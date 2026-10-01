"""One training policy for all four models: learn every epoch, or say why not.

The encoder, Branch A, Branch B and DeepOP each had their own loop, and none of
them had a learning-rate schedule, a warmup, or any protection against a
non-finite loss; only DeepOP clipped gradients, and patience ran from 2 to 5.
A run could therefore spend hours on epochs that did nothing, or step once on a
NaN and continue from garbage weights with nothing in the log to say so.

TrainingGuard wraps the optimizer step and the end of each epoch:

  per step    * linear LR warmup over the first `warmup_steps` updates, so the
                first batches cannot throw the weights somewhere they never
                recover from (the "collapsed at epoch 1" failure);
              * gradient-norm clipping, with the pre-clip norm recorded;
              * a non-finite loss or gradient is SKIPPED -- no optimizer step --
                and counted.

  per epoch   * improved   -> snapshot weights + optimizer state in memory;
              * `step_back_after` epochs without improvement (default 2)
                -> STEP BACK: restore the best snapshot, multiply the LR by
                   `lr_factor` (default 0.5), print a diagnosis of what looks
                   wrong, and carry on from the best point;
              * `patience` epochs without improvement (default 3) -> STOP. The
                best weights are already saved, so stopping costs nothing;
              * an unstable epoch (non-finite score or > `max_nonfinite_frac`
                of steps skipped) steps back immediately.

The diagnosis is the "take a step back and think what's wrong" step, made
concrete: it compares the train-loss trend with the validation trend, reads
the gradient norms and clipping rate, and repeats the trainer's own health
checks (a head predicting one class, a world model worse than persistence),
then states which of the usual causes the numbers point to.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

IMPROVED, WAIT, STEP_BACK, STOP = "improved", "wait", "step_back", "stop"


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class TrainingGuard:
    def __init__(
        self,
        name: str,
        modules: Sequence[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        *,
        mode: str = "max",
        patience: int = 3,
        step_back_after: int = 2,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        max_step_backs: int = 2,
        warmup_steps: int = 0,
        clip_norm: Optional[float] = None,
        max_nonfinite_frac: float = 0.01,
        min_delta: float = 1e-4,
        log: Callable[[str], None] = print,
        graph_undo: Optional[bool] = None,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if patience and step_back_after >= patience:
            raise ValueError("step_back_after must be smaller than patience, or the "
                             "run stops before it ever gets a second chance")
        self.name = name
        self.modules = list(modules)
        self.optimizer = optimizer
        self.mode = mode
        self.patience = int(patience)
        self.step_back_after = int(step_back_after)
        self.lr_factor = float(lr_factor)
        self.min_lr = float(min_lr)
        self.max_step_backs = int(max_step_backs)
        self.warmup_steps = int(warmup_steps)
        self.clip_norm = clip_norm
        self.max_nonfinite_frac = float(max_nonfinite_frac)
        self.min_delta = float(min_delta)
        self.log = log

        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.lr_scale = 1.0
        self.global_step = 0

        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.since_improve = 0
        self.step_backs = 0
        self.epoch = 0
        self.stop_reason: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        self._best_state: Optional[Dict[str, Any]] = None
        self._pending = None            # a deferred step whose outcome is not read yet
        self._snap = None
        #: Replay the deferred step's snapshot copy and its where-undo as two
        #: CUDA graphs (one launch each instead of one per tensor). Same
        #: kernels on the same tensors: bit-identical. Re-captured whenever the
        #: tensors move (e.g. a step-back reloads the optimizer state).
        #: None: on unless CYBERWORLD_CUDA_GRAPH=0 (it only acts on CUDA, in the
        #: deferred step). Every trainer, the encoder included, gets it without
        #: an argument -- bita/train.py is part of the shared-setup cache key.
        if graph_undo is None:
            import os as _os
            graph_undo = _os.environ.get("CYBERWORLD_CUDA_GRAPH", "1") not in ("0", "false", "False", "")
        self.graph_undo = bool(graph_undo)
        self._undo_graphs = None
        self._reset_epoch_stats()
        self._apply_lr()

    # ------------------------------------------------------------------ steps
    def _params(self) -> Iterable[torch.nn.Parameter]:
        for m in self.modules:
            yield from (p for p in m.parameters() if p.requires_grad)

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_lr(self) -> None:
        warm = 1.0
        if self.warmup_steps and self.global_step < self.warmup_steps:
            warm = 0.1 + 0.9 * (self.global_step / self.warmup_steps)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * self.lr_scale * warm

    def backward_step(self, loss: torch.Tensor) -> bool:
        """backward + clip + step. Returns False (and skips the step) when the
        loss or the gradient is not finite.

        ONE device->host synchronisation per step. The loss-finite check and
        the gradient norm are read together after backward, instead of one
        sync before backward (isfinite) and one after it (the norm). The
        outcome is the same: for a non-finite loss the reference skips
        backward and sets the grads to None; here backward runs, and the
        grads (non-finite, possibly clipped) are then set to None -- no
        optimizer step, no counter or diagnostic touched but n_nonfinite.
        The first step of an epoch (n_grad == 0) records the per-parameter
        gradient norms, which the reference records only for a finite loss;
        that step keeps the reference's order exactly.
        """
        self.n_steps += 1
        if self.n_grad == 0 or not torch.is_tensor(loss) or loss.device.type == "cpu":
            if not bool(torch.isfinite(loss.detach()).all()):
                self.optimizer.zero_grad(set_to_none=True)
                self.n_nonfinite += 1
                return False
            loss.backward()
            return self.step_after_backward()
        loss_ok = torch.isfinite(loss.detach()).all()
        loss.backward()
        return self.step_after_backward(loss_ok=loss_ok)

    def _record_top_grads(self) -> None:
        """Which parameters carry the gradient, once per epoch (first step).

        When every step hits the clip, the update direction is whatever the
        largest-gradient tensor says, and the rest of the model barely moves.
        Naming that tensor turns "gradients are large" into a cause.
        """
        norms = []
        for mi, m in enumerate(self.modules):
            for n, p in m.named_parameters():
                if p.requires_grad and p.grad is not None:
                    norms.append((float(p.grad.detach().float().norm()),
                                  f"{type(m).__name__}.{n}" if len(self.modules) > 1 else n))
        total = math.sqrt(sum(v * v for v, _ in norms)) or 1.0
        norms.sort(reverse=True)
        self.top_grads = [(name, v, (v / total) ** 2) for v, name in norms[:3]]

    def step_after_backward(self, loss_ok: Optional[torch.Tensor] = None) -> bool:
        """For loops that call backward themselves (e.g. gradient accumulation).

        `loss_ok` (a device bool, from backward_step) is read in the same
        synchronisation as the gradient norm; a False skips the step exactly
        as a non-finite loss does in backward_step."""
        if self.n_grad == 0:
            self._record_top_grads()
        params = list(self._params())
        norm = torch.nn.utils.clip_grad_norm_(
            params, self.clip_norm if self.clip_norm else float("inf"))
        if loss_ok is not None and norm.device == loss_ok.device:
            ok_f, norm_f = torch.stack([loss_ok.to(norm.dtype), norm]).tolist()   # the one sync
            if not ok_f:
                self.optimizer.zero_grad(set_to_none=True)
                self.n_nonfinite += 1
                return False
        else:
            if loss_ok is not None and not bool(loss_ok):
                self.optimizer.zero_grad(set_to_none=True)
                self.n_nonfinite += 1
                return False
            norm_f = float(norm)
        if not math.isfinite(norm_f):
            self.optimizer.zero_grad(set_to_none=True)
            self.n_nonfinite += 1
            return False
        self.grad_norm_sum += norm_f
        self.grad_norm_max = max(self.grad_norm_max, norm_f)
        self.n_grad += 1
        if self.clip_norm and norm_f > self.clip_norm:
            self.n_clipped += 1
        self._apply_lr()
        self.optimizer.step()
        self.global_step += 1
        return True

    # ------------------------------------------------- deferred (no-sync) step
    def deferred_supported(self) -> bool:
        """backward_step_deferred needs an optimizer whose skipped step it can
        undo exactly: torch.optim.Adam/AdamW, foreach or single-tensor (not
        fused, not capturable, no amsgrad), parameters on one CUDA device."""
        import os
        if os.environ.get("CYBERWORLD_GUARD_SYNC", "") in ("1", "true", "True"):
            return False
        opt = self.optimizer
        if type(opt) not in (torch.optim.Adam, torch.optim.AdamW):
            return False
        for g in opt.param_groups:
            if g.get("fused") or g.get("capturable") or g.get("amsgrad") or g.get("differentiable"):
                return False
            if torch.is_tensor(g.get("lr")):
                return False
        devs = {p.device for p in self._params()}
        return len(devs) == 1 and next(iter(devs)).type == "cuda"

    def backward_step_deferred(self, loss: torch.Tensor):
        """backward_step without a device->host synchronisation.

        Same decisions, counters, diagnostics and weights as backward_step,
        bit for bit (tests/test_training_guard_one_sync.py). The step is
        taken on the device speculatively and undone on the device if the
        loss or the gradient is not finite:

          snapshot   params, exp_avg, exp_avg_sq  (one multi-tensor copy)
          step       the optimizer's own step, unchanged
          undo       x = where(ok, x, snapshot)   (selects, never computes)

        The outcome (ok, grad norm) is copied to pinned host memory without
        waiting and read at the NEXT step (or flush()), by which time the
        device has long finished it; only then are the host-side counters,
        the global step (the warmup LR) and, for a skipped step, Adam's step
        counts updated -- before anything reads them. Returns a device bool
        (or a Python bool when it fell back to backward_step).
        """
        self._resolve()
        if self.n_grad == 0:            # first step of an epoch: records top grads, keeps the exact order
            return self.backward_step(loss)
        params = list(self._params())
        self.n_steps += 1
        loss_ok = torch.isfinite(loss.detach()).all()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, self.clip_norm if self.clip_norm else float("inf"))
        ok = loss_ok & torch.isfinite(norm)
        live, stepped = [], []
        opt = self.optimizer
        for g in opt.param_groups:
            for q in g["params"]:
                if q.grad is None:
                    continue
                st = opt.state.get(q)
                if not st or "exp_avg" not in st:
                    # a parameter's first step creates its state; nothing to snapshot. Rare
                    # (first use of a parameter): settle this step synchronously.
                    return self._settle_now(ok, norm)
                live += [q, st["exp_avg"], st["exp_avg_sq"]]
                stepped.append(st["step"])
        snap = self._snapshot_buffers(live)
        graphs = self._graphs_for(live, snap) if self.graph_undo else None
        if graphs is not None:
            graphs["snap"].replay()
        else:
            with torch.no_grad():
                torch._foreach_copy_(snap, live)
        self._apply_lr()
        opt.step()
        if graphs is not None:
            graphs["ok"].copy_(ok)
            graphs["undo"].replay()
        else:
            with torch.no_grad():
                for x, y in zip(live, snap):
                    torch.where(ok, x, y, out=x)
        out = torch.stack([ok.to(norm.dtype), norm.detach()])
        host = torch.empty(2, dtype=out.dtype, pin_memory=True)
        host.copy_(out, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending = (host, ev, stepped)
        return ok

    def deferred_step_graphed(self, whole, args):
        """backward_step_deferred with the forward, loss, backward, gradient
        clipping, the finite checks and the snapshot replayed as ONE CUDA graph
        (`whole`: cyberworld_v4.graphed_step.WholeStepGraph).

        The same kernels in the same order as the eager deferred step, so the
        same decisions, counters and weights bit for bit
        (tests/test_graphed_step.py). The optimizer step stays eager -- Adam's
        bias correction is computed on the host from its step count -- and the
        undo is the graph_undo graph. Falls back to computing the loss eagerly
        and calling backward_step_deferred whenever the graph cannot be used:
        the first step of an epoch, another batch shape, no optimizer state yet.

        Returns (loss, ok) as backward_step_deferred's caller would see them.
        """
        self._resolve()
        live = None
        if self.n_grad != 0 and whole.usable(args):
            live, stepped = self._graph_live(whole)
        if live is None or not whole.ensure(args, live, self._snapshot_buffers(live), self,
                                            {live[i] for i in range(0, len(live), 3)}):
            loss = whole.eager(args)
            return loss, self.backward_step_deferred(loss)
        self.n_steps += 1
        whole.replay(args)
        loss, norm, ok = whole.loss, whole.norm, whole.ok
        snap = self._snapshot_buffers(live)
        graphs = self._graphs_for(live, snap)
        self._apply_lr()
        self.optimizer.step()
        if graphs is not None:
            graphs["ok"].copy_(ok)
            graphs["undo"].replay()
        else:
            with torch.no_grad():
                for x, y in zip(live, snap):
                    torch.where(ok, x, y, out=x)
        out = torch.stack([ok.to(norm.dtype), norm.detach()])
        host = torch.empty(2, dtype=out.dtype, pin_memory=True)
        host.copy_(out, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending = (host, ev, stepped)
        return loss, ok

    def _graph_live(self, whole):
        """(live, stepped) for the parameters the graph gives gradients to, in
        optimizer order -- what backward_step_deferred collects from `.grad`."""
        opt = self.optimizer
        # Before the first capture: the parameters that have Adam state are the
        # ones that have received gradients. The capture verifies the graph
        # gives gradients to exactly these.
        want = whole.grad_params or {q for g in opt.param_groups for q in g["params"]
                                     if "exp_avg" in opt.state.get(q, {})}
        if not want:
            return None, None
        live, stepped = [], []
        for g in opt.param_groups:
            for q in g["params"]:
                if q not in want:
                    continue
                st = opt.state.get(q)
                if not st or "exp_avg" not in st:
                    return None, None
                live += [q, st["exp_avg"], st["exp_avg_sq"]]
                stepped.append(st["step"])
        return live, stepped

    def _graphs_for(self, live, snap):
        """{"snap", "undo", "ok"}: the snapshot copy and the undo as CUDA graphs,
        keyed by the tensors' addresses. None if capture is not possible."""
        key = tuple(x.data_ptr() for x in live) + tuple(y.data_ptr() for y in snap)
        g = self._undo_graphs
        if g is not None and g["key"] == key:
            return g
        try:
            ok_static = torch.zeros((), dtype=torch.bool, device=live[0].device)
            side = torch.cuda.Stream(device=live[0].device)
            side.wait_stream(torch.cuda.current_stream(live[0].device))
            gs, gu = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            # capture records, it does not execute: no tensor is modified here
            with torch.no_grad():
                # thread_local: the DataLoader's pin-memory thread may make CUDA
                # calls meanwhile (see cyberworld_v4/graphed_step.thread_local_capture)
                with torch.cuda.graph(gs, stream=side, capture_error_mode="thread_local"):
                    torch._foreach_copy_(snap, live)
                with torch.cuda.graph(gu, stream=side, capture_error_mode="thread_local"):
                    for x, y in zip(live, snap):
                        torch.where(ok_static, x, y, out=x)
            torch.cuda.current_stream(live[0].device).wait_stream(side)
        except Exception as e:                       # stay eager
            self.graph_undo = False
            self.log(f"[{self.name}] guard graphs unavailable ({type(e).__name__}: {e}); eager undo")
            return None
        self._undo_graphs = {"key": key, "snap": gs, "undo": gu, "ok": ok_static}
        return self._undo_graphs

    def _snapshot_buffers(self, live):
        key = [(x.shape, x.dtype, x.device) for x in live]
        if self._snap is None or self._snap[0] != key:
            self._snap = (key, [torch.empty_like(x) for x in live])
        return self._snap[1]

    def _settle_now(self, ok, norm) -> bool:
        ok_f, norm_f = torch.stack([ok.to(norm.dtype), norm]).tolist()
        return self._account(bool(ok_f), norm_f, step=True)

    def _account(self, ok: bool, norm_f: float, step: bool) -> bool:
        """The host-side bookkeeping of one step, as step_after_backward does it."""
        if not ok:
            if step:
                self.optimizer.zero_grad(set_to_none=True)
            self.n_nonfinite += 1
            return False
        self.grad_norm_sum += norm_f
        self.grad_norm_max = max(self.grad_norm_max, norm_f)
        self.n_grad += 1
        if self.clip_norm and norm_f > self.clip_norm:
            self.n_clipped += 1
        if step:
            self._apply_lr()
            self.optimizer.step()
        self.global_step += 1
        return True

    def _resolve(self) -> None:
        """Read the outcome of the pending deferred step (see backward_step_deferred)."""
        if self._pending is None:
            return
        host, ev, stepped = self._pending
        self._pending = None
        ev.synchronize()                # long done in practice: a whole accumulation group ago
        ok_f, norm_f = host.tolist()
        ok = bool(ok_f)
        if not ok:
            # The device already undid the weights and moments; Adam's step
            # counts live on the host and were advanced by the step: undo them.
            for t in stepped:
                t.sub_(1)
        self._account(ok, norm_f, step=False)

    def flush(self) -> None:
        """Settle any deferred step (call before reading counters/state)."""
        self._resolve()

    def _reset_epoch_stats(self) -> None:
        self.n_steps = 0
        self.n_nonfinite = 0
        self.n_grad = 0
        self.n_clipped = 0
        self.grad_norm_sum = 0.0
        self.grad_norm_max = 0.0
        self.top_grads = []

    # --------------------------------------------------------------- epochs
    def _improved(self, score: float) -> bool:
        if self.best is None:
            return True
        tol = max(self.min_delta, self.min_delta * abs(self.best))
        return score > self.best + tol if self.mode == "max" else score < self.best - tol

    def _snapshot(self) -> None:
        self._best_state = {
            "modules": [{k: v.detach().clone() for k, v in m.state_dict().items()}
                        for m in self.modules],
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def _restore(self) -> bool:
        if self._best_state is None:
            return False
        for m, sd in zip(self.modules, self._best_state["modules"]):
            m.load_state_dict(sd)
        self.optimizer.load_state_dict(copy.deepcopy(self._best_state["optimizer"]))
        self._apply_lr()   # load_state_dict restored the snapshot's LR; re-apply the scale
        return True

    def end_epoch(self, score: float, train_loss: Optional[float] = None,
                  health: Optional[Sequence[str]] = None) -> str:
        """Record one epoch and decide what happens next. Returns the action."""
        self._resolve()
        self.epoch += 1
        health = [h for h in (health or []) if h]
        frac_bad = self.n_nonfinite / max(self.n_steps, 1)
        unstable = (not _finite(score) or (train_loss is not None and not _finite(train_loss))
                    or frac_bad > self.max_nonfinite_frac)
        rec = {
            "epoch": self.epoch, "score": float(score) if _finite(score) else None,
            "train_loss": float(train_loss) if _finite(train_loss) else None,
            "lr": self.current_lr(), "grad_norm_mean": self.grad_norm_sum / max(self.n_grad, 1),
            "grad_norm_max": self.grad_norm_max,
            "clipped_frac": self.n_clipped / max(self.n_grad, 1),
            "nonfinite_steps": self.n_nonfinite, "steps": self.n_steps, "health": list(health),
            "top_grads": [{"param": n, "norm": v, "share": s} for n, v, s in self.top_grads],
        }

        if not unstable and self._improved(float(score)):
            self.best, self.best_epoch, self.since_improve = float(score), self.epoch, 0
            self._snapshot()
            action = IMPROVED
        else:
            self.since_improve += 1
            if unstable:
                action = STEP_BACK
            elif self.patience and self.since_improve >= self.patience:
                action = STOP
            elif self.since_improve == self.step_back_after:
                action = STEP_BACK
            else:
                action = WAIT

        if action == STEP_BACK:
            next_lr = max(b * self.lr_scale * self.lr_factor for b in self.base_lrs)
            if self.step_backs >= self.max_step_backs or next_lr < self.min_lr:
                action = STOP
                self.stop_reason = (f"no improvement after {self.step_backs} step-back(s); "
                                    f"the LR is already at {self.current_lr():.2e}")
            else:
                self.lr_scale *= self.lr_factor
                self.step_backs += 1
                restored = self._restore()
                self._apply_lr()
                rec["restored_epoch"] = self.best_epoch if restored else None
        if action == STOP and self.stop_reason is None:
            self.stop_reason = (f"{self.since_improve} epochs without improvement "
                                f"(patience {self.patience}); best epoch {self.best_epoch}")

        rec["action"] = action
        rec["lr_after"] = self.current_lr()
        self.history.append(rec)
        self._log_epoch(rec, unstable, frac_bad)
        if action in (STEP_BACK, STOP):
            for line in self.diagnose(unstable=unstable, frac_bad=frac_bad):
                self.log(f"  [{self.name} guard]   {line}")
        self._reset_epoch_stats()
        return action

    # ------------------------------------------------------------ reporting
    def _log_epoch(self, rec: Dict[str, Any], unstable: bool, frac_bad: float) -> None:
        s = rec["score"]
        score_txt = "nan" if s is None else f"{s:.4f}"
        best_txt = "none" if self.best is None else f"{self.best:.4f}"
        clip_txt = f" clipped {rec['clipped_frac']:.0%}" if self.clip_norm else ""
        msg = (f"[{self.name} guard] epoch {rec['epoch']}: score={score_txt} best={best_txt} "
               f"(epoch {self.best_epoch}) | lr {rec['lr']:.2e} | grad norm mean "
               f"{rec['grad_norm_mean']:.3g} max {rec['grad_norm_max']:.3g}{clip_txt}"
               f" | skipped {rec['nonfinite_steps']}/{rec['steps']} | -> {rec['action'].upper()}")
        if rec["action"] == STEP_BACK:
            msg += (f" (restored epoch {rec.get('restored_epoch')}, lr -> {rec['lr_after']:.2e})")
        self.log(msg)
        if self.clip_norm and rec["clipped_frac"] > 0.5 and rec["top_grads"]:
            self.log(f"  [{self.name} guard] largest gradients: " + ", ".join(
                f"{g['param']} {g['norm']:.3g} ({g['share']:.0%})" for g in rec["top_grads"]))
        for h in rec["health"]:
            self.log(f"  [{self.name} guard] health: {h}")

    def diagnose(self, unstable: bool = False, frac_bad: float = 0.0) -> List[str]:
        """What the recent epochs say is wrong, in plain words."""
        out: List[str] = []
        h = self.history
        last = h[-1]
        if unstable:
            out.append(f"numerically unstable: {last['nonfinite_steps']} of {last['steps']} steps "
                       f"had a non-finite loss/gradient ({frac_bad:.1%}) or the score was NaN. "
                       f"Usual cause: learning rate too high. Weights restored, LR cut.")
        losses = [r["train_loss"] for r in h[-3:] if r["train_loss"] is not None]
        scores = [r["score"] for r in h[-3:] if r["score"] is not None]
        if len(losses) >= 2:
            rel = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)
            if rel > 0.01 and not unstable:
                out.append(f"train loss still falling ({losses[0]:.4g} -> {losses[-1]:.4g}) while "
                           f"validation did not improve: over-fitting or a train/validation shift, "
                           f"not a failure to learn. The best weights are the ones kept.")
            elif abs(rel) <= 0.01:
                out.append(f"train loss flat ({losses[0]:.4g} -> {losses[-1]:.4g}): the model has "
                           f"stopped learning on the training data itself -- converged, under-capacity, "
                           f"or an LR too small to move it (a lower LR follows).")
            elif rel < -0.01:
                out.append(f"train loss RISING ({losses[0]:.4g} -> {losses[-1]:.4g}): steps are too "
                           f"large; the LR cut is the right response.")
        if last["grad_norm_mean"] < 1e-7 and last["steps"]:
            out.append("gradients are ~0: nothing reaches the weights (dead units or a detached "
                       "graph). A lower LR will not fix this.")
        if self.clip_norm and last["clipped_frac"] > 0.5:
            top = last.get("top_grads") or []
            dom = (f" Dominated by {top[0]['param']} ({top[0]['share']:.0%} of the squared norm): "
                   f"that tensor sets every update's direction." if top and top[0]["share"] > 0.5 else "")
            out.append(f"{last['clipped_frac']:.0%} of steps hit the clip norm {self.clip_norm}: "
                       f"gradients are persistently large -- an LR that is too high, or an input "
                       f"on the wrong scale.{dom}")
        seen = set()
        for r in h[-3:]:
            for x in r["health"]:
                if x not in seen:
                    seen.add(x)
                    out.append(f"health check: {x}")
        if scores and len(scores) >= 2 and max(scores) - min(scores) < self.min_delta * 10:
            out.append("validation score has not moved at all: if a health check above says the "
                       "head predicts one class, the model has collapsed onto the prior.")
        action = last["action"]
        if action == STEP_BACK:
            out.append(f"action: rolled back to epoch {last.get('restored_epoch')} and continued at "
                       f"lr {last['lr_after']:.2e} (step-back {self.step_backs}/{self.max_step_backs}).")
        elif action == STOP:
            out.append(f"action: stopped -- {self.stop_reason}. Best epoch {self.best_epoch} "
                       f"(score {self.best}) is what is saved.")
        return out

    def should_stop(self) -> bool:
        return bool(self.history) and self.history[-1]["action"] == STOP

    # ------------------------------------------------------------- resume
    _RESUME_KEYS = ("lr_scale", "global_step", "best", "best_epoch", "since_improve",
                    "step_backs", "epoch", "stop_reason", "history", "_best_state")

    def state_dict(self) -> Dict[str, Any]:
        """Everything the guard knows, including the best snapshot, for a resume point."""
        self._resolve()
        return {k: getattr(self, k) for k in self._RESUME_KEYS}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._pending = None
        for k in self._RESUME_KEYS:
            setattr(self, k, sd[k])
        self._reset_epoch_stats()
        self._apply_lr()

    def summary(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch, "best_score": self.best,
            "step_backs": self.step_backs, "stop_reason": self.stop_reason,
            "final_lr": self.current_lr(), "patience": self.patience,
            "step_back_after": self.step_back_after, "history": self.history,
        }


def default_warmup_steps(steps_per_epoch: int, cap: int = 500) -> int:
    """~5% of the first epoch, at most `cap` updates, at least 1."""
    return max(1, min(cap, steps_per_epoch // 20))


# ---------------------------------------------------------------- crash resume
def _rng_state() -> Dict[str, Any]:
    import random
    import numpy as np
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: Dict[str, Any]) -> None:
    import random
    import numpy as np
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available() and len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(st["cuda"])


def run_fingerprint(args, ignore: Iterable[str] = ()) -> Dict[str, str]:
    """The arguments a resume point must match. Epoch ceilings, worker counts
    and the device may change between attempts; anything else means a
    different run, whose half-trained state must not be continued."""
    skip = set(ignore) | {"resume", "no_resume"}
    return {k: repr(v) for k, v in sorted(vars(args).items()) if k not in skip}


class ResumePoint:
    """Crash recovery: the state at the end of the last finished epoch.

    A run killed at hour five -- power cut, OOM, a reboot -- restarted from
    epoch 0 before this. Now each trainer writes one file per run (atomically:
    temp file + os.replace, so a crash mid-write leaves the previous epoch's
    file intact) holding the weights, the optimizer, the TrainingGuard (best
    snapshot included) and the RNG states. Re-running the same command
    continues after the last finished epoch; a successful run deletes it.

    A file whose arguments differ from this run's is set aside (renamed
    `.stale`), never loaded: continuing a different run's weights would be
    silently wrong. An unreadable file is set aside as `.corrupt`.
    """

    def __init__(self, path, fingerprint: Dict[str, str], *, enabled: bool = True,
                 log: Callable[[str], None] = print):
        self.path = str(path)
        self.fingerprint = dict(fingerprint)
        self.enabled = enabled
        self.log = log

    def _set_aside(self, suffix: str) -> None:
        import os
        try:
            os.replace(self.path, self.path + suffix)
        except OSError:
            pass

    def load(self) -> Optional[Dict[str, Any]]:
        import os
        if not self.enabled or not os.path.exists(self.path):
            return None
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as exc:  # truncated or foreign file
            self.log(f"[resume] {self.path} unreadable ({exc}); set aside as .corrupt, starting fresh")
            self._set_aside(".corrupt")
            return None
        theirs = payload.get("fingerprint") or {}
        if theirs != self.fingerprint:
            diff = sorted(k for k in set(theirs) | set(self.fingerprint)
                          if theirs.get(k) != self.fingerprint.get(k))
            self.log(f"[resume] {self.path} is from a run with different arguments "
                     f"({', '.join(diff[:8])}); set aside as .stale, starting fresh")
            self._set_aside(".stale")
            return None
        if payload.get("rng"):
            _set_rng_state(payload["rng"])
        self.log(f"[resume] continuing from {self.path}: {payload.get('done_epochs')} epoch(s) "
                 f"already finished")
        return payload

    def save(self, done_epochs: int, **state: Any) -> None:
        import os
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        torch.save({"fingerprint": self.fingerprint, "done_epochs": int(done_epochs),
                    "rng": _rng_state(), **state}, tmp)
        os.replace(tmp, self.path)
        # Test hook: die here, as a power cut would, so a test can prove the
        # next run picks up from this file (tests/test_training_resume.py).
        if os.environ.get("CYBERWORLD_TEST_CRASH_AFTER_EPOCH") == str(int(done_epochs)):
            self.log(f"[resume] TEST CRASH after epoch {done_epochs}")
            os._exit(99)

    def clear(self) -> None:
        import os
        for p in (self.path, self.path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
