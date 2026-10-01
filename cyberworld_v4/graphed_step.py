"""Replay a training step's forward + backward as CUDA graphs.

## Why

Branch A, Branch B and DeepOP are small models: a training step is ~1 ms of
GPU work issued as a few hundred tiny kernels, and launching them from Python
cost ~3.4 ms per step (measured, Branch A). The GPU sat ~20% busy at idle
clocks while one CPU thread was pegged. A CUDA graph records the launches
once and replays them with one call.

## What is graphed, and why it is bit-identical

`loss = fn(*inputs)` and its backward, via torch.cuda.make_graphed_callables:
the same kernels, on the same inputs, in the same order. The optimizer step,
clipping and the guard stay eager (Adam's bias correction is computed on the
host from a Python step count, which a graph would freeze).

Two things would otherwise make it differ, and both are handled:

  * make_graphed_callables runs warm-up iterations, which consume the CUDA
    (dropout) and CPU RNG streams. Both states are saved before and restored
    after, so the first real step draws exactly the masks eager would.
  * A graph has fixed shapes. A batch of any other shape (the last, partial
    batch of an epoch) runs the eager callable -- the same function.

tests/test_graphed_step.py checks losses and every weight against eager over
hundreds of steps with dropout live; the trainers' end-to-end runs are
compared leaf by leaf (see PERF_REPORT.md).

Training mode is captured; call the eager module for evaluation.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional, Sequence

import torch


def graphs_enabled(default: bool = True) -> bool:
    """CYBERWORLD_CUDA_GRAPH=0 turns the graphed steps off everywhere."""
    v = os.environ.get("CYBERWORLD_CUDA_GRAPH")
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


@contextlib.contextmanager
def thread_local_capture():
    """Capture with capture_error_mode="thread_local" while this is active.

    The default ("global") makes CUDA reject an unsafe API call from ANY
    thread during a capture -- and the DataLoader's pin-memory thread pins
    host buffers concurrently, which invalidated a DeepOP capture. Only this
    thread's stream is being captured, so only this thread needs checking.
    make_graphed_callables does not take the mode, so `torch.cuda.graph` is
    swapped for the duration (it is looked up at call time).
    """
    orig = torch.cuda.graph

    class _TL(orig):
        def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="thread_local"):
            super().__init__(cuda_graph, pool=pool, stream=stream,
                             capture_error_mode=capture_error_mode)

    torch.cuda.graph = _TL
    try:
        yield
    finally:
        torch.cuda.graph = orig


class _Wrap(torch.nn.Module):
    def __init__(self, fn: Callable, modules: Sequence[torch.nn.Module]):
        super().__init__()
        self.mods = torch.nn.ModuleList(modules)    # so make_graphed_callables sees the parameters
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class GraphedLoss:
    """`GraphedLoss(fn, modules, example)(*inputs) == fn(*inputs)`, replayed as a graph.

    (WholeStepGraph takes the same kind of `fn`; its monitors are in
    `last_extras` after each step.)

    `fn` returns a scalar loss, or a tuple whose first element is the loss and
    whose other elements are DETACHED monitors (a monitor that required grad
    would get a zero grad_output in the graphed backward), from the tensors it
    is given and the parameters of `modules`. `example` fixes the graphed shapes
    and dtypes; inputs of any other shape fall back to calling `fn` eagerly.
    Built lazily on the first matching call, so it costs nothing when unused.
    """

    def __init__(self, fn: Callable, modules: Sequence[torch.nn.Module], enabled: bool = True,
                 on_error: str = "eager", log: Optional[Callable[[str], None]] = print):
        self.fn = fn
        self.on_error = on_error
        self.log = log
        self.error = None
        self.modules = list(modules)
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self._graphed = None
        self._sig = None
        self.n_graphed = 0
        self.n_eager = 0

    @staticmethod
    def _signature(args):
        return tuple((tuple(a.shape), a.dtype, a.device) if torch.is_tensor(a) else ("x", a)
                     for a in args)

    def build(self, example) -> None:
        dev = next((a.device for a in example if torch.is_tensor(a)), None)
        if dev is None or dev.type != "cuda":
            self.enabled = False
            return
        cuda_rng = torch.cuda.get_rng_state(dev)
        cpu_rng = torch.get_rng_state()
        # warm-up and capture must not see .grad from a previous step
        saved = [(p, p.grad) for m in self.modules for p in m.parameters()]
        for p, _ in saved:
            p.grad = None
        static = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in example)
        try:
            with thread_local_capture():
                self._graphed = torch.cuda.make_graphed_callables(
                    _Wrap(self.fn, self.modules), static, allow_unused_input=True)
            self._sig = self._signature(example)
        except Exception as e:          # an op that cannot be captured: stay eager
            self._graphed = None
            self.enabled = False
            self.error = f"{type(e).__name__}: {e}"
            if self.on_error == "raise":
                raise
            if self.log is not None:
                self.log(f"  cuda graph unavailable ({self.error}); running eagerly")
        finally:
            for p, g in saved:
                p.grad = g
            torch.cuda.set_rng_state(cuda_rng, dev)
            torch.set_rng_state(cpu_rng)

    def __call__(self, *args):
        if self.enabled and all(m.training for m in self.modules):
            sig = self._signature(args)
            if self._graphed is None:
                self.build(args)
            if self._graphed is not None and sig == self._sig:
                self.n_graphed += 1
                return self._graphed(*args)
        self.n_eager += 1
        return self.fn(*args)


class WholeStepGraph:
    """Forward + loss + backward + gradient clipping + finite checks + the
    guard's snapshot as one CUDA graph, for TrainingGuard.deferred_step_graphed.

    GraphedLoss graphs the forward/backward only; the guard's clip, checks and
    snapshot then still launched eagerly, and autograd cloned every gradient
    out of the graph's buffers. Here the gradients live in static buffers that
    the graph writes and `.grad` points at. Same kernels, same order: the
    recorded step is the eager deferred step.

    Captured lazily, re-captured when the optimizer's tensors move (a
    step-back reloads them) or the guard's clip norm changes. The CUDA and
    CPU RNG states are restored after warm-up and capture.
    """

    def __init__(self, fn: Callable, modules: Sequence[torch.nn.Module], enabled: bool = True,
                 log: Optional[Callable[[str], None]] = print):
        self.fn = fn
        self.modules = list(modules)
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.log = log
        self.error = None
        self.grad_params = set()
        self._g = None
        self._key = None
        self._sig = None
        self.n_graphed = 0

    def _params(self):
        for m in self.modules:
            yield from (p for p in m.parameters() if p.requires_grad)

    def usable(self, args) -> bool:
        if not self.enabled or not all(m.training for m in self.modules):
            return False
        if not all(torch.is_tensor(a) and a.is_cuda for a in args):
            return False
        if self._sig is None:
            # which parameters get a gradient is known after the first capture;
            # until then any full-shape batch may trigger one
            return True
        return GraphedLoss._signature(args) == self._sig

    def ensure(self, args, live, snap, guard, expected_grad_params) -> bool:
        key = (GraphedLoss._signature(args), guard.clip_norm,
               tuple(x.data_ptr() for x in live), tuple(y.data_ptr() for y in snap))
        if self._g is not None and key == self._key:
            return True
        if self._g is not None and key[0] != self._sig:
            return False                    # another batch shape: eager
        try:
            self._capture(args, live, snap, guard.clip_norm, expected_grad_params)
        except Exception as e:
            self.enabled = False
            self._g = None
            self.error = f"{type(e).__name__}: {e}"
            if self.log is not None:
                self.log(f"  whole-step cuda graph unavailable ({self.error}); running eagerly")
            return False
        self._key = (GraphedLoss._signature(args), guard.clip_norm,
                     tuple(x.data_ptr() for x in live), tuple(y.data_ptr() for y in snap))
        return True

    def _capture(self, args, live, snap, clip_norm, expected_grad_params):
        dev = args[0].device
        params = list(self._params())
        cuda_rng, cpu_rng = torch.cuda.get_rng_state(dev), torch.get_rng_state()
        saved = [(p, p.grad) for p in params]
        clip = clip_norm if clip_norm else float("inf")
        static = [a.detach().clone() for a in args]

        def body():
            out = self.fn(*static)
            loss, extras = (out[0], tuple(out[1:])) if isinstance(out, tuple) else (out, ())
            loss_ok = torch.isfinite(loss.detach()).all()
            # The gradients from autograd.grad, assigned to .grad: what
            # backward() does into a None .grad (AccumulateGrad hands the
            # buffer over), minus the AccumulateGrad nodes -- those may have
            # been created on another stream by an earlier eager step, which
            # breaks capture.
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            gp = []
            for p, gr in zip(params, grads):
                if gr is not None:
                    p.grad = gr
                    gp.append(p)
            norm = torch.nn.utils.clip_grad_norm_(gp, clip)
            ok = loss_ok & torch.isfinite(norm)
            with torch.no_grad():
                torch._foreach_copy_(snap, live)
            return loss.detach(), norm, ok, gp, extras

        try:
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):                         # warm-up (lazy init, cuDNN plans)
                    for p in params:
                        p.grad = None
                    body()
            torch.cuda.current_stream(dev).wait_stream(side)
            for p in params:
                p.grad = None
            g = torch.cuda.CUDAGraph()
            # The parameters' AccumulateGrad nodes were created by eager steps
            # on the default stream; let autograd redirect them to the capture
            # stream (it is what the stream is during replay as well).
            _ovr = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
            if _ovr is not None:
                _ovr(True)
            try:
                with torch.cuda.graph(g, stream=side, capture_error_mode="thread_local"):
                    loss, norm, ok, gp, extras = body()
            finally:
                if _ovr is not None:
                    _ovr(False)
            if set(gp) != set(expected_grad_params):
                raise RuntimeError("the graph's gradient set differs from the optimizer's state")
            self._g, self._static = g, static
            self.loss, self.norm, self.ok = loss, norm, ok
            self._extras = extras
            self._grad_of = [(p, p.grad) for p in gp]
            self.grad_params = set(gp)
            self._sig = GraphedLoss._signature(args)
        finally:
            for p, gr in saved:
                p.grad = gr
            torch.cuda.set_rng_state(cuda_rng, dev)
            torch.set_rng_state(cpu_rng)

    def replay(self, args):
        for s, a in zip(self._static, args):
            s.copy_(a, non_blocking=True)
        self._g.replay()
        for p, gr in self._grad_of:
            p.grad = gr
        self.last_extras = self._extras
        self.n_graphed += 1

    def eager(self, args):
        """`fn(*args)` eagerly; returns the loss, keeps any monitors in last_extras."""
        out = self.fn(*args)
        if isinstance(out, tuple):
            self.last_extras = tuple(out[1:])
            return out[0]
        self.last_extras = ()
        return out


class GraphedBody:
    """Replay `body(acc, *inputs)` as a CUDA graph -- for validation loops.

    `body` updates the tensors in the dict `acc` in place (sums, histograms,
    confusion matrices) from the batch tensors `inputs`, with no host-side
    effect: a graph replays only device work, so anything Python-side (a host
    counter, a flag) must be done by the caller, outside the body. Same kernels
    in the same order as calling `body` eagerly, so the accumulators come out
    identical.

    Warm-up runs on scratch copies of `acc`, so the real accumulators are only
    ever touched by real batches. Inputs of another shape (the last batch)
    run eagerly. Capture failure falls back to eager, with a note.
    """

    def __init__(self, body: Callable, acc: dict, enabled: bool = True,
                 log: Optional[Callable[[str], None]] = print):
        self.body, self.acc = body, acc
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.log = log
        self.error = None
        self._g = None
        self._sig = None
        self.n_graphed = 0
        self.n_eager = 0

    def __call__(self, *inputs):
        if self.enabled and all(torch.is_tensor(x) and x.is_cuda for x in inputs):
            sig = GraphedLoss._signature(inputs)
            if self._g is None:
                self._capture(inputs)
            if self._g is not None and sig == self._sig:
                for s, x in zip(self._static, inputs):
                    s.copy_(x, non_blocking=True)
                self._g.replay()
                self.n_graphed += 1
                return
        self.n_eager += 1
        self.body(self.acc, *inputs)

    def _capture(self, inputs):
        dev = inputs[0].device
        cuda_rng, cpu_rng = torch.cuda.get_rng_state(dev), torch.get_rng_state()
        try:
            static = [x.detach().clone() for x in inputs]
            scratch = {k: v.clone() for k, v in self.acc.items()}
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(2):
                    self.body(scratch, *static)
            torch.cuda.current_stream(dev).wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, stream=side, capture_error_mode="thread_local"):
                self.body(self.acc, *static)
            self._g, self._static = g, static
            self._sig = GraphedLoss._signature(inputs)
        except Exception as e:
            self._g = None
            self.enabled = False
            self.error = f"{type(e).__name__}: {e}"
            if self.log is not None:
                self.log(f"  validation cuda graph unavailable ({self.error}); running eagerly")
        finally:
            torch.cuda.set_rng_state(cuda_rng, dev)
            torch.set_rng_state(cpu_rng)
