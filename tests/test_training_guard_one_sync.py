"""TrainingGuard.backward_step reads the loss-finite flag and the gradient norm
in ONE device->host sync. Its decisions, counters, diagnostics and the
weights it produces must be exactly those of the original two-sync order
(isfinite before backward, the norm after), including across non-finite
losses and non-finite gradients."""

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from cyberworld_v4.training_guard import TrainingGuard  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the one-sync path is the device path")


def _reference_backward_step(g: TrainingGuard, loss) -> bool:
    """The original TrainingGuard.backward_step + step_after_backward."""
    g.n_steps += 1
    if not bool(torch.isfinite(loss.detach()).all()):
        g.optimizer.zero_grad(set_to_none=True)
        g.n_nonfinite += 1
        return False
    loss.backward()
    if g.n_grad == 0:
        g._record_top_grads()
    params = list(g._params())
    norm = torch.nn.utils.clip_grad_norm_(params, g.clip_norm if g.clip_norm else float("inf"))
    norm_f = float(norm)
    if not math.isfinite(norm_f):
        g.optimizer.zero_grad(set_to_none=True)
        g.n_nonfinite += 1
        return False
    g.grad_norm_sum += norm_f
    g.grad_norm_max = max(g.grad_norm_max, norm_f)
    g.n_grad += 1
    if g.clip_norm and norm_f > g.clip_norm:
        g.n_clipped += 1
    g._apply_lr()
    g.optimizer.step()
    g.global_step += 1
    return True


def _make(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=1e-5)
    g = TrainingGuard("t", [m], opt, warmup_steps=4, clip_norm=0.5, log=lambda *_: None)
    return m, opt, g


# step index -> what goes wrong: a NaN/inf loss, or a NaN only in the gradient
_FAULTS = {0: "nan_loss", 3: "nan_loss", 5: "inf_loss", 7: "nan_grad", 9: "big", 12: "nan_grad"}


def _run(step_fn):
    m, opt, g = _make()
    torch.manual_seed(1)
    out = []
    for epoch in range(2):
        for i in range(15):
            opt.zero_grad()
            x = torch.randn(32, 6, device="cuda")
            loss = m(x).square().mean()
            f = _FAULTS.get(i)
            if f == "nan_loss":
                loss = loss * float("nan")
            elif f == "inf_loss":
                loss = loss + float("inf")
            elif f == "nan_grad":
                loss = loss + (m[0].weight.sum() * 0.0).sqrt()      # finite loss, NaN gradient
            elif f == "big":
                loss = loss * 1e4
            out.append(step_fn(g, loss))
        stats = (g.n_steps, g.n_nonfinite, g.n_grad, g.n_clipped, g.grad_norm_sum, g.grad_norm_max,
                 copy.deepcopy(g.top_grads), g.global_step, g.current_lr())
        out.append(stats)
        g._reset_epoch_stats()
    return out, [p.detach().cpu().clone() for p in m.parameters()], opt.state_dict()


def test_one_sync_step_is_the_two_sync_step():
    ref_out, ref_params, ref_opt = _run(_reference_backward_step)
    new_out, new_params, new_opt = _run(lambda g, loss: g.backward_step(loss))
    assert ref_out == new_out
    assert any(o is False for o in ref_out) and any(o is True for o in ref_out)
    # both kinds of skip happened: a non-finite loss and a non-finite gradient
    assert ref_out[-1][1] == 5 and ref_out[-1][0] == 15
    for a, b in zip(ref_params, new_params):
        assert torch.equal(a, b)
    for k, st in ref_opt["state"].items():
        for name, v in st.items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(new_opt["state"][k][name])), name
