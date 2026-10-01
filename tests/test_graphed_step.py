"""GraphedLoss (cyberworld_v4/graphed_step.py) must train EXACTLY like eager.

Same losses at every step and the same final weights, with dropout live, a
partial batch in the middle (eager fallback), and the sync-free guard step --
the configuration the trainers run.
"""

import pytest
import torch

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from cyberworld_v4.graphed_step import GraphedLoss
from cyberworld_v4.training_guard import TrainingGuard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _data(n, B=128):
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(n):
        b = 37 if i == n // 2 else B                  # one partial batch
        out.append((torch.randn(b, 15, 27, generator=g), -torch.rand(b, 15, generator=g) * 100,
                    torch.rand(b, generator=g), torch.randint(0, 14, (b,), generator=g),
                    torch.randint(0, 4, (b,), generator=g)))
    return out


def _run(data, graphed):
    torch.manual_seed(0)
    m = MultiTaskLSTM(input_dim=27, num_techniques=14, num_gradations=4,
                      risk_objective="soft_bce", **MultiTaskLSTM.PAPER_ARCH).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = TrainingGuard("a", [m], opt, mode="max", patience=3, step_back_after=2,
                      warmup_steps=20, clip_norm=1.0, log=lambda s: None)
    m.train()
    torch.manual_seed(123)

    def f(x, t, r, tc, gr):
        return m.compute_loss(m(x, t_history=t), {"risk": r, "technique": tc, "gradation": gr})[0]
    fn = GraphedLoss(f, [m], enabled=graphed)
    losses = []
    for d in data:
        d = [t.cuda() for t in d]
        opt.zero_grad(set_to_none=True)
        loss = fn(*d)
        g.backward_step_deferred(loss)
        m.uncertainty_loss.project_()
        losses.append(loss.detach().clone())
    g.flush()
    return torch.stack(losses).cpu(), {k: v.cpu() for k, v in m.state_dict().items()}, fn


def test_graphed_training_is_bit_identical_to_eager():
    data = _data(120)
    L0, W0, _ = _run(data, graphed=False)
    L1, W1, fn = _run(data, graphed=True)
    assert fn.n_graphed == 119 and fn.n_eager == 1
    assert torch.equal(L0, L1)
    for k in W0:
        assert torch.equal(W0[k].nan_to_num(3.0), W1[k].nan_to_num(3.0)), k


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_support_index_loss_matches_mask(device):
    from deepop_decoder.forecast_decoder import smoothed_and_plain_ce
    g = torch.Generator().manual_seed(3)
    for sup_list in ([0, 3, 4, 9], list(range(10)), [5]):
        sup = torch.zeros(10, dtype=torch.bool)
        sup[sup_list] = True
        logits = torch.randn(64, 5, 10, generator=g).to(device).requires_grad_(True)
        tgt = torch.randint(0, 10, (64, 5), generator=g).to(device)
        a, pa = smoothed_and_plain_ce(logits, tgt, support=sup.to(device))
        ga, = torch.autograd.grad(a, logits)
        b, pb = smoothed_and_plain_ce(logits, tgt, support=sup.to(device),
                                      support_index=sup.nonzero().squeeze(1).to(device))
        gb, = torch.autograd.grad(b, logits)
        assert torch.equal(a, b) and torch.equal(pa, pb) and torch.equal(ga, gb)


def _guard_run(graph_undo):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(27, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                            torch.nn.Linear(64, 3)).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = TrainingGuard("t", [m], opt, mode="min", patience=3, step_back_after=2, warmup_steps=5,
                      clip_norm=1.0, log=lambda s: None, graph_undo=graph_undo)
    gen = torch.Generator().manual_seed(1)
    torch.manual_seed(5)
    for i in range(80):
        x = torch.randn(32, 27, generator=gen).cuda()
        y = torch.randn(32, 3, generator=gen).cuda()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(m(x), y)
        if i in (13, 14, 40):
            loss = loss * float("nan")          # a skipped step: the undo must restore exactly
        if i == 50:
            g.flush()                           # optimizer state reloaded: tensors move
            opt.load_state_dict(opt.state_dict())
        g.backward_step_deferred(loss)
    g.flush()
    return ({k: v.cpu() for k, v in m.state_dict().items()},
            (g.n_steps, g.n_nonfinite, g.n_grad, g.n_clipped, g.grad_norm_sum, g.global_step))


def test_guard_graph_undo_is_bit_identical():
    w0, c0 = _guard_run(False)
    w1, c1 = _guard_run(True)
    assert c0 == c1 and c0[1] == 3
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


def _whole_run(mode, n=150):
    from cyberworld_v4.graphed_step import WholeStepGraph
    torch.manual_seed(0)
    m = MultiTaskLSTM(input_dim=27, num_techniques=14, num_gradations=4,
                      risk_objective="soft_bce", **MultiTaskLSTM.PAPER_ARCH).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = TrainingGuard("a", [m], opt, mode="max", patience=3, step_back_after=2,
                      warmup_steps=20, clip_norm=1.0, log=lambda s: None, graph_undo=True)
    m.train()

    def f(x, t, r, tc, gr):
        return m.compute_loss(m(x, t_history=t), {"risk": r, "technique": tc, "gradation": gr})[0]
    whole = WholeStepGraph(f, [m], log=None)
    data = _data(n)
    torch.manual_seed(123)
    losses, epoch_stats = [], []
    for i, d in enumerate(data):
        d = [t.cuda() for t in d]
        if i in (30, 31, 97):
            d[0] = d[0].clone()
            d[0][0, 0, 0] = float("inf")          # non-finite loss: the step must be undone
        if i == 75:                                # epoch boundary + optimizer reload (tensors move)
            g.flush()
            epoch_stats.append((g.n_steps, g.n_nonfinite, g.n_grad, g.n_clipped, g.grad_norm_sum))
            g._reset_epoch_stats()
            opt.load_state_dict(opt.state_dict())
        opt.zero_grad(set_to_none=True)
        if mode == "whole":
            loss, ok = g.deferred_step_graphed(whole, d)
        else:
            loss = f(*d)
            ok = g.backward_step_deferred(loss)
        m.uncertainty_loss.project_()
        losses.append(loss.detach().clone())
    g.flush()
    epoch_stats.append((g.n_steps, g.n_nonfinite, g.n_grad, g.n_clipped, g.grad_norm_sum, g.global_step))
    return (torch.stack(losses).cpu(), {k: v.cpu() for k, v in m.state_dict().items()},
            epoch_stats, whole)


def test_whole_step_graph_is_bit_identical():
    L0, W0, S0, _ = _whole_run("eager")
    L1, W1, S1, whole = _whole_run("whole")
    assert whole.n_graphed > 100, whole.n_graphed
    assert S0 == S1 and S0[0][1] == 2 and S0[1][1] == 1
    assert torch.equal(L0.nan_to_num(9.0, 9.0, -9.0), L1.nan_to_num(9.0, 9.0, -9.0))
    for k in W0:
        assert torch.equal(W0[k].nan_to_num(3.0), W1[k].nan_to_num(3.0)), k
