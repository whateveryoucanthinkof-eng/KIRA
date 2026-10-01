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
