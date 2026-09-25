"""Fast path level 3 (CUDA-graphed embedding + heads + losses) against level 2.

Same weights, same batches, same seeds, a real training loop (Adam through the
TrainingGuard, gradient accumulation over 4 batches, a short last batch that
runs eagerly): losses, weights, memory and pending messages must be
BIT-IDENTICAL, because the graphs replay the kernels eager launches and draw
the same dropout offsets (bita/fast/graphs.py).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bita"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _train(level, n_batches=27, bs=64, backprop_every=4, seed=0):
    from test_fast_tgn_equivalence import _Data, _graph, _model
    from utils.utils import get_neighbor_finder
    from fast import enable_fast_tgn
    from cyberworld_v4.training_guard import TrainingGuard
    import train as T
    src, dst, ts, eidx, ef, nf = _graph()
    labels = (np.arange(len(src)) % 3 == 0).astype(np.int64)
    finder = get_neighbor_finder(_Data(src, dst, ts, eidx), uniform=False, max_node_idx=len(nf) - 1)
    m = _model(ef, nf, finder, seed=seed)
    if level:                                   # level 0: the reference model, losses as bita/train.py
        enable_fast_tgn(m, level=level)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    guard = TrainingGuard("t", [m], opt, warmup_steps=2, clip_norm=100.0, log=lambda *_: None)
    ec = torch.nn.BCELoss()
    cc = T.FocalLoss(alpha=torch.tensor([0.5, 2.0]), gamma=2.0).cuda()
    rng = np.random.default_rng(1)
    torch.manual_seed(123)
    m.train()
    m.memory.__init_memory__()
    losses, outs = [], []
    n_total = len(src)
    for g in range(0, n_batches, backprop_every):
        opt.zero_grad()
        el, cl = 0.0, 0.0
        for j in range(backprop_every):
            b = g + j
            if b >= n_batches:
                continue
            sl = slice(b * bs, min((b + 1) * bs, n_total if b == n_batches - 1 else (b + 1) * bs))
            if b == n_batches - 1:
                sl = slice(b * bs, b * bs + bs // 2)          # a short last batch
            neg = rng.integers(1, len(nf), sl.stop - sl.start)
            if level:
                e, c, (p, n, lg) = m.fast_batch_losses(src[sl], dst[sl], neg, ts[sl], eidx[sl], labels[sl], ec, cc,
                                                       n_neighbors=10)
            else:
                p, n, lg = m.compute_edge_probabilities_and_categories(src[sl], dst[sl], neg, ts[sl], eidx[sl],
                                                                       n_neighbors=10)
                size = len(p)
                e = ec(p.squeeze(-1), torch.ones(size, device="cuda")) + \
                    ec(n.squeeze(-1), torch.zeros(size, device="cuda"))
                c = cc(lg, torch.as_tensor(labels[sl], device="cuda"))
            outs.append(torch.cat([p.detach().reshape(-1), n.detach().reshape(-1), lg.detach().reshape(-1)]).cpu())
            el = el + e
            cl = cl + c
        total = (el + 15.0 * cl) / backprop_every
        assert guard.backward_step(total)
        losses.append(float(total))
        m.memory.detach_memory()
    params = [p.detach().cpu().clone() for p in m.parameters()]
    return losses, outs, params, m.memory.memory.detach().cpu().clone(), m


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def test_level3_is_bit_identical_to_level2(deterministic):
    l2, o2, p2, m2, _ = _train(level=2)
    l3, o3, p3, m3, model3 = _train(level=3)
    assert getattr(model3, "_graph_slots", None), "level 3 captured no graph"
    assert l2 == l3
    for a, b in zip(o2, o3):
        assert torch.equal(a, b)
    for a, b in zip(p2, p3):
        assert torch.equal(a, b)
    assert torch.equal(m2, m3)


def test_level3_eval_and_no_grad_run_eagerly():
    _, _, _, _, m = _train(level=3, n_batches=8)
    n_slots = sum(len(v) for v in m._graph_slots.values())
    m.eval()
    from test_fast_tgn_equivalence import _graph
    src, dst, ts, eidx, ef, nf = _graph()
    with torch.no_grad():
        m.compute_edge_probabilities_and_categories(src[600:664], dst[600:664], dst[600:664], ts[600:664],
                                                    eidx[600:664], n_neighbors=10)
    assert sum(len(v) for v in m._graph_slots.values()) == n_slots


def test_level4_bita_graphs_within_fp32_rounding(deterministic):
    """Level 4 also replays BiTA + the memory updater on padded shapes: the
    padding is inert, so only fp32 rounding may differ (GEMMs over more rows).
    Measured against the envelope already accepted for level 2: level 4's
    distance from level 2 must not exceed level 2's distance from the
    reference (weights after Adam steps amplify rounding in small
    gradients, so both are compared on the same run)."""
    l0, o0, p0, m0, _ = _train(level=0)
    l2, o2, p2, m2, _ = _train(level=2)
    l4, o4, p4, m4, model4 = _train(level=4)
    kinds = {k[-1] for k in model4._graph_slots}
    assert "bita" in kinds, "level 4 captured no BiTA graph"

    def rel_loss(a, b):
        return max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b))

    def dmax(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    assert rel_loss(l2, l4) < 1e-5
    assert dmax(o2, o4) < 1e-5
    assert dmax(p2, p4) <= max(dmax(p0, p2), 1e-6), (dmax(p2, p4), dmax(p0, p2))
    assert float((m2 - m4).abs().max()) <= max(float((m0 - m2).abs().max()), 1e-6)
