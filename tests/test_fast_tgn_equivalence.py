"""The opt-in fast TGN step (bita/fast) against the reference, on one GPU.

Two copies of the same ExtendedTGN (same weights) run the same batches, each
batch from the same RNG seed so dropout draws the same masks, under
deterministic algorithms (the reference's index_add_ is otherwise atomic and
not reproducible even against itself). No optimizer step, so weights stay
equal and differences cannot compound.

Level 1: every forward output, the memory table, last_update and the pending
messages are BIT-IDENTICAL; gradients agree to fp32 rounding (they are summed
in a different order). Level 2 (Triton BiGRU): forward within fp32 rounding.
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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Data:
    def __init__(self, s, d, t, e):
        self.sources, self.destinations, self.timestamps, self.edge_idxs = s, d, t, e


def _graph(seed=0, n_nodes=120, n_edges=2600):
    rng = np.random.default_rng(seed)
    # A few hubs (repeated peers, long per-edge message sequences) plus a tail.
    hub = rng.random(n_edges) < 0.4
    src = np.where(hub, rng.integers(1, 6, n_edges), rng.integers(1, n_nodes, n_edges))
    dst = rng.integers(1, n_nodes, n_edges)
    dst = np.where(dst == src, dst % (n_nodes - 1) + 1, dst)
    ts = 1.3e9 + np.sort(rng.integers(0, 4000, n_edges)).astype(np.float64)   # ties included
    eidx = np.arange(1, n_edges + 1)
    edge_features = rng.normal(size=(n_edges + 1, 12)).astype(np.float32)
    node_features = rng.normal(size=(n_nodes + 1, 12)).astype(np.float32)
    return src, dst, ts, eidx, edge_features, node_features


def _model(edge_features, node_features, finder, seed=0):
    from model.extentedtgn import ExtendedTGN
    from model.time_encoding import TimeEncode
    torch.manual_seed(seed)
    m = ExtendedTGN(neighbor_finder=finder, node_features=node_features, edge_features=edge_features,
                    device=torch.device("cuda"), n_layers=1, n_heads=2, dropout=0.1, use_memory=True,
                    message_dimension=100, memory_dimension=12, embedding_module_type="graph_attention",
                    message_function="identity", aggregator_type="bigru_transformer",
                    memory_updater_type="gru", n_neighbors=10, num_categories=2).cuda()
    for mod in m.modules():
        if isinstance(mod, TimeEncode):
            mod.requires_grad_(False)
    return m


def _run(level, n_batches=24, bs=64, backprop_every=4):
    from utils.utils import get_neighbor_finder
    from fast import enable_fast_tgn
    src, dst, ts, eidx, ef, nf = _graph()
    finder = get_neighbor_finder(_Data(src, dst, ts, eidx), uniform=False, max_node_idx=len(nf) - 1)
    ref = _model(ef, nf, finder)
    fast = _model(ef, nf, finder)
    fast.load_state_dict(ref.state_dict())
    enable_fast_tgn(fast, level=level)
    rng = np.random.default_rng(1)
    for m in (ref, fast):
        m.train()
        m.memory.__init_memory__()
    worst = dict(out=0.0, grad=0.0)
    for g in range(0, n_batches, backprop_every):
        for m in (ref, fast):
            m.zero_grad(set_to_none=True)
        totals = {}
        outs = {id(ref): [], id(fast): []}
        negs = [rng.integers(1, len(nf), bs) for _ in range(backprop_every)]
        for m in (ref, fast):
            total = 0.0
            for j in range(backprop_every):
                b = g + j
                sl = slice(b * bs, (b + 1) * bs)
                torch.manual_seed(1000 + b)
                p, n, logits = m.compute_edge_probabilities_and_categories(
                    src[sl], dst[sl], negs[j], ts[sl], eidx[sl], n_neighbors=10)
                outs[id(m)].append((p, n, logits))
                total = total + p.sum() + (1 - n).sum() + logits.square().sum()
            totals[id(m)] = total
        for (a, b) in zip(outs[id(ref)], outs[id(fast)]):
            for x, y in zip(a, b):
                d = float((x - y).abs().max())
                worst["out"] = max(worst["out"], d)
                if level == 1:
                    assert torch.equal(x, y), f"level 1 forward must be bit-identical (max |d| {d})"
        for m in (ref, fast):
            totals[id(m)].backward()
        for (name, pa), (_, pb) in zip(ref.named_parameters(), fast.named_parameters()):
            if pa.grad is None and pb.grad is None:
                continue
            ga, gb = pa.grad, pb.grad
            rel = float((ga - gb).abs().max() / ga.abs().max().clamp_min(1e-12))
            worst["grad"] = max(worst["grad"], rel)
        if level == 1:
            assert torch.equal(ref.memory.memory.detach(), fast.memory.memory.detach())
            assert torch.equal(ref.memory.last_update.detach(), fast.memory.last_update.detach())
            _assert_same_pending(ref.memory, fast.memory)
        for m in (ref, fast):
            m.memory.detach_memory()
    return worst


def _assert_same_pending(ref_mem, fast_mem):
    pool = fast_mem._pool
    nodes = [k for k, v in ref_mem.messages.items() if len(v)]
    assert sorted(nodes) == sorted(np.flatnonzero(pool.cnt > 0).tolist())
    for k in nodes:
        msgs = ref_mem.messages[k]
        rows = np.arange(pool.start[k], pool.start[k] + pool.cnt[k])
        assert torch.equal(torch.stack([m[0].detach() for m in msgs]), pool.pool[torch.from_numpy(rows).cuda()])
        assert np.array_equal(np.array([float(m[1]) for m in msgs]), pool.row_t[rows])
        assert np.array_equal(np.array([int(m[2]) for m in msgs]), pool.row_peer[rows])


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.use_deterministic_algorithms(prev)
    torch.backends.cudnn.allow_tf32 = prev_tf32


def test_level1_forward_bit_identical_grads_rounding(deterministic):
    worst = _run(level=1)
    assert worst["out"] == 0.0
    assert worst["grad"] < 1e-5, worst


def test_level2_within_fp32_rounding(deterministic):
    worst = _run(level=2)
    assert worst["out"] < 1e-5, worst
    assert worst["grad"] < 1e-4, worst


def test_backup_restore_round_trip(deterministic):
    from utils.utils import get_neighbor_finder
    from fast import enable_fast_tgn
    src, dst, ts, eidx, ef, nf = _graph(seed=3)
    m = _model(ef, nf, get_neighbor_finder(_Data(src, dst, ts, eidx), uniform=False, max_node_idx=len(nf) - 1))
    enable_fast_tgn(m, level=1)
    m.eval()
    m.memory.__init_memory__()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for b in range(10):
            sl = slice(b * 64, (b + 1) * 64)
            m.compute_edge_probabilities_and_categories(src[sl], dst[sl], rng.integers(1, len(nf), 64),
                                                        ts[sl], eidx[sl], n_neighbors=10)
        snap = m.memory.backup_memory()
        sl = slice(640, 704)
        neg = rng.integers(1, len(nf), 64)
        a = m.compute_edge_probabilities_and_categories(src[sl], dst[sl], neg, ts[sl], eidx[sl], n_neighbors=10)
        m.memory.restore_memory(snap)
        b = m.compute_edge_probabilities_and_categories(src[sl], dst[sl], neg, ts[sl], eidx[sl], n_neighbors=10)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
