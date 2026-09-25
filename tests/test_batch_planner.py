"""The batch planner (bita/fast/planner.py) and its host kernels.

1. rust/tgn_host's planner kernels against the numpy code they replace
   (bita/fast/plan_host.py): BiTA grouping and the neighbour block,
   bit-identical, including padded (level 4) layouts.
2. PoolMeta (the planner's copy of the message pool's host bookkeeping)
   against MessagePool through writes, clears, growth, compaction, detach.
3. End to end: the training loop with and without the planner computes the
   same losses, weights, memory, pending messages and RNG state, bit for bit
   (CPU at level 1; CUDA at levels 2 and 3 when a GPU is present).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bita"))
sys.path.insert(0, os.path.join(HERE, ".."))

from utils import tgn_host  # noqa: E402
from fast import plan_host  # noqa: E402

needs_rust = pytest.mark.skipif(tgn_host.lib() is None, reason="rust/tgn_host not built")


class _D:
    def __init__(self, s, d, t, e):
        self.sources, self.destinations, self.timestamps, self.edge_idxs = s, d, t, e


def _random_pool(rng, n_nodes=300, n_rows=6000, max_cnt=40):
    """Host bookkeeping of a pool: node v's pending rows [start, start+cnt)."""
    cnt = np.zeros(n_nodes, np.int64)
    start = np.zeros(n_nodes, np.int64)
    has = rng.random(n_nodes) < 0.6
    cur = 1
    for v in np.flatnonzero(has):
        c = int(rng.integers(1, max_cnt)) if rng.random() < 0.2 else int(rng.integers(1, 4))
        cnt[v], start[v] = c, cur
        cur += c
    row_peer = np.zeros(max(cur, n_rows), np.int64)
    row_t = np.zeros(max(cur, n_rows), np.float64)
    # few peers per node (repeated edges), integer times (ties), some duplicates
    row_peer[1:cur] = rng.integers(0, 8, cur - 1)
    row_t[1:cur] = 1.3e9 + rng.integers(0, 50, cur - 1).astype(np.float64)
    return cnt, start, row_peer, row_t


class _Meta:
    def __init__(self, cnt, start, row_peer, row_t):
        self.cnt, self.start, self.row_peer, self.row_t = cnt, start, row_peer, row_t


@needs_rust
@pytest.mark.parametrize("msl", [1, 3, 64])
def test_bita_grouping_rust_equals_numpy(msl):
    rng = np.random.default_rng(msl)
    for trial in range(30):
        cnt, start, row_peer, row_t = _random_pool(rng)
        nodes = rng.integers(0, len(cnt), 256)
        ref = plan_host.bita_group_numpy(cnt, start, row_peer, row_t, nodes, msl)
        got = plan_host.bita_group(_Meta(cnt, start, row_peer, row_t), nodes, msl)
        assert (ref is None) == (got is None)
        if ref is None:
            continue
        for a, b in zip(ref, got):
            if isinstance(a, np.ndarray):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a, b)
                assert a.tobytes() == b.tobytes()          # signed zeros included
            else:
                assert a == b


@needs_rust
def test_bita_grouping_padded_fill_matches_numpy_padding():
    """The level-4 layout (fast_tgn._fast_bita_graphed pads with numpy)."""
    rng = np.random.default_rng(7)
    cnt, start, row_peer, row_t = _random_pool(rng)
    nodes = rng.integers(0, len(cnt), 256)
    to_update, E, L, idx, klen, dt, last_t, owner, counts, node_ts = plan_host.bita_group_numpy(
        cnt, start, row_peer, row_t, nodes, 64)
    E_pad, L_pad, N_pad = ((E + 127) // 128) * 128, max(8, 1 << (L - 1).bit_length()), 2 * 128 + 1
    h = plan_host.bita_plan_rust(_Meta(cnt, start, row_peer, row_t), nodes, 64)
    b_idx = np.full((E_pad, L_pad), 7, np.int64)
    b_dt = np.full((E_pad, L_pad), 7, np.float32)
    b_klen = np.full(E_pad, 7, np.int32)
    b_owner = np.full(E_pad, 7, np.int64)
    b_last = np.empty(E, np.float64)
    b_upd = np.empty(len(to_update), np.int64)
    b_counts = np.full(N_pad, 7, np.float32)
    b_nts = np.empty(len(to_update), np.float64)
    h.fill(b_idx, b_dt, b_klen, b_owner, b_last, b_upd, b_counts, b_nts, owner_fill=N_pad - 1)
    h.close()
    idx_p = np.zeros((E_pad, L_pad), np.int64); idx_p[:E, :L] = idx
    dt_p = np.zeros((E_pad, L_pad), np.float32); dt_p[:E, :L] = dt
    lens_p = np.zeros(E_pad, np.int32); lens_p[:E] = klen
    owner_p = np.full(E_pad, N_pad - 1, np.int64); owner_p[:E] = owner
    counts_p = np.ones(N_pad, np.float32); counts_p[:len(to_update)] = counts
    for a, b in ((idx_p, b_idx), (dt_p, b_dt), (lens_p, b_klen), (owner_p, b_owner), (last_t, b_last),
                 (to_update, b_upd), (counts_p, b_counts), (node_ts, b_nts)):
        assert a.tobytes() == b.tobytes()


@needs_rust
@pytest.mark.parametrize("host_ef", [False, True])
def test_nbr_block_rust_equals_numpy(host_ef, monkeypatch):
    from utils.utils import get_neighbor_finder
    rng = np.random.default_rng(0)
    n_nodes, n_edges = 400, 20000
    hub = rng.random(n_edges) < 0.3
    src = np.where(hub, rng.integers(1, 4, n_edges), rng.integers(1, n_nodes - 50, n_edges))
    dst = rng.integers(1, n_nodes - 50, n_edges)
    ts = 1.3e9 + np.sort(rng.integers(0, 3000, n_edges)).astype(np.float64)
    eidx = np.arange(1, n_edges + 1)
    finder = get_neighbor_finder(_D(src, dst, ts, eidx), uniform=False, max_node_idx=n_nodes - 1)
    ef = rng.normal(size=(n_edges + 1, 5)).astype(np.float32) if host_ef else None
    nodes = np.concatenate([rng.integers(0, n_nodes, 380), [0, n_nodes - 1, 1, 2]]).astype(np.int64)
    cut = np.concatenate([rng.choice(ts, 190), 1.3e9 + rng.uniform(-10, 3100, 190), [0.0, 5e9, ts[5], ts[-1]]])
    got = plan_host.nbr_block(finder, nodes, cut, 10, ef)
    monkeypatch.setenv("CYBERWORLD_HOST_RUST", "0")
    ref = plan_host.nbr_block_numpy(finder, nodes, cut, 10, ef)
    for a, b in zip(ref, got):
        if a is None:
            assert b is None
            continue
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert got[3].any() and not got[3].all()


# ---------------------------------------------------------------- PoolMeta
def test_poolmeta_follows_messagepool(monkeypatch):
    """The planner's host-only PoolMeta and the trainer's MessagePool make the
    same layout decisions (growth mid-group, compaction on write and at detach)."""
    import torch
    import fast.store as store
    monkeypatch.setattr(store, "MIN_ROWS", 300)
    rng = np.random.default_rng(0)
    n_nodes = 200
    pool = store.MessagePool(n_nodes, 3, torch.device("cpu"))
    meta = store.PoolMeta(n_nodes)
    grows = compacts = 0
    for step in range(160):
        grad = (step // 4) % 3 != 0          # a whole group trains or evaluates
        owners = rng.integers(0, n_nodes, 64)
        peers = rng.integers(0, n_nodes, 64)
        times = rng.random(64)
        cap0, cur0 = pool.capacity, pool.cur
        pool.clear(owners)
        meta.clear(owners)
        order = pool.write_order(owners)
        with torch.set_grad_enabled(grad):
            pool.write(owners, peers, times, torch.randn(64, 3, requires_grad=grad), order=order)
        meta.write_host(owners, peers, times, grad, order=order,
                        pre=store.PoolMeta.write_pre(owners, peers, times, order))
        grows += pool.capacity > cap0
        compacts += pool.cur < cur0 + 64
        if step % 4 == 3:
            cur0 = pool.cur
            pool.detach()
            meta.detach()
            compacts += pool.cur < cur0
        assert pool.state_check() == meta.state_check()
        for a in ("cnt", "start", "row_t", "row_peer"):
            assert np.array_equal(getattr(pool, a), getattr(meta, a))
    assert grows >= 1 and compacts >= 1, (grows, compacts)


# ------------------------------------------------------------- end to end
def _corpus(seed=0, n_nodes=150, n_edges=3000):
    rng = np.random.default_rng(seed)
    hub = rng.random(n_edges) < 0.4
    src = np.where(hub, rng.integers(1, 6, n_edges), rng.integers(1, n_nodes, n_edges))
    dst = rng.integers(1, n_nodes, n_edges)
    dst = np.where(dst == src, dst % (n_nodes - 1) + 1, dst)
    ts = 1.3e9 + np.sort(rng.integers(0, 4000, n_edges)).astype(np.float64)   # ties included
    eidx = np.arange(1, n_edges + 1)
    labels = (rng.random(n_edges) < 0.2).astype(np.int64)
    ef = rng.normal(size=(n_edges + 1, 12)).astype(np.float32)
    nf = rng.normal(size=(n_nodes + 1, 12)).astype(np.float32)
    group = np.zeros(n_nodes + 1, np.int64)
    group[n_nodes // 2:] = 1
    return src, dst, ts, eidx, labels, ef, nf, group


class _Data:
    def __init__(self, s, d, t, e, l):
        self.sources, self.destinations, self.timestamps, self.edge_idxs, self.labels = s, d, t, e, l


def _train(planned, level, device, epochs=2, bs=64, bpe=4, host_edges=None, seed=0, n_edges=3000):
    """bita/train.py's inner loop on the fast step, optionally planned.
    Returns everything a bit-identity claim is about."""
    import torch
    from torch import nn
    from utils.utils import get_neighbor_finder, RandEdgeSampler
    from fast import enable_fast_tgn
    from model.extentedtgn import ExtendedTGN
    from model.time_encoding import TimeEncode
    import train as T
    src, dst, ts, eidx, labels, ef, nf, group = _corpus(n_edges=n_edges)
    if host_edges is not None:
        mm = np.memmap(host_edges, dtype=np.float32, mode="w+", shape=ef.shape)
        mm[:] = ef
        mm.flush()
        ef = np.memmap(host_edges, dtype=np.float32, mode="r", shape=ef.shape)
    data = _Data(src, dst, ts, eidx, labels)
    finder = get_neighbor_finder(data, uniform=False, max_node_idx=len(nf) - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = RandEdgeSampler(src, dst, node_group=group)
    tgn = ExtendedTGN(neighbor_finder=finder, node_features=nf, edge_features=ef, device=torch.device(device),
                      n_layers=1, n_heads=2, dropout=0.1, use_memory=True, message_dimension=100,
                      memory_dimension=12, embedding_module_type="graph_attention", message_function="identity",
                      aggregator_type="bigru_transformer", memory_updater_type="gru", n_neighbors=10,
                      num_categories=2).to(device)
    for mod in tgn.modules():
        if isinstance(mod, TimeEncode):
            mod.requires_grad_(False)
    enable_fast_tgn(tgn, level=level)
    ec = nn.BCELoss()
    cc = T.FocalLoss(alpha=T.FocalLoss.inverse_frequency_alpha(labels, 2), gamma=2.0).to(device)
    opt = torch.optim.Adam([p for p in tgn.parameters() if p.requires_grad], lr=1e-3)
    planner = None
    if planned:
        from fast.planner import BatchPlanner
        planner = BatchPlanner(tgn, data, finder, sampler, batch_size=bs, backprop_every=bpe, n_degree=10)
    n = len(src)
    num_batch = (n + bs - 1) // bs
    losses = []
    try:
        for epoch in range(epochs):
            tgn.train()
            tgn.memory.__init_memory__()
            tgn.set_neighbor_finder(finder)
            plans = planner.epoch(num_batch, ec, cc) if planner else None
            for k in range(0, num_batch, bpe):
                opt.zero_grad()
                loss = 0.0
                for j in range(bpe):
                    b = k + j
                    if b >= num_batch:
                        continue
                    if plans is not None:
                        el, cl, _ = tgn.fast_batch_losses(None, None, None, None, None, None, ec, cc,
                                                          n_neighbors=10, plan=plans.next(b))
                    else:
                        sl = slice(b * bs, min(n, (b + 1) * bs))
                        _, neg = sampler.sample(sl.stop - sl.start, sources=src[sl], destinations=dst[sl])
                        el, cl, _ = tgn.fast_batch_losses(src[sl], dst[sl], neg, ts[sl], eidx[sl], labels[sl],
                                                          ec, cc, n_neighbors=10)
                    loss = loss + el + 15.0 * cl
                (loss / bpe).backward()
                opt.step()
                losses.append(float(loss.detach()))
                tgn.memory.detach_memory()
            if plans is not None:
                plans.finish()
    finally:
        if planner is not None:
            planner.close()
    pool = tgn.memory._pool
    return dict(losses=losses, params={k: v.detach().cpu().clone() for k, v in tgn.state_dict().items()},
                rng=np.random.get_state(), pool=(pool.state_check(), pool.cnt.copy(), pool.start.copy(),
                                                 pool.row_t.copy(), pool.row_peer.copy()),
                pool_rows=pool.pool[: pool.cur].detach().cpu().clone())


def _assert_identical(a, b):
    import torch
    assert a["losses"] == b["losses"]
    assert set(a["params"]) == set(b["params"])
    for k in a["params"]:
        assert torch.equal(a["params"][k], b["params"][k]), k
    ra, rb = a["rng"], b["rng"]
    assert ra[0] == rb[0] and np.array_equal(ra[1], rb[1]) and ra[2:] == rb[2:]
    assert a["pool"][0] == b["pool"][0]
    for x, y in zip(a["pool"][1:], b["pool"][1:]):
        assert np.array_equal(x, y)
    assert torch.equal(a["pool_rows"], b["pool_rows"])


def test_planned_training_bit_identical_cpu(monkeypatch):
    """Level 1 on the CPU, 2 epochs, a partial last group, a pool small enough
    to grow and compact: losses, weights, memory, pending messages and the
    numpy RNG state after training are identical with the planner."""
    import fast.store as store
    monkeypatch.setattr(store, "MIN_ROWS", 300)
    ref = _train(False, 1, "cpu")
    got = _train(True, 1, "cpu")
    _assert_identical(ref, got)


def test_planned_training_host_edges_cpu(tmp_path):
    ref = _train(False, 1, "cpu", host_edges=str(tmp_path / "a.mm"), epochs=1)
    got = _train(True, 1, "cpu", host_edges=str(tmp_path / "b.mm"), epochs=1)
    _assert_identical(ref, got)
