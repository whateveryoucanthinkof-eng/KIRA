"""Host-side, value-independent parts of the fast TGN step (numpy + Rust, no torch).

Everything here is a function of the data, the batch order and the message
pool's HOST bookkeeping (fast.store.PoolMeta) -- never of a model value -- so
the batch planner (fast/planner.py) can run it ahead of the trainer, in
another process. The trainer's own path calls the same functions, so the
planned and unplanned steps share one definition:

  bita_group      BiTA grouping of a batch's pending messages (was inline in
                  fast_tgn._fast_bita; bita_group_numpy is that code verbatim)
  nbr_block       neighbour lookup + the attention mask/time deltas + host
                  edge-feature rows (was inline in fast_tgn._fast_nbr)

Each has a numpy definition and a rust/tgn_host kernel that computes the same
bits (tests/test_batch_planner.py); CYBERWORLD_HOST_RUST=0 forces numpy.
"""
from __future__ import annotations

import os

import numpy as np


# ------------------------------------------------------------------ BiTA
def bita_group_numpy(cnt, start, row_peer, row_t, nodes_np, max_seq_len):
    """(to_update, E, L, idx, klen, dt, last_t, owner, counts, node_ts) or None.

    BiTAAggregator.aggregate's grouping for the nodes in `nodes_np` with
    pending messages: node ascending (np.unique), one edge per peer in order
    of the peer's first message (dict order), each edge's messages sorted by
    time (stable), the last max_seq_len kept, left-aligned, zero-padded."""
    cand = np.unique(nodes_np)
    to_update = cand[cnt[cand] > 0]
    n_nodes = len(to_update)
    if n_nodes == 0:
        return None
    cnts = cnt[to_update]
    R = int(cnts.sum())
    seg0 = np.cumsum(cnts) - cnts
    node_pos = np.repeat(np.arange(n_nodes), cnts)
    ins = np.arange(R) - np.repeat(seg0, cnts)
    rows = np.repeat(start[to_update], cnts) + ins          # insertion order
    peer = row_peer[rows]
    t = row_t[rows]
    # Latest message time per node, over ALL its pending messages.
    node_ts = np.maximum.reduceat(t, seg0)

    # Edges in the reference's order: node ascending (np.unique), then
    # peers in order of first appearance in the node's list (dict order).
    o = np.lexsort((ins, peer, node_pos))
    new_grp = np.ones(R, dtype=bool)
    new_grp[1:] = (node_pos[o][1:] != node_pos[o][:-1]) | (peer[o][1:] != peer[o][:-1])
    gid = np.cumsum(new_grp) - 1
    first_ins = np.empty(R, dtype=np.int64)
    first_ins[o] = ins[o][new_grp][gid]
    # Within an edge: `sorted(seq, key=time)`, which is stable.
    f = np.lexsort((ins, t, first_ins, node_pos))
    np_f, fi_f = node_pos[f], first_ins[f]
    new_e = np.ones(R, dtype=bool)
    new_e[1:] = (np_f[1:] != np_f[:-1]) | (fi_f[1:] != fi_f[:-1])
    eid = np.cumsum(new_e) - 1
    estart = np.flatnonzero(new_e)
    E = len(estart)
    elen = np.diff(np.append(estart, R))
    pos_in_e = np.arange(R) - estart[eid]
    drop = np.maximum(elen - max_seq_len, 0)                      # keep the last max_seq_len
    keep = pos_in_e >= drop[eid]
    klen = elen - drop
    col = (pos_in_e - drop[eid])[keep]
    L = int(klen.max())
    idx = np.zeros((E, L), dtype=np.int64)                        # pool row 0 is all zeros
    idx[eid[keep], col] = rows[f][keep]
    times = np.zeros((E, L), dtype=np.float64)
    times[eid[keep], col] = t[f][keep]
    last_t = times[np.arange(E), klen - 1]
    valid = np.arange(L)[None, :] < klen[:, None]
    dt = ((np.maximum(last_t[:, None] - times, 0.0)) * valid).astype(np.float32)
    owner = np_f[new_e]
    counts = np.bincount(owner, minlength=n_nodes).astype(np.float32)
    return to_update, E, L, idx, klen, dt, last_t, owner, counts, node_ts


def bita_plan_rust(meta, nodes_np, max_seq_len):
    """The rust/tgn_host grouping handle (tgn_host.BitaPlan) or None."""
    from utils.tgn_host import bita_plan
    return bita_plan(meta.cnt, meta.start, meta.row_peer, meta.row_t, nodes_np, max_seq_len)


def bita_group(meta, nodes_np, max_seq_len):
    """bita_group_numpy over `meta` (a fast.store.PoolMeta / MessagePool),
    through rust/tgn_host when available: the same arrays, bit for bit."""
    h = bita_plan_rust(meta, nodes_np, max_seq_len)
    if h is None:
        return bita_group_numpy(meta.cnt, meta.start, meta.row_peer, meta.row_t, nodes_np, max_seq_len)
    try:
        if h.n_upd == 0:
            return None
        E, L, n = h.E, h.L, h.n_upd
        idx = np.empty((E, L), np.int64)
        dt = np.empty((E, L), np.float32)
        klen32 = np.empty(E, np.int32)
        owner = np.empty(E, np.int64)
        last_t = np.empty(E, np.float64)
        to_update = np.empty(n, np.int64)
        counts = np.empty(n, np.float32)
        node_ts = np.empty(n, np.float64)
        h.fill(idx, dt, klen32, owner, last_t, to_update, counts, node_ts)
        return to_update, E, L, idx, klen32.astype(np.int64), dt, last_t, owner, counts, node_ts
    finally:
        h.close()


# ------------------------------------------------------------ neighbours
def _rust_finder_ok(finder, k):
    return (getattr(finder, "_csr", None) is not None and not finder.uniform and k > 0
            and os.environ.get("TGNE_REFERENCE_SAMPLER", "") not in ("1", "true", "True"))


def nbr_block_numpy(finder, nodes, ts3, k, host_edges):
    """(ints, deltas, mask_fixed, invalid, ef_n): the host part of
    fast_tgn._fast_nbr. ints = [nodes ; neighbours ; edge idxs] (int64)."""
    neighbors, edge_idxs, edge_times = finder.get_temporal_neighbor(nodes, ts3, n_neighbors=k)
    deltas = (ts3[:, None] - edge_times).astype(np.float32)
    mask = neighbors == 0
    invalid = mask.all(axis=1, keepdims=True)
    mask_fixed = mask.copy()
    mask_fixed[invalid[:, 0], 0] = False
    nbr = neighbors.reshape(-1).astype(np.int64)
    all_nodes = np.concatenate([nodes, nbr])
    ints = np.concatenate([all_nodes, edge_idxs.reshape(-1).astype(np.int64)])
    ef_n = None if host_edges is None else np.take(host_edges, edge_idxs.astype(np.int64), axis=0)
    return ints, deltas, mask_fixed, invalid, ef_n


def nbr_block_into(finder, nodes, ts3, k, host_edges, ints, deltas, mask, invalid, ef_n):
    """nbr_block_numpy written into the given buffers, through rust/tgn_host
    when it can (same bits); numpy otherwise."""
    if _rust_finder_ok(finder, k):
        from utils.tgn_host import nbr_plan
        flat_nbr, flat_eidx, flat_ts, offsets = finder._csr
        if nbr_plan(flat_nbr, flat_eidx, flat_ts, offsets, np.ascontiguousarray(nodes, np.int64),
                    np.ascontiguousarray(ts3, np.float64), k,
                    None if host_edges is None else np.asarray(host_edges), ints, deltas, mask, invalid, ef_n):
            return
    a, b, c, d, e = nbr_block_numpy(finder, nodes, ts3, k, host_edges)
    ints[...] = a
    deltas[...] = b
    mask[...] = c
    invalid[...] = d
    if e is not None:
        ef_n[...] = e


def nbr_block(finder, nodes, ts3, k, host_edges):
    """nbr_block_numpy's arrays, through rust/tgn_host when available."""
    if not _rust_finder_ok(finder, k):
        return nbr_block_numpy(finder, nodes, ts3, k, host_edges)
    n = len(nodes)
    ints = np.empty(n + 2 * n * k, np.int64)
    deltas = np.empty((n, k), np.float32)
    mask = np.empty((n, k), bool)
    invalid = np.empty((n, 1), bool)
    ef_n = None if host_edges is None else np.empty((n, k, host_edges.shape[1]), np.float32)
    nbr_block_into(finder, nodes, ts3, k, host_edges, ints, deltas, mask, invalid, ef_n)
    return ints, deltas, mask, invalid, ef_n
