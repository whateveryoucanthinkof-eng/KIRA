"""Batch planner: the fast TGN step's value-independent host work, run ahead
of the trainer in a separate process (opt-in: bita/train.py --batch_planner).

What a training batch needs from the host that no model value influences
(verified against the code, fast/plan_host.py and fast/store.PoolMeta):

  * the batch's slices of the training data and its labels;
  * the negative destinations: RandEdgeSampler.sample draws from the global
    numpy RNG (the train sampler has no seed). The planner owns that stream
    for an epoch: it starts from the trainer's np.random state at the epoch
    start, and hands its end state back, which the trainer installs. Nothing
    else in the training loop draws from numpy's global RNG;
  * the message pool's row bookkeeping: which rows a batch's messages go to,
    when the pool grows or compacts, which rows hold each node's pending
    messages. It depends only on the batches written, the nodes cleared and
    the detach points (every backprop_every batches), so the planner runs its
    own PoolMeta (the same class the trainer's MessagePool extends) through
    the same sequence. Every plan records the pool state it assumed and the
    trainer checks it against its own before using the plan;
  * from that bookkeeping, the BiTA grouping (rust/tgn_host);
  * the host copy of last_update (a timestamp per node: data), hence the raw
    messages' time deltas;
  * the neighbour lookup, masks and time deltas, and host edge-feature rows
    (rust/tgn_host).

The planner process is forked from the trainer (like DataLoader workers) and
never touches torch or CUDA. Plans travel through an anonymous shared-memory
ring laid out exactly as the device buffers the unplanned step uploads
(fast_tgn uses upload_many for each group; here each group starts at a
512-byte offset, the alignment of a fresh device allocation, so every device
tensor has the alignment it had before). The trainer copies a plan into one
pinned buffer, uploads it with ONE copy, and takes views; a small header
travels through a pipe. The trainer consumes plans strictly in batch order.

Validation and test are not planned (they run the unplanned step, with its own
seeded samplers); each training epoch restarts from fresh memory, so the
planner needs nothing from them. Equivalence: tests/test_batch_planner.py,
bench.py --planner --compare.
"""
from __future__ import annotations

import copy
import math
import mmap
import multiprocessing as mp
import os
import signal
import time
import traceback
import warnings

import numpy as np

from fast.plan_host import bita_group_numpy, bita_plan_rust, nbr_block_into
from fast.store import PoolMeta

_ALIGN_GROUP = 512
_CTRL = 4096                      # control page: [0] = bytes consumed by the trainer
_BITA_GRAPH_MAX_L = 32            # == fast_tgn._BITA_GRAPH_MAX_L

VAR_NONE, VAR_EAGER, VAR_GRAPH, VAR_HOST = 0, 1, 2, 3


def _r8(n):
    return (n + 7) & ~7


_NPDT = {"i8": np.int64, "i4": np.int32, "f4": np.float32, "f8": np.float64, "b1": np.bool_}
_ITEM = {"i8": 8, "i4": 4, "f4": 4, "f8": 8, "b1": 1}


def _npdt(code):
    return _NPDT[code]


def layout(B, k, de_b, de_n, U, n_upd, E, L, variant, C, N_pad):
    """name -> (offset, dtype code, shape, count), device bytes, total bytes.

    Device groups (each as upload_many would lay it out, starting at a
    512-byte offset) then host-only arrays."""
    out = {}
    off = 0

    def group(items):
        nonlocal off
        off = (off + _ALIGN_GROUP - 1) // _ALIGN_GROUP * _ALIGN_GROUP
        for name, code, shape in items:
            count = math.prod(shape)
            out[name] = (off, code, shape, count)
            off += _r8(count * _ITEM[code])

    n = 3 * B
    g1 = [("b_ints", "i8", (6 * B,))]      # [nodes (3B) ; eidx (B) ; write order (2B)]
    if de_b:
        g1.append(("b_ef", "f4", (B, de_b)))
    group(g1)
    if variant == VAR_EAGER:
        n_groups = (E + C - 1) // C
        group([("g_ints", "i8", (E + n_upd,)), ("g_idx", "i8", (E, L)), ("g_lens", "i4", (E,)),
               ("g_f32", "f4", (E * L + n_upd,)), ("g_f64", "f8", (E + n_upd,)), ("g_pad", "b1", (n_groups * C,))])
    elif variant == VAR_GRAPH:
        E_pad = ((E + C - 1) // C) * C
        L_pad = max(8, 1 << (L - 1).bit_length())
        group([("g_ints", "i8", (E_pad + n_upd,)), ("g_idx", "i8", (E_pad, L_pad)), ("g_lens", "i4", (E_pad,)),
               ("g_f32", "f4", (E_pad * L_pad + N_pad,)), ("g_f64", "f8", (E + n_upd,))])
    group([("delta", "f4", (2 * B,))])
    g4 = [("n_ints", "i8", (n + 2 * n * k,)), ("n_deltas", "f4", (n, k)), ("n_mask", "b1", (n, k)),
          ("n_invalid", "b1", (n, 1))]
    if de_n:
        g4.append(("n_ef", "f4", (n, k, de_n)))
    group(g4)
    group([("labels", "i8", (B,))])
    dev_nbytes = off
    host = [("t_sorted", "f8", (2 * B,)), ("p_sorted", "i8", (2 * B,)), ("uniq", "i8", (U,)),
            ("first", "i8", (U,)), ("counts", "i8", (U,))]
    if variant == VAR_HOST:
        host += [("h_to_update", "i8", (n_upd,)), ("h_idx", "i8", (E, L)), ("h_klen", "i8", (E,)),
                 ("h_dt", "f4", (E, L)), ("h_last_t", "f8", (E,)), ("h_owner", "i8", (E,)),
                 ("h_counts", "f4", (n_upd,)), ("h_node_ts", "f8", (n_upd,))]
    for name, code, shape in host:
        count = math.prod(shape)
        out[name] = (off, code, shape, count)
        off += _r8(count * _ITEM[code])
    return out, dev_nbytes, off


def _views(buf, lay):
    return {name: np.frombuffer(buf, dtype=_NPDT[code], count=count, offset=o).reshape(shape)
            for name, (o, code, shape, count) in lay.items()}


class _HostViews(dict):
    """numpy views of a plan's host copy, made on first access (the trainer
    reads about half of them)."""

    __slots__ = ("buf", "lay")

    def __init__(self, buf, lay):
        super().__init__()
        self.buf, self.lay = buf, lay

    def __missing__(self, name):
        o, code, shape, count = self.lay[name]
        v = np.frombuffer(self.buf, dtype=_NPDT[code], count=count, offset=o).reshape(shape)
        self[name] = v
        return v

    def __contains__(self, name):
        return name in self.lay


# =================================================================== child
class _Ring:
    def __init__(self, mm, size):
        self.mm = mm
        self.size = size
        self.ctrl = np.frombuffer(mm, dtype=np.int64, count=8, offset=0)
        self.produced = 0

    def reserve(self, nbytes, alive):
        """(offset in the data area, monotonic end) of a free region."""
        n = (nbytes + _ALIGN_GROUP - 1) // _ALIGN_GROUP * _ALIGN_GROUP
        pos = self.produced % self.size
        if pos + n > self.size:
            self.produced += self.size - pos
            pos = 0
        end = self.produced + n
        while end - int(self.ctrl[0]) > self.size:
            if not alive():
                raise SystemExit(0)
            time.sleep(0.0002)
        self.produced = end
        return pos, end


def _parent_alive(ppid):
    return lambda: os.getppid() == ppid


class _Cfg:
    """What the child needs, captured at fork time (read-only there)."""


def _child_main(cfg, cmd_conn, out_conn, mm, ring_size):
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except Exception:
        pass
    ring = _Ring(mm, ring_size)
    alive = _parent_alive(cfg.ppid)
    data_mv = memoryview(mm)[_CTRL:]
    while True:
        try:
            cmd = cmd_conn.recv()
        except (EOFError, OSError):
            return
        if cmd[0] == "stop":
            return
        try:
            if cmd[0] == "train":
                _plan_train_epoch(cfg, cmd[1], ring, data_mv, out_conn, alive)
            else:
                raise ValueError(f"unknown planner command {cmd[0]!r}")
        except SystemExit:
            return
        except BaseException:
            try:
                out_conn.send(("error", traceback.format_exc()))
            except Exception:
                return


def _plan_train_epoch(cfg, p, ring, data_mv, out_conn, alive):
    """One training epoch's plans, in batch order (bita/train.py's loop)."""
    rs = np.random.RandomState()
    rs.set_state(p["rng"])
    sampler = copy.copy(cfg.sampler)
    sampler.seed = 0                      # makes _rng() return random_state ...
    sampler.random_state = rs             # ... which continues the trainer's global stream
    meta = PoolMeta(p["n_nodes"])
    lu = np.zeros(p["n_nodes"], np.float64)   # FastMemory._lu after __init_memory__
    data = cfg.data
    bs, bpe, num_batch, k = cfg.batch_size, cfg.backprop_every, p["num_batch"], cfg.n_degree
    num_instance = len(data.sources)
    for g in range(0, num_batch, bpe):
        for j in range(bpe):
            b = g + j
            if b >= num_batch:
                continue
            s, e = b * bs, min(num_instance, (b + 1) * bs)
            src, dst = data.sources[s:e], data.destinations[s:e]
            _, neg = sampler.sample(len(src), sources=src, destinations=dst)
            hdr = _plan_batch(cfg, p, meta, lu, b, src, dst, neg, data.timestamps[s:e], data.edge_idxs[s:e],
                              data.labels[s:e], k, ring, data_mv, alive)
            out_conn.send(hdr)
        meta.detach()                     # tgn.memory.detach_memory() after every group
    out_conn.send(("end", rs.get_state()))


def _plan_batch(cfg, p, meta, lu, b, src, dst, neg, ts, eidx, labels, k, ring, data_mv, alive):
    B = len(src)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    neg = np.asarray(neg, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.float64)
    eidx = np.asarray(eidx, dtype=np.int64)
    nodes = np.concatenate([src, dst, neg])
    ts3 = np.concatenate([ts, ts, ts])
    pos = nodes[: 2 * B]
    peers = np.concatenate([dst, src])
    ts2 = ts3[: 2 * B]
    write_order = np.argsort(pos, kind="stable")
    pre = PoolMeta.write_pre(pos, peers, ts2, write_order)
    U = len(pre[2])
    state = meta.state_check()

    # BiTA grouping over the pool as it is before this batch's write.
    h = bita_plan_rust(meta, pos, cfg.max_seq_len)
    res = None
    if h is None:
        res = bita_group_numpy(meta.cnt, meta.start, meta.row_peer, meta.row_t, pos, cfg.max_seq_len)
        n_upd, E, L = (0, 0, 0) if res is None else (len(res[0]), res[1], res[2])
    else:
        n_upd, E, L = h.n_upd, h.E, h.L
    if n_upd == 0:
        variant = VAR_NONE
    elif cfg.level < 2:
        variant = VAR_HOST
    elif p["bita_graph"] and B == p["graph_batch"] and L <= _BITA_GRAPH_MAX_L:
        variant = VAR_GRAPH
    else:
        variant = VAR_EAGER
    C, N_pad = cfg.context_size, 2 * p["graph_batch"] + 1
    lay, dev_nbytes, total = layout(B, k, cfg.de_b, cfg.de_n, U, n_upd, E, L, variant, C, N_pad)
    off, end = ring.reserve(total, alive)
    buf = data_mv[off: off + total]
    v = _views(buf, lay)

    v["b_ints"][:] = np.concatenate([nodes, eidx, write_order])
    if cfg.de_b:
        np.take(cfg.host_ef_b, eidx, axis=0, out=v["b_ef"])
    v["t_sorted"][:], v["p_sorted"][:], v["uniq"][:], v["first"][:], v["counts"][:] = pre

    if variant != VAR_NONE:
        to_update, node_ts = _fill_bita(v, variant, h, res, n_upd, E, L, C, N_pad)
        # FastMemory._fast_update_memory
        assert (lu[to_update] <= node_ts).all(), "Trying to update memory to time in the past"
        lu[to_update] = node_ts
    if h is not None:
        h.close()
    # clear_messages(pos) then the write of this batch's messages
    meta.clear(pos, uniq=pre[2])
    meta.write_host(pos, peers, ts2, grad=True, order=write_order, pre=pre)
    lu_pos = lu[pos]
    v["delta"][:] = np.where(lu_pos > 0, ts2 - lu_pos, 0.0).astype(np.float32)
    nbr_block_into(cfg.finder, nodes, ts3, k, cfg.host_ef_n, v["n_ints"], v["n_deltas"], v["n_mask"],
                   v["n_invalid"], v.get("n_ef"))
    v["labels"][:] = np.asarray(labels)
    return ("plan", b, off, end, B, U, n_upd, E, L, variant, state)


def _fill_bita(v, variant, h, res, n_upd, E, L, C, N_pad):
    """Write the grouping in the variant's layout; returns (to_update, node_ts) views."""
    if variant == VAR_HOST:
        if h is not None:
            k32 = np.empty(E, np.int32)
            h.fill(v["h_idx"], v["h_dt"], k32, v["h_owner"], v["h_last_t"], v["h_to_update"],
                   v["h_counts"], v["h_node_ts"])
            v["h_klen"][:] = k32
        else:
            to_update, _, _, idx, klen, dt, last_t, owner, counts, node_ts = res
            for name, a in (("h_to_update", to_update), ("h_idx", idx), ("h_klen", klen), ("h_dt", dt),
                            ("h_last_t", last_t), ("h_owner", owner), ("h_counts", counts),
                            ("h_node_ts", node_ts)):
                v[name][...] = a
        return v["h_to_update"], v["h_node_ts"]
    ints, idx, lens, f32, f64 = v["g_ints"], v["g_idx"], v["g_lens"], v["g_f32"], v["g_f64"]
    e_rows, l_cols = idx.shape
    owner = ints[:e_rows]
    to_update = ints[e_rows:]
    dt = f32[: e_rows * l_cols].reshape(e_rows, l_cols)
    counts = f32[e_rows * l_cols:]
    last_t, node_ts = f64[:E], f64[E:]
    fill_owner = N_pad - 1 if variant == VAR_GRAPH else 0
    if h is not None:
        h.fill(idx, dt, lens, owner, last_t, to_update, counts, node_ts, owner_fill=fill_owner)
    else:
        r_upd, _, _, r_idx, r_klen, r_dt, r_last, r_owner, r_counts, r_nts = res
        idx[...] = 0
        idx[:E, :L] = r_idx
        dt[...] = 0
        dt[:E, :L] = r_dt
        lens[...] = 0
        lens[:E] = r_klen
        owner[...] = fill_owner
        owner[:E] = r_owner
        counts[...] = 1.0
        counts[:n_upd] = r_counts
        to_update[:] = r_upd
        last_t[:] = r_last
        node_ts[:] = r_nts
    if variant == VAR_EAGER:
        pad = v["g_pad"]
        pad[:] = True
        pad[:E] = False
    return to_update, node_ts


# ================================================================= trainer
class Plan:
    """One planned batch, on the trainer side: host views (numpy, into a
    pinned staging buffer) and device views (one uploaded buffer)."""

    __slots__ = ("batch", "B", "k", "U", "n_upd", "E", "L", "variant", "state", "h", "d", "_stage", "_dev")


class _TorchDT:
    table = None

    @classmethod
    def get(cls, code):
        if cls.table is None:
            import torch
            cls.table = {"i8": torch.int64, "i4": torch.int32, "f4": torch.float32, "f8": torch.float64,
                         "b1": torch.bool}
        return cls.table[code]


class BatchPlanner:
    """Trainer-side handle of the planner process.

        planner = BatchPlanner(tgn, train_data, train_ngh_finder, train_rand_sampler,
                               batch_size=128, backprop_every=8, n_degree=10)
        for epoch ...:
            plans = planner.epoch(num_batch, edge_criterion, category_criterion)
            ... tgn.fast_batch_losses(..., plan=plans.next(batch_idx)) ...
            plans.finish()          # installs the planner's end-of-epoch RNG state
        planner.close()
    """

    def __init__(self, tgn, data, finder, sampler, batch_size, backprop_every, n_degree,
                 ring_bytes=64 << 20):
        from fast.fast_tgn import _host_edge_array
        from model.extentedtgn import legacy_category_pass
        if not hasattr(tgn, "fast_batch_losses"):
            raise ValueError("--batch_planner needs the fast step (--fast_step)")
        if legacy_category_pass():
            raise ValueError("--batch_planner does not plan the legacy two-pass category head")
        if getattr(finder, "uniform", False):
            raise ValueError("--batch_planner: uniform neighbour sampling draws from the global RNG "
                             "inside the step; not planned")
        if not getattr(tgn, "_fast_ga1", False):
            raise ValueError("--batch_planner plans the 1-layer graph-attention embedding only")
        if n_degree < 1:
            raise ValueError("--batch_planner needs n_degree >= 1")
        if getattr(sampler, "seed", None) is not None:
            raise ValueError("--batch_planner expects the train sampler to draw from the global RNG")
        self.tgn = tgn
        agg = tgn.message_aggregator
        cfg = _Cfg()
        cfg.ppid = os.getpid()
        cfg.data, cfg.finder, cfg.sampler = data, finder, sampler
        cfg.batch_size, cfg.backprop_every, cfg.n_degree = int(batch_size), int(backprop_every), int(n_degree)
        cfg.level = int(tgn._fast_level)
        cfg.max_seq_len, cfg.context_size = int(agg.max_seq_len), int(agg.context_size)
        cfg.host_ef_b = _host_edge_array(tgn.edge_raw_features)
        cfg.host_ef_n = _host_edge_array(tgn.embedding_module.edge_features)
        cfg.de_b = 0 if cfg.host_ef_b is None else int(cfg.host_ef_b.shape[1])
        cfg.de_n = 0 if cfg.host_ef_n is None else int(cfg.host_ef_n.shape[1])
        self.cfg = cfg
        self.finder = finder
        self.ring_size = int(ring_bytes)
        self.mm = mmap.mmap(-1, _CTRL + self.ring_size, flags=mmap.MAP_SHARED)
        self.ctrl = np.frombuffer(self.mm, dtype=np.int64, count=8, offset=0)
        self.data = memoryview(self.mm)[_CTRL:]
        ctx = mp.get_context("fork")
        self.cmd, cmd_child = ctx.Pipe()
        self.out_child, self.out = None, None
        out_parent, out_child = ctx.Pipe(duplex=False)
        self.out = out_parent
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)   # fork in a threaded process: the child is numpy-only
            self.proc = ctx.Process(target=_child_main, args=(cfg, cmd_child, out_child, self.mm, self.ring_size),
                                    daemon=True, name="batch-planner")
            self.proc.start()
        cmd_child.close()
        out_child.close()
        self.wait_s = 0.0
        self.n_plans = 0

    # -------------------------------------------------------------- epochs
    def epoch(self, num_batch, edge_criterion=None, category_criterion=None):
        tgn = self.tgn
        if tgn.embedding_module.neighbor_finder is not self.finder:
            raise RuntimeError("batch planner: the model's neighbour finder is not the planned one")
        bita_graph, graph_batch = False, getattr(tgn, "_graph_batch", None) or self.cfg.batch_size
        if tgn._fast_level >= 4 and edge_criterion is not None:
            probe = type("_B", (), {"B": graph_batch})()
            bita_graph = tgn._graph_region_ok(probe, edge_criterion, category_criterion) is not None
            graph_batch = getattr(tgn, "_graph_batch", None) or graph_batch
        self._graph_batch = int(graph_batch)
        self.cmd.send(("train", dict(rng=np.random.get_state(), n_nodes=int(tgn.memory._pool.n_nodes),
                                     num_batch=int(num_batch), bita_graph=bool(bita_graph),
                                     graph_batch=int(graph_batch))))
        return _EpochPlans(self, num_batch)

    def _recv(self):
        t0 = time.perf_counter()
        msg = self.out.recv()
        self.wait_s += time.perf_counter() - t0
        if msg[0] == "error":
            raise RuntimeError("batch planner process failed:\n" + msg[1])
        return msg

    def _take(self, hdr):
        import torch
        _, b, off, end, B, U, n_upd, E, L, variant, state = hdr
        cfg = self.cfg
        lay, dev_nbytes, total = layout(B, cfg.n_degree, cfg.de_b, cfg.de_n, U, n_upd, E, L, variant,
                                        cfg.context_size, 2 * self._graph_batch + 1)
        dev = self.tgn.device
        cuda = dev.type == "cuda"
        stage = torch.empty(total, dtype=torch.uint8, pin_memory=cuda)
        host = stage.numpy()
        host[:] = np.frombuffer(self.data, dtype=np.uint8, count=total, offset=off)
        self.ctrl[0] = end                                   # the ring region is free again
        dbuf = stage[:dev_nbytes].to(dev, non_blocking=True) if cuda else stage[:dev_nbytes]
        p = Plan()
        p.batch, p.B, p.k, p.U, p.n_upd, p.E, p.L, p.variant, p.state = \
            b, B, cfg.n_degree, U, n_upd, E, L, variant, state
        p.h = _HostViews(host, lay)
        d = {}
        for name, (o, code, shape, count) in lay.items():
            if o >= dev_nbytes:
                continue
            d[name] = dbuf[o: o + count * _ITEM[code]].view(_TorchDT.get(code)).view(shape)
        p.d = d
        p._stage, p._dev = stage, dbuf
        self.n_plans += 1
        return p

    def cpu_seconds(self) -> float:
        """CPU time the planner process has used (user + system)."""
        try:
            with open(f"/proc/{self.proc.pid}/stat") as fh:
                f = fh.read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, AttributeError):
            return float("nan")

    def close(self):
        if getattr(self, "proc", None) is None:
            return
        try:
            self.cmd.send(("stop",))
        except Exception:
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        self.proc = None

    __del__ = close


class _EpochPlans:
    def __init__(self, planner, num_batch):
        self.p = planner
        self.num_batch = num_batch
        self.next_batch = 0

    def next(self, batch_idx):
        if batch_idx != self.next_batch:
            raise RuntimeError(f"batch planner: batch {batch_idx} requested, {self.next_batch} is next "
                               "(plans are consumed strictly in batch order)")
        msg = self.p._recv()
        if msg[0] != "plan" or msg[1] != batch_idx:
            raise RuntimeError(f"batch planner: expected the plan of batch {batch_idx}, got {msg[:2]}")
        self.next_batch += 1
        return self.p._take(msg)

    def finish(self):
        if self.next_batch != self.num_batch:
            raise RuntimeError(f"batch planner: epoch ended after {self.next_batch} of {self.num_batch} batches")
        msg = self.p._recv()
        if msg[0] != "end":
            raise RuntimeError(f"batch planner: expected the end of the epoch, got {msg[0]}")
        np.random.set_state(msg[1])
