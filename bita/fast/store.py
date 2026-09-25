"""Tensor storage for the TGN memory's autograd overlay and its pending messages.

Two facts about the reference training step make these structures possible.

1. A node's pending messages always come from ONE batch. With
   memory_update_at_start, every node in a batch's positives has its pending
   messages consumed and cleared before the batch stores its own, and a node
   only gains messages when it is itself a positive. So a node's pending list
   is exactly "its occurrences in the last batch it appeared in", written in
   one go -- which lets a node's messages live in one contiguous row range of
   a flat pool instead of a Python list of tensor views.

2. All message METADATA is known on the host. Owner, peer and timestamp come
   from the numpy batch arrays; only the 48-wide raw message rows are device
   tensors. So the BiTA grouping (by peer, time-sorted, capped, padded) is
   computed in numpy with no device synchronisation, and the device does one
   gather.

Gradients. In the reference, rows written to memory inside the current
backprop group carry autograd history (the GRU output that produced them), and
messages built from them do too; `detach_memory()` at the end of each group
cuts both. Here the device tables (memory, message pool) are always plain,
detached values, and rows written since the last detach are ALSO kept, with
their history, in an "overlay" list of chunks. A read gathers the detached
table and, for rows the host knows are live, substitutes the overlay row with
torch.where. The forward values are identical (the table holds exactly the
detached copy of the live row); the gradient reaches the same producers. The
difference is cost: the reference's in-place writes into an [n_nodes, 12]
Parameter made every write and every full-table clone an O(n_nodes) node in
the backward graph, and its Python message lists made detach/backup
O(pending messages) per group.
"""
from __future__ import annotations

import numpy as np
import torch


def upload(a: np.ndarray, device, dtype=None) -> torch.Tensor:
    """Host array -> device without a stream synchronisation.

    `torch.from_numpy(x).to(cuda)` from pageable memory synchronises the
    stream (non_blocking=False), which the reference did ~500 times per batch
    and which serialises the host and the device. Staging through the caching
    pinned allocator makes it asynchronous and safe: the pinned block is not
    reused until the copy has run.
    """
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def upload_many(device, *arrays):
    """Several host arrays -> device in ONE pinned buffer and ONE copy.

    Returns device tensors (views into one byte buffer) with each array's
    dtype and shape. Offsets are 8-byte aligned so every dtype can be viewed.
    """
    arrs = [np.ascontiguousarray(a) for a in arrays]
    offs, total = [], 0
    for a in arrs:
        offs.append(total)
        total += (a.nbytes + 7) & ~7
    buf = torch.empty(max(total, 8), dtype=torch.uint8, pin_memory=device.type == "cuda")
    host = buf.numpy()
    for a, o in zip(arrs, offs):
        host[o: o + a.nbytes] = a.reshape(-1).view(np.uint8)
    dev = buf.to(device, non_blocking=True)
    out = []
    for a, o in zip(arrs, offs):
        t = dev[o: o + a.nbytes].view(torch.from_numpy(np.empty(0, a.dtype)).dtype)
        out.append(t.view(a.shape))
    return out


class Overlay:
    """Rows written since the last detach, with their autograd history.

    `ptr_of(keys)` maps host keys to positions in cat(chunks), -1 if not live.
    Keys are node ids (memory) or pool row ids (messages).
    """

    def __init__(self):
        self.chunks = []
        self.n = 0
        self._cat = None

    def append(self, values: torch.Tensor) -> int:
        start = self.n
        self.chunks.append(values)
        self.n += values.shape[0]
        self._cat = None
        return start

    def cat(self) -> torch.Tensor:
        if self._cat is None:
            self._cat = self.chunks[0] if len(self.chunks) == 1 else torch.cat(self.chunks, 0)
        return self._cat

    def clear(self):
        self.chunks = []
        self.n = 0
        self._cat = None

    @property
    def active(self) -> bool:
        return bool(self.chunks)


#: Smallest pool (rows). Tests lower it to force compaction and growth.
MIN_ROWS = 1 << 16


def _capacity_for(n_live: int) -> int:
    """Pool rows to allocate for n_live live rows: 1.5x headroom (+32k), not
    a power-of-two jump. At full scale (~10-18M pending rows of 192 B) a 4x
    or doubling policy alone would cost several GB of GPU memory."""
    return max(MIN_ROWS, int(n_live * 1.5) + min(1 << 15, MIN_ROWS))


class PoolMeta:
    """The HOST bookkeeping of a MessagePool, numpy only.

    Which pool rows hold which node's pending messages, with their peers and
    times; where the next write goes; when the pool grows or compacts. None
    of it depends on a model value -- only on the batches written, the nodes
    cleared and the detach points -- so the batch planner (fast/planner.py)
    runs its own PoolMeta ahead of the trainer and computes the same row
    layout. MessagePool is a PoolMeta plus the device rows: the device work
    happens in the _dev_* hooks, called where the host layout changes, so
    the two run one definition of every layout decision.

    `_held` (MessagePool: overlay.active) says rows written since the last
    detach are held with autograd history, which forbids compaction."""

    def __init__(self, n_nodes: int, capacity: int = None):
        self.n_nodes = n_nodes
        self.cnt = np.zeros(n_nodes, dtype=np.int64)
        self.n_live = 0                  # == cnt.sum(), maintained incrementally
        self.start = np.zeros(n_nodes, dtype=np.int64)
        self._held = False
        self._alloc(capacity or MIN_ROWS)
        self.live_row0 = self.cur        # rows >= live_row0 are in the overlay

    # device hooks (no-ops here; MessagePool moves the device rows)
    def _dev_alloc(self):
        pass

    def _dev_grow(self, cap):
        pass

    def _dev_compact(self, cap, n_live, old_rows):
        pass

    def _held_active(self) -> bool:
        return self._held

    def _release_held(self):
        self._held = False

    def _alloc(self, capacity):
        self.capacity = int(capacity)
        self._dev_alloc()
        self.row_t = np.zeros(self.capacity, dtype=np.float64)
        self.row_peer = np.zeros(self.capacity, dtype=np.int64)
        self.cur = 1

    def ensure_nodes(self, n_nodes):
        if n_nodes > self.n_nodes:
            extra = n_nodes - self.n_nodes
            self.cnt = np.concatenate([self.cnt, np.zeros(extra, np.int64)])
            self.start = np.concatenate([self.start, np.zeros(extra, np.int64)])
            self.n_nodes = n_nodes

    # ----------------------------------------------------------- capacity
    def _grow(self, need):
        cap = self.capacity
        while cap - self.cur < need:
            cap = int(cap * 1.5) + need
        self._dev_grow(cap)
        self.row_t = np.concatenate([self.row_t, np.zeros(cap - self.capacity)])
        self.row_peer = np.concatenate([self.row_peer, np.zeros(cap - self.capacity, np.int64)])
        self.capacity = cap

    def live_rows(self):
        """(nodes with pending messages, their counts, their rows in order)."""
        nodes = np.flatnonzero(self.cnt > 0)
        cnts = self.cnt[nodes]
        n_live = int(cnts.sum())
        rows = np.repeat(self.start[nodes], cnts) + (
            np.arange(n_live) - np.repeat(np.cumsum(cnts) - cnts, cnts))
        return nodes, cnts, rows

    def compact(self):
        """Keep only live rows. Only legal when no overlay chunk is held."""
        assert not self._held_active()
        nodes, cnts, old_rows = self.live_rows()
        n_live = len(old_rows)
        cap = _capacity_for(n_live)
        self._dev_compact(cap, n_live, old_rows)
        row_t = np.zeros(cap, np.float64)
        row_peer = np.zeros(cap, np.int64)
        row_t[1: 1 + n_live] = self.row_t[old_rows]
        row_peer[1: 1 + n_live] = self.row_peer[old_rows]
        self.start[nodes] = 1 + np.cumsum(cnts) - cnts
        self.row_t, self.row_peer, self.capacity = row_t, row_peer, cap
        self.cur = 1 + n_live
        self.live_row0 = self.cur
        self.n_live = n_live

    # --------------------------------------------------------------- write
    def write_order(self, owners: np.ndarray) -> np.ndarray:
        """The row order write() will use (callers may upload it early)."""
        return np.argsort(owners, kind="stable")

    @staticmethod
    def write_pre(owners, peers, times, order):
        """What write_meta derives from the batch alone (the planner computes
        it ahead): times and peers in row order, np.unique of the owners."""
        o_sorted = owners[order]
        uniq, first, counts = np.unique(o_sorted, return_index=True, return_counts=True)
        return times[order], peers[order], uniq, first, counts

    def write_meta(self, owners, peers, times, order, pre=None) -> int:
        """Host part of a write: make room (compact or grow), place the rows,
        replace each owner's pending list. Returns the first row."""
        n = len(owners)
        if self.capacity - self.cur < n:
            if not self._held_active() and self.cur > (self.capacity >> 1):
                self.compact()
            if self.capacity - self.cur < n:
                self._grow(n)
        if pre is None:
            pre = self.write_pre(owners, peers, times, order)
        t_sorted, p_sorted, uniq, first, counts = pre
        base = self.cur
        rows = slice(base, base + n)
        self.row_t[rows] = t_sorted
        self.row_peer[rows] = p_sorted
        self.start[uniq] = base + first
        self.n_live += int(counts.sum()) - int(self.cnt[uniq].sum())
        self.cnt[uniq] = counts
        self.cur = base + n
        return base

    def write_host(self, owners, peers, times, grad: bool, order=None, pre=None) -> int:
        """A write without device rows (the planner): write_meta plus
        MessagePool.write's overlay rule (rows are held while grad is on, or
        while rows are held)."""
        if order is None:
            order = self.write_order(owners)
        base = self.write_meta(owners, peers, times, order, pre)
        if grad or self._held:
            self._held = True
        return base

    def clear(self, nodes: np.ndarray, uniq: np.ndarray = None):
        u = np.unique(nodes) if uniq is None else uniq
        self.n_live -= int(self.cnt[u].sum())
        self.cnt[u] = 0

    def detach(self):
        self._release_held()
        self.live_row0 = self.cur
        # Dead rows (consumed or superseded lists) are reclaimed here, at a
        # group boundary, where no overlay row is referenced by position. Once
        # dead rows outnumber half the live ones: O(live) work per ~live/2
        # rows written, i.e. amortised O(1) per row, and the pool stays within
        # ~1.5-2x the live rows instead of growing with every row ever written.
        dead = self.cur - 1 - self.n_live
        if dead > max(self.n_live // 2, MIN_ROWS // 2):
            self.compact()

    def state_check(self):
        """What a plan records to prove the planner's pool is the trainer's."""
        return (int(self.cur), int(self.n_live), int(self.capacity), int(self.live_row0))


class MessagePool(PoolMeta):
    """Pending raw messages: a device row pool plus host metadata (PoolMeta).

    Row 0 is reserved and stays zero: it is what padded positions gather, as
    the reference padded with zeros. Node v's pending messages are pool rows
    [start[v], start[v] + cnt[v]) in the reference's insertion order.
    """

    def __init__(self, n_nodes: int, width: int, device, capacity: int = None):
        self.width = width
        self.device = device
        self.overlay = Overlay()
        super().__init__(n_nodes, capacity)

    # ------------------------------------------------------ device hooks
    def _dev_alloc(self):
        self.pool = torch.zeros(self.capacity, self.width, device=self.device)

    def _dev_grow(self, cap):
        pool = torch.zeros(cap, self.width, device=self.device)
        pool[: self.cur] = self.pool[: self.cur]
        self.pool = pool

    def _dev_compact(self, cap, n_live, old_rows):
        pool = torch.zeros(cap, self.width, device=self.device)
        if n_live:
            pool[1: 1 + n_live] = self.pool.index_select(0, upload(old_rows, self.device, torch.int64))
        self.pool = pool

    def _held_active(self) -> bool:
        return self.overlay.active

    def _release_held(self):
        self.overlay.clear()

    # --------------------------------------------------------------- write
    def write(self, owners: np.ndarray, peers: np.ndarray, times: np.ndarray, raw: torch.Tensor,
              order: np.ndarray = None, order_g: torch.Tensor = None, pre=None):
        """Replace each owner's pending list with its rows of `raw`, in order.

        `raw` rows are in the reference's append order (all source-side
        messages, then all destination-side ones); a stable sort by owner
        reproduces each node's list order. `pre` is PoolMeta.write_pre's
        result when the batch planner computed it.
        """
        n = len(owners)
        if order is None:
            order = self.write_order(owners)
        base = self.write_meta(owners, peers, times, order, pre)
        if order_g is None:
            order_g = upload(order, self.device, torch.int64)
        rows = slice(base, base + n)
        sorted_raw = raw.index_select(0, order_g)
        with torch.no_grad():
            self.pool[rows] = sorted_raw
        if torch.is_grad_enabled() or self.overlay.active:
            assert self.overlay.n == base - self.live_row0
            self.overlay.append(sorted_raw)

    # ---------------------------------------------------------------- read
    def gather(self, rows: np.ndarray, rows_g: torch.Tensor = None) -> torch.Tensor:
        """Rows of the pool (any shape of row ids), live rows with history."""
        if rows_g is None:
            rows_g = upload(rows, self.device, torch.int64)
        base = self.pool[rows_g]
        if not self.overlay.active or not (rows >= self.live_row0).any():
            return base
        live = rows_g >= self.live_row0
        # index_select, not advanced indexing: every non-live position maps to
        # overlay row 0, and advanced indexing's backward (a sorted
        # index_put_ with accumulate) sums those duplicates serially -- up to
        # ~0.5 ms of GPU per batch. index_select's backward is index_add_;
        # the duplicates carry exact zeros (the where below masks them), and
        # no live row is gathered twice, so the gradient values are the same.
        flat = (rows_g - self.live_row0).clamp(min=0).reshape(-1)
        picked = self.overlay.cat().index_select(0, flat).view(*rows_g.shape, -1)
        return torch.where(live.unsqueeze(-1), picked, base)

    # ---------------------------------------------------------- snapshots
    def snapshot(self):
        nodes, cnts, rows = self.live_rows()
        n_live = len(rows)
        data = self.pool.index_select(0, upload(rows, self.device, torch.int64)).detach().clone() \
            if n_live else torch.zeros(0, self.width, device=self.device)
        return dict(nodes=nodes, cnts=cnts, data=data, t=self.row_t[rows].copy(),
                    peer=self.row_peer[rows].copy(), n_nodes=self.n_nodes)

    def restore(self, snap):
        self.overlay.clear()
        self.cnt[:] = 0
        n_live = len(snap["t"])
        cap = _capacity_for(n_live)
        self._alloc(cap)
        if n_live:
            self.pool[1: 1 + n_live] = snap["data"]
        self.row_t[1: 1 + n_live] = snap["t"]
        self.row_peer[1: 1 + n_live] = snap["peer"]
        cnts = snap["cnts"]
        self.cnt[snap["nodes"]] = cnts
        self.n_live = n_live
        self.start[snap["nodes"]] = 1 + np.cumsum(cnts) - cnts
        self.cur = 1 + n_live
        self.live_row0 = self.cur


class MemoryOverlay:
    """Node memory: the Parameter holds detached current values; rows written
    since the last detach are also held with history in an overlay."""

    def __init__(self, n_nodes: int, device):
        self.device = device
        self.ptr = np.full(n_nodes, -1, dtype=np.int64)          # host copy: "is anything live?"
        self.ptr_g = torch.full((n_nodes,), -1, dtype=torch.int64, device=device)
        self.touched = []
        self.touched_g = []
        self.overlay = Overlay()

    def ensure_nodes(self, n_nodes):
        if n_nodes > len(self.ptr):
            extra = n_nodes - len(self.ptr)
            self.ptr = np.concatenate([self.ptr, np.full(extra, -1, np.int64)])
            self.ptr_g = torch.cat([self.ptr_g, torch.full((extra,), -1, dtype=torch.int64, device=self.device)])

    def write(self, table: torch.Tensor, nodes_np: np.ndarray, nodes_gpu: torch.Tensor, values: torch.Tensor):
        """`nodes_np` must be unique."""
        with torch.no_grad():
            table[nodes_gpu] = values
        if torch.is_grad_enabled() or self.overlay.active:
            start = self.overlay.append(values)
            self.ptr[nodes_np] = start + np.arange(len(nodes_np))
            self.ptr_g[nodes_gpu] = torch.arange(start, start + len(nodes_np), device=self.device)
            self.touched.append(nodes_np)
            self.touched_g.append(nodes_gpu)

    def read(self, table: torch.Tensor, nodes_np: np.ndarray, nodes_gpu: torch.Tensor) -> torch.Tensor:
        base = table[nodes_gpu]
        if not self.overlay.active or not (self.ptr[nodes_np] >= 0).any():
            return base
        p = self.ptr_g[nodes_gpu]
        picked = self.overlay.cat().index_select(0, p.clamp(min=0))
        return torch.where((p >= 0).unsqueeze(-1), picked, base)

    def detach(self):
        for t in self.touched:
            self.ptr[t] = -1
        if self.touched_g:
            # index_fill_, not `ptr_g[idx] = -1`: the latter copies the scalar
            # through a synchronising host->device transfer.
            self.ptr_g.index_fill_(0, torch.cat(self.touched_g), -1)
        self.touched = []
        self.touched_g = []
        self.overlay.clear()
