import os
import shutil

import numpy as np
import torch


# ------------------------------------------------------------ out-of-core store
#
# The encoder was sized for ~34M edges. The full PCAP + CTU-13 corpus is ~133M,
# and at the measured ~288 B/edge peak (42.2M-edge probe: 12.2 GB) that is
# ~38 GB of anonymous memory on a 22 GB machine. Almost all of it is
# read-mostly per-edge columns: the loaded columns, the split's per-subset
# copies, the two neighbour finders' CSR arrays and the edge features.
#
# DiskStore puts those in files and hands back numpy arrays mapped onto them.
# Pages of a file mapping are page cache: the kernel reclaims them under
# pressure (clean ones for free, dirty ones after writeback) instead of the
# cgroup OOM-killing the run, and only the pages a batch touches need to be
# resident. Values, dtypes and shapes are exactly those of the in-RAM arrays.
#
# `store=None` everywhere means "plain RAM arrays", the previous behaviour.

CHUNK = 1 << 22        # rows per streaming pass: bounds every transient


class DiskStore:
  """A directory of file-backed numpy arrays. Never put it on tmpfs (= RAM)."""

  def __init__(self, root):
    self.root = os.path.abspath(root)
    shutil.rmtree(self.root, ignore_errors=True)   # a crashed run's files are never reused
    os.makedirs(self.root, exist_ok=True)
    self._n = 0

  def _path(self, name):
    self._n += 1
    return os.path.join(self.root, f"{self._n:04d}_{name}.bin")

  def empty(self, name, shape, dtype, path=None):
    """A writable array mapped onto a new file (plain ndarray view, not np.memmap)."""
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    path = path or self._path(name)
    if int(np.prod(shape)) == 0:
      return np.empty(shape, dtype=dtype)
    return np.asarray(np.memmap(path, dtype=dtype, mode="w+", shape=shape))

  def memmap(self, name, shape, dtype):
    """Like empty() but returns the np.memmap object itself (TGN detects it)."""
    shape = tuple(int(s) for s in shape)
    return np.memmap(self._path(name), dtype=dtype, mode="w+", shape=shape)

  def remove(self, arr):
    """Delete the file behind `arr` (the caller must drop its references)."""
    base = arr
    while base is not None and not isinstance(base, np.memmap):
      base = getattr(base, "base", None)
    fn = getattr(base, "filename", None)
    if fn and os.path.dirname(os.path.abspath(fn)) == self.root:
      try:
        os.remove(fn)          # unlinked now; the pages go when the mapping does
      except OSError:
        pass

  def cleanup(self):
    shutil.rmtree(self.root, ignore_errors=True)


def store_empty(store, name, shape, dtype):
  return np.empty(shape, dtype=dtype) if store is None else store.empty(name, shape, dtype)


def masked_copy(col, mask, store=None, name="col", count=None):
  """`col[mask]`, streamed into the store when one is given (identical values)."""
  if store is None:
    return col[mask]
  n_out = int(np.count_nonzero(mask)) if count is None else int(count)
  out = store.empty(name, (n_out,) + col.shape[1:], col.dtype)
  p = 0
  for a in range(0, len(col), CHUNK):
    part = col[a:a + CHUNK][mask[a:a + CHUNK]]
    out[p:p + len(part)] = part
    p += len(part)
  assert p == n_out
  return out


def _as_nonneg_int(arr, limit=None):
  """(min, max) when `arr` is a non-negative integer array a bitmap can cover, else None."""
  arr = np.asarray(arr)
  if arr.dtype.kind not in "iu" or arr.ndim != 1 or arr.size == 0:
    return None
  lo, hi = int(arr.min()), int(arr.max())
  if lo < 0 or (limit is not None and hi >= limit):
    return None
  return lo, hi


def mark_present(seen, arr):
  """seen[arr] = True, streamed."""
  for a in range(0, len(arr), CHUNK):
    seen[arr[a:a + CHUNK]] = True
  return seen


def sorted_unique(arr):
  """np.unique(arr) for a 1-D array, via a bitmap when the values are small
  non-negative integers: no sorted copy of the whole column. Same values, same
  dtype, same (ascending) order as np.unique."""
  arr = np.asarray(arr)
  r = _as_nonneg_int(arr, limit=max(1 << 26, 4 * arr.size))
  if r is None:
    return np.unique(arr)
  seen = mark_present(np.zeros(r[1] + 1, dtype=bool), arr)
  return np.flatnonzero(seen).astype(arr.dtype, copy=False)


def count_unique(*arrays):
  """len(set(a) | set(b) | ...) without building the set."""
  rs = [_as_nonneg_int(a, limit=1 << 34) for a in arrays]
  if any(r is None for r in rs):
    s = set()
    for a in arrays:
      s |= set(np.asarray(a).tolist())
    return len(s)
  seen = np.zeros(max(r[1] for r in rs) + 1, dtype=bool)
  for a in arrays:
    mark_present(seen, a)
  return int(np.count_nonzero(seen))


def first_seen_order(arr, max_id=None):
  """The distinct values of `arr` in order of FIRST occurrence (streamed).

  Inserting a sequence into a Python set and inserting only its first
  occurrences, in the same order, produce the same set in the same internal
  layout -- a duplicate insert is a lookup that changes nothing (no resize is
  triggered by it). So `set(first_seen_order(a).tolist())` iterates exactly as
  `set(a)` does, and anything drawn from `list(that set)` is the same draw."""
  arr = np.asarray(arr)
  if len(arr) == 0:
    return arr[:0].copy()
  hi = int(arr.max()) if max_id is None else int(max_id)
  seen = np.zeros(hi + 1, dtype=bool)
  out = []
  for a in range(0, len(arr), CHUNK):
    c = arr[a:a + CHUNK]
    u, first = np.unique(c, return_index=True)
    u = u[np.argsort(first, kind="stable")]
    u = u[~seen[u]]
    seen[u] = True
    out.append(u)
  return np.concatenate(out)


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  """Random negative destinations for link prediction.

  `node_group` (optional, node id -> group id, e.g. the capture a node belongs
  to) makes negatives come from the positive edge's OWN group when `sources`
  is passed to sample(). Drawn from every capture at once, a negative could be
  a 2011 CTU-13 host scored at a 2018 timestamp: its neighbour time deltas are
  ~2e8 s, which (a) makes the negative trivially separable, inflating AP, and
  (b) drove the time encoder's gradient to ~1e7 on every step of the dry run.
  """

  def __init__(self, src_list, dst_list, seed=None, node_group=None):
    self.seed = None
    # sorted_unique == np.unique (same values, dtype, order) without a sorted
    # copy of a whole ~100M-edge column.
    self.src_list = sorted_unique(src_list)
    self.dst_list = sorted_unique(dst_list)
    self.node_group = None
    if node_group is not None:
      self.node_group = np.asarray(node_group)
      groups = self.node_group[self.dst_list]
      self._dst_by_group = {g: self.dst_list[groups == g] for g in np.unique(groups)}

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def _rng(self):
    return np.random if self.seed is None else self.random_state

  def sample(self, size, sources=None, destinations=None):
    """`destinations` (the positive edges' own) are never returned as their
    negative. With per-capture pools, the inductive evaluation's pool is often
    a single server: every negative WAS the positive destination, the two
    scored identically, and inductive AUC/AP read exactly 0.5000 on every
    epoch of the dry run -- half of the encoder's selection score was a
    constant."""
    rng = self._rng()
    src_index = rng.randint(0, len(self.src_list), size)
    avoid = None if destinations is None else np.asarray(destinations)
    if self.node_group is None or sources is None:
      return self.src_list[src_index], self._draw(rng, self.dst_list, avoid, size)
    dst = np.empty(size, dtype=self.dst_list.dtype)
    groups = self.node_group[np.asarray(sources)]
    for g in np.unique(groups):
      pos = np.nonzero(groups == g)[0]
      pool = self._dst_by_group.get(g)
      if pool is None or len(pool) == 0:
        pool = self.dst_list          # no same-group destination: fall back
      dst[pos] = self._draw(rng, pool, None if avoid is None else avoid[pos], len(pos))
    return self.src_list[src_index], dst

  def _draw(self, rng, pool, avoid, n):
    """n uniform draws from the sorted-unique `pool`; row i excludes avoid[i].

    Uniform over pool minus {avoid[i]}: draw from n-1 slots and skip the
    excluded one. A pool that is only {avoid[i]} falls back to the whole
    destination list, and only if that too is {avoid[i]} does the collision
    stand (nothing else exists to sample).
    """
    if avoid is None:
      return pool[rng.randint(0, len(pool), n)]
    k = np.searchsorted(pool, avoid)
    inside = (k < len(pool)) & (pool[np.minimum(k, len(pool) - 1)] == avoid)
    out = np.empty(n, dtype=pool.dtype)
    free = ~inside
    if free.any():
      out[free] = pool[rng.randint(0, len(pool), int(free.sum()))]
    if inside.any() and len(pool) > 1:
      idx = rng.randint(0, len(pool) - 1, int(inside.sum()))
      idx += idx >= k[inside]
      out[inside] = pool[idx]
    elif inside.any():
      out[inside] = self._draw(rng, self.dst_list, avoid[inside], int(inside.sum()))         if pool is not self.dst_list else pool[0]
    return out

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


class _CSRRows:
  """Per-node view into one flat array, indexed by node id.

  `finder.node_to_neighbors[node]` returns a zero-copy numpy VIEW of that
  node's slice, so slicing (`[:i]`, `[-n:]`) and fancy indexing behave exactly
  as they did when each node owned its own small array.
  """

  __slots__ = ("flat", "offsets")

  def __init__(self, flat, offsets):
    self.flat = flat
    self.offsets = offsets

  def __getitem__(self, node):
    return self.flat[self.offsets[node]:self.offsets[node + 1]]

  def __len__(self):
    return len(self.offsets) - 1


def _is_sorted(ts):
  for a in range(0, len(ts), CHUNK):
    c = ts[a:a + CHUNK + 1]
    if not np.all(c[1:] >= c[:-1]):       # False on NaN too: falls back
      return False
  return True


def _streamed_csr(src, dst, eidx, ts, n_nodes, store):
  """The CSR get_neighbor_finder builds, without its ~64 B per directed entry
  of transients, written straight into `store`.

  The reference order is np.lexsort((ts2, owner)) over the two directions
  concatenated: by owner, then time, then position in the concatenation
  (lexsort is stable), i.e. (owner, ts, direction, k). Equivalently: a STABLE
  sort by owner of the sequence ordered by (ts, direction, k). When the edges
  are already time-sorted (every split of the loader's output is), that
  sequence is produced chunk by chunk -- chunks cut only between distinct
  timestamps, so no tie spans two chunks -- and a stable counting sort by owner
  scatters each chunk into place: two passes, O(chunk) memory.
  """
  n = len(src)
  counts = np.zeros(n_nodes, dtype=np.int64)
  for a in range(0, n, CHUNK):
    counts += np.bincount(src[a:a + CHUNK], minlength=n_nodes)
    counts += np.bincount(dst[a:a + CHUNK], minlength=n_nodes)
  offsets = np.zeros(n_nodes + 1, dtype=np.int64)
  np.cumsum(counts, out=offsets[1:])
  del counts
  total = int(offsets[-1])
  flat_nbr = store.empty("csr_nbr", total, np.int32)
  flat_eidx = store.empty("csr_eidx", total, np.int32)
  flat_ts = store.empty("csr_ts", total, np.float64)
  cursor = offsets[:-1].copy()

  a = 0
  while a < n:
    b = min(n, a + CHUNK)
    if b < n:                              # never split a run of equal timestamps
      b = int(np.searchsorted(ts, ts[b - 1], side="right"))
    m = b - a
    s_c = np.asarray(src[a:b]).astype(np.int64, copy=False)
    d_c = np.asarray(dst[a:b]).astype(np.int64, copy=False)
    t_c = np.asarray(ts[a:b], dtype=np.float64)
    e_c = np.asarray(eidx[a:b])
    owner = np.concatenate([s_c, d_c])
    # (ts, direction, k): position in the local concatenation encodes (direction, k)
    seq = np.lexsort((np.arange(2 * m), np.concatenate([t_c, t_c])))
    owner = owner[seq]
    by_owner = np.argsort(owner, kind="stable")
    take = seq[by_owner]                   # local concat positions, final order
    o_sorted = owner[by_owner]
    del owner, seq, by_owner
    L = len(o_sorted)
    starts = np.flatnonzero(np.r_[True, o_sorted[1:] != o_sorted[:-1]])
    sizes = np.diff(np.r_[starts, L])
    rank = np.arange(L, dtype=np.int64) - np.repeat(starts, sizes)
    pos = cursor[o_sorted] + rank
    cursor[o_sorted[starts]] += sizes
    del rank, o_sorted
    first = take < m                       # source direction: peer is dst
    k = np.where(first, take, take - m)
    flat_nbr[pos] = np.where(first, d_c[k], s_c[k]).astype(np.int32, copy=False)
    flat_eidx[pos] = e_c[k].astype(np.int32, copy=False)
    flat_ts[pos] = t_c[k]
    del take, first, k, pos
    a = b
  assert np.array_equal(cursor, offsets[1:])
  return flat_nbr, flat_eidx, flat_ts, offsets


def get_neighbor_finder(data, uniform, max_node_idx=None, store=None):
  """Build a NeighborFinder without materialising a Python object per edge.

  The previous implementation built `adj_list = [[] for _ in range(n_nodes)]`
  and appended a Python tuple `(destination, edge_idx, timestamp)` for every
  edge in BOTH directions. Measured cost: **293 bytes per edge** -- at full
  corpus density (~34M edges) that is **9.3 GiB** for the intermediate alone,
  and NeighborFinder then built numpy arrays from it while the list was still
  alive, so the true peak was ~11 GiB on top of the loader. That, not the data
  volume, is what made full-density training look impossible and forced a
  stride.

  This builds the same structure in CSR form with one vectorised lexsort: no
  Python tuples, no per-node lists, ~24 bytes per directed edge.

  The lexsort key order is (timestamp, node), so within each node's slice the
  entries come out sorted by timestamp -- the invariant `find_before`'s
  `np.searchsorted` depends on. A stable sort on node alone would NOT do: the
  two directions are concatenated, so their timestamps interleave.

  With `store` (a DiskStore) and time-sorted edges, the same arrays are built
  by _streamed_csr() straight into files: identical contents, O(chunk)
  transient memory, and served from page cache.
  """
  src = np.asarray(data.sources)
  dst = np.asarray(data.destinations)
  eidx = np.asarray(data.edge_idxs)
  ts = np.asarray(data.timestamps, dtype=np.float64)

  max_node_idx = int(max(src.max(), dst.max())) if max_node_idx is None else int(max_node_idx)
  n_nodes = max_node_idx + 1

  if (store is not None and len(src) > 0 and src.dtype.kind in "iu" and dst.dtype.kind in "iu"
          and int(min(src.min(), dst.min())) >= 0
          and int(max(src.max(), dst.max())) < n_nodes and _is_sorted(ts)):
    return NeighborFinder(None, uniform=uniform,
                          _csr=_streamed_csr(src, dst, eidx, ts, n_nodes, store))

  # Each undirected edge appears once per endpoint.
  owner = np.concatenate([src, dst]).astype(np.int64, copy=False)
  peer = np.concatenate([dst, src]).astype(np.int32, copy=False)
  eidx2 = np.concatenate([eidx, eidx]).astype(np.int32, copy=False)
  ts2 = np.concatenate([ts, ts])

  order = np.lexsort((ts2, owner))          # primary: owner, secondary: time
  owner_sorted = owner[order]
  flat_nbr = peer[order]
  flat_eidx = eidx2[order]
  flat_ts = ts2[order]
  del peer, eidx2, ts2, order, owner

  # offsets[k] = first position belonging to node k
  offsets = np.searchsorted(owner_sorted, np.arange(n_nodes + 1), side="left")
  del owner_sorted

  return NeighborFinder(
    None, uniform=uniform,
    _csr=(flat_nbr, flat_eidx, flat_ts, offsets),
  )


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None, _csr=None):
    """`adj_list` is the original list-of-lists form, kept for the small graphs
    built directly in branch_a_gnn_lstm and multi_dataset_stream. Large graphs
    come in through get_neighbor_finder(), which passes `_csr` instead and
    never materialises a Python object per edge."""
    self._csr = _csr
    if _csr is not None:
      flat_nbr, flat_eidx, flat_ts, offsets = _csr
      self.node_to_neighbors = _CSRRows(flat_nbr, offsets)
      self.node_to_edge_idxs = _CSRRows(flat_eidx, offsets)
      self.node_to_edge_timestamps = _CSRRows(flat_ts, offsets)
    else:
      self.node_to_neighbors = []
      self.node_to_edge_idxs = []
      self.node_to_edge_timestamps = []

      for neighbors in adj_list:
        # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
        # We sort the list based on timestamp
        sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
        self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
        self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
        self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def _segment_searchsorted(self, starts, ends, cut_times):
    """Vectorised `np.searchsorted` inside each node's CSR slice.

    numpy has no segmented searchsorted, and the flat timestamp array is only
    sorted WITHIN a segment, never globally -- so one global searchsorted is
    wrong. This runs a branchless binary search over all rows at once:
    O(log max_degree) vectorised passes instead of one Python-level
    searchsorted per node.
    """
    lo = starts.astype(np.int64, copy=True)
    hi = ends.astype(np.int64, copy=True)
    flat_ts = self._csr[2]
    # ceil(log2) of the widest segment bounds the iteration count.
    span = int(max(1, (ends - starts).max()))
    for _ in range(int(np.ceil(np.log2(span + 1))) + 1):
      active = lo < hi
      if not active.any():
        break
      mid = (lo + hi) >> 1
      # `mid` is within [starts, ends) for active rows. Inactive rows must
      # still gather *something*, and `starts` is not safe for them: a node
      # with no edges after the last owner has starts == len(flat_ts), which
      # is out of bounds. Clamp into the array.
      safe_mid = np.where(active, mid, 0)
      np.clip(safe_mid, 0, max(0, len(flat_ts) - 1), out=safe_mid)
      go_right = active & (flat_ts[safe_mid] < cut_times)
      lo = np.where(go_right, mid + 1, lo)
      hi = np.where(active & ~go_right, mid, hi)
    return lo

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """Vectorised over the batch when CSR storage is available.

    The reference implementation loops over every node in the batch, doing its
    own searchsorted and slicing. At batch 128 that is 384 Python iterations
    per batch (source, destination, negative) and ~35M per epoch, all
    GIL-bound on one core -- which is why 15 of 16 cores and ~95% of the GPU
    sat idle while an epoch took 13.5 minutes.

    `_get_temporal_neighbor_reference` below is the original, kept verbatim as
    the oracle that tests/test_temporal_neighbor_vectorised.py checks against.
    """
    # TGNE_REFERENCE_SAMPLER=1 forces the original per-node Python loop. It
    # exists so a controlled A/B run can compare the two end to end, not just
    # in unit tests -- the vectorised path is the default.
    if (self._csr is None or n_neighbors <= 0
            or os.environ.get("TGNE_REFERENCE_SAMPLER", "") in ("1", "true", "True")):
      return self._get_temporal_neighbor_reference(source_nodes, timestamps, n_neighbors)

    flat_nbr, flat_eidx, flat_ts, offsets = self._csr
    nodes = np.asarray(source_nodes, dtype=np.int64)
    cut = np.asarray(timestamps, dtype=np.float64)
    assert len(nodes) == len(cut)

    if not self.uniform:
      # The same computation in rust/tgn_host (one binary search per row
      # instead of ~20 vectorised numpy passes): bit-identical, ~0.3 ms less
      # host time per batch. None when the library is unavailable.
      from utils.tgn_host import recent_neighbors
      out = recent_neighbors(flat_nbr, flat_eidx, flat_ts, offsets, nodes, cut, n_neighbors)
      if out is not None:
        return out

    starts = offsets[nodes]
    ends = offsets[nodes + 1]
    # Index one past the last interaction strictly before cut_time.
    stop = self._segment_searchsorted(starts, ends, cut)
    avail = stop - starts

    B = len(nodes)
    col = np.arange(n_neighbors, dtype=np.int64)

    if self.uniform:
      # Match the reference exactly: it draws n_neighbors indices WITH
      # replacement from [0, len(source_neighbors)), then re-sorts by time.
      # Rows with no history stay all-zero.
      has = avail > 0
      draw = np.zeros((B, n_neighbors), dtype=np.int64)
      if has.any():
        r = np.random.randint(0, np.maximum(avail, 1)[:, None], size=(B, n_neighbors))
        draw = starts[:, None] + r
      pick = np.where(has[:, None], draw, 0)
      nb = np.where(has[:, None], flat_nbr[pick], 0)
      ei = np.where(has[:, None], flat_eidx[pick], 0)
      et = np.where(has[:, None], flat_ts[pick], 0.0)
      order = np.argsort(et, axis=1, kind="stable")
      rows = np.arange(B)[:, None]
      return (nb[rows, order].astype(np.int32),
              ei[rows, order].astype(np.int32),
              et[rows, order].astype(np.float64))

    # Most-recent-n: take positions [stop - k, stop), right-aligned in the
    # output, which is what the reference's negative slicing produces.
    src_idx = stop[:, None] - n_neighbors + col[None, :]
    valid = (src_idx >= starts[:, None]) & (src_idx < stop[:, None])
    safe = np.where(valid, src_idx, 0)
    zero_i = np.zeros((), dtype=np.int32)
    neighbors = np.where(valid, flat_nbr[safe], zero_i).astype(np.int32)
    edge_idxs = np.where(valid, flat_eidx[safe], zero_i).astype(np.int32)
    edge_times = np.where(valid, flat_ts[safe], 0.0).astype(np.float64)
    return neighbors, edge_idxs, edge_times

  def _get_temporal_neighbor_reference(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float64)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times
