import os

import numpy as np
import torch


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
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

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


def get_neighbor_finder(data, uniform, max_node_idx=None):
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
  """
  src = np.asarray(data.sources)
  dst = np.asarray(data.destinations)
  eidx = np.asarray(data.edge_idxs)
  ts = np.asarray(data.timestamps, dtype=np.float64)

  max_node_idx = int(max(src.max(), dst.max())) if max_node_idx is None else int(max_node_idx)
  n_nodes = max_node_idx + 1

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
              et[rows, order].astype(np.float32))

    # Most-recent-n: take positions [stop - k, stop), right-aligned in the
    # output, which is what the reference's negative slicing produces.
    src_idx = stop[:, None] - n_neighbors + col[None, :]
    valid = (src_idx >= starts[:, None]) & (src_idx < stop[:, None])
    safe = np.where(valid, src_idx, 0)
    zero_i = np.zeros((), dtype=np.int32)
    neighbors = np.where(valid, flat_nbr[safe], zero_i).astype(np.int32)
    edge_idxs = np.where(valid, flat_eidx[safe], zero_i).astype(np.int32)
    edge_times = np.where(valid, flat_ts[safe], 0.0).astype(np.float32)
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
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
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
