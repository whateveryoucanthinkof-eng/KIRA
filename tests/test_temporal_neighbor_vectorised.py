"""The vectorised sampler must equal the per-node Python loop exactly.

`get_temporal_neighbor` is called three times per training batch (source,
destination, negative). At batch 128 that is 384 Python-level iterations per
batch and ~35 million per epoch, each doing its own searchsorted and slicing,
all GIL-bound on one core. Measured while training: 97.9% CPU (one core of
sixteen), GPU at 0-12% drawing 3W of 80W, 13.5 minutes per epoch.

This is a pure-performance change, so the bar is equality with the original --
which is kept verbatim as `_get_temporal_neighbor_reference` and used as the
oracle here. Anything less than exact agreement is a bug, not a tradeoff.
"""

import numpy as np
import pytest

from bita.utils.utils import get_neighbor_finder


class _Data:
    def __init__(self, s, d, e, t):
        self.sources, self.destinations, self.edge_idxs, self.timestamps = s, d, e, t


def _graph(n_edges=20_000, n_nodes=900, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(1, n_nodes, n_edges)
    d = rng.integers(1, n_nodes, n_edges)
    e = np.arange(1, n_edges + 1)
    t = np.sort(rng.random(n_edges) * 1e6)
    return _Data(s, d, e, t), n_nodes


def _both(nf, nodes, times, k):
    return (nf.get_temporal_neighbor(nodes, times, k),
            nf._get_temporal_neighbor_reference(nodes, times, k))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [1, 10, 20])
def test_most_recent_sampling_is_identical(seed, k):
    data, n_nodes = _graph(seed=seed)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data.sources), 256)
    fast, ref = _both(nf, data.sources[idx], data.timestamps[idx], k)
    for a, b, name in zip(fast, ref, ("neighbors", "edge_idxs", "edge_times")):
        np.testing.assert_array_equal(a, b, err_msg=f"{name} differs")


def test_nodes_with_no_history_are_all_zero():
    """A node whose first interaction is at or after cut_time has no past."""
    data, n_nodes = _graph(seed=1)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    nodes = data.sources[:64]
    early = np.full(64, data.timestamps[0] - 1.0)     # before everything
    fast, ref = _both(nf, nodes, early, 10)
    assert not fast[0].any()
    np.testing.assert_array_equal(fast[0], ref[0])


def test_isolated_nodes_are_handled():
    data = _Data(np.array([5]), np.array([6]), np.array([1]), np.array([10.0]))
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=20)
    nodes = np.array([1, 2, 5, 6, 19])
    times = np.full(5, 100.0)
    fast, ref = _both(nf, nodes, times, 5)
    for a, b in zip(fast, ref):
        np.testing.assert_array_equal(a, b)


def test_fewer_neighbours_than_requested_are_right_aligned():
    """The reference left-pads with zeros; alignment must match exactly."""
    data, n_nodes = _graph(n_edges=200, n_nodes=40, seed=4)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    nodes = data.sources[:32]
    times = data.timestamps[:32]
    fast, ref = _both(nf, nodes, times, 20)
    for a, b in zip(fast, ref):
        np.testing.assert_array_equal(a, b)


def test_cut_time_is_strict():
    """An interaction exactly AT cut_time must be excluded, as searchsorted
    left does in the reference."""
    data, n_nodes = _graph(seed=2)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    nodes = data.sources[:128]
    exact = data.timestamps[:128]
    fast, ref = _both(nf, nodes, exact, 10)
    for a, b in zip(fast, ref):
        np.testing.assert_array_equal(a, b)


def test_dtypes_match_the_reference():
    data, n_nodes = _graph(seed=3)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    fast, ref = _both(nf, data.sources[:16], data.timestamps[:16], 10)
    for a, b in zip(fast, ref):
        assert a.dtype == b.dtype, f"{a.dtype} vs {b.dtype}"
        assert a.shape == b.shape


def test_segment_binary_search_matches_numpy_searchsorted():
    """The core primitive, checked directly against numpy per segment."""
    data, n_nodes = _graph(seed=5)
    nf = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    _nbr, _eidx, flat_ts, offsets = nf._csr
    rng = np.random.default_rng(5)
    nodes = rng.integers(1, n_nodes, 300)
    cuts = rng.random(300) * 1e6
    got = nf._segment_searchsorted(offsets[nodes], offsets[nodes + 1], cuts)
    for i, (node, c) in enumerate(zip(nodes, cuts)):
        a, b = offsets[node], offsets[node + 1]
        expected = a + np.searchsorted(flat_ts[a:b], c)
        assert got[i] == expected, f"node {node}: {got[i]} vs {expected}"


def test_uniform_sampling_keeps_the_same_distribution_and_shape():
    """Uniform draws are random, so compare structure and support, not values."""
    data, n_nodes = _graph(seed=7)
    nf = get_neighbor_finder(data, uniform=True, max_node_idx=n_nodes)
    nodes = data.sources[:64]
    times = data.timestamps[64:128]
    np.random.seed(0)
    nb, ei, et = nf.get_temporal_neighbor(nodes, times, 10)
    assert nb.shape == ei.shape == et.shape == (64, 10)
    # every sampled edge must genuinely precede its cut time
    for i, t in enumerate(times):
        real = et[i][et[i] > 0]
        assert np.all(real < t), f"row {i} sampled a future interaction"
    # and each row must come out sorted by time, as the reference re-sorts
    for i in range(64):
        assert np.all(np.diff(et[i]) >= 0), f"row {i} not time-ordered"
