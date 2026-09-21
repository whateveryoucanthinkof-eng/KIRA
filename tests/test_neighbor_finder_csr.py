"""The CSR neighbour finder must be identical to the list-of-tuples original.

The original built `adj_list = [[] for _ in range(n_nodes)]` and appended a
Python tuple per edge per direction: measured **293 bytes per edge**, i.e.
9.3 GiB of intermediate at full corpus density (~34M edges), with the numpy
conversion happening while that list was still alive. That -- not the data
volume -- is what made full-density training look impossible and got a stride
put in front of it.

CSR stores the same thing in three flat arrays (~24 B per directed edge) and
hands out zero-copy views. These tests pin the equivalence, because a silent
difference here corrupts every embedding the encoder produces.
"""

import numpy as np
import pytest

from bita.utils.utils import NeighborFinder, get_neighbor_finder


class _Data:
    def __init__(self, s, d, e, t):
        self.sources, self.destinations, self.edge_idxs, self.timestamps = s, d, e, t


def _reference(data, max_node_idx):
    """The original implementation, kept verbatim as the oracle."""
    adj = [[] for _ in range(max_node_idx + 1)]
    for s, d, e, t in zip(data.sources, data.destinations, data.edge_idxs, data.timestamps):
        adj[s].append((d, e, t))
        adj[d].append((s, e, t))
    return NeighborFinder(adj, uniform=False)


def _random_graph(n_edges=4000, n_nodes=300, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(1, n_nodes, n_edges)
    d = rng.integers(1, n_nodes, n_edges)
    e = np.arange(1, n_edges + 1)
    t = np.sort(rng.random(n_edges) * 1e6)   # edges arrive in time order
    return _Data(s, d, e, t), n_nodes


@pytest.mark.parametrize("seed", range(5))
def test_every_node_matches_the_reference(seed):
    data, n_nodes = _random_graph(seed=seed)
    ref = _reference(data, n_nodes)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)

    for node in range(n_nodes + 1):
        rn, re_, rt = ref.node_to_neighbors[node], ref.node_to_edge_idxs[node], ref.node_to_edge_timestamps[node]
        cn, ce, ct = csr.node_to_neighbors[node], csr.node_to_edge_idxs[node], csr.node_to_edge_timestamps[node]
        assert len(cn) == len(rn), f"node {node}: {len(cn)} neighbours vs {len(rn)}"
        np.testing.assert_array_equal(np.asarray(cn), np.asarray(rn), err_msg=f"node {node} neighbours")
        np.testing.assert_array_equal(np.asarray(ce), np.asarray(re_), err_msg=f"node {node} edge idxs")
        np.testing.assert_allclose(np.asarray(ct), np.asarray(rt), err_msg=f"node {node} timestamps")


def test_each_nodes_slice_is_sorted_by_timestamp():
    """find_before() uses np.searchsorted on these, so order is an invariant.

    A stable sort on node alone would NOT give this: the two edge directions
    are concatenated, so their timestamps interleave. Hence the lexsort.
    """
    data, n_nodes = _random_graph(seed=7)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    for node in range(n_nodes + 1):
        ts = np.asarray(csr.node_to_edge_timestamps[node])
        assert np.all(np.diff(ts) >= 0), f"node {node} timestamps out of order"


@pytest.mark.parametrize("seed", range(3))
def test_find_before_matches_the_reference(seed):
    data, n_nodes = _random_graph(seed=seed)
    ref = _reference(data, n_nodes)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        node = int(rng.integers(1, n_nodes))
        cut = float(rng.random() * 1e6)
        a = ref.find_before(node, cut)
        b = csr.find_before(node, cut)
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x, dtype=np.float64),
                                       np.asarray(y, dtype=np.float64))


def test_get_temporal_neighbor_matches_the_reference():
    """The method the model actually calls every batch."""
    data, n_nodes = _random_graph(seed=11)
    ref = _reference(data, n_nodes)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    nodes = data.sources[:500]
    times = data.timestamps[:500]
    for a, b in zip(ref.get_temporal_neighbor(nodes, times, 20),
                    csr.get_temporal_neighbor(nodes, times, 20)):
        np.testing.assert_array_equal(a, b)


def test_isolated_nodes_return_empty_slices():
    data = _Data(np.array([5]), np.array([6]), np.array([1]), np.array([1.0]))
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=10)
    for node in (1, 2, 3, 4, 7, 8, 9, 10):
        assert len(csr.node_to_neighbors[node]) == 0
    assert len(csr.node_to_neighbors[5]) == 1
    assert csr.node_to_neighbors[5][0] == 6
    assert csr.node_to_neighbors[6][0] == 5


def test_self_loops_appear_twice_as_in_the_original():
    data = _Data(np.array([3]), np.array([3]), np.array([1]), np.array([1.0]))
    ref = _reference(data, 5)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=5)
    assert len(csr.node_to_neighbors[3]) == len(ref.node_to_neighbors[3]) == 2


def test_views_are_zero_copy_into_one_flat_array():
    """The point of CSR: no per-node allocation."""
    data, n_nodes = _random_graph(seed=3)
    csr = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes)
    v = csr.node_to_neighbors[10]
    assert isinstance(v, np.ndarray)
    assert v.base is not None, "slice should be a view, not a copy"


def test_the_list_of_lists_constructor_still_works():
    """branch_a_gnn_lstm and multi_dataset_stream build small graphs directly."""
    adj = [[] for _ in range(4)]
    adj[1].append((2, 1, 5.0))
    adj[2].append((1, 1, 5.0))
    nf = NeighborFinder(adj, uniform=False)
    assert list(nf.node_to_neighbors[1]) == [2]
    assert len(nf.node_to_neighbors[0]) == 0
