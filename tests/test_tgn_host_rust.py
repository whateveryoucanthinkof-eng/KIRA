"""rust/tgn_host against the numpy code it replaces: bit-identical.

NeighborFinder.get_temporal_neighbor (most-recent-n) through the Rust library
vs the vectorised numpy path (CYBERWORLD_HOST_RUST=0), on a graph with hubs,
isolated nodes, timestamp ties exactly at the cut, cuts before the first and
after the last interaction, and the node with the largest id.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bita"))

from utils import tgn_host  # noqa: E402
from utils.utils import get_neighbor_finder  # noqa: E402

pytestmark = pytest.mark.skipif(tgn_host.lib() is None, reason="rust/tgn_host not built")


class _D:
    def __init__(self, s, d, t, e):
        self.sources, self.destinations, self.timestamps, self.edge_idxs = s, d, t, e


def _numpy(finder, nodes, ts, k, monkeypatch):
    monkeypatch.setenv("CYBERWORLD_HOST_RUST", "0")
    try:
        return finder.get_temporal_neighbor(nodes, ts, n_neighbors=k)
    finally:
        monkeypatch.delenv("CYBERWORLD_HOST_RUST")


@pytest.mark.parametrize("k", [1, 10, 20])
def test_recent_neighbors_bit_identical(k, monkeypatch):
    rng = np.random.default_rng(0)
    n_nodes, n_edges = 400, 20000
    hub = rng.random(n_edges) < 0.3
    src = np.where(hub, rng.integers(1, 4, n_edges), rng.integers(1, n_nodes - 50, n_edges))
    dst = rng.integers(1, n_nodes - 50, n_edges)
    ts = np.sort(rng.integers(0, 3000, n_edges)).astype(np.float64)          # many ties
    eidx = np.arange(1, n_edges + 1)
    finder = get_neighbor_finder(_D(src, dst, ts, eidx), uniform=False, max_node_idx=n_nodes - 1)
    q_nodes = np.concatenate([rng.integers(0, n_nodes, 3000), [0, n_nodes - 1, 1, 2, 3]])
    q_ts = np.concatenate([rng.choice(ts, 1500), rng.uniform(-10, 3100, 1500), [0.0, 5e3, -1.0, 1e9, ts[100]]])
    got = finder.get_temporal_neighbor(q_nodes, q_ts, n_neighbors=k)
    ref = _numpy(finder, q_nodes, q_ts, k, monkeypatch)
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)
    assert (got[0] != 0).any() and (got[0] == 0).any()


def test_falls_back_on_unhandled_dtypes():
    flat = np.zeros(4, np.int64)                                          # not int32
    assert tgn_host.recent_neighbors(flat, flat, np.zeros(4), np.zeros(2, np.int64),
                                     np.zeros(1, np.int64), np.zeros(1), 3) is None
