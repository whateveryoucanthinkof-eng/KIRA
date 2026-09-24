"""The encoder's graph keys nodes by (capture, host), not by address alone.

Found by scripts/dry_run_plan.py: with TGN memory on, a host present in two
captures was one node. The split is taken per capture, so training advanced
that node's memory to the later capture's time and validating the earlier
capture's edges raised "Trying to update memory to time in the past" -- on the
first validation pass of the real run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "bita", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="module")
def two_ctu_captures(tmp_path_factory):
    import random
    import dry_run_plan as d
    root = tmp_path_factory.mktemp("ctu")
    rng = random.Random(0)
    # Two lock-named train scenarios on different dates, SAME host addresses.
    d.make_ctu(root, "1/capture20110810.binetflow", (2011, 8, 10), rng)
    d.make_ctu(root, "2/capture20110811.binetflow", (2011, 8, 11), rng)
    return root


@pytest.fixture(scope="module")
def graph(two_ctu_captures):
    from train import load_and_preprocess_unified_dataset
    return load_and_preprocess_unified_dataset(ctu13_dir=str(two_ctu_captures), splits=("train",))


def test_the_same_address_in_two_captures_is_two_nodes(graph):
    graph_df, _e, node_features, _m = graph
    caps = graph_df["capture"].values
    assert len(np.unique(caps)) == 2
    nodes_by_cap = [set(graph_df.u[caps == c]) | set(graph_df.i[caps == c]) for c in np.unique(caps)]
    assert not (nodes_by_cap[0] & nodes_by_cap[1]), "a node spans two captures"


def test_features_still_come_from_the_address(graph):
    graph_df, _e, node_features, _m = graph
    caps = graph_df["capture"].values
    a = graph_df.u[caps == np.unique(caps)[0]].values
    b = graph_df.u[caps == np.unique(caps)[1]].values
    # the first source in each capture is the same address (147.32.84.100)
    assert a[0] != b[0]
    np.testing.assert_array_equal(node_features[a[0]], node_features[b[0]])


def test_every_node_is_chronological_across_the_per_capture_split(graph):
    """Train edges of a node precede its val and test edges -- what memory needs."""
    from train import split_data
    graph_df, edge_features, node_features, _m = graph
    _nf, _ef, _full, train, val, test, _nnv, _nnt = split_data(
        graph_df, edge_features, node_features, different_new_nodes=True)
    last_train = {}
    for u, v, t in zip(train.sources, train.destinations, train.timestamps):
        last_train[u] = max(last_train.get(u, -np.inf), t)
        last_train[v] = max(last_train.get(v, -np.inf), t)
    for split in (val, test):
        for u, v, t in zip(split.sources, split.destinations, split.timestamps):
            for n in (u, v):
                assert t >= last_train.get(n, -np.inf), "memory would be updated into the past"
