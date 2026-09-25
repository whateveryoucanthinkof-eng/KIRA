"""The encoder must receive the SAME node features at inference as in training.

`bita/train.py` trains the TGNE with 12-D intrinsic IP features. Inference goes
through `HostTrajectoryExtractor.extract_trajectories`, which rebuilds
`tgn.node_raw_features` for the hosts in the current window -- and it used to
build `torch.zeros`.

That is a train/serve mismatch of exactly the kind that is invisible in tests
of either side alone: shapes match, nothing raises, and the encoder simply
receives an input distribution it never saw. It would have silently undone the
node-feature work that took inductive AUC from 0.5043 to 0.8330, for every
downstream model and for live serving.
"""

import inspect

import numpy as np
import pytest

from data_unification.ip_features import build_node_feature_matrix, ip_node_features


def test_inference_builds_real_features_not_zeros():
    import data_unification.multi_dataset_stream as mds
    src = inspect.getsource(mds.HostTrajectoryExtractor.extract_trajectories)
    assert "build_node_feature_matrix" in src, (
        "extract_trajectories must build intrinsic IP node features"
    )
    assert "torch.zeros((n_nodes, 12)" not in src, (
        "extract_trajectories is still zeroing node features"
    )


def test_same_ip_gives_the_same_features_everywhere():
    """Training and inference must agree per address, or the model sees drift."""
    for ip in ("172.31.69.25", "192.168.10.50", "8.8.8.8", "10.0.0.1"):
        train_row = build_node_feature_matrix({ip: 1})[1]
        serve_row = build_node_feature_matrix({ip: 1}, n_nodes=5000)[1]
        np.testing.assert_array_equal(train_row, serve_row)
        np.testing.assert_array_equal(train_row, np.asarray(ip_node_features(ip), dtype=np.float32))


def test_padding_rows_stay_zero():
    """Row 0 is padding, and slack rows for unseen hosts must read as unknown."""
    m = build_node_feature_matrix({"10.0.0.1": 1, "10.0.0.2": 2}, n_nodes=64)
    assert m.shape == (64, 12)
    assert not m[0].any()
    assert m[1].any() and m[2].any()
    assert not m[3:].any()


def test_matrix_is_never_smaller_than_the_ip_map():
    """An n_nodes smaller than the map would drop hosts off the end."""
    ips = {f"10.0.0.{i}": i for i in range(1, 40)}
    m = build_node_feature_matrix(ips, n_nodes=5)
    assert m.shape[0] >= 40
    for ip, nid in ips.items():
        np.testing.assert_array_equal(m[nid], np.asarray(ip_node_features(ip), dtype=np.float32))


def test_node_ids_out_of_range_are_ignored_not_crashing():
    m = build_node_feature_matrix({"10.0.0.1": 1}, n_nodes=3)
    assert m.shape == (3, 12) and m[1].any()


def test_a_zero_feature_matrix_would_be_detectably_different():
    """Guard the guard: prove zeros and real features are distinguishable, so
    the parity assertions above are meaningful."""
    real = build_node_feature_matrix({"172.31.69.25": 1, "8.8.8.8": 2}, n_nodes=8)
    zeros = np.zeros((8, 12), dtype=np.float32)
    assert not np.array_equal(real, zeros)
