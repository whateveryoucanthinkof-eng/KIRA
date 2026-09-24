"""The graph latent's neighbour cut-off must be measured, and pinned to training.

Each host's 12-D TGNE latent is built from at most `n_neighbors` of its most
recent flows in the 2 s window (bita/train.py --n_degree, default 10). Earlier
flows are not down-weighted -- they are absent. So `n_neighbors` benign flows
sent after an attack flow remove the attack from the latent altogether; only
the 15 aggregate attributes still count it.

That is a property of the architecture, not a bug with a local fix: the
attention weights were fitted to this cut-off, and changing it at serve time
would be a train/serve mismatch. What was wrong is that nothing measured it
and the serving value was a literal that matched training by coincidence.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from data_unification.multi_dataset_stream import (
    BoundedHostSlidingBuffer,
    HostTrajectoryExtractor,
    format_neighbor_exposure,
)
from data_unification.unified_schema import UnifiedFlowRecord

REPO = Path(__file__).resolve().parents[1]
TARGET, ATTACKER = "10.0.2.10", "203.0.113.7"


def _encoder(n_neighbors=None, uniform=None):
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder

    torch.manual_seed(0)
    n0 = 64
    tgn = ExtendedTGN(
        neighbor_finder=NeighborFinder([[] for _ in range(n0)], uniform=False),
        node_features=np.zeros((n0, 12), np.float32), edge_features=np.zeros((10, 12), np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=False,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5).eval()
    if n_neighbors is not None:
        tgn.serving_n_neighbors = n_neighbors
    if uniform is not None:
        tgn.serving_neighbor_uniform = uniform
    return tgn


def _flow(src, dst, t, dport, attack=False, fwd_bytes=600):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=40000, dst_port=dport, protocol=6,
        start_time=t, end_time=t + 0.01, fwd_bytes=fwd_bytes, bwd_bytes=900,
        fwd_packets=5, bwd_packets=5, raw_label="ATTACK" if attack else "BENIGN",
        raw_label_source="TEST", is_attack=attack,
        coarse_category="InitialAccess" if attack else "Benign",
        attck_technique_ids=["T1190"] if attack else [], metadata={})


def _window(n_benign_after, with_attack=True, t0=1000.0):
    recs = [_flow(TARGET, "10.0.3.99", t0, 443)]  # anchors the window grid
    if with_attack:
        recs.append(_flow(ATTACKER, TARGET, t0 + 0.10, 4444, attack=True, fwd_bytes=250_000))
    for i in range(n_benign_after):
        recs.append(_flow(f"10.0.3.{i + 1}", TARGET, t0 + 0.20 + 0.01 * i, 80))
    return recs


def _target_snapshot(extractor, records):
    buf = BoundedHostSlidingBuffer()
    extractor.extract_trajectories(records, sliding_buffer=buf)
    (snap,) = buf.get_host_trajectory(TARGET)
    return snap


@pytest.mark.parametrize("n_after", [3, 9])
def test_the_latent_sees_an_attack_followed_by_fewer_than_n_flows(n_after):
    ex = HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=10), window_size_sec=2.0)
    hit = _target_snapshot(ex, _window(n_after, with_attack=True))
    clean = _target_snapshot(ex, _window(n_after, with_attack=False))
    assert not np.allclose(hit.embedding, clean.embedding), "the attack flow should reach the latent"


@pytest.mark.parametrize("n_after", [10, 25])
def test_n_benign_flows_after_the_attack_remove_it_from_the_latent(n_after):
    """The characterisation this file exists for: eviction, not dilution."""
    ex = HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=10), window_size_sec=2.0)
    hit = _target_snapshot(ex, _window(n_after, with_attack=True))
    clean = _target_snapshot(ex, _window(n_after, with_attack=False))
    np.testing.assert_allclose(hit.embedding, clean.embedding, atol=1e-6,
                               err_msg="latent should be blind to an evicted attack flow")
    # The attributes still count every flow, so the attack is not invisible
    # to the model as a whole -- only to its graph half.
    assert not np.allclose(hit.temporal_attrs, clean.temporal_attrs)
    assert hit.is_attack and not clean.is_attack


def test_the_exposure_report_counts_truncation_and_hidden_attacks():
    ex = HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=10), window_size_sec=2.0)
    ex.extract_trajectories(_window(12), sliding_buffer=BoundedHostSlidingBuffer())
    r = ex.neighbor_exposure_report(reset=True)
    # hosts: target (14 flows), attacker (1), anchor peer (1), 12 benign peers (1 each)
    assert r["host_windows"] == 15 and r["flows"] == 28
    assert r["truncated_host_windows"] == 1 and r["flows_outside_latent"] == 4
    assert r["attack_host_windows"] == 2  # target and attacker, attack_role="either"
    assert r["attack_host_windows_all_attack_flows_outside_latent"] == 1
    assert r["attack_hidden_from_latent_rate"] == 0.5
    assert "EVERY attack flow outside it" in format_neighbor_exposure(r, "train")

    assert ex.neighbor_exposure_report()["host_windows"] == 0, "reset=True must zero the counters"

    ex.extract_trajectories(_window(5), sliding_buffer=BoundedHostSlidingBuffer())
    r = ex.neighbor_exposure_report()
    assert r["truncated_host_windows"] == 0
    assert r["attack_host_windows_all_attack_flows_outside_latent"] == 0


def test_uniform_sampling_is_not_counted_as_a_fixed_cutoff():
    ex = HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=10, uniform=True),
                                 window_size_sec=2.0)
    assert ex.neighbor_uniform
    ex.extract_trajectories(_window(12), sliding_buffer=BoundedHostSlidingBuffer())
    r = ex.neighbor_exposure_report()
    assert r["counted"] is False and r["host_windows"] == 0
    assert "not counted" in format_neighbor_exposure(r, "train")


def test_serving_takes_the_cutoff_from_the_encoder_not_a_literal():
    assert HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=20)).n_neighbors == 20
    assert HostTrajectoryExtractor(tgne_ta_model=_encoder()).n_neighbors == 10
    assert HostTrajectoryExtractor(tgne_ta_model=_encoder(n_neighbors=20), n_neighbors=5).n_neighbors == 5
    src = (REPO / "data_unification" / "multi_dataset_stream.py").read_text(encoding="utf-8")
    assert "n_neighbors=10\n" not in src and "uniform=False))" not in src


def test_the_encoder_config_carries_the_sampling_it_was_trained_with():
    from branch_a_gnn_lstm.train_branch_a import _attach_neighbor_sampling

    tgn = _encoder()
    _attach_neighbor_sampling(tgn, {"n_neighbors": 20, "neighbor_sampling": "uniform"})
    assert tgn.serving_n_neighbors == 20 and tgn.serving_neighbor_uniform is True
    _attach_neighbor_sampling(tgn, {})  # a config written before the keys existed
    assert tgn.serving_n_neighbors == 10 and tgn.serving_neighbor_uniform is False
    with pytest.raises(ValueError):
        _attach_neighbor_sampling(tgn, {"neighbor_sampling": "recent"})
    with pytest.raises(ValueError):
        _attach_neighbor_sampling(tgn, {"n_neighbors": 0})

    train_src = (REPO / "bita" / "train.py").read_text(encoding="utf-8")
    assert '"n_neighbors": args.n_degree' in train_src
    assert '"neighbor_sampling": "uniform" if args.uniform else "most_recent"' in train_src
    from scripts.write_encoder_config import DEFAULTS
    assert DEFAULTS["n_neighbors"] == 10 and DEFAULTS["neighbor_sampling"] == "most_recent"
