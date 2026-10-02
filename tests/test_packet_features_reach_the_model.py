"""The 30 packet-level features must be able to reach a model, and attack
labels must be attributable to hosts rather than only to instants.

Two defects on the PCAP path, both on the same line:

    for host_ip, (flows, _packet_features) in per_host.items():

  * `_packet_features` -- `telemetry/packet/pcap_engine.py` computes 30
    packet-level features per host-window (TTL variance, TCP window size,
    inter-arrival coefficient of variation, SYN-without-response ratio,
    vertical/horizontal scan scores, retransmission ratio, DNS domain
    entropy). `pcap_adapter` and `pcap_bridge` both paid for the computation
    and then discarded it, so no model has ever seen one -- while the flow
    attributes that did reach the model cannot express periodicity or
    scan shape at all.

  * `is_attack` came from `label_at(mid_ts)` and was stamped on every flow of
    every host in the window. A CIC-2018 day directory holds ~445 per-host
    captures, so an attack hour labelled ~445 hosts when one or two were
    involved, making the target nearly a function of the clock.
"""

import numpy as np
import pytest
import torch

from data_unification.attack_participants import ParticipantMap, apply_participants
from data_unification.attack_windows import AttackInterval, DerivedWindows
from data_unification.host_attributes import (
    EXTENDED_HOST_ATTRIBUTES,
    EXTENDED_HOST_ATTR_DIM,
    HOST_ATTRIBUTES,
    HOST_ATTR_DIM,
    PACKET_ATTRIBUTES,
    PACKET_ATTRIBUTE_INDEX,
    PACKET_ATTR_DIM,
    normalize_packet_features,
)
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.unified_schema import LabelSource, UnifiedFlowRecord


class _StubTGN:
    device = "cpu"

    def get_host_embeddings(self, node_ids, timestamp, n_neighbors=10):
        return torch.zeros((len(node_ids), 12), dtype=torch.float32)


def _rec(src="10.0.0.5", dst="10.0.0.6", *, t=0.0, metadata=None):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=1234, dst_port=443, protocol=6,
        start_time=t, end_time=t + 0.1, fwd_bytes=500, bwd_bytes=800,
        fwd_packets=4, bwd_packets=5, raw_label="BENIGN",
        raw_label_source=LabelSource.CIC2018.value, is_attack=False,
        coarse_category="Benign", attck_technique_ids=[], metadata=metadata,
    )


# ---------------------------------------------------------------------------
# The names and the normalisation
# ---------------------------------------------------------------------------


def test_the_names_come_from_the_engine_so_they_cannot_drift():
    from telemetry.packet.pcap_engine import PCAP_BEHAVIORAL_COLUMNS

    assert PACKET_ATTRIBUTES == list(PCAP_BEHAVIORAL_COLUMNS)
    assert PACKET_ATTR_DIM == 30
    assert EXTENDED_HOST_ATTR_DIM == HOST_ATTR_DIM + PACKET_ATTR_DIM == 45
    assert EXTENDED_HOST_ATTRIBUTES[:HOST_ATTR_DIM] == list(HOST_ATTRIBUTES)


def test_an_empty_window_normalises_to_zeros_not_nan():
    from telemetry.packet.pcap_engine import LivePCAPEngine

    v = normalize_packet_features(LivePCAPEngine().extract_features([], 2.0))
    assert v.shape == (PACKET_ATTR_DIM,)
    assert np.isfinite(v).all() and (v == 0).all()


def test_every_packet_feature_lands_in_the_unit_interval():
    """Extreme and hostile values, including the ones a scan actually produces."""
    extreme = {n: 1e9 for n in PACKET_ATTRIBUTES}
    v = normalize_packet_features(extreme)
    assert v.min() >= 0.0 and v.max() <= 1.0

    nasty = {n: float("nan") for n in PACKET_ATTRIBUTES}
    nasty["ttl_mean"] = float("inf")
    nasty["pkt_iat_cv"] = -3.0
    v = normalize_packet_features(nasty)
    assert np.isfinite(v).all() and v.min() >= 0.0 and v.max() <= 1.0


def test_ratios_pass_through_unchanged_so_they_stay_interpretable():
    v = normalize_packet_features({"syn_no_response_ratio": 0.9,
                                   "tcp_handshake_completion_ratio": 0.25})
    assert v[PACKET_ATTRIBUTE_INDEX["syn_no_response_ratio"]] == pytest.approx(0.9)
    assert v[PACKET_ATTRIBUTE_INDEX["tcp_handshake_completion_ratio"]] == pytest.approx(0.25)


def test_a_missing_key_is_zero_not_a_crash():
    v = normalize_packet_features({"ttl_mean": 64.0})
    assert v[PACKET_ATTRIBUTE_INDEX["ttl_mean"]] == pytest.approx(64.0 / 255.0)
    assert v[PACKET_ATTRIBUTE_INDEX["dns_query_count"]] == 0.0


# ---------------------------------------------------------------------------
# They reach the extractor
# ---------------------------------------------------------------------------


def test_the_default_contract_is_unchanged():
    """15 attributes and a 27-D model input, for every existing checkpoint."""
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0)
    assert ex.n_temporal_attrs == 15
    assert ex.compute_host_temporal_attributes("10.0.0.5", [_rec()], 2.0).shape == (15,)


def test_opting_in_widens_the_vector_to_45():
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0,
                                 include_packet_features=True)
    assert ex.n_temporal_attrs == EXTENDED_HOST_ATTR_DIM
    assert ex.compute_host_temporal_attributes("10.0.0.5", [_rec()], 2.0).shape == (45,)


def test_the_flow_attributes_are_identical_either_way():
    """Opting in must ADD information, never perturb what was already there."""
    recs = [_rec(t=0.0), _rec(dst="10.0.0.7", t=0.3)]
    plain = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0)
    wide = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0,
                                   include_packet_features=True)
    a = plain.compute_host_temporal_attributes("10.0.0.5", recs, 2.0)
    b = wide.compute_host_temporal_attributes("10.0.0.5", recs, 2.0)
    assert np.allclose(a, b[:HOST_ATTR_DIM])


def test_packet_features_travel_on_the_record_metadata():
    pf = {"ttl_mean": 64.0, "syn_no_response_ratio": 0.8, "pkt_iat_cv": 0.04}
    shared = {"source": "pcap", "packet_features": pf}
    recs = [_rec(t=0.0, metadata=shared), _rec(t=0.4, metadata=shared)]
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0,
                                 include_packet_features=True)
    attrs = ex.compute_host_temporal_attributes("10.0.0.5", recs, 2.0)
    assert attrs[HOST_ATTR_DIM + PACKET_ATTRIBUTE_INDEX["syn_no_response_ratio"]] == pytest.approx(0.8)
    assert attrs[HOST_ATTR_DIM + PACKET_ATTRIBUTE_INDEX["ttl_mean"]] == pytest.approx(64 / 255)


def test_the_wide_vector_survives_a_full_extraction():
    pf = {"vertical_scan_score": 65535.0, "syn_no_response_ratio": 0.95}
    shared = {"source": "pcap", "packet_features": pf}
    recs = [_rec(t=0.0, metadata=shared), _rec(t=0.5, metadata=shared)]
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0,
                                 include_packet_features=True)
    snaps = list(ex.extract_trajectories(recs)["10.0.0.5"])
    assert snaps
    assert snaps[0].temporal_attrs.shape == (EXTENDED_HOST_ATTR_DIM,)
    i = HOST_ATTR_DIM + PACKET_ATTRIBUTE_INDEX["vertical_scan_score"]
    assert snaps[0].temporal_attrs[i] == pytest.approx(1.0)


def test_csv_records_without_packet_features_are_zero_padded_not_broken():
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=2.0,
                                 include_packet_features=True)
    attrs = ex.compute_host_temporal_attributes("10.0.0.5", [_rec()], 2.0)
    assert np.all(attrs[HOST_ATTR_DIM:] == 0.0)
    assert attrs[:HOST_ATTR_DIM].any(), "flow attributes should still be populated"


def test_a_conflicting_width_is_refused_rather_than_silently_truncated():
    with pytest.raises(ValueError, match="include_packet_features"):
        HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), n_temporal_attrs=20,
                                include_packet_features=True)


def test_the_bridge_no_longer_throws_them_away():
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "data_unification" / "pcap_bridge.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "(flows, _packet_features)" not in src, (
        "pcap_bridge still unpacks the engine's features into a throwaway name"
    )
    assert '"packet_features"' in src


# ---------------------------------------------------------------------------
# Labels are scoped to participants
# ---------------------------------------------------------------------------


ATTACKER, VICTIM, BYSTANDER = "18.221.219.4", "172.31.69.25", "172.31.69.90"


def test_an_unscoped_interval_still_labels_everyone():
    """No participant set cannot mean 'nobody' -- that would delete the labels."""
    iv = AttackInterval("DDoS", 0.0, 10.0, 5)
    assert not iv.scoped
    assert iv.involves(BYSTANDER)


def test_a_scoped_interval_excludes_a_bystander():
    iv = AttackInterval("DDoS", 0.0, 10.0, 5, frozenset({ATTACKER, VICTIM}))
    assert iv.scoped
    assert iv.involves(ATTACKER) and iv.involves(VICTIM)
    assert not iv.involves(BYSTANDER)
    assert iv.involves(BYSTANDER, VICTIM), "either endpoint may match"


def test_derived_windows_reports_whether_it_can_attribute_labels():
    unscoped = DerivedWindows(intervals=[AttackInterval("DDoS", 0.0, 10.0, 5)])
    scoped = DerivedWindows(intervals=[
        AttackInterval("DDoS", 0.0, 10.0, 5, frozenset({ATTACKER}))])
    assert not unscoped.scoped
    assert scoped.scoped
    assert scoped.participant_summary()["scoped_intervals"] == 1


def test_a_participant_map_fills_a_day_the_csv_could_not_answer_for():
    d = DerivedWindows(intervals=[AttackInterval("DoS attacks-Hulk", 0.0, 10.0, 5)])
    m = ParticipantMap({"wed_14": {"Hulk": {ATTACKER, VICTIM}}}, provenance="test")
    rep = apply_participants(d, "wed_14_pcap", m)
    assert rep["scoped_from_map"] == 1
    assert d.intervals[0].involves(VICTIM)
    assert not d.intervals[0].involves(BYSTANDER)
    assert d.scoped


def test_the_map_never_overrides_what_the_data_already_said():
    """A transcription of a published schedule loses to observed addresses."""
    d = DerivedWindows(intervals=[
        AttackInterval("Hulk", 0.0, 10.0, 5, frozenset({ATTACKER}))])
    m = ParticipantMap({"wed_14": {"Hulk": {"1.2.3.4"}}}, provenance="test")
    rep = apply_participants(d, "wed_14", m)
    assert rep["scoped_from_map"] == 0
    assert d.intervals[0].participants == frozenset({ATTACKER})


def test_an_unmatched_label_is_reported_as_time_only_not_silently_dropped():
    d = DerivedWindows(intervals=[AttackInterval("Infiltration", 0.0, 10.0, 5)])
    rep = apply_participants(d, "thu_1", ParticipantMap({}, provenance="empty"))
    assert rep["time_only_labels"] == ["Infiltration"]
    assert rep["fully_scoped"] is False
    # and the label still applies to everyone, i.e. old behaviour, made visible
    assert d.intervals[0].involves(BYSTANDER)


def test_a_missing_map_file_is_normal_not_an_error():
    from data_unification.attack_participants import load_participant_map

    m = load_participant_map(pytest.importorskip("pathlib").Path("does_not_exist.json"))
    assert not m
    assert m.lookup("wed_14", "Hulk") == set()


def test_explanations_name_the_packet_attributes():
    """A --packet-features Branch A is 57-D; its attributions must name the 30
    packet-level attributes (SYN/scan signatures, TTL, timing...) in the
    extractor's order, each with a console group."""
    from data_unification.host_attributes import EXTENDED_HOST_ATTRIBUTES, PACKET_ATTRIBUTES
    from explainability.unified_explanation import FEATURE_NAMES, feature_names_for
    assert feature_names_for(27) == FEATURE_NAMES
    names = feature_names_for(57)
    assert names[12:] == list(EXTENDED_HOST_ATTRIBUTES)
    src = open("control_backend/model_adapter.py").read()
    ns = {}
    exec(compile(src.split("class AntigravityModelAdapter")[0], "ma", "exec"), ns)
    groups = {ns["FEATURE_GROUP_MAP"][n] for n in PACKET_ATTRIBUTES}
    assert "Packet: TCP flags" in groups and "Packet: Scan signature" in groups


def test_scan_scores_separate_a_scan_from_a_single_connection():
    """They were max/unique ratios: 1.0 for a host on one port of one server and
    for a 10,000-port sweep alike. Now counts, log-scaled."""
    from data_unification.host_attributes import normalize_packet_features, PACKET_ATTRIBUTE_INDEX
    i = PACKET_ATTRIBUTE_INDEX["vertical_scan_score"]
    client = normalize_packet_features({"vertical_scan_score": 1.0})[i]
    scan = normalize_packet_features({"vertical_scan_score": 10000.0})[i]
    assert scan > 0.8 and client < 0.1
