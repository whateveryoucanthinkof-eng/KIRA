"""PCAP flow labels: attacking PAIRS, and UNKNOWN for intervals nobody can attribute.

Two defects, both in how a CIC-2018 attack interval (a time range from the
label CSV) becomes per-flow labels:

1. Time-only intervals. Nine of ten CIC-2018 label CSVs carry no addresses and
   config/attack_participants.json ships empty, so every flow of every host
   active during an attack interval was labelled with the attack -- ~445 hosts
   per day. The encoder run of 2026-09-25 trained on 9.0M InitialAccess edges
   and predicted InitialAccess for every validation edge. Default now: such
   flows are UNKNOWN (no claim either way); CYBERWORLD_UNSCOPED_LABELS=time_only
   restores the old stamping.

2. Participant SETS. "Either endpoint is a participant" labels every client of
   a victim server as attacking it. A CSV row asserts a conversation, so the
   interval now keeps {src, dst} pairs and labels a flow only when its pair is
   one of them.

The Rust-path column builder and the record path (pcap_bridge) must agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_unification.attack_windows import AttackInterval, DerivedWindows, derive_windows

ATT, VIC, CLIENT, OTHER = "18.218.115.60", "172.31.69.25", "172.31.64.90", "172.31.64.91"
DAY = "tue_20"
# 20 Feb 2018 10:12:00 local -> 14:12 UTC
T0 = 1519135920.0


def _csv(tmp_path: Path, with_ips: bool) -> Path:
    hdr = ["Src IP", "Dst IP", "Timestamp", "Label"] if with_ips else ["Timestamp", "Label"]
    rows = []
    for s in range(0, 60, 5):
        ts = f"20/02/2018 10:12:{s:02d}"
        rows.append(([ATT, VIC] if with_ips else []) + [ts, "DDoS attacks-LOIC-HTTP"])
        rows.append(([CLIENT, VIC] if with_ips else []) + [ts, "Benign"])
    p = tmp_path / ("ips.csv" if with_ips else "noips.csv")
    p.write_text("\n".join(",".join(r) for r in [hdr] + rows) + "\n")
    return p


def test_derive_windows_keeps_the_attacking_pairs(tmp_path):
    dw = derive_windows(_csv(tmp_path, True))
    (iv,) = dw.intervals
    assert iv.pairs == frozenset({frozenset((ATT, VIC))})
    assert iv.flow_hit(ATT, VIC) and iv.flow_hit(VIC, ATT)
    # the victim's legitimate client is NOT the attack, though VIC participates
    assert iv.flow_hit(CLIENT, VIC) is False
    assert iv.involves(CLIENT, VIC)         # the old participant-set answer


def test_csv_without_addresses_is_unscoped(tmp_path):
    (iv,) = derive_windows(_csv(tmp_path, False)).intervals
    assert not iv.scoped and iv.flow_hit(ATT, VIC) is None


def test_participant_map_roles_become_pairs(tmp_path):
    import json
    from data_unification.attack_participants import apply_participants, load_participant_map
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"days": {DAY: {"LOIC-HTTP": {"attackers": [ATT], "victims": [VIC]}}}}))
    dw = derive_windows(_csv(tmp_path, False))
    apply_participants(dw, DAY, load_participant_map(m))
    (iv,) = dw.intervals
    assert iv.scoped and iv.pairs == frozenset({frozenset((ATT, VIC))})


def test_apply_participants_is_called_by_the_day_loaders():
    """It had no caller: a filled-in map changed nothing."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("data_unification/training_sources.py", "rust/pcap_fast/pcap_fast_py.py",
                "scripts/retrain_future_models_live.py"):
        assert "apply_participants(dw, day)" in (root / rel).read_text(), rel


# --------------------------------------------------------------- the labellers
def _interval(scoped: bool) -> DerivedWindows:
    iv = AttackInterval("DDoS attacks-LOIC-HTTP", T0, T0 + 60, 12)
    if scoped:
        iv.participants = frozenset((ATT, VIC))
        iv.pairs = frozenset({frozenset((ATT, VIC))})
    return DerivedWindows(intervals=[iv], evidence={"plausible": True})


FLOWS = [(ATT, VIC), (CLIENT, VIC), (CLIENT, OTHER)]


def _rust_path_labels(dw):
    """DayColumnBuilder.chunk on one window, VIC's capture holding all three flows."""
    from data_unification.label_resolver import get_default_resolver
    from rust.pcap_fast.pcap_fast_py import FLOW_DT, HW_DT, WIN_DT, DayColumnBuilder
    strings = [ATT, VIC, CLIENT, OTHER]
    sid = {s: i for i, s in enumerate(strings)}
    w = 2.0
    bucket = int((T0 + 10) // w)
    wins = np.array([(bucket, 1)], dtype=WIN_DT)
    hws = np.zeros(1, dtype=HW_DT)
    hws["host"], hws["n_flows"] = 0, len(FLOWS)
    flows = np.zeros(len(FLOWS), dtype=FLOW_DT)
    for k, (a, b) in enumerate(FLOWS):
        flows[k]["src"], flows[k]["dst"] = sid[a], sid[b]
        flows[k]["dport"], flows[k]["proto"] = 80, 6
        flows[k]["start"] = flows[k]["end"] = bucket * w + 0.5
    b = DayColumnBuilder(dw, [VIC], w, get_default_resolver())
    _u, _i, _ts, lbl, _e = b.chunk(strings, wins, hws, flows)
    return [b.cats[x] for x in lbl]


def _record_path_labels(dw, monkeypatch):
    import data_unification.pcap_bridge as pb
    w = 2.0
    bucket = int((T0 + 10) // w)
    per_host = {VIC: ([{"src_ip": a, "dst_ip": b, "dst_port": 80, "protocol": 6,
                        "start_time": bucket * w + 0.5, "end_time": bucket * w + 0.5}
                       for a, b in FLOWS], None)}
    monkeypatch.setattr(pb, "iter_merged_day_windows", lambda *a, **k: iter([(bucket, per_host)]))
    (_s, _e, recs), = list(pb.iter_day_records("unused", dw, scenario_id=DAY, window_seconds=w))
    return [r.coarse_category for r in recs]


@pytest.mark.parametrize("path", ["rust", "record"])
def test_pairs_label_only_the_attacking_conversation(path, monkeypatch):
    got = (_rust_path_labels(_interval(True)) if path == "rust"
           else _record_path_labels(_interval(True), monkeypatch))
    # (ATT, VIC) is the attack; VIC's own client and an unrelated flow are not --
    # even though VIC owns this capture and is a participant.
    assert got == ["Impact", "Benign", "Benign"]


@pytest.mark.parametrize("path", ["rust", "record"])
def test_unscoped_interval_is_unknown_not_attack(path, monkeypatch):
    monkeypatch.delenv("CYBERWORLD_UNSCOPED_LABELS", raising=False)
    got = (_rust_path_labels(_interval(False)) if path == "rust"
           else _record_path_labels(_interval(False), monkeypatch))
    assert got == ["UNKNOWN"] * 3


@pytest.mark.parametrize("path", ["rust", "record"])
def test_time_only_policy_restores_the_old_stamping(path, monkeypatch):
    monkeypatch.setenv("CYBERWORLD_UNSCOPED_LABELS", "time_only")
    got = (_rust_path_labels(_interval(False)) if path == "rust"
           else _record_path_labels(_interval(False), monkeypatch))
    assert got == ["Impact"] * 3


def test_unknown_records_claim_nothing(monkeypatch):
    """UNKNOWN is the label resolver's third state: is_attack False, and
    label_filter.drop_unresolved removes it from every supervised set."""
    import data_unification.pcap_bridge as pb
    from data_unification.label_filter import drop_unresolved
    monkeypatch.delenv("CYBERWORLD_UNSCOPED_LABELS", raising=False)
    w = 2.0
    bucket = int((T0 + 10) // w)
    per_host = {VIC: ([{"src_ip": ATT, "dst_ip": VIC, "start_time": bucket * w,
                        "end_time": bucket * w}], None)}
    monkeypatch.setattr(pb, "iter_merged_day_windows", lambda *a, **k: iter([(bucket, per_host)]))
    (_s, _e, recs), = list(pb.iter_day_records("unused", _interval(False), scenario_id=DAY))
    assert not recs[0].is_attack and recs[0].raw_label == "DDoS attacks-LOIC-HTTP"
    kept, _cov = drop_unresolved(recs)
    assert kept == []


def test_downstream_pcap_reader_drops_unknown():
    from data_unification.capture_columns import ColumnSpec
    spec = ColumnSpec("pcap_windows", "PCAP2018", "tue_20_pcap", "/x", 2.0)
    assert spec.drops_unresolved


def test_cache_keys_change_with_the_policy(monkeypatch):
    from data_unification.capture_columns import ColumnSpec
    spec = ColumnSpec("pcap_windows", "PCAP2018", "tue_20_pcap", "/x", 2.0)
    monkeypatch.delenv("CYBERWORLD_UNSCOPED_LABELS", raising=False)
    a = spec.source_key()
    monkeypatch.setenv("CYBERWORLD_UNSCOPED_LABELS", "time_only")
    assert spec.source_key() != a
