"""Regression tests for the data/contract layer defects found on 2026-09-22.

Each test pins one defect that ran cleanly and produced a wrong number. The
measurements quoted in the assertions' messages were taken at full density on
the real corpora; see the module comments in the files under test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from cyberworld_v4.config import get_contract
from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter
from data_unification.multi_dataset_stream import (
    HostTrajectoryExtractor,
    heuristic_label_override_enabled,
)
from data_unification.time_utils import timestamp_resolution_seconds
from data_unification.trajectory_store import (
    RESIDENT_FEATS_MAX_BYTES,
    FEAT_DIM,
    _resident_or_random_advised,
)
from data_unification.unified_schema import UnifiedFlowRecord

W = get_contract().window_seconds
T0 = 1_500_000_000.0


def _rec(t, src="10.0.0.1", dst="10.0.0.2", dport=53, fwd_p=2, bwd_p=2,
         fwd_b=120, bwd_b=140, dur=0.0, attack=False, coarse="Benign"):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=40000, dst_port=dport, protocol=6,
        start_time=t, end_time=t + dur,
        fwd_bytes=fwd_b, bwd_bytes=bwd_b, fwd_packets=fwd_p, bwd_packets=bwd_p,
        raw_label="BENIGN" if not attack else "DoS Hulk",
        raw_label_source="CIC2017", is_attack=attack, coarse_category=coarse,
        attck_technique_ids=[],
    )


# ---------------------------------------------------------------------------
# 1. The window clock must not fall behind the data after a gap
# ---------------------------------------------------------------------------
#
# The advance was `current_win_start += window_size_sec` -- exactly one window
# per boundary-crossing record -- so any gap wider than one window left the
# grid permanently behind. Measured on CIC-2017 Wednesday (the held-out TEST
# split): 14,676 of 15,183 emitted windows (96.66%) held a record outside
# their own interval, the worst 116 s away.

def test_every_record_lands_in_a_window_that_contains_it():
    # 4 records at t0, then a 60 s gap, then 4 more: the old code emitted one
    # window per post-gap record with a nominal start 60 s behind the data.
    times = [T0, T0 + 0.5, T0 + 1.0, T0 + 1.5,
             T0 + 60.0, T0 + 60.5, T0 + 61.0, T0 + 120.0]
    recs = [_rec(t) for t in times]
    es = FlowToTemporalEventAdapter(window_size_sec=W).process_records(recs)

    covered = 0
    for win_start, win_end, s, e in es.window_boundaries:
        if s >= e:
            continue
        seg = es.timestamps[s:e]
        assert np.all(seg >= win_start), (
            f"record before its window start: {seg.min()} < {win_start}")
        assert np.all(seg < win_start + W), (
            f"record past its window end: {seg.max()} >= {win_start + W}")
        assert win_end == pytest.approx(win_start + W), (
            "a window must be exactly one window_size wide; the final window "
            "used max(start+W, last_timestamp) and stretched to 116 s on real data")
        covered += e - s
    assert covered == len(recs), "every record must belong to exactly one window"


def test_window_grid_stays_anchored_to_the_first_timestamp():
    times = [T0, T0 + 600.0, T0 + 600.5]
    es = FlowToTemporalEventAdapter(window_size_sec=W).process_records(
        [_rec(t) for t in times])
    # The post-gap window must be the grid cell that holds t0+600, not the
    # second cell of the grid.
    last_start = es.window_boundaries[-1][0]
    assert last_start == pytest.approx(T0 + 600.0), (
        f"window clock lagged: expected {T0 + 600.0}, got {last_start} "
        f"(lag {T0 + 600.0 - last_start:.1f} s)")


def test_empty_windows_are_not_emitted_but_indices_still_partition():
    times = [T0 + i * 0.2 for i in range(10)] + [T0 + 1000.0]
    recs = [_rec(t) for t in times]
    es = FlowToTemporalEventAdapter(window_size_sec=W).process_records(recs)
    idx = []
    for _ws, _we, s, e in es.window_boundaries:
        idx.append((s, e))
    flat = [i for s, e in idx for i in range(s, e)]
    assert flat == list(range(len(recs)))


# ---------------------------------------------------------------------------
# 2. The behavioural heuristic must not overwrite the dataset's own label
# ---------------------------------------------------------------------------
#
# Measured on CIC-2017 Monday (529,601 flows, ZERO attack rows in the corpus):
# the heuristic relabelled 28,748 of 203,760 host-windows (14.11%) as
# attack/Execution. On the held-out CIC-2017 Wednesday test split, 9,481 of
# the 9,659 resulting positives (98.2%) were the heuristic, not the data.

class _StubTGN:
    """Duck-types just enough of TGNE-TA. No `embedding_module`, so the
    neighbour-finder setup in extract_trajectories is skipped."""

    device = "cpu"

    def get_host_embeddings(self, host_ids, timestamp=None, n_neighbors=10):
        return torch.zeros((len(host_ids), 12), dtype=torch.float32)


def _benign_windows_that_trip_the_heuristic():
    """Ordinary small DNS-shaped exchanges on port 53, all labelled Benign.

    fingerprint_flow calls these C2Beaconing and marks
    is_port_mismatch = (dst_port not in {80,443,8000,8080,8443}), so two of
    them in one window satisfy `port_mismatch_count >= 2`.
    """
    recs = []
    for i in range(6):
        recs.append(_rec(T0 + 0.1 * i, src="10.0.0.1", dst=f"10.0.0.{10 + i}",
                         dport=53, fwd_p=2, bwd_p=2, fwd_b=120, bwd_b=140,
                         dur=0.9))
    return recs


def _extract(recs, monkeypatch, override):
    if override is None:
        monkeypatch.delenv("CYBERWORLD_HEURISTIC_LABEL_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("CYBERWORLD_HEURISTIC_LABEL_OVERRIDE", override)
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=W)
    return ex.extract_trajectories(recs)


def test_override_is_off_by_default():
    assert heuristic_label_override_enabled() is False or \
        os.environ.get("CYBERWORLD_HEURISTIC_LABEL_OVERRIDE"), \
        "the heuristic label override must default to OFF"


def test_benign_traffic_stays_benign_by_default(monkeypatch):
    store = _extract(_benign_windows_that_trip_the_heuristic(), monkeypatch, None)
    n = store.n_snapshots
    assert n > 0
    assert not np.asarray(store.is_attack).any(), (
        "a heuristic relabelled ground-truth-benign traffic as an attack; "
        "on CIC-2017 Monday that mislabelled 14.11% of a capture with no "
        "attacks in it")
    assert set(store.categories) == {"Benign"}
    assert float(np.asarray(store.risk_score).max()) == 0.0


def test_the_override_still_works_when_explicitly_requested(monkeypatch):
    store = _extract(_benign_windows_that_trip_the_heuristic(), monkeypatch, "1")
    assert np.asarray(store.is_attack).any(), (
        "with CYBERWORLD_HEURISTIC_LABEL_OVERRIDE=1 the heuristic must still "
        "fire -- this test pins that the switch, not the code, was removed")
    assert "Execution" in store.categories
    assert float(np.asarray(store.risk_score).max()) >= 0.65


def test_ground_truth_attack_labels_survive_either_way(monkeypatch):
    recs = [_rec(T0 + 0.1 * i, dport=53, attack=True, coarse="Impact",
                 dur=0.9) for i in range(4)]
    for flag in (None, "1"):
        store = _extract(list(recs), monkeypatch, flag)
        assert np.asarray(store.is_attack).all()


# ---------------------------------------------------------------------------
# 3. The "resident" feature block must actually be resident
# ---------------------------------------------------------------------------
#
# np.ascontiguousarray on an already C-contiguous memmap returns a VIEW, so
# the read-amplification fix (52.5 -> 8.1 batch/s, 41.8 TiB read from a
# 2.30 GiB block) was inert: the block got neither the copy nor MADV_RANDOM.

def test_small_feature_block_is_read_into_ram(tmp_path):
    n = 4096
    p = tmp_path / "blk.feats"
    data = np.asarray(np.random.rand(n, FEAT_DIM), dtype=np.float32)
    data.tofile(p)
    mm = np.memmap(p, dtype=np.float32, mode="r", shape=(n, FEAT_DIM))
    assert mm.nbytes <= RESIDENT_FEATS_MAX_BYTES

    out = _resident_or_random_advised(mm, n)
    assert not isinstance(out, np.memmap)
    assert out.base is None and out.flags["OWNDATA"], (
        "the block is still a view over the spill file; ascontiguousarray does "
        "not copy an already-contiguous memmap")
    np.testing.assert_allclose(out, data)

    # Decisive: overwriting the backing file must not change a RAM copy.
    with open(p, "r+b") as f:
        f.write(np.float32(-999.0).tobytes())
        f.flush()
        os.fsync(f.fileno())
    assert float(out[0, 0]) != -999.0, "still mapped to disk"


def test_oversized_block_stays_memmapped(tmp_path, monkeypatch):
    import data_unification.trajectory_store as ts
    monkeypatch.setattr(ts, "RESIDENT_FEATS_MAX_BYTES", 16)
    n = 64
    p = tmp_path / "big.feats"
    np.zeros((n, FEAT_DIM), dtype=np.float32).tofile(p)
    mm = np.memmap(p, dtype=np.float32, mode="r", shape=(n, FEAT_DIM))
    assert isinstance(ts._resident_or_random_advised(mm, n), np.memmap)


# ---------------------------------------------------------------------------
# 4. get_heldout_test_records must forward `stride`
# ---------------------------------------------------------------------------
#
# It dropped it while both sibling accessors forward it, so a capped test
# sample silently reverted to a chronological PREFIX -- which measured 0%
# attack on the frozen val split.

def test_heldout_test_records_forwards_stride():
    from data_unification.split_manager import ScientificSplitManager

    seen = {}

    class _Spy(ScientificSplitManager):
        def __init__(self):
            pass

        def records_for(self, split, max_per_source=None, stride=1):
            seen.update(split=split, max_per_source=max_per_source, stride=stride)
            return []

    _Spy().get_heldout_test_records(max_per_source=2000, stride=37)
    assert seen == {"split": "test", "max_per_source": 2000, "stride": 37}


# ---------------------------------------------------------------------------
# 5. chunked extraction must not raise on its own bookkeeping
# ---------------------------------------------------------------------------
#
# `_Col` defines neither __len__ nor __getitem__, so `if builder._window_idx:`
# was always True and the slice that followed raised TypeError on chunk 1.

def test_chunked_extraction_runs_and_keeps_window_idx_monotonic():
    from data_unification.chunked_extraction import extract_trajectories_chunked

    recs = [_rec(T0 + i * 0.7) for i in range(400)]
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=W)
    store = extract_trajectories_chunked(ex, iter(recs), chunk_size=100,
                                         history_seconds=4.0)
    assert store.n_snapshots > 0
    for host in store:
        rows = np.asarray(store._rows_by_host[host])
        w = np.asarray(store.window_idx)[rows]
        s = np.asarray(store.window_start)[rows]
        order = np.argsort(s, kind="stable")
        assert np.all(np.diff(w[order]) >= 0), (
            "window_idx must stay monotonic in time across chunk boundaries")


# ---------------------------------------------------------------------------
# 6. Timestamp resolution is measured, not assumed
# ---------------------------------------------------------------------------
#
# 7 of 8 CIC-2017 captures carry `d/m/Y H:M` with no seconds, so their real
# grid is 60 s under a 2 s contract. On Friday-PortScan that makes the
# smallest non-zero seconds-to-next-attack 60.0 s, and hazard_risk(tau=10)
# takes exactly two values.

def test_timestamp_resolution_detects_a_minute_grid():
    minute = np.array([T0 + 60 * i for i in range(10)], dtype=np.float64)
    assert timestamp_resolution_seconds(minute) == pytest.approx(60.0)
    second = np.array([T0 + i for i in range(10)], dtype=np.float64)
    assert timestamp_resolution_seconds(second) == pytest.approx(1.0)
    assert timestamp_resolution_seconds(np.array([T0])) == 0.0
    # invalid stamps must not be mistaken for a fine grid
    assert timestamp_resolution_seconds(
        np.array([0.0, -1.0, np.nan, T0, T0 + 60.0])) == pytest.approx(60.0)


def test_coarse_resolution_is_reported(caplog):
    import logging

    import data_unification.time_utils as tu

    tu._RESOLUTION_WARNED.clear()
    minute = np.array([T0 + 60 * i for i in range(10)], dtype=np.float64)
    with caplog.at_level(logging.WARNING):
        res = tu.warn_if_resolution_too_coarse(minute, W, source="fake_capture.csv")
    assert res == pytest.approx(60.0)
    assert any("fake_capture.csv" in r.getMessage() for r in caplog.records)
    # and only once per source
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        tu.warn_if_resolution_too_coarse(minute, W, source="fake_capture.csv")
    assert not caplog.records
