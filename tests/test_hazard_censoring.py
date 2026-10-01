"""Hazard targets that look ahead into UNKNOWN traffic are censored (NaN).

UNKNOWN traffic (an attack interval nobody can attribute, an unmapped label)
is dropped before extraction. Without censoring, a host's windows just before
such a span saw no "next attack" and got hazard 0: confident negatives on the
pre-attack windows early warning has to learn from.
"""

from __future__ import annotations

import numpy as np
import torch

from data_unification.capture_columns import unknown_intervals
from data_unification.trajectory_store import TrajectoryStoreBuilder


def _store(attack_windows=(), unknown=None, n=60, ns="pcap/day"):
    b = TrajectoryStoreBuilder(spill_dir=None)
    b.set_namespace(ns)
    for w in range(n):
        atk = w in attack_windows
        b.append(host_ip="10.0.0.1", host_id=1, window_idx=w, window_start=2.0 * w,
                 window_end=2.0 * w + 2, embedding=np.zeros(12, np.float32),
                 temporal_attrs=np.zeros(15, np.float32), is_attack=atk,
                 coarse_category="Impact" if atk else "Benign",
                 technique_ids=["T1498"] if atk else [], risk_score=0.8 if atk else 0.0)
    if unknown:
        b.add_unknown_intervals(unknown)
    return b.finalize()


def test_intervals_merge():
    iv = unknown_intervals([10, 11, 30, 31.5], [10.5, 12, 31, 32])
    assert iv == [[10.0, 12.0], [30.0, 32.0]]


def test_windows_before_an_unknown_span_are_censored_not_negative():
    tau = 10.0
    st = _store(unknown=[[60.0, 80.0]])          # windows 30..40 dropped as UNKNOWN
    h = st.hazard_risk(tau)
    t = 2.0 * np.arange(60)
    # right before the span: unknown, not 0
    assert np.isnan(h[(t >= 30) & (t < 60)]).all()
    # far before it (> 4.6 tau): the target is ~0 whatever happened there -> kept
    assert (h[t < 10] == 0).all()
    # after the span with no later attack: genuinely 0
    assert (h[t > 80] == 0).all()


def test_a_known_attack_before_the_span_is_kept():
    st = _store(attack_windows=(20,), unknown=[[60.0, 80.0]])
    h = st.hazard_risk(10.0)
    assert h[20] == 1.0                           # the attack window itself
    assert np.isfinite(h[15:21]).all()            # its next event is the known attack


def test_other_captures_are_not_censored():
    st = _store(unknown=None)
    st2 = _store(unknown=[[60.0, 80.0]], ns="other")
    assert np.isfinite(st.hazard_risk(10.0)).all()
    assert np.isnan(st2.hazard_risk(10.0)).any()


def test_branch_a_loss_ignores_censored_targets():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    m = MultiTaskLSTM(input_dim=27, risk_objective="soft_bce", **{
        k: v for k, v in MultiTaskLSTM.PAPER_ARCH.items()})
    pred = {"risk_score": torch.tensor([0.9, 0.1, 0.5]),
            "technique_logits": torch.zeros(3, m.num_techniques),
            "gradation_score": torch.zeros(3), "gradation_logits": torch.zeros(3, 4)}
    batch = {"technique": torch.zeros(3, dtype=torch.long), "gradation": torch.zeros(3, dtype=torch.long)}
    a = m.task_losses(pred, {**batch, "risk": torch.tensor([1.0, 0.0, float("nan")])})[0]
    b = m.task_losses({k: v[:2] for k, v in pred.items()},
                      {"technique": batch["technique"][:2], "gradation": batch["gradation"][:2],
                       "risk": torch.tensor([1.0, 0.0])})[0]
    assert torch.isfinite(a) and torch.allclose(a, b)


def test_column_load_records_when_unknown_traffic_was_dropped(tmp_path):
    from data_unification import capture_columns as cc
    from data_unification.unified_schema import UnifiedFlowRecord

    def rec(t, cat):
        return UnifiedFlowRecord(src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=80,
                                 protocol=6, start_time=t, end_time=t + 0.5, fwd_bytes=10,
                                 bwd_bytes=10, fwd_packets=1, bwd_packets=1, raw_label="x",
                                 raw_label_source="CIC2018", is_attack=False, coarse_category=cat)
    recs = [rec(0.0, "Benign"), rec(100.0, "UNKNOWN"), rec(101.0, "UNKNOWN"), rec(200.0, "Benign")]
    spec = cc.ColumnSpec("read_capture", "CIC2018", "x", "/x.csv", 2.0)
    cc.build_columns(spec, tmp_path / "c", records=recs)
    cols = cc.CaptureColumns.load(tmp_path / "c", drop_unresolved=True)
    assert len(cols) == 2
    assert cols.meta["unknown_intervals"] == [[100.0, 101.5]]
