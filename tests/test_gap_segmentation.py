"""Opt-in time-gap segmentation for Branch A's sequence samples.

Rows in a trajectory are a host's ACTIVE windows, not its clock ticks, so the
"next window" is whenever the host next appears. Measured with the real
adapters on the train split: consecutive active windows of a CTU-13 host are a
median 736 s apart and over an hour apart in 35% of pairs, against a 2 s
window and a 10 s forecast horizon.

max_gap_seconds cuts a trajectory where two consecutive windows are further
apart than that, and no sample's history or target crosses a cut. Off by
default, because enabling it removes samples and the no-dilution rule applies.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data_unification.trajectory_store import TrajectoryStoreBuilder
from branch_a_gnn_lstm.sequence_dataset import LazyHostSequenceDataset

EMB = np.zeros(12, dtype=np.float32)


def _store(times, ip="10.0.0.1"):
    """One host, one active window at each given start time."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    for i, t in enumerate(times):
        attrs = np.full(15, float(i), dtype=np.float32)   # row i is recognisable
        b.append(host_ip=ip, host_id=0, window_idx=i,
                 window_start=float(t), window_end=float(t) + 2.0,
                 embedding=EMB, temporal_attrs=attrs, is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    return b.finalize()


def test_default_is_unchanged():
    st = _store([0, 2, 4, 3600, 3602])
    a = LazyHostSequenceDataset(st, seq_len=3)
    b = LazyHostSequenceDataset(st, seq_len=3, max_gap_seconds=None)
    assert len(a) == len(b) == 4          # ends 1..4
    assert b.n_dropped_by_gap == 0


def test_a_target_after_a_gap_is_dropped():
    # windows at 0,2,4 | gap | 3600,3602  -> the target at 3600 opens a segment
    st = _store([0, 2, 4, 3600, 3602])
    ds = LazyHostSequenceDataset(st, seq_len=3, max_gap_seconds=10.0)
    assert ds.n_dropped_by_gap == 1
    assert len(ds) == 3                   # ends at 2, 4, and 3602
    ends = sorted(int(e) for e in ds._pos)
    assert ends == [1, 2, 4]


def test_history_never_reaches_across_a_gap():
    st = _store([0, 2, 4, 3600, 3602, 3604])
    ds = LazyHostSequenceDataset(st, seq_len=5, max_gap_seconds=10.0)
    # sample whose target is row 5 (t=3604): history may only be rows 3,4
    idx = [i for i in range(len(ds)) if int(ds._pos[i]) == 5][0]
    x = ds[idx]["features"].numpy()
    # Row i's attributes are all equal to i, so column 12 names the row.
    # Rows 0-2 are before the gap; only rows 3 and 4 may appear, left-padded.
    assert x.shape == (5, 27)
    assert np.all(x[:3] == 0), "the three missing steps must be zero padding"
    assert [int(r[12]) for r in x[3:]] == [3, 4], "history reached across the gap"


def test_no_gap_means_no_change_even_when_enabled():
    st = _store([0, 2, 4, 6, 8])
    a = LazyHostSequenceDataset(st, seq_len=3)
    b = LazyHostSequenceDataset(st, seq_len=3, max_gap_seconds=10.0)
    assert len(a) == len(b) and b.n_dropped_by_gap == 0
    for i in range(len(a)):
        np.testing.assert_array_equal(a[i]["features"].numpy(), b[i]["features"].numpy())


def test_the_gap_threshold_is_strict_greater_than():
    """A gap exactly equal to the threshold does not cut."""
    st = _store([0, 10, 20])
    ds = LazyHostSequenceDataset(st, seq_len=3, max_gap_seconds=10.0)
    assert ds.n_dropped_by_gap == 0 and len(ds) == 2
