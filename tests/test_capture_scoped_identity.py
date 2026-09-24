"""A host in one capture is never the same trajectory as a host in another.

Hosts were keyed by the bare IP string, and the trainers feed one shared
builder capture after capture, so the same string in two captures became ONE
trajectory. Measured on the train split: fri_16 & wed_14 share all 350 hosts
(9 of 10 CIC-2018 days fabricate `192.168.10.{i % 250 + 1}` from one pool),
two CTU-13 scenarios share 28,474, and tue_20 (2018) shares 59 with a 2011
CTU-13 scenario. Rows are appended in file order, so a merged host's history
could come from 2018 and its "next window" from 2011.
"""
import numpy as np
import pytest

from data_unification.trajectory_store import TrajectoryStoreBuilder, capture_namespace

EMB = np.zeros(12, dtype=np.float32)
ATTRS = np.zeros(15, dtype=np.float32)


def _add(b, ip, w, t0=0.0, atk=False):
    b.append(host_ip=ip, host_id=0, window_idx=w,
             window_start=t0 + w * 2.0, window_end=t0 + w * 2.0 + 2.0,
             embedding=EMB, temporal_attrs=ATTRS, is_attack=atk,
             coarse_category="C2" if atk else "Benign",
             technique_ids=["T1071"] if atk else [], risk_score=0.82 if atk else 0.0)


def test_the_same_ip_in_two_captures_is_two_trajectories():
    b = TrajectoryStoreBuilder(spill_dir=None)
    b.set_namespace("CSV/wed_14_csv")
    for w in range(5):
        _add(b, "192.168.10.5", w)
    b.set_namespace("CSV/fri_16_csv")
    for w in range(5):
        _add(b, "192.168.10.5", w, t0=1e6)
    st = b.finalize()
    assert len(st) == 2, f"expected two trajectories, got {list(st)}"
    assert all(len(st[h]) == 5 for h in st)


def test_the_same_ip_within_one_capture_is_one_trajectory():
    b = TrajectoryStoreBuilder(spill_dir=None)
    b.set_namespace("CSV/wed_14_csv")
    for w in range(7):
        _add(b, "192.168.10.5", w)
    st = b.finalize()
    assert len(st) == 1 and len(next(iter(st.values()))) == 7


def test_no_namespace_keeps_bare_ip_keys():
    """Backward compatible: live serving and older callers are unchanged."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    _add(b, "10.0.0.1", 0)
    st = b.finalize()
    assert list(st) == ["10.0.0.1"]


def test_a_later_capture_can_no_longer_supply_an_earlier_one_s_target():
    """The time-reversal case: a 2018 capture appended before a 2011 one."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    b.set_namespace("CSV/tue_20_csv")
    for w in range(4):
        _add(b, "147.32.84.165", w, t0=1.5e9)        # 2017-2018 epoch
    b.set_namespace("1/capture20110810")
    for w in range(4):
        _add(b, "147.32.84.165", w, t0=1.31e9, atk=True)  # 2011 epoch
    st = b.finalize()
    for h in st:
        starts = np.asarray(st.window_start)[st._rows_by_host[h]]
        assert np.all(np.diff(starts) >= 0), f"{h}: trajectory runs backwards in time"


def test_the_hazard_target_does_not_reach_across_captures():
    """An attack in capture B must not raise risk for the same address in A."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    b.set_namespace("A")
    for w in range(4):
        _add(b, "10.0.0.9", w)                       # clean in A
    b.set_namespace("B")
    for w in range(4):
        _add(b, "10.0.0.9", w, t0=10.0, atk=True)    # attacked in B, 10 s later
    st = b.finalize()
    h = st.hazard_risk(tau_seconds=10.0)
    clean_rows = st._rows_by_host["10.0.0.9@A"]
    assert np.all(h[clean_rows] == 0.0), "capture B's attack leaked into capture A"


def test_capture_namespace_distinguishes_ctu13_scenarios():
    """CTU-13 disambiguates scenarios by directory, so the parent is required."""
    a = capture_namespace("/data/CTU-13-Dataset/1/capture20110810.binetflow")
    b = capture_namespace("/data/CTU-13-Dataset/2/capture20110810.binetflow")
    assert a != b
    assert capture_namespace("/data/CSV/wed_14_csv.csv") == "CSV/wed_14_csv"
