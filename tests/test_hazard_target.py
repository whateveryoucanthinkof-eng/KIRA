"""A forward-looking risk target, and why the old one could not be learned.

`risk_score` as written during extraction is
`base_severity(tactic) + 0.04*density + 0.04*volume` for an attack window and
exactly 0.0 otherwise. Measured consequences on the 2026-09-21 full-density
run:

  * 82.5% of validation targets are exactly 0.0, the rest cluster near 0.76
  * the smooth-L1-optimal constant is 0.1400, whose MAE is 0.2279
  * Branch A's trained risk head scored 0.2268 -- it had converged to the
    unconditional mean, and could not have beaten the constant 0 (MAE 0.1334)
    whatever it learned, because smooth L1 is mean-seeking and MAE is
    median-seeking

It is also not a forecast: it describes whether THIS window contains attack
traffic, so predicting it is delayed detection rather than forecasting. And it
is close to a function of the coarse category, which another head predicts.

`hazard_risk(tau)` = exp(-seconds_until_next_attack / tau) replaces all three
problems: continuous, forward-looking, and independent of the tactic label.
"""
import numpy as np
import pytest

from data_unification.trajectory_store import TrajectoryStoreBuilder

EMB = np.zeros(12, dtype=np.float32)
ATTRS = np.zeros(15, dtype=np.float32)


def _store(spec, window=2.0):
    """spec: {host: [is_attack, ...]} one entry per consecutive window."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    for host, flags in spec.items():
        for w, atk in enumerate(flags):
            b.append(host_ip=host, host_id=0, window_idx=w,
                     window_start=float(w * window), window_end=float(w * window + window),
                     embedding=EMB, temporal_attrs=ATTRS,
                     is_attack=bool(atk), coarse_category="C2" if atk else "Benign",
                     technique_ids=["T1071"] if atk else [],
                     risk_score=0.82 if atk else 0.0)
    return b.finalize()


def test_time_to_next_attack_counts_forward_only():
    st = _store({"10.0.0.1": [0, 0, 1, 0, 0]})
    dt = st.time_to_next_attack()
    rows = st._rows_by_host["10.0.0.1"]
    got = dt[rows]
    # windows start at 0,2,4,6,8; the attack is at t=4
    assert got[0] == pytest.approx(4.0)
    assert got[1] == pytest.approx(2.0)
    assert got[2] == pytest.approx(0.0), "an attack window is zero seconds away"
    # nothing after the attack -> never
    assert not np.isfinite(got[3]) and not np.isfinite(got[4])


def test_a_host_never_attacked_is_infinitely_far():
    st = _store({"10.0.0.9": [0, 0, 0]})
    dt = st.time_to_next_attack()
    assert np.all(~np.isfinite(dt[st._rows_by_host["10.0.0.9"]]))


def test_hosts_do_not_bleed_into_each_other():
    """The bug this guards: computing the hazard globally rather than per host
    would make one host's attack lower another host's risk."""
    st = _store({"10.0.0.1": [0, 0, 0], "10.0.0.2": [1, 1, 1]})
    dt = st.time_to_next_attack()
    assert np.all(~np.isfinite(dt[st._rows_by_host["10.0.0.1"]])), "clean host must stay clean"
    assert np.all(dt[st._rows_by_host["10.0.0.2"]] == 0.0)


def test_hazard_is_one_during_an_attack_and_decays_before_it():
    st = _store({"10.0.0.1": [0, 0, 0, 0, 0, 1]})
    h = st.hazard_risk(tau_seconds=10.0)
    rows = st._rows_by_host["10.0.0.1"]
    got = h[rows]
    assert got[-1] == pytest.approx(1.0), "attack window scores 1.0"
    assert np.all(np.diff(got) > 0), "risk must rise monotonically toward the attack"
    # one full horizon out (10s -> windows 5 back) is exp(-1)
    assert got[0] == pytest.approx(np.exp(-1.0), abs=1e-6)


def test_hazard_is_zero_for_a_host_never_attacked():
    st = _store({"10.0.0.9": [0, 0, 0]})
    assert np.all(st.hazard_risk(10.0) == 0.0)


def test_hazard_is_continuous_not_bimodal():
    """The property that makes it learnable: the old target took two values,
    this one spreads across the range."""
    st = _store({f"10.0.0.{i}": [0] * i + [1] for i in range(1, 12)})
    h = st.hazard_risk(tau_seconds=10.0)
    distinct = np.unique(np.round(h, 4))
    assert len(distinct) > 8, f"expected a spread of values, got {distinct}"
    # and the old target really is bimodal, for contrast
    old = np.asarray(st.risk_score)
    assert len(np.unique(np.round(old, 4))) == 2


def test_hazard_does_not_depend_on_the_tactic_label():
    """The old target was base_severity(tactic) -- i.e. a relabelling of the
    category, which a different head already predicts."""
    a = _store({"h": [0, 0, 1]})
    b = TrajectoryStoreBuilder(spill_dir=None)
    for w, atk in enumerate([0, 0, 1]):
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=EMB, temporal_attrs=ATTRS, is_attack=bool(atk),
                 coarse_category="Impact" if atk else "Benign",   # different tactic
                 technique_ids=["T1485"] if atk else [],
                 risk_score=0.96 if atk else 0.0)                  # different severity
    other = b.finalize()
    np.testing.assert_allclose(a.hazard_risk(10.0), other.hazard_risk(10.0))
    assert not np.allclose(np.asarray(a.risk_score), np.asarray(other.risk_score))


def test_tau_must_be_positive():
    st = _store({"h": [0, 1]})
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="tau_seconds"):
            st.hazard_risk(bad)


def test_out_of_order_rows_are_sorted_not_silently_wrong():
    st = _store({"h": [0, 0, 1]})
    # scramble the host's row order the way a merge across captures could
    st._rows_by_host["h"] = st._rows_by_host["h"][::-1]
    got = st.hazard_risk(10.0)[st._rows_by_host["h"]]
    # reversed rows: [attack, 2s before, 4s before]
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(np.exp(-0.2), abs=1e-6)
    assert got[2] == pytest.approx(np.exp(-0.4), abs=1e-6)


def test_use_hazard_target_swaps_the_column_every_consumer_reads():
    """Branch A, Branch B and DeepOP all read store.risk_score, so the swap
    has to happen there rather than in three separate datasets."""
    st = _store({"h": [0, 0, 0, 1]})
    before = np.asarray(st.risk_score).copy()
    info = st.use_hazard_target(tau_seconds=10.0)
    assert not np.allclose(np.asarray(st.risk_score), before)
    np.testing.assert_allclose(np.asarray(st.risk_score_severity), before)
    assert info["zero_fraction_after"] < info["zero_fraction_before"]
    assert info["distinct_after"] > info["distinct_before"]


def test_applying_the_hazard_twice_is_refused():
    """Decaying an already-decayed target would silently square it."""
    st = _store({"h": [0, 1]})
    st.use_hazard_target(10.0)
    with pytest.raises(RuntimeError, match="already applied"):
        st.use_hazard_target(10.0)


def test_the_lazy_dataset_picks_up_the_swapped_target():
    """End to end through the reader Branch A actually uses."""
    pytest.importorskip("torch")
    from branch_a_gnn_lstm.sequence_dataset import LazyHostSequenceDataset
    st = _store({"h": [0, 0, 0, 0, 1]})
    ds_before = LazyHostSequenceDataset(st, seq_len=3)
    r_before = [float(ds_before[i]["risk"]) for i in range(len(ds_before))]
    st.use_hazard_target(10.0)
    ds_after = LazyHostSequenceDataset(st, seq_len=3)
    r_after = [float(ds_after[i]["risk"]) for i in range(len(ds_after))]
    assert r_before != r_after
    assert all(0.0 <= v <= 1.0 for v in r_after)
    assert r_after == sorted(r_after), "risk must rise as the attack approaches"


# --- the spill block must not become a disk-thrashing memmap ---------------

def test_a_small_feature_block_is_held_in_ram_not_memmapped(tmp_path):
    """Why: training reads the block in random order, 20.66M gathers an epoch,
    and a memmap page fault pulls the kernel's default 128 KiB readahead for a
    108-byte row. On 2026-09-22 four workers each read 10.4 TiB from a 2.30 GiB
    block and throughput fell 52.5 -> 8.1 batch/s."""
    b = TrajectoryStoreBuilder(spill_dir=str(tmp_path))
    for w in range(200):
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=EMB, temporal_attrs=ATTRS, is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    st = b.finalize()
    assert not isinstance(st.feats, np.memmap), "small block should be resident"
    assert st.feats.shape == (200, 27)
    assert st.n_snapshots == 200


def test_a_block_over_the_threshold_stays_memmapped(tmp_path, monkeypatch):
    """The escape hatch still works: a block too large to hold stays mapped."""
    import data_unification.trajectory_store as ts
    monkeypatch.setattr(ts, "RESIDENT_FEATS_MAX_BYTES", 1)   # force the mapped path
    b = ts.TrajectoryStoreBuilder(spill_dir=str(tmp_path))
    for w in range(50):
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=EMB, temporal_attrs=ATTRS, is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    st = b.finalize()
    assert isinstance(st.feats, np.memmap)
    assert st.feats.shape == (50, 27)


def test_resident_block_still_reads_back_the_right_values(tmp_path):
    """Correctness of the copy, not just its type."""
    rng = np.random.default_rng(4)
    embs, attrs = [], []
    b = TrajectoryStoreBuilder(spill_dir=str(tmp_path))
    for w in range(120):
        e = rng.random(12).astype(np.float32); a = rng.random(15).astype(np.float32)
        embs.append(e); attrs.append(a)
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=e, temporal_attrs=a, is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    st = b.finalize()
    expect = np.concatenate([np.stack(embs), np.stack(attrs)], axis=1)
    np.testing.assert_allclose(np.asarray(st.feats), expect, rtol=0, atol=0)


def test_both_splits_must_be_swapped_together():
    """Swapping only one leaves train and validation on different scales,
    which would look like a generalisation gap and is really a unit mismatch."""
    a = _store({"h": [0, 0, 1]})
    b = _store({"h2": [0, 1, 0]})
    a.use_hazard_target(10.0)
    assert np.asarray(a.risk_score).max() <= 1.0
    # b untouched: still the bimodal severity. Compared with tolerance, not
    # set equality -- risk_score is float32, so 0.82 round-trips as
    # 0.81999999284... and an exact set comparison against a Python float
    # fails on dtype rather than on anything meaningful.
    vals = np.unique(np.round(np.asarray(b.risk_score), 4))
    assert len(vals) == 2, f"expected a bimodal target, got {vals}"
    np.testing.assert_allclose(sorted(vals), [0.0, 0.82], atol=1e-4)
    b.use_hazard_target(10.0)
    assert len(np.unique(np.round(np.asarray(b.risk_score), 4))) > 1


def test_the_downstream_trainer_exposes_the_flag():
    """Pins that Branch B / DeepOP can actually reach the hazard target."""
    src = open("scripts/retrain_future_models_live.py").read()
    assert '"--risk-target"' in src
    assert "use_hazard_target" in src
    # applied to BOTH stores
    assert 'for _nm, _st in (("train", train_traj), ("val", val_traj))' in src
