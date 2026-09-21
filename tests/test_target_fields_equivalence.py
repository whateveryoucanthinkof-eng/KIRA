"""The fast target reads must be bit-identical to building a full snapshot.

`LazyHostSequenceDataset.__getitem__` and `LazyHostRolloutDataset.__getitem__`
used to call `TrajectoryStore._materialize(row)` and then read three or four
fields off the resulting `HostWindowSnapshot`. That cost ~86us per call and ran
in the main process, so the GPU waited on it (measured: 29.6 min/epoch of pure
target construction at 20.66M samples).

They now read the columns directly. These tests pin that the shortcut returns
exactly what the snapshot would have -- if it ever diverges, the model silently
trains on different labels, which is the worst way for this to fail.
"""
import numpy as np
import pytest

from data_unification.trajectory_store import TrajectoryStoreBuilder


def _store(n_hosts=12, n_windows=60, seed=7):
    rng = np.random.default_rng(seed)
    cats = ["Benign", "Execution", "InitialAccess", "C2", "Recon"]
    techs = [["T1071"], ["T1059", "T1105"], [], ["T1190"]]
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(n_hosts):
        for w in range(n_windows):
            k = int(rng.integers(0, len(cats)))
            b.append(
                host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                window_start=float(w * 2), window_end=float(w * 2 + 2),
                embedding=rng.random(12).astype(np.float32),
                temporal_attrs=rng.random(15).astype(np.float32),
                is_attack=(k != 0), coarse_category=cats[k],
                technique_ids=list(techs[int(rng.integers(0, len(techs)))]),
                risk_score=float(rng.random()),
            )
    return b.finalize()


def test_target_fields_matches_materialize_on_every_row():
    st = _store()
    for row in range(st.n_snapshots):
        snap = st._materialize(row)
        risk, category, first_tech, widx = st.target_fields(row)

        assert risk == snap.risk_score
        assert category == snap.coarse_category
        assert widx == snap.window_idx
        expected = snap.technique_ids[0] if snap.technique_ids else None
        assert first_tech == expected, f"row {row}: {first_tech!r} != {expected!r}"


def test_branch_a_dataset_items_match_the_materialize_path():
    """Same assertion one level up: through the actual Dataset."""
    torch = pytest.importorskip("torch")
    from branch_a_gnn_lstm.sequence_dataset import (
        LazyHostSequenceDataset, TECH_TO_IDX, GRADATION_LEVELS)

    st = _store()
    ds = LazyHostSequenceDataset(st, seq_len=5)
    assert len(ds) > 0

    for idx in range(0, len(ds), max(1, len(ds) // 200)):
        item = ds[idx]
        host = ds.hosts[int(ds._host_idx[idx])]
        end = int(ds._pos[idx])
        rows = st._rows_by_host[host]

        # recompute the old way, from a fully materialised snapshot
        snap = st._materialize(int(rows[end]))
        tech = snap.technique_ids[0] if snap.technique_ids else "Benign"

        assert item["risk"].item() == pytest.approx(snap.risk_score)
        assert item["technique"].item() == TECH_TO_IDX.get(
            tech, TECH_TO_IDX.get("Benign", 0))
        assert item["gradation"].item() == GRADATION_LEVELS.get(snap.coarse_category, 0)
        assert item["window_idx"] == snap.window_idx
        assert item["host_ip"] == host


def test_branch_b_risk_future_matches_materialize_path():
    torch = pytest.importorskip("torch")
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset

    st = _store()
    ds = LazyHostRolloutDataset(st, K=5, T=15)
    assert len(ds) > 0

    for idx in range(0, len(ds), max(1, len(ds) // 200)):
        item = ds[idx]
        host = ds.hosts[int(ds._host_idx[idx])]
        i = int(ds._pos[idx])
        rows = st._rows_by_host[host]
        fut_rows = rows[i:i + ds.K]

        expected = np.asarray(
            [st._materialize(int(r)).risk_score for r in fut_rows], dtype=np.float32)
        if len(fut_rows) < ds.K:
            expected = np.pad(expected, (0, ds.K - len(fut_rows)), mode="edge")

        np.testing.assert_array_equal(item["risk_future"].numpy(), expected)


def test_target_fields_is_actually_faster():
    """Guards the reason the shortcut exists; a regression here is a silent
    return to a GPU-starving data path."""
    import time
    st = _store(n_hosts=4, n_windows=200)
    rows = list(range(st.n_snapshots)) * 6

    for r in rows[:500]:                      # warm both paths, and the
        st._materialize(r)                    # multi_dataset_stream import
        st.target_fields(r)                   # that _materialize does lazily

    t = time.perf_counter()
    for r in rows:
        st._materialize(r)
    slow = time.perf_counter() - t

    t = time.perf_counter()
    for r in rows:
        st.target_fields(r)
    fast = time.perf_counter() - t

    ratio = slow / max(fast, 1e-12)
    # Measured ~3.6x on an in-RAM store. Asserting 2x leaves room for a loaded
    # machine while still failing if someone reintroduces snapshot building.
    assert ratio > 2.0, f"target_fields only {ratio:.1f}x faster than _materialize"
