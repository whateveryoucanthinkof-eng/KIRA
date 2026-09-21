"""A spilled feature block must not leave a file behind.

`TrajectoryStoreBuilder(spill_dir=...)` writes the bulk (N, 27) float32 block
to disk and memmaps it back, which is what makes full-density extraction fit
in RAM. Nothing ever deleted those files. At full density each of Branch A,
Branch B and DeepOP writes a multi-GB block, and this machine's disk is
already 90% full -- a handful of runs would have filled it. A 216 MB orphan
from an earlier run is what exposed it.

The fix unlinks the file immediately after mapping. On POSIX the inode
survives while the mapping holds a reference, so the data stays valid and the
space is reclaimed automatically -- including on a crash or a kill, which no
explicit cleanup path can guarantee.
"""

import os

import numpy as np
import pytest

from data_unification.trajectory_store import FEAT_DIM, TrajectoryStoreBuilder


def _add(b, host, widx, seed=0):
    rng = np.random.default_rng(seed)
    b.append(
        host_ip=host,
        host_id=hash(host) % 100000,
        window_idx=widx,
        window_start=float(widx * 2),
        window_end=float(widx * 2 + 2),
        embedding=rng.random(12).astype(np.float32),
        temporal_attrs=rng.random(15).astype(np.float32),
        is_attack=False,
        coarse_category="Benign",
        risk_score=0.1,
        technique_ids=["T1046"],
    )


def _build(spill_dir, n_hosts=8, n_win=9000):
    b = TrajectoryStoreBuilder(spill_dir=str(spill_dir))
    for h in range(n_hosts):
        for w in range(n_win):
            _add(b, f"10.0.0.{h}", w, seed=h * 100 + w)
    return b


def test_no_file_is_left_in_the_spill_directory(tmp_path):
    store = _build(tmp_path).finalize()
    assert store.n_snapshots == 8 * 9000
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"spill files left behind: {[p.name for p in leftovers]}"


def test_the_data_is_still_readable_after_unlinking(tmp_path):
    """The whole point: unlinking must not break the live mapping."""
    store = _build(tmp_path).finalize()
    for host in store:
        snaps = store[host]
        assert len(snaps) == 9000
        for s in snaps:
            assert s.embedding.shape == (12,)
            assert s.temporal_attrs.shape == (15,)
            assert np.isfinite(s.embedding).all()


def test_spilled_and_in_memory_stores_agree_exactly(tmp_path):
    """Spilling is a memory strategy, never a change in content."""
    spilled = _build(tmp_path).finalize()
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(8):
        for w in range(9000):
            _add(b, f"10.0.0.{h}", w, seed=h * 100 + w)
    in_ram = b.finalize()

    assert sorted(spilled.keys()) == sorted(in_ram.keys())
    for host in in_ram:
        a, c = spilled[host], in_ram[host]
        assert len(a) == len(c)
        for x, y in zip(a, c):
            np.testing.assert_array_equal(x.embedding, y.embedding)
            np.testing.assert_array_equal(x.temporal_attrs, y.temporal_attrs)
            assert x.window_idx == y.window_idx
            assert x.coarse_category == y.coarse_category


def test_the_block_really_went_to_disk(tmp_path):
    """Guard the guard: if it never spilled, the cleanup test is vacuous."""
    b = _build(tmp_path)
    assert b._spill_path is not None
    assert os.path.exists(b._spill_path), "nothing was written to disk"
    store = b.finalize()
    assert store.memory_report()["feats_on_disk"] is True
    assert not os.path.exists(b._spill_path), "file survived finalize()"


def test_an_empty_store_spills_nothing_and_leaves_nothing(tmp_path):
    store = TrajectoryStoreBuilder(spill_dir=str(tmp_path)).finalize()
    assert store.n_snapshots == 0
    assert list(tmp_path.iterdir()) == []
