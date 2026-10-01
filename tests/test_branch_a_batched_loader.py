"""The batched Branch A loader must yield EXACTLY what the per-sample one did.

`LazyHostSequenceDataset.gather_batch` (vectorised, host-major features) and
`PermutationBatchSampler` (no 68M-element list) are speed changes only. These
tests pin bit-identity against the original path -- `[ds[i] for i in batch]`
through `default_collate`, sampled by `DataLoader(shuffle=True)` -- for every
tensor of every batch, the batch order, and the global RNG state afterwards.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "perf"))
from synth_store import make_store  # noqa: E402

from branch_a_gnn_lstm.sequence_dataset import (  # noqa: E402
    BatchedSequenceView, LazyHostSequenceDataset, PermutationBatchSampler,
    collate_prebatched)
from data_unification.host_major import flat_row_order  # noqa: E402


def _same(a, b):
    assert a.keys() == b.keys()
    assert list(a) == list(b)
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].dtype == b[k].dtype, k
            assert a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.fixture(scope="module", params=["ram", "memmap"])
def store(request, tmp_path_factory):
    path = None
    if request.param == "memmap":
        path = str(tmp_path_factory.mktemp("feats") / "f.bin")
    # spread a few long trajectories over many windows, so time gaps vary
    return make_store(60_000, 3_000, n_windows=5_000, seed=3, feats_path=path,
                      small_frac=0.8)


CASES = [dict(), dict(min_history_steps=3), dict(min_history_steps=1, max_gap_seconds=40.0)]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("host_major", [False, True])
def test_gather_batch_is_default_collate_of_getitem(store, kw, host_major, tmp_path):
    ds = LazyHostSequenceDataset(store, seq_len=15, min_trajectory_len=1, **kw)
    assert len(ds) > 1000
    ds.enable_batched(host_major=host_major, spill_dir=str(tmp_path))
    rng = np.random.default_rng(0)
    padded = 0
    for _ in range(40):
        idx = rng.integers(0, len(ds), 128)
        ref = default_collate([ds[int(i)] for i in idx])
        new = ds.gather_batch(idx)
        _same(ref, new)
        padded += int((ref["features"] == 0).all(-1).any(-1).sum())
    if kw.get("min_history_steps"):
        assert padded > 0, "the case meant to exercise left padding had none"


def test_hazard_target_risk_is_identical(tmp_path):
    st = make_store(20_000, 800, n_windows=3_000, seed=5, small_frac=0.8)
    st.use_hazard_target(10.0)
    ds = LazyHostSequenceDataset(st, seq_len=15, min_trajectory_len=1, min_history_steps=2)
    ds.enable_batched()
    idx = np.arange(0, len(ds), 7)[:512]
    _same(default_collate([ds[int(i)] for i in idx]), ds.gather_batch(idx))


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_order_content_and_rng(store, workers):
    ds = LazyHostSequenceDataset(store, seq_len=15, min_trajectory_len=1, min_history_steps=4)
    ds.enable_batched()

    def run(new):
        torch.manual_seed(1234)
        if new:
            dl = DataLoader(BatchedSequenceView(ds),
                            batch_sampler=PermutationBatchSampler(len(ds), 128),
                            collate_fn=collate_prebatched, num_workers=workers)
        else:
            dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=workers)
        out = []
        for _epoch in range(2):
            out.append(list(dl))
        return out, torch.get_rng_state(), len(dl)

    ref, ref_rng, ref_len = run(False)
    new, new_rng, new_len = run(True)
    assert ref_len == new_len
    assert torch.equal(ref_rng, new_rng)
    for e_ref, e_new in zip(ref, new):
        assert len(e_ref) == len(e_new)
        for a, b in zip(e_ref, e_new):
            _same(a, b)


def test_flat_row_order_fallback_matches_views():
    base = np.arange(100, dtype=np.int64)[::-1].copy()
    views = [base[0:10], base[10:11], base[11:60], base[60:100]]
    flat, off = flat_row_order(views)
    assert flat is base
    for v, o in zip(views, off):
        assert np.array_equal(flat[o:o + len(v)], v)
    loose = [np.array([5, 3, 9]), np.array([1]), np.array([7, 2])]
    flat, off = flat_row_order(loose)
    for v, o in zip(loose, off):
        assert np.array_equal(flat[o:o + len(v)], v)


def test_host_major_copy_multi_bucket_from_memmap(tmp_path):
    from data_unification.host_major import host_major_copy
    rng = np.random.default_rng(9)
    src = np.memmap(tmp_path / "s.bin", dtype=np.float32, mode="w+", shape=(50_001, 27))
    src[:] = rng.standard_normal(src.shape, dtype=np.float32)
    src.flush()
    ro = np.memmap(tmp_path / "s.bin", dtype=np.float32, mode="r", shape=(50_001, 27))
    flat = rng.permutation(50_001)[:40_000]
    out = host_major_copy(ro, flat, str(tmp_path), bucket_bytes=27 * 4 * 3_333)
    assert isinstance(out, np.memmap)
    assert np.array_equal(np.asarray(out), np.asarray(ro)[flat])
