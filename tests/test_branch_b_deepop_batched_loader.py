"""Branch B / DeepOP batched loaders must yield EXACTLY what the per-sample ones did.

`gather_batch` (vectorised, host-major features, per-row token table), the
vectorised DeepOP oversampling mask and `target_token_histogram` are speed
changes only. Each is compared with the original per-sample code -- the
references below are the pre-2026-10 implementations, verbatim in substance.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "perf"))
from synth_store import make_store  # noqa: E402

from branch_b_world_model.train_branch_b import LazyHostRolloutDataset  # noqa: E402
from data_unification.host_major import batched_loader  # noqa: E402
from deepop_decoder.joint_vocab import get_joint_vocab  # noqa: E402
from deepop_decoder.train_cwa_decoder import LazyCWADataset  # noqa: E402


def _same(a, b):
    assert list(a) == list(b)
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.fixture(scope="module", params=["ram", "memmap"])
def store(request, tmp_path_factory):
    path = None
    if request.param == "memmap":
        path = str(tmp_path_factory.mktemp("f") / "f.bin")
    st = make_store(50_000, 2_500, n_windows=4_000, seed=11, feats_path=path,
                    small_frac=0.85, attack_rate=0.3)
    st.use_hazard_target(10.0)
    return st


def _check_batches(ds, n_batches=30, bs=128, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        idx = rng.integers(0, len(ds), bs)
        _same(default_collate([ds[int(i)] for i in idx]), ds.gather_batch(idx))
    # the edges: first and last samples (short trajectories, edge padding)
    for idx in (np.arange(min(bs, len(ds))), np.arange(max(0, len(ds) - bs), len(ds))):
        _same(default_collate([ds[int(i)] for i in idx]), ds.gather_batch(idx))


@pytest.mark.parametrize("host_major", [False, True])
@pytest.mark.parametrize("emit_times", [True, False])
def test_branch_b_gather_batch(store, host_major, emit_times, tmp_path):
    ds = LazyHostRolloutDataset(store, T=15, K=5, emit_times=emit_times)
    assert len(ds) > 1000
    ds.enable_batched(host_major=host_major, spill_dir=str(tmp_path))
    _check_batches(ds)
    # edge padding of the future must actually occur in this store
    n_short = sum(1 for h, p in zip(ds._host_idx, ds._pos)
                  if p + 5 > len(store._rows_by_host[ds.hosts[h]]))
    assert n_short > 0


@pytest.mark.parametrize("host_major", [False, True])
def test_branch_b_short_histories_are_edge_padded_identically(store, host_major, tmp_path):
    """min_history_steps=1: histories shorter than T are edge-padded (first
    real state and its time repeated) in both access paths, as serving pads."""
    ds = LazyHostRolloutDataset(store, T=15, K=5, min_history_steps=1)
    full = LazyHostRolloutDataset(store, T=15, K=5)
    assert len(ds) > len(full)
    ds.enable_batched(host_major=host_major, spill_dir=str(tmp_path))
    _check_batches(ds)
    k = int(np.flatnonzero(ds._pos == 1)[0])          # one real history step
    s = ds[k]
    assert torch.equal(s["h_history"][0], s["h_history"][-1])
    assert float(s["t_history"][0]) == 0.0 and float(s["t_history"][-1]) == 0.0


@pytest.mark.parametrize("host_major", [False, True])
@pytest.mark.parametrize("T,oversample", [(15, True), (15, False), (0, True), (3, False)])
def test_deepop_gather_batch(store, host_major, T, oversample, tmp_path):
    vocab = get_joint_vocab(network_observable_only=True)
    ds = LazyCWADataset(store, vocab, K=5, T=T, oversample=oversample)
    assert len(ds) > 1000
    ds.enable_batched(host_major=host_major, spill_dir=str(tmp_path))
    _check_batches(ds)


def _old_index(store, K, T, oversample):
    """LazyCWADataset.__init__'s index, with the original per-start loop."""
    benign_cat = store.categories.index("Benign") if "Benign" in store.categories else None
    first_i = 1 if T > 0 else 0
    host_idx, pos = [], []
    hi = 0
    for h in store:
        rows = store._rows_by_host[h]
        n = len(rows)
        if n < K or n < K + first_i:
            continue
        starts = np.arange(first_i, n - K + 1, dtype=np.int32)
        cats = store.cat_id[rows]
        active = np.array([bool((cats[i:i + K] != benign_cat).any()) for i in starts],
                          dtype=bool) if benign_cat is not None else np.zeros(len(starts), bool)
        sel = np.concatenate([starts, starts[active]]) if oversample else starts
        host_idx.append(np.full(len(sel), hi, dtype=np.int32))
        pos.append(sel)
        hi += 1
    return np.concatenate(host_idx), np.concatenate(pos)


def _old_histogram(ds):
    st = ds.store
    n_rows = int(len(st.cat_id))
    first_tech = np.where(
        st.tech_off[1:n_rows + 1] > st.tech_off[:n_rows],
        st.tech_flat[np.minimum(st.tech_off[:n_rows], max(len(st.tech_flat) - 1, 0))],
        -1).astype(np.int64)
    cats = np.asarray(st.cat_id[:n_rows], dtype=np.int64)
    key = cats * (len(st.techniques) + 1) + (first_tech + 1)
    uniq, inv = np.unique(key, return_inverse=True)
    lut = np.empty(len(uniq), dtype=np.int64)
    for j, k in enumerate(uniq):
        c = int(k) // (len(st.techniques) + 1)
        t = int(k) % (len(st.techniques) + 1) - 1
        lut[j] = ds.vocab.encode(st.categories[c], st.techniques[t] if t >= 0 else "None")
    row_tok = lut[inv]
    hist = np.zeros(ds.vocab.vocab_size, dtype=np.int64)
    for host_i, i in zip(ds._host_idx, ds._pos):
        rows = st._rows_by_host[ds.hosts[int(host_i)]]
        hist += np.bincount(row_tok[rows[int(i):int(i) + ds.K]], minlength=ds.vocab.vocab_size)
    return hist


@pytest.mark.parametrize("T,oversample", [(15, True), (0, True), (15, False)])
def test_deepop_index_and_histogram(store, T, oversample):
    vocab = get_joint_vocab(network_observable_only=True)
    ds = LazyCWADataset(store, vocab, K=5, T=T, oversample=oversample)
    h_ref, p_ref = _old_index(store, 5, T, oversample)
    assert np.array_equal(ds._host_idx, h_ref) and np.array_equal(ds._pos, p_ref)
    assert ds._pos.dtype == p_ref.dtype
    assert np.array_equal(ds.target_token_histogram(), _old_histogram(ds))
    # and the per-row token table is `_token` on every row
    tok = ds._row_tokens()
    for r in np.random.default_rng(0).integers(0, len(store.cat_id), 3000):
        assert int(tok[r]) == ds._token(int(r))


@pytest.mark.parametrize("workers", [0, 2])
def test_loaders_identical_order_and_rng(store, workers):
    vocab = get_joint_vocab(network_observable_only=True)
    for ds, bs in ((LazyHostRolloutDataset(store, T=15, K=5), 128),
                   (LazyCWADataset(store, vocab, K=5, T=15), 64)):
        ds.enable_batched()

        def run(new):
            torch.manual_seed(7)
            dl = (batched_loader(ds, bs, True, num_workers=workers) if new
                  else DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers))
            return [b for _, b in zip(range(60), dl)], torch.get_rng_state()

        ref, ref_rng = run(False)
        new, new_rng = run(True)
        assert torch.equal(ref_rng, new_rng)
        assert len(ref) == len(new)
        for a, b in zip(ref, new):
            _same(a, b)
