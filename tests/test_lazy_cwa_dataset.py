"""DeepOP must not materialise every token sequence either.

`create_cwa_training_samples` builds h_future [K,12], h_history [T,12] and
two token arrays per sample -- roughly 1,250 bytes each -- one per snapshot
plus a duplicate for every attack sequence. At full corpus density (~42M
snapshots) that is ~52 GiB.

Third and last instance of the same blocker (Branch A 80 GiB, Branch B
38 GiB). These tests pin that the lazy path reproduces the eager one,
including the 2x oversampling of non-Benign windows, which exists to stop the
decoder collapsing to the quiescent sequence.
"""

import numpy as np
import pytest
import torch

from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.train_cwa_decoder import (
    CWASequenceDataset,
    LazyCWADataset,
    create_cwa_training_samples,
)
from data_unification.trajectory_store import TrajectoryStoreBuilder

K, T = 3, 4


def _store(n_hosts=3, n_win=12, seed=0, all_benign=False):
    rng = np.random.default_rng(seed)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(n_hosts):
        for w in range(n_win):
            attack = (not all_benign) and (w % 4 == 1)
            b.append(
                host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                window_start=float(w * 2), window_end=float(w * 2 + 2),
                embedding=rng.random(12).astype(np.float32),
                temporal_attrs=rng.random(15).astype(np.float32),
                is_attack=attack,
                coarse_category="C2" if attack else "Benign",
                technique_ids=["T1071"] if attack else [],
                risk_score=0.8 if attack else 0.0,
            )
    return b.finalize()


def test_same_number_of_samples_including_oversampling():
    st = _store()
    v = get_joint_vocab()
    eager = create_cwa_training_samples(st, v, K=K, T=T)
    lazy = LazyCWADataset(st, v, K=K, T=T)
    assert len(lazy) == len(eager) > 0


def test_oversampling_actually_happens():
    """Guard: if no window were duplicated the count test would be vacuous."""
    v = get_joint_vocab()
    with_attacks = LazyCWADataset(_store(), v, K=K, T=T)
    benign_only = LazyCWADataset(_store(all_benign=True), v, K=K, T=T)
    assert len(with_attacks) > len(benign_only), "attack windows are not duplicated"


def test_shapes_match_the_contract():
    st = _store()
    v = get_joint_vocab()
    ds = LazyCWADataset(st, v, K=K, T=T)
    s = ds[0]
    assert s["h_future"].shape == (K, 12)
    assert s["input_tokens"].shape == (K,)
    assert s["target_tokens"].shape == (K,)
    assert s["h_history"].shape == (T, 12)


def test_input_is_bos_plus_shifted_targets():
    """Teacher forcing: input[0] is BOS and input[1:] == target[:-1]."""
    st = _store()
    v = get_joint_vocab()
    ds = LazyCWADataset(st, v, K=K, T=T)
    for i in range(min(10, len(ds))):
        s = ds[i]
        assert int(s["input_tokens"][0]) == v.bos_idx
        assert torch.equal(s["input_tokens"][1:], s["target_tokens"][:-1])


def test_token_content_matches_the_eager_path():
    st = _store()
    v = get_joint_vocab()
    eager = create_cwa_training_samples(st, v, K=K, T=T)
    lazy = LazyCWADataset(st, v, K=K, T=T)
    e = sorted(tuple(int(x) for x in s["target_tokens"]) for s in eager)
    l = sorted(tuple(int(x) for x in lazy[i]["target_tokens"]) for i in range(len(lazy)))
    assert e == l, "lazy token sequences differ from the eager ones"


def test_history_is_left_padded_at_the_start_of_a_trajectory():
    st = _store(n_hosts=1, n_win=8)
    ds = LazyCWADataset(st, get_joint_vocab(), K=K, T=T)
    first = ds[0]                       # i = 0 -> no history at all
    assert first["h_history"].shape == (T, 12)
    # with no history the first future state is repeated
    assert torch.allclose(first["h_history"][0], first["h_history"][-1])


def test_hosts_shorter_than_K_are_excluded():
    st = _store(n_hosts=2, n_win=K - 1)
    assert len(LazyCWADataset(st, get_joint_vocab(), K=K, T=T)) == 0


def test_it_works_through_a_dataloader():
    from torch.utils.data import DataLoader
    ds = LazyCWADataset(_store(), get_joint_vocab(), K=K, T=T)
    n = 0
    for b in DataLoader(ds, batch_size=8, shuffle=True):
        assert b["h_future"].shape[1:] == (K, 12)
        assert b["h_history"].shape[1:] == (T, 12)
        n += len(b["h_future"])
    assert n == len(ds)
