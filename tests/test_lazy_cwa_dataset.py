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


def _burst_store(n_hosts=6, n_win=25, seed=0):
    """One contiguous attack burst on half the hosts -- bursty like the corpus,
    unlike `_store`'s every-fourth-window pattern."""
    rng = np.random.default_rng(seed)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(n_hosts):
        for w in range(n_win):
            attack = (h % 2 == 0) and (8 <= w < 14)
            b.append(host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                     window_start=float(w * 2), window_end=float(w * 2 + 2),
                     embedding=rng.random(12).astype(np.float32),
                     temporal_attrs=rng.random(15).astype(np.float32),
                     is_attack=attack,
                     coarse_category="C2" if attack else "Benign",
                     technique_ids=["T1071"] if attack else [],
                     risk_score=0.8 if attack else 0.0)
    return b.finalize()


@pytest.mark.parametrize("T_", [0, T])
@pytest.mark.parametrize("oversample", [True, False])
def test_same_number_of_samples_including_oversampling(T_, oversample):
    st = _store()
    v = get_joint_vocab()
    eager = create_cwa_training_samples(st, v, K=K, T=T_, oversample=oversample)
    lazy = LazyCWADataset(st, v, K=K, T=T_, oversample=oversample)
    assert len(lazy) == len(eager) > 0


def test_oversampled_eager_samples_are_not_the_same_object():
    """The duplicate used to be the SAME dict appended twice, so an in-place
    edit of one sample silently edited its twin."""
    st = _store()
    eager = create_cwa_training_samples(st, get_joint_vocab(), K=K, T=T)
    assert all(eager[i] is not eager[j]
               for i in range(len(eager)) for j in range(i + 1, min(i + 3, len(eager))))


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
    assert s["h_future"].shape == (K, 27)
    assert s["input_tokens"].shape == (K,)
    assert s["target_tokens"].shape == (K,)
    assert s["h_history"].shape == (T, 27)


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


def test_history_is_left_padded_with_a_real_past_state():
    """i == 1 has exactly one real past state; the pad repeats THAT, not a target."""
    st = _store(n_hosts=1, n_win=8)
    ds = LazyCWADataset(st, get_joint_vocab(), K=K, T=T)
    first = ds[0]                       # i = 1 -> one real history row
    assert first["h_history"].shape == (T, 27)
    assert torch.allclose(first["h_history"][0], first["h_history"][-1])
    assert torch.allclose(first["h_history"][-1],
                          torch.from_numpy(st.feats[st._rows_by_host["10.0.0.0"][0]]))


def test_no_window_takes_its_history_from_its_own_target():
    """The leak this guards.

    `LazyCWADataset` used to emit a window at i == 0, whose h_history was
    `np.repeat(h_fut[:1], T, axis=0)` -- T copies of the FIRST TARGET's own
    latent. `train_deepop_live` rolls Branch B forward from h_history, so for
    those samples the decoder was conditioned on the oracle latent of the very
    step it was asked to predict. From the retrain's own aggregates that is
    bounded at roughly 0.5-4.5% of the live validation set: small, but it
    leaks in the direction that flatters the model, so it cannot stay.

    Windows with no history at all are dropped when T > 0 -- they cannot be
    forecast from history by construction, so they are not data being thrown
    away, they are samples that could only ever be fabricated.
    """
    st = _store(n_hosts=3, n_win=12)
    v = get_joint_vocab()
    ds = LazyCWADataset(st, v, K=K, T=T)
    assert (ds._pos >= 1).all(), "a window with no history survived"
    assert ds.n_windows_dropped_no_history == 3, "one per host"
    for i in range(len(ds)):
        s = ds[i]
        for hrow in s["h_history"]:
            for frow in s["h_future"]:
                assert not torch.allclose(hrow, frow, atol=1e-6), (
                    "a history row equals a target row's latent")


def test_windows_with_no_history_are_kept_when_nothing_reads_history():
    """T == 0 means h_history is never built or consumed, so i == 0 is a
    perfectly ordinary window and dropping it WOULD be losing data."""
    st = _store(n_hosts=3, n_win=12)
    v = get_joint_vocab()
    with_hist = LazyCWADataset(st, v, K=K, T=T)
    no_hist = LazyCWADataset(st, v, K=K, T=0)
    assert len(no_hist) > len(with_hist)
    assert (no_hist._pos == 0).any()
    assert no_hist.n_windows_dropped_no_history == 0


def test_oversampling_can_be_turned_off_for_the_eval_split():
    """The 2x duplication of attack windows is a training device. Applied to
    the validation split it moves the class prior every metric is read at --
    accuracy, macro-F1, the persistence baseline and the majority baseline all
    shift with it. `scripts/retrain_future_models_live.py` builds train and val
    with the same call, so every DeepOP validation number logged so far was
    computed at an inflated attack rate.
    """
    # `_store`'s default puts an attack in every 4th window, so with K=3
    # almost every window is 'active' and the duplication is nearly uniform.
    # Use a bursty store, which is what the real corpus looks like
    # (val_label_churn 0.0524: a host's label rarely changes).
    st = _burst_store()
    v = get_joint_vocab()
    over = LazyCWADataset(st, v, K=K, T=T, oversample=True)
    plain = LazyCWADataset(st, v, K=K, T=T, oversample=False)
    assert len(over) > len(plain)

    h_over, h_plain = over.target_token_histogram(), plain.target_token_histogram()
    ben = v.encode("Benign", None)
    share_over = h_over[ben] / h_over.sum()
    share_plain = h_plain[ben] / h_plain.sum()
    assert share_plain > share_over + 0.05, (
        f"oversampling must visibly move the prior: {share_plain:.4f} -> {share_over:.4f}")


def test_target_token_histogram_matches_a_brute_force_count():
    """It is the cheap way to say which of the 10 joint tokens can occur.

    Measured on the shipped corpus from the categories the retrain logged:
    4 of 10 tokens occur in train (Benign.None, C2.T1071, Impact.T1498,
    InitialAccess.T1190) and 3 of 10 in val. Cross-entropy is normalised over
    all 10 regardless, and label smoothing at eps=0.04 assigns 6*eps/V = 2.4%
    of every target's mass to tokens that cannot occur.
    """
    v = get_joint_vocab()
    for T_ in (0, T):
        for ov in (True, False):
            ds = LazyCWADataset(_store(), v, K=K, T=T_, oversample=ov)
            ref = np.zeros(v.vocab_size, dtype=np.int64)
            for i in range(len(ds)):
                for t in ds[i]["target_tokens"].tolist():
                    ref[t] += 1
            assert np.array_equal(ds.target_token_histogram(), ref), (T_, ov)


def test_obs_token_is_strictly_the_past():
    """`obs_token` feeds the persistence baseline and the step-0 continuity
    bonus. If it were ever a target the baseline would be scoring against the
    answer. It is `rows[i-1]`, the row immediately before the first target row.
    """
    st = _store()
    v = get_joint_vocab()
    ds = LazyCWADataset(st, v, K=K, T=T)
    for i in range(len(ds)):
        pos = int(ds._pos[i])
        rows = st._rows_by_host[ds.hosts[int(ds._host_idx[i])]]
        assert pos >= 1
        assert int(rows[pos - 1]) < int(rows[pos]), "obs row is not before the window"
        assert int(ds[i]["obs_token"]) == ds._token(int(rows[pos - 1]))


def test_hosts_shorter_than_K_are_excluded():
    st = _store(n_hosts=2, n_win=K - 1)
    assert len(LazyCWADataset(st, get_joint_vocab(), K=K, T=T)) == 0


def test_it_works_through_a_dataloader():
    from torch.utils.data import DataLoader
    ds = LazyCWADataset(_store(), get_joint_vocab(), K=K, T=T)
    n = 0
    for b in DataLoader(ds, batch_size=8, shuffle=True):
        assert b["h_future"].shape[1:] == (K, 27)
        assert b["h_history"].shape[1:] == (T, 27)
        n += len(b["h_future"])
    assert n == len(ds)
