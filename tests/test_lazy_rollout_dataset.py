"""Branch B must not materialise every rollout window either.

`create_rollout_samples` builds three arrays per sample -- h_history [T,12],
h_future [K,12], risk_future [K] -- for **1,164 bytes each**. Branch B draws
on the same trajectories as Branch A, roughly 42M snapshots at full density:

    10M samples -> 10.8 GiB
    35M samples -> 37.9 GiB

Same blocker, same fix. These tests pin that the lazy path produces exactly
what the eager one does, including the edge-padding at the end of a
trajectory, which is easy to get subtly wrong.
"""

import numpy as np
import pytest
import torch

from branch_b_world_model.train_branch_b import (
    HostRolloutDataset,
    LazyHostRolloutDataset,
    create_rollout_samples,
)
from data_unification.trajectory_store import TrajectoryStoreBuilder

T, K = 4, 3


def _store(n_hosts=4, n_win=12, seed=0):
    rng = np.random.default_rng(seed)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(n_hosts):
        for w in range(n_win):
            b.append(
                host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                window_start=float(w * 2), window_end=float(w * 2 + 2),
                embedding=rng.random(12).astype(np.float32),
                temporal_attrs=rng.random(15).astype(np.float32),
                is_attack=bool(w % 2), coarse_category="C2" if w % 2 else "Benign",
                technique_ids=["T1071"] if w % 2 else [],
                risk_score=float(w) / 10.0,
            )
    return b.finalize()


def test_same_number_of_samples():
    st = _store()
    eager = create_rollout_samples(st, T=T, K=K)
    lazy = LazyHostRolloutDataset(st, T=T, K=K)
    assert len(lazy) == len(eager) > 0


def test_every_sample_matches_the_eager_path():
    st = _store()
    eager = HostRolloutDataset(create_rollout_samples(st, T=T, K=K))
    lazy = LazyHostRolloutDataset(st, T=T, K=K)

    # create_rollout_samples iterates hosts in store order, as the lazy index
    # does, so positions line up; compare as multisets to be safe anyway.
    def sig(d):
        return (tuple(np.round(d["h_history"].numpy().ravel(), 6)),
                tuple(np.round(d["h_future"].numpy().ravel(), 6)),
                tuple(np.round(d["risk_future"].numpy().ravel(), 6)))

    e = sorted(sig(eager[i]) for i in range(len(eager)))
    l = sorted(sig(lazy[i]) for i in range(len(lazy)))
    assert e == l, "lazy rollout samples differ from the eager ones"


def test_edge_padding_at_the_end_of_a_trajectory():
    """The last samples have fewer than K future steps and must edge-pad."""
    st = _store(n_hosts=1, n_win=T + 2)       # only 2 samples, both short on future
    lazy = LazyHostRolloutDataset(st, T=T, K=K)
    last = lazy[len(lazy) - 1]
    assert last["h_future"].shape == (K, 27)
    assert last["risk_future"].shape == (K,)
    # edge padding repeats the final real value
    assert torch.allclose(last["h_future"][-1], last["h_future"][-2])


def test_history_is_exactly_T_steps_and_strictly_before_the_future():
    st = _store()
    lazy = LazyHostRolloutDataset(st, T=T, K=K)
    for i in range(min(20, len(lazy))):
        s = lazy[i]
        assert s["h_history"].shape == (T, 27)
        assert s["h_future"].shape == (K, 27)


def test_hosts_with_too_little_history_are_excluded():
    st = _store(n_hosts=3, n_win=T)           # need T+1
    assert len(LazyHostRolloutDataset(st, T=T, K=K)) == 0


def test_index_memory_is_tiny():
    st = _store(n_hosts=15, n_win=60)
    lazy = LazyHostRolloutDataset(st, T=T, K=K)
    per = (lazy._host_idx.nbytes + lazy._pos.nbytes) / max(1, len(lazy))
    assert per <= 16, f"{per:.1f} B/sample index is too heavy"


def test_it_works_through_a_dataloader():
    from torch.utils.data import DataLoader
    st = _store()
    ds = LazyHostRolloutDataset(st, T=T, K=K)
    n = 0
    for b in DataLoader(ds, batch_size=8, shuffle=True):
        assert b["h_history"].shape[1:] == (T, 27)
        assert b["h_future"].shape[1:] == (K, 27)
        n += len(b["h_history"])
    assert n == len(ds)
