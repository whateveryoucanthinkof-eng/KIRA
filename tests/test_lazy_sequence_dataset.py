"""Branch A must not materialise every training window.

`create_host_sequence_samples` builds a [seq_len, 27] float32 array per
sample: **2,053 bytes each**, 1,620 of it the window. At full corpus density
Branch A produces roughly 42M samples (measured snapshot ratios: 1.99 per
record for CIC-2018, 0.72 for CTU-13):

      8.5M samples ->  16.3 GiB
       42M samples ->  80.3 GiB

That does not fit, and thinning the data is not an option. Nothing needs those
arrays to exist -- a window is seq_len consecutive rows of a trajectory the
store already holds in one memmapped block.

LazyHostSequenceDataset keeps two int32 columns (~8 B/sample, 336 MB at 42M)
and gathers the window on __getitem__. These tests pin that it produces
EXACTLY what the eager path produces; a memory optimisation that changes the
data is not an optimisation.
"""

import numpy as np
import pytest
import torch

from branch_a_gnn_lstm.sequence_dataset import (
    HostSequenceDataset,
    LazyHostSequenceDataset,
    create_host_sequence_samples,
)
from data_unification.trajectory_store import TrajectoryStoreBuilder

SEQ = 5


def _store(n_hosts=5, n_win=14, seed=0):
    rng = np.random.default_rng(seed)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for h in range(n_hosts):
        for w in range(n_win):
            b.append(
                host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                window_start=float(w * 2), window_end=float(w * 2 + 2),
                embedding=rng.random(12).astype(np.float32),
                temporal_attrs=rng.random(15).astype(np.float32),
                is_attack=bool(w % 3 == 0),
                coarse_category=["Benign", "C2", "Impact"][w % 3],
                technique_ids=["T1046"] if w % 3 else [],
                risk_score=float(w) / 20.0,
            )
    return b.finalize()


def test_lazy_and_eager_have_the_same_length():
    st = _store()
    eager = HostSequenceDataset(create_host_sequence_samples(st, seq_len=SEQ,
                                                             min_trajectory_len=2),
                                seq_len=SEQ)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    assert len(lazy) == len(eager) > 0


def test_every_sample_matches_the_eager_path():
    """The whole point: identical data, less memory."""
    st = _store()
    eager_samples = create_host_sequence_samples(st, seq_len=SEQ, min_trajectory_len=2)
    eager = HostSequenceDataset(eager_samples, seq_len=SEQ)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)

    # order may differ; index both by (host, window_idx)
    def key(d):
        return (d["host_ip"], int(d["window_idx"]))

    e = {key(eager[i]): eager[i] for i in range(len(eager))}
    l = {key(lazy[i]): lazy[i] for i in range(len(lazy))}
    assert set(e) == set(l), "the two paths cover different (host, window) pairs"

    for k in e:
        torch.testing.assert_close(e[k]["features"], l[k]["features"])
        torch.testing.assert_close(e[k]["risk"], l[k]["risk"])
        assert e[k]["technique"] == l[k]["technique"], k
        assert e[k]["gradation"] == l[k]["gradation"], k


def test_windows_are_left_padded_when_padding_is_explicitly_requested():
    """Padding still works -- it is just no longer the default. `min_history_steps=1`
    restores the old behaviour for a caller who wants it."""
    st = _store(n_hosts=1, n_win=3)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2,
                                   min_history_steps=1)
    first = lazy[0]                       # end_idx = 1 -> only 1 real step
    assert first["features"].shape == (SEQ, 27)
    assert torch.count_nonzero(first["features"][:SEQ - 1]) == 0, "should be left-padded"


def test_the_default_refuses_a_sample_that_would_be_mostly_padding():
    """The shipped checkpoint records a MEDIAN host trajectory of one window,
    so under the old default the median training example was 14/15 zeros."""
    st = _store(n_hosts=1, n_win=3)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    assert len(lazy) == 0, (
        f"a 3-window host cannot fill a {SEQ}-step window; it should yield no samples"
    )


def test_every_sample_under_the_default_is_fully_observed():
    st = _store(n_hosts=1, n_win=SEQ + 6)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    assert len(lazy) == 6, f"expected n - seq_len samples, got {len(lazy)}"
    for i in range(len(lazy)):
        f = lazy[i]["features"]
        assert torch.count_nonzero(f.sum(dim=1)) == SEQ, "a step is all zeros (padding)"


def test_the_drop_is_reported_not_silent():
    st = _store(n_hosts=1, n_win=3)
    rep = {}
    LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2, report=rep)
    assert rep["kept"] == 0 and rep["dropped_short_history"] == 2
    assert rep["min_history_steps"] == SEQ


def test_lazy_and_eager_agree_on_the_history_floor():
    """The two sample builders must not disagree about what counts as a sample."""
    from branch_a_gnn_lstm.sequence_dataset import create_host_sequence_samples

    st = _store(n_hosts=1, n_win=SEQ + 4)
    traj = {h: list(st[h]) for h in st}
    eager = create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=2)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    assert len(eager) == len(lazy) == 4


def test_the_target_is_strictly_in_the_future():
    """The nowcast bug: the target must never be the last input step."""
    st = _store(n_hosts=1, n_win=10)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    rows = st._rows_by_host["10.0.0.0"]
    for i in range(len(lazy)):
        end = int(lazy._pos[i])
        target_w = int(lazy[i]["window_idx"])
        last_input_w = st._materialize(int(rows[end - 1])).window_idx
        assert target_w > last_input_w, (
            f"target window {target_w} is not after the last input {last_input_w}"
        )


def test_short_trajectories_are_excluded():
    st = _store(n_hosts=3, n_win=1)       # every host has 1 snapshot
    assert len(LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)) == 0


def test_index_memory_is_tiny_compared_to_materialising():
    """~8 bytes per sample instead of 2,053."""
    st = _store(n_hosts=20, n_win=40)
    lazy = LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2)
    idx_bytes = lazy._host_idx.nbytes + lazy._pos.nbytes
    per_sample = idx_bytes / max(1, len(lazy))
    assert per_sample <= 16, f"{per_sample:.1f} B/sample index is too heavy"


def test_it_works_through_a_dataloader():
    from torch.utils.data import DataLoader
    st = _store()
    dl = DataLoader(LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2),
                    batch_size=8, shuffle=True)
    seen = 0
    for batch in dl:
        assert batch["features"].shape[1:] == (SEQ, 27)
        seen += len(batch["features"])
    assert seen == len(LazyHostSequenceDataset(st, seq_len=SEQ, min_trajectory_len=2))
