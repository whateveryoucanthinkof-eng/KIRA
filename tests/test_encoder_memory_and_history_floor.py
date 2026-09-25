"""TGN's memory module works and its cost is stated; samples are not mostly padding.

Two separate fixes, both about training on what is actually there.

**Memory.** The shipped encoder runs with `use_memory: False`. TGN's per-node
memory is the mechanism by which a host accumulates state across windows --
the premise of a host-trajectory model -- and it was switched off. It is not a
free switch: memory makes batch N depend on batch N-1, so it forbids batch
shuffling, and shuffling exists because 58.6% of time-ordered batches hold a
single class. The error has to say that, or someone flips the flag and quietly
gets a worse encoder.

**Padding.** The shipped Branch A checkpoint records `val_traj_len_median = 1.0`
over 2,172,277 hosts: the MEDIAN host appears in exactly one window. With
`seq_len=15` that made the median training example fourteen-fifteenths zeros.
"""

import inspect

import numpy as np
import pytest
import torch

from branch_a_gnn_lstm.sequence_dataset import (
    LazyHostSequenceDataset,
    create_host_sequence_samples,
    format_history_report,
)
from data_unification.multi_dataset_stream import HostWindowSnapshot

SEQ = 15


# ---------------------------------------------------------------------------
# TGN memory
# ---------------------------------------------------------------------------


def _tgn(use_memory):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bita"))
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder

    n_nodes, n_edges = 40, 200
    rng = np.random.default_rng(0)
    src = rng.integers(1, n_nodes, n_edges)
    dst = rng.integers(1, n_nodes, n_edges)
    ts = np.sort(rng.uniform(0, 400, n_edges))
    adj = [[] for _ in range(n_nodes)]
    for i, (u, v, t) in enumerate(zip(src, dst, ts)):
        adj[u].append((v, i, t)); adj[v].append((u, i, t))
    return ExtendedTGN(
        neighbor_finder=NeighborFinder(adj, uniform=False),
        node_features=rng.random((n_nodes, 12)).astype(np.float32),
        edge_features=rng.random((n_edges, 12)).astype(np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.1,
        use_memory=use_memory, message_dimension=100, memory_dimension=12,
        embedding_module_type="graph_attention", message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru",
        num_categories=5,
    )


def test_the_memory_path_actually_runs():
    """Before recommending it, check it is not bit-rotted."""
    tgn = _tgn(True)
    emb = tgn.get_host_embeddings(np.array([1, 2, 3]), timestamp=300.0, n_neighbors=10)
    assert emb.shape == (3, 12)
    assert torch.isfinite(emb).all()


def test_memory_adds_real_capacity():
    """If it added nothing, turning it on would not be worth a retrain."""
    off = sum(p.numel() for p in _tgn(False).parameters())
    on = sum(p.numel() for p in _tgn(True).parameters())
    assert on > 3 * off, f"memory added only {on - off} parameters ({off} -> {on})"


def test_the_latent_width_is_unchanged_by_memory():
    """Memory must not silently change the 12-D contract every model downstream
    is built on."""
    for use_mem in (False, True):
        emb = _tgn(use_mem).get_host_embeddings(np.array([1]), timestamp=300.0)
        assert emb.shape[1] == 12


def test_turning_memory_on_states_what_it_costs():
    """Flipping the flag without raising --backprop_every gives a worse encoder,
    so the error must name the mitigation, not just the prohibition."""
    import bita.train as T

    src = inspect.getsource(T)
    i = src.index("--shuffle_batches cannot be combined")
    msg = src[i:i + 2000]
    assert "backprop_every" in msg, "the error forbids the combination but offers no way forward"
    assert "58.6" in msg, "the error does not say why shuffling exists"


def test_a_small_backprop_every_with_memory_is_warned_about():
    import bita.train as T

    src = inspect.getsource(T)
    assert "args.use_memory and args.backprop_every < 4" in src


# ---------------------------------------------------------------------------
# History floor
# ---------------------------------------------------------------------------


def _snap(i):
    # Non-zero features on purpose. With zeros everywhere a padded step and a
    # real step are indistinguishable -- which is exactly the ambiguity padding
    # introduces, and it would make the assertions below vacuous.
    return HostWindowSnapshot("h", 1, i, i * 2.0, i * 2.0 + 2.0,
                              np.full(12, i + 1, np.float32),
                              np.full(15, 0.5, np.float32),
                              False, "Benign", [], 0.0)


def test_a_one_window_host_yields_nothing_by_default():
    """The median host on this corpus. It used to yield a 14/15-padding sample."""
    traj = {"h": [_snap(i) for i in range(2)]}
    assert create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1) == []


def test_padding_is_still_available_when_asked_for():
    traj = {"h": [_snap(i) for i in range(2)]}
    got = create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1,
                                       min_history_steps=1)
    assert len(got) == 1
    assert np.count_nonzero(got[0]["features"][:SEQ - 1]) == 0


def test_every_default_sample_is_fully_observed():
    traj = {"h": [_snap(i) for i in range(SEQ + 5)]}
    got = create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1)
    assert len(got) == 5
    for s in got:
        assert np.count_nonzero(s["features"].sum(axis=1)) == SEQ, "a step is all zeros"


def test_the_drop_is_reported_rather_than_silent():
    """Dropping most of a corpus silently is as bad as padding it silently."""
    traj = {"short": [_snap(i) for i in range(3)],
            "long": [_snap(i) for i in range(SEQ + 5)]}
    rep = {}
    create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1, report=rep)
    assert rep["kept"] == 5
    assert rep["dropped_short_history"] > 0
    text = format_history_report(rep)
    assert "kept 5 samples" in text and "dropped" in text


def test_a_mostly_dropped_split_warns_loudly():
    traj = {f"h{i}": [_snap(j) for j in range(3)] for i in range(20)}
    traj["long"] = [_snap(j) for j in range(SEQ + 2)]
    rep = {}
    create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1, report=rep)
    assert "WARNING" in format_history_report(rep)


def test_the_two_sample_builders_agree():
    """LazyHostSequenceDataset and create_host_sequence_samples must not
    disagree about what counts as a sample -- one is used at scale, the other
    in tests, and a divergence would be invisible."""
    from data_unification.trajectory_store import TrajectoryStoreBuilder

    b = TrajectoryStoreBuilder()
    for i in range(SEQ + 6):
        b.append(host_ip="h", host_id=1, window_idx=i, window_start=i * 2.0,
                 window_end=i * 2.0 + 2.0, embedding=np.full(12, i + 1, np.float32),
                 temporal_attrs=np.full(15, 0.5, np.float32), is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    store = b.finalize()
    traj = {h: list(store[h]) for h in store}
    eager = create_host_sequence_samples(traj, seq_len=SEQ, min_trajectory_len=1)
    lazy = LazyHostSequenceDataset(store, seq_len=SEQ, min_trajectory_len=1)
    assert len(eager) == len(lazy) == 6
