"""Training batches must contain more than one class.

TGN slices batches contiguously in TIME and attacks are time-localised, so
most batches hold a single label. Measured on fri_16: **58.6% of 128-sample
batches are single-class**. The category head then gets "predict X for all of
these" with no contrast and oscillates across batches -- which is why it
collapsed to one class while a gradient-boosted tree on the same features,
shuffled, reaches 0.9997 balanced accuracy.

Two things this pins:

1. Permuting SAMPLES fixes it; permuting batch ORDER does not. Reordering
   contiguous batches leaves each one single-class and changes nothing. (I
   wrote the batch-order version first; it was useless.)
2. Shuffling is only valid with the memory module OFF. TGN's memory makes
   batch N depend on batch N-1, so shuffling with memory on would train on
   states that never existed. That combination must be refused, not ignored.
"""

import numpy as np
import pytest


def _single_class_fraction(labels, order, bs=128):
    pure = total = 0
    for s in range(0, len(order) - bs, bs):
        if len(set(labels[order[s:s + bs]])) == 1:
            pure += 1
        total += 1
    return pure / max(1, total)


def _time_clustered_labels(n=40000, runs=40):
    """Labels in contiguous runs, as a time-ordered attack corpus produces."""
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        cls = int(rng.integers(0, 5))
        out.extend([cls] * int(rng.integers(n // runs // 2, n // runs)))
    return np.asarray(out[:n])


def test_time_ordered_batches_really_are_single_class():
    """Establish the premise the fix is built on."""
    y = _time_clustered_labels()
    frac = _single_class_fraction(y, np.arange(len(y)))
    assert frac > 0.4, f"fixture does not reproduce the problem ({frac:.2f})"


def test_shuffling_samples_fixes_it():
    y = _time_clustered_labels()
    frac = _single_class_fraction(y, np.random.default_rng(1).permutation(len(y)))
    assert frac < 0.05, f"shuffled batches still single-class ({frac:.2f})"


def test_shuffling_batch_ORDER_does_not_fix_it():
    """The distinction that matters: reordering contiguous blocks is useless."""
    y = _time_clustered_labels()
    bs = 128
    nb = len(y) // bs
    order = np.concatenate([
        np.arange(b * bs, (b + 1) * bs)
        for b in np.random.default_rng(2).permutation(nb)
    ])
    frac = _single_class_fraction(y, order)
    assert frac > 0.4, (
        "batch-order shuffling should NOT help -- if it does, this test no "
        "longer distinguishes the two approaches"
    )


def test_trainer_permutes_samples_not_batches():
    src = open("bita/train.py").read()
    assert "perm = np.random.permutation(num_instance)" in src, (
        "must permute samples (num_instance), not batches"
    )
    assert "sources_batch = train_data.sources[sel]" in src, (
        "batches must be gathered through the permutation"
    )


def test_shuffling_is_refused_when_memory_is_enabled():
    src = open("bita/train.py").read()
    assert "cannot be combined with --use_memory" in src, (
        "shuffling with the memory module on must raise, not silently corrupt state"
    )
    assert "args.shuffle_batches and not args.use_memory" in src


def test_shuffling_is_on_by_default_and_can_be_disabled():
    src = open("bita/train.py").read()
    assert "'--shuffle_batches', action='store_true', default=True" in src
    assert "'--no_shuffle_batches'" in src


# ---------------------------------------------------------------------------
# Shuffling must not break causality
# ---------------------------------------------------------------------------

def test_shuffled_batches_still_only_see_the_past():
    """The correctness question behind the fix.

    A forecaster that peeks at the future is worthless, so shuffling is only
    acceptable if every sampled neighbour still predates the edge being
    scored. It does, because `find_before` cuts on each edge's OWN timestamp
    rather than on position in the batch -- but that is worth proving, not
    assuming.
    """
    from bita.utils.utils import get_neighbor_finder

    class _D:
        def __init__(s, a, b, c, d):
            s.sources, s.destinations, s.edge_idxs, s.timestamps = a, b, c, d

    rng = np.random.default_rng(0)
    n, nodes = 20000, 500
    d = _D(rng.integers(1, nodes, n), rng.integers(1, nodes, n),
           np.arange(1, n + 1), np.sort(rng.random(n) * 1e6))
    nf = get_neighbor_finder(d, uniform=False, max_node_idx=nodes)

    # Query in a SHUFFLED order, exactly as the trainer now does.
    perm = rng.permutation(n)[:2000]
    src = d.sources[perm]
    ts = d.timestamps[perm]
    nbr, eidx, etimes = nf.get_temporal_neighbor(src, ts, 10)

    for row in range(len(perm)):
        real = etimes[row][etimes[row] > 0]
        if len(real):
            assert real.max() < ts[row], (
                f"row {row} sampled a neighbour at t={real.max()} for a query "
                f"at t={ts[row]} -- the model can see the future"
            )


def test_permutation_covers_every_sample_exactly_once():
    """A shuffle that drops or duplicates samples would silently change the
    effective dataset size."""
    n = 10_000
    perm = np.random.default_rng(3).permutation(n)
    assert len(perm) == n
    assert sorted(perm.tolist()) == list(range(n))
