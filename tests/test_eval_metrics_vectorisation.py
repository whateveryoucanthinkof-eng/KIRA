"""The vectorised Hits@K / MRR must equal the loops they replaced.

The original implementations were O(N x P x C) and stalled an epoch's
evaluation for many minutes on a million-interaction set. They are replaced by
exact vectorised equivalents; these tests pin the equivalence so the speedup
can never quietly become a behaviour change.
"""

import numpy as np
import pytest


# ---- the ORIGINAL implementations, kept verbatim as the reference ----------

def _reference_hits_and_mrr(y_true, y_scores):
    hits_at_k = lambda k: np.mean([
        y_true[i] in np.argsort(-y_scores[i])[:k] for i in range(len(y_true))
    ])
    ranks = np.argsort(-y_scores, axis=1)
    correct_ranks = np.array(
        [np.where(ranks[i] == y_true[i])[0][0] + 1 for i in range(len(y_true))]
    )
    return hits_at_k(1), hits_at_k(3), hits_at_k(5), np.mean(1 / correct_ranks)


def _reference_class_mrr(y_true_bin, y_score_class):
    ranks_c = np.argsort(-y_score_class)
    true_indices = np.where(y_true_bin == 1)[0]
    if len(true_indices) == 0:
        return 0.0
    return float(np.mean([
        1 / (np.where(ranks_c == idx)[0][0] + 1) for idx in true_indices
    ]))


# ---- the NEW implementations, mirroring the shipped code ------------------

def _fast_hits_and_mrr(y_true, y_scores):
    row = np.arange(len(y_true))
    true_class_score = y_scores[row, y_true]
    correct_ranks = 1 + (y_scores > true_class_score[:, None]).sum(axis=1)
    return (
        float(np.mean(correct_ranks <= 1)),
        float(np.mean(correct_ranks <= 3)),
        float(np.mean(correct_ranks <= 5)),
        float(np.mean(1.0 / correct_ranks)),
    )


def _fast_class_mrr(y_true_bin, y_score_class):
    true_indices = np.where(y_true_bin == 1)[0]
    if len(true_indices) == 0:
        return 0.0
    order = np.argsort(-y_score_class)
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order))
    return float(np.mean(1.0 / (position[true_indices] + 1)))


# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_hits_and_mrr_match_the_reference(seed):
    rng = np.random.default_rng(seed)
    n, c = 500, 6
    logits = rng.normal(size=(n, c))
    y_scores = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y_true = rng.integers(0, c, size=n)

    ref = _reference_hits_and_mrr(y_true, y_scores)
    fast = _fast_hits_and_mrr(y_true, y_scores)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_per_class_mrr_matches_the_reference(seed):
    """This is the exact-permutation case -- it must be bit-identical."""
    rng = np.random.default_rng(seed)
    n = 800
    y_score_class = rng.random(n)
    y_true_bin = (rng.random(n) < 0.2).astype(int)

    assert _fast_class_mrr(y_true_bin, y_score_class) == pytest.approx(
        _reference_class_mrr(y_true_bin, y_score_class), rel=1e-12
    )


def test_per_class_mrr_with_no_positives_is_zero():
    assert _fast_class_mrr(np.zeros(50, dtype=int), np.random.random(50)) == 0.0


def test_per_class_mrr_when_every_sample_is_positive():
    n = 100
    got = _fast_class_mrr(np.ones(n, dtype=int), np.random.random(n))
    expected = float(np.mean(1.0 / np.arange(1, n + 1)))
    assert got == pytest.approx(expected)


def test_a_perfect_ranker_scores_mrr_one():
    n, c = 200, 5
    y_true = np.random.default_rng(0).integers(0, c, size=n)
    y_scores = np.full((n, c), 0.01)
    y_scores[np.arange(n), y_true] = 0.99
    _, _, _, mrr = _fast_hits_and_mrr(y_true, y_scores)
    assert mrr == pytest.approx(1.0)


def test_hits_at_k_is_monotone_in_k():
    rng = np.random.default_rng(3)
    n, c = 300, 6
    y_scores = rng.random((n, c))
    y_scores /= y_scores.sum(1, keepdims=True)
    y_true = rng.integers(0, c, size=n)
    h1, h3, h5, _ = _fast_hits_and_mrr(y_true, y_scores)
    assert h1 <= h3 <= h5


def test_ties_resolve_optimistically_and_are_documented():
    """All-equal scores rank the true class first. The old argsort version
    agreed only when the true class had the lowest index; this is the one
    deliberate behavioural difference, and it is unreachable with real softmax
    outputs."""
    y_scores = np.full((1, 4), 0.25)
    y_true = np.array([3])
    _, _, _, mrr = _fast_hits_and_mrr(y_true, y_scores)
    assert mrr == pytest.approx(1.0)
