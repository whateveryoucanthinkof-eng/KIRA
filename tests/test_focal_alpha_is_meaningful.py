"""Per-class focal weights must actually differentiate the classes.

This function has now failed twice, in opposite directions:

1. Plain 1/frequency: absent classes got astronomical raw weights, and after
   normalising every present class collapsed to ~0. Measured
   {Benign: 0.0, C2: 0.0, Impact: 2.0, ...} -- the majority classes
   contributed no loss at all.
2. Arithmetic-mean normalisation: one ultra-rare class dominates the mean and
   crushes everyone else onto the clip floor. Measured on the full corpus,
   {Benign: 0.2, C2: 0.2, Impact: 0.2, InitialAccess: 0.2, Recon: 4.911} --
   four of five classes weighted IDENTICALLY, i.e. no balancing at all among
   the four carrying the data.

These are multiplicative weights, so the geometric mean is the correct centre.
"""

import numpy as np
import pytest
import torch

from bita.train import FocalLoss

alpha = FocalLoss.inverse_frequency_alpha


def _labels(counts: dict):
    """counts maps class index -> number of samples."""
    return np.concatenate([np.full(n, c, dtype=np.int64) for c, n in counts.items()])


def test_the_real_corpus_distribution_yields_distinct_weights():
    """The regression: Recon ~5 orders of magnitude rarer than Benign."""
    y = _labels({0: 9_000_000, 1: 1_500_000, 2: 1_000_000, 3: 300_000, 4: 120})
    w = alpha(y, 5).numpy()
    major = w[:4]
    assert len(set(np.round(major, 4))) == 4, (
        f"the four major classes must be weighted differently, got {major}"
    )


def test_rarer_classes_get_larger_weights():
    y = _labels({0: 100_000, 1: 10_000, 2: 1_000, 3: 100})
    w = alpha(y, 4).numpy()
    assert w[0] < w[1] < w[2] < w[3], f"not monotone in rarity: {w}"


def test_weights_stay_inside_the_clip():
    y = _labels({0: 10_000_000, 1: 5})
    w = alpha(y, 2).numpy()
    assert w.min() >= 0.2 - 1e-9 and w.max() <= 5.0 + 1e-9


def test_a_balanced_corpus_gives_near_equal_weights():
    y = _labels({0: 1000, 1: 1000, 2: 1000})
    w = alpha(y, 3).numpy()
    np.testing.assert_allclose(w, np.ones(3), rtol=1e-6)


def test_an_absent_class_keeps_weight_one_and_does_not_skew_the_rest():
    """Weight exactly 1.0 for an absent class is the diagnostic signal that
    found the corpus-split bug -- it must be preserved, and it must not
    contaminate the present classes' normalisation."""
    y = _labels({0: 10_000, 1: 1_000})          # classes 2,3 absent
    w = alpha(y, 4).numpy()
    assert w[2] == pytest.approx(1.0) and w[3] == pytest.approx(1.0)
    assert w[0] != pytest.approx(1.0) or w[1] != pytest.approx(1.0)


def test_geometric_centring_survives_one_extreme_outlier():
    """With the arithmetic mean the three common classes all hit the floor."""
    y = _labels({0: 5_000_000, 1: 2_000_000, 2: 500_000, 3: 10})
    w = alpha(y, 4).numpy()
    assert len(set(np.round(w[:3], 4))) == 3, f"common classes collapsed: {w[:3]}"


def test_weights_are_a_tensor_of_the_right_shape():
    w = alpha(_labels({0: 10, 1: 10}), 5)
    assert isinstance(w, torch.Tensor) and w.shape == (5,) and w.dtype == torch.float


def test_single_class_data_does_not_divide_by_zero():
    w = alpha(_labels({0: 500}), 3).numpy()
    assert np.all(np.isfinite(w))


def test_the_loss_actually_applies_the_weights():
    """A weight vector that never reaches the loss is decoration."""
    logits = torch.zeros(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    flat = FocalLoss(alpha=torch.ones(3), gamma=2.0)(logits, targets)
    skewed = FocalLoss(alpha=torch.tensor([5.0, 1.0, 1.0]), gamma=2.0)(logits, targets)
    assert skewed > flat, "up-weighting class 0 must raise the loss on class-0 samples"
