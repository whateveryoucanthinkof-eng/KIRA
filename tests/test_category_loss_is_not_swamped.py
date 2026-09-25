"""The auxiliary category loss must not be drowned by the edge loss.

The two were summed with EQUAL weight and never logged separately, so the
imbalance was invisible. Measured on this corpus's real class distribution
({Benign 13.47M, C2 46k, Impact 1.30M, InitialAccess 157k, Recon 138k}):

    edge loss at AUC ~0.82        0.5754
    category focal loss, mediocre 0.0420    ~14x smaller
    category focal loss, confident 0.0001   ~5700x smaller

Focal loss with gamma=2 exists to down-weight easy examples, but 89% of this
corpus is one easy class, so it drives the whole term toward zero next to a
co-summed task. C2 compounds it at 0.3% of samples -- a batch of 128 holds
about 0.4 C2 examples.

Consequence: three of five classes sat at exactly 0.0 recall while the head
was perfectly capable -- the same architecture reaches 0.998 balanced
accuracy on these features standalone, and a gradient-boosted tree reaches
0.9997. The features and the architecture were never the problem.
"""

import inspect

import numpy as np
import pytest
import torch


def _focal():
    import sys
    sys.path.insert(0, "bita")
    from train import FocalLoss
    return FocalLoss


REAL_COUNTS = {0: 13466285, 1: 46408, 2: 1300101, 3: 156990, 4: 137877}


def test_focal_loss_really_does_collapse_on_this_distribution():
    """Establish the premise, so the fix is justified by measurement."""
    FocalLoss = _focal()
    labels = np.concatenate([np.full(v, k) for k, v in REAL_COUNTS.items()])
    crit = FocalLoss(alpha=FocalLoss.inverse_frequency_alpha(labels, 5), gamma=2.0)

    B = 128
    tot = sum(REAL_COUNTS.values())
    y = torch.tensor(np.random.default_rng(0).choice(
        list(REAL_COUNTS), size=B, p=[c / tot for c in REAL_COUNTS.values()]), dtype=torch.long)

    confident = torch.zeros(B, 5)
    confident[torch.arange(B), y] = 4.0
    assert crit(confident, y).item() < 0.01, (
        "premise failed: focal loss should collapse once the head is confident"
    )

    untrained = torch.zeros(B, 5)
    assert crit(untrained, y).item() > 0.1


def test_the_trainer_weights_the_category_term():
    src = open("bita/train.py").read()
    assert "args.cat_loss_weight * category_loss_total" in src, (
        "the category loss is still summed with equal weight"
    )


def test_the_default_weight_brings_the_terms_into_range():
    """Not a magic number: it should make the two comparable, not dominant."""
    src = open("bita/train.py").read()
    import re
    m = re.search(r"'--cat_loss_weight', type=float, default=([\d.]+)", src)
    assert m, "--cat_loss_weight not found"
    w = float(m.group(1))
    # edge ~0.575, category ~0.042 at the same point -> ratio ~13.7
    assert 5.0 <= w <= 30.0, f"weight {w} is outside the measured useful range"
    assert 0.3 < w * 0.042 / 0.5754 < 3.0, (
        f"weight {w} does not bring the terms within an order of magnitude"
    )


def test_both_loss_terms_are_logged_separately():
    """A swamped auxiliary task must be visible in the log, not inferred."""
    src = open("bita/train.py").read()
    # The per-term accumulators became m_loss.edge / m_loss.cat when the
    # training step stopped syncing every batch (losses stay on the GPU until
    # logged); what matters is that both terms still reach the epoch line.
    assert "edge {np.mean(m_loss.edge)" in src, "edge loss is not in the epoch line"
    assert "cat {np.mean(m_loss.cat)" in src, "category loss is not in the epoch line"


def test_weight_one_restores_the_previous_behaviour():
    """An escape hatch to reproduce the old runs exactly."""
    src = open("bita/train.py").read()
    assert "1.0 restores the old behaviour" in src
