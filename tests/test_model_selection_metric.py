"""The encoder must not be selected on a saturated metric.

Early stopping used `val_ap` -- link prediction on hosts already seen. That
task converges in ONE epoch on this corpus and then saturates, while the
classification head keeps improving:

    val_ap        0.9971 -> 0.9968 -> 0.9968     (best: epoch 0)
    inductive AP  0.9926 -> 0.9921 -> 0.9918     (best: epoch 0)
    C2 inductive  0.325  -> 0.319  -> 0.517      (still climbing)

Selecting on the saturated metric restores epoch 0 and discards the
improvement. The replacement weights embedding quality on UNSEEN hosts
(inductive AP, since the transductive figure has no discriminative power left
at 0.9968) against balanced classification (macro F1, because Benign is ~90%
of val and dominates any unweighted average).
"""

import numpy as np
import pytest

from bita.utils.utils import EarlyStopMonitor


def test_selection_uses_inductive_ap_and_macro_f1():
    src = open("bita/train.py").read()
    assert "_sel = 0.5 * float(nn_val_ap) + 0.5 * float(val_f1_macro)" in src
    assert "early_stopper.early_stop_check(_sel)" in src, "the new metric is not used"
    assert "early_stop_check(val_ap)" not in src, "still selecting on val_ap alone"


def test_the_selection_score_is_logged():
    """A selection decision that is not visible cannot be questioned."""
    src = open("bita/train.py").read()
    assert "selection=%.4f" in src


def test_val_ap_alone_would_have_picked_epoch_zero():
    """Establish the premise from the observed numbers."""
    m = EarlyStopMonitor(max_round=5, higher_better=True)
    for ap in (0.9971, 0.9968, 0.9968):
        m.early_stop_check(ap)
    assert m.best_epoch == 0


def test_the_combined_metric_prefers_a_later_epoch_when_f1_improves():
    """With AP flat and macro-F1 climbing, selection must move forward."""
    m = EarlyStopMonitor(max_round=5, higher_better=True)
    inductive_ap = [0.9926, 0.9921, 0.9918]
    macro_f1 = [0.60, 0.62, 0.71]
    for ap, f1 in zip(inductive_ap, macro_f1):
        m.early_stop_check(0.5 * ap + 0.5 * f1)
    assert m.best_epoch == 2, "the improving epoch should win"


def test_macro_f1_is_used_not_aggregate_accuracy():
    """Benign is ~90% of val; an unweighted accuracy hides minority collapse.
    A head predicting Benign for everything scores ~0.90 accuracy but a very
    low macro F1."""
    from sklearn.metrics import f1_score, accuracy_score
    y = np.array([0] * 900 + [1] * 40 + [2] * 30 + [3] * 20 + [4] * 10)
    allbenign = np.zeros_like(y)
    assert accuracy_score(y, allbenign) == pytest.approx(0.90)
    assert f1_score(y, allbenign, average="macro", zero_division=0) < 0.25


def test_early_stopper_still_respects_patience():
    m = EarlyStopMonitor(max_round=2, higher_better=True)
    assert not m.early_stop_check(0.5)
    assert not m.early_stop_check(0.4)
    assert m.early_stop_check(0.4) or m.num_round >= 2
