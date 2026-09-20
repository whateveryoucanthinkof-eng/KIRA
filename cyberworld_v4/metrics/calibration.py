"""Probability calibration (spec 33, 35).

Two things v3 conflated:

  * a learned temperature used *inside the training loss* is not calibration.
    Post-hoc temperature scaling (Guo et al., ICML 2017) freezes the model and
    fits a single scalar on held-out data. Fitting it on training data inherits
    the training fit and calibrates nothing.

  * "calibrated" is a measured property, not a label. Report ECE, Brier, NLL
    and a reliability curve before and after, or do not use the word.

The temperature is fitted on the CALIBRATION split, which is disjoint from
validation so it does not inherit model-selection optimism.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> Tuple[float, Dict[str, Any]]:
    """ECE with equal-width bins, plus the reliability curve behind it."""
    y_true = np.asarray(y_true).astype(float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    curve = {"bin_center": [], "confidence": [], "accuracy": [], "count": []}
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y_true[m].mean())
        ece += (cnt / n) * abs(acc - conf)
        curve["bin_center"].append(float((edges[b] + edges[b + 1]) / 2))
        curve["confidence"].append(conf)
        curve["accuracy"].append(acc)
        curve["count"].append(cnt)

    return float(ece), curve


def calibration_report(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-7, 1 - 1e-7)
    ece, curve = expected_calibration_error(y_true, p, n_bins)
    return {
        "ece": ece,
        "brier": float(np.mean((p - y_true) ** 2)),
        "nll": float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y_true.mean()),
        "reliability": curve,
    }


class TemperatureScaler:
    """Post-hoc temperature scaling for binary probabilities.

    Fit on the calibration split only. `fit` operates on logits; `fit_from_probs`
    accepts probabilities and inverts them first.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.fitted: bool = False
        self._before: Optional[Dict[str, Any]] = None
        self._after: Optional[Dict[str, Any]] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, logits: np.ndarray, y_true: np.ndarray, *, grid: Optional[np.ndarray] = None) -> "TemperatureScaler":
        """Pick T minimising NLL. A 1-D grid search is exact enough and cannot
        diverge, unlike an unconstrained optimiser on a near-flat objective."""
        logits = np.asarray(logits, dtype=float).ravel()
        y = np.asarray(y_true, dtype=float).ravel()
        grid = grid if grid is not None else np.concatenate(
            [np.linspace(0.05, 1.0, 96)[:-1], np.linspace(1.0, 10.0, 181)]
        )

        self._before = calibration_report(y, self._sigmoid(logits))
        best_t, best_nll = 1.0, np.inf
        for t in grid:
            p = np.clip(self._sigmoid(logits / t), 1e-7, 1 - 1e-7)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = float(nll), float(t)

        self.temperature = best_t
        self.fitted = True
        self._after = calibration_report(y, self._sigmoid(logits / best_t))
        return self

    def fit_from_probs(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        p = np.clip(np.asarray(probs, dtype=float).ravel(), 1e-6, 1 - 1e-6)
        return self.fit(np.log(p / (1 - p)), y_true)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.asarray(logits, dtype=float) / self.temperature)

    def transform_probs(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        return self.transform_logits(np.log(p / (1 - p)))

    def report(self) -> Dict[str, Any]:
        """Before/after evidence. This is what justifies the word 'calibrated'."""
        return {
            "temperature": self.temperature,
            "fitted": self.fitted,
            "before": self._before,
            "after": self._after,
            "ece_improvement": (
                None
                if not (self._before and self._after)
                else float(self._before["ece"] - self._after["ece"])
            ),
        }
