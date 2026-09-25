"""Per-horizon forecasting metrics and the degradation curve (spec 44).

A single aggregate number hides the thing that matters about a forecaster:
how fast it decays with horizon. Everything here is reported per step k.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .detection import detection_metrics


def _nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def horizon_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    window_seconds: float,
    threshold: float = 0.5,
    hours_observed: Optional[float] = None,
) -> Dict[str, Any]:
    """Metrics per forecast step. y_* are [N, K]."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")

    K = y_true.shape[1]
    per_step: List[Dict[str, Any]] = []
    for k in range(K):
        m = detection_metrics(
            y_true[:, k], y_score[:, k], threshold=threshold, hours_observed=hours_observed
        )
        m["step"] = k + 1
        m["horizon_seconds"] = (k + 1) * window_seconds
        m["nll"] = _nll(y_true[:, k].astype(float), y_score[:, k])
        per_step.append(m)

    pr = [m["pr_auc"] for m in per_step]
    return {
        "per_step": per_step,
        "pr_auc_by_step": pr,
        "brier_by_step": [m["brier"] for m in per_step],
        "nll_by_step": [m["nll"] for m in per_step],
        # How much predictive power is lost across the horizon. The headline
        # number for a forecasting claim.
        "degradation_pr_auc": (
            float(pr[0] - pr[-1]) if K > 1 and np.isfinite(pr[0]) and np.isfinite(pr[-1]) else float("nan")
        ),
        "horizon_seconds": [m["horizon_seconds"] for m in per_step],
    }


def cumulative_onset_metrics(
    onset_step: np.ndarray,
    hazard_pred: np.ndarray,
    *,
    window_seconds: float,
    no_onset_value: int = -1,
) -> Dict[str, Any]:
    """Score cumulative onset probability derived from predicted hazards.

    P(onset <= t+k) = 1 - prod_{j<=k}(1-h_j). Never max(h).
    """
    onset_step = np.asarray(onset_step).astype(int).ravel()
    h = np.clip(np.asarray(hazard_pred, dtype=float), 0.0, 1.0)
    cum = 1.0 - np.cumprod(1.0 - h, axis=1)

    K = h.shape[1]
    per_step = []
    for k in range(K):
        y = ((onset_step != no_onset_value) & (onset_step <= k + 1)).astype(int)
        m = detection_metrics(y, cum[:, k])
        m["step"] = k + 1
        m["horizon_seconds"] = (k + 1) * window_seconds
        per_step.append(m)

    return {
        "per_step": per_step,
        "pr_auc_by_step": [m["pr_auc"] for m in per_step],
        "final_cumulative_pr_auc": per_step[-1]["pr_auc"] if per_step else float("nan"),
    }
