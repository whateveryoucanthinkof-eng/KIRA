"""Detection metrics (spec 44).

PR-AUC leads, not accuracy. Under the class imbalance in this problem a
constant "benign" predictor scores high accuracy and is useless; average
precision reflects the minority class the system exists to find.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    hours_observed: Optional[float] = None,
) -> Dict[str, Any]:
    """Binary detection metrics with the operational rate a SOC actually feels.

    `hours_observed` enables false_alarms_per_hour — the number an analyst
    cares about, and one that a precision figure alone hides.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUCs are undefined with a single class present; say so rather than emit 0.5.
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
    else:
        out["pr_auc"] = out["roc_auc"] = out["brier"] = float("nan")
        out["note"] = "single class present; AUC undefined"

    # PR-AUC is bounded below by the base rate. At an extreme base rate it is
    # ~1.0 for any ranking whatsoever, including a constant one, so quoting it
    # without the base rate beside it is misleading. Measured on a real run: a
    # test split that was 99.97% positive gave PR-AUC 0.9998 for both the model
    # AND persistence.
    pr = out.get("pr_auc")
    if pr == pr and (out["positive_rate"] > 0.95 or out["positive_rate"] < 0.05):
        out["pr_auc_baseline"] = out["positive_rate"]
        out["pr_auc_lift"] = float(pr - out["positive_rate"])
        out["extreme_base_rate"] = True
        out["note"] = (
            f"base rate {out['positive_rate']:.4f} is extreme; PR-AUC {pr:.4f} is close to the "
            f"rate a constant predictor achieves. Read pr_auc_lift ({pr - out['positive_rate']:+.4f}) "
            f"and balanced_accuracy instead."
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) else 0.0

    if hours_observed and hours_observed > 0:
        out["false_alarms_per_hour"] = float(fp / hours_observed)

    return out


def multilabel_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> Dict[str, Any]:
    """Multilabel technique metrics (spec 47). y_* are [N, n_labels]."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    present = y_true.sum(axis=0) > 0
    if present.any():
        out["mAP"] = float(
            average_precision_score(y_true[:, present], y_score[:, present], average="macro")
        )
    else:
        out["mAP"] = float("nan")

    for k in (1, 3):
        if y_score.shape[1] >= k:
            topk = np.argsort(-y_score, axis=1)[:, :k]
            hits = np.take_along_axis(y_true, topk, axis=1)
            out[f"precision_at_{k}"] = float(hits.sum(axis=1).mean() / k)
            denom = np.maximum(y_true.sum(axis=1), 1)
            out[f"recall_at_{k}"] = float((hits.sum(axis=1) / denom).mean())

    out["label_support"] = y_true.sum(axis=0).astype(int).tolist()
    return out
