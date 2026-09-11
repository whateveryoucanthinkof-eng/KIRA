"""
Detection Metrics for Intrusion and Attack Stage Classification.

Computes comprehensive cybersecurity detection performance:
1. Binary Detection: Precision, Recall, F1, AUROC, AUPRC, FPR@95% TPR.
2. Attack-Stage Multi-Class: Macro-F1, Micro-F1, per-class F1, Confusion Matrix.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from world_model.data.feature_schema import ATTACK_STAGE_NAMES, N_ATTACK_STAGES


def compute_detection_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Binary intrusion detection metrics.
    
    Args:
        y_true: [N] binary 0/1 true labels
        y_scores: [N] predicted probabilities in [0, 1]
        threshold: decision threshold for binary classification
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_scores))
    except ValueError:
        auprc = 0.0

    # FPR at 95% Recall (TPR)
    fpr_at_95 = 1.0
    try:
        fprs, tprs, _ = roc_curve(y_true, y_scores)
        idx = np.where(tprs >= 0.95)[0]
        if len(idx) > 0:
            fpr_at_95 = float(fprs[idx[0]])
    except Exception:
        pass

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr_at_95,
    }


def compute_attack_stage_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """
    Multi-class MITRE ATT&CK stage metrics (5 classes).
    
    Args:
        y_true: [N] ground-truth stage indices (0..4)
        y_logits: [N, 5] predicted unnormalized logits or probabilities
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.argmax(y_logits, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    per_class_f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {}
    for i, f1_val in enumerate(per_class_f1_vals):
        name = ATTACK_STAGE_NAMES.get(i, f"STAGE_{i}")
        per_class_f1[name] = float(f1_val)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_ATTACK_STAGES))).tolist()

    return {
        "stage_accuracy": acc,
        "stage_macro_f1": macro_f1,
        "stage_micro_f1": micro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }
