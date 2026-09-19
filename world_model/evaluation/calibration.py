"""
Probability Calibration and Reliability Metrics for World Model Predictions.

Implements:
1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across M bins (Naeini et al., 2015).
2. Brier Score: strictly proper scoring rule measuring accuracy of probabilistic forecasts.
3. Temperature Scaling: post-hoc Platt / Guo et al. (2017) calibration parameter optimization.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes ECE, MCE, and Brier score.
    
    Args:
        probs: [N] predicted probabilities in [0, 1]
        labels: [N] ground-truth binary labels in {0, 1}
        n_bins: number of confidence bins (default 10)
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    N = len(probs)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0}

    # Brier Score: MSE between probability and true outcome
    brier_score = float(np.mean((probs - labels) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
    }


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling (Guo et al., 2017).
    Optimizes a single temperature parameter T > 0 on validation set using NLL.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """Finds optimal temperature minimizing NLL."""
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        criterion = nn.CrossEntropyLoss() if val_logits.dim() > 1 and val_logits.size(1) > 1 else nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())
