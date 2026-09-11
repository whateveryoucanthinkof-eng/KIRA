"""
Branch B: Infiltration Risk Prediction Head.

Estimates per-step compromise likelihood and cumulative horizon risk
over predicted future host states H_{t+1..t+K}[v].
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiltrationRiskHead(nn.Module):
    """
    MLP head estimating compromise probability from latent host states.
    r_{t+k}[v] = sigmoid(MLP(H_{t+k}[v]))
    R_{cumul}[v] = 1 - prod_{k=1}^K (1 - r_{t+k}[v])
    """

    def __init__(self, d_latent: int = 12, hidden_dim: int = 32, dropout: float = 0.1):
        super(InfiltrationRiskHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self.forward_trajectory(x)[0]
        return self.forward_step(x)

    def forward_step(self, h_state: torch.Tensor) -> torch.Tensor:
        """
        Computes single-step compromise probability.
        Args:
            h_state: [batch_size, d_latent]
        Returns:
            risk: [batch_size] in [0, 1]
        """
        return self.mlp(h_state).squeeze(-1)

    def forward_trajectory(self, h_rollout: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-step ATT&CK Severity-Derived Risk Score S_{t+k} in [0, 1]
        and cumulative horizon risk across K forward steps.

        Args:
            h_rollout: [batch_size, K, d_latent]
        Returns:
            step_risks: [batch_size, K] per-step severity scores S_{t+k} in [0, 1]
            cumulative_risk: [batch_size] peak forecast severity over horizon max_k S_{t+k}
        """
        B, K, D = h_rollout.shape
        h_flat = h_rollout.reshape(B * K, D)
        step_risks = self.mlp(h_flat).reshape(B, K)

        # Peak forecast severity across horizon K (deterministic peak risk)
        peak_risk = torch.max(step_risks, dim=-1)[0]
        cumulative_risk = peak_risk

        return step_risks, cumulative_risk
