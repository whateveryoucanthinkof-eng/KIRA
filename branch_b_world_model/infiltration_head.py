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
            cumulative_risk: [batch_size] accumulated hazard over the horizon,
                1 - prod_k (1 - S_{t+k}). Use peak_risk() for the max.
        """
        B, K, D = h_rollout.shape
        h_flat = h_rollout.reshape(B * K, D)
        step_risks = self.mlp(h_flat).reshape(B, K)

        # Cumulative hazard over the horizon: 1 - prod_k (1 - r_k).
        #
        # This returned max(step_risks) -- the PEAK -- under the name
        # `cumulative_risk`, contradicting this class's own docstring and the
        # v4 metrics module, whose test
        # (tests/test_v4_metrics.py::test_cumulative_onset_uses_one_minus_product_not_max)
        # demonstrates why peak is the wrong statistic here:
        #
        #   [0.50, 0, 0, 0, 0]        peak 0.50, cumulative 0.500
        #   [0.30, 0.30, 0.30, 0, 0]  peak 0.30, cumulative 0.657  <- the real onset
        #
        # Peak ranks the single spike above the sustained threat; the product
        # form ranks them correctly. That matters precisely where this value
        # is used -- correlation/trajectory_assembler.py feeds it to
        # HostAttackTrajectory.cumulative_forecast_risk, which is how hosts
        # are ordered for an analyst. A host under persistent moderate
        # pressure should outrank one with a single noisy spike.
        #
        # Computed in log space for numerical stability across the horizon.
        cumulative_risk = 1.0 - torch.exp(
            torch.log1p(-step_risks.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        )

        return step_risks, cumulative_risk

    @staticmethod
    def peak_risk(step_risks: torch.Tensor) -> torch.Tensor:
        """Largest single-step risk over the horizon.

        Kept available because it answers a different question -- "how bad
        does it get at worst" rather than "how likely is compromise at all
        across the horizon". It is NOT the cumulative figure and must not be
        used to rank hosts by onset risk.
        """
        return torch.max(step_risks, dim=-1)[0]
