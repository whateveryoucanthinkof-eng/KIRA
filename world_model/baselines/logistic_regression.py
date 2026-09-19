"""
Linear / Ridge Autoregressive Baseline for Latent Dynamics.

Predicts the next latent state ẑ_{t+1} as a linear map over the flattened
history window: ẑ_{t+1} = W · vec(z_{t-H+1:t}) + b + z_t.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from world_model.data.feature_schema import D_Z


class LinearDynamicsBaseline(nn.Module):
    """
    Linear autoregressive baseline with residual connection.
    """

    def __init__(self, history_len: int = 16, d_z: int = D_Z):
        super().__init__()
        self.history_len = history_len
        self.d_z = d_z
        self.linear = nn.Linear(history_len * d_z, d_z)

    def forward(
        self,
        z_sequence: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_sequence: [B, S, d_z] where S >= history_len
        """
        B, S, _ = z_sequence.shape
        H = self.history_len

        preds = []
        for i in range(H, S + 1):
            window = z_sequence[:, i - H : i, :].reshape(B, -1)
            delta = self.linear(window)
            last_z = z_sequence[:, i - 1, :]
            pred = last_z + delta
            preds.append(pred.unsqueeze(1))

        z_pred_tail = torch.cat(preds, dim=1) if preds else z_sequence
        return {
            "z_pred": z_pred_tail,
            "z_next": preds[-1].squeeze(1) if preds else z_sequence[:, -1, :],
        }

    @torch.no_grad()
    def rollout(
        self,
        z_history: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        K: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Iteratively rolls forward K steps."""
        self.eval()
        B, H, _ = z_history.shape
        curr_context = z_history.clone()
        future_list = []

        for _ in range(K):
            window = curr_context[:, -self.history_len:, :].reshape(B, -1)
            delta = self.linear(window)
            z_next = curr_context[:, -1, :] + delta
            future_list.append(z_next.unsqueeze(1))
            curr_context = torch.cat([curr_context, z_next.unsqueeze(1)], dim=1)

        z_future = torch.cat(future_list, dim=1)
        return {"z_future": z_future}
