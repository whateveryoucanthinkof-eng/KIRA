"""
LSTM Baseline for Latent Dynamics Forecasting.

A 2-layer LSTM serving as the standard recurrent sequence forecasting baseline,
operating on the exact same latent state history z_{t-H+1:t} and predicting ẑ_{t+1:t+K}.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from world_model.data.feature_schema import D_Z


class LSTMDynamicsBaseline(nn.Module):
    """
    2-layer LSTM baseline for latent sequence transition dynamics.
    """

    def __init__(
        self,
        d_z: int = D_Z,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_z = d_z
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=d_z,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, d_z)

    def forward(
        self,
        z_sequence: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over sequence of length S.
        Predicts ẑ_{i+1} = z_i + output_proj(lstm_out_i)
        """
        out, _ = self.lstm(z_sequence)  # [B, S, hidden_dim]
        delta = self.output_proj(out)   # [B, S, d_z]
        z_pred = z_sequence + delta     # [B, S, d_z]

        return {
            "z_pred": z_pred,
            "delta_pred": delta,
        }

    @torch.no_grad()
    def rollout(
        self,
        z_history: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        K: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressively rolls forward K steps using LSTM hidden state."""
        self.eval()
        B, H, _ = z_history.shape
        out, (h, c) = self.lstm(z_history)

        curr_z = z_history[:, -1:, :]
        last_hidden = out[:, -1:, :]
        z_next = curr_z + self.output_proj(last_hidden)

        future_list = [z_next]
        for _ in range(1, K):
            curr_z = z_next
            step_out, (h, c) = self.lstm(curr_z, (h, c))
            z_next = curr_z + self.output_proj(step_out)
            future_list.append(z_next)

        z_future = torch.cat(future_list, dim=1)  # [B, K, d_z]
        return {"z_future": z_future}
