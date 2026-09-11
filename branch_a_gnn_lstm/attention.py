"""
Temporal Sequence Attention for Branch A LSTM.
Computes attention distribution over sequence timesteps and outputs context vector + weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    Bahdanau / Feed-Forward self-attention over LSTM hidden states:
    e_t = v^T tanh(W h_t + b)
    alpha = softmax(e)
    context = sum_t (alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super(TemporalSelfAttention, self).__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            lstm_outputs: [batch_size, seq_len, hidden_dim]
            mask: Optional [batch_size, seq_len] boolean mask (True for padding)
        Returns:
            context: [batch_size, hidden_dim]
            weights: [batch_size, seq_len]
        """
        # [batch_size, seq_len, attn_dim]
        energy = torch.tanh(self.proj(lstm_outputs))
        # [batch_size, seq_len]
        scores = self.v(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        # [batch_size, 1, seq_len] @ [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context, weights
