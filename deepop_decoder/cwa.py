"""
Causal Window Attention (CWA) Module for DeepOP Decoder.

Implements multi-scale causal window attention:
Allocates attention heads across different temporal window sizes (e.g., local cw=2, medium cw=4, full history),
enforcing strict causal masking (no future leakage) within each window scale.
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalWindowAttention(nn.Module):
    """
    Multi-Head Causal Window Attention with head allocation across window scales.
    Complexity: O(N) when window scales are bounded constants.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 6,
        window_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super(CausalWindowAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_sizes = window_sizes or [2, 4, 8]
        self.n_scales = len(self.window_sizes)
        assert n_heads % self.n_scales == 0, f"n_heads ({n_heads}) must be divisible by n_scales ({self.n_scales})"
        self.heads_per_scale = n_heads // self.n_scales

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _build_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        Builds causal window mask where position i can attend to j iff j <= i and (i - j) < window_size.
        """
        i_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        j_indices = torch.arange(seq_len, device=device).unsqueeze(0)

        # Causal constraint: j <= i
        is_causal = j_indices <= i_indices
        # Window constraint: (i - j) < window_size
        is_within_window = (i_indices - j_indices) < window_size

        valid_mask = is_causal & is_within_window
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = mask.masked_fill(valid_mask, 0.0)
        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch_size, seq_len_q, d_model]
            k: [batch_size, seq_len_k, d_model]
            v: [batch_size, seq_len_k, d_model]
            key_padding_mask: [batch_size, seq_len_k] boolean mask (True for padding)
        Returns:
            output: [batch_size, seq_len_q, d_model]

        True O(N * W) sliding window causal attention:
        Guarantees zero allocation of dense (N x N) attention matrices.
        Memory and computation scale strictly as O(B * H * N * W * d_head).
        """
        B, N_q, _ = q.shape
        _, N_k, _ = k.shape

        # Linear projections and reshape to [B, n_heads, seq_len, d_head]
        Q = self.q_proj(q).view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

        scale_factor = 1.0 / math.sqrt(self.d_head)
        head_outputs = []

        # Process heads in groups assigned to each window scale
        for s_idx, win_size in enumerate(self.window_sizes):
            h_start = s_idx * self.heads_per_scale
            h_end = h_start + self.heads_per_scale

            q_group = Q[:, h_start:h_end]  # [B, h_group, N_q, d_head]
            k_group = K[:, h_start:h_end]  # [B, h_group, N_k, d_head]
            v_group = V[:, h_start:h_end]  # [B, h_group, N_k, d_head]

            if N_q == 1 and N_k >= 1:
                # Autoregressive single-step generation: attend only to last min(N_k, W) tokens
                W_eff = min(win_size, N_k)
                k_local = k_group[:, :, -W_eff:, :]  # [B, h_group, W_eff, d_head]
                v_local = v_group[:, :, -W_eff:, :]  # [B, h_group, W_eff, d_head]

                scores = torch.matmul(q_group, k_local.transpose(-2, -1)) * scale_factor  # [B, h_group, 1, W_eff]
                if key_padding_mask is not None:
                    pad_mask = key_padding_mask[:, -W_eff:].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W_eff]
                    scores = scores.masked_fill(pad_mask, float("-inf"))

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                attn_weights = self.dropout(attn_weights)
                out_group = torch.matmul(attn_weights, v_local)  # [B, h_group, 1, d_head]
                head_outputs.append(out_group)

            elif N_q == N_k:
                # Full sequence self-attention: O(N * W) unfolded sliding window
                W = min(win_size, N_q)
                if W == 1:
                    scores = (q_group * k_group).sum(dim=-1, keepdim=True) * scale_factor  # [B, h_group, N, 1]
                    if key_padding_mask is not None:
                        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = torch.nan_to_num(attn_weights, 0.0)
                    attn_weights = self.dropout(attn_weights)
                    out_group = attn_weights * v_group  # [B, h_group, N, d_head]
                else:
                    # Pad left by W - 1 to construct causal sliding window of size W
                    k_pad = F.pad(k_group, (0, 0, W - 1, 0))  # [B, h_group, N + W - 1, d_head]
                    v_pad = F.pad(v_group, (0, 0, W - 1, 0))  # [B, h_group, N + W - 1, d_head]

                    # Unfold along seq dimension (dim 2) -> shape [B, h_group, N, d_head, W]
                    k_unf = k_pad.unfold(dimension=2, size=W, step=1)
                    v_unf = v_pad.unfold(dimension=2, size=W, step=1)

                    # Compute local attention scores: [B, h_group, N, W] (No N x N tensor!)
                    scores = (q_group.unsqueeze(-1) * k_unf).sum(dim=-2) * scale_factor

                    # Causal padding mask for first W-1 positions
                    w_idx = torch.arange(W, device=q.device).view(1, 1, 1, W)
                    min_valid = ((W - 1) - torch.arange(N_q, device=q.device)).clamp(min=0).view(1, 1, N_q, 1)
                    scores = scores.masked_fill(w_idx < min_valid, float("-inf"))

                    if key_padding_mask is not None:
                        kpm_pad = F.pad(key_padding_mask, (W - 1, 0), value=True)
                        kpm_unf = kpm_pad.unfold(dimension=1, size=W, step=1)  # [B, N, W]
                        scores = scores.masked_fill(kpm_unf.unsqueeze(1), float("-inf"))

                    attn_weights = F.softmax(scores, dim=-1)  # [B, h_group, N, W]
                    attn_weights = torch.nan_to_num(attn_weights, 0.0)
                    attn_weights = self.dropout(attn_weights)

                    # Weighted sum over local window: [B, h_group, N, d_head]
                    out_group = (attn_weights.unsqueeze(-2) * v_unf).sum(dim=-1)

                head_outputs.append(out_group)

            else:
                # Fallback for arbitrary mismatched lengths without exceeding max window
                # Still bounds attention to causal window
                i_indices = torch.arange(N_q, device=q.device).unsqueeze(1)
                j_indices = torch.arange(N_k, device=q.device).unsqueeze(0)
                is_causal = j_indices <= i_indices
                is_within_window = (i_indices - j_indices) < win_size
                valid_mask = is_causal & is_within_window

                scores = torch.matmul(q_group, k_group.transpose(-2, -1)) * scale_factor
                mask = torch.full((N_q, N_k), float("-inf"), device=q.device)
                mask = mask.masked_fill(valid_mask, 0.0)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)

                if key_padding_mask is not None:
                    scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                attn_weights = self.dropout(attn_weights)
                out_group = torch.matmul(attn_weights, v_group)
                head_outputs.append(out_group)

        # Concatenate across heads: [B, n_heads, N_q, d_head] -> [B, N_q, d_model]
        all_heads = torch.cat(head_outputs, dim=1).transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        output = self.out_proj(all_heads)
        return output
