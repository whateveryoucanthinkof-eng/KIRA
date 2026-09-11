"""
Temporal Attention Layer for TGN.
Aggregates temporal neighborhood features using multi-head attention.
"""

import torch
import torch.nn as nn
from utils.utils import MergeLayer


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_neighbors_features: int,
        n_edge_features: int,
        time_dim: int,
        output_dimension: int,
        n_head: int = 2,
        dropout: float = 0.1,
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.n_head = n_head
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.query_dim = n_node_features + time_dim
        self.key_dim = n_neighbors_features + time_dim + n_edge_features

        self.merger = MergeLayer(
            self.query_dim, n_node_features, n_node_features, output_dimension
        )

        self.multi_head_target = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.key_dim,
            num_heads=n_head,
            dropout=dropout,
        )

    def forward(
        self,
        src_node_features: torch.Tensor,
        src_time_features: torch.Tensor,
        neighbors_features: torch.Tensor,
        neighbors_time_features: torch.Tensor,
        edge_features: torch.Tensor,
        neighbors_padding_mask: torch.Tensor,
    ):
        # src_node_features: [batch_size, feat_dim]
        # src_time_features: [batch_size, 1, time_dim]
        # neighbors_features: [batch_size, n_neighbors, feat_dim]
        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        key = torch.cat(
            [neighbors_features, edge_features, neighbors_time_features], dim=2
        )

        # PyTorch multihead attention expects [seq_len, batch_size, feat_dim]
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)

        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(-1), 0] = False

        attn_output, attn_output_weights = self.multi_head_target(
            query=query,
            key=key,
            value=key,
            key_padding_mask=neighbors_padding_mask,
        )

        attn_output = attn_output.squeeze(0)
        attn_output_weights = attn_output_weights.squeeze(1)

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(
            invalid_neighborhood_mask, 0
        )

        attn_output = self.merger(attn_output, src_node_features)
        return attn_output, attn_output_weights
