"""
Static GCN Baseline for Network Dynamics.

A non-temporal static Graph Convolutional Network (Kipf & Welling, 2017)
operating on snapshot graphs without temporal memory, serving as the spatial-only baseline.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.data.feature_schema import D_Z


class StaticGCNLayer(nn.Module):
    """Standard Graph Convolutional Layer: H' = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features] or [B, N, in_features]
        adj: [N, N] normalized adjacency matrix with self-loops
        """
        support = self.linear(x)
        if x.dim() == 2:
            out = torch.spmm(adj, support) if adj.is_sparse else torch.matmul(adj, support)
        else:
            out = torch.matmul(adj, support)
        return out + self.bias


class StaticGCNBaseline(nn.Module):
    """
    Static GCN baseline that processes graph snapshots independently
    and projects pooled graph embeddings into next-state predictions.
    """

    def __init__(self, in_features: int = 7, hidden_dim: int = 64, d_z: int = D_Z):
        super().__init__()
        self.gcn1 = StaticGCNLayer(in_features, hidden_dim)
        self.gcn2 = StaticGCNLayer(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, d_z)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(node_feats, adj))
        h = F.relu(self.gcn2(h, adj))
        # Mean readout
        if h.dim() == 3:
            pooled = h.mean(dim=1)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        z_pred = self.readout(pooled)
        return z_pred
