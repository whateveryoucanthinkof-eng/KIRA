"""
Message Aggregator implementations for TGN / BiTA.
Aggregates sequence of raw messages for each node before memory update.
"""

import math
from collections import defaultdict
import numpy as np
import torch
from torch import nn


class MessageAggregator(nn.Module):
    def __init__(self, device="cpu"):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        pass


class LastMessageAggregator(MessageAggregator):
    def __init__(self, device="cpu"):
        super(LastMessageAggregator, self).__init__(device=device)

    def aggregate(self, node_ids, messages):
        unique_nodes = []
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []
        for node_id in node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)

        to_update_node_ids = np.unique(to_update_node_ids)
        for node_id in to_update_node_ids:
            if len(messages[node_id]) > 0:
                unique_nodes.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = (
            torch.stack(unique_messages)
            if len(unique_messages) > 0
            else torch.zeros(0, device=self.device)
        )
        unique_timestamps = (
            torch.stack(unique_timestamps)
            if len(unique_timestamps) > 0
            else torch.zeros(0, device=self.device)
        )

        return unique_nodes, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device="cpu"):
        super(MeanMessageAggregator, self).__init__(device=device)

    def aggregate(self, node_ids, messages):
        unique_nodes = []
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []
        for node_id in node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)

        to_update_node_ids = np.unique(to_update_node_ids)
        for node_id in to_update_node_ids:
            if len(messages[node_id]) > 0:
                unique_nodes.append(node_id)
                node_msgs = torch.stack([m[0] for m in messages[node_id]])
                unique_messages.append(node_msgs.mean(dim=0))
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = (
            torch.stack(unique_messages)
            if len(unique_messages) > 0
            else torch.zeros(0, device=self.device)
        )
        unique_timestamps = (
            torch.stack(unique_timestamps)
            if len(unique_timestamps) > 0
            else torch.zeros(0, device=self.device)
        )

        return unique_nodes, unique_messages, unique_timestamps


class BiGRUTransformerAggregator(MessageAggregator):
    """
    BiGRU + Transformer message aggregator as described in BiTA / TGNE-TA.
    Aggregates temporal sequence of messages using bidirectional GRU and self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super(BiGRUTransformerAggregator, self).__init__(device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def aggregate(self, node_ids, messages):
        unique_nodes = []
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = np.unique([n for n in node_ids if len(messages[n]) > 0])
        if len(to_update_node_ids) == 0:
            return [], torch.zeros(0, device=self.device), torch.zeros(0, device=self.device)

        for node_id in to_update_node_ids:
            unique_nodes.append(node_id)
            node_msgs = torch.stack([m[0] for m in messages[node_id]]).unsqueeze(0)  # [1, seq_len, dim]
            gru_out, _ = self.bigru(node_msgs)
            attn_out, _ = self.attn(gru_out, gru_out, gru_out)
            agg = self.proj(attn_out[:, -1, :]).squeeze(0)
            unique_messages.append(agg)
            unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages)
        unique_timestamps = torch.stack(unique_timestamps)
        return unique_nodes, unique_messages, unique_timestamps


def get_message_aggregator(
    aggregator_type: str,
    device: str = "cpu",
    input_dim: int = 100,
    hidden_dim: int = 100,
    n_heads: int = 2,
    dropout: float = 0.1,
):
    agg_type = aggregator_type.lower()
    if agg_type == "last":
        return LastMessageAggregator(device=device)
    elif agg_type == "mean":
        return MeanMessageAggregator(device=device)
    elif "bigru" in agg_type or "transformer" in agg_type:
        return BiGRUTransformerAggregator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            device=device,
        )
    else:
        return LastMessageAggregator(device=device)
