"""
Message Aggregator implementations for TGN / BiTA.
Aggregates sequence of raw messages for each node before memory update.
"""

import math
from collections import defaultdict
from typing import Optional

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


class BiTAAggregator(MessageAggregator):
    """BiTA message aggregation, Algorithm 1 of Makki Nayeri and Rezvani,
    "BiTA: Bidirectional Gated Recurrent Unit-Transformer Aggregator in a
    Temporal Graph Network Framework for Alert Prediction in Computer
    Networks" (arXiv:2604.22781).

    For every node v due a memory update, with its pending messages grouped by
    the edge (v, peer) they arrived on:

      Eq. 1   m        = MSG(raw message)                   (TGN message function)
      Eq. 3   x        = m + TimeEnc(dt)                    (dt = age of the message
                                                             within its edge sequence)
      Eq. 4-5 z_temp   = BiGRU(x)[-1]                       (final forward and backward
                                                             states, concatenated: 2*d_h)
      Eq. 6   e        = W_e z_temp + b_e                   (-> d_trans)
      Eq. 7-8 z_ctx    = Transformer({e_ij | (i,j) in E_t})  (self-attention ACROSS the
                                                             edges of the batch)
      Eq. 9   h_bar_v  = mean over edges incident to v of z_ctx   (readout)

    h_bar_v is what the GRU memory updater consumes (Eq. 11).

    The class this replaces ran BiGRU -> a bare MultiheadAttention over ONE
    node's message sequence -> last position -> Linear. It had no time
    encoding, no Transformer block (no FFN, residual or norm), no attention
    across edges, and read out the last token instead of mean-pooling. It was
    also never executed: TGN only builds an aggregator when use_memory=True,
    and every shipped encoder ran with memory off.

    Two bounds keep this tractable on flow data, where one 2 s window can hold
    thousands of flows (the paper's alert batches are 128 edges):
      * max_seq_len: at most this many most-recent messages per edge.
      * context_size: edges attend to each other in time-ordered groups of this
        size, i.e. E_t is the paper's training batch of 128 edges.
    """

    #: TGN applies the message function BEFORE this aggregator (paper Step 1),
    #: not after it as it does for the heuristic last/mean aggregators.
    applies_message_function = True

    def __init__(
        self,
        message_dim: int,
        d_h: int,
        d_trans: int,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        context_size: int = 128,
        device: str = "cpu",
    ):
        super(BiTAAggregator, self).__init__(device=device)
        from model.time_encoding import TimeEncode

        self.message_dim = message_dim
        self.d_h = d_h
        self.d_trans = d_trans
        self.max_seq_len = max_seq_len
        self.context_size = context_size
        self.time_encoder = TimeEncode(dimension=message_dim)
        self.bigru = nn.GRU(input_size=message_dim, hidden_size=d_h,
                            bidirectional=True, batch_first=True)
        self.W_e = nn.Linear(2 * d_h, d_trans)
        layer = nn.TransformerEncoderLayer(
            d_model=d_trans, nhead=n_heads, dim_feedforward=2 * d_trans,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)

    @staticmethod
    def _peer(message) -> int:
        # Messages are (raw, t, peer). Older stores hold (raw, t); treat those
        # as a single edge per node rather than failing.
        return int(message[2]) if len(message) > 2 else -1

    def aggregate(self, node_ids, messages, message_function=None):
        to_update = np.unique([n for n in node_ids if len(messages[n]) > 0])
        if len(to_update) == 0:
            return [], torch.zeros(0, device=self.device), torch.zeros(0, device=self.device)

        # Step 2: per-edge, time-ordered message sequences.
        edge_owner, edge_seqs = [], []
        for pos, node_id in enumerate(to_update):
            by_peer = defaultdict(list)
            for msg in messages[node_id]:
                by_peer[self._peer(msg)].append(msg)
            for seq in by_peer.values():
                seq = sorted(seq, key=lambda mm: float(mm[1]))[-self.max_seq_len:]
                edge_owner.append(pos)
                edge_seqs.append(seq)

        n_edges = len(edge_seqs)
        lengths = torch.tensor([len(sq) for sq in edge_seqs], dtype=torch.long)
        L = int(lengths.max())
        raw_dim = edge_seqs[0][0][0].shape[-1]
        dev = edge_seqs[0][0][0].device
        raw = torch.zeros(n_edges, L, raw_dim, device=dev)
        times = torch.zeros(n_edges, L, device=dev)
        for k, seq in enumerate(edge_seqs):
            raw[k, : len(seq)] = torch.stack([mm[0] for mm in seq])
            times[k, : len(seq)] = torch.stack([mm[1].float().reshape(()) for mm in seq])

        # Step 1 (Eq. 1): the message function.
        m = message_function.compute_message(raw) if message_function is not None else raw
        if m.shape[-1] != self.message_dim:
            raise ValueError(
                f"BiTA expects {self.message_dim}-D messages, got {m.shape[-1]}. With "
                f"message_function='identity' the message dimension is the raw "
                f"message width.")

        # Step 3 (Eq. 3): add the time encoding. dt is each message's age
        # relative to the newest message on its edge, so the encoding does not
        # depend on the absolute clock; encoding raw epoch seconds would put
        # every message at an arbitrary phase of every frequency.
        lengths_d = lengths.to(dev)
        last_t = times.gather(1, (lengths_d - 1).clamp(min=0).unsqueeze(1))
        valid = torch.arange(L, device=dev).unsqueeze(0) < lengths_d.unsqueeze(1)
        dt = (last_t - times).clamp(min=0.0) * valid
        x = m + self.time_encoder(dt)

        # Step 4.1 (Eq. 4-5): BiGRU, final hidden state of both directions.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.bigru(packed)                       # [2, E, d_h]
        z_temp = torch.cat([h_n[0], h_n[1]], dim=-1)      # [E, 2*d_h]

        # Step 4.2 (Eq. 6-8): project, then self-attention ACROSS edges, in
        # time-ordered groups of context_size edges.
        e = self.W_e(z_temp)                              # [E, d_trans]
        order = torch.argsort(last_t.squeeze(1))
        C = self.context_size
        n_groups = (n_edges + C - 1) // C
        grouped = torch.zeros(n_groups * C, self.d_trans, device=dev, dtype=e.dtype)
        grouped[:n_edges] = e[order]
        pad = torch.ones(n_groups * C, dtype=torch.bool, device=dev)
        pad[:n_edges] = False
        ctx = self.transformer(grouped.view(n_groups, C, self.d_trans),
                               src_key_padding_mask=pad.view(n_groups, C))
        z_ctx = torch.empty_like(e)
        z_ctx[order] = ctx.reshape(n_groups * C, self.d_trans)[:n_edges]

        # Step 4.3 (Eq. 9): mean-pool each node's incident edges.
        owner = torch.tensor(edge_owner, dtype=torch.long, device=dev)
        n_nodes = len(to_update)
        summed = torch.zeros(n_nodes, self.d_trans, device=dev, dtype=z_ctx.dtype).index_add_(0, owner, z_ctx)
        counts = torch.zeros(n_nodes, device=dev, dtype=z_ctx.dtype).index_add_(
            0, owner, torch.ones(n_edges, device=dev, dtype=z_ctx.dtype))
        h_bar = summed / counts.unsqueeze(1)

        timestamps = torch.stack([
            torch.stack([mm[1] for mm in messages[n]]).max() for n in to_update])
        return list(to_update), h_bar, timestamps


#: Aggregators that are actually implemented. The BiTA paper also ablates
#: BiTransformer, relative-position and stacked variants and a TCN block; those
#: are not implemented here. Asking for one used to silently return the
#: BiGRU-Transformer instead, so a result would be reported under the wrong name.
_BITA_NAMES = ("bita", "bigru_transformer")


def get_message_aggregator(
    aggregator_type: str,
    device: str = "cpu",
    input_dim: int = 100,
    hidden_dim: int = 100,
    n_heads: int = 2,
    dropout: float = 0.1,
    d_trans: Optional[int] = None,
):
    agg_type = aggregator_type.lower()
    if agg_type == "last":
        return LastMessageAggregator(device=device)
    if agg_type == "mean":
        return MeanMessageAggregator(device=device)
    if agg_type in _BITA_NAMES:
        d_trans = d_trans or hidden_dim
        return BiTAAggregator(
            message_dim=input_dim,
            d_h=max(1, d_trans // 2),
            d_trans=d_trans,
            n_heads=n_heads,
            dropout=dropout,
            device=device,
        )
    raise ValueError(
        f"Unknown or unimplemented aggregator {aggregator_type!r}. Implemented: "
        f"'last', 'mean', 'bigru_transformer' (BiTA).")
