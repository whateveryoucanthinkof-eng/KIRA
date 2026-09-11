"""
Temporal Graph Network (TGN) core architecture.
Rossi et al. / BiTA temporal graph engine.
"""

import logging
from collections import defaultdict
import numpy as np
import torch
from torch import nn

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(nn.Module):
    def __init__(
        self,
        neighbor_finder,
        node_features,
        edge_features,
        device,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        memory_update_at_start=True,
        message_dimension=100,
        memory_dimension=100,
        embedding_module_type="graph_attention",
        message_function="mlp",
        mean_time_shift_src=0,
        std_time_shift_src=1,
        mean_time_shift_dst=0,
        std_time_shift_dst=1,
        n_neighbors=20,
        aggregator_type="last",
        memory_updater_type="gru",
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        raw_message_dimension=None,
    ):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = (
            torch.from_numpy(node_features.astype(np.float32)).to(device)
            if node_features is not None
            else None
        )
        self.edge_raw_features = (
            torch.from_numpy(edge_features.astype(np.float32)).to(device)
            if edge_features is not None
            else None
        )

        self.n_node_features = (
            self.node_raw_features.shape[1]
            if self.node_raw_features is not None
            else 12
        )
        self.n_nodes = (
            self.node_raw_features.shape[0]
            if self.node_raw_features is not None
            else 50000
        )
        self.n_edge_features = (
            self.edge_raw_features.shape[1]
            if self.edge_raw_features is not None
            else 12
        )
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start

            if raw_message_dimension is None:
                raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension

            self.message_function = get_message_function(
                module_type=message_function,
                raw_message_dimension=raw_message_dimension,
                message_dimension=message_dimension,
            )
            self.message_aggregator = get_message_aggregator(
                aggregator_type=aggregator_type,
                device=device,
                input_dim=raw_message_dimension,
                hidden_dim=raw_message_dimension,
                n_heads=n_heads,
                dropout=dropout,
            )
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=self.memory_dimension,
                input_dimension=message_dimension,
                message_dimension=message_dimension,
                device=device,
            )
            self.memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=message_dimension,
                memory_dimension=self.memory_dimension,
                device=device,
            )

        self.embedding_module = get_embedding_module(
            module_type=embedding_module_type,
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=self.n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_node_features,
            embedding_dimension=self.embedding_dimension,
            device=self.device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
            n_neighbors=self.n_neighbors,
        )

        self.affinity_score = MergeLayer(
            self.n_node_features, self.n_node_features, self.n_node_features, 1
        )

    def compute_temporal_embeddings(
        self,
        source_nodes,
        destination_nodes,
        negative_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=20,
    ):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all unique nodes involved in the batch
                self.update_memory(positives, self.memory.messages)
                self.memory.clear_messages(positives)

            unique_sources, source_id_to_messages = self.get_raw_messages(
                source_nodes,
                source_nodes,
                destination_nodes,
                destination_nodes,
                edge_times,
                edge_idxs,
            )
            unique_destinations, destination_id_to_messages = self.get_raw_messages(
                destination_nodes,
                destination_nodes,
                source_nodes,
                source_nodes,
                edge_times,
                edge_idxs,
            )
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)

            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update

            source_time_diffs = (
                torch.from_numpy(edge_times).float().to(self.device)
                - last_update[source_nodes].float()
            )
            source_time_diffs = (
                source_time_diffs - self.mean_time_shift_src
            ) / self.std_time_shift_src

            destination_time_diffs = (
                torch.from_numpy(edge_times).float().to(self.device)
                - last_update[destination_nodes].float()
            )
            destination_time_diffs = (
                destination_time_diffs - self.mean_time_shift_dst
            ) / self.std_time_shift_dst

            negative_time_diffs = (
                torch.from_numpy(edge_times).float().to(self.device)
                - last_update[negative_nodes].float()
            )
            negative_time_diffs = (
                negative_time_diffs - self.mean_time_shift_dst
            ) / self.std_time_shift_dst

            time_diffs = torch.cat(
                [source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0
            )

        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples : 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples :]

        if self.use_memory and not self.memory_update_at_start:
            self.update_memory(positives, self.memory.messages)
            self.memory.clear_messages(positives)
            self.memory.store_raw_messages(unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_edge_probabilities(
        self,
        source_nodes,
        destination_nodes,
        negative_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=20,
    ):
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = (
            self.compute_temporal_embeddings(
                source_nodes,
                destination_nodes,
                negative_nodes,
                edge_times,
                edge_idxs,
                n_neighbors,
            )
        )

        score = self.affinity_score(
            torch.cat([source_node_embedding, source_node_embedding], dim=0),
            torch.cat([destination_node_embedding, negative_node_embedding], dim=0),
        ).squeeze(dim=-1)

        pos_score = score[:n_samples].sigmoid()
        neg_score = score[n_samples:].sigmoid()

        return pos_score, neg_score

    def update_memory(self, nodes, messages):
        unique_nodes, unique_messages, unique_timestamps = (
            self.message_aggregator.aggregate(nodes, messages)
        )
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
            self.memory_updater.update_memory(
                unique_nodes, unique_messages, timestamps=unique_timestamps
            )

    def get_updated_memory(self, nodes, messages):
        unique_nodes, unique_messages, unique_timestamps = (
            self.message_aggregator.aggregate(nodes, messages)
        )
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
            updated_memory, updated_last_update = (
                self.memory_updater.get_updated_memory(
                    unique_nodes, unique_messages, timestamps=unique_timestamps
                )
            )
        else:
            updated_memory = self.memory.memory.data.clone()
            updated_last_update = self.memory.last_update.data.clone()
        return updated_memory, updated_last_update

    def get_raw_messages(
        self,
        source_nodes,
        source_node_embedding,
        destination_nodes,
        destination_node_embedding,
        edge_times,
        edge_idxs,
    ):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = (
            self.edge_raw_features[edge_idxs]
            if self.edge_raw_features is not None
            else torch.zeros((len(source_nodes), self.n_edge_features), device=self.device)
        )

        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(1)).view(
            len(source_nodes), -1
        )

        source_message = torch.cat(
            [source_memory, destination_memory, edge_features, source_time_delta_encoding],
            dim=1,
        )

        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
