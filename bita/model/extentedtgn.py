"""
ExtendedTGN Architecture for BiTA / TGNE-TA.

Extends standard TGN with:
1. Multi-class edge category prediction (Focal loss / Cross Entropy).
2. Direct per-host state extraction: get_host_embeddings(node_ids, timestamp) -> H_t.
3. Global state pooling: get_global_state(H_t) -> h_hat_t.
"""

import numpy as np
import torch
import torch.nn as nn
from model.tgn import TGN


class ExtendedTGN(TGN):
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
        num_categories=4,
    ):
        super(ExtendedTGN, self).__init__(
            neighbor_finder=neighbor_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
            memory_update_at_start=memory_update_at_start,
            message_dimension=message_dimension,
            memory_dimension=memory_dimension,
            embedding_module_type=embedding_module_type,
            message_function=message_function,
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            n_neighbors=n_neighbors,
            aggregator_type=aggregator_type,
            memory_updater_type=memory_updater_type,
            use_destination_embedding_in_message=use_destination_embedding_in_message,
            use_source_embedding_in_message=use_source_embedding_in_message,
            dyrep=dyrep,
        )

        self.num_categories = num_categories
        self.category_predictor = nn.Linear(self.embedding_dimension, num_categories)

    def compute_edge_probabilities_and_categories(
        self,
        source_nodes,
        destination_nodes,
        negative_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=20,
    ):
        pos_score, neg_score = super().compute_edge_probabilities(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors
        )

        source_node_embedding, destination_node_embedding, _ = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors
        )

        combined_embeddings = source_node_embedding + destination_node_embedding
        category_logits = self.category_predictor(combined_embeddings)

        return pos_score, neg_score, category_logits

    def get_host_embeddings(
        self,
        node_ids: np.ndarray,
        timestamp: float,
        n_neighbors: int = 20,
    ) -> torch.Tensor:
        """
        Extracts structural and temporal node embeddings H_t for specified host IDs at given timestamp.
        Interface contract: H_t ∈ R^{|V_t| × d}.
        """
        if len(node_ids) == 0:
            return torch.zeros((0, self.embedding_dimension), device=self.device)

        timestamps = np.full(len(node_ids), timestamp, dtype=float)

        memory = None
        time_diffs = None
        if self.use_memory:
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update
            time_diffs = (
                torch.from_numpy(timestamps).float().to(self.device)
                - last_update[node_ids].float()
            )
            time_diffs = (time_diffs - self.mean_time_shift_src) / self.std_time_shift_src

        node_embeddings = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=node_ids,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )

        return node_embeddings

    def get_global_state(self, host_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes pooled global network summary state ĥ_t = pool(H_t) ∈ R^d.
        """
        if host_embeddings.shape[0] == 0:
            return torch.zeros(self.embedding_dimension, device=self.device)
        return host_embeddings.mean(dim=0)
