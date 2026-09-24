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

        # Category head: [src_emb ; dst_emb ; edge_features] -> class.
        #
        # This was `nn.Linear(embedding_dimension, num_categories)` applied to
        # `src_emb + dst_emb`. Two defects, the first fatal:
        #
        #   1. It never saw the EDGE. The label (Benign / C2 / Impact /
        #      InitialAccess / Recon) is a property of the FLOW -- its ports,
        #      byte volumes, duration, protocol. With only node embeddings, a
        #      benign flow and an attack flow between the SAME host pair are
        #      literally identical inputs. The head could not separate them
        #      even in principle, so it collapsed to a constant prediction.
        #
        #   2. Addition is symmetric, so A->B and B->A were indistinguishable.
        #      Direction is most of what separates a scanner from its target.
        #
        # Measured collapse, both runs predicting one class for everything:
        #   old alpha  -> always Benign,        aggregate CatAcc 0.8351 (looked fine)
        #   new alpha  -> always InitialAccess, aggregate CatAcc 0.2115
        # Only which class it collapsed onto changed, following whichever the
        # loss favoured -- the signature of a head with no discriminative
        # signal.
        #
        # Concatenation keeps direction; the edge features supply the flow;
        # the hidden layer adds the nonlinearity a single Linear lacked.
        edge_dim = (
            self.edge_raw_features.shape[1]
            if self.edge_raw_features is not None
            else self.embedding_dimension
        )
        cat_in = self.embedding_dimension * 2 + edge_dim
        hidden = max(32, cat_in)
        self.category_predictor = nn.Sequential(
            nn.Linear(cat_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_categories),
        )

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

        # Concatenate, do not sum: summing erases direction. Append the edge's
        # own features so the head can see what the flow actually is.
        edge_feat = self.edge_raw_features[edge_idxs]
        if edge_feat.device != source_node_embedding.device:
            edge_feat = edge_feat.to(source_node_embedding.device)
        combined_embeddings = torch.cat(
            [source_node_embedding, destination_node_embedding, edge_feat.float()], dim=1
        )
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
                torch.from_numpy(timestamps).double().to(self.device)
                - last_update[node_ids].double()
            )
            time_diffs = ((time_diffs - self.mean_time_shift_src) / self.std_time_shift_src).float()

        node_embeddings = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=node_ids,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )

        return node_embeddings

    # ------------------------------------------------------------------
    # Streaming memory (window-by-window extraction and live serving)
    #
    # Training (bita/train.py) drives memory through compute_temporal_embeddings
    # one batch at a time. Trajectory extraction and serving do not score
    # edges, they read host embeddings at the end of each 2 s window, so they
    # need the same causal protocol without the link-prediction machinery:
    #
    #   1. update_memory_for(nodes)   memory <- BiTA(messages stored EARLIER)
    #   2. get_host_embeddings(...)   read embeddings from that memory
    #   3. store_interactions(...)    queue this window's messages for later
    #
    # This is TGN's memory_update_at_start order (Rossi et al. 2020, and BiTA
    # Section "Causality"): a window's own interactions never reach the
    # memory that its own embedding is computed from.
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Forget all memory and pending messages (start of a new capture/session)."""
        if self.use_memory:
            self.memory.__init_memory__()

    def ensure_capacity(self, n_nodes: int) -> None:
        if self.use_memory:
            self.memory.ensure_capacity(n_nodes)

    @torch.no_grad()
    def update_memory_for(self, node_ids) -> None:
        if not self.use_memory or len(node_ids) == 0:
            return
        nodes = np.unique(np.asarray(node_ids, dtype=int))
        self.update_memory(nodes, self.memory.messages)
        self.memory.clear_messages(nodes)

    @torch.no_grad()
    def store_interactions(self, sources, destinations, timestamps, edge_idxs) -> None:
        if not self.use_memory or len(sources) == 0:
            return
        sources = np.asarray(sources, dtype=int)
        destinations = np.asarray(destinations, dtype=int)
        timestamps = np.asarray(timestamps, dtype=float)
        edge_idxs = np.asarray(edge_idxs, dtype=int)
        u_src, src_msgs = self.get_raw_messages(
            sources, sources, destinations, destinations, timestamps, edge_idxs)
        u_dst, dst_msgs = self.get_raw_messages(
            destinations, destinations, sources, sources, timestamps, edge_idxs)
        self.memory.store_raw_messages(u_src, src_msgs)
        self.memory.store_raw_messages(u_dst, dst_msgs)

    def get_global_state(self, host_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes pooled global network summary state ĥ_t = pool(H_t) ∈ R^d.
        """
        if host_embeddings.shape[0] == 0:
            return torch.zeros(self.embedding_dimension, device=self.device)
        return host_embeddings.mean(dim=0)
