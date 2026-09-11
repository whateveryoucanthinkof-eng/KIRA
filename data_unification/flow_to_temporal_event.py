"""
Flow to Temporal Event Stream Adapter.

Converts windowed UnifiedFlowRecord streams into TemporalEvent format expected by TGNE-TA.
Maintains persistent IP-to-Node-ID mappings, extracts 12-dimensional normalized edge features,
enforces non-decreasing causal timestamps, and partitions events into discrete time windows (e.g. 60s).
"""

import math
from typing import List, Dict, Tuple, Iterator, Optional
import numpy as np

from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory


class TemporalEventStream:
    """Encapsulates arrays and lookups for a batch or full stream of temporal graph events."""

    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_features: np.ndarray,
        edge_idxs: np.ndarray,
        labels: np.ndarray,
        is_attack: np.ndarray,
        ip_to_id: Dict[str, int],
        id_to_ip: Dict[int, str],
        window_boundaries: Optional[List[Tuple[float, float, int, int]]] = None,
    ):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_features = edge_features
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.is_attack = is_attack
        self.ip_to_id = ip_to_id
        self.id_to_ip = id_to_ip
        self.window_boundaries = window_boundaries or []
        self.n_interactions = len(sources)
        self.n_nodes = len(ip_to_id) + 1  # 0 is padding


class FlowToTemporalEventAdapter:
    """Converts UnifiedFlowRecord sequences into structured TemporalEventStream."""

    COARSE_TO_IDX = {
        CoarseCategory.BENIGN.value: 0,
        CoarseCategory.RECON.value: 1,
        CoarseCategory.INITIAL_ACCESS.value: 2,
        CoarseCategory.EXECUTION.value: 3,
        CoarseCategory.C2.value: 4,
        CoarseCategory.LATERAL_MOVEMENT.value: 5,
        CoarseCategory.EXFILTRATION.value: 6,
        CoarseCategory.IMPACT.value: 7,
        CoarseCategory.UNKNOWN.value: 0,
    }

    def __init__(
        self,
        window_size_sec: float = 60.0,
        initial_ip_map: Optional[Dict[str, int]] = None,
    ):
        self.window_size_sec = window_size_sec
        # Reserve node ID 0 for padding/unseen
        self.ip_to_id: Dict[str, int] = initial_ip_map.copy() if initial_ip_map else {}
        self.id_to_ip: Dict[int, str] = {v: k for k, v in self.ip_to_id.items()}
        self._next_id = max(self.ip_to_id.values(), default=0) + 1

    def get_or_create_node_id(self, ip: str) -> int:
        if ip not in self.ip_to_id:
            self.ip_to_id[ip] = self._next_id
            self.id_to_ip[self._next_id] = ip
            self._next_id += 1
        return self.ip_to_id[ip]

    def extract_edge_feature(self, record: UnifiedFlowRecord) -> np.ndarray:
        """
        Extracts 12-dimensional normalized feature vector for TGNE-TA edge input using canonical schema.
        """
        from data_unification.tgne_features import extract_flow_record_edge_features
        return extract_flow_record_edge_features(record)

    def process_records(
        self, records: List[UnifiedFlowRecord], sort_by_time: bool = True
    ) -> TemporalEventStream:
        """Processes a list of records into a validated TemporalEventStream."""
        if sort_by_time:
            records = sorted(records, key=lambda r: r.start_time)

        n = len(records)
        sources = np.zeros(n, dtype=int)
        destinations = np.zeros(n, dtype=int)
        timestamps = np.zeros(n, dtype=float)
        edge_features = np.zeros((n, 12), dtype=np.float32)
        edge_idxs = np.arange(n, dtype=int)
        labels = np.zeros(n, dtype=int)
        is_attack = np.zeros(n, dtype=bool)

        for i, r in enumerate(records):
            sources[i] = self.get_or_create_node_id(r.src_ip)
            destinations[i] = self.get_or_create_node_id(r.dst_ip)
            timestamps[i] = r.start_time
            edge_features[i] = self.extract_edge_feature(r)
            labels[i] = self.COARSE_TO_IDX.get(r.coarse_category, 0)
            is_attack[i] = r.is_attack

        # Validate non-decreasing timestamps for causal ordering without silent mutation
        for i in range(1, n):
            if timestamps[i] < timestamps[i - 1]:
                if sort_by_time:
                    raise ValueError(f"Non-monotonic timestamp detected after sorting at index {i}: {timestamps[i]} < {timestamps[i-1]}")
                else:
                    raise ValueError(f"Out-of-order timestamp at index {i} ({timestamps[i]} < {timestamps[i-1]}); pass sort_by_time=True or pre-sort telemetry.")

        # Compute 60s window boundaries: (win_start, win_end, start_idx, end_idx)
        window_boundaries = []
        if n > 0:
            first_t = timestamps[0]
            last_t = timestamps[-1]
            current_win_start = first_t
            start_idx = 0

            for i in range(n):
                if timestamps[i] >= current_win_start + self.window_size_sec:
                    window_boundaries.append(
                        (current_win_start, current_win_start + self.window_size_sec, start_idx, i)
                    )
                    current_win_start = current_win_start + self.window_size_sec
                    start_idx = i
            window_boundaries.append(
                (current_win_start, max(current_win_start + self.window_size_sec, last_t), start_idx, n)
            )

        return TemporalEventStream(
            sources=sources,
            destinations=destinations,
            timestamps=timestamps,
            edge_features=edge_features,
            edge_idxs=edge_idxs,
            labels=labels,
            is_attack=is_attack,
            ip_to_id=self.ip_to_id,
            id_to_ip=self.id_to_ip,
            window_boundaries=window_boundaries,
        )
