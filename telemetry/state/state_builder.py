#!/usr/bin/env python3
"""
telemetry/state/state_builder.py
Continuous 2.0-second capture window and in-memory sequence buffer.
Builds packet and flow features for the capture-only telemetry stream.
Dual-Branch/DeepOP inference runs in control_backend, not in this module.
"""

from collections import deque
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from telemetry.flow.flow_table import LiveFlowTable, FLOW_COLUMNS
from telemetry.packet.pcap_engine import LivePCAPEngine, PCAP_BEHAVIORAL_COLUMNS

# Capture feature contract (42 flow + 30 packet-behavioral features).
CANONICAL_72_COLUMNS = FLOW_COLUMNS + PCAP_BEHAVIORAL_COLUMNS

FEATURE_NAMES = CANONICAL_72_COLUMNS


class LiveStateBuilder:
    def __init__(
        self,
        window_sec: float = 2.0,
        history_len: int = 15,
        dim_mode: int = 72,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None
    ):
        self.window_sec = window_sec
        self.history_len = history_len
        self.dim_mode = dim_mode
        self.feature_columns = CANONICAL_72_COLUMNS

        self.flow_table = LiveFlowTable()
        self.pcap_engine = LivePCAPEngine()

        self.current_window_packets: List[Dict[str, Any]] = []
        self.window_start_time = time.time()
        self.window_id = 0

        # In-Memory Rolling Sequence Buffer: shape [L, D]
        self.sequence_buffer: deque = deque(maxlen=history_len)

        # Preprocessing parameters (StandardScaler from frozen V3.1 checkpoint)
        self.scaler_mean = scaler_mean if scaler_mean is not None else np.zeros(len(self.feature_columns), dtype=np.float32)
        self.scaler_scale = scaler_scale if scaler_scale is not None else np.ones(len(self.feature_columns), dtype=np.float32)

    def ingest_packet(self, pkt: Dict[str, Any]):
        """Ingests a packet into both the flow table and the current window buffer."""
        self.current_window_packets.append(pkt)
        self.flow_table.process_packet(pkt)

    def is_window_ready(self, current_time: Optional[float] = None) -> bool:
        """Checks if the 2.0-second temporal boundary has elapsed."""
        now = current_time or time.time()
        return (now - self.window_start_time) >= self.window_sec

    def close_window(self, close_ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Closes current 2.0-second window, extracts flow & PCAP features,
        applies log1p & scaling, enqueues to sequence buffer, and records latency.
        """
        t_close = close_ts or time.time()
        t_state_start = time.perf_counter()

        packets = self.current_window_packets
        n_packets = len(packets)

        # Snapshot 5-tuples before aggregate extract (which may prune idle flows)
        flow_snapshot = self.flow_table.snapshot_flows(max_flows=256)

        # 1. In-memory Flow features (42)
        flow_feats = self.flow_table.extract_window_features(self.window_sec)

        # 2. In-memory PCAP behavioral features (30)
        pcap_feats = self.pcap_engine.extract_features(packets, self.window_sec)

        raw_vector = []
        for col in FLOW_COLUMNS:
            raw_vector.append(float(flow_feats.get(col, 0.0)))
        for col in PCAP_BEHAVIORAL_COLUMNS:
            raw_vector.append(float(pcap_feats.get(col, 0.0)))

        raw_np = np.array(raw_vector, dtype=np.float32)

        # 3. Preprocessing: log1p + StandardScaler z-score normalization
        # Formula: X_scaled = (log1p(clip(X_raw, 0)) - mean) / (scale + 1e-7)
        log_np = np.log1p(np.clip(raw_np, 0.0, None))
        scaled_np = (log_np - self.scaler_mean) / (self.scaler_scale + 1e-7)
        scaled_np = np.nan_to_num(scaled_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 4. Enqueue into rolling sequence buffer
        self.sequence_buffer.append(scaled_np)
        self.window_id += 1

        t_state_end = time.perf_counter()
        state_latency_ms = (t_state_end - t_state_start) * 1000.0

        # Reset window state
        self.current_window_packets = []
        self.window_start_time = t_close

        return {
            'window_id': self.window_id,
            'timestamp': t_close,
            'window_end': t_close,
            'packet_count': n_packets,
            'active_flows': flow_feats.get('active_flows_count', 0),
            'flows': flow_snapshot,
            'raw_state': raw_np,
            'scaled_state': scaled_np,
            'is_buffer_full': len(self.sequence_buffer) == self.history_len,
            'buffer_len': len(self.sequence_buffer),
            'buffer_length': len(self.sequence_buffer),
            'state_latency_ms': state_latency_ms,
            'pipeline_latency_ms': state_latency_ms,
            'feature_dim': len(self.feature_columns)
        }

    def get_current_sequence_tensor(self) -> Optional[np.ndarray]:
        """
        Returns sequence tensor of shape [1, L, D] if buffer is full, else None.
        Zero disk I/O, pure in-memory slice.
        """
        if len(self.sequence_buffer) < self.history_len:
            return None
        seq_array = np.array(self.sequence_buffer, dtype=np.float32)
        return np.expand_dims(seq_array, axis=0)  # Shape [1, 15, 72]

    def get_causal_sequence_tensor(self) -> Optional[np.ndarray]:
        """Alias for get_current_sequence_tensor."""
        return self.get_current_sequence_tensor()

