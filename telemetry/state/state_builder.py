#!/usr/bin/env python3
"""
telemetry/state/state_builder.py
Continuous 2.0-second capture window over the SPAN feed.

Emits the per-window 5-tuple flow snapshot and window metadata. That snapshot is the
sensor's entire contract with the backend: control_backend rebuilds host state from the
flows (TGNE-TA latent + host temporal attributes) and runs Dual-Branch/DeepOP inference
there. No feature vector is built here.
"""

import time
from typing import Dict, List, Optional, Any

from cyberworld_v4.config import get_contract
from telemetry.flow.flow_table import LiveFlowTable


class LiveStateBuilder:
    def __init__(self, window_sec: Optional[float] = None):
        # Default to the authoritative temporal contract, not a hardcoded
        # literal -- a caller that wants the served window size can simply
        # omit window_sec instead of importing/duplicating the constant.
        self.window_sec = window_sec if window_sec is not None else get_contract().window_seconds

        self.flow_table = LiveFlowTable()

        self.current_window_packets: List[Dict[str, Any]] = []
        self.window_start_time = time.time()
        self.window_id = 0

    def seek_to(self, ts: float):
        """
        Anchors the window clock to a capture timestamp.

        Live capture leaves this alone: packet timestamps and time.time() share the wall
        clock. PCAP replay must call it with the first packet's timestamp, which is
        historical -- otherwise `ts - window_start_time` is hugely negative and no window
        ever closes.
        """
        self.window_start_time = ts

    def ingest_packet(self, pkt: Dict[str, Any]):
        """Ingests a packet into both the flow table and the current window buffer."""
        self.current_window_packets.append(pkt)
        self.flow_table.process_packet(pkt)

    def is_window_ready(self, current_time: Optional[float] = None) -> bool:
        """Checks if the 2.0-second temporal boundary has elapsed."""
        now = current_time or time.time()
        return (now - self.window_start_time) >= self.window_sec

    def close_window(self, close_ts: Optional[float] = None) -> Dict[str, Any]:
        """Closes the current 2.0-second window and returns its flow snapshot."""
        t_close = close_ts or time.time()
        t_state_start = time.perf_counter()

        n_packets = len(self.current_window_packets)

        # Snapshot 5-tuples before aggregate extract (which may prune idle flows)
        flow_snapshot = self.flow_table.snapshot_flows(max_flows=256, with_flags=True)
        flow_feats = self.flow_table.extract_window_features(self.window_sec)

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
            'state_latency_ms': state_latency_ms,
            'pipeline_latency_ms': state_latency_ms,
        }
