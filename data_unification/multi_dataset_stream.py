"""
Multi-Dataset Stream & Host Trajectory Extractor.

Aggregates flows from CIC-IDS2017, CTU-13, and Warden into unified time-windowed graph snapshots,
runs TGNE-TA to compute per-host embeddings H_t[v], computes the 15 per-host window temporal attributes,
and produces structured host sequence dictionaries.
"""

import os
import collections
from typing import List, Dict, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import pandas as pd

from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory
from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter, TemporalEventStream
from data_unification.cic2017_adapter import CIC2017Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.warden_adapter import WardenAdapter
from data_unification.auth_log_adapter import AuthEventRecord, AuthLogAdapter, AuthEventType, AuthLogSource
from data_unification.behavioral_fingerprint import BehavioralFlowFingerprinter, BehavioralProfile
from data_unification.trajectory_store import TrajectoryStore, TrajectoryStoreBuilder
from dataclasses import field


@dataclass(slots=True)
class HostWindowSnapshot:
    """Per-host state snapshot for a single time window t.

    slots=True and the absence of the old auth_metrics/behavioral_metrics dicts
    are load-bearing, not style: a full-corpus run holds ~25M of these at once,
    where the per-instance __dict__ plus two populated metric dicts cost ~305
    bytes each (~7.6 GB measured). Those two dicts were written here but never
    read by any consumer -- they are consumed locally in extract_trajectories to
    adjust risk_score before the snapshot is built, and sequence_dataset /
    train_branch_b / train_cwa_decoder only ever touch embedding, temporal_attrs,
    is_attack, coarse_category, technique_ids, window_idx, window_end and
    risk_score. Adding a field here is not free; check the memory budget first.
    """
    host_ip: str
    host_id: int
    window_idx: int
    window_start: float
    window_end: float
    embedding: np.ndarray  # H_t[v] in R^d (d=12 from TGNE-TA)
    temporal_attrs: np.ndarray  # 15 temporal scalar attributes
    is_attack: bool
    coarse_category: str
    technique_ids: List[str]
    risk_score: float  # Ground truth compromise/risk score in [0, 1]


class BoundedHostSlidingBuffer:
    """
    Bounded sliding window buffer for per-host state snapshots.
    
    Prevents Python heap memory bloat and garbage collection latency spikes
    by enforcing strict O(active_hosts * max_windows) memory bounds with O(1) eviction
    and idle endpoint pruning.
    """

    def __init__(self, max_windows_per_host: int = 16, max_idle_ttl_sec: float = 3600.0):
        self.max_windows = max_windows_per_host
        self.max_idle_ttl_sec = max_idle_ttl_sec
        self._buffers: Dict[str, collections.deque] = {}
        self._last_active_time: Dict[str, float] = {}

    def append(self, snapshot: HostWindowSnapshot):
        """Appends a snapshot, automatically evicting oldest window beyond max_windows in O(1)."""
        hip = snapshot.host_ip
        if hip not in self._buffers:
            self._buffers[hip] = collections.deque(maxlen=self.max_windows)
        self._buffers[hip].append(snapshot)
        self._last_active_time[hip] = max(self._last_active_time.get(hip, 0.0), snapshot.window_end)

    def extend_host(self, host_ip: str, snapshots: List[HostWindowSnapshot]):
        """Appends multiple snapshots for a host."""
        for s in snapshots:
            self.append(s)

    def get_host_trajectory(self, host_ip: str) -> List[HostWindowSnapshot]:
        """Retrieves current sliding window history for a host."""
        return list(self._buffers.get(host_ip, []))

    def get_all_trajectories(self) -> Dict[str, List[HostWindowSnapshot]]:
        """Returns snapshot lists for all currently tracked hosts."""
        return {hip: list(dq) for hip, dq in self._buffers.items()}

    def prune_stale_hosts(self, current_time: float, max_idle_sec: Optional[float] = None) -> int:
        """Evicts endpoints that have been idle past max_idle_sec."""
        ttl = max_idle_sec or self.max_idle_ttl_sec
        stale_ips = [
            hip for hip, last_ts in self._last_active_time.items()
            if (current_time - last_ts) > ttl
        ]
        for hip in stale_ips:
            self._buffers.pop(hip, None)
            self._last_active_time.pop(hip, None)
        return len(stale_ips)

    def get_memory_stats(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Reports active buffer memory utilization and capacity metrics."""
        total_snaps = sum(len(dq) for dq in self._buffers.values())
        tracked_hosts = len(self._buffers)
        # Approximate footprint: ~320 bytes per HostWindowSnapshot numpy array reference
        est_kb = (total_snaps * 0.32) + (tracked_hosts * 0.1)
        active_10m = 0
        if current_time is not None:
            active_10m = sum(1 for ts in self._last_active_time.values() if (current_time - ts) <= 600.0)

        return {
            "tracked_hosts": tracked_hosts,
            "total_snapshots": total_snaps,
            "max_windows_per_host": self.max_windows,
            "estimated_memory_kb": round(est_kb, 2),
            "active_hosts_last_10m": active_10m,
        }

    def __len__(self) -> int:
        return len(self._buffers)

    def __contains__(self, host_ip: str) -> bool:
        return host_ip in self._buffers


class HostTrajectoryExtractor:
    """Extracts per-host embedding and temporal attribute trajectories across time windows."""

    def __init__(
        self,
        tgne_ta_model,
        window_size_sec: float = 60.0,
        n_temporal_attrs: int = 15,
        auth_events: Optional[List[AuthEventRecord]] = None,
        spill_dir: Optional[str] = None,
    ):
        self.tgn = tgne_ta_model
        self.window_size_sec = window_size_sec
        self.n_temporal_attrs = n_temporal_attrs
        # When set, the bulk (N, 27) feature block is written here and mapped
        # back read-only, so a full-density corpus does not have to fit in RAM.
        self.spill_dir = spill_dir
        self.auth_events: List[AuthEventRecord] = list(auth_events) if auth_events else []

    def add_auth_events(self, events: List[AuthEventRecord]):
        """Ingests additional host authentication security events."""
        self.auth_events.extend(events)

    def compute_host_temporal_attributes(
        self,
        host_ip: str,
        window_records: List[UnifiedFlowRecord],
        window_duration: float,
    ) -> np.ndarray:
        """
        Computes 15 per-host temporal scalar attributes for window t:
        0: flow_count
        1: log1p(fwd_bytes)
        2: log1p(bwd_bytes)
        3: log1p(total_bytes)
        4: log1p(fwd_packets)
        5: log1p(bwd_packets)
        6: log1p(total_packets)
        7: unique_peers_count
        8: unique_dst_ports_count
        9: tcp_ratio
        10: udp_ratio
        11: avg_duration
        12: byte_rate (normalized)
        13: packet_rate (normalized)
        14: active_connection_density
        """
        attrs = np.zeros(self.n_temporal_attrs, dtype=np.float32)
        n = len(window_records)
        if n == 0:
            return attrs

        fwd_b = sum(r.fwd_bytes for r in window_records)
        bwd_b = sum(r.bwd_bytes for r in window_records)
        tot_b = fwd_b + bwd_b
        fwd_p = sum(r.fwd_packets for r in window_records)
        bwd_p = sum(r.bwd_packets for r in window_records)
        tot_p = fwd_p + bwd_p

        peers = set()
        ports = set()
        tcp_count = 0
        udp_count = 0
        tot_dur = 0.0

        for r in window_records:
            peer = r.dst_ip if r.src_ip == host_ip else r.src_ip
            peers.add(peer)
            ports.add(r.dst_port)
            if r.protocol == 6:
                tcp_count += 1
            elif r.protocol == 17:
                udp_count += 1
            tot_dur += r.duration

        dur = max(1.0, window_duration)
        attrs[0] = min(1.0, np.log1p(n) / 10.0)
        attrs[1] = min(1.0, np.log1p(fwd_b) / 20.0)
        attrs[2] = min(1.0, np.log1p(bwd_b) / 20.0)
        attrs[3] = min(1.0, np.log1p(tot_b) / 20.0)
        attrs[4] = min(1.0, np.log1p(fwd_p) / 10.0)
        attrs[5] = min(1.0, np.log1p(bwd_p) / 10.0)
        attrs[6] = min(1.0, np.log1p(tot_p) / 10.0)
        attrs[7] = min(1.0, np.log1p(len(peers)) / 5.0)
        attrs[8] = min(1.0, np.log1p(len(ports)) / 5.0)
        attrs[9] = float(tcp_count) / n
        attrs[10] = float(udp_count) / n
        attrs[11] = min(1.0, (tot_dur / n) / 300.0)
        attrs[12] = min(1.0, np.log1p(tot_b / dur) / 15.0)
        attrs[13] = min(1.0, np.log1p(tot_p / dur) / 10.0)
        attrs[14] = min(1.0, float(len(peers)) / max(1, n))

        return attrs

    def extract_trajectories(
        self,
        records: List[UnifiedFlowRecord],
        adapter: Optional[FlowToTemporalEventAdapter] = None,
        auth_events: Optional[List[AuthEventRecord]] = None,
        sliding_buffer: Optional["BoundedHostSlidingBuffer"] = None,
        builder: Optional[TrajectoryStoreBuilder] = None,
        emit_after: Optional[float] = None,
        window_idx_base: int = 0,
    ) -> "TrajectoryStore | Dict[str, List[HostWindowSnapshot]]":
        """
        Groups flows by 60s windows, extracts TGNE-TA embeddings H_t,
        correlates multimodal host authentication logs (breaking L4 visibility ceiling),
        and constructs per-host timelines using either bounded circular buffers or unbounded lists.
        """
        adapter = adapter or FlowToTemporalEventAdapter(window_size_sec=self.window_size_sec)
        event_stream = adapter.process_records(records, sort_by_time=True)

        all_auth_events: List[AuthEventRecord] = self.auth_events + (list(auth_events) if auth_events else [])

        if hasattr(self.tgn, "embedding_module") and len(event_stream.sources) > 0:
            from utils.utils import NeighborFinder
            n_nodes = max(event_stream.n_nodes + 10, getattr(self.tgn, "n_nodes", 0))
            adj_list = [[] for _ in range(n_nodes)]
            for i in range(len(event_stream.sources)):
                u = int(event_stream.sources[i])
                v = int(event_stream.destinations[i])
                t = float(event_stream.timestamps[i])
                adj_list[u].append((v, i, t))
                adj_list[v].append((u, i, t))
            nf = NeighborFinder(adj_list, uniform=False)
            self.tgn.neighbor_finder = nf
            self.tgn.embedding_module.neighbor_finder = nf
            edge_feats_t = torch.from_numpy(event_stream.edge_features).float().to(self.tgn.device)
            self.tgn.edge_raw_features = edge_feats_t
            self.tgn.embedding_module.edge_features = edge_feats_t
            # Intrinsic IP node features, matching TRAINING exactly.
            #
            # This was torch.zeros(...). The encoder is trained with 12-D
            # features derived from each host's address (see
            # data_unification/ip_features.py), and handing it zeros at
            # inference is a train/serve mismatch that silently destroys the
            # inductive capability those features exist to provide -- the
            # difference between inductive AUC 0.83 and 0.50.
            from data_unification.ip_features import build_node_feature_matrix
            node_feats_t = torch.from_numpy(
                build_node_feature_matrix(event_stream.ip_to_id, n_nodes=n_nodes)
            ).float().to(self.tgn.device)
            self.tgn.node_raw_features = node_feats_t
            self.tgn.embedding_module.node_features = node_feats_t
            self.tgn.n_nodes = n_nodes

        # Map window boundaries to lists of records
        # `builder` lets a caller accumulate across several chunked calls, so a
        # full-density corpus can be processed a slice at a time while the
        # records for each slice are freed. `emit_after` drops snapshots from a
        # chunk's warm-up overlap -- those windows were already emitted by the
        # previous chunk, and the overlap exists only so TGNE sees the same
        # neighbour history it would have seen processing everything at once.
        owns_builder = builder is None
        if builder is None:
            builder = TrajectoryStoreBuilder(spill_dir=self.spill_dir)
        sorted_records = sorted(records, key=lambda r: r.start_time)

        for win_idx, (win_start, win_end, s_idx, e_idx) in enumerate(event_stream.window_boundaries):
            if s_idx >= e_idx:
                continue

            win_recs = sorted_records[s_idx:e_idx]
            # Identify active hosts in this window
            active_ips = set()
            for r in win_recs:
                active_ips.add(r.src_ip)
                active_ips.add(r.dst_ip)

            # Also check if any hosts had auth events in this window
            for aev in all_auth_events:
                if win_start <= aev.timestamp <= win_end:
                    active_ips.add(aev.host_ip)

            active_ips_sorted = sorted(list(active_ips))
            active_host_ids = np.array([adapter.ip_to_id.get(ip, 0) for ip in active_ips_sorted], dtype=int)

            # Compute TGNE-TA embeddings H_t
            with torch.no_grad():
                H_t = self.tgn.get_host_embeddings(
                    active_host_ids, timestamp=win_end, n_neighbors=10
                ).cpu().numpy()

            # For each active host, compute temporal attributes, auth indicators & labels
            for idx, ip in enumerate(active_ips_sorted):
                host_recs = [r for r in win_recs if r.src_ip == ip or r.dst_ip == ip]
                attrs = self.compute_host_temporal_attributes(
                    ip, host_recs, self.window_size_sec
                )

                # Determine if host was victim/target of attack or attacking
                atk_recs = [r for r in host_recs if r.is_attack]
                is_atk = len(atk_recs) > 0
                coarse = atk_recs[0].coarse_category if is_atk else "Benign"
                techs = list(atk_recs[0].attck_technique_ids) if is_atk else []
                # Risk score: Grounded in MITRE ATT&CK tactic progression & volume intensity
                TACTIC_BASE_SEVERITY = {
                    "Benign": 0.0,
                    "Recon": 0.35,
                    "Reconnaissance": 0.35,
                    "Discovery": 0.38,
                    "InitialAccess": 0.60,
                    "CredentialAccess": 0.65,
                    "Execution": 0.72,
                    "Persistence": 0.75,
                    "PrivilegeEscalation": 0.78,
                    "DefenseEvasion": 0.75,
                    "C2": 0.82,
                    "CommandAndControl": 0.82,
                    "LateralMovement": 0.85,
                    "Exfiltration": 0.92,
                    "Impact": 0.96,
                }
                if is_atk:
                    base_sev = TACTIC_BASE_SEVERITY.get(coarse, 0.50)
                    atk_density = min(1.0, len(atk_recs) / max(1, len(host_recs)))
                    vol_scale = min(1.0, float(np.log1p(len(atk_recs)) / 5.0))
                    risk = min(1.0, max(0.20, base_sev + 0.04 * atk_density + 0.04 * vol_scale))
                else:
                    risk = 0.0

                # Extract and correlate multimodal authentication metrics
                auth_metrics = {}
                if all_auth_events:
                    auth_metrics = AuthLogAdapter.extract_window_auth_metrics(
                        all_auth_events, ip, win_start, win_end
                    )
                    # Break the L4 visibility ceiling: detect credential access / password spraying
                    if auth_metrics.get("is_brute_force_flag", 0.0) == 1.0 or auth_metrics.get("auth_failed_count", 0.0) >= 5:
                        is_atk = True
                        coarse = "CredentialAccess"
                        if "T1110" not in techs:
                            techs = ["T1110"] + [t for t in techs if t != "T1110"]
                        brute_score = auth_metrics.get("auth_brute_force_score", 0.5)
                        risk = max(risk, min(1.0, 0.75 + 0.25 * brute_score))

                # Compute behavioral flow size & protocol profile metrics (independent of static ports)
                behavioral_metrics = BehavioralFlowFingerprinter.compute_host_behavioral_metrics(ip, host_recs)
                if behavioral_metrics.get("is_shell_detected", 0.0) == 1.0 or behavioral_metrics.get("port_mismatch_count", 0.0) >= 2:
                    is_atk = True
                    if coarse == "Benign":
                        coarse = "Execution"
                    if "T1059" not in techs:
                        techs = ["T1059"] + techs
                    risk = max(risk, min(1.0, 0.65 + 0.35 * behavioral_metrics.get("behavioral_threat_score", 0.5)))

                if sliding_buffer is not None:
                    sliding_buffer.append(HostWindowSnapshot(
                        host_ip=ip,
                        host_id=adapter.ip_to_id.get(ip, 0),
                        window_idx=win_idx,
                        window_start=win_start,
                        window_end=win_end,
                        embedding=H_t[idx],
                        temporal_attrs=attrs,
                        is_attack=is_atk,
                        coarse_category=coarse,
                        technique_ids=techs,
                        risk_score=float(risk),
                    ))
                elif emit_after is None or win_start >= emit_after:
                    builder.append(
                        host_ip=ip,
                        host_id=adapter.ip_to_id.get(ip, 0),
                        window_idx=win_idx + window_idx_base,
                        window_start=win_start,
                        window_end=win_end,
                        embedding=H_t[idx],
                        temporal_attrs=attrs,
                        is_attack=is_atk,
                        coarse_category=coarse,
                        technique_ids=techs,
                        risk_score=float(risk),
                    )

        if sliding_buffer is not None:
            return sliding_buffer.get_all_trajectories()
        return builder.finalize() if owns_builder else builder
