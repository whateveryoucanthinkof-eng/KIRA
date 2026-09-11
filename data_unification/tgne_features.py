"""
data_unification/tgne_features.py
Canonical TGNE Feature Extraction & Authoritative Schema (Version 1.0.0).

Enforces strict train/inference representation parity across all pipeline components:
- bita/train.py
- data_unification/flow_to_temporal_event.py
- data_unification/multi_dataset_stream.py (HostTrajectoryExtractor)
- control_backend/model_adapter.py
- simulation/engine.py
- evaluation/scientific_benchmark.py

Zero Target Label Leakage:
Features are computed strictly from observable network telemetry (packet/byte volumes,
timing deltas, transport protocols, port numbers, flow asymmetry) and NEVER inspect
ground-truth labels (`is_attack`, `coarse_category`, `attck_technique_ids`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

SCHEMA_VERSION: str = "1.0.0"

# Authoritative 12-D Edge Feature Names
EDGE_FEATURE_NAMES: List[str] = [
    "log1p_fwd_bytes",
    "log1p_bwd_bytes",
    "log1p_fwd_packets",
    "log1p_bwd_packets",
    "duration_norm_300s",
    "log1p_byte_rate_norm",
    "log1p_packet_rate_norm",
    "is_tcp",
    "is_udp",
    "is_icmp",
    "dst_port_norm_65535",
    "directional_flow_asymmetry",
]

# Authoritative 15-D Temporal Host Attribute Names
HOST_TEMPORAL_ATTR_NAMES: List[str] = [
    "flow_rate_per_sec",
    "byte_rate_per_sec",
    "packet_rate_per_sec",
    "fwd_to_bwd_byte_ratio",
    "unique_dst_ports_norm_100",
    "tcp_flow_ratio",
    "udp_flow_ratio",
    "icmp_flow_ratio",
    "fan_out_degree_norm_50",
    "fan_in_degree_norm_50",
    "inter_arrival_time_mean_norm",
    "inter_arrival_time_std_norm",
    "flow_duration_mean_norm",
    "burstiness_index",
    "active_connection_entropy",
]

TOTAL_LATENT_DIM: int = 12
TOTAL_TEMPORAL_ATTR_DIM: int = 15
TOTAL_BRANCH_A_INPUT_DIM: int = TOTAL_LATENT_DIM + TOTAL_TEMPORAL_ATTR_DIM  # 27


@dataclass(frozen=True)
class TGNEFeatureSchema:
    """Authoritative metadata and contract for TGNE feature extraction."""
    schema_version: str = SCHEMA_VERSION
    edge_dim: int = TOTAL_LATENT_DIM
    temporal_attr_dim: int = TOTAL_TEMPORAL_ATTR_DIM
    total_branch_a_dim: int = TOTAL_BRANCH_A_INPUT_DIM
    edge_feature_names: List[str] = field(default_factory=lambda: list(EDGE_FEATURE_NAMES))
    host_temporal_names: List[str] = field(default_factory=lambda: list(HOST_TEMPORAL_ATTR_NAMES))
    edge_dtype: str = "float32"
    observable_only: bool = True
    zero_label_leakage: bool = True


SCHEMA: TGNEFeatureSchema = TGNEFeatureSchema()


def extract_canonical_edge_features(
    fwd_bytes: float,
    bwd_bytes: float,
    fwd_packets: float,
    bwd_packets: float,
    duration_sec: float,
    byte_rate: float,
    packet_rate: float,
    protocol: int,
    dst_port: int,
) -> np.ndarray:
    """
    Computes canonical 12-D edge feature vector from raw observable flow properties.
    Exact, deterministic formula applied uniformly across all datasets.
    """
    feat = np.zeros(12, dtype=np.float32)
    feat[0] = np.float32(np.log1p(max(0.0, float(fwd_bytes))))
    feat[1] = np.float32(np.log1p(max(0.0, float(bwd_bytes))))
    feat[2] = np.float32(np.log1p(max(0.0, float(fwd_packets))))
    feat[3] = np.float32(np.log1p(max(0.0, float(bwd_packets))))
    feat[4] = np.float32(min(max(0.0, float(duration_sec)), 300.0) / 300.0)
    feat[5] = np.float32(min(1.0, np.log1p(max(0.0, float(byte_rate))) / 20.0))
    feat[6] = np.float32(min(1.0, np.log1p(max(0.0, float(packet_rate))) / 10.0))
    feat[7] = 1.0 if int(protocol) == 6 else 0.0
    feat[8] = 1.0 if int(protocol) == 17 else 0.0
    feat[9] = 1.0 if int(protocol) == 1 else 0.0
    feat[10] = np.float32(min(65535, max(0, int(dst_port))) / 65535.0)

    # Feature 11: Directional flow asymmetry in [-1.0, 1.0]
    tot_bytes = max(0.0, float(fwd_bytes)) + max(0.0, float(bwd_bytes))
    feat[11] = np.float32((float(fwd_bytes) - float(bwd_bytes)) / (tot_bytes + 1e-5))
    return feat


def extract_flow_record_edge_features(record: Any) -> np.ndarray:
    """
    Adapter extracting canonical 12-D edge features from a UnifiedFlowRecord.
    """
    fwd_bytes = getattr(record, "fwd_bytes", 0.0)
    bwd_bytes = getattr(record, "bwd_bytes", 0.0)
    fwd_packets = getattr(record, "fwd_packets", 0.0)
    bwd_packets = getattr(record, "bwd_packets", 0.0)
    duration = getattr(record, "duration", 0.0)
    byte_rate = getattr(record, "byte_rate", 0.0)
    packet_rate = getattr(record, "packet_rate", 0.0)
    protocol = getattr(record, "protocol", 6)
    dst_port = getattr(record, "dst_port", 80)

    return extract_canonical_edge_features(
        fwd_bytes=fwd_bytes,
        bwd_bytes=bwd_bytes,
        fwd_packets=fwd_packets,
        bwd_packets=bwd_packets,
        duration_sec=duration,
        byte_rate=byte_rate,
        packet_rate=packet_rate,
        protocol=protocol,
        dst_port=dst_port,
    )


def extract_canonical_host_temporal_attributes(
    host_ip: str,
    window_records: List[Any],
    window_duration_sec: float = 2.0,
) -> np.ndarray:
    """
    Computes canonical 15-D host temporal behavioral attributes for a time window.
    """
    attrs = np.zeros(15, dtype=np.float32)
    host_flows = [r for r in window_records if getattr(r, "src_ip", None) == host_ip or getattr(r, "dst_ip", None) == host_ip]

    if not host_flows or window_duration_sec <= 0:
        return attrs

    n_flows = len(host_flows)
    total_bytes = sum(getattr(r, "fwd_bytes", 0) + getattr(r, "bwd_bytes", 0) for r in host_flows)
    total_packets = sum(getattr(r, "fwd_packets", 0) + getattr(r, "bwd_packets", 0) for r in host_flows)
    fwd_bytes = sum(getattr(r, "fwd_bytes", 0) for r in host_flows)
    bwd_bytes = sum(getattr(r, "bwd_bytes", 0) for r in host_flows)

    # 1-3. Rates
    attrs[0] = np.float32(min(1.0, (n_flows / window_duration_sec) / 50.0))
    attrs[1] = np.float32(min(1.0, np.log1p(total_bytes / window_duration_sec) / 15.0))
    attrs[2] = np.float32(min(1.0, np.log1p(total_packets / window_duration_sec) / 10.0))

    # 4. Fwd/Bwd byte ratio
    attrs[3] = np.float32(fwd_bytes / (bwd_bytes + 1e-4) if bwd_bytes > 0 else min(10.0, fwd_bytes / 1e-4) / 10.0)
    attrs[3] = np.float32(min(1.0, attrs[3] / 10.0))

    # 5. Unique destination ports
    dst_ports = set(getattr(r, "dst_port", 0) for r in host_flows if getattr(r, "src_ip", None) == host_ip)
    attrs[4] = np.float32(min(1.0, len(dst_ports) / 100.0))

    # 6-8. Protocol distributions
    tcp_count = sum(1 for r in host_flows if getattr(r, "protocol", 0) == 6)
    udp_count = sum(1 for r in host_flows if getattr(r, "protocol", 0) == 17)
    icmp_count = sum(1 for r in host_flows if getattr(r, "protocol", 0) == 1)
    attrs[5] = np.float32(tcp_count / n_flows)
    attrs[6] = np.float32(udp_count / n_flows)
    attrs[7] = np.float32(icmp_count / n_flows)

    # 9-10. Fan-out / Fan-in degrees
    out_peers = set(getattr(r, "dst_ip", None) for r in host_flows if getattr(r, "src_ip", None) == host_ip)
    in_peers = set(getattr(r, "src_ip", None) for r in host_flows if getattr(r, "dst_ip", None) == host_ip)
    attrs[8] = np.float32(min(1.0, len(out_peers) / 50.0))
    attrs[9] = np.float32(min(1.0, len(in_peers) / 50.0))

    # 11-12. Inter-arrival time statistics
    times = sorted([getattr(r, "start_time", 0.0) for r in host_flows])
    if len(times) > 1:
        deltas = np.diff(times)
        attrs[10] = np.float32(min(1.0, float(np.mean(deltas)) / max(0.1, window_duration_sec)))
        attrs[11] = np.float32(min(1.0, float(np.std(deltas)) / max(0.1, window_duration_sec)))

    # 13. Mean flow duration
    durations = [getattr(r, "duration", 0.0) for r in host_flows]
    attrs[12] = np.float32(min(1.0, float(np.mean(durations)) / 60.0)) if durations else 0.0

    # 14. Burstiness index (variance / mean of flow arrival count per quarter-window)
    q_dur = max(0.01, window_duration_sec / 4.0)
    if times:
        t0 = times[0]
        q_counts = [0, 0, 0, 0]
        for t in times:
            q_idx = min(3, max(0, int((t - t0) / q_dur)))
            q_counts[q_idx] += 1
        q_mean = np.mean(q_counts)
        q_std = np.std(q_counts)
        attrs[13] = np.float32(min(1.0, float(q_std / (q_mean + 1e-4))))

    # 15. Port connection entropy
    if dst_ports:
        port_counts = {}
        for r in host_flows:
            p = getattr(r, "dst_port", 0)
            port_counts[p] = port_counts.get(p, 0) + 1
        probs = [c / n_flows for c in port_counts.values()]
        entropy = -sum(p * np.log2(p + 1e-9) for p in probs)
        attrs[14] = np.float32(min(1.0, entropy / 6.0))

    return attrs


def build_scoped_host_id(
    dataset_source: str,
    scenario_id: str,
    host_ip: str,
    capture_id: Optional[str] = None,
) -> str:
    """
    Builds a globally unique, scoped host identifier to prevent IP collisions across datasets.
    Example: 'cicids2017:Friday_DDoS:192.168.10.50'
    """
    clean_src = str(dataset_source).strip().lower()
    clean_scen = str(scenario_id).strip()
    clean_ip = str(host_ip).strip()
    if capture_id:
        return f"{clean_src}:{clean_scen}:{capture_id}:{clean_ip}"
    return f"{clean_src}:{clean_scen}:{clean_ip}"
