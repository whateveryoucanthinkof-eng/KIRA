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
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from data_unification.host_attributes import HOST_ATTRIBUTES

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
HOST_TEMPORAL_ATTR_NAMES: List[str] = HOST_ATTRIBUTES

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


#: Edge features to zero out, by name, for ablation studies. Set via
#: CYBERWORLD_ABLATE_EDGE_FEATURES as a comma-separated list of names from
#: EDGE_FEATURE_NAMES. Empty (the default) changes nothing.
#:
#: ## Why this exists
#:
#: A reviewer asked how these twelve features were chosen and what makes them
#: the right ones. The honest answer was that nothing in the repository
#: justified them -- they are conventional NetFlow summaries, with no ablation
#: and no importance analysis behind them.
#:
#: `dst_port_norm_65535` is the sharp end of that question. Several CIC-2018
#: attack classes sit on fixed destination ports, so a model can learn
#: "port => class" and score well without learning behaviour at all -- and
#: that collapses the moment an attacker changes port. Zeroing one feature and
#: re-measuring inductive AUC turns the question into a number.
#:
#: Zeroing rather than removing keeps the 12-D contract intact, so an ablated
#: encoder still loads everywhere a normal one does and the comparison is not
#: confounded by an architecture change.
def _ablation_mask() -> "np.ndarray | None":
    raw = os.environ.get("CYBERWORLD_ABLATE_EDGE_FEATURES", "").strip()
    if not raw:
        return None
    wanted = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in EDGE_FEATURE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown edge feature(s) to ablate: {unknown}. "
            f"Valid names: {EDGE_FEATURE_NAMES}")
    mask = np.ones(len(EDGE_FEATURE_NAMES), dtype=np.float32)
    for n in wanted:
        mask[EDGE_FEATURE_NAMES.index(n)] = 0.0
    return mask


_ABLATION_MASK = None
_ABLATION_READ = False


def ablation_mask():
    """Cached so the env var is read once, not once per flow record."""
    global _ABLATION_MASK, _ABLATION_READ
    if not _ABLATION_READ:
        _ABLATION_MASK = _ablation_mask()
        _ABLATION_READ = True
        if _ABLATION_MASK is not None:
            dropped = [n for n, m in zip(EDGE_FEATURE_NAMES, _ABLATION_MASK) if m == 0.0]
            print(f"EDGE FEATURE ABLATION ACTIVE: zeroing {dropped}", flush=True)
    return _ABLATION_MASK


def reset_ablation_cache():
    """For tests, which change the env var between cases."""
    global _ABLATION_MASK, _ABLATION_READ
    _ABLATION_MASK, _ABLATION_READ = None, False


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

    _m = ablation_mask()
    if _m is not None:
        feat *= _m
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




def build_scoped_host_id(
    dataset_source: str,
    scenario_id: str,
    host_ip: str,
    capture_id: Optional[str] = None,
) -> str:
    """
    Builds a scoped host identifier, e.g. 'cicids2017:Friday_DDoS:192.168.10.50'.

    Unused. The trainers scope hosts with TrajectoryStoreBuilder.set_namespace
    and trajectory_store.capture_namespace(path) instead, which is keyed by the
    capture file itself. If this is ever wired in, `scenario_id` must be unique
    per capture: the CIC-2018 adapter does not set one, and a shared value
    would merge the same address across days again.
    """
    clean_src = str(dataset_source).strip().lower()
    clean_scen = str(scenario_id).strip()
    clean_ip = str(host_ip).strip()
    if capture_id:
        return f"{clean_src}:{clean_scen}:{capture_id}:{clean_ip}"
    return f"{clean_src}:{clean_scen}:{clean_ip}"
