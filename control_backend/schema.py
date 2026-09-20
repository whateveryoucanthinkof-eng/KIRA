"""
control_backend/schema.py
Pydantic data models for the Antigravity Predictive Attack-Trajectory Event Contract.
Decouples React dashboard from specific model architectures, dimensions, or checkpoints.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Current UTC time as the dashboard's wire format: ISO-8601 with a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelMetadata(BaseModel):
    name: str = Field(..., description="Model identifier, e.g. 'Antigravity-DualBranch-DeepOP'")
    version: str = Field(..., description="Model version string, e.g. '3.2-SOC'")
    feature_count: int = Field(..., description="Dimensionality of state vector, e.g. 27 (12 graph + 15 temporal)")
    history_steps: int = Field(..., description="Causal sequence length, e.g. 5")
    window_seconds: float = Field(..., description="Temporal window duration in seconds, e.g. 2.0")
    forecast_steps: int = Field(..., description="Number of forward prediction steps, e.g. 8")
    checkpoint: Optional[str] = Field(None, description="Model checkpoint file name")
    threshold: float = Field(..., description="Calibrated operating alert threshold, e.g. 0.40")


class StateMetadata(BaseModel):
    window_id: int
    sequence_ready: bool
    packet_count: int
    active_flows: Optional[int] = None
    pipeline_latency_ms: float
    buffer_length: int


class ForecastPoint(BaseModel):
    horizon_seconds: float
    risk: float
    confidence: Optional[float] = None
    predicted_stage: Optional[str] = None


class ExplainabilityGroup(BaseModel):
    name: str
    percentage: float


class ExplainabilityFeature(BaseModel):
    feature: str
    score: float
    group: str


class ExplainabilityPayload(BaseModel):
    available: bool = False
    method: Optional[str] = None
    groups: List[ExplainabilityGroup] = []
    top_features: List[ExplainabilityFeature] = []


class PredictionData(BaseModel):
    risk: float
    max_future_risk: float
    predicted_risk_prior: Optional[float] = None
    forecast_error: Optional[float] = None
    hazard_score: Optional[float] = None
    malicious_confidence: Optional[float] = None
    precursor_confidence: Optional[float] = None
    alert: bool
    alert_level: str  # NOMINAL, WARNING, ELEVATED, CRITICAL
    threshold: float
    predicted_stage: str
    mitre_tactic: Optional[str] = None
    mitre_technique: Optional[str] = None
    mitre_tactic_id: Optional[str] = None
    mitre_description: Optional[str] = None
    stage_probabilities: Optional[Dict[str, float]] = None
    technique_confidence: Optional[float] = None
    stage_provenance: Optional[Dict[str, str]] = None

    # Provenance of the displayed risk (spec 21, 41). The SOC layer may blend
    # model output with deterministic rules, but the three numbers must be
    # separable or a rule-driven demo can be mistaken for a model result.
    ml_risk: Optional[float] = None          # model output, untouched
    rule_risk: Optional[float] = None        # deterministic SOC rules alone
    rules_applied: Optional[bool] = None     # whether `risk` was rule-adjusted


class LatencyData(BaseModel):
    telemetry_ms: float
    inference_ms: float
    total_ms: float


class EarlyWarningData(BaseModel):
    is_alert: bool
    alert_timestamp: Optional[float] = None
    actual_milestone_timestamp: Optional[float] = None
    lead_time_seconds: Optional[float] = None
    target_milestone_desc: Optional[str] = None


class FocusEdge(BaseModel):
    src: str
    dst: str


class PredictionEvent(BaseModel):
    type: str = "prediction"
    mode: str = "LIVE"  # "STANDBY" | "LIVE"
    timestamp: str
    wall_clock: str
    model: ModelMetadata
    state: StateMetadata
    prediction: PredictionData
    forecast: List[ForecastPoint]
    explainability: ExplainabilityPayload
    latency: LatencyData
    early_warning: Optional[EarlyWarningData] = None
    attack_active: bool = False
    attack_phase: Optional[str] = None
    # Discovery binding — highlight observed IPs/edges, never fixed lab node IDs
    focus_ips: List[str] = Field(default_factory=list)
    focus_edges: List[FocusEdge] = Field(default_factory=list)
    target_ip: Optional[str] = None
    throughput: float = 0.0


class TopologyNode(BaseModel):
    id: str
    ip: str
    role: str  # internal | external | sensor | unknown
    zone: str = "enterprise"
    label: Optional[str] = None
    bytes_in: int = 0
    bytes_out: int = 0
    last_seen: float = 0.0
    risk: float = 0.0
    stale: bool = False


class TopologyEdge(BaseModel):
    src: str
    dst: str
    protocol: int = 0
    dst_port: int = 0
    bytes: int = 0
    packets: int = 0
    last_seen: float = 0.0


class TopologyStats(BaseModel):
    nodes: int = 0
    edges: int = 0
    external_nodes: int = 0
    windows_applied: int = 0


class TopologyEvent(BaseModel):
    type: str = "topology_update"
    site_id: str
    generated_at: float
    nodes: List[TopologyNode] = Field(default_factory=list)
    edges: List[TopologyEdge] = Field(default_factory=list)
    stats: TopologyStats = Field(default_factory=TopologyStats)


class SystemStatusEvent(BaseModel):
    type: str = "system_status"
    mode: str = "STANDBY"  # "LIVE" or "STANDBY"

    # Authoritative 5 independent runtime states
    network: str = "stopped"         # "stopped" | "starting" | "running" | "stopping"
    sensor: str = "stopped"          # "stopped" | "starting" | "running" | "stopping"
    normal_traffic: str = "stopped"  # "stopped" | "running"
    attack: str = "stopped"          # "stopped" | "running"
    ml: str = "stopped"              # "stopped" | "starting" | "running" | "stopping"

    # Backward-compatible flags
    network_online: bool
    nodes_running: int
    total_nodes: int
    sensor_active: bool
    telemetry_active: bool
    ml_active: bool = False
    ml_status: str = "standby"  # "live" or "standby"
    workloads_active: bool
    attack_active: bool
    demo_active: bool = False
    active_command: Optional[str] = None
    model_loaded: bool
    model_meta: Optional[ModelMetadata] = None
    # Site / discovery (Phase 4+)
    lab_mode: bool = False
    site_id: Optional[str] = None
    topology_nodes: int = 0
    topology_edges: int = 0
    sensor_interface: Optional[str] = None
    # Live operational reality metrics
    uptime: int = 0
    throughput: float = 0.0
    latency: float = 0.0
    packetLoss: float = 0.0
    activeConnections: int = 0
    anomalyScore: float = 0.0
    threatLevel: str = "low"
    timestamp: Optional[str] = None


class CommandEvent(BaseModel):
    type: str  # "command_started", "command_output", "command_completed"
    command: str
    timestamp: str
    line: Optional[str] = None
    exit_code: Optional[int] = None
    success: Optional[bool] = None


class AttackEvent(BaseModel):
    type: str = "attack_event"
    stage: str
    timestamp: float
    details: str
