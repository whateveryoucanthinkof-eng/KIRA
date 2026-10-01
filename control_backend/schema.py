"""
control_backend/schema.py
Pydantic data models for the Antigravity Predictive Attack-Trajectory Event Contract.
Decouples React dashboard from specific model architectures, dimensions, or checkpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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
    forecast_step_seconds: Optional[float] = Field(
        None, description="Seconds per forecast step (may differ from window_seconds)")
    rules_enabled: bool = Field(
        False, description="Whether the advisory SOC rule layer is computed. It never "
                            "changes `risk`, `predicted_stage` or `alert`.")


class TemporalContractInfo(BaseModel):
    window_seconds: float
    history_steps: int
    forecast_steps: int
    forecast_step_seconds: float
    # "checkpoints" when read from the loaded models, "config" otherwise.
    source: str


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
    # Split-conformal band on `risk` for this step, fitted on validation
    # (control_backend/forecast_band.py). None when the checkpoint has none.
    risk_lower: Optional[float] = None
    risk_upper: Optional[float] = None
    # Kill-chain lane of predicted_stage (control_backend/tactics.py), so the
    # console can place the step without parsing the token.
    tactic_lane: Optional[str] = None


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
    # Kill-chain lane of predicted_stage (control_backend/tactics.py).
    tactic_lane: Optional[str] = None

    # Provenance (spec 21, 41). `risk`, `predicted_stage` and `alert` are always
    # the model's output. The optional SOC rule layer is advisory: its opinion
    # is reported here, next to the model's, and never replaces it.
    ml_risk: Optional[float] = None          # model output (== risk)
    ml_technique: Optional[str] = None       # model technique (== predicted_stage)
    rule_risk: Optional[float] = None        # advisory rule layer, None when off/not fired
    rule_technique: Optional[str] = None     # advisory rule label, None when off/not fired
    rules_applied: Optional[bool] = None     # always False: rules never adjust `risk`
    risk_source: str = "model"               # what produced `risk`; only "model" today

    # Mitigation is recorded by the dashboard, not enforced by it. The model
    # keeps scoring the traffic it actually sees; these say whether traffic
    # that a recorded block/isolation should have stopped is still present.
    mitigation_status: Optional[str] = None  # None | "recorded_quiet" | "traffic_persists"
    mitigation_bypass_flows: Optional[int] = None


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


class StateDim(BaseModel):
    """One of the 27 dimensions Branch A read for this window."""
    index: int
    feature: str          # H_emb_0..11, then the 15 host attributes
    group: str            # FEATURE_GROUP_MAP
    value: float
    attribution: float    # share of |input x gradient|, sums to 1 over the 27


class BranchStep(BaseModel):
    horizon_seconds: float
    tactic_lane: str
    technique: Optional[str] = None
    probability: float    # DeepOP's probability for this step's token


class ForecastBranch(BaseModel):
    """One of DeepOP's three most likely continuations (forecast_branches.py).

    Ranked by the probability of the first forecast step's token; each is then
    decoded greedily, exactly as the main forecast is. Fields no model
    produces yet are None and the console marks them under development.
    """
    id: str                       # "A" | "B" | "C"
    kind: str                     # escalation | pivot | backoff
    label: str
    stage: str                    # kill-chain lane the branch reaches
    technique: str                # ATT&CK id, or "—" for a back-off
    probability: float            # share of the top-3 mass; the three sum to 1
    probability_raw: float        # the first step's actual probability
    confidence: float             # DeepOP's probability for the technique's step
    horizon_seconds: float
    path: List[BranchStep] = Field(default_factory=list)
    hops: Optional[List[Dict[str, str]]] = None   # not predicted: models are per-host
    packets: Optional[float] = None               # not predicted
    bytes: Optional[float] = None                 # not predicted
    peak_risk: Optional[float] = None             # not predicted per branch


class FlowFlags(BaseModel):
    syn: int = 0
    ack: int = 0
    psh: int = 0
    rst: int = 0
    fin: int = 0
    urg: int = 0


class FlowRecordOut(BaseModel):
    """A flow the sensor exported for this window, as the console lists it."""
    id: str
    window: int
    ts_us: int
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str         # TCP | UDP | ICMP | OTHER
    fwd_bytes: int
    bwd_bytes: int
    fwd_packets: int
    bwd_packets: int
    duration_ms: float
    flags: FlowFlags
    direction: str        # inbound | outbound | internal (site CIDRs)
    on_path: bool         # touches the host scored this window


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

    # Evidence and structure behind the verdict. Each is None when the
    # pipeline did not produce it for this window.
    state_vector: Optional[List[StateDim]] = None
    branches: Optional[List[ForecastBranch]] = None
    flows: Optional[List[FlowRecordOut]] = None
    flows_in_window: Optional[int] = None
    attention: Optional[Dict[str, Any]] = None     # attention_probe.py
    campaign: Optional[Dict[str, Any]] = None      # correlation_service.py
    incidents: Optional[List[Dict[str, Any]]] = None


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
    # Why inference is unavailable when model_loaded is False (missing, stale or
    # mismatched checkpoints). None when the models are loaded.
    model_error: Optional[str] = None
    # The temporal contract the console should lay its timeline out on: the
    # loaded checkpoints' when there are models, cyberworld_v4.config's when not.
    contract: Optional[TemporalContractInfo] = None
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
