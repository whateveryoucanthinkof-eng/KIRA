export type ServiceStatus = "running" | "stopped" | "error" | "unknown";
export type ThreatLevel = "low" | "medium" | "high" | "critical" | "nominal" | "warning" | "elevated";
export type EventSeverity = "info" | "warning" | "error" | "critical";
export type AttackStatus = "none" | "active" | "mitigated" | "contained";

export interface SystemStatus {
  networkStatus: ServiceStatus;
  telemetryStatus: ServiceStatus;
  predictionStatus: ServiceStatus;
  attackStatus: AttackStatus;
  uptime: number; // seconds
  lastUpdate: string; // ISO timestamp
  anomalyScore: number; // 0–100
  threatLevel: ThreatLevel;
  packetLoss: number; // %
  latency: number; // ms
  throughput: number; // Mbps
  activeConnections: number;
  [key: string]: any; // Allow raw backend fields (network_online, ml_active, etc.)
}

export interface TelemetryPoint {
  timestamp: string;
  observed: number;
  predicted: number;
  upperBound: number;
  lowerBound: number;
  confidence: number;
  anomalyScore: number;
}

// ─── Explainability (mirrors control_backend/schema.py) ──────────────────────
// ExplainabilityGroup / ExplainabilityFeature / ExplainabilityPayload are the
// exact wire shapes emitted by ModelAdapter._explain() (Input x Gradient).

export interface ExplainabilityGroup {
  name: string;       // e.g. "TGNE Latent", "Volume", "Connectivity", "Timing"
  percentage: number; // 0–100, share of total attribution for that group
}

export interface ExplainabilityFeature {
  feature: string; // raw feature name, e.g. "byte_rate" or "H_emb_3"
  score: number;   // 0–1, share of total |input x gradient| attribution
  group: string;
}

export interface ExplainabilityPayload {
  available: boolean;
  method?: string | null; // e.g. "Input x Gradient Saliency"
  groups: ExplainabilityGroup[];
  top_features: ExplainabilityFeature[];
}

export interface PredictionResult {
  timestamp: string;
  value: number;
  confidence: number;
  horizon: number; // seconds — forecast_steps x window_seconds from ModelMetadata
  model: string;
  signals: ContributingSignal[];
  // Real model attributions, straight off the wire.
  explainability?: ExplainabilityPayload | null;
  // PredictionData fields (control_backend/schema.py:PredictionData)
  risk?: number;
  max_future_risk?: number;
  predicted_risk_prior?: number | null;
  forecast_error?: number | null;
  hazard_score?: number | null;
  malicious_confidence?: number | null;
  precursor_confidence?: number | null;
  alert?: boolean;
  alert_level?: string; // NOMINAL | WARNING | ELEVATED | CRITICAL
  threshold?: number;
  predicted_stage?: string;
  mitre_tactic?: string | null;
  mitre_technique?: string | null;
  mitre_tactic_id?: string | null;
  mitre_description?: string | null;
  stage_probabilities?: Record<string, number> | null;
  technique_confidence?: number | null;
  stage_provenance?: Record<string, string>;
  // Provenance of the displayed risk (schema.py:PredictionData, spec 21/41):
  // the SOC rule layer may blend model output with deterministic rules.
  ml_risk?: number | null;
  rule_risk?: number | null;
  rules_applied?: boolean | null;
  branch_a_risk?: number;
  branch_b_risk?: number;
}

export interface ContributingSignal {
  name: string;
  weight: number; // 0–1
  direction: "positive" | "negative" | "neutral";
  value: string;
}

export interface ForecastPoint {
  timestamp: string;
  // True horizon of this step, in SECONDS, straight from the backend
  // (schema.py:ForecastPoint.horizon_seconds). Steps are window_seconds apart.
  horizonSeconds: number;
  predictedStage?: string | null;
  predicted: number;
  upperBound: number;
  lowerBound: number;
  confidence: number;
}

export interface AttackEvent {
  id: string;
  timestamp: string;
  type: string;
  sourceIp: string;
  targetIp: string;
  severity: EventSeverity;
  stage: string;
  bytes: number;
  packets: number;
  mitigated: boolean;
}

export interface NetworkEvent {
  id: string;
  timestamp: string;
  severity: EventSeverity;
  category: string;
  source: string;
  destination?: string;
  message: string;
  protocol?: string;
  port?: number;
  raw?: string;
}

export interface TopologyNode {
  id: string;
  label: string;
  type: "router" | "switch" | "server" | "host" | "firewall" | "internet" | "attacker";
  ip?: string;
  status: "online" | "offline" | "degraded" | "compromised" | "warning";
  services?: string[];
  os?: string;
  x: number;
  y: number;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  bandwidth?: number; // Mbps
  utilization?: number; // %
  status: "active" | "saturated" | "down" | "suspicious";
  protocol?: string;
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  lastUpdated: string;
}

export interface SiteInfo {
  name: string;
  location: string;
  timezone: string;
  subnet: string;
  externalIp: string;
  description: string;
}

export interface CommandResult {
  command: string;
  status: "started" | "completed" | "failed";
  output?: string;
  timestamp: string;
}

export interface MitigationPayload {
  target: string;
  action: "block" | "rate_limit" | "redirect" | "isolate";
  duration?: number;
  reason?: string;
}

// WebSocket event payloads
export type WSEventType =
  | "prediction"
  | "topology_update"
  | "ml_reset"
  | "system_status"
  | "command_started"
  | "command_completed"
  | "command_output"
  | "attack_event";

export interface WSEvent {
  type: WSEventType;
  timestamp: string;
  payload: unknown;
}

export interface WSHandlers {
  onPrediction?: (data: PredictionResult) => void;
  onTopologyUpdate?: (data: Topology) => void;
  onMLReset?: (data: { reason: string }) => void;
  onSystemStatus?: (data: SystemStatus) => void;
  onCommandStarted?: (data: { command: string }) => void;
  onCommandCompleted?: (data: CommandResult) => void;
  onCommandOutput?: (data: { command: string; line: string }) => void;
  onAttackEvent?: (data: AttackEvent) => void;
  onError?: (err: Event) => void;
  onClose?: () => void;
  onOpen?: () => void;
}
