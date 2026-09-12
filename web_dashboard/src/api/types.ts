export type ServiceStatus = "running" | "stopped" | "error" | "unknown";
export type ThreatLevel = "low" | "medium" | "high" | "critical";
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

export interface PredictionResult {
  timestamp: string;
  value: number;
  confidence: number;
  horizon: number; // minutes
  model: string;
  signals: ContributingSignal[];
  stage_provenance?: Record<string, string>;
}

export interface ContributingSignal {
  name: string;
  weight: number; // 0–1
  direction: "positive" | "negative" | "neutral";
  value: string;
}

export interface ForecastPoint {
  timestamp: string;
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
