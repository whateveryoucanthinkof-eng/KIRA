/**
 * View-model types for the live event stream.
 *
 * `src/api/types.ts` mirrors `control_backend/schema.py` and is the wire
 * contract — it is not edited here. These are the client-side shapes the
 * pages render, accumulated from that stream (the backend caches only the
 * latest prediction, so all history is built here).
 */

/** One inference window. Every field is measured, none are synthesized. */
export interface LivePoint {
  /** epoch ms */
  t: number;
  /** HH:MM:SS tick label */
  label: string;
  windowId: number | null;
  /** Observed risk for this window, 0–1 (PredictionData.risk). */
  risk: number;
  /** Peak risk across the forecast rollout, 0–1 (PredictionData.max_future_risk). */
  maxFutureRisk: number;
  /** Model output before the SOC rule layer, 0–1. */
  mlRisk: number | null;
  /** Deterministic rule layer alone, 0–1. */
  ruleRisk: number | null;
  /** Mbps, measured. */
  throughput: number;
  packets: number;
  flows: number;
  latencyMs: number;
}

export interface LatencyBudget {
  telemetry_ms?: number;
  inference_ms?: number;
  total_ms?: number;
}

export interface EarlyWarning {
  is_alert?: boolean;
  alert_timestamp?: string | null;
  actual_milestone_timestamp?: string | null;
  lead_time_seconds?: number | null;
  target_milestone_desc?: string | null;
}

export interface StateMeta {
  window_id?: number;
  sequence_ready?: boolean;
  packet_count?: number;
  active_flows?: number;
  pipeline_latency_ms?: number;
  buffer_length?: number;
}

/**
 * The prediction event minus `prediction`/`forecast`/`explainability`, which
 * App already lifts into their own state. Carries the operational context the
 * pages need: pipeline timing, early warning, focus hosts.
 */
export interface PredictionEnvelope {
  mode?: string;
  timestamp?: string;
  wall_clock?: string;
  state?: StateMeta;
  latency?: LatencyBudget;
  early_warning?: EarlyWarning | null;
  attack_active?: boolean;
  attack_phase?: string | null;
  focus_ips?: string[];
  target_ip?: string | null;
  throughput?: number;
}

export type Theme = "ink" | "paper";
