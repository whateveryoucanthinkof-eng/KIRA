/**
 * Offline replay report — the response shape of `POST /api/replay`
 * (`control_backend/main.py:237`).
 *
 * The backend disables the SOC rule layer for this path regardless of the
 * server setting, because an offline analysis is an evaluation and a heuristic
 * that floors risk at 0.40 would contaminate it. So `risk` on a replay row is
 * pure model output — `rules_disabled` is always true.
 *
 * `packets`, `bytes` and `flows` are measured from each window's flow records
 * (control_backend/main.py:_analyse_capture); flows are capped per window.
 */

import type { FlowRecord } from "./evidence";

export interface ReplayForecastPoint {
  horizon_seconds: number;
  risk: number;
}

export interface ReplayFeature {
  feature: string;
  score: number;
  group: string;
}

export interface ReplayRow {
  window: number;
  target: string;
  risk: number;
  ml_risk: number | null;
  alert: boolean;
  stage: string;
  mitre_tactic: string | null;
  mitre_technique: string | null;
  forecast: ReplayForecastPoint[];
  top_features: ReplayFeature[];
  /** Kill-chain lane of `stage` (control_backend/tactics.py). */
  tactic_lane?: string | null;
  /** Packets in the window, both directions. */
  packets?: number;
  /** Bytes in the window, both directions. */
  bytes?: number;
  /** The window's flow records, first-packet order (capped). */
  flows?: FlowRecord[];
}

export interface ReplayReport {
  filename: string;
  /** "pcap" for .pcap/.pcapng, "csv" for .csv/.binetflow */
  kind: "pcap" | "csv";
  windows_analyzed: number;
  flagged_windows: number;
  /** Always true — the rule layer is forced off for offline analysis. */
  rules_disabled: boolean;
  /** Seconds per window the capture was scored on. */
  window_seconds?: number;
  /** The alert threshold the windows were scored against. */
  threshold?: number;
  results: ReplayRow[];
}

/** A built-in capture the Replay page can analyse without an upload. */
export interface ReplaySample {
  id: string;
  /** Filename on disk. */
  name: string;
  label: string;
  kind: "pcap" | "csv";
  bytes: number;
  /** Windows in the capture, when known before parsing (demo fixtures). */
  windows: number | null;
  source: string;
  note: string;
}
