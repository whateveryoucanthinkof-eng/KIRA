/**
 * Offline replay report — the response shape of `POST /api/replay`
 * (`control_backend/main.py:237`).
 *
 * The backend disables the SOC rule layer for this path regardless of the
 * server setting, because an offline analysis is an evaluation and a heuristic
 * that floors risk at 0.40 would contaminate it. So `risk` on a replay row is
 * pure model output — `rules_disabled` is always true.
 *
 * `packets`, `bytes` and `flows` are not returned by the endpoint yet. It holds
 * each window's flows while scoring (`windows[widx]` in `replay_file`) and
 * sends only the scores. The demo populates them; against a live backend they
 * are absent and the DVR's volume trace and flow stream say so.
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
  /** Packets in the window, both directions. Not sent by `/api/replay` yet. */
  packets?: number;
  /** Bytes in the window, both directions. Not sent by `/api/replay` yet. */
  bytes?: number;
  /** The window's flow records, first-packet order. Not sent by `/api/replay` yet. */
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
  results: ReplayRow[];
}
