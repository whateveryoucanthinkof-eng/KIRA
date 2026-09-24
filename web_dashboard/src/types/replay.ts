/**
 * Offline replay report — the response shape of `POST /api/replay`
 * (`control_backend/main.py:237`).
 *
 * The backend disables the SOC rule layer for this path regardless of the
 * server setting, because an offline analysis is an evaluation and a heuristic
 * that floors risk at 0.40 would contaminate it. So `risk` on a replay row is
 * pure model output — `rules_disabled` is always true.
 */

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
