/**
 * Kill-chain timeline.
 *
 * The console renders risk well but never showed the *narrative* — and the
 * narrative is what "attack forecasting" has to demonstrate. This strip is the
 * claim, stated in one line: here is how far the intrusion has actually got,
 * and here is where the model says it goes next.
 *
 * Lanes are the real ordering from `correlation/causal_edge_scorer.py:19`
 * TACTIC_ORDER. Stage assignment comes off the wire — `predicted_stage` on
 * PredictionData for the present, and on each ForecastPoint for the rollout.
 *
 * Encoding follows the system's rule that observed is warm and solid while
 * forecast is cool and hatched, so provenance survives greyscale.
 */

import type { ForecastPoint } from "../api/types";
import { Micro, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, OBSERVED } from "../design/charts";

/** correlation/causal_edge_scorer.py:TACTIC_ORDER */
const LANES = [
  { key: "Recon", label: "Recon" },
  { key: "InitialAccess", label: "Access" },
  { key: "Execution", label: "Exec" },
  { key: "C2", label: "C2" },
  { key: "LateralMovement", label: "Lateral" },
  { key: "Exfiltration", label: "Exfil" },
  { key: "Impact", label: "Impact" },
];

type Cell = {
  key: string;
  label: string;
  state: "done" | "now" | "forecast" | "idle";
  technique: string | null;
  horizon: string | null;
  risk: number | null;
};

function horizonLabel(seconds: number): string {
  if (!Number.isFinite(seconds)) return "";
  return seconds < 60 ? `+${seconds % 1 === 0 ? seconds.toFixed(0) : seconds.toFixed(1)}s` : `+${(seconds / 60).toFixed(1)}m`;
}

export default function KillChain({
  stage,
  technique,
  forecast,
  threshold = 0.65,
  compact,
}: {
  /** PredictionData.predicted_stage — where the intrusion is now. */
  stage: string | null | undefined;
  /** PredictionData.mitre_technique — displayed on the current lane. */
  technique?: string | null;
  forecast: ForecastPoint[];
  threshold?: number;
  /** Tighter variant for the Predictions rail. */
  compact?: boolean;
}) {
  const nowIdx = LANES.findIndex((l) => l.key === stage);

  // First forecast point that lands on each lane ahead of the present.
  const forecastAt = new Map<string, ForecastPoint>();
  for (const f of forecast) {
    const k = f.predictedStage;
    if (!k || forecastAt.has(k)) continue;
    const idx = LANES.findIndex((l) => l.key === k);
    if (idx > nowIdx) forecastAt.set(k, f);
  }

  const cells: Cell[] = LANES.map((lane, i) => {
    if (nowIdx >= 0 && i === nowIdx) {
      return {
        key: lane.key,
        label: lane.label,
        state: "now",
        technique: technique ? technique.split(" ")[0] : null,
        horizon: "NOW",
        risk: null,
      };
    }
    if (nowIdx >= 0 && i < nowIdx) {
      return { key: lane.key, label: lane.label, state: "done", technique: null, horizon: null, risk: null };
    }
    const f = forecastAt.get(lane.key);
    if (f) {
      return {
        key: lane.key,
        label: lane.label,
        state: "forecast",
        technique: null,
        horizon: horizonLabel(f.horizonSeconds),
        risk: f.predicted / 100,
      };
    }
    return { key: lane.key, label: lane.label, state: "idle", technique: null, horizon: null, risk: null };
  });

  const barH = compact ? 8 : 14;

  return (
    <div>
      {/* Hatch pattern for forecast cells — cool, and readable in greyscale. */}
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden>
        <defs>
          <pattern id="kc-hatch" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">
            <rect width="4" height="4" fill="transparent" />
            <line x1="0" y1="0" x2="0" y2="4" stroke={FORECAST} strokeWidth="1.6" />
          </pattern>
        </defs>
      </svg>

      <div style={{ display: "grid", gridTemplateColumns: `repeat(${LANES.length}, minmax(0, 1fr))`, gap: 1 }}>
        {cells.map((c) => {
          const active = c.state === "now";
          const done = c.state === "done";
          const fc = c.state === "forecast";

          return (
            <div key={c.key} style={{ minWidth: 0 }}>
              {/* Lane label */}
              <div
                className="t-micro"
                style={{
                  color: active ? "var(--paper-000)" : done ? "var(--paper-400)" : "var(--paper-600)",
                  marginBottom: 6,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {c.label}
              </div>

              {/* The bar */}
              <div
                style={{
                  height: barH,
                  border: "var(--hair)",
                  borderLeft: active ? "3px solid var(--paper-000)" : "var(--hair)",
                  background: done ? OBSERVED : active ? "var(--paper-000)" : "var(--ink-200)",
                  position: "relative",
                  overflow: "hidden",
                  transition: "background var(--dur-fill) var(--ease)",
                }}
              >
                {fc && (
                  <svg width="100%" height="100%" style={{ display: "block" }} aria-hidden>
                    <rect width="100%" height="100%" fill="url(#kc-hatch)" opacity={0.75} />
                  </svg>
                )}
              </div>

              {/* Technique / horizon caption */}
              <div style={{ marginTop: 5, minHeight: compact ? 12 : 26 }}>
                {c.technique && (
                  <div className="t-data-s" style={{ color: "var(--paper-000)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {c.technique}
                  </div>
                )}
                {c.horizon && (
                  <div
                    className="t-data-s"
                    style={{
                      color: active ? "var(--paper-400)" : fc ? FORECAST : "var(--paper-600)",
                      letterSpacing: active ? "0.1em" : undefined,
                    }}
                  >
                    {c.horizon}
                  </div>
                )}
                {!compact && fc && c.risk != null && (
                  <div className="t-data-s" style={{ color: sevColor(sevFromRisk(c.risk, threshold)) }}>
                    {c.risk.toFixed(2)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {!compact && (
        <div style={{ display: "flex", gap: "var(--s-4)", marginTop: "var(--s-3)", paddingTop: "var(--s-2)", borderTop: "var(--hair)" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
            <span style={{ width: 14, height: 8, background: OBSERVED, display: "inline-block" }} />
            <Micro>observed</Micro>
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
            <span style={{ width: 14, height: 8, background: "var(--paper-000)", display: "inline-block" }} />
            <Micro>current stage</Micro>
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
            <svg width="14" height="8" style={{ display: "inline-block" }} aria-hidden>
              <rect width="14" height="8" fill="url(#kc-hatch)" />
            </svg>
            <Micro>forecast</Micro>
          </span>
        </div>
      )}
    </div>
  );
}
