/**
 * Attack storyline.
 *
 * The kill chain as one rail, each stage pinned to a moment: when it was
 * first observed (t −82s), the present (NOW), or when the forecast expects it
 * (+8s). Observed stages come from the campaign graph's OBSERVED nodes and
 * their start times; the present from predicted_stage; the future from the
 * leading forecast branch and the rollout's predicted stages.
 *
 * Encoding is the console's: observed is warm and solid, the present carries
 * the verdict's severity, forecast is cool and dashed.
 */

import type { ForecastPoint, PredictionResult } from "../api/types";
import type { Campaign } from "../types/campaign";
import type { ForecastBranch } from "../types/forecast";
import { sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, OBSERVED } from "../design/charts";
import { ahead } from "../design/time";

import { LANES, TECHNIQUE_NAMES } from "../design/lanes";

type State = "done" | "now" | "forecast" | "idle";

interface Cell {
  key: string;
  label: string;
  state: State;
  technique: string | null;
  anchor: string | null;
  /** Seconds relative to the present; negative is past. */
  at: number | null;
  probability: number | null;
}

function signed(seconds: number): string {
  const s = Math.round(seconds);
  return s === 0 ? "NOW" : s < 0 ? `t −${Math.abs(s)}s` : ahead(s);
}

export default function Storyline({
  prediction,
  forecast,
  campaign,
  branches,
  futureSeconds,
}: {
  prediction: PredictionResult | null;
  forecast: ForecastPoint[];
  campaign: Campaign | null;
  branches: ForecastBranch[] | null;
  /** When the cursor is inside the forecast horizon: how far ahead. */
  futureSeconds: number | null;
}) {
  const stage = prediction?.predicted_stage ?? null;
  const nowIdx = LANES.findIndex((l) => l.key === stage);
  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);
  const nowColour = sevColor(sevFromRisk(risk, threshold));
  const techniqueNow = prediction?.mitre_technique ? prediction.mitre_technique.split(" ")[0] : null;

  // Earliest observed node per lane, as seconds before the campaign's end.
  const observed = new Map<string, { technique: string; at: number }>();
  const projected = new Map<string, { technique: string; at: number }>();
  if (campaign) {
    for (const n of campaign.nodes) {
      const at = n.start_time - campaign.end_time;
      const into = n.provenance === "OBSERVED" ? observed : projected;
      const prev = into.get(n.coarse_category);
      if (!prev || at < prev.at) into.set(n.coarse_category, { technique: n.technique_id, at });
    }
  }

  // Forecast: the leading branch first, then the rollout's own stages.
  const upcoming = new Map<string, { technique: string | null; at: number; probability: number | null }>();
  for (const b of branches ?? []) {
    if (b.kind === "backoff" || upcoming.has(b.stage)) continue;
    upcoming.set(b.stage, { technique: b.technique, at: b.horizon_seconds, probability: b.probability });
  }
  for (const f of forecast) {
    const k = f.predictedStage;
    if (!k || upcoming.has(k)) continue;
    upcoming.set(k, { technique: projected.get(k)?.technique ?? null, at: f.horizonSeconds, probability: null });
  }
  for (const [k, v] of projected) {
    if (!upcoming.has(k) && v.at > 0) upcoming.set(k, { technique: v.technique, at: v.at, probability: null });
  }

  const cells: Cell[] = LANES.map((lane, i) => {
    if (i === nowIdx) {
      return { key: lane.key, label: lane.label, state: "now", technique: techniqueNow, anchor: "NOW", at: 0, probability: null };
    }
    const o = observed.get(lane.key);
    if (o && (nowIdx < 0 || i < nowIdx)) {
      return { key: lane.key, label: lane.label, state: "done", technique: o.technique, anchor: signed(o.at), at: o.at, probability: null };
    }
    const a = upcoming.get(lane.key);
    if (a && (nowIdx < 0 || i > nowIdx)) {
      return {
        key: lane.key,
        label: lane.label,
        state: "forecast",
        technique: a.technique,
        anchor: signed(a.at),
        at: a.at,
        probability: a.probability,
      };
    }
    return { key: lane.key, label: lane.label, state: "idle", technique: null, anchor: null, at: null, probability: null };
  });

  // How each pair of neighbouring stages is joined on the rail.
  const joint = (a: Cell | undefined, b: Cell | undefined): string => {
    if (!a || !b) return "none";
    if (a.state === "idle" || b.state === "idle") return "faint";
    if (a.state === "forecast" || b.state === "forecast") return "forecast";
    return "observed";
  };

  return (
    <div className="sl">
      {cells.map((c, i) => {
        const reached = c.state === "forecast" && futureSeconds != null && c.at != null && c.at <= futureSeconds;
        const name = c.technique ? TECHNIQUE_NAMES[c.technique] : null;
        return (
          <div
            key={c.key}
            className={`sl-cell is-${c.state}${reached ? " is-reached" : ""}${LANES[i].dev ? " is-udev" : ""}`}
            title={LANES[i].dev ? "Under development: no deployed head emits this stage yet." : undefined}
          >
            <span className={`sl-rail is-left is-${joint(cells[i - 1], c)}`} />
            <span className={`sl-rail is-right is-${joint(c, cells[i + 1])}`} />
            <span
              className={c.state === "now" ? "sl-mark is-ticking-frame" : "sl-mark"}
              style={
                c.state === "done"
                  ? { background: OBSERVED, borderColor: OBSERVED }
                  : c.state === "now"
                    ? { background: nowColour, borderColor: nowColour }
                    : c.state === "forecast"
                      ? { borderColor: FORECAST }
                      : undefined
              }
            />
            <div className="sl-lane">{c.label}</div>
            <div className="sl-line" title={name ?? undefined}>
              {c.technique ? <b>{c.technique}</b> : <span className="sl-none">{LANES[i].dev ? "in dev" : "—"}</span>}
              {c.anchor && (
                <span className="sl-anchor" style={c.state === "now" ? { color: nowColour } : undefined}>
                  {c.anchor}
                  {c.probability != null && <span className="sl-prob">{Math.round(c.probability * 100)}%</span>}
                </span>
              )}
            </div>
            {name && <div className="sl-name">{name}</div>}
          </div>
        );
      })}
    </div>
  );
}
