/**
 * One prediction event off the wire (control_backend/schema.py:
 * PredictionEvent) into the shape the console renders.
 *
 * Both streams go through here: the real /ws socket and the demo's
 * MockWebSocket (src/api/mock.ts), which emits the backend's exact wire
 * format. So the demo exercises the same mapping the live console does.
 */

import type { ContributingSignal, ExplainabilityPayload } from "./types";


/**
 * Real horizon of the forecast in seconds: forecast_steps x the seconds per
 * forecast step (5 x 30 s = 150 s under the current contract; older
 * single-scale checkpoints step by window_seconds). Falls back to the last
 * forecast point's horizon_seconds.
 */
function horizonSeconds(p: any): number {
  const steps = Number(p?.model?.forecast_steps);
  const step = Number(p?.model?.forecast_step_seconds ?? p?.model?.window_seconds);
  if (Number.isFinite(steps) && Number.isFinite(step) && steps > 0 && step > 0) {
    return steps * step;
  }
  const fc: any[] = Array.isArray(p?.forecast) ? p.forecast : [];
  const last = fc.length ? Number(fc[fc.length - 1]?.horizon_seconds) : NaN;
  return Number.isFinite(last) ? last : 0;
}

/** Normalize schema.py:ExplainabilityPayload off the wire. */
function normalizeExplainability(raw: any): ExplainabilityPayload | null {
  if (!raw || typeof raw !== "object") return null;
  const groups = Array.isArray(raw.groups) ? raw.groups : [];
  const feats = Array.isArray(raw.top_features) ? raw.top_features : [];
  return {
    available: Boolean(raw.available),
    method: raw.method ?? null,
    groups: groups
      .filter((g: any) => g && g.name != null)
      .map((g: any) => ({ name: String(g.name), percentage: Number(g.percentage) || 0 })),
    top_features: feats
      .filter((f: any) => f && f.feature != null)
      .map((f: any) => ({
        feature: String(f.feature),
        score: Number(f.score) || 0,
        group: String(f.group ?? "General"),
      })),
  };
}

/**
 * The model's real Input x Gradient attributions, in the shape the existing
 * signal components render. `weight` is the feature's share of total
 * attribution (0–1); the attribution is a magnitude, so it has no sign —
 * hence direction "neutral". `value` carries the feature group label.
 */
function explainabilitySignals(ex: ExplainabilityPayload | null): ContributingSignal[] {
  if (!ex || !ex.available) return [];
  return ex.top_features.map((f) => ({
    name: f.feature,
    weight: f.score,
    direction: "neutral" as const,
    value: f.group,
  }));
}


/** The /ws "prediction" payload, normalised for the console. */
export function normalizePrediction(p: any): any {
  const explain = normalizeExplainability(p.explainability);
  return {
    ...p,
    prediction: p.prediction
      ? {
          // Spread the wire payload first so every PredictionData field
          // survives: alert_level, threshold, mitre_tactic, mitre_technique,
          // stage_probabilities, stage_provenance, …
          ...p.prediction,
          // The console keys its kill-chain lanes by predicted_stage. The
          // backend sends the technique there and the lane in tactic_lane
          // (control_backend/tactics.py); the technique stays available as
          // ml_technique.
          predicted_stage: p.prediction.tactic_lane ?? p.prediction.predicted_stage,
          ml_technique: p.prediction.ml_technique ?? p.prediction.predicted_stage,
          horizon: horizonSeconds(p),
          value: p.prediction.risk || 0,
          confidence: p.prediction.malicious_confidence || 0,
          branch_a_risk: p.prediction.risk ?? 0,
          branch_b_risk: p.prediction.max_future_risk ?? p.prediction.risk ?? 0,
          model: "K.I.R.A. Ensemble",
          explainability: explain,
          // Real per-feature attributions (Input x Gradient), as signals.
          signals: explainabilitySignals(explain),
        }
      : null,
  };
}
