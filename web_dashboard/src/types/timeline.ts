/**
 * Time travel.
 *
 * Every window the stream delivers is kept as a frame — the verdict, the
 * rollout, the campaign, the topology and the evidence exactly as they were.
 * A single cursor chooses what every view renders: the live window, a past
 * frame, or a point inside the forecast horizon.
 */

import type { ForecastPoint, PredictionResult, Topology } from "../api/types";
import type { Campaign } from "./campaign";
import type { FlowRecord, StateDim } from "./evidence";
import type { ForecastBranch } from "./forecast";
import type { AttentionMatrix } from "./attention";
import type { LivePoint, PredictionEnvelope } from "./live";

/*
 * The temporal contract, taken from /api/status (schema.py:TemporalContractInfo):
 * the loaded checkpoints' when there are models, cyberworld_v4/config.py's
 * when not. These are live ES-module bindings: setContract() updates them and
 * every view reads the current value on its next render. The defaults are the
 * configured contract (2 s windows, 5 forecast steps of 30 s), used until the
 * first status response arrives.
 */

/** Seconds per input window (telemetry/state/state_builder.py). */
export let WINDOW_SECONDS = 2;
/** Seconds per forecast step — the rollout's resolution. */
export let FORECAST_STEP_SECONDS = 30;
/** Number of forecast steps. */
export let FORECAST_STEPS = 5;
/** The rollout's horizon: FORECAST_STEPS × FORECAST_STEP_SECONDS. */
export let HORIZON_SECONDS = 150;

export function setContract(c: {
  window_seconds?: number;
  forecast_steps?: number;
  forecast_step_seconds?: number;
} | null | undefined): void {
  if (!c) return;
  const w = Number(c.window_seconds);
  const k = Number(c.forecast_steps);
  const step = Number(c.forecast_step_seconds);
  if (Number.isFinite(w) && w > 0) WINDOW_SECONDS = w;
  if (Number.isFinite(step) && step > 0) FORECAST_STEP_SECONDS = step;
  if (Number.isFinite(k) && k > 0) FORECAST_STEPS = k;
  HORIZON_SECONDS = FORECAST_STEPS * FORECAST_STEP_SECONDS;
}

export interface Frame {
  /** Client-side sequence. Window ids can repeat when the scenario is steered. */
  seq: number;
  point: LivePoint;
  prediction: PredictionResult;
  envelope: PredictionEnvelope;
  forecast: ForecastPoint[];
  campaign: Campaign | null;
  topology: Topology | null;
  stateVector: StateDim[] | null;
  flows: FlowRecord[];
  flowsInWindow: number | null;
  branches: ForecastBranch[] | null;
  attention: AttentionMatrix | null;
}

export type Cursor =
  | { kind: "live" }
  /** A recorded window, by sequence. */
  | { kind: "past"; seq: number }
  /** A point inside the forecast horizon, seconds ahead of the live window. */
  | { kind: "future"; seconds: number };
