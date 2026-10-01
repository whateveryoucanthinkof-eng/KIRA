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

/** Seconds per window — telemetry/state/state_builder.py window size. */
export const WINDOW_SECONDS = 2;
/** The rollout's horizon: K=8 steps of one window each. */
export const HORIZON_SECONDS = 16;

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
