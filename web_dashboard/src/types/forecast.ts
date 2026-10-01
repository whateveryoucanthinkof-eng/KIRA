/**
 * Alternate futures — the branches the forecast head weighs from the present.
 *
 * Branch B of the model rolls the hidden state forward; the stage head reads a
 * distribution over what happens next. This is that distribution collapsed to
 * its three most likely continuations, each with the path it would take
 * through the network and what it would cost on the wire.
 *
 * Not yet on /ws: the backend forwards the rollout's risk and predicted stage
 * per step (ForecastPoint) but not the competing continuations. The demo
 * stream carries this shape; against a live backend the field is absent and
 * the forecast tree says so.
 */

export type BranchKind = "escalation" | "pivot" | "backoff";

export interface BranchHop {
  ip: string;
  name: string;
}

export interface ForecastBranch {
  /** A, B, C by probability — A is the most likely continuation. */
  id: "A" | "B" | "C";
  kind: BranchKind;
  /** What happens on this branch, in one line. */
  label: string;
  /** Kill-chain lane the branch reaches (correlation TACTIC_ORDER). */
  stage: string;
  /** ATT&CK technique id, or "—" for a back-off. */
  technique: string;
  /** Share of the forecast mass on this branch; the three sum to 1. */
  probability: number;
  /** Technique-head confidence for the branch's technique. */
  confidence: number;
  /** When the branch's first milestone is expected, seconds from now. */
  horizon_seconds: number;
  /** The hosts the branch passes through, in order. */
  hops: BranchHop[];
  /** Forecast packets and bytes over the horizon if this branch plays out. */
  packets: number;
  bytes: number;
  /** Peak risk the rollout reaches along this branch. */
  peak_risk: number;
}
