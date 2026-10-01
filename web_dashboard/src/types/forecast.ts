/**
 * Alternate futures — the continuations DeepOP weighs from the present.
 *
 * control_backend/forecast_branches.py: step 0 of the served decoding
 * (DeepOPForecastDecoder.forecast_sequence) is reproduced exactly, the three
 * most probable first tokens are kept, and each is decoded greedily through
 * the same decoder. Branch A is therefore the served forecast itself.
 *
 * No model predicts which hosts a branch passes through, its packet and byte
 * volume, or a per-branch risk peak (Branch B rolls out one future, not
 * three). Those fields arrive null and the console marks them under
 * development; the demo fills them so the layout can be seen.
 */

export type BranchKind = "escalation" | "pivot" | "backoff";

export interface BranchHop {
  ip: string;
  name: string;
}

/** One forecast step along a branch. */
export interface BranchStep {
  horizon_seconds: number;
  tactic_lane: string;
  technique: string | null;
  /** DeepOP's probability for this step's token. */
  probability: number;
}

export interface ForecastBranch {
  /** A, B, C by probability — A is the most likely continuation. */
  id: "A" | "B" | "C";
  kind: BranchKind;
  /** What happens on this branch, in one line. */
  label: string;
  /** Kill-chain lane the branch reaches (control_backend/tactics.py LANES). */
  stage: string;
  /** ATT&CK technique id, or "—" for a back-off. */
  technique: string;
  /** Share of the top-3 first-step mass on this branch; the three sum to 1. */
  probability: number;
  /** The first step's actual probability under DeepOP. */
  probability_raw?: number;
  /** DeepOP's probability for the branch's technique step. */
  confidence: number;
  /** When the branch's technique step lands, seconds from now. */
  horizon_seconds: number;
  /** Every forecast step of the branch. */
  path?: BranchStep[];
  /** The hosts the branch passes through. Not predicted by any model yet. */
  hops: BranchHop[] | null;
  /** Forecast packets and bytes over the horizon. Not predicted yet. */
  packets: number | null;
  bytes: number | null;
  /** Peak risk along this branch. Not predicted per branch yet. */
  peak_risk: number | null;
}
