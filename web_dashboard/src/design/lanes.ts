/**
 * Kill-chain lanes and technique names — one copy for every view.
 *
 * Mirrors control_backend/tactics.py:LANES. Lane keys are what arrives as
 * `tactic_lane` on the verdict and on each forecast step. CredentialAccess is
 * a lane because DeepOP emits CredentialAccess.T1110 (brute force).
 * Execution and LateralMovement are drawn but `dev`: neither Branch A nor
 * DeepOP has a token for them (do_this_in_next_session_ml_review.md §8).
 */

export interface Lane {
  key: string;
  /** Short label for the compact kill-chain strip. */
  short: string;
  label: string;
  /** No deployed head emits this lane yet. */
  dev?: boolean;
}

export const LANES: Lane[] = [
  { key: "Recon", short: "Recon", label: "Recon" },
  { key: "CredentialAccess", short: "Creds", label: "Credential Access" },
  { key: "InitialAccess", short: "Access", label: "Initial Access" },
  { key: "Execution", short: "Exec", label: "Execution", dev: true },
  { key: "C2", short: "C2", label: "C2" },
  { key: "LateralMovement", short: "Lateral", label: "Lateral Movement", dev: true },
  { key: "Exfiltration", short: "Exfil", label: "Exfiltration" },
  { key: "Impact", short: "Impact", label: "Impact" },
];

export const LANE_KEYS = LANES.map((l) => l.key);

/**
 * Names for every technique a head can emit: Branch A's TECHNIQUE_VOCAB and
 * DeepOP's tokens (control_backend/model_adapter.py:TECHNIQUE_TO_MITRE).
 */
export const TECHNIQUE_NAMES: Record<string, string> = {
  T1046: "Network Service Discovery",
  T1595: "Active Scanning",
  T1110: "Brute Force",
  T1190: "Exploit Public-Facing App",
  T1189: "Drive-by Compromise",
  T1071: "Application Layer Protocol",
  "T1071.001": "Web Protocols",
  "T1568.001": "Fast Flux DNS",
  T1204: "User Execution",
  T1005: "Data from Local System",
  T1498: "Network Denial of Service",
  "T1498.001": "Direct Network Flood",
  T1020: "Automated Exfiltration",
};
