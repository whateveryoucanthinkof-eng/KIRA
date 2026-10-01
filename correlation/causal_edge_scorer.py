"""
Heuristic causal-edge scorer. HAND-SET, NOT LEARNED.

Scores how plausible it is that trajectory entry (u, t1, tech1) led to
(v, t2, tech2), for linking alerts into candidate campaign chains.

What this file used to be, and why it changed:

* It was titled "Learned Causal Edge Scorer" and the architecture diagram
  called it an ML-compatible causal scorer. Its `score_edge` ran a 7->32->32->1
  MLP whose weights were never trained, never saved and never loaded -- every
  edge score was the output of a randomly initialised network. The features fed
  into it (0.1 / 0.85 / 0.95 - 0.15*gap / floor 0.2, threshold 0.35) were
  typed in by hand, fitted to nothing and taken from no paper.
* It linked events up to 3600 s apart and measured time in hours, on top of a
  model that sees 30 s of history and forecasts 150 s ahead. It drew hour-long
  attack chains from 30 seconds of evidence.

It is now what it always actually was: a transparent heuristic. Every constant
lives in HEURISTIC_PARAMS so it can be read, questioned and replaced. The
default linking window is the model's own evidence horizon (history +
forecast, from cyberworld_v4/config.py) instead of an hour. A learned scorer
would need labelled campaign chains to fit against; none exist in this
repository, so none is claimed.
"""

from typing import Dict, List, Optional, Set, Tuple

from cyberworld_v4.config import get_contract
from correlation.trajectory_assembler import Provenance, TrajectoryEntry

_CONTRACT = get_contract()

#: Longest gap over which two events may be linked by default: the span the
#: models can actually see (history) plus the span they forecast. Linking
#: across more than this asserts a connection no model output supports.
EVIDENCE_HORIZON_SEC: float = _CONTRACT.history_seconds + _CONTRACT.forecast_seconds

# Kill-chain stage ordering (hand-set). -1 = not an attack stage.
#: CredentialAccess sits between Recon and InitialAccess, as the console's
#: kill-chain lanes draw it (control_backend/tactics.py). It was missing, so
#: DeepOP's CredentialAccess.T1110 (brute force) ranked -1 and scored as "not an
#: attack stage" (plausibility 0.1): a brute-force episode could never link into
#: the campaign it opens. In this corpus T1110 is external password guessing
#: against an exposed service, i.e. the step that GAINS access, which is why it
#: precedes InitialAccess here rather than following it as in ATT&CK's
#: enterprise matrix (where it is post-compromise credential theft).
TACTIC_ORDER = {
    "Recon": 0,
    "CredentialAccess": 1,
    "InitialAccess": 2,
    "Execution": 3,
    "C2": 4,
    "LateralMovement": 5,
    "Exfiltration": 6,
    "Impact": 7,
    "Benign": -1,
    "Unknown": -1,
    "UNKNOWN": -1,
}

#: Every tunable number in the scorer. None of these were fitted.
HEURISTIC_PARAMS: Dict[str, float] = {
    "non_attack_plausibility": 0.1,   # either side benign/unknown
    "same_stage_plausibility": 0.85,  # repeated scans, persistent C2
    "forward_base": 0.95,             # next kill-chain stage
    "forward_step_penalty": 0.15,     # per skipped stage
    "forward_floor": 0.2,
    "backward_plausibility": 0.2,     # e.g. Impact -> Recon
    "same_host_locality": 1.0,
    "observed_link_locality": 0.8,    # traffic seen between the two hosts
    "unknown_link_locality": 0.5,     # cross-host, no communication data given
    "provenance_both_observed": 1.0,
    "provenance_mixed": 0.75,
    "provenance_both_forecast": 0.5,
    "min_score_threshold": 0.35,
}


def compute_transition_plausibility(coarse1: str, coarse2: str) -> float:
    """Hand-set kill-chain transition prior in [0, 1]."""
    p = HEURISTIC_PARAMS
    s1 = TACTIC_ORDER.get(coarse1, -1)
    s2 = TACTIC_ORDER.get(coarse2, -1)
    if s1 == -1 or s2 == -1:
        return p["non_attack_plausibility"]
    if s1 == s2:
        return p["same_stage_plausibility"]
    if s2 > s1:
        gap = s2 - s1
        return round(max(p["forward_floor"], p["forward_base"] - p["forward_step_penalty"] * gap), 4)
    return p["backward_plausibility"]


class HeuristicCausalEdgeScorer:
    """Deterministic pairwise scorer: plausibility x time decay x locality x provenance."""

    def __init__(self, max_time_delta_sec: float = EVIDENCE_HORIZON_SEC):
        self.max_time_delta_sec = float(max_time_delta_sec)

    def _locality(self, e1: TrajectoryEntry, e2: TrajectoryEntry,
                  communicating_pairs: Optional[Set[Tuple[str, str]]]) -> float:
        p = HEURISTIC_PARAMS
        if e1.host_ip == e2.host_ip:
            return p["same_host_locality"]
        if communicating_pairs is None:
            return p["unknown_link_locality"]
        linked = ((e1.host_ip, e2.host_ip) in communicating_pairs
                  or (e2.host_ip, e1.host_ip) in communicating_pairs)
        return p["observed_link_locality"] if linked else 0.0

    @staticmethod
    def _provenance(e1: TrajectoryEntry, e2: TrajectoryEntry) -> float:
        p = HEURISTIC_PARAMS
        obs = Provenance.OBSERVED.value
        fc = Provenance.FORECAST.value
        if e1.provenance == obs and e2.provenance == obs:
            return p["provenance_both_observed"]
        if e1.provenance == fc and e2.provenance == fc:
            return p["provenance_both_forecast"]
        return p["provenance_mixed"]

    def score_edge(
        self,
        e1: TrajectoryEntry,
        e2: TrajectoryEntry,
        communicating_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> float:
        dt = e2.timestamp - e1.timestamp
        if dt < 0 or dt > self.max_time_delta_sec:
            return 0.0  # causality needs non-decreasing time, within the evidence horizon
        time_decay = 1.0 - dt / self.max_time_delta_sec if self.max_time_delta_sec > 0 else 0.0
        return float(
            compute_transition_plausibility(e1.coarse_category, e2.coarse_category)
            * time_decay
            * self._locality(e1, e2, communicating_pairs)
            * self._provenance(e1, e2)
        )

    def score_candidate_edges(
        self,
        entries: List[TrajectoryEntry],
        communicating_pairs: Optional[Set[Tuple[str, str]]] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """(src_idx, dst_idx, score) for every pair within the evidence horizon."""
        threshold = (HEURISTIC_PARAMS["min_score_threshold"]
                     if min_score_threshold is None else min_score_threshold)
        indexed = sorted(enumerate(entries), key=lambda x: x[1].timestamp)
        out: List[Tuple[int, int, float]] = []
        for pos, (i, e1) in enumerate(indexed):
            for j, e2 in (x for x in indexed[pos + 1:]):
                if e2.timestamp - e1.timestamp > self.max_time_delta_sec:
                    break  # sorted: every later entry is further away
                if e1.coarse_category == "Benign" and e2.coarse_category == "Benign":
                    continue
                score = self.score_edge(e1, e2, communicating_pairs)
                if score >= threshold:
                    out.append((i, j, score))
        return out
