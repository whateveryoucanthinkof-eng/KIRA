"""
GRAIN-Inspired Learned Causal Edge Scorer.

Scores causal plausibility between security trajectory entries (u, t1, tech1) -> (v, t2, tech2)
without relying on rigid predefined attack templates.
Uses a learned classifier over temporal distance, transition plausibility,
host identity, communication links, and risk deltas.
"""

from typing import List, Tuple, Optional, Set
import numpy as np
import torch
import torch.nn as nn

from correlation.trajectory_assembler import TrajectoryEntry, Provenance


# Standard Kill-Chain / ATT&CK Stage Transitions Prior Matrix
TACTIC_ORDER = {
    "Recon": 0,
    "InitialAccess": 1,
    "Execution": 2,
    "C2": 3,
    "LateralMovement": 4,
    "Exfiltration": 5,
    "Impact": 6,
    "Benign": -1,
    "Unknown": -1,
}


def compute_transition_plausibility(coarse1: str, coarse2: str) -> float:
    """Computes empirical attack stage transition plausibility."""
    s1 = TACTIC_ORDER.get(coarse1, -1)
    s2 = TACTIC_ORDER.get(coarse2, -1)

    if s1 == -1 or s2 == -1:
        return 0.1  # Low baseline for benign/unknown transitions

    # Same stage continuation (e.g. repeated scans or persistent C2)
    if s1 == s2:
        return 0.85

    # Forward kill-chain progression (e.g. Recon -> InitialAccess -> C2)
    if s2 > s1:
        gap = s2 - s1
        return round(max(0.2, 0.95 - 0.15 * gap), 4)

    # Backward jump (e.g. Impact -> Recon, less likely in single chain)
    return 0.2


class CausalEdgeScorer(nn.Module):
    """
    MLP-based pairwise causality scoring module.

    Unreachable from any live serving path as of the 2026-09-22 serving
    review (grep for `CausalEdgeScorer(` / `score_candidate_edges` finds no
    caller anywhere outside this module and its re-export in
    correlation/__init__.py). Nothing in this repository ever trains or
    saves a checkpoint for it either, so `self.mlp` is permanently the
    random initialisation PyTorch gives `nn.Linear` unless something loads
    real weights into it.

    That combination -- unreachable today, but a plain nn.Module anyone
    could import and wire into a pipeline tomorrow -- is exactly how a
    `min_score_threshold=0.35` gate over noise would start silently
    dropping (or keeping) edges without anyone deciding that on purpose.
    The guard below makes that impossible: score_edge()/score_candidate_edges()
    refuse to run until this instance has had real weights loaded into it
    (via load_state_dict(), the same call every other model in this repo
    uses to load its checkpoint). This is a fail-loud guard, not a
    pass-everything-through-with-a-warning one -- deliberately, because
    nothing depends on this scorer's current behaviour (there is no live
    caller to keep working), so there is no compatibility reason to prefer
    a quieter default, and a warning is too easy to lose in a log versus a
    raised exception that stops a caller from ever trusting an untrained
    score.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        super(CausalEdgeScorer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Set only by load_state_dict() below, i.e. only when something has
        # actually loaded weights into this specific instance -- constructing
        # a new CausalEdgeScorer never inherits another instance's trained
        # status.
        self._weights_loaded = False

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._weights_loaded = True
        return result

    def _require_trained_weights(self) -> None:
        if not self._weights_loaded:
            raise RuntimeError(
                "CausalEdgeScorer.score_edge()/score_candidate_edges() called "
                "with untrained (randomly initialised) weights -- this MLP is "
                "never loaded from a checkpoint anywhere in this repository. "
                "Scoring with it would silently filter or pass edges based on "
                "noise, not learned causal plausibility. Load real weights "
                "with load_state_dict(...) before scoring."
            )

    def extract_pair_features(
        self,
        e1: TrajectoryEntry,
        e2: TrajectoryEntry,
        communicating_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> np.ndarray:
        """
        Features:
        0: delta_t in hours (clipped to [0, 24])
        1: is_same_host (1.0 or 0.0)
        2: is_lateral_link (traffic observed between e1.host and e2.host)
        3: transition_plausibility in [0, 1]
        4: delta_risk (risk2 - risk1)
        5: min_confidence (min(conf1, conf2))
        6: provenance_weight (1.0 if both observed, 0.6 if forecast)
        """
        feat = np.zeros(7, dtype=np.float32)
        dt_hours = max(0.0, e2.timestamp - e1.timestamp) / 3600.0
        feat[0] = min(1.0, dt_hours / 24.0)
        feat[1] = 1.0 if e1.host_ip == e2.host_ip else 0.0

        is_lat = False
        if communicating_pairs and e1.host_ip != e2.host_ip:
            if (e1.host_ip, e2.host_ip) in communicating_pairs or (e2.host_ip, e1.host_ip) in communicating_pairs:
                is_lat = True
        feat[2] = 1.0 if is_lat else 0.0

        feat[3] = compute_transition_plausibility(e1.coarse_category, e2.coarse_category)
        feat[4] = float(np.clip(e2.risk_score - e1.risk_score, -1.0, 1.0))
        feat[5] = min(e1.confidence, e2.confidence)

        if e1.provenance == Provenance.OBSERVED.value and e2.provenance == Provenance.OBSERVED.value:
            feat[6] = 1.0
        elif e1.provenance == Provenance.FORECAST.value and e2.provenance == Provenance.FORECAST.value:
            feat[6] = 0.5
        else:
            feat[6] = 0.75

        return feat

    def score_edge(
        self,
        e1: TrajectoryEntry,
        e2: TrajectoryEntry,
        communicating_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> float:
        self._require_trained_weights()
        if e2.timestamp < e1.timestamp:
            return 0.0  # Causality requires non-decreasing time

        device = next(self.parameters()).device
        feat = self.extract_pair_features(e1, e2, communicating_pairs)
        t_feat = torch.from_numpy(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            score = self.mlp(t_feat).item()
        return float(score)

    def score_candidate_edges(
        self,
        entries: List[TrajectoryEntry],
        max_time_delta_sec: float = 3600.0,
        communicating_pairs: Optional[Set[Tuple[str, str]]] = None,
        min_score_threshold: float = 0.35,
    ) -> List[Tuple[int, int, float]]:
        """
        Builds and scores candidate causality edges between all trajectory entries within max_time_delta.
        Returns: list of (src_idx, dst_idx, score).
        """
        self._require_trained_weights()
        # Sort indices by timestamp to reduce complexity from O(N^2) to temporal window O(N * W)
        indexed_entries = sorted(enumerate(entries), key=lambda x: x[1].timestamp)
        n = len(indexed_entries)
        candidate_edges = []

        for pos, (i, e1) in enumerate(indexed_entries):
            for next_pos in range(pos + 1, n):
                j, e2 = indexed_entries[next_pos]
                dt = e2.timestamp - e1.timestamp
                if dt > max_time_delta_sec:
                    # Non-decreasing timestamp invariant: subsequent entries are also > max_time_delta_sec
                    break

                # Ignore transitions between purely benign background events
                if e1.coarse_category == "Benign" and e2.coarse_category == "Benign":
                    continue

                # Causal plausibility: must be same host progression or observed network communication
                if e1.host_ip != e2.host_ip and communicating_pairs is not None:
                    if (e1.host_ip, e2.host_ip) not in communicating_pairs and (e2.host_ip, e1.host_ip) not in communicating_pairs:
                        continue

                score = self.score_edge(e1, e2, communicating_pairs)
                if score >= min_score_threshold:
                    candidate_edges.append((i, j, score))

        return candidate_edges
