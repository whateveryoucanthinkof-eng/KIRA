"""
Attack Trajectory Assembly & Correlation Package (GRAIN + APTShield Inspired).

Not wired into control_backend: nothing in the live path calls it, and the
dashboard's Campaign page is populated only by demo fixtures. The edge scorer
is a hand-set heuristic, not a learned model (see causal_edge_scorer.py).
"""

from correlation.trajectory_assembler import (
    Provenance,
    TrajectoryEntry,
    HostAttackTrajectory,
    AttackTrajectoryAssembler,
)
from correlation.causal_edge_scorer import EVIDENCE_HORIZON_SEC, HeuristicCausalEdgeScorer
from correlation.graph_compaction import CompactedAlertNode, GraphCompactor
from correlation.campaign_merge import AttackCampaign, CampaignMerger

__all__ = [
    "Provenance",
    "TrajectoryEntry",
    "HostAttackTrajectory",
    "AttackTrajectoryAssembler",
    "EVIDENCE_HORIZON_SEC",
    "HeuristicCausalEdgeScorer",
    "CompactedAlertNode",
    "GraphCompactor",
    "AttackCampaign",
    "CampaignMerger",
]
