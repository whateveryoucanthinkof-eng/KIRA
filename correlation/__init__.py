"""
Attack Trajectory Assembly & Correlation Package (GRAIN + APTShield Inspired).
"""

from correlation.trajectory_assembler import (
    Provenance,
    TrajectoryEntry,
    HostAttackTrajectory,
    AttackTrajectoryAssembler,
)
from correlation.causal_edge_scorer import CausalEdgeScorer
from correlation.graph_compaction import CompactedAlertNode, GraphCompactor
from correlation.campaign_merge import AttackCampaign, CampaignMerger

__all__ = [
    "Provenance",
    "TrajectoryEntry",
    "HostAttackTrajectory",
    "AttackTrajectoryAssembler",
    "CausalEdgeScorer",
    "CompactedAlertNode",
    "GraphCompactor",
    "AttackCampaign",
    "CampaignMerger",
]
