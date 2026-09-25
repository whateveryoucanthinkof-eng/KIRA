"""Campaign correlation may not claim more than the models can see.

The models see 30 s of history and forecast 150 s ahead. The correlation layer
used to link events up to an hour apart (in hours, via an untrained random MLP)
and merge any connected component with no gap check, so unrelated incidents
hours apart became one campaign.
"""

import pytest

from cyberworld_v4.config import get_contract
from correlation.campaign_merge import CampaignMerger
from correlation.causal_edge_scorer import (
    EVIDENCE_HORIZON_SEC,
    HeuristicCausalEdgeScorer,
)
from correlation.graph_compaction import CompactedAlertNode
from correlation.trajectory_assembler import TrajectoryEntry


def _entry(host, t, cat="Recon", tech="T1046", prov="OBSERVED"):
    return TrajectoryEntry(host_ip=host, window_idx=0, timestamp=t, provenance=prov,
                           coarse_category=cat, technique_id=tech, confidence=0.9,
                           risk_score=0.8)


def _node(i, host, start, end=None, cat="Recon"):
    return CompactedAlertNode(node_id=i, host_ip=host, coarse_category=cat,
                              technique_id="T1046", provenance="OBSERVED",
                              start_time=start, end_time=start if end is None else end,
                              hit_count=1, max_risk_score=0.8, mean_confidence=0.9)


def test_default_horizon_is_the_models_evidence_horizon():
    c = get_contract()
    assert EVIDENCE_HORIZON_SEC == c.history_seconds + c.forecast_seconds
    assert EVIDENCE_HORIZON_SEC < 3600


def test_scorer_is_deterministic_and_not_a_neural_net():
    import torch.nn as nn
    s = HeuristicCausalEdgeScorer()
    assert not isinstance(s, nn.Module)
    a, b = _entry("10.0.0.1", 0.0), _entry("10.0.0.1", 10.0, cat="InitialAccess")
    assert s.score_edge(a, b) == HeuristicCausalEdgeScorer().score_edge(a, b) > 0


def test_events_beyond_the_horizon_are_not_linked():
    s = HeuristicCausalEdgeScorer()
    a = _entry("10.0.0.1", 0.0)
    b = _entry("10.0.0.1", EVIDENCE_HORIZON_SEC + 1.0, cat="InitialAccess")
    assert s.score_edge(a, b) == 0.0
    assert s.score_candidate_edges([a, b]) == []


def test_score_decays_with_time():
    s = HeuristicCausalEdgeScorer()
    a = _entry("10.0.0.1", 0.0)
    near = s.score_edge(a, _entry("10.0.0.1", 5.0, cat="InitialAccess"))
    far = s.score_edge(a, _entry("10.0.0.1", EVIDENCE_HORIZON_SEC * 0.9, cat="InitialAccess"))
    assert near > far > 0


def test_incidents_hours_apart_sharing_a_host_are_separate_campaigns():
    six_h = 6 * 3600.0
    nodes = [_node(0, "10.0.0.5", 0.0), _node(1, "10.0.0.5", 30.0),
             _node(2, "10.0.0.5", six_h), _node(3, "10.0.0.5", six_h + 30.0)]
    # An edge spanning six hours, as the old 3600 s scorer chain could produce.
    edges = [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9)]
    camps = CampaignMerger().merge_campaigns(nodes, edges)
    assert len(camps) == 2
    assert all(c.duration_sec <= EVIDENCE_HORIZON_SEC for c in camps)


def test_chain_of_short_hops_cannot_stretch_past_a_gap():
    # Nodes connected only by bridging edges, but with a hole in the middle.
    nodes = [_node(0, "10.0.0.5", 0.0, 60.0), _node(1, "10.0.0.6", 100.0),
             _node(2, "10.0.0.6", 100.0 + EVIDENCE_HORIZON_SEC + 50.0)]
    edges = [(0, 1, 0.9), (1, 2, 0.9)]
    camps = CampaignMerger().merge_campaigns(nodes, edges)
    assert sorted(len(c.nodes) for c in camps) == [1, 2]


def test_close_events_still_merge():
    nodes = [_node(0, "10.0.0.5", 0.0), _node(1, "10.0.0.6", 20.0, cat="InitialAccess")]
    camps = CampaignMerger().merge_campaigns(nodes, [(0, 1, 0.8)])
    assert len(camps) == 1 and camps[0].involved_hosts == ["10.0.0.5", "10.0.0.6"]
