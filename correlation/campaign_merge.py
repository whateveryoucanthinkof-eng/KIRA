"""
Campaign Merge & Forensic Graph Assembly Module.

Computes connected components over the compacted causal graph to isolate
candidate multi-host attack campaigns. Formats campaign subgraphs for
forensic inspection, dashboard rendering, and explainability.
"""

from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass, asdict
import numpy as np

from correlation.graph_compaction import CompactedAlertNode


@dataclass
class AttackCampaign:
    """A reconstructed multi-step, multi-host attack campaign."""
    campaign_id: int
    involved_hosts: List[str]
    root_cause_node_ids: List[int]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]  # [{"src": u, "dst": v, "causality_score": s}]
    max_risk_score: float
    start_time: float
    end_time: float
    duration_sec: float
    attack_techniques: List[str]
    has_forecast_components: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CampaignMerger:
    """Extracts coherent campaign subgraphs from causal graph."""

    def merge_campaigns(
        self,
        nodes: List[CompactedAlertNode],
        edges: List[Tuple[int, int, float]],
    ) -> List[AttackCampaign]:
        if not nodes:
            return []

        n = len(nodes)
        # Build undirected adjacency for connected components
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        in_degrees: Dict[int, int] = {i: 0 for i in range(n)}

        for u, v, _ in edges:
            adj[u].append(v)
            adj[v].append(u)
            in_degrees[v] += 1

        visited: Set[int] = set()
        campaigns: List[AttackCampaign] = []

        for i in range(n):
            if i in visited:
                continue

            # BFS / DFS traversal
            comp_node_ids = []
            queue = [i]
            visited.add(i)

            while queue:
                curr = queue.pop(0)
                comp_node_ids.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Sort nodes in component chronologically
            comp_nodes = [nodes[idx] for idx in comp_node_ids]
            comp_nodes = sorted(comp_nodes, key=lambda nd: nd.start_time)

            comp_id_set = set(comp_node_ids)
            comp_edges = [
                {"src": u, "dst": v, "causality_score": score}
                for u, v, score in edges
                if u in comp_id_set and v in comp_id_set
            ]

            involved_hosts = sorted(list(set(nd.host_ip for nd in comp_nodes)))
            root_causes = [nd.node_id for nd in comp_nodes if in_degrees[nd.node_id] == 0]
            if not root_causes:
                root_causes = [comp_nodes[0].node_id]

            max_risk = max(nd.max_risk_score for nd in comp_nodes)
            start_t = min(nd.start_time for nd in comp_nodes)
            end_t = max(nd.end_time for nd in comp_nodes)
            dur = max(0.0, end_t - start_t)

            techs = []
            has_forecast = False
            for nd in comp_nodes:
                if nd.technique_id and nd.technique_id != "None":
                    techs.append(nd.technique_id)
                if nd.provenance == "FORECAST":
                    has_forecast = True

            campaign = AttackCampaign(
                campaign_id=len(campaigns) + 1,
                involved_hosts=involved_hosts,
                root_cause_node_ids=root_causes,
                nodes=[
                    {
                        "node_id": nd.node_id,
                        "host_ip": nd.host_ip,
                        "coarse_category": nd.coarse_category,
                        "technique_id": nd.technique_id,
                        "provenance": nd.provenance,
                        "start_time": nd.start_time,
                        "end_time": nd.end_time,
                        "hit_count": nd.hit_count,
                        "risk_score": nd.max_risk_score,
                    }
                    for nd in comp_nodes
                ],
                edges=comp_edges,
                max_risk_score=float(max_risk),
                start_time=float(start_t),
                end_time=float(end_t),
                duration_sec=float(dur),
                attack_techniques=techs,
                has_forecast_components=has_forecast,
            )
            campaigns.append(campaign)

        # Sort campaigns by risk descending
        campaigns = sorted(campaigns, key=lambda c: c.max_risk_score, reverse=True)
        return campaigns
