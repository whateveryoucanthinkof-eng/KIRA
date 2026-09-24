"""
Campaign Merge & Forensic Graph Assembly Module.

Computes connected components over the compacted causal graph to isolate
candidate multi-host attack campaigns. Formats campaign subgraphs for
forensic inspection, dashboard rendering, and explainability.

Components are split on time. This used to merge anything connected with no
gap check, so two unrelated incidents six hours apart that shared one host
became one "campaign", and a chain of individually short edges could stretch
across a whole day. An edge whose endpoints are further apart than
`max_gap_sec` is dropped, and a component is cut wherever its nodes, in time
order, leave a gap wider than that. The default is the models' evidence horizon
(history + forecast); linking across more is not supported by any model output.
"""

from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass, asdict

from correlation.causal_edge_scorer import EVIDENCE_HORIZON_SEC
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

    def __init__(self, max_gap_sec: float = EVIDENCE_HORIZON_SEC):
        self.max_gap_sec = float(max_gap_sec)

    def _gap(self, a: CompactedAlertNode, b: CompactedAlertNode) -> float:
        """Seconds between two nodes' time spans (0 if they overlap)."""
        return max(0.0, max(a.start_time, b.start_time) - min(a.end_time, b.end_time))

    def _split_on_time(self, comp: List[int], nodes: List[CompactedAlertNode]) -> List[List[int]]:
        ordered = sorted(comp, key=lambda i: nodes[i].start_time)
        parts: List[List[int]] = [[ordered[0]]]
        reach = nodes[ordered[0]].end_time
        for i in ordered[1:]:
            if nodes[i].start_time - reach > self.max_gap_sec:
                parts.append([i])
                reach = nodes[i].end_time
            else:
                parts[-1].append(i)
                reach = max(reach, nodes[i].end_time)
        return parts

    def merge_campaigns(
        self,
        nodes: List[CompactedAlertNode],
        edges: List[Tuple[int, int, float]],
    ) -> List[AttackCampaign]:
        if not nodes:
            return []

        edges = [(u, v, s) for u, v, s in edges
                 if self._gap(nodes[u], nodes[v]) <= self.max_gap_sec]

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

            for part in self._split_on_time(comp_node_ids, nodes):
                campaigns.append(self._build_campaign(part, nodes, edges, in_degrees, len(campaigns) + 1))

        # Sort campaigns by risk descending
        campaigns = sorted(campaigns, key=lambda c: c.max_risk_score, reverse=True)
        return campaigns

    def _build_campaign(self, comp_node_ids, nodes, edges, in_degrees, campaign_id) -> AttackCampaign:
        comp_nodes = sorted((nodes[idx] for idx in comp_node_ids), key=lambda nd: nd.start_time)

        comp_id_set = set(comp_node_ids)
        comp_edges = [
            {"src": u, "dst": v, "causality_score": score}
            for u, v, score in edges
            if u in comp_id_set and v in comp_id_set
        ]

        involved_hosts = sorted({nd.host_ip for nd in comp_nodes})
        root_causes = [nd.node_id for nd in comp_nodes if in_degrees[nd.node_id] == 0]
        if not root_causes:
            root_causes = [comp_nodes[0].node_id]

        start_t = min(nd.start_time for nd in comp_nodes)
        end_t = max(nd.end_time for nd in comp_nodes)

        return AttackCampaign(
            campaign_id=campaign_id,
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
            max_risk_score=float(max(nd.max_risk_score for nd in comp_nodes)),
            start_time=float(start_t),
            end_time=float(end_t),
            duration_sec=float(max(0.0, end_t - start_t)),
            attack_techniques=[nd.technique_id for nd in comp_nodes
                               if nd.technique_id and nd.technique_id != "None"],
            has_forecast_components=any(nd.provenance == "FORECAST" for nd in comp_nodes),
        )
