"""
APTShield-Inspired Graph Compaction & Pruning Module.

Applies two core mechanisms to keep streaming correlation tractable:
1. Redundant Semantics Skipping: Merges duplicate technique alerts on the same host within short time windows.
2. Non-Viable Node Pruning: Eliminates isolated alert nodes with zero high-confidence causal edges.
"""

from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
import numpy as np

from correlation.trajectory_assembler import TrajectoryEntry


@dataclass
class CompactedAlertNode:
    """A consolidated alert node surviving semantics skipping and compaction."""
    node_id: int
    host_ip: str
    coarse_category: str
    technique_id: str
    provenance: str
    start_time: float
    end_time: float
    hit_count: int
    max_risk_score: float
    mean_confidence: float
    original_entries: List[TrajectoryEntry] = field(default_factory=list)


class GraphCompactor:
    """
    Compacts and prunes trajectory event graphs.
    """

    def __init__(
        self,
        merge_time_window_sec: float = 180.0,
        min_causal_edge_threshold: float = 0.40,
        prune_isolated_benign: bool = True,
    ):
        self.merge_time_window_sec = merge_time_window_sec
        self.min_causal_edge_threshold = min_causal_edge_threshold
        self.prune_isolated_benign = prune_isolated_benign

    def skip_redundant_semantics(
        self, entries: List[TrajectoryEntry]
    ) -> List[CompactedAlertNode]:
        """
        Merges consecutive duplicate (host, coarse_category, technique_id, provenance)
        events occurring within merge_time_window_sec into a single compacted node.
        """
        if not entries:
            return []

        sorted_entries = sorted(entries, key=lambda e: (e.host_ip, e.timestamp))
        compacted_nodes: List[CompactedAlertNode] = []

        current_group: List[TrajectoryEntry] = []

        def flush_group(group: List[TrajectoryEntry]):
            if not group:
                return
            first = group[0]
            last = group[-1]
            max_risk = max(e.risk_score for e in group)
            mean_conf = float(np.mean([e.confidence for e in group]))

            node = CompactedAlertNode(
                node_id=len(compacted_nodes),
                host_ip=first.host_ip,
                coarse_category=first.coarse_category,
                technique_id=first.technique_id,
                provenance=first.provenance,
                start_time=first.timestamp,
                end_time=last.timestamp,
                hit_count=len(group),
                max_risk_score=max_risk,
                mean_confidence=mean_conf,
                original_entries=list(group),
            )
            compacted_nodes.append(node)

        for entry in sorted_entries:
            if not current_group:
                current_group.append(entry)
                continue

            prev = current_group[-1]
            is_same_host = entry.host_ip == prev.host_ip
            is_same_tech = (entry.coarse_category == prev.coarse_category and
                            entry.technique_id == prev.technique_id and
                            entry.provenance == prev.provenance)
            is_within_win = (entry.timestamp - prev.timestamp) <= self.merge_time_window_sec

            if is_same_host and is_same_tech and is_within_win:
                current_group.append(entry)
            else:
                flush_group(current_group)
                current_group = [entry]

        flush_group(current_group)
        return compacted_nodes

    def prune_non_viable_nodes(
        self,
        nodes: List[CompactedAlertNode],
        edges: List[Tuple[int, int, float]],
        return_details: bool = False,
    ) -> Tuple:
        """
        Prunes nodes that have no incident causal edges with score >= min_causal_edge_threshold,
        except high-severity unmerged active attacks.
        
        Computes compaction and fidelity metrics:
        - Compaction Ratio: percentage reduction in graph elements
        - Attack Event Recall (AER): fraction of true attack entries preserved post-pruning
        - False Pruning Rate (FPR): fraction of pruned entries that were actually malicious
        """
        # Filter edges by threshold
        valid_edges = [e for e in edges if e[2] >= self.min_causal_edge_threshold]

        connected_node_ids: Set[int] = set()
        for u, v, _ in valid_edges:
            connected_node_ids.add(u)
            connected_node_ids.add(v)

        surviving_nodes = []
        old_to_new_id: Dict[int, int] = {}

        for old_id, node in enumerate(nodes):
            is_connected = old_id in connected_node_ids
            # SOC-Grade Adaptive Preservation:
            # 1. Connected nodes in causal chains (causal score >= threshold)
            # 2. Any verified attack evidence (non-benign category or technique)
            # 3. Any elevated risk event (max_risk_score >= 0.20)
            is_attack_evidence = (
                node.coarse_category not in ["Benign", "None", "Unknown"]
                or (node.technique_id and node.technique_id not in ["Benign", "None", ""])
                or node.max_risk_score >= 0.20
            )

            if is_connected or is_attack_evidence:
                new_id = len(surviving_nodes)
                old_to_new_id[old_id] = new_id
                node.node_id = new_id
                surviving_nodes.append(node)

        # Re-index edges
        reindexed_edges = []
        for u, v, score in valid_edges:
            if u in old_to_new_id and v in old_to_new_id:
                reindexed_edges.append((old_to_new_id[u], old_to_new_id[v], score))

        # Compaction ratio metric: 1 - (N_surv + E_surv) / (N_orig + E_orig)
        n_orig = max(1, len(nodes) + len(edges))
        n_surv = len(surviving_nodes) + len(reindexed_edges)
        compaction_ratio = max(0.0, 1.0 - (n_surv / n_orig))

        # Fidelity & Recall Analysis
        def count_attacks(entry_list):
            return sum(1 for e in entry_list if e.coarse_category != "Benign" or e.risk_score > 0.5)

        total_orig_entries = sum(len(n.original_entries) for n in nodes)
        total_attack_entries = sum(count_attacks(n.original_entries) for n in nodes)

        surv_entries_count = sum(len(n.original_entries) for n in surviving_nodes)
        surv_attack_entries = sum(count_attacks(n.original_entries) for n in surviving_nodes)

        pruned_entries_count = max(0, total_orig_entries - surv_entries_count)
        pruned_attack_entries = max(0, total_attack_entries - surv_attack_entries)

        attack_event_recall = float(surv_attack_entries / max(1, total_attack_entries)) if total_attack_entries > 0 else 1.0
        false_pruning_rate = float(pruned_attack_entries / max(1, pruned_entries_count)) if pruned_entries_count > 0 else 0.0

        self.last_metrics = {
            "compaction_ratio": float(compaction_ratio),
            "attack_event_recall": float(attack_event_recall),
            "false_pruning_rate": float(false_pruning_rate),
            "total_attack_entries": total_attack_entries,
            "surviving_attack_entries": surv_attack_entries,
            "total_pruned_entries": pruned_entries_count,
            "pruned_attack_entries": pruned_attack_entries,
            "original_nodes": len(nodes),
            "surviving_nodes": len(surviving_nodes),
        }

        if return_details:
            return surviving_nodes, reindexed_edges, float(compaction_ratio), self.last_metrics
        return surviving_nodes, reindexed_edges, float(compaction_ratio)
