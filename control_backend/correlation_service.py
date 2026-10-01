"""
control_backend/correlation_service.py
Campaigns and incidents for the console, built from the models' own output.

`correlation/` assembles multi-host campaigns from per-window verdicts but was
never wired into serving. This feeds it the live stream:

  observed   one entry per scored window whose technique is an attack lane,
             at the window's time (Branch A: technique, confidence, risk);
  forecast   the latest window's rollout steps that land on an attack lane,
             at window time + horizon (DeepOP token, Branch B risk).

and runs the existing pipeline unchanged: GraphCompactor merges repeats,
HeuristicCausalEdgeScorer links nodes, prune_non_viable_nodes drops the
unlinked, CampaignMerger splits on the evidence horizon. Every number in
that pipeline is hand-set (see HEURISTIC_PARAMS); none is learned.

Incidents group a host's alerting windows: a new incident opens on a host's
first alert, or on an alert more than the evidence horizon after its last
one. The server never acknowledges or closes an incident -- that is the
analyst's call, made in the console.
"""

import itertools
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from correlation.campaign_merge import CampaignMerger
from correlation.causal_edge_scorer import EVIDENCE_HORIZON_SEC, HeuristicCausalEdgeScorer
from correlation.graph_compaction import GraphCompactor
from correlation.trajectory_assembler import Provenance, TrajectoryEntry
from control_backend.tactics import BENIGN, LANES, lane_of, split_token

_SEVERITY = {"CRITICAL": "critical", "ELEVATED": "elevated", "WARNING": "warning", "NOMINAL": "warning"}
_MAX_INCIDENTS = 50


class LiveCorrelation:
    def __init__(self, horizon_sec: float = EVIDENCE_HORIZON_SEC):
        self.horizon_sec = float(horizon_sec)
        self.scorer = HeuristicCausalEdgeScorer(max_time_delta_sec=horizon_sec)
        self.merger = CampaignMerger(max_gap_sec=horizon_sec)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.observed: List[TrajectoryEntry] = []
            self.forecast: List[TrajectoryEntry] = []
            self.pairs: Set[Tuple[str, str]] = set()
            self.incidents: Dict[str, Dict[str, Any]] = {}   # id -> incident
            self.open_by_host: Dict[str, str] = {}
            self._ids = itertools.count(1)
            self._window = 0

    # -- feeding ----------------------------------------------------------

    def observe(self, event, window_end: float, flows=(), contained_hosts: Set[str] = frozenset()):
        """Add one scored window. Returns (campaign, incidents) for the wire."""
        pred = event.prediction
        host = event.target_ip
        with self._lock:
            self._window += 1
            for r in flows:
                if r.src_ip and r.dst_ip:
                    self.pairs.add((r.src_ip, r.dst_ip))

            lane, technique = split_token(pred.predicted_stage)
            if host and lane != BENIGN:
                self.observed.append(TrajectoryEntry(
                    host_ip=host, window_idx=self._window, timestamp=float(window_end),
                    provenance=Provenance.OBSERVED.value, coarse_category=lane,
                    technique_id=technique or "None",
                    confidence=float(pred.technique_confidence or 0.0),
                    risk_score=float(pred.risk),
                ))
            # Only the newest rollout is a live forecast; older ones are superseded.
            self.forecast = []
            if host:
                for f in event.forecast:
                    f_lane, f_tech = split_token(f.predicted_stage)
                    if f_lane == BENIGN:
                        continue
                    self.forecast.append(TrajectoryEntry(
                        host_ip=host, window_idx=self._window,
                        timestamp=float(window_end) + float(f.horizon_seconds),
                        provenance=Provenance.FORECAST.value, coarse_category=f_lane,
                        technique_id=f_tech or "None",
                        confidence=float(f.confidence or 0.0), risk_score=float(f.risk),
                    ))
            cutoff = float(window_end) - self.horizon_sec
            self.observed = [e for e in self.observed if e.timestamp >= cutoff]

            campaign = self._campaign(host)
            self._track_incident(event, window_end, campaign, contained_hosts)
            return campaign, self._incident_list(contained_hosts)

    # -- campaign -----------------------------------------------------------

    def _campaign(self, host: Optional[str]) -> Optional[Dict[str, Any]]:
        entries = self.observed + self.forecast
        if not entries:
            return None
        compactor = GraphCompactor(merge_time_window_sec=self.horizon_sec)
        nodes = compactor.skip_redundant_semantics(entries)
        # Score node to node: one representative entry per compacted node.
        reps = [TrajectoryEntry(
            host_ip=n.host_ip, window_idx=0, timestamp=n.start_time, provenance=n.provenance,
            coarse_category=n.coarse_category, technique_id=n.technique_id,
            confidence=n.mean_confidence, risk_score=n.max_risk_score,
        ) for n in nodes]
        edges = self.scorer.score_candidate_edges(reps, communicating_pairs=self.pairs or None)
        nodes, edges, _ = compactor.prune_non_viable_nodes(nodes, edges)
        campaigns = self.merger.merge_campaigns(nodes, edges)
        if not campaigns:
            return None
        # The campaign holding the scored host's most recent observed activity:
        # what is happening now, not an older, unlinked episode on the same host.
        def latest_on_host(c) -> float:
            ts = [nd["end_time"] for nd in c.nodes
                  if nd["host_ip"] == host and nd["provenance"] == Provenance.OBSERVED.value]
            return max(ts) if ts else float("-inf")
        on_host = [c for c in campaigns if host in c.involved_hosts]
        chosen = max(on_host, key=latest_on_host) if on_host else campaigns[0]
        by_id = {n.node_id: n for n in nodes}
        out = chosen.to_dict()
        out["nodes"] = [{
            "node_id": d["node_id"],
            "host_ip": d["host_ip"],
            "coarse_category": d["coarse_category"],
            "technique_id": d["technique_id"],
            "provenance": d["provenance"],
            "start_time": d["start_time"],
            "end_time": d["end_time"],
            "hit_count": d["hit_count"],
            "max_risk_score": d["risk_score"],
            "mean_confidence": by_id[d["node_id"]].mean_confidence,
        } for d in out["nodes"]]
        out["lanes"] = LANES
        out["method"] = "correlation/ heuristic (hand-set parameters, not learned)"
        return out

    # -- incidents ------------------------------------------------------------

    def _track_incident(self, event, window_end, campaign, contained_hosts) -> None:
        pred = event.prediction
        host = event.target_ip
        if not host or not pred.alert:
            return
        inc_id = self.open_by_host.get(host)
        inc = self.incidents.get(inc_id) if inc_id else None
        if inc is not None and window_end - inc["lastSeen"] > self.horizon_sec:
            inc = None
        lane = lane_of(pred.predicted_stage)
        forecast_only = lane == BENIGN
        if inc is None:
            inc_id = f"INC-{next(self._ids):04d}"
            ew = event.early_warning
            inc = {
                "id": inc_id,
                "host": host,
                "hostLabel": host,
                "title": "",
                "technique": None,
                "tactic": None,
                "severity": "warning",
                "status": "new",
                "opened": float(window_end),
                "lastSeen": float(window_end),
                "peakRisk": 0.0,
                "alertCount": 0,
                "assignee": None,
                "leadTimeSeconds": ew.lead_time_seconds if ew else None,
                "campaignId": None,
                "notes": [{"at": float(window_end), "text": (
                    "Opened by a forecast alert: the rollout crosses the threshold."
                    if forecast_only else "Opened by a model alert on observed traffic.")}],
            }
            self.incidents[inc_id] = inc
            self.open_by_host[host] = inc_id
            while len(self.incidents) > _MAX_INCIDENTS:
                oldest = min(self.incidents.values(), key=lambda i: i["lastSeen"])
                self.incidents.pop(oldest["id"])
                if self.open_by_host.get(oldest["host"]) == oldest["id"]:
                    self.open_by_host.pop(oldest["host"])
        peak = max(float(pred.risk), float(pred.max_future_risk))
        inc["lastSeen"] = float(window_end)
        inc["alertCount"] += 1
        if peak >= inc["peakRisk"]:
            inc["peakRisk"] = peak
            inc["severity"] = _SEVERITY.get(str(pred.alert_level).upper(), "warning")
        if not forecast_only:
            inc["technique"] = pred.mitre_technique
            inc["tactic"] = pred.mitre_tactic
            inc["title"] = pred.mitre_technique or pred.predicted_stage
        elif not inc["title"]:
            target = (event.early_warning.target_milestone_desc if event.early_warning else None)
            inc["title"] = f"Forecast: {target}" if target else "Forecast alert"
        if campaign and host in campaign.get("involved_hosts", []):
            inc["campaignId"] = campaign["campaign_id"]

    def _incident_list(self, contained_hosts: Set[str]) -> List[Dict[str, Any]]:
        out = []
        for inc in self.incidents.values():
            item = dict(inc)
            # Recorded containment is reported, not inferred: the host is
            # marked contained only while a mitigation is recorded for it.
            if inc["host"] in contained_hosts and item["status"] == "new":
                item["status"] = "contained"
            out.append(item)
        return sorted(out, key=lambda i: i["lastSeen"], reverse=True)
