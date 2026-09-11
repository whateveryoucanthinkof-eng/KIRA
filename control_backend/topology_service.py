"""
control_backend/topology_service.py
Discovery-first live graph from SPAN flow windows.

Nodes/edges appear only when observed; internal vs external from site CIDRs.
TTL eviction + max_nodes caps keep noisy sites bounded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from control_backend.schema import (
    TopologyEdge,
    TopologyEvent,
    TopologyNode,
    TopologyStats,
)
from control_backend.site_config import SiteConfig, get_site_config


@dataclass
class _NodeState:
    ip: str
    role: str
    zone: str
    label: str = ""
    bytes_in: int = 0
    bytes_out: int = 0
    last_seen: float = 0.0
    risk: float = 0.0


@dataclass
class _EdgeState:
    src: str
    dst: str
    protocol: int = 0
    dst_port: int = 0
    bytes: int = 0
    packets: int = 0
    last_seen: float = 0.0


class TopologyService:
    """Mutable live topology owned by the control backend."""

    def __init__(self, site: Optional[SiteConfig] = None):
        self._lock = threading.Lock()
        self._nodes: Dict[str, _NodeState] = {}
        self._edges: Dict[Tuple[str, str, int, int], _EdgeState] = {}
        self._windows_applied = 0
        self._site_override = site
        self._min_edge_bytes = 1  # Phase 5: collapse near-empty chatter

    def _site(self) -> SiteConfig:
        return self._site_override or get_site_config()

    def reset(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._windows_applied = 0

    def apply_window(
        self,
        flows: Sequence[Any],
        window_end: Optional[float] = None,
        now: Optional[float] = None,
    ) -> TopologyEvent:
        """Upsert nodes/edges from one SPAN window, then TTL-evict."""
        site = self._site()
        ts = float(now if now is not None else (window_end or time.time()))

        with self._lock:
            for f in flows or []:
                src, dst, proto, dport, fwd_b, bwd_b, fwd_p, bwd_p, end_t = _flow_fields(f)
                if not src or not dst:
                    continue
                seen = float(end_t or ts)

                self._touch_node(site, src, bytes_out=fwd_b, bytes_in=bwd_b, last_seen=seen)
                self._touch_node(site, dst, bytes_out=bwd_b, bytes_in=fwd_b, last_seen=seen)

                total_b = int(fwd_b + bwd_b)
                total_p = int(fwd_p + bwd_p)
                if total_b < self._min_edge_bytes and total_p <= 0:
                    continue

                key = (src, dst, int(proto), int(dport))
                edge = self._edges.get(key)
                if edge is None:
                    self._edges[key] = _EdgeState(
                        src=src,
                        dst=dst,
                        protocol=int(proto),
                        dst_port=int(dport),
                        bytes=total_b,
                        packets=total_p,
                        last_seen=seen,
                    )
                else:
                    edge.bytes += total_b
                    edge.packets += total_p
                    edge.last_seen = max(edge.last_seen, seen)

            self._windows_applied += 1
            self._evict_locked(site, ts)
            return self._snapshot_locked(site, ts)

    def attach_risks(self, risk_by_ip: Dict[str, float]) -> None:
        with self._lock:
            for ip, risk in (risk_by_ip or {}).items():
                node = self._nodes.get(ip)
                if node is not None:
                    node.risk = float(max(0.0, min(1.0, risk)))

    def snapshot(self, now: Optional[float] = None) -> TopologyEvent:
        site = self._site()
        ts = float(now if now is not None else time.time())
        with self._lock:
            self._evict_locked(site, ts)
            return self._snapshot_locked(site, ts)

    def _touch_node(
        self,
        site: SiteConfig,
        ip: str,
        *,
        bytes_out: int,
        bytes_in: int,
        last_seen: float,
    ) -> None:
        role = site.classify_ip(ip)
        zone = "external" if role == "external" else "enterprise"
        label = _asset_label(site, ip)
        node = self._nodes.get(ip)
        if node is None:
            self._nodes[ip] = _NodeState(
                ip=ip,
                role=role,
                zone=zone,
                label=label,
                bytes_in=int(bytes_in),
                bytes_out=int(bytes_out),
                last_seen=last_seen,
            )
        else:
            node.bytes_in += int(bytes_in)
            node.bytes_out += int(bytes_out)
            node.last_seen = max(node.last_seen, last_seen)
            node.role = role
            node.zone = zone
            if label:
                node.label = label

    def _evict_locked(self, site: SiteConfig, now: float) -> None:
        node_ttl = max(1, int(site.node_ttl_sec))
        edge_ttl = max(1, int(site.edge_ttl_sec))
        max_nodes = max(1, int(site.max_nodes))

        dead_edges = [
            k
            for k, e in self._edges.items()
            if (now - e.last_seen) > edge_ttl
        ]
        for k in dead_edges:
            del self._edges[k]

        dead_nodes = [
            ip
            for ip, n in self._nodes.items()
            if (now - n.last_seen) > node_ttl
        ]
        for ip in dead_nodes:
            del self._nodes[ip]

        # Drop edges whose endpoints vanished
        self._edges = {
            k: e
            for k, e in self._edges.items()
            if e.src in self._nodes and e.dst in self._nodes
        }

        if len(self._nodes) > max_nodes:
            ranked = sorted(self._nodes.values(), key=lambda n: n.last_seen)
            overflow = len(self._nodes) - max_nodes
            for n in ranked[:overflow]:
                self._nodes.pop(n.ip, None)
            self._edges = {
                k: e
                for k, e in self._edges.items()
                if e.src in self._nodes and e.dst in self._nodes
            }

    def _snapshot_locked(self, site: SiteConfig, now: float) -> TopologyEvent:
        node_ttl = max(1, int(site.node_ttl_sec))
        nodes: List[TopologyNode] = []
        for n in sorted(self._nodes.values(), key=lambda x: x.ip):
            age = now - n.last_seen
            nodes.append(
                TopologyNode(
                    id=n.ip,
                    ip=n.ip,
                    role=n.role,
                    zone=n.zone,
                    label=n.label or None,
                    bytes_in=n.bytes_in,
                    bytes_out=n.bytes_out,
                    last_seen=n.last_seen,
                    risk=n.risk,
                    stale=age > (node_ttl * 0.75),
                )
            )
        edges = [
            TopologyEdge(
                src=e.src,
                dst=e.dst,
                protocol=e.protocol,
                dst_port=e.dst_port,
                bytes=e.bytes,
                packets=e.packets,
                last_seen=e.last_seen,
            )
            for e in sorted(
                self._edges.values(),
                key=lambda x: (x.src, x.dst, x.protocol, x.dst_port),
            )
        ]
        external = sum(1 for n in nodes if n.role == "external")
        return TopologyEvent(
            type="topology_update",
            site_id=site.site_id,
            generated_at=now,
            nodes=nodes,
            edges=edges,
            stats=TopologyStats(
                nodes=len(nodes),
                edges=len(edges),
                external_nodes=external,
                windows_applied=self._windows_applied,
            ),
        )


def _asset_label(site: SiteConfig, ip: str) -> str:
    for a in site.assets_of_interest:
        if a.ip == ip and a.name:
            return a.name
    return ""


def _flow_fields(f: Any) -> Tuple[str, str, int, int, int, int, int, int, float]:
    if isinstance(f, dict):
        return (
            str(f.get("src_ip") or ""),
            str(f.get("dst_ip") or ""),
            int(f.get("protocol") or 0),
            int(f.get("dst_port") or 0),
            int(f.get("fwd_bytes") or 0),
            int(f.get("bwd_bytes") or 0),
            int(f.get("fwd_packets") or 0),
            int(f.get("bwd_packets") or 0),
            float(f.get("end_time") or 0.0),
        )
    return (
        str(getattr(f, "src_ip", "") or ""),
        str(getattr(f, "dst_ip", "") or ""),
        int(getattr(f, "protocol", 0) or 0),
        int(getattr(f, "dst_port", 0) or 0),
        int(getattr(f, "fwd_bytes", 0) or 0),
        int(getattr(f, "bwd_bytes", 0) or 0),
        int(getattr(f, "fwd_packets", 0) or 0),
        int(getattr(f, "bwd_packets", 0) or 0),
        float(getattr(f, "end_time", 0.0) or 0.0),
    )


topology_service = TopologyService()
