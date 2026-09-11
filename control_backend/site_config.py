"""
control_backend/site_config.py
Load per-deployment site profiles (CIDRs, sensor, lab_mode, assets of interest).

Override with env:
  CYBERWORLD_SITE=containerlab-enterprise|local-default|<name>
  CYBERWORLD_SITE_CONFIG=/absolute/or/relative/path/to/site.yaml
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from control_backend.lab_config import REPO_ROOT

logger = logging.getLogger("antigravity.site_config")

SITES_DIR = os.path.join(REPO_ROOT, "config", "sites")
DEFAULT_SITE_ID = "containerlab-enterprise"


@dataclass(frozen=True)
class AssetOfInterest:
    ip: str
    name: str = ""
    role: str = ""


@dataclass(frozen=True)
class SiteConfig:
    site_id: str
    display_name: str
    lab_mode: bool
    enterprise_cidrs: Tuple[str, ...]
    external_cidrs: Tuple[str, ...]
    zones: Dict[str, List[str]]
    assets_of_interest: Tuple[AssetOfInterest, ...]
    sensor_mode: str
    sensor_interface: str
    sensor_container: Optional[str]
    node_ttl_sec: int
    edge_ttl_sec: int
    max_nodes: int
    operator_hints: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    def asset_ips(self) -> Tuple[str, ...]:
        return tuple(a.ip for a in self.assets_of_interest)

    def external_traffic_hint(self) -> str:
        return (
            self.operator_hints.get("external_traffic")
            or "External hosts appear when SPAN observes traffic outside enterprise_cidrs."
        ).strip()

    def is_enterprise_ip(self, ip: str) -> bool:
        return _ip_in_cidrs(ip, self.enterprise_cidrs)

    def is_external_ip(self, ip: str) -> bool:
        if not ip:
            return False
        if self.external_cidrs and _ip_in_cidrs(ip, self.external_cidrs):
            return True
        return not self.is_enterprise_ip(ip)

    def classify_ip(self, ip: str) -> str:
        """Return 'internal' | 'external' | 'unknown'."""
        if not ip:
            return "unknown"
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return "unknown"
        if self.is_enterprise_ip(ip):
            return "internal"
        return "external"

    def as_public_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "display_name": self.display_name,
            "lab_mode": self.lab_mode,
            "enterprise_cidrs": list(self.enterprise_cidrs),
            "external_cidrs": list(self.external_cidrs),
            "zones": self.zones,
            "assets_of_interest": [
                {"ip": a.ip, "name": a.name, "role": a.role} for a in self.assets_of_interest
            ],
            "sensor": {
                "mode": self.sensor_mode,
                "interface": self.sensor_interface,
                "container": self.sensor_container,
            },
            "topology": {
                "node_ttl_sec": self.node_ttl_sec,
                "edge_ttl_sec": self.edge_ttl_sec,
                "max_nodes": self.max_nodes,
            },
            "operator_hints": dict(self.operator_hints),
            "source_path": self.source_path,
        }


def _ip_in_cidrs(ip: str, cidrs: Sequence[str]) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for c in cidrs:
        try:
            if addr in ipaddress.ip_network(c, strict=False):
                return True
        except ValueError:
            continue
    return False


def _parse_assets(raw: Any) -> Tuple[AssetOfInterest, ...]:
    out: List[AssetOfInterest] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(AssetOfInterest(ip=item))
        elif isinstance(item, dict) and item.get("ip"):
            out.append(
                AssetOfInterest(
                    ip=str(item["ip"]),
                    name=str(item.get("name") or ""),
                    role=str(item.get("role") or ""),
                )
            )
    return tuple(out)


def resolve_site_config_path() -> str:
    explicit = os.environ.get("CYBERWORLD_SITE_CONFIG")
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(REPO_ROOT, explicit)
        return os.path.abspath(path)

    site_id = os.environ.get("CYBERWORLD_SITE", DEFAULT_SITE_ID).strip() or DEFAULT_SITE_ID
    return os.path.abspath(os.path.join(SITES_DIR, f"{site_id}.yaml"))


def load_site_config(path: Optional[str] = None) -> SiteConfig:
    cfg_path = os.path.abspath(path or resolve_site_config_path())
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Site config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sensor = raw.get("sensor") or {}
    topo = raw.get("topology") or {}
    hints = raw.get("operator_hints") or {}

    site = SiteConfig(
        site_id=str(raw.get("site_id") or os.path.splitext(os.path.basename(cfg_path))[0]),
        display_name=str(raw.get("display_name") or raw.get("site_id") or "site"),
        lab_mode=bool(raw.get("lab_mode", False)),
        enterprise_cidrs=tuple(str(c) for c in (raw.get("enterprise_cidrs") or [])),
        external_cidrs=tuple(str(c) for c in (raw.get("external_cidrs") or [])),
        zones={str(k): list(v) for k, v in (raw.get("zones") or {}).items()},
        assets_of_interest=_parse_assets(raw.get("assets_of_interest")),
        sensor_mode=str(sensor.get("mode") or "local"),
        sensor_interface=str(
            os.environ.get("CYBERWORLD_SENSOR_IFACE")
            or sensor.get("interface")
            or "eth1"
        ),
        sensor_container=sensor.get("container"),
        node_ttl_sec=int(topo.get("node_ttl_sec") or 120),
        edge_ttl_sec=int(topo.get("edge_ttl_sec") or 60),
        max_nodes=int(topo.get("max_nodes") or 500),
        operator_hints={str(k): str(v) for k, v in hints.items()},
        source_path=cfg_path,
    )
    logger.info(
        "Loaded site config site_id=%s lab_mode=%s path=%s",
        site.site_id,
        site.lab_mode,
        cfg_path,
    )
    return site


@lru_cache(maxsize=1)
def get_site_config() -> SiteConfig:
    return load_site_config()


def reload_site_config() -> SiteConfig:
    get_site_config.cache_clear()
    return get_site_config()


def flow_endpoint_activity(flows: Sequence[Any]) -> Dict[str, float]:
    """
    Score IPs by approximate activity (bytes + packets) across a flow window.
    Accepts UnifiedFlowRecord-like objects or dicts.
    """
    scores: Dict[str, float] = {}

    def _add(ip: str, weight: float) -> None:
        if not ip:
            return
        scores[ip] = scores.get(ip, 0.0) + weight

    for f in flows or []:
        if isinstance(f, dict):
            src = str(f.get("src_ip") or "")
            dst = str(f.get("dst_ip") or "")
            fwd_b = float(f.get("fwd_bytes") or 0)
            bwd_b = float(f.get("bwd_bytes") or 0)
            fwd_p = float(f.get("fwd_packets") or 0)
            bwd_p = float(f.get("bwd_packets") or 0)
        else:
            src = str(getattr(f, "src_ip", "") or "")
            dst = str(getattr(f, "dst_ip", "") or "")
            fwd_b = float(getattr(f, "fwd_bytes", 0) or 0)
            bwd_b = float(getattr(f, "bwd_bytes", 0) or 0)
            fwd_p = float(getattr(f, "fwd_packets", 0) or 0)
            bwd_p = float(getattr(f, "bwd_packets", 0) or 0)

        # Weight: bytes dominate, packets break ties; +1 ensures presence counts.
        _add(src, fwd_b + bwd_b * 0.25 + fwd_p + 1.0)
        _add(dst, bwd_b + fwd_b * 0.25 + bwd_p + 1.0)

    return scores


def select_primary_target_ip(
    flows: Sequence[Any],
    site: Optional[SiteConfig] = None,
    fallback: Optional[str] = None,
) -> str:
    """
    Choose which host to score for a window — site-aware, not lab-hardcoded.

    Priority:
      1) assets_of_interest present in the window (highest activity among them)
      2) most active internal (enterprise) IP
      3) most active IP overall
      4) first configured asset_of_interest
      5) explicit fallback or empty string
    """
    site = site or get_site_config()
    scores = flow_endpoint_activity(flows)
    if not scores:
        assets = site.asset_ips()
        if assets:
            return assets[0]
        return fallback or ""

    asset_set = set(site.asset_ips())
    present_assets = {ip: sc for ip, sc in scores.items() if ip in asset_set}
    if present_assets:
        return max(present_assets.items(), key=lambda kv: kv[1])[0]

    internal = {
        ip: sc for ip, sc in scores.items() if site.classify_ip(ip) == "internal"
    }
    if internal:
        return max(internal.items(), key=lambda kv: kv[1])[0]

    return max(scores.items(), key=lambda kv: kv[1])[0]
