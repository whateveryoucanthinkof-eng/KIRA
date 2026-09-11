"""
tests/test_site_config.py
Phase 1: site profiles + CIDR-aware primary host selection (no lab IP hardcoding).
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "bita"))

from control_backend.site_config import (
    load_site_config,
    reload_site_config,
    select_primary_target_ip,
    get_site_config,
)
from control_backend.model_adapter import flows_from_span_dicts, select_primary_target


@pytest.fixture
def local_site(monkeypatch):
    path = os.path.join(PROJECT_ROOT, "config", "sites", "local-default.yaml")
    monkeypatch.setenv("CYBERWORLD_SITE_CONFIG", path)
    reload_site_config()
    yield get_site_config()
    monkeypatch.delenv("CYBERWORLD_SITE_CONFIG", raising=False)
    reload_site_config()


@pytest.fixture
def lab_site(monkeypatch):
    path = os.path.join(PROJECT_ROOT, "config", "sites", "containerlab-enterprise.yaml")
    monkeypatch.setenv("CYBERWORLD_SITE_CONFIG", path)
    reload_site_config()
    yield get_site_config()
    monkeypatch.delenv("CYBERWORLD_SITE_CONFIG", raising=False)
    reload_site_config()


def test_load_containerlab_site():
    path = os.path.join(PROJECT_ROOT, "config", "sites", "containerlab-enterprise.yaml")
    site = load_site_config(path)
    assert site.site_id == "containerlab-enterprise"
    assert site.lab_mode is True
    assert site.classify_ip("10.0.3.10") == "internal"
    assert site.classify_ip("192.168.100.10") == "external"
    assert "10.0.3.10" in site.asset_ips()


def test_load_local_default_site():
    path = os.path.join(PROJECT_ROOT, "config", "sites", "local-default.yaml")
    site = load_site_config(path)
    assert site.site_id == "local-default"
    assert site.lab_mode is False
    assert site.classify_ip("10.50.1.20") == "internal"
    assert site.classify_ip("203.0.113.10") == "external"


def test_select_primary_prefers_internal_over_external(local_site):
    """Synthetic non-lab IPs: external→internal traffic scores the internal host."""
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "203.0.113.10",
                "dst_ip": "10.50.1.20",
                "src_port": 4444,
                "dst_port": 80,
                "protocol": 6,
                "fwd_bytes": 5000,
                "bwd_bytes": 200,
                "fwd_packets": 20,
                "bwd_packets": 5,
                "start_time": 1.0,
                "end_time": 2.0,
            }
        ]
    )
    target = select_primary_target_ip(flows, site=local_site)
    assert target == "10.50.1.20"


def test_select_primary_assets_of_interest_win(lab_site):
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "10.0.1.11",
                "dst_ip": "10.0.2.40",
                "src_port": 50000,
                "dst_port": 8000,
                "protocol": 6,
                "fwd_bytes": 100,
                "bwd_bytes": 100,
                "fwd_packets": 2,
                "bwd_packets": 2,
                "start_time": 1.0,
                "end_time": 2.0,
            },
            {
                "src_ip": "10.0.1.11",
                "dst_ip": "10.0.3.10",
                "src_port": 50001,
                "dst_port": 80,
                "protocol": 6,
                "fwd_bytes": 50,
                "bwd_bytes": 50,
                "fwd_packets": 1,
                "bwd_packets": 1,
                "start_time": 1.0,
                "end_time": 2.0,
            },
        ]
    )
    # Both are assets; higher activity asset should win (srv-app has more weight here)
    target = select_primary_target_ip(flows, site=lab_site)
    assert target in ("10.0.2.40", "10.0.3.10")
    assert target == "10.0.2.40"


def test_model_adapter_select_primary_uses_site(local_site):
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "203.0.113.10",
                "dst_ip": "10.50.1.20",
                "src_port": 1,
                "dst_port": 443,
                "protocol": 6,
                "fwd_bytes": 9000,
                "bwd_bytes": 100,
                "fwd_packets": 30,
                "bwd_packets": 2,
            }
        ]
    )
    assert select_primary_target(flows) == "10.50.1.20"


def test_api_site_endpoint(local_site):
    from fastapi.testclient import TestClient
    from control_backend.main import app

    client = TestClient(app)
    resp = client.get("/api/site")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "local-default"
    assert data["lab_mode"] is False
    assert "10.0.0.0/8" in data["enterprise_cidrs"]


def test_api_topology_is_site_hint_not_fixed_attacker(lab_site):
    from fastapi.testclient import TestClient
    from control_backend.main import app
    from control_backend.topology_service import topology_service

    topology_service.reset()
    client = TestClient(app)
    resp = client.get("/api/topology")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "topology_update"
    assert data["site_id"] == "containerlab-enterprise"
    assert "attacker_zone" not in data
    # Empty until flows — no permanent attacker glyph in API
    assert all(n.get("id") != "attacker" for n in data.get("nodes", []))
