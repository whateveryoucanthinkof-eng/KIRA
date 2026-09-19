"""
tests/test_topology_service.py
Phase 2–5: discovery topology, TTL, CIDR roles, API contract.
"""

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "bita"))

from control_backend.site_config import reload_site_config, get_site_config
from control_backend.topology_service import TopologyService
from control_backend.main import app
from control_backend.commands import executor, LAB_ONLY_COMMANDS


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


def test_empty_until_traffic(local_site):
    svc = TopologyService(site=local_site)
    snap = svc.snapshot(now=1000.0)
    assert snap.stats.nodes == 0
    assert snap.nodes == []


def test_external_appears_only_when_observed(local_site):
    svc = TopologyService(site=local_site)
    now = 2000.0
    evt = svc.apply_window(
        [
            {
                "src_ip": "203.0.113.10",
                "dst_ip": "10.50.1.20",
                "src_port": 4444,
                "dst_port": 80,
                "protocol": 6,
                "fwd_bytes": 900,
                "bwd_bytes": 1200,
                "fwd_packets": 5,
                "bwd_packets": 8,
                "end_time": now,
            }
        ],
        now=now,
    )
    roles = {n.ip: n.role for n in evt.nodes}
    assert roles["10.50.1.20"] == "internal"
    assert roles["203.0.113.10"] == "external"
    assert evt.stats.external_nodes == 1
    assert len(evt.edges) == 1


def test_ttl_evicts_idle_nodes(local_site):
    svc = TopologyService(site=local_site)
    t0 = 3000.0
    svc.apply_window(
        [
            {
                "src_ip": "10.1.1.1",
                "dst_ip": "10.1.1.2",
                "protocol": 6,
                "dst_port": 443,
                "fwd_bytes": 100,
                "bwd_bytes": 100,
                "fwd_packets": 1,
                "bwd_packets": 1,
                "end_time": t0,
            }
        ],
        now=t0,
    )
    assert svc.snapshot(now=t0).stats.nodes == 2
    # Past node_ttl_sec (120)
    later = t0 + local_site.node_ttl_sec + 5
    snap = svc.snapshot(now=later)
    assert snap.stats.nodes == 0
    assert snap.stats.edges == 0


def test_max_nodes_cap(local_site):
    from dataclasses import replace

    tiny = replace(local_site, max_nodes=3, node_ttl_sec=3600)
    svc = TopologyService(site=tiny)
    now = 4000.0
    flows = []
    for i in range(6):
        flows.append(
            {
                "src_ip": f"10.9.0.{i}",
                "dst_ip": "10.9.0.100",
                "protocol": 6,
                "dst_port": 80,
                "fwd_bytes": 10 + i,
                "bwd_bytes": 10,
                "fwd_packets": 1,
                "bwd_packets": 1,
                "end_time": now + i,
            }
        )
        svc.apply_window([flows[-1]], now=now + i)
    snap = svc.snapshot(now=now + 10)
    assert snap.stats.nodes <= 3


def test_api_topology_live_graph(local_site):
    from control_backend.topology_service import topology_service

    topology_service.reset()
    topology_service._site_override = local_site
    try:
        topology_service.apply_window(
            [
                {
                    "src_ip": "203.0.113.10",
                    "dst_ip": "10.50.1.20",
                    "protocol": 6,
                    "dst_port": 80,
                    "fwd_bytes": 500,
                    "bwd_bytes": 500,
                    "fwd_packets": 2,
                    "bwd_packets": 2,
                    "end_time": time.time(),
                }
            ]
        )
        client = TestClient(app)
        resp = client.get("/api/topology")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "topology_update"
        assert data["stats"]["nodes"] >= 2
        ips = {n["ip"] for n in data["nodes"]}
        assert "203.0.113.10" in ips
        assert "10.50.1.20" in ips
        assert all(n.get("id") != "attacker" for n in data["nodes"])
        assert all(n.get("id") != "dmz-web" for n in data["nodes"])
    finally:
        topology_service._site_override = None
        topology_service.reset()


def test_api_status_exposes_lab_mode(local_site):
    client = TestClient(app)
    data = client.get("/api/status").json()
    assert data["lab_mode"] is False
    assert data["site_id"] == "local-default"
    assert "topology_nodes" in data


def test_lab_commands_blocked_when_not_lab_mode(local_site):
    with pytest.raises(RuntimeError, match="Lab Mode"):
        executor.execute_command("start_network")
    assert "start_network" in LAB_ONLY_COMMANDS


def test_lab_commands_allowed_in_lab_mode(lab_site, monkeypatch):
    # Don't actually run deploy — just verify gate passes into busy/shell path.
    # start_network is shell-backed; stub _run_shell_worker to no-op after gate.
    called = {}

    def fake_start(self, cmd_name):
        called["cmd"] = cmd_name
        with self.lock:
            self.active_command = None

    monkeypatch.setattr(type(executor), "_run_shell_worker", fake_start)
    res = executor.execute_command("start_network")
    assert res["status"] == "started"
    # Give thread a moment if started
    time.sleep(0.05)
