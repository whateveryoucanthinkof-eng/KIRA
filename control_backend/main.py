"""
control_backend/main.py
FastAPI Control Backend for the V3 SOC dashboard — wired to real Containerlab.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from control_backend.commands import executor, ALLOWED_COMMANDS
from control_backend.event_broker import broker
from control_backend.lab_config import TOTAL_NODES, LAB_NAME_FILTER
from control_backend.site_config import get_site_config
from control_backend.schema import SystemStatusEvent, ModelMetadata
from control_backend.telemetry_service import telemetry_service
from control_backend.topology_service import topology_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("antigravity.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    broker.set_loop(asyncio.get_running_loop())
    site = get_site_config()
    logger.info(
        "CyberWorld SOC backend online — site=%s lab_mode=%s (discovery topology).",
        site.site_id,
        site.lab_mode,
    )
    yield
    try:
        telemetry_service.stop_ml()
    except Exception:
        pass
    try:
        telemetry_service.stop_sensor()
    except Exception:
        pass
    logger.info("CyberWorld SOC backend shutting down.")


app = FastAPI(
    title="CyberWorld SOC — SPAN Discovery + Dual-Branch/DeepOP",
    version="3.3.0-discovery",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "web_dashboard", "dist")
)


def _count_lab_containers() -> list[str]:
    try:
        res = subprocess.run(
            ["podman", "ps", "--filter", f"name={LAB_NAME_FILTER}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.stdout.strip():
            return [c.strip() for c in res.stdout.strip().split("\n") if c.strip()]
    except Exception:
        pass
    return []


@app.get("/api/status", response_model=SystemStatusEvent)
async def get_system_status():
    site = get_site_config()
    topo = topology_service.snapshot()
    containers = _count_lab_containers() if site.lab_mode else []
    nodes_running = len(containers)

    # Lab Mode: "online" = Containerlab up. Non-lab: sensor streaming / recent windows.
    stream_fresh = (
        telemetry_service.is_running
        and telemetry_service.last_window_at > 0
        and (time.time() - telemetry_service.last_window_at) < 30.0
    )
    if site.lab_mode:
        network_online = nodes_running >= 12
    else:
        network_online = bool(telemetry_service.is_running or stream_fresh)
    executor.network_online = network_online

    status = executor.get_status()
    active_cmd = status.get("active_command")

    if active_cmd == "start_network":
        network_state = "starting"
    elif active_cmd == "stop_network":
        network_state = "stopping"
    elif network_online:
        network_state = "running"
    else:
        network_state = "stopped"

    if active_cmd == "start_telemetry":
        sensor_state = "starting"
    elif active_cmd == "stop_telemetry":
        sensor_state = "stopping"
    elif telemetry_service.is_running:
        sensor_state = "running"
    else:
        sensor_state = "stopped"

    workloads_running = bool(site.lab_mode and network_online and executor.is_workloads_running())
    attack_running = bool(executor.is_attack_running())

    if active_cmd == "start_ml":
        ml_state = "starting"
    elif active_cmd == "stop_ml":
        ml_state = "stopping"
    elif telemetry_service.is_ml_active:
        ml_state = "running"
    else:
        ml_state = "stopped"

    ml_live = ml_state == "running"
    model_meta = ModelMetadata(
        name="Antigravity-DualBranch-DeepOP",
        version="3.3-SOC",
        feature_count=27,
        history_steps=5,
        window_seconds=2.0,
        forecast_steps=8,
        checkpoint="host_wdt.pt + branch_a_lstm.pt + cwa_forecast_decoder.pt",
        threshold=0.65,
    )

    return SystemStatusEvent(
        type="system_status",
        mode="LIVE" if (network_online or telemetry_service.is_running or ml_live) else "STANDBY",
        network=network_state,
        sensor=sensor_state,
        normal_traffic="running" if workloads_running else "stopped",
        attack="running" if attack_running else "stopped",
        ml=ml_state,
        network_online=network_online,
        nodes_running=nodes_running if site.lab_mode else topo.stats.nodes,
        total_nodes=TOTAL_NODES if site.lab_mode else max(topo.stats.nodes, site.max_nodes),
        sensor_active=telemetry_service.is_running,
        telemetry_active=telemetry_service.is_running,
        ml_active=telemetry_service.is_ml_active,
        ml_status="live" if ml_live else "standby",
        workloads_active=workloads_running,
        attack_active=attack_running,
        demo_active=False,
        active_command=active_cmd,
        model_loaded=True,
        model_meta=model_meta,
        lab_mode=site.lab_mode,
        site_id=site.site_id,
        topology_nodes=topo.stats.nodes,
        topology_edges=topo.stats.edges,
        sensor_interface=site.sensor_interface,
    )


@app.get("/api/scenarios")
async def get_scenarios():
    """External-traffic mode: no synthetic scenario playback; no fixed attacker node."""
    site = get_site_config()
    assets = site.asset_ips()
    return [
        {
            "id": "external",
            "title": "External traffic (observed via SPAN)",
            "description": site.external_traffic_hint(),
            "difficulty": "operator",
            "target_host": assets[0] if assets else None,
            "attacker_ip": None,
            "steps": 0,
            "site_id": site.site_id,
        }
    ]


@app.get("/api/site")
async def get_site():
    """Active site profile (CIDRs, lab_mode, sensor, assets)."""
    return get_site_config().as_public_dict()


@app.get("/api/topology")
async def get_topology():
    """Live discovery graph from observed SPAN flows (empty until traffic)."""
    evt = topology_service.snapshot()
    return evt.dict() if hasattr(evt, "dict") else evt.model_dump()


@app.post("/api/command/{cmd_name}")
async def trigger_command(cmd_name: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    if cmd_name not in ALLOWED_COMMANDS:
        raise HTTPException(
            status_code=400,
            detail=f"Command '{cmd_name}' not allowed. Allowed: {list(ALLOWED_COMMANDS.keys())}",
        )
    try:
        return executor.execute_command(cmd_name, payload or {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mitigate")
async def trigger_mitigation(payload: Dict[str, Any] = Body(...)):
    action = payload.get("action", "ISOLATE_HOST")
    target = payload.get("target")
    msg = telemetry_service.apply_mitigation(action, target)
    return {"status": "applied", "message": msg}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await broker.connect(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if data.get("action") == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": data.get("timestamp")})
                    )
            except Exception:
                pass
    except WebSocketDisconnect:
        broker.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        broker.disconnect(websocket)


if os.path.exists(FRONTEND_DIST_PATH):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(FRONTEND_DIST_PATH, "assets")),
        name="assets",
    )

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_react_app(full_path: str):
        file_path = os.path.join(FRONTEND_DIST_PATH, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST_PATH, "index.html"))
