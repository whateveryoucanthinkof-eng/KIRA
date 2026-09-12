import asyncio
import json
import logging
import os
import random
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fake_backend")

app = FastAPI(title="CyberWorld SOC - Fake Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo Simulation State
demo_state = {
    "is_attack_triggered": False,
    "attack_type": "DDoS Flood",
    "trigger_time": 0.0,
    "attack_phase": 0,  # 0: normal, 1: predicted early spike, 2: actual observed spike
    "current_observed": 15.0,
    "current_predicted": 12.0,
    "current_risk": 0.12,
    "current_throughput": 45.0,
    "current_anomaly": 8.0,
    "uptime_seconds": 7200,
    "window_id": 1,
}

active_connections = set()

FRONTEND_DIST_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "web_dashboard", "dist")
)


async def broadcast_event(event: Dict[str, Any]):
    """Broadcast an event to all connected dashboard websockets."""
    if not active_connections:
        return
    message = json.dumps(event)
    to_remove = set()
    for ws in list(active_connections):
        try:
            await ws.send_text(message)
        except Exception:
            to_remove.add(ws)
    for ws in to_remove:
        active_connections.discard(ws)


def stop_or_cancel_attack():
    """Immediately disarm the attack state and broadcast cancellation."""
    global demo_state
    demo_state["is_attack_triggered"] = False
    demo_state["attack_phase"] = 0
    logger.info("Attack cancelled/stopped. Restoring baseline metrics.")


@app.post("/api/trigger")
async def trigger_attack(attack_type: str = "DDoS Flood"):
    """Trigger the attack animation: predicted spikes first, then observed follows."""
    global demo_state
    demo_state["is_attack_triggered"] = True
    demo_state["attack_type"] = attack_type
    demo_state["trigger_time"] = time.time()
    demo_state["attack_phase"] = 1
    logger.info(f"Triggered attack simulation: {attack_type}")

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    timestr = time.strftime("%H:%M:%S")

    # Broadcast log warning
    await broadcast_event({
        "type": "command_output",
        "command": "attack_trigger",
        "line": f"[{timestr}] [EARLY-WARNING] Anomaly detected in forecast horizon. High risk anticipated for {attack_type}!",
        "timestamp": now_iso,
    })

    return {"status": "ok", "message": f"Triggered {attack_type} attack animation"}


@app.post("/api/cancel")
async def cancel_attack():
    """Cancel the ongoing attack and bring down graphs and metrics."""
    stop_or_cancel_attack()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    timestr = time.strftime("%H:%M:%S")

    # Clear threats in UI
    await broadcast_event({
        "type": "attack_event",
        "id": "atk-sim-1",
        "stage": "STOPPED",
        "mitigated": True,
        "timestamp": now_iso,
    })

    await broadcast_event({
        "type": "command_completed",
        "command": "stop_attack",
        "timestamp": now_iso,
    })

    await broadcast_event({
        "type": "command_output",
        "command": "mitigate",
        "line": f"[{timestr}] [SOAR] Attack cancelled. Active defenses engaged. Telemetry returning to baseline.",
        "timestamp": now_iso,
    })

    return {"status": "ok", "message": "Attack cancelled and dashboard returning to normal"}


@app.post("/api/reset")
async def reset_demo():
    """Reset the dashboard immediately to normal baseline."""
    global demo_state
    stop_or_cancel_attack()
    demo_state["current_observed"] = 15.0
    demo_state["current_predicted"] = 12.0
    demo_state["current_risk"] = 0.12
    demo_state["current_throughput"] = 45.0
    demo_state["current_anomaly"] = 8.0

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    await broadcast_event({
        "type": "attack_event",
        "id": "atk-sim-1",
        "stage": "STOPPED",
        "mitigated": True,
        "timestamp": now_iso,
    })
    await broadcast_event({
        "type": "ml_reset",
        "timestamp": now_iso,
    })
    return {"status": "ok", "message": "Demo reset to baseline"}


@app.post("/api/command/{command_name}")
async def handle_command(command_name: str):
    """Handle UI buttons in the Controls tab."""
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    timestr = time.strftime("%H:%M:%S")

    await broadcast_event({
        "type": "command_started",
        "command": command_name,
        "timestamp": now_iso,
    })

    if command_name in ("stop_attack", "reset_environment", "stop_network"):
        stop_or_cancel_attack()
        await broadcast_event({
            "type": "attack_event",
            "id": "atk-sim-1",
            "stage": "STOPPED",
            "mitigated": True,
            "timestamp": now_iso,
        })
        msg = f"[{timestr}] [CONTROL] Executed {command_name}. Attack disarmed and baseline restored."
    elif command_name in ("start_attack", "arm_external"):
        demo_state["is_attack_triggered"] = True
        demo_state["trigger_time"] = time.time()
        demo_state["attack_phase"] = 1
        msg = f"[{timestr}] [CONTROL] External attack monitor armed."
    else:
        msg = f"[{timestr}] [CONTROL] Executed command: {command_name}."

    await broadcast_event({
        "type": "command_output",
        "command": command_name,
        "line": msg,
        "timestamp": now_iso,
    })

    await broadcast_event({
        "type": "command_completed",
        "command": command_name,
        "timestamp": now_iso,
    })

    return {"status": "ok", "command": command_name}


@app.post("/api/mitigate")
async def handle_mitigate(request: Request):
    """Handle defensive SOAR buttons (Isolate Host, Block IP, Block Port, etc.)."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    action = data.get("action", "MITIGATE")
    target = data.get("target", "ALL")
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    timestr = time.strftime("%H:%M:%S")

    # Cancels the attack and restores normal metrics
    stop_or_cancel_attack()

    await broadcast_event({
        "type": "attack_event",
        "id": "atk-sim-1",
        "stage": "STOPPED",
        "mitigated": True,
        "timestamp": now_iso,
    })

    await broadcast_event({
        "type": "command_output",
        "command": "mitigate",
        "line": f"[{timestr}] [SOAR] Mitigation applied: {action} ({target}). Traffic successfully mitigated.",
        "timestamp": now_iso,
    })

    return {"status": "ok", "action": action, "target": target}


@app.get("/api/status")
async def get_system_status():
    return {
        "network": "running",
        "sensor": "running",
        "ml": "running",
        "attack": "running" if demo_state["is_attack_triggered"] else "stopped",
        "uptime": demo_state["uptime_seconds"],
        "timestamp": time.time(),
        "anomalyScore": round(demo_state["current_anomaly"]),
        "threatLevel": "critical" if demo_state["attack_phase"] == 2 else ("elevated" if demo_state["attack_phase"] == 1 else "low"),
        "packetLoss": 8.5 if demo_state["attack_phase"] == 2 else 0.0,
        "latency": 320 if demo_state["attack_phase"] == 2 else 12,
        "throughput": round(demo_state["current_throughput"]),
        "activeConnections": 15200 if demo_state["attack_phase"] == 2 else 120,
    }


@app.get("/api/site")
async def get_site():
    return {
        "name": "Enterprise Data Center (Demo)",
        "location": "Virtual Demonstration",
        "timezone": "UTC",
        "subnet": "10.0.0.0/8",
        "externalIp": "192.168.100.10",
        "description": "Demonstration Environment",
    }


@app.get("/api/topology")
async def get_topology():
    is_hot = demo_state["attack_phase"] == 2
    return {
        "nodes": [
            {"id": "ext", "label": "Internet", "role": "external", "status": "online"},
            {"id": "fw", "label": "Core Firewall", "role": "sensor", "status": "online"},
            {"id": "web1", "label": "Web Server 1", "role": "host", "status": "compromised" if is_hot else ("warning" if demo_state["is_attack_triggered"] else "online")},
            {"id": "web2", "label": "Web Server 2", "role": "host", "status": "online"},
            {"id": "db", "label": "Database", "role": "host", "status": "online"},
        ],
        "edges": [
            {"src": "ext", "dst": "fw", "protocol": 6, "bytes": 1000000000 if is_hot else 1000000},
            {"src": "fw", "dst": "web1", "protocol": 6, "bytes": 850000000 if is_hot else 800000},
            {"src": "fw", "dst": "web2", "protocol": 6, "bytes": 400000},
            {"src": "web1", "dst": "db", "protocol": 6, "bytes": 100000},
            {"src": "web2", "dst": "db", "protocol": 6, "bytes": 100000},
        ],
        "stats": {"nodes": 5, "edges": 5},
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Operator WebSocket client connected. Active: {len(active_connections)}")

    # Send initial greeting
    await websocket.send_text(json.dumps({
        "type": "connected",
        "message": "Connected to CyberWorld SOC Event Bus (Demo Mode)."
    }))

    try:
        while True:
            await asyncio.sleep(1.5)
            demo_state["uptime_seconds"] += 2
            demo_state["window_id"] += 1
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

            # ── Attack Phase State Machine ──
            if demo_state["is_attack_triggered"]:
                elapsed = time.time() - demo_state["trigger_time"]
                if elapsed < 4.0:
                    # Phase 1: PREDICTED SPIKE FIRST (Early Warning!)
                    demo_state["attack_phase"] = 1
                    target_predicted = 88.0 + random.uniform(-2, 2)
                    target_observed = 15.0 + random.uniform(-1, 2)  # Still normal!
                    target_risk = 0.88
                    target_throughput = 48.0
                    target_anomaly = 85.0
                    threat_level = "elevated"
                else:
                    # Phase 2: OBSERVED SPIKE FOLLOWS
                    if demo_state["attack_phase"] == 1:
                        demo_state["attack_phase"] = 2
                        logger.info("Attack traffic arrived: Observed graph shooting up to follow prediction!")
                        timestr = time.strftime("%H:%M:%S")
                        await broadcast_event({
                            "type": "command_output",
                            "command": "attack_impact",
                            "line": f"[{timestr}] [CRITICAL] Incoming {demo_state['attack_type']} flood detected! Core ingress saturated.",
                            "timestamp": now_iso,
                        })

                    demo_state["attack_phase"] = 2
                    target_predicted = 94.0 + random.uniform(-2, 2)
                    target_observed = 92.0 + random.uniform(-3, 3)  # Observed shoots up!
                    target_risk = 0.94
                    target_throughput = 920.0 + random.uniform(-15, 15)
                    target_anomaly = 95.0
                    threat_level = "critical"
            else:
                # Normal Baseline (or Recovering after Cancellation)
                demo_state["attack_phase"] = 0
                target_predicted = 12.0 + random.uniform(-1, 1.5)
                target_observed = 15.0 + random.uniform(-1, 1.5)
                target_risk = 0.12
                target_throughput = 45.0 + random.uniform(-2, 2)
                target_anomaly = 8.0
                threat_level = "low"

            # Smoothly ease current values towards targets
            alpha = 0.65
            demo_state["current_observed"] += (target_observed - demo_state["current_observed"]) * alpha
            demo_state["current_predicted"] += (target_predicted - demo_state["current_predicted"]) * alpha
            demo_state["current_risk"] += (target_risk - demo_state["current_risk"]) * alpha
            demo_state["current_throughput"] += (target_throughput - demo_state["current_throughput"]) * alpha
            demo_state["current_anomaly"] += (target_anomaly - demo_state["current_anomaly"]) * alpha

            cur_obs = round(demo_state["current_observed"], 1)
            cur_pred = round(demo_state["current_predicted"], 1)
            cur_risk = round(demo_state["current_risk"], 2)
            cur_tp = round(demo_state["current_throughput"])
            cur_anom = round(demo_state["current_anomaly"])

            # 1. Prediction Event (drives the Live Traffic Chart and Prediction Summary)
            prediction_event = {
                "type": "prediction",
                "timestamp": now_iso,
                "state": {
                    "active_flows": cur_obs,
                    "packet_count": cur_obs,
                    "window_id": demo_state["window_id"],
                },
                "prediction": {
                    "risk": cur_risk,
                    "malicious_confidence": 0.95 if demo_state["is_attack_triggered"] else 0.35,
                    "hazard_score": cur_risk,
                    "forecast_error": 0.04,
                    "max_future_risk": min(1.0, round(cur_risk + 0.05, 2)),
                },
                "horizon": 30,
                "forecast": [
                    {
                        "horizon_seconds": (i + 1) * 30,
                        "risk": cur_risk,
                        "confidence": 0.92 if demo_state["is_attack_triggered"] else 0.5,
                    }
                    for i in range(8)
                ],
            }

            # 2. System Status Event (drives the Top Metric Cards)
            status_event = {
                "type": "system_status",
                "network": "running",
                "sensor": "running",
                "ml": "running",
                "attack": "running" if demo_state["is_attack_triggered"] else "stopped",
                "uptime": demo_state["uptime_seconds"],
                "timestamp": now_iso,
                "anomalyScore": cur_anom,
                "threatLevel": threat_level,
                "packetLoss": 8.5 if demo_state["attack_phase"] == 2 else 0.0,
                "latency": 320 if demo_state["attack_phase"] == 2 else 12,
                "throughput": cur_tp,
                "activeConnections": 15200 if demo_state["attack_phase"] == 2 else 120,
            }

            try:
                await websocket.send_text(json.dumps(prediction_event))
                await websocket.send_text(json.dumps(status_event))

                # 3. If in Phase 2, push active attack event to Recent Events
                if demo_state["attack_phase"] == 2:
                    attack_event = {
                        "type": "attack_event",
                        "id": "atk-sim-1",
                        "attack_type": demo_state["attack_type"],
                        "stage": "Exploitation",
                        "sourceIp": "192.168.100.10",
                        "targetIp": "10.0.3.10",
                        "packets": 500000,
                        "bytes": 500000000,
                        "mitigated": False,
                        "timestamp": now_iso,
                    }
                    await websocket.send_text(json.dumps(attack_event))

            except Exception as e:
                logger.error(f"Error sending websocket data: {e}")
                break

    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        active_connections.discard(websocket)


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
