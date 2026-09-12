import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# State
demo_state = {
    "is_attack_triggered": False,
    "attack_type": None,
    "trigger_time": 0.0,
    "attack_phase": 0, # 0: normal, 1: predicted spike, 2: actual spike
}

active_connections = set()

FRONTEND_DIST_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "web_dashboard", "dist")
)

@app.post("/api/trigger")
async def trigger_attack(attack_type: str = "DDoS"):
    """Trigger the attack animation on the dashboard."""
    global demo_state
    demo_state["is_attack_triggered"] = True
    demo_state["attack_type"] = attack_type
    demo_state["trigger_time"] = time.time()
    demo_state["attack_phase"] = 1
    logger.info(f"Triggered fake attack: {attack_type}")
    return {"status": "ok", "message": f"Triggered {attack_type} attack animation"}

@app.post("/api/reset")
async def reset_demo():
    """Reset the dashboard to normal state."""
    global demo_state
    demo_state["is_attack_triggered"] = False
    demo_state["attack_type"] = None
    demo_state["trigger_time"] = 0.0
    demo_state["attack_phase"] = 0
    logger.info("Reset fake demo state to normal.")
    return {"status": "ok", "message": "Demo reset to normal"}

@app.get("/api/status")
async def get_system_status():
    return {
        "network": "running",
        "sensor": "running",
        "ml": "running",
        "attack": "active" if demo_state["is_attack_triggered"] else "none",
        "uptime": 7200,
        "timestamp": time.time(),
        "anomalyScore": 95 if demo_state["attack_phase"] >= 2 else (85 if demo_state["attack_phase"] == 1 else 12),
        "threatLevel": "high" if demo_state["is_attack_triggered"] else "low",
        "packetLoss": 12.5 if demo_state["attack_phase"] >= 2 else 0.1,
        "latency": 450 if demo_state["attack_phase"] >= 2 else 15,
        "throughput": 9500 if demo_state["attack_phase"] >= 2 else 450,
        "activeConnections": 15000 if demo_state["attack_phase"] >= 2 else 1240,
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
    return {
        "nodes": [
            {"id": "ext", "label": "Internet", "role": "external", "status": "online"},
            {"id": "fw", "label": "Core Firewall", "role": "sensor", "status": "online"},
            {"id": "web1", "label": "Web Server 1", "role": "host", "status": "warning" if demo_state["is_attack_triggered"] else "online"},
            {"id": "web2", "label": "Web Server 2", "role": "host", "status": "online"},
            {"id": "db", "label": "Database", "role": "host", "status": "online"},
        ],
        "edges": [
            {"src": "ext", "dst": "fw", "protocol": 6, "bytes": 1000000000 if demo_state["attack_phase"] >= 2 else 1000000},
            {"src": "fw", "dst": "web1", "protocol": 6, "bytes": 800000000 if demo_state["attack_phase"] >= 2 else 800000},
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
    try:
        while True:
            await asyncio.sleep(2.0)
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            
            # Logic to progress phases
            if demo_state["is_attack_triggered"]:
                elapsed = time.time() - demo_state["trigger_time"]
                if elapsed > 10 and demo_state["attack_phase"] == 1:
                    demo_state["attack_phase"] = 2
                    logger.info("Moving to attack phase 2 (Actual traffic spike)")
            
            # Normal values
            risk_value = 0.1
            predicted_value = 0.15
            throughput = 450
            
            if demo_state["attack_phase"] == 1:
                # Predicted spike
                risk_value = 0.85
                predicted_value = 0.9
                throughput = 450 # Actual still normal
            elif demo_state["attack_phase"] == 2:
                # Actual spike
                risk_value = 0.95
                predicted_value = 0.95
                throughput = 850 # Nearing capacity
            
            prediction_event = {
                "type": "prediction",
                "timestamp": now_iso,
                "prediction": {
                    "risk": risk_value,
                    "malicious_confidence": 0.92 if demo_state["is_attack_triggered"] else 0.4,
                    "hazard_score": risk_value,
                    "forecast_error": 0.05,
                    "max_future_risk": risk_value + 0.05
                },
                "horizon": 30
            }
            
            status_event = {
                "type": "system_status",
                "network": "running",
                "sensor": "running",
                "ml": "running",
                "attack": "active" if demo_state["is_attack_triggered"] else "none",
                "uptime": 7200,
                "timestamp": now_iso,
                "anomalyScore": risk_value * 100,
                "threatLevel": "high" if demo_state["is_attack_triggered"] else "low",
                "packetLoss": 12.5 if demo_state["attack_phase"] >= 2 else 0.1,
                "latency": 450 if demo_state["attack_phase"] >= 2 else 15,
                "throughput": throughput,
                "activeConnections": 15000 if demo_state["attack_phase"] >= 2 else 1240,
            }
            
            try:
                await websocket.send_text(json.dumps(prediction_event))
                await websocket.send_text(json.dumps(status_event))
                
                # If phase 2, send attack event
                if demo_state["attack_phase"] >= 2:
                    attack_event = {
                        "type": "attack_event",
                        "id": f"atk-{int(time.time())}",
                        "attack_type": demo_state["attack_type"],
                        "stage": "Exploitation",
                        "sourceIp": "192.168.100.10",
                        "targetIp": "10.0.3.10",
                        "packets": 500000,
                        "bytes": 500000000,
                        "mitigated": False,
                        "timestamp": now_iso
                    }
                    await websocket.send_text(json.dumps(attack_event))
                    
            except Exception as e:
                logger.error(f"Error sending to ws: {e}")
                break
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            active_connections.remove(websocket)
        except KeyError:
            pass


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
