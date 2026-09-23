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
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from control_backend.commands import executor, ALLOWED_COMMANDS
from control_backend.event_broker import broker
import model_contract
from control_backend.lab_config import TOTAL_NODES, LAB_NAME_FILTER
from control_backend.site_config import get_site_config
from control_backend.schema import SystemStatusEvent, ModelMetadata, utc_now_iso
from control_backend.telemetry_service import telemetry_service
from control_backend.topology_service import topology_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("antigravity.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    broker.set_loop(asyncio.get_running_loop())
    site = get_site_config()
    logger.info(
        "cyberworld SOC backend online — site=%s lab_mode=%s (discovery topology).",
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
    logger.info("cyberworld SOC backend shutting down.")


app = FastAPI(
    title="cyberworld SOC — SPAN Discovery + Dual-Branch/DeepOP",
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
    # Report the contract the loaded checkpoints actually carry.
    #
    # These were literals -- history_steps=5, forecast_steps=8 -- which were
    # the v3 values. After the v4 retrain the served models use history 15 /
    # forecast 5, so /api/status was telling an operator the wrong temporal
    # contract for the model that was answering their queries. The adapter
    # already adopts the real values from the checkpoints in _adopt_contract
    # and refuses to load if they disagree with each other, so it is the one
    # source worth reporting.
    from control_backend.model_adapter import model_adapter as _served
    model_meta = ModelMetadata(
        name="Antigravity-DualBranch-DeepOP",
        version="3.3-SOC",
        feature_count=model_contract.BRANCH_A_INPUT_DIM,
        history_steps=_served.history_steps,
        window_seconds=_served.window_seconds,
        forecast_steps=_served.forecast_steps,
        checkpoint="host_wdt.pt + branch_a_lstm.pt + cwa_forecast_decoder.pt",
        threshold=_served.alert_threshold,
    )

    now_iso = utc_now_iso()
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
        uptime=int(time.time() - getattr(telemetry_service, "service_start_time", time.time())),
        throughput=getattr(telemetry_service, "current_throughput", 35.0),
        latency=getattr(telemetry_service, "current_latency", 8.5),
        packetLoss=getattr(telemetry_service, "current_packet_loss", 0.0),
        activeConnections=getattr(telemetry_service, "current_active_connections", 24),
        anomalyScore=getattr(telemetry_service, "current_anomaly_score", 8.0),
        threatLevel=getattr(telemetry_service, "current_threat_level", "low"),
        timestamp=now_iso,
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


# A private adapter instance dedicated to /api/replay -- see
# _get_replay_adapter() below for why replay must never score on the live
# `model_adapter` singleton. Built lazily (once) on first use, not at import
# time, so importing this module never pays the model-loading cost and test
# collection doesn't require checkpoints on disk unless /api/replay actually
# runs.
_replay_adapter: Optional[Any] = None


def _get_replay_adapter():
    """Return the adapter /api/replay scores against -- never the live singleton.

    Replay used to call `predict_window()` directly on `model_adapter`, the
    SAME instance the live tail worker scores on concurrently, and mutated
    its shared state: `rules_enabled` (restored in `finally`) and
    `reset_history()` -- NOT restored, and not scoped to any one target
    either: `AntigravityModelAdapter.reset_history()` clears
    `h_state_history_by_target` and `feature_history_by_target` in full, so
    one `/api/replay` call wiped the rolling window for every host the live
    tail worker was tracking, not just whatever the upload analysed. Past
    that, for the rest of the replay run any window whose target IP
    coincides with a live host's IP (likely -- both draw from the same
    site's asset IPs) would splice replayed windows into that host's real
    rolling history as it rebuilds.

    A private `AntigravityModelAdapter` instance starts with its own, empty
    `feature_history_by_target`/`h_state_history_by_target` dicts and its own
    `rules_enabled`, so replay can never read or clear a live host's history
    or toggle rules out from under the tail worker, no matter what IPs appear
    in the uploaded capture and no matter how the two overlap in time. Live
    history is therefore untouched by a replay, always.
    """
    global _replay_adapter
    if _replay_adapter is None:
        from control_backend.model_adapter import AntigravityModelAdapter
        _replay_adapter = AntigravityModelAdapter()
    return _replay_adapter


@app.post("/api/replay")
async def replay_file(file: UploadFile = File(...), max_windows: int = 200):
    """Offline analysis of an uploaded capture or flow CSV.

    PS 26153 asks for a demo interface accepting "a PCAP or CSV file", running
    fully offline. The live path only ever consumed a SPAN feed, so a CSV of
    flow records — the form both CIC-IDS-2018 and CTU-13 ship in — had no way in
    at all.

    Runs entirely locally: no network egress, no cloud dependency. The upload is
    written to a temp file, parsed, scored window by window, and deleted.

    Scored on a private adapter instance (see _get_replay_adapter()), never on
    the live serving singleton, so a replay can run concurrently with the live
    tail worker without corrupting its state.

    Rules are disabled for this path regardless of the server's setting: an
    offline analysis is an evaluation, and a heuristic that floors risk at 0.40
    would make every uploaded file look alarming.
    """
    import tempfile

    from control_backend.model_adapter import select_primary_target

    replay_adapter = _get_replay_adapter()

    name = (file.filename or "upload").lower()
    suffix = Path(name).suffix
    if suffix not in (".pcap", ".pcapng", ".csv", ".binetflow", ""):
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Expected .pcap, .pcapng or .csv")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    tmp = Path(tempfile.mkstemp(suffix=suffix or ".bin")[1])
    tmp.write_bytes(data)

    try:
        # replay_adapter is private to this endpoint (see _get_replay_adapter),
        # so forcing rules off and clearing history here can never affect the
        # live tail worker's adapter or its state.
        replay_adapter.rules_enabled = False
        replay_adapter.reset_history()

        if suffix in (".csv", ".binetflow"):
            records = _parse_flow_file(tmp, suffix)
            windows = _group_records_into_windows(records, replay_adapter.window_seconds)
        else:
            windows = _replay_pcap_windows(tmp, max_windows, replay_adapter.window_seconds)

        results = []
        for widx, flows in enumerate(windows[:max_windows]):
            if not flows:
                continue
            target = select_primary_target(flows)
            if not target:
                continue
            ev = replay_adapter.predict_window(target_ip=target, flows=flows, window_id=widx)
            results.append({
                "window": widx,
                "target": target,
                "risk": ev.prediction.risk,
                "ml_risk": ev.prediction.ml_risk,
                "alert": ev.prediction.alert,
                "stage": ev.prediction.predicted_stage,
                "mitre_tactic": ev.prediction.mitre_tactic,
                "mitre_technique": ev.prediction.mitre_technique,
                "forecast": [
                    {"horizon_seconds": f.horizon_seconds, "risk": f.risk} for f in ev.forecast
                ],
                "top_features": [
                    {"feature": f.feature, "score": f.score, "group": f.group}
                    for f in ev.explainability.top_features[:5]
                ],
            })

        flagged = [r for r in results if r["alert"]]
        return {
            "filename": file.filename,
            "kind": "csv" if suffix in (".csv", ".binetflow") else "pcap",
            "windows_analyzed": len(results),
            "flagged_windows": len(flagged),
            "rules_disabled": True,
            "results": results,
        }
    finally:
        tmp.unlink(missing_ok=True)


def _parse_flow_file(path: Path, suffix: str):
    """CSV/binetflow -> UnifiedFlowRecord via the same adapters training uses."""
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter

    if suffix == ".binetflow":
        return list(CTU13Adapter().parse_netflow_csv(str(path), max_rows=100000))
    return list(CIC2018Adapter().parse_file(str(path), max_rows=100000))


def _group_records_into_windows(records, window_seconds: float):
    """Bucket flow records into fixed windows by end_time."""
    if not records:
        return []
    recs = sorted(records, key=lambda r: r.end_time)
    t0 = recs[0].end_time
    buckets: Dict[int, list] = {}
    for r in recs:
        buckets.setdefault(int((r.end_time - t0) // window_seconds), []).append(r)
    return [buckets[k] for k in sorted(buckets)]


def _replay_pcap_windows(path: Path, max_windows: int, window_seconds: float):
    """Reuse the sensor's own window builder so replay matches live exactly.

    `window_seconds` is passed in by the caller (the replay adapter's own
    served contract) rather than read from the live `model_adapter` singleton
    here, so this function never touches live serving state.
    """
    import struct

    from telemetry.capture.sniffer import StreamingPacketSniffer
    from telemetry.state.state_builder import LiveStateBuilder
    from control_backend.model_adapter import flows_from_span_dicts

    sb = LiveStateBuilder(window_sec=window_seconds)
    windows, anchored = [], False
    with open(path, "rb") as f:
        if len(f.read(24)) < 24:
            raise HTTPException(400, "Not a valid PCAP file")
        while len(windows) < max_windows:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            sec, usec, incl, _orig = struct.unpack("=IIII", hdr)
            ts = sec + usec / 1e6
            if not anchored:
                sb.seek_to(ts)
                anchored = True
            pkt = StreamingPacketSniffer.parse_frame(f.read(incl), ts)
            if pkt:
                sb.ingest_packet(pkt)
            if sb.is_window_ready(current_time=ts):
                st = sb.close_window(close_ts=ts)
                windows.append(flows_from_span_dicts(st.get("flows") or []))
    return windows


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
        return FileResponse(
            os.path.join(FRONTEND_DIST_PATH, "index.html"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
