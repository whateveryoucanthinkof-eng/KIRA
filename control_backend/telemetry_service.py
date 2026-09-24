"""
control_backend/telemetry_service.py
Live Containerlab SPAN telemetry + Dual-Branch / DeepOP inference.
SPAN capture via scripts/run_telemetry.sh (--no-inference); ML runs in-process.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Any

from control_backend.capture_accounting import CaptureAccounting
from control_backend.event_broker import broker
from control_backend.lab_config import (
    MONOREPO_ROOT,
    STATE_STREAM_PATH,
    ML_TRIGGER_FILE,
    SENSOR_CONTAINER,
)
from control_backend.model_adapter import (
    model_adapter,
    flows_from_span_dicts,
    select_primary_target,
)
from control_backend.schema import CommandEvent, utc_now_iso
from control_backend.topology_service import topology_service

logger = logging.getLogger("antigravity.telemetry_service")


class LiveTelemetryService:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.tail_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_ml_active = False
        self.adapter = model_adapter
        self.windows_streamed = 0
        self.last_window_at: float = 0.0
        self.external_attack_armed = False

        self.service_start_time: float = time.time()
        # Measured values only. These used to start at plausible-looking
        # constants (35 Mbps, 8.5 ms, 24 connections) that the status bar
        # showed before a single packet had been seen.
        self.current_throughput: float = 0.0
        self.current_latency: float = 0.0
        self.current_packet_loss: float = 0.0
        self.current_active_connections: int = 0
        self.current_anomaly_score: float = 0.0
        self.current_threat_level: str = "low"
        # Windows never delivered, or built while the kernel dropped frames.
        self.capture = CaptureAccounting()

        self.isolated_hosts: set = set()
        self.blocked_ips: set = set()
        self.blocked_ports: set = set()
        self.credentials_revoked: bool = False

        if os.path.exists(ML_TRIGGER_FILE):
            try:
                os.remove(ML_TRIGGER_FILE)
            except Exception:
                pass

    def start_sensor(self) -> Dict[str, Any]:
        if self.is_running:
            return {"status": "already_running"}

        for path in (STATE_STREAM_PATH, ML_TRIGGER_FILE):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        from control_backend.site_config import env, get_site_config

        site = get_site_config()
        sensor_name = site.sensor_container or SENSOR_CONTAINER
        iface = site.sensor_interface or "eth1"

        # Lab sensor path: require container when site declares one (typical Containerlab).
        if site.sensor_container:
            check = subprocess.run(
                ["podman", "ps", "--filter", f"name={sensor_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
            )
            if sensor_name not in check.stdout:
                raise RuntimeError(
                    f"Cannot start telemetry: {sensor_name} is not running. "
                    "Start the Containerlab network first (Lab Mode), or switch site profile."
                )
            cmd = [
                "./scripts/run_telemetry.sh",
                "--record-state",
                STATE_STREAM_PATH,
                "--no-inference",
            ]
        else:
            # Local SPAN: sniff the configured NIC directly (needs CAP_NET_RAW / root).
            py = sys.executable
            cmd = [
                py,
                "telemetry/run_telemetry.py",
                "--interface",
                iface,
                "--record-state",
                STATE_STREAM_PATH,
                "--no-inference",
            ]
            replay_pcap = env("REPLAY_PCAP")
            if replay_pcap:
                cmd.extend(["--replay", replay_pcap])

        logger.info(
            "Launching SPAN telemetry (site=%s iface=%s lab=%s): %s",
            site.site_id,
            iface,
            site.lab_mode,
            " ".join(cmd),
        )
        self.process = subprocess.Popen(
            cmd,
            cwd=MONOREPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.is_running = True
        self.is_ml_active = False
        self.windows_streamed = 0
        self.capture.reset_sequence()
        self.last_window_at = 0.0
        self.adapter.reset_history()
        topology_service.reset()

        self.tail_thread = threading.Thread(target=self._tail_worker, daemon=True)
        self.tail_thread.start()
        threading.Thread(target=self._stdout_logger, daemon=True).start()

        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="start_telemetry",
                timestamp=utc_now_iso(),
                line=f"[+] Passive SPAN sensor active on {iface} (flows → Dual-Branch/DeepOP).",
            )
        )
        return {"status": "started", "interface": iface, "ml_status": "standby"}

    def stop_sensor(self) -> Dict[str, Any]:
        self.is_running = False
        self.is_ml_active = False

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

        broker.clear_prediction()
        topology_service.reset()
        empty = topology_service.snapshot()
        broker.broadcast_sync(empty)
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="stop_telemetry",
                timestamp=utc_now_iso(),
                line="[-] Passive SPAN sensor stopped.",
            )
        )
        return {"status": "stopped", "windows_streamed": self.windows_streamed}

    def start_ml(self) -> Dict[str, Any]:
        if not self.is_running:
            raise RuntimeError("Cannot start ML: telemetry sensor must be running first.")
        if self.is_ml_active:
            return {"status": "already_active"}

        self.is_ml_active = True
        self.adapter.reset_history()
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="start_ml",
                timestamp=utc_now_iso(),
                line="[★] Antigravity Dual-Branch + DeepOP CWA LIVE on SPAN flows.",
            )
        )
        return {"status": "started", "ml_status": "live"}

    def stop_ml(self) -> Dict[str, Any]:
        self.is_ml_active = False
        self.adapter.reset_history()
        broker.clear_prediction()
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="stop_ml",
                timestamp=utc_now_iso(),
                line="[■] Antigravity ML STOPPED — STANDBY (predictions cleared).",
            )
        )
        return {"status": "stopped", "ml_status": "standby"}

    def is_mitigated(self) -> bool:
        if self.isolated_hosts or self.blocked_ips or self.blocked_ports:
            return True
        if self.credentials_revoked:
            return True
        return False

    def apply_mitigation(self, action: str, target: Optional[str] = None) -> str:
        from control_backend.site_config import get_site_config

        act = action.upper().strip()
        note = " (dashboard-recorded; wire to firewall/policy for enforcement)"
        site = get_site_config()
        default_host = site.asset_ips()[0] if site.asset_ips() else None
        if act in ("ISOLATE_HOST", "ISOLATE"):
            host = target or default_host
            if not host:
                return "[?] ISOLATE_HOST requires a target IP (no site assets_of_interest configured)."
            self.isolated_hosts.add(host)
            msg = f"[🛡️ SOAR]: Marked host {host} isolated.{note}"
        elif act in ("BLOCK_ATTACKER_IP", "BLOCK_IP"):
            ip = target
            if not ip:
                return "[?] BLOCK_IP requires a target IP (no hardcoded external/attacker address)."
            self.blocked_ips.add(ip)
            msg = f"[🚫 SOAR]: Marked block for IP {ip}.{note}"
        elif act in ("BLOCK_PORT", "BLOCK"):
            p = int(target) if target and str(target).isdigit() else 80
            self.blocked_ports.add(p)
            msg = f"[🔒 SOAR]: Marked block for port {p}.{note}"
        elif act in ("REVOKE_CREDENTIALS", "REVOKE_CREDS"):
            self.credentials_revoked = True
            msg = f"[🔑 SOAR]: Marked credentials revoked.{note}"
        elif act in ("CLEAR", "CLEAR_DEFENSES"):
            self.isolated_hosts.clear()
            self.blocked_ips.clear()
            self.blocked_ports.clear()
            self.credentials_revoked = False
            self.adapter.reset_history()
            msg = "[🔄 SOAR]: Cleared recorded defenses."
        else:
            msg = f"[?] Unknown mitigation action: {act}"

        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="mitigate_threat",
                timestamp=utc_now_iso(),
                line=msg,
            )
        )
        return msg

    def _mitigation_bypass_flows(self, flows) -> int:
        """Flows a recorded block/isolation should have stopped but did not.

        Mitigations here are recorded intents, not enforcement. This used to be
        _filter_flows(), which DROPPED those flows before scoring -- so if the
        real block had not been applied, the attack traffic was hidden from the
        model and the dashboard reported the host as quiet. The flows are now
        always scored; this count is how the operator learns the block failed.
        """
        if not self.is_mitigated():
            return 0
        n = 0
        for r in flows:
            if (
                r.src_ip in self.blocked_ips or r.dst_ip in self.blocked_ips
                or r.dst_port in self.blocked_ports
                or r.src_ip in self.isolated_hosts or r.dst_ip in self.isolated_hosts
            ):
                n += 1
        return n

    def _stdout_logger(self):
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            cleaned = line.rstrip()
            if cleaned:
                broker.broadcast_sync(
                    CommandEvent(
                        type="command_output",
                        command="telemetry",
                        timestamp=utc_now_iso(),
                        line=cleaned,
                    )
                )

    def _tail_worker(self):
        waited = 0.0
        while self.is_running and not os.path.exists(STATE_STREAM_PATH):
            time.sleep(0.2)
            waited += 0.2
            if waited > 15.0:
                logger.warning("State stream file was not created within 15 seconds.")
                break

        if not os.path.exists(STATE_STREAM_PATH):
            return

        with open(STATE_STREAM_PATH, "r", encoding="utf-8") as f:
            while self.is_running:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                try:
                    record = json.loads(line)
                    self.capture.observe(record)
                    raw_flows = record.get("flows") or []
                    flows = flows_from_span_dicts(raw_flows)
                    target = select_primary_target(flows)
                    self.last_window_at = time.time()
                    window_end = float(
                        record.get("window_end")
                        or record.get("timestamp")
                        or self.last_window_at
                    )

                    topo = topology_service.apply_window(
                        raw_flows,
                        window_end=window_end,
                        now=self.last_window_at,
                    )

                    # Live reality metrics -- MEASURED, not synthesised.
                    #
                    # These three are labelled "Live operational reality
                    # metrics" in schema.SystemStatusEvent and are read by an
                    # operator as observations of their network. All three were
                    # partly invented:
                    #
                    #   throughput  = max(real, len(flows)*0.35 + 20.0)
                    #                 -> never below 20 Mbps, whatever the wire
                    #                    was actually carrying; on a quiet
                    #                    network the displayed figure was pure
                    #                    flow-count arithmetic.
                    #   packetLoss  = a measured 0.0 was OVERWRITTEN with
                    #                 (int(time.time()) % 4) * 0.1 -- a 0.0-0.3%
                    #                 loss figure derived from the wall clock.
                    #                 A healthy link could not report healthy.
                    #   latency     = real + min(120, len(flows)*0.25 + 8.0)
                    #                 + a clock-derived "jitter" term, floored
                    #                 at 4 ms. The pipeline's own measurement
                    #                 was a minority of the number shown.
                    #
                    # The window divisor was also the literal 2.0 rather than
                    # the served contract, so throughput would silently be
                    # wrong by the ratio of the two if the window ever changed.
                    window_s = float(getattr(self.adapter, "window_seconds", 2.0)) or 2.0
                    total_bytes = sum((getattr(f, "fwd_bytes", 0) + getattr(f, "bwd_bytes", 0)) for f in flows)
                    self.current_throughput = round((total_bytes * 8.0) / (window_s * 1_000_000.0), 2)
                    self.current_active_connections = len(flows)

                    # Share of flows with no reverse packets. Not true packet
                    # loss, but it is what the sensor can actually observe.
                    unanswered = sum(1 for f in flows if getattr(f, "bwd_packets", 0) == 0)
                    loss_pct = (unanswered / max(1, len(flows))) * 100.0
                    self.current_packet_loss = round(min(100.0, loss_pct), 1)

                    self.current_latency = round(float(record.get("pipeline_latency_ms", 0.0)), 1)

                    if self.is_ml_active:
                        self.windows_streamed += 1
                        event = self.adapter.predict_window(
                            target_ip=target,
                            flows=flows,
                            window_id=int(record.get("window_id", self.windows_streamed)),
                            # Display-only: never an input to scoring.
                            attack_active=self.external_attack_armed,
                            attack_phase="EXTERNAL" if self.external_attack_armed else None,
                            mitigation_recorded=self.is_mitigated(),
                            mitigation_bypass_flows=self._mitigation_bypass_flows(flows),
                            packet_count=int(record.get("packet_count", 0)),
                            pipeline_latency_ms=float(record.get("pipeline_latency_ms", 0.0)),
                            active_flows=int(record.get("active_flows", len(flows)) or 0),
                            throughput=self.current_throughput,
                        )
                        if target and event.prediction:
                            topology_service.attach_risks(
                                {
                                    target: float(
                                        max(
                                            event.prediction.risk,
                                            event.prediction.max_future_risk,
                                        )
                                    )
                                }
                            )
                            topo = topology_service.snapshot(now=self.last_window_at)
                        broker.broadcast_sync(topo)
                        broker.broadcast_sync(event)
                    else:
                        event = None
                        broker.broadcast_sync(topo)
                        broker.broadcast_sync(
                            {
                                "type": "telemetry_update",
                                "ml_status": "standby",
                                "state": {
                                    "window_id": int(record.get("window_id", 0)),
                                    "packet_count": int(record.get("packet_count", 0)),
                                    "pipeline_latency_ms": float(
                                        record.get("pipeline_latency_ms", 0.0)
                                    ),
                                    "active_flows": record.get("active_flows"),
                                    "flow_export_count": len(raw_flows),
                                    "timestamp": float(record.get("timestamp", time.time())),
                                },
                            }
                        )

                    if event and getattr(event, "prediction", None):
                        pred_obj = event.prediction
                        r_val = max(getattr(pred_obj, "risk", 0.0), getattr(pred_obj, "max_future_risk", 0.0))
                        self.current_anomaly_score = round(r_val * 100.0, 1)
                        raw_alert = getattr(pred_obj, "alert_level", "NOMINAL").upper()
                        if raw_alert == "CRITICAL":
                            self.current_threat_level = "critical"
                        elif raw_alert == "ELEVATED":
                            self.current_threat_level = "high"
                        elif raw_alert == "WARNING":
                            self.current_threat_level = "medium"
                        else:
                            self.current_threat_level = "low"
                    else:
                        self.current_anomaly_score = 0.0
                        self.current_threat_level = "low"

                    now_iso = utc_now_iso()
                    broker.broadcast_sync({
                        "type": "system_status",
                        "mode": "LIVE" if self.is_running else "STANDBY",
                        "network": "running",
                        "sensor": "running" if self.is_running else "stopped",
                        "normal_traffic": "running" if len(flows) > 0 else "stopped",
                        # The alerting cut is the adapter's fitted operating
                        # point, not a literal 65. `alert_threshold` is taken
                        # from the checkpoint (_adopt_risk_semantics), and the
                        # bce objective is expected to fit one well below 0.65
                        # -- at a fitted 0.40, a risk of 0.55 raised
                        # prediction.alert=True / alert_level=ELEVATED /
                        # threatLevel="high" on the same bus while this field
                        # still said "stopped".
                        "attack": (
                            "running"
                            if self.current_anomaly_score
                            >= float(getattr(self.adapter, "alert_threshold", 0.65)) * 100.0
                            else "stopped"
                        ),
                        "ml": "running" if self.is_ml_active else "stopped",
                        "network_online": True,
                        "sensor_active": self.is_running,
                        "telemetry_active": self.is_running,
                        "ml_active": self.is_ml_active,
                        "uptime": int(time.time() - self.service_start_time),
                        "throughput": self.current_throughput,
                        "latency": self.current_latency,
                        "packetLoss": self.current_packet_loss,
                        "activeConnections": self.current_active_connections,
                        "anomalyScore": self.current_anomaly_score,
                        "threatLevel": self.current_threat_level,
                        **self.capture.status(),
                        "timestamp": now_iso,
                    })
                except Exception as e:
                    logger.exception("Error parsing/inferring live stream line: %s", e)


telemetry_service = LiveTelemetryService()
