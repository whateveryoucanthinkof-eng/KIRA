"""
control_backend/commands.py
Allowlisted executor driving the real Containerlab cyber-range in the monorepo root.
Attacks are external: ARM marks that an outside campaign is expected; no simulated flows.
"""

from datetime import datetime
import logging
import subprocess
import threading
import time
from typing import Dict, List, Optional, Any

from control_backend.event_broker import broker
from control_backend.lab_config import (
    MONOREPO_ROOT,
    WORKLOAD_PROFILES,
    WORKSTATION_CONTAINERS,
)
from control_backend.site_config import get_site_config
from control_backend.schema import CommandEvent, AttackEvent
from control_backend.telemetry_service import telemetry_service
from control_backend.topology_service import topology_service

logger = logging.getLogger("antigravity.commands")

ALLOWED_COMMANDS = {
    "build_environment": {
        "description": "Build CyberWorld node container image",
        "cmd": ["./scripts/build.sh"],
    },
    "start_network": {
        "description": "Deploy Containerlab enterprise network topology & services",
        "cmd": ["./scripts/deploy.sh"],
    },
    "stop_network": {
        "description": "Teardown Containerlab enterprise network topology & containers",
        "cmd": ["./scripts/destroy.sh"],
    },
    "healthcheck": {
        "description": "Run automated network connectivity and policy audit",
        "cmd": ["./scripts/healthcheck.sh"],
    },
    "verify_telemetry": {
        "description": "Verify passive SPAN sensor packet capture and state builder",
        "cmd": ["./scripts/verify_telemetry.sh"],
    },
    "start_telemetry": {
        "description": "Start live passive packet sniffer on eth_sensor",
        "custom_handler": "start_telemetry_sensor",
    },
    "stop_telemetry": {
        "description": "Stop live passive sniffer",
        "custom_handler": "stop_telemetry_sensor",
    },
    "start_normal_traffic": {
        "description": "Launch realistic background employee traffic profiles",
        "custom_handler": "start_workloads",
    },
    "stop_normal_traffic": {
        "description": "Stop background employee traffic workloads",
        "custom_handler": "stop_workloads",
    },
    "start_attack": {
        "description": "Arm dashboard for external attack (run attack outside containers)",
        "custom_handler": "arm_external_attack",
    },
    "stop_attack": {
        "description": "Disarm external-attack monitoring flag",
        "custom_handler": "disarm_external_attack",
    },
    "start_ml": {
        "description": "Activate Antigravity Dual-Branch + DeepOP CWA on live SPAN flows",
        "custom_handler": "start_ml_inference",
    },
    "stop_ml": {
        "description": "Deactivate V3 ML inference and return model to standby",
        "custom_handler": "stop_ml_inference",
    },
    "reset_environment": {
        "description": "Reset telemetry state and return all components to standby",
        "custom_handler": "reset_environment_state",
    },
    "mitigate_threat": {
        "description": "Record defensive SOAR playbook action",
        "custom_handler": "execute_mitigation",
    },
}


LAB_ONLY_COMMANDS = frozenset(
    {
        "build_environment",
        "start_network",
        "stop_network",
        "healthcheck",
        "verify_telemetry",
        "start_normal_traffic",
        "stop_normal_traffic",
    }
)


class CommandExecutor:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_command: Optional[str] = None
        self.active_process: Optional[subprocess.Popen] = None
        self.running_thread: Optional[threading.Thread] = None
        self.workloads_running: bool = False
        self.attack_running: bool = False  # external campaign armed
        self.network_online: bool = False  # overridden by status endpoint

    def is_running(self) -> bool:
        with self.lock:
            return self.active_command is not None

    def is_workloads_running(self) -> bool:
        with self.lock:
            return self.workloads_running

    def is_attack_running(self) -> bool:
        with self.lock:
            return self.attack_running

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "is_running": self.active_command is not None,
                "active_command": self.active_command,
                "network_online": self.network_online,
                "workloads_running": self.workloads_running,
                "attack_running": self.attack_running,
            }

    def execute_command(self, cmd_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if cmd_name not in ALLOWED_COMMANDS:
            raise ValueError(f"Command '{cmd_name}' is not in the allowlist.")

        site = get_site_config()
        if cmd_name in LAB_ONLY_COMMANDS and not site.lab_mode:
            raise RuntimeError(
                f"Command '{cmd_name}' requires Lab Mode "
                f"(site '{site.site_id}' has lab_mode=false). "
                "Set CYBERWORLD_SITE=containerlab-enterprise for Containerlab orchestration."
            )

        with self.lock:
            if self.active_command is not None:
                raise RuntimeError(
                    f"Cannot run '{cmd_name}': command '{self.active_command}' is already in progress."
                )
            self.active_command = cmd_name

        # Long-running shell scripts run in background; quick handlers run sync then clear.
        cfg = ALLOWED_COMMANDS[cmd_name]
        if "cmd" in cfg:
            self.running_thread = threading.Thread(
                target=self._run_shell_worker,
                args=(cmd_name,),
                daemon=True,
            )
            self.running_thread.start()
            return {"status": "started", "command": cmd_name}

        try:
            handler_name = cfg["custom_handler"]
            handler = getattr(self, handler_name, None)
            if not handler:
                raise NotImplementedError(f"Handler '{handler_name}' not implemented.")

            broker.broadcast_sync(
                CommandEvent(
                    type="command_started",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line=f"[*] Executing {cmd_name}...",
                )
            )
            res = handler(args or {})
            broker.broadcast_sync(
                CommandEvent(
                    type="command_completed",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    exit_code=0,
                    success=True,
                    line=f"[✓] {cmd_name} completed successfully.",
                )
            )
            return {"command": cmd_name, "status": "completed", "result": res}
        except Exception as e:
            broker.broadcast_sync(
                CommandEvent(
                    type="command_completed",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    exit_code=1,
                    success=False,
                    line=f"[✗] Error: {e}",
                )
            )
            raise
        finally:
            with self.lock:
                self.active_command = None

    def _run_shell_worker(self, cmd_name: str):
        cfg = ALLOWED_COMMANDS[cmd_name]
        broker.broadcast_sync(
            CommandEvent(
                type="command_started",
                command=cmd_name,
                timestamp=datetime.utcnow().isoformat() + "Z",
                line=f"Executing operation: {cfg['description']}...",
            )
        )
        exit_code = 1
        success = False
        try:
            exit_code = self._execute_process(cfg["cmd"], cmd_name)
            success = exit_code == 0
            if cmd_name == "start_network" and success:
                self.network_online = True
            elif cmd_name == "stop_network":
                self.network_online = False
                self.attack_running = False
                self.workloads_running = False
                telemetry_service.external_attack_armed = False
                try:
                    telemetry_service.stop_sensor()
                except Exception:
                    pass
        except Exception as e:
            logger.error("Error executing %s: %s", cmd_name, e)
            broker.broadcast_sync(
                CommandEvent(
                    type="command_output",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line=f"[ERROR] {e}",
                )
            )
        finally:
            with self.lock:
                self.active_command = None
                self.active_process = None
            broker.broadcast_sync(
                CommandEvent(
                    type="command_completed",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    exit_code=exit_code,
                    success=success,
                    line=f"Operation '{cmd_name}' finished with exit code {exit_code}.",
                )
            )

    def _execute_process(self, cmd: List[str], cmd_name: str) -> int:
        logger.info("Starting process: %s in %s", " ".join(cmd), MONOREPO_ROOT)
        proc = subprocess.Popen(
            cmd,
            cwd=MONOREPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.active_process = proc
        for raw_line in proc.stdout or []:
            line = raw_line.rstrip()
            if not line:
                continue
            broker.broadcast_sync(
                CommandEvent(
                    type="command_output",
                    command=cmd_name,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line=line,
                )
            )
        proc.wait()
        return proc.returncode

    def start_telemetry_sensor(self, args: Dict[str, Any]) -> int:
        telemetry_service.start_sensor()
        return 0

    def stop_telemetry_sensor(self, args: Dict[str, Any]) -> int:
        telemetry_service.stop_sensor()
        return 0

    def start_workloads(self, args: Dict[str, Any]) -> int:
        for container, profile_args in WORKLOAD_PROFILES:
            cmd = [
                "podman",
                "exec",
                "-d",
                container,
                "python3",
                "/app/workloads/user_workload.py",
                *profile_args,
            ]
            subprocess.run(cmd, check=False)
            broker.broadcast_sync(
                CommandEvent(
                    type="command_output",
                    command="start_normal_traffic",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line=f"[+] Started workload on {container}: {' '.join(profile_args)}",
                )
            )
        self.workloads_running = True
        return 0

    def stop_workloads(self, args: Dict[str, Any]) -> int:
        for container in WORKSTATION_CONTAINERS:
            subprocess.run(
                ["podman", "exec", container, "pkill", "-f", "user_workload.py"],
                check=False,
            )
            broker.broadcast_sync(
                CommandEvent(
                    type="command_output",
                    command="stop_normal_traffic",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line=f"[-] Terminated workloads in {container}",
                )
            )
        self.workloads_running = False
        return 0

    def arm_external_attack(self, args: Dict[str, Any]) -> int:
        """Operator arms UI expecting outside→enterprise traffic (no topology fixture)."""
        self.attack_running = True
        telemetry_service.external_attack_armed = True
        site = get_site_config()
        hint = site.external_traffic_hint()
        broker.broadcast_sync(
            AttackEvent(
                stage="[EXTERNAL TRAFFIC ARMED]",
                timestamp=time.time(),
                details=hint,
            )
        )
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="start_attack",
                timestamp=datetime.utcnow().isoformat() + "Z",
                line=(
                    "[⚔️ EXTERNAL MODE] Dashboard armed for outside traffic. "
                    f"{hint} No simulated campaign is launched."
                ),
            )
        )
        if not telemetry_service.is_running:
            broker.broadcast_sync(
                CommandEvent(
                    type="command_output",
                    command="start_attack",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    line="[!] Tip: start sensor + ML so SPAN traffic is observed.",
                )
            )
        return 0

    def disarm_external_attack(self, args: Dict[str, Any]) -> int:
        self.attack_running = False
        telemetry_service.external_attack_armed = False
        broker.broadcast_sync(
            AttackEvent(
                stage="STOPPED",
                timestamp=time.time(),
                details="External attack monitoring disarmed.",
            )
        )
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="stop_attack",
                timestamp=datetime.utcnow().isoformat() + "Z",
                line="[-] External attack monitoring disarmed.",
            )
        )
        return 0

    def start_ml_inference(self, args: Dict[str, Any]) -> int:
        telemetry_service.start_ml()
        return 0

    def stop_ml_inference(self, args: Dict[str, Any]) -> int:
        telemetry_service.stop_ml()
        return 0

    def execute_mitigation(self, args: Dict[str, Any]) -> str:
        return telemetry_service.apply_mitigation(
            args.get("action", "ISOLATE_HOST"),
            args.get("target"),
        )

    def reset_environment_state(self, args: Dict[str, Any]) -> int:
        self.disarm_external_attack({})
        if get_site_config().lab_mode:
            self.stop_workloads({})
        try:
            self.stop_ml_inference({})
        except Exception:
            pass
        telemetry_service.isolated_hosts.clear()
        telemetry_service.blocked_ips.clear()
        telemetry_service.blocked_ports.clear()
        telemetry_service.credentials_revoked = False
        topology_service.reset()
        broker.broadcast_sync(topology_service.snapshot())
        broker.broadcast_sync(
            CommandEvent(
                type="command_output",
                command="reset_environment",
                timestamp=datetime.utcnow().isoformat() + "Z",
                line="[✓] Environment reset: external armed cleared, topology cleared, ML standby.",
            )
        )
        return 0


executor = CommandExecutor()
