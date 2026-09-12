#!/usr/bin/env python3
"""
workloads/attacker_scenario_aggressive.py

AGGRESSIVE but CONTAINED ContainerLab workload for CyberFortress prediction testing.

Goals:
- Generate substantially higher telemetry variance.
- Create abrupt temporal transitions.
- Stress temporal prediction / early-warning behavior.
- Produce overlapping attack signatures.
- Exercise burst -> quiet -> burst transitions.
- Create concurrent traffic against multiple lab targets.
- Keep the workload non-destructive: no filesystem destruction,
  persistence, credential theft, or real reverse shells.

Stages:
1. RECON_BURST
2. WEB_PROBE_STORM
3. EXPLOIT_SIMULATION
4. MIXED_ATTACK_BURSTS
5. LATERAL_PIVOT_STORM
6. RECOVERY / COOL-DOWN

The "actual milestone" is deliberately recorded immediately before
the lateral movement stage so model lead-time can be evaluated.

Expected evaluation:

    prediction_time < t_actual_milestone

and preferably:

    prediction_time << t_actual_milestone
"""

import json
import random
import socket
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Any


# ============================================================
# CONFIGURATION
# ============================================================

DMZ_HOST = "10.0.3.10"
DMZ_WEB_URL = f"http://{DMZ_HOST}:80"

INTERNAL_APP_HOST = "10.0.2.40"
INTERNAL_DB_HOST = "10.0.2.50"

ATTACKER_IP = "192.168.100.10"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# Broad service fingerprinting set.
PORTS_TO_SCAN = [
    21, 22, 23, 25, 53,
    80, 81, 88, 110, 111, 135,
    139, 143, 389, 443, 445,
    465, 587, 636, 993, 995,
    1433, 1521, 2049, 2375,
    3000, 3306, 3389, 5000,
    5432, 5601, 5900,
    6379, 6443, 8000,
    8080, 8081, 8443, 9000
]


# Non-destructive malicious-looking HTTP paths.
FUZZ_PATHS = [
    "/admin",
    "/admin/login.php",
    "/manager/html",
    "/api/v1/system/debug",
    "/.env",
    "/wp-config.php.bak",
    "/actuator/health",
    "/api/v1/users",
    "/api/v1/customers",
    "/api/v1/system",
    "/api/v1/debug",
    "/api/v1/exec",
    "/login?user=admin%27%20OR%20%271%27%3D%271",
    "/search?q=%27%20UNION%20SELECT%201%2C2%2C3--",
    "/api/v1/query?input=%27%20OR%201%3D1--",
    "/cgi-bin/test.cgi?cmd=id",
    "/shell.php?cmd=whoami",
    "/debug?cmd=uname",
    "/api/v1/exec?cmd=cat%20/etc/shadow",
    "/login?user=admin' OR '1'='1'--",
    "/shell.php?cmd=id;whoami;uname -a",
    "/cgi-bin/test.cgi?cmd=rm -rf /"
]


USER_AGENTS = [
    "Mozilla/5.0",
    "curl/8.0",
    "python-requests/2.x",
    "sqlmap/1.x",
    "nikto",
    "Nmap Scripting Engine",
    "ExploitFramework-Test",
]


# ============================================================
# LOGGING
# ============================================================

def log_event(
    stage: str,
    message: str,
    timestamp: float = None
):
    ts = timestamp if timestamp is not None else time.time()

    timestr = time.strftime(
        "%H:%M:%S",
        time.localtime(ts)
    )

    print(
        f"[{timestr}] "
        f"[ATTACKER] "
        f"[{stage}] "
        f"{message}"
    )

    sys.stdout.flush()


# ============================================================
# LOW-LEVEL NETWORK PRIMITIVES
# ============================================================

def tcp_probe(
    host: str,
    port: int,
    timeout: float = 0.05
) -> bool:

    try:
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        sock.settimeout(timeout)

        result = sock.connect_ex(
            (host, port)
        )

        sock.close()

        return result == 0

    except Exception:
        return False


def http_probe(
    url: str,
    timeout: float = 0.5
) -> bool:

    try:

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "*/*",
                "X-Forwarded-For": ATTACKER_IP,
                "Connection": "keep-alive",
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=timeout
        ) as resp:

            resp.read(256)

        return True

    except Exception:
        return False


# ============================================================
# STAGE 1
# EXTREME RECONNAISSANCE BURST
# ============================================================

def run_reconnaissance(
    target_ip: str = DMZ_HOST,
    duration_sec: float = 10.0
) -> Dict[str, Any]:

    t_start = time.time()

    log_event(
        "STAGE 1: RECON_BURST",
        f"Starting aggressive service discovery against {target_ip}"
    )

    probes_sent = 0
    open_ports = []

    end_time = t_start + duration_sec

    while time.time() < end_time:

        # Randomize ordering to avoid a perfectly periodic signature.
        ports = PORTS_TO_SCAN.copy()
        random.shuffle(ports)

        for port in ports:

            result = tcp_probe(
                target_ip,
                port,
                timeout=random.uniform(0.025, 0.08)
            )

            probes_sent += 1

            if result:
                open_ports.append(port)

            # Highly variable inter-packet timing.
            time.sleep(
                random.uniform(
                    0.002,
                    0.025
                )
            )

    t_end = time.time()

    log_event(
        "STAGE 1: RECON_BURST",
        f"{probes_sent} probes generated; "
        f"{len(set(open_ports))} responsive ports"
    )

    return {
        "stage": "RECON_BURST",
        "start": t_start,
        "end": t_end,
        "probes": probes_sent,
        "responsive_ports": sorted(set(open_ports)),
    }


# ============================================================
# STAGE 2
# WEB PROBE STORM
# ============================================================

def run_web_probe_worker(
    target_url: str,
    end_time: float,
    counter: List[int]
):

    while time.time() < end_time:

        path = random.choice(FUZZ_PATHS)

        # Random query padding produces changing request sizes.
        padding_length = random.choice(
            [0, 16, 64, 128, 256, 512]
        )

        if padding_length:
            path += (
                "&pad=" +
                ("A" * padding_length)
            )

        url = target_url + path

        http_probe(url)

        counter[0] += 1

        time.sleep(
            random.uniform(
                0.005,
                0.08
            )
        )


def run_suspicious_probing(
    target_url: str = DMZ_WEB_URL,
    duration_sec: float = 14.0
) -> Dict[str, Any]:

    t_start = time.time()

    log_event(
        "STAGE 2: WEB_PROBE_STORM",
        f"Launching concurrent HTTP probing against {target_url}"
    )

    request_counter = [0]

    end_time = t_start + duration_sec

    # Multiple concurrent request streams.
    workers = []

    for _ in range(5):

        thread = threading.Thread(
            target=run_web_probe_worker,
            args=(
                target_url,
                end_time,
                request_counter
            ),
            daemon=True
        )

        thread.start()
        workers.append(thread)

    # Deliberate burst modulation.
    while time.time() < end_time:

        burst_pause = random.choice([
            0.05,
            0.10,
            0.20,
            0.40
        ])

        time.sleep(burst_pause)

    for thread in workers:
        thread.join(timeout=1)

    t_end = time.time()

    log_event(
        "STAGE 2: WEB_PROBE_STORM",
        f"Generated {request_counter[0]} HTTP probes"
    )

    return {
        "stage": "WEB_PROBE_STORM",
        "start": t_start,
        "end": t_end,
        "requests": request_counter[0],
    }


# ============================================================
# STAGE 3
# EXPLOIT SIMULATION
# ============================================================

def run_exploit_attempt(
    target_url: str = DMZ_WEB_URL,
    duration_sec: float = 10.0
) -> Dict[str, Any]:

    t_start = time.time()

    log_event(
        "STAGE 3: EXPLOIT_SIMULATION",
        "Beginning high-frequency exploit-pattern simulation"
    )

    payloads_sent = 0

    payload_variants = [
        {
            "exploit": "CVE-TEST-001",
            "action": "command_injection_probe",
        },
        {
            "exploit": "CVE-TEST-002",
            "action": "template_injection_probe",
        },
        {
            "exploit": "CVE-TEST-003",
            "action": "path_traversal_probe",
        },
        {
            "exploit": "CVE-TEST-004",
            "action": "deserialization_probe",
        },
        {
            "exploit": "CVE-TEST-005",
            "action": "authentication_bypass_probe",
        },
    ]

    end_time = t_start + duration_sec

    while time.time() < end_time:

        variant = random.choice(
            payload_variants
        )

        payload = json.dumps({
            **variant,
            "test": True,
            "padding": "X" * random.randint(
                64,
                4096
            ),
        }).encode()

        try:

            req = urllib.request.Request(
                f"{target_url}/api/v1/system/exec",
                data=payload,
                headers={
                    "Content-Type":
                        "application/json",

                    "User-Agent":
                        random.choice(USER_AGENTS),

                    "X-Exploit-Test":
                        "true",

                    "X-Test-Vector":
                        variant["action"],
                }
            )

            with urllib.request.urlopen(
                req,
                timeout=0.8
            ) as resp:

                resp.read(128)

        except Exception:
            pass

        payloads_sent += 1

        # Very irregular timing.
        time.sleep(
            random.uniform(
                0.01,
                0.15
            )
        )

    t_end = time.time()

    log_event(
        "STAGE 3: EXPLOIT_SIMULATION",
        f"Generated {payloads_sent} exploit-pattern requests"
    )

    return {
        "stage": "EXPLOIT_SIMULATION",
        "start": t_start,
        "end": t_end,
        "payloads": payloads_sent,
    }


# ============================================================
# STAGE 4
# MIXED ATTACK BURSTS
# ============================================================

def run_mixed_attack_bursts(
    duration_sec: float = 12.0
) -> Dict[str, Any]:

    t_start = time.time()

    log_event(
        "STAGE 4: MIXED_ATTACK_BURSTS",
        "Starting overlapping reconnaissance + HTTP + pivot telemetry"
    )

    end_time = t_start + duration_sec

    recon_count = 0
    http_count = 0
    pivot_count = 0

    internal_targets = [
        (INTERNAL_APP_HOST, 8000),
        (INTERNAL_DB_HOST, 5432),
    ]

    while time.time() < end_time:

        mode = random.choice([
            "RECON",
            "HTTP",
            "PIVOT",
            "RECON",
            "HTTP",
            "BURST",
        ])

        if mode == "RECON":

            for _ in range(
                random.randint(5, 20)
            ):

                port = random.choice(
                    PORTS_TO_SCAN
                )

                tcp_probe(
                    DMZ_HOST,
                    port,
                    timeout=0.04
                )

                recon_count += 1

        elif mode == "HTTP":

            for _ in range(
                random.randint(3, 12)
            ):

                path = random.choice(
                    FUZZ_PATHS
                )

                http_probe(
                    DMZ_WEB_URL + path,
                    timeout=0.5
                )

                http_count += 1

        elif mode == "PIVOT":

            host, port = random.choice(
                internal_targets
            )

            for _ in range(
                random.randint(2, 8)
            ):

                tcp_probe(
                    host,
                    port,
                    timeout=0.08
                )

                pivot_count += 1

        elif mode == "BURST":

            # Short simultaneous burst.
            threads = []

            for _ in range(8):

                thread = threading.Thread(
                    target=tcp_probe,
                    args=(
                        DMZ_HOST,
                        random.choice(
                            PORTS_TO_SCAN
                        ),
                        0.03,
                    ),
                    daemon=True
                )

                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join(timeout=0.2)

        time.sleep(
            random.uniform(
                0.01,
                0.12
            )
        )

    t_end = time.time()

    log_event(
        "STAGE 4: MIXED_ATTACK_BURSTS",
        f"recon={recon_count}, "
        f"http={http_count}, "
        f"pivot={pivot_count}"
    )

    return {
        "stage": "MIXED_ATTACK_BURSTS",
        "start": t_start,
        "end": t_end,
        "recon": recon_count,
        "http": http_count,
        "pivot": pivot_count,
    }


# ============================================================
# STAGE 5
# TARGET MILESTONE + LATERAL STORM
# ============================================================

def run_lateral_pivot_attempt(
    duration_sec: float = 10.0
) -> Dict[str, Any]:

    t_start = time.time()

    targets = [
        (INTERNAL_APP_HOST, 8000),
        (INTERNAL_APP_HOST, 8080),
        (INTERNAL_DB_HOST, 5432),
        (INTERNAL_DB_HOST, 3306),
    ]

    log_event(
        "STAGE 5: LATERAL_PIVOT_STORM",
        f"Targeting internal services: {targets}"
    )

    attempts = 0

    end_time = t_start + duration_sec

    while time.time() < end_time:

        # Rapid target switching.
        shuffled = targets.copy()
        random.shuffle(shuffled)

        for host, port in shuffled:

            tcp_probe(
                host,
                port,
                timeout=random.uniform(
                    0.02,
                    0.08
                )
            )

            attempts += 1

            # Occasionally create a microburst.
            if random.random() < 0.35:

                for _ in range(
                    random.randint(3, 10)
                ):

                    tcp_probe(
                        host,
                        port,
                        timeout=0.025
                    )

                    attempts += 1

            time.sleep(
                random.uniform(
                    0.005,
                    0.06
                )
            )

    t_end = time.time()

    log_event(
        "STAGE 5: LATERAL_PIVOT_STORM",
        f"Completed {attempts} internal-service attempts"
    )

    return {
        "stage": "LATERAL_PIVOT_STORM",
        "start": t_start,
        "end": t_end,
        "attempts": attempts,
    }


# ============================================================
# STAGE 6
# RECOVERY / COOL-DOWN
# ============================================================

def run_recovery(
    duration_sec: float = 8.0
) -> Dict[str, Any]:

    t_start = time.time()

    log_event(
        "STAGE 6: RECOVERY",
        "Stopping attack traffic; observing telemetry recovery"
    )

    time.sleep(duration_sec)

    t_end = time.time()

    return {
        "stage": "RECOVERY",
        "start": t_start,
        "end": t_end,
    }


# ============================================================
# FULL CAMPAIGN
# ============================================================

def execute_full_attack_campaign(
    delay_before_start: float = 0.0
) -> Dict[str, Any]:

    if delay_before_start > 0:

        log_event(
            "CAMPAIGN",
            f"Baseline period: {delay_before_start:.1f}s"
        )

        time.sleep(
            delay_before_start
        )

    t_campaign_start = time.time()

    log_event(
        "CAMPAIGN",
        "=================================================="
    )

    log_event(
        "CAMPAIGN",
        "AGGRESSIVE CyberFortress TEST CAMPAIGN STARTED"
    )

    log_event(
        "CAMPAIGN",
        f"Random seed = {RANDOM_SEED}"
    )

    log_event(
        "CAMPAIGN",
        "=================================================="
    )


    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    s1 = run_reconnaissance(
        duration_sec=10.0
    )

    # Very short transition.
    time.sleep(0.3)


    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    s2 = run_suspicious_probing(
        duration_sec=14.0
    )

    time.sleep(0.3)


    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    s3 = run_exploit_attempt(
        duration_sec=10.0
    )

    time.sleep(0.3)


    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    s4 = run_mixed_attack_bursts(
        duration_sec=12.0
    )


    # --------------------------------------------------------
    # TARGET MILESTONE
    # --------------------------------------------------------

    t_actual_milestone = time.time()

    log_event(
        "CAMPAIGN",
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    log_event(
        "CAMPAIGN",
        "TARGET MILESTONE REACHED"
    )

    log_event(
        "CAMPAIGN",
        f"T_actual = {t_actual_milestone:.6f}"
    )

    log_event(
        "CAMPAIGN",
        "Model should ideally have predicted this BEFORE now."
    )

    log_event(
        "CAMPAIGN",
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )


    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------

    s5 = run_lateral_pivot_attempt(
        duration_sec=10.0
    )


    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    s6 = run_recovery(
        duration_sec=8.0
    )


    t_campaign_end = time.time()


    timeline = {

        "random_seed": RANDOM_SEED,

        "campaign_start":
            t_campaign_start,

        "campaign_end":
            t_campaign_end,

        "t_actual_milestone":
            t_actual_milestone,

        "stages": [
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
        ],
    }


    log_event(
        "CAMPAIGN",
        "=================================================="
    )

    log_event(
        "CAMPAIGN",
        "AGGRESSIVE CAMPAIGN COMPLETE"
    )

    log_event(
        "CAMPAIGN",
        f"Total duration: "
        f"{t_campaign_end - t_campaign_start:.2f}s"
    )

    log_event(
        "CAMPAIGN",
        "=================================================="
    )

    return timeline


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    delay = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else 0.0
    )

    result = execute_full_attack_campaign(
        delay_before_start=delay
    )

    print("\n--- ATTACK TIMELINE JSON ---")

    print(
        json.dumps(
            result,
            indent=2
        )
    )