"""
Paths into the cyber-range (Containerlab scripts + live telemetry).
Site identity (CIDRs, lab_mode, assets) lives in config/sites/*.yaml via site_config.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Alias kept for callers that still name the lab root MONOREPO_ROOT
MONOREPO_ROOT = REPO_ROOT
V3_ROOT = REPO_ROOT  # legacy alias after V3→root promotion

STATE_STREAM_PATH = "/tmp/cyberworld_live_stream.jsonl"
ML_TRIGGER_FILE = "/tmp/cyberworld_ml_enabled"

# Containerlab orchestration constants (only meaningful when site.lab_mode=true)
TOTAL_NODES = 15
SENSOR_CONTAINER = "clab-enterprise-sensor"
LAB_NAME_FILTER = "clab-enterprise"

WORKLOAD_PROFILES = [
    ("clab-enterprise-ws-office", ["office", "ws-office"]),
    ("clab-enterprise-ws-web", ["web_heavy", "ws-web"]),
    ("clab-enterprise-ws-file", ["file_heavy", "ws-file"]),
    ("clab-enterprise-ws-app", ["app_heavy", "ws-app"]),
]

WORKSTATION_CONTAINERS = [c for c, _ in WORKLOAD_PROFILES]
