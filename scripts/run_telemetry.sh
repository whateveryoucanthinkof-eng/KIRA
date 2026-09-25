#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

PYTHON_BIN="/var/home/samito/.pyenv/versions/3.12.14/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Check if PCAP replay mode was requested
IS_REPLAY=0
for arg in "$@"; do
    if [ "$arg" == "--pcap" ]; then
        IS_REPLAY=1
        break
    fi
done

if [ "$IS_REPLAY" -eq 1 ]; then
    echo "[*] Running cyberworld SPAN telemetry in REPLAY Mode (capture only)..."
    exec "$PYTHON_BIN" telemetry/run_telemetry.py --no-inference "$@"
fi

# Default: Live Mode from Sensor node (SPAN capture; ML runs in control_backend)
echo "================================================================="
echo "STARTING cyberworld LIVE SPAN TELEMETRY (CAPTURE ONLY)"
echo "================================================================="

if ! podman ps --filter "name=clab-enterprise-sensor" --format "{{.Names}}" | grep -q "clab-enterprise-sensor"; then
    echo "[-] Error: 'clab-enterprise-sensor' container is not running."
    echo "    Please run './scripts/deploy.sh' first."
    exit 1
fi

SENSOR_PID=$(podman inspect clab-enterprise-sensor --format '{{.State.Pid}}')

if [ -z "$SENSOR_PID" ] || [ "$SENSOR_PID" -eq 0 ]; then
    echo "[-] Error: Could not determine PID for clab-enterprise-sensor."
    exit 1
fi

echo "[*] Sensor container PID: $SENSOR_PID"
echo "[*] Ingesting live packets from interface 'eth_sensor' via kernel network namespace..."
echo "[*] Capture: 2.0s windows + 5-tuple flow snapshots for Dual-Branch/DeepOP"
echo "[*] Inference: disabled here (Antigravity stack runs in control_backend)"
echo "[*] In-memory hot path active (Zero CSV / Zero PCAP disk overhead)"
echo "-----------------------------------------------------------------"

# Default to capture-only unless caller explicitly wants in-process inference
HAS_NO_INF=0
for arg in "$@"; do
    if [ "$arg" == "--no-inference" ]; then
        HAS_NO_INF=1
        break
    fi
done

EXTRA_ARGS=()
if [ "$HAS_NO_INF" -eq 0 ]; then
    EXTRA_ARGS+=(--no-inference)
fi

exec podman unshare nsenter -t "$SENSOR_PID" -n "$PYTHON_BIN" telemetry/run_telemetry.py --interface eth_sensor "${EXTRA_ARGS[@]}" "$@"
