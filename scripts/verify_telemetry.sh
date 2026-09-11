#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================="
echo "CYBERWORLD TELEMETRY VERIFICATION"
echo "================================================================="

# 1. Verify clab-enterprise-sensor container is up
if ! podman ps --filter "name=clab-enterprise-sensor" --format "{{.Names}}" | grep -q "clab-enterprise-sensor"; then
    echo "[-] Error: 'clab-enterprise-sensor' is not running."
    echo "    Please run './scripts/deploy.sh' first."
    exit 1
fi

SENSOR_PID=$(podman inspect clab-enterprise-sensor --format '{{.State.Pid}}')
if [ -z "$SENSOR_PID" ] || [ "$SENSOR_PID" -eq 0 ]; then
    echo "[-] Error: Could not determine PID for sensor container."
    exit 1
fi

echo "[*] Sensor container PID: $SENSOR_PID"
echo "[*] Running 4-window live telemetry test on eth_sensor..."

OUTPUT=$(podman unshare nsenter -t "$SENSOR_PID" -n python3 telemetry/run_telemetry.py --interface eth_sensor --max-windows 4 2>&1)
echo "$OUTPUT"

if echo "$OUTPUT" | grep -q "Window #0004"; then
    echo "================================================================="
    echo "[✓] TELEMETRY PIPELINE VERIFICATION PASSED (4/4 windows processed)"
    echo "================================================================="
    exit 0
else
    echo "[-] Telemetry verification failed to complete 4 windows."
    exit 1
fi
