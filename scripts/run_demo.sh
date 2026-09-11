#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================="
echo "CYBERWORLD LAB DEMO"
echo "  Use: python run_dashboard.py"
echo "  Then: START NETWORK → SENSOR → ML → WORKLOADS → ARM EXTERNAL"
echo "  Attack from outside toward 10.0.3.10 (DMZ) — SPAN observes it."
echo "================================================================="
exec python3 run_dashboard.py "$@"
