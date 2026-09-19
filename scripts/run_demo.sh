#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================="
echo "cyberworld LAB DEMO"
echo "  Live: python run_dashboard.py"
echo "  Demo: python run_dashboard.py --replay <sample.pcap>"
echo "================================================================="

if [ "${1:-}" == "--replay" ] && [ -n "${2:-}" ]; then
  PCAP="$2"
  echo "[+] Booting dashboard in demo (replay) mode with: $PCAP"
  exec python3 run_dashboard.py --replay "$PCAP"
else
  echo "[+] Booting dashboard in standard mode."
  echo "[i] URL will be http://localhost:8000"
  exec python3 run_dashboard.py "$@"
fi
