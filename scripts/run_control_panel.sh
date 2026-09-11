#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

PYTHON_BIN="/var/home/samito/.pyenv/versions/3.12.14/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

DEV_MODE=0
for arg in "$@"; do
    if [ "$arg" == "--dev" ]; then
        DEV_MODE=1
    fi
done

echo "================================================================="
echo "STARTING CYBERWORLD SOC CONTROL PANEL (Dual-Branch + DeepOP)"
echo "================================================================="

# Ensure bita wins over ambient model.py shadows
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/bita${PYTHONPATH:+:$PYTHONPATH}"

# 1. Build frontend if dist/ does not exist
if [ ! -d "web_dashboard/dist" ]; then
    echo "[*] Building React dashboard production bundle..."
    (cd web_dashboard && npm run build)
fi

# 2. Start FastAPI Control Backend
echo "[*] Starting FastAPI Control Backend on port 8000..."
"$PYTHON_BIN" -m uvicorn control_backend.main:app --host 0.0.0.0 --port 8000 --log-level info &
BACKEND_PID=$!

trap "echo '[*] Shutting down Control Panel...'; kill $BACKEND_PID 2>/dev/null || true; exit 0" INT TERM EXIT

sleep 1

if [ "$DEV_MODE" -eq 1 ]; then
    echo "[*] Starting Vite Development Server on port 5174..."
    (cd web_dashboard && npm run dev -- --port 5174) &
    DEV_PID=$!
    trap "echo '[*] Shutting down...'; kill $BACKEND_PID $DEV_PID 2>/dev/null || true; exit 0" INT TERM EXIT
    echo "================================================================="
    echo "CONTROL PANEL ACTIVE (DEV MODE):"
    echo "  Vite Hot-Reload GUI:  http://localhost:5174"
    echo "  Control Backend API:  http://localhost:8000/docs"
    echo "  WebSocket Stream:     ws://localhost:8000/ws"
    echo "================================================================="
    wait $DEV_PID
else
    echo "================================================================="
    echo "CONTROL PANEL ACTIVE:"
    echo "  Operator SOC Dashboard: http://localhost:8000"
    echo "  REST API Documentation: http://localhost:8000/docs"
    echo "  WebSocket Event Stream: ws://localhost:8000/ws"
    echo "================================================================="
    wait $BACKEND_PID
fi
