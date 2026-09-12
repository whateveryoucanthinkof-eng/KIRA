#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "BUILDING CyberFortress CONTAINER IMAGE (localhost/CyberFortress-node:latest)"
echo "================================================================="

cd "${ROOT_DIR}"
podman build -t localhost/CyberFortress-node:latest -f nodes/base/Dockerfile .

echo "[*] Image build successful: localhost/CyberFortress-node:latest"
