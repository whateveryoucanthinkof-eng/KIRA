#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "TEARING DOWN CYBERWORLD ENTERPRISE RANGE"
echo "================================================================="

cd "${ROOT_DIR}"

# Fast kill running lab containers to avoid 10s SIGTERM per-container timeout
RUNNING=$(podman ps -q --filter "name=clab-enterprise" || true)
if [ -n "$RUNNING" ]; then
    podman kill $RUNNING >/dev/null 2>&1 || true
fi

if [ -f "containerlab/enterprise.clab.yml" ]; then
    echo "[*] Destroying Containerlab topology..."
    podman unshare --rootless-netns bash -c "
        mkdir -p /tmp/proc_mock
        cat /proc/modules > /tmp/proc_mock/modules
        echo 'ip_tables 28672 0 - Live 0x0000000000000000' >> /tmp/proc_mock/modules
        echo 'ip6_tables 32768 0 - Live 0x0000000000000000' >> /tmp/proc_mock/modules
        mount --bind /tmp/proc_mock/modules /proc/modules 2>/dev/null || true

        mkdir -p /run/podman
        podman system service -t 0 unix:///run/podman/podman.sock &
        SRV_PID=\$!
        sleep 2

        containerlab destroy -t containerlab/enterprise.clab.yml --max-workers 1 -c -r podman || true
        kill \$SRV_PID 2>/dev/null || true
    " || true
fi

# Clean any leftover containers matching clab-enterprise
LEFTOVERS=$(podman ps -a --filter "name=clab-enterprise" --format "{{.ID}}" || true)
if [ -n "$LEFTOVERS" ]; then
    echo "[*] Cleaning leftover containers: $LEFTOVERS"
    podman rm -f $LEFTOVERS || true
fi

echo "================================================================="
echo "CYBERWORLD TEARDOWN COMPLETE."
echo "================================================================="
