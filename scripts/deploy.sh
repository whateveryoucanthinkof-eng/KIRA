#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "DEPLOYING CYBERWORLD ENTERPRISE RANGE & STARTING SERVICES"
echo "================================================================="

cd "${ROOT_DIR}"

# 1. Build image if not present
if ! podman image exists localhost/cyberworld-node:latest; then
    echo "[*] Building required node image..."
    ./scripts/build.sh
fi

# Pre-cleanup any lingering enterprise containers
LEFTOVERS=$(podman ps -a --filter "name=clab-enterprise" -q || true)
if [ -n "$LEFTOVERS" ]; then
    echo "[*] Removing previous enterprise containers..."
    podman rm -f $LEFTOVERS >/dev/null 2>&1 || true
fi

# 2. Deploy Containerlab topology under rootless netns
echo "[*] Deploying Containerlab enterprise topology..."
podman unshare --rootless-netns bash -c "
    # Clean stale netns files if unreferenced
    python3 -c '
import ctypes, glob, os
libc = ctypes.CDLL(\"libc.so.6\", use_errno=True)
for p in glob.glob(\"/run/user/1000/netns/netns-*\"):
    try:
        libc.umount2(p.encode(), 2)
        os.remove(p)
    except Exception:
        pass
' 2>/dev/null || true

    mkdir -p /tmp/proc_mock
    cat /proc/modules > /tmp/proc_mock/modules
    echo 'ip_tables 28672 0 - Live 0x0000000000000000' >> /tmp/proc_mock/modules
    echo 'ip6_tables 32768 0 - Live 0x0000000000000000' >> /tmp/proc_mock/modules
    mount --bind /tmp/proc_mock/modules /proc/modules

    mkdir -p /run/podman
    podman system service -t 0 unix:///run/podman/podman.sock &
    SRV_PID=\$!
    sleep 2

    rm -rf containerlab/clab-enterprise || true
    containerlab deploy -t containerlab/enterprise.clab.yml --reconfigure --max-workers 1 -r podman
    kill \$SRV_PID 2>/dev/null || true
"

# 3. Configure networking, routes, firewall rules & SPAN mirroring
echo "[*] Configuring network interfaces, routing, and SPAN tap..."
./containerlab/configs/setup_networking.sh

# 4. Launch Enterprise Services
echo "[*] Starting Enterprise Background Services..."
podman exec -d clab-enterprise-srv-dns python3 /app/nodes/services/dns_server.py
podman exec -d clab-enterprise-srv-id python3 /app/nodes/services/identity_server.py
podman exec -d clab-enterprise-srv-file python3 /app/nodes/services/file_server.py
podman exec -d clab-enterprise-srv-app python3 /app/nodes/services/app_server.py
podman exec -d clab-enterprise-srv-db python3 /app/nodes/services/db_server.py
podman exec -d clab-enterprise-dmz-web python3 /app/nodes/services/web_dmz.py

sleep 2

# 5. Launch User Workload Profiles
echo "[*] Starting Heterogeneous Employee Workloads..."
podman exec -d clab-enterprise-ws-office python3 /app/workloads/user_workload.py office ws-office
podman exec -d clab-enterprise-ws-web python3 /app/workloads/user_workload.py web_heavy ws-web
podman exec -d clab-enterprise-ws-file python3 /app/workloads/user_workload.py file_heavy ws-file
podman exec -d clab-enterprise-ws-app python3 /app/workloads/user_workload.py app_heavy ws-app

echo "================================================================="
echo "CYBERWORLD ENTERPRISE LAB DEPLOYED & OPERATIONAL!"
echo "  Run './scripts/healthcheck.sh' to verify connectivity & policy."
echo "  Run './scripts/run_telemetry.sh' to stream live telemetry states."
echo "================================================================="
