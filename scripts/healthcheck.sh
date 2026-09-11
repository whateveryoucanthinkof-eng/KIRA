#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "CYBERWORLD ENTERPRISE RANGE HEALTHCHECK & POLICY AUDIT"
echo "================================================================="

FAILED_CHECKS=0
TOTAL_CHECKS=0

pass() {
    echo -e "  \e[32m[PASS]\e[0m $1"
}

fail() {
    echo -e "  \e[31m[FAIL]\e[0m $1"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

run_check() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    local desc="$1"
    shift
    if "$@"; then
        pass "$desc"
    else
        fail "$desc"
    fi
}

echo ""
echo "--- 1. NODE STATUS & SERVICE PROCESS CHECKS ---"

check_container_running() {
    local node="$1"
    podman ps --filter "name=$node" --filter "status=running" --format "{{.Names}}" | grep -q "$node"
}

for node in router-edge router-firewall router-core ws-office ws-web ws-file ws-app srv-dns srv-id srv-file srv-app srv-db dmz-web attacker sensor; do
    run_check "Node 'clab-enterprise-$node' is running" check_container_running "clab-enterprise-$node"
done

echo ""
echo "--- 2. INTERNAL CONNECTIVITY & ENTERPRISE SERVICE REACHABILITY ---"

# Check DNS resolution from workstation
test_dns() {
    local res
    res=$(podman exec clab-enterprise-ws-office nslookup app.corp.local 10.0.2.10 2>&1 || true)
    echo "$res" | grep -q "10.0.2.40"
}
run_check "ws-office resolves 'app.corp.local' via srv-dns (10.0.2.10)" test_dns

# Check HTTP Application endpoint from workstation
test_app_http() {
    local code
    code=$(podman exec clab-enterprise-ws-office curl -s -o /dev/null -w "%{http_code}" http://10.0.2.40:8000/api/v1/system/status || true)
    [ "$code" = "200" ]
}
run_check "ws-office connects to srv-app REST API (http://10.0.2.40:8000/api/v1/system/status -> 200 OK)" test_app_http

# Check File server endpoint from workstation
test_file_http() {
    local code
    code=$(podman exec clab-enterprise-ws-file curl -s -o /dev/null -w "%{http_code}" http://10.0.2.30:8080/files/document.pdf || true)
    [ "$code" = "200" ]
}
run_check "ws-file downloads file from srv-file (http://10.0.2.30:8080/files/document.pdf -> 200 OK)" test_file_http

# Check DMZ web reachability from workstation
test_dmz_from_user() {
    local code
    code=$(podman exec clab-enterprise-ws-web curl -s -o /dev/null -w "%{http_code}" http://10.0.3.10:80/ || true)
    [ "$code" = "200" ]
}
run_check "ws-web accesses DMZ portal (http://10.0.3.10:80/ -> 200 OK)" test_dmz_from_user

# Check Backend App -> DB communication
test_app_to_db() {
    # Test query via app server
    local body
    body=$(podman exec clab-enterprise-srv-app curl -s http://127.0.0.1:8000/api/v1/customers || true)
    echo "$body" | grep -q "Acme Corp"
}
run_check "srv-app successfully queries srv-db backend (http://10.0.2.40:8000/api/v1/customers returns records)" test_app_to_db

echo ""
echo "--- 3. SECURITY POLICY & SEGMENTATION ENFORCEMENT ---"

# Attacker CAN access DMZ web
test_attacker_to_dmz() {
    local code
    code=$(podman exec clab-enterprise-attacker curl -s --connect-timeout 2 -o /dev/null -w "%{http_code}" http://10.0.3.10:80/ || true)
    [ "$code" = "200" ]
}
run_check "PERMIT: Attacker can reach public DMZ Web (10.0.3.10:80)" test_attacker_to_dmz

# Attacker CANNOT reach User zone
test_attacker_to_users_blocked() {
    local code
    code=$(podman exec clab-enterprise-attacker curl -s --connect-timeout 2 -o /dev/null -w "%{http_code}" http://10.0.1.10/ 2>&1 || true)
    [ "$code" != "200" ]
}
run_check "BLOCK: Attacker cannot reach User workstation (10.0.1.10)" test_attacker_to_users_blocked

# Attacker CANNOT reach Server zone
test_attacker_to_servers_blocked() {
    local code
    code=$(podman exec clab-enterprise-attacker curl -s --connect-timeout 2 -o /dev/null -w "%{http_code}" http://10.0.2.40:8000/api/v1/system/status 2>&1 || true)
    [ "$code" != "200" ]
}
run_check "BLOCK: Attacker cannot reach Internal Server zone (10.0.2.40:8000)" test_attacker_to_servers_blocked

# Users CANNOT directly access DB server (port 5432)
test_user_to_db_blocked() {
    # DB server strictly rejects IPs other than 10.0.2.40
    local res
    res=$(podman exec clab-enterprise-ws-office python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2.0)
try:
    s.connect(('10.0.2.50', 5432))
    data = s.recv(1024)
    print(data.decode('utf-8', errors='ignore'))
except Exception as e:
    print('ERROR:', e)
finally:
    s.close()
" 2>&1 || true)
    echo "$res" | grep -q "Connection refused" || echo "$res" | grep -q "permission denied" || echo "$res" | grep -q "timed out"
}
run_check "POLICY: Direct workstation access to DB (10.0.2.50:5432) is blocked" test_user_to_db_blocked

echo ""
echo "--- 4. PASSIVE SPAN SENSOR OBSERVABILITY ---"

# Verify sensor receives packets passively
test_sensor_rx() {
    # Trigger a ping while sensor monitors
    local count
    count=$(podman exec clab-enterprise-sensor timeout 3 tcpdump -c 5 -ni eth_sensor 2>&1 | grep -c "packets captured" || true)
    [ "$count" -ge 1 ]
}
run_check "Sensor passively observes network traffic via SPAN tap (eth_sensor)" test_sensor_rx

echo ""
echo "================================================================="
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "\e[32mALL $TOTAL_CHECKS HEALTHCHECKS & SECURITY AUDITS PASSED SUCCESSFULLY!\e[0m"
    echo "================================================================="
    exit 0
else
    echo -e "\e[31m$FAILED_CHECKS OUT OF $TOTAL_CHECKS CHECKS FAILED!\e[0m"
    echo "================================================================="
    exit 1
fi
