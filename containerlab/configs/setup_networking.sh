#!/bin/bash
set -euo pipefail

echo "================================================================="
echo "CONFIGURING ENTERPRISE NETWORK INTERFACES, ROUTES & POLICIES"
echo "================================================================="

PREFIX="clab-enterprise"

pexec() {
    local node="$1"
    shift
    podman exec "${PREFIX}-${node}" "$@"
}

echo "[1/7] Configuring External Attacker..."
pexec attacker ip addr flush dev eth1 || true
pexec attacker ip addr add 192.168.100.10/24 dev eth1
pexec attacker ip link set eth1 up
pexec attacker ip route replace default via 192.168.100.1

echo "[2/7] Configuring Edge Router..."
pexec router-edge sysctl -w net.ipv4.ip_forward=1 >/dev/null
pexec router-edge ip addr flush dev eth1 || true
pexec router-edge ip addr flush dev eth2 || true
pexec router-edge ip addr add 192.168.100.1/24 dev eth1
pexec router-edge ip addr add 172.16.0.1/30 dev eth2
pexec router-edge ip link set eth1 up
pexec router-edge ip link set eth2 up
pexec router-edge ip route replace 10.0.0.0/8 via 172.16.0.2

echo "[3/7] Configuring Perimeter Firewall Router & Security Policies..."
pexec router-firewall sysctl -w net.ipv4.ip_forward=1 >/dev/null
pexec router-firewall ip addr flush dev eth1 || true
pexec router-firewall ip addr flush dev eth2 || true
pexec router-firewall ip addr add 172.16.0.2/30 dev eth1
pexec router-firewall ip addr add 10.0.0.1/30 dev eth2
pexec router-firewall ip link set eth1 up
pexec router-firewall ip link set eth2 up
pexec router-firewall ip route replace 192.168.100.0/24 via 172.16.0.1
pexec router-firewall ip route replace 10.0.1.0/24 via 10.0.0.2
pexec router-firewall ip route replace 10.0.2.0/24 via 10.0.0.2
pexec router-firewall ip route replace 10.0.3.0/24 via 10.0.0.2

# Enforce Enterprise Firewall Rules
pexec router-firewall iptables -F FORWARD || true
# 1. Allow established/related connections
pexec router-firewall iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT || true
# 2. Allow internal enterprise communications (10.0.0.0/8 <-> 10.0.0.0/8)
# But BLOCK workstations (10.0.1.0/24) from directly connecting to Database (10.0.2.50:5432)
pexec router-firewall iptables -A FORWARD -s 10.0.1.0/24 -d 10.0.2.50 -p tcp --dport 5432 -j REJECT --reject-with tcp-reset || true
pexec router-firewall iptables -A FORWARD -s 10.0.0.0/8 -d 10.0.0.0/8 -j ACCEPT || true
# 3. Allow External/Attacker to DMZ Web (10.0.3.10:80)
pexec router-firewall iptables -A FORWARD -s 192.168.100.0/24 -d 10.0.3.10 -p tcp --dport 80 -j ACCEPT || true
# 4. Strictly DROP and block External access to internal Users and Servers
pexec router-firewall iptables -A FORWARD -s 192.168.100.0/24 -d 10.0.1.0/24 -j DROP || true
pexec router-firewall iptables -A FORWARD -s 192.168.100.0/24 -d 10.0.2.0/24 -j DROP || true

echo "[4/7] Configuring Core Router, Bridges & SPAN Mirroring..."
pexec router-core sysctl -w net.ipv4.ip_forward=1 >/dev/null
pexec router-core ip addr flush dev eth_fw || true
pexec router-core ip addr flush dev eth_dmz || true
pexec router-core ip addr add 10.0.0.2/30 dev eth_fw
pexec router-core ip addr add 10.0.3.1/24 dev eth_dmz
pexec router-core ip link set eth_fw up
pexec router-core ip link set eth_dmz up
pexec router-core ip link set eth_sensor up
pexec router-core ip route replace default via 10.0.0.1

# Create Internal Bridges for Users and Servers subnets
pexec router-core ip link add br-users type bridge || true
pexec router-core ip addr flush dev br-users || true
pexec router-core ip addr add 10.0.1.1/24 dev br-users
for iface in eth_u1 eth_u2 eth_u3 eth_u4; do
    pexec router-core ip link set "$iface" master br-users up || true
done
pexec router-core ip link set br-users up

pexec router-core ip link add br-servers type bridge || true
pexec router-core ip addr flush dev br-servers || true
pexec router-core ip addr add 10.0.2.1/24 dev br-servers
for iface in eth_s1 eth_s2 eth_s3 eth_s4 eth_s5; do
    pexec router-core ip link set "$iface" master br-servers up || true
done
pexec router-core ip link set br-servers up

# Core Router Inter-VLAN Security: Block direct workstation access to DB (10.0.2.50:5432)
pexec router-core iptables -F FORWARD || true
pexec router-core iptables -A FORWARD -s 10.0.1.0/24 -d 10.0.2.50 -p tcp --dport 5432 -j REJECT --reject-with tcp-reset || true
pexec router-core iptables -A FORWARD -j ACCEPT || true

# Configure SPAN Port Mirroring (tc mirred) -> eth_sensor
for mirr_dev in br-users br-servers eth_dmz eth_fw; do
    pexec router-core tc qdisc del dev "$mirr_dev" ingress 2>/dev/null || true
    pexec router-core tc qdisc add dev "$mirr_dev" handle ffff: ingress || true
    pexec router-core tc filter add dev "$mirr_dev" parent ffff: matchall action mirred egress mirror dev eth_sensor || true
done

echo "[5/7] Configuring Users Subnet (Workstations)..."
USERS=("ws-office:11" "ws-web:12" "ws-file:13" "ws-app:14")
for entry in "${USERS[@]}"; do
    node="${entry%%:*}"
    ip_suffix="${entry##*:}"
    pexec "$node" ip addr flush dev eth1 || true
    pexec "$node" ip addr add "10.0.1.${ip_suffix}/24" dev eth1
    pexec "$node" ip link set eth1 up
    pexec "$node" ip route replace default via 10.0.1.1
done

echo "[6/7] Configuring Servers & DMZ..."
SERVERS=("srv-dns:10" "srv-id:20" "srv-file:30" "srv-app:40" "srv-db:50")
for entry in "${SERVERS[@]}"; do
    node="${entry%%:*}"
    ip_suffix="${entry##*:}"
    pexec "$node" ip addr flush dev eth1 || true
    pexec "$node" ip addr add "10.0.2.${ip_suffix}/24" dev eth1
    pexec "$node" ip link set eth1 up
    pexec "$node" ip route replace default via 10.0.2.1
done

pexec dmz-web ip addr flush dev eth1 || true
pexec dmz-web ip addr add 10.0.3.10/24 dev eth1
pexec dmz-web ip link set eth1 up
pexec dmz-web ip route replace default via 10.0.3.1

echo "[7/7] Configuring Passive Sensor Promiscuous TAP..."
pexec sensor ip link set eth_sensor up promisc on
pexec sensor ip addr flush dev eth_sensor || true

echo "================================================================="
echo "ENTERPRISE NETWORK CONFIGURATION COMPLETE!"
echo "================================================================="
