# Local SPAN bring-up (CyberWorld)

## Lab (Containerlab)

```bash
export CYBERWORLD_SITE=containerlab-enterprise
./scripts/deploy.sh
python run_dashboard.py --site containerlab-enterprise
# UI: START NETWORK (if needed) → START SENSOR → START ML
```

Sensor interface: `eth_sensor` inside `clab-enterprise-sensor` (via `scripts/run_telemetry.sh`).

## Generic local NIC (no Containerlab)

1. Attach the NIC that receives the SPAN/mirror to this host.
2. Edit `config/sites/local-default.yaml`: set `enterprise_cidrs` and `sensor.interface`.
3. Run with privileges for AF_PACKET / promiscuous capture:

```bash
sudo -E python run_dashboard.py --site local-default --interface eth1
# UI: START SENSOR → START ML (no deploy/destroy)
```

Or:

```bash
export CYBERWORLD_SITE=local-default
sudo -E python run_dashboard.py --interface eth1
```

**Permissions:** raw capture typically needs `CAP_NET_RAW` (or root). Promiscuous mode may be required on the mirror port.

**Visibility:** topology only includes conversations present in the mirrored feed.

## Checks

- Fresh UI: empty topology until flows.
- Internal traffic → internal nodes.
- Outside→inside → `external` node appears; expires after `topology.node_ttl_sec`.
- Predictions highlight `focus_ips`, not fixed lab IDs.
