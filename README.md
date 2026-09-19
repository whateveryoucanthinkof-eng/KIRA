# cyberworld — Predictive Network SOC (SPAN → Topology → Dual-Branch / DeepOP)

cyberworld turns a **SPAN / port-mirror feed** into a **live host graph** and **ATT&CK-aware risk forecasts** for operators.

> Passively watch live traffic → discover who is talking to whom → forecast near-future attack evolution → show it on a SOC console.

It is **not** an inline firewall/IPS, and it does **not** rely on a hardcoded attacker glyph or a fixed 15-node cartoon topology. Topology is **discovery-first**: nodes and edges appear only when SPAN observes them.

---

## One-sentence product

**cyberworld turns a SPAN/mirror feed into a live host graph and predictive ATT&CK-aware risk forecasts for SOC operators.**

---

## What you get

| Capability | Behavior |
|------------|----------|
| Capture | Local AF_PACKET sniff on the mirror NIC (lab sensor netns or real interface) |
| Topology | Empty until traffic; fills from observed IPs/edges; external hosts appear only when seen |
| ML | Dual-Branch + DeepOP on 2.0s flow windows (inference in control backend, not on the sniffer) |
| Predictions | Bind to `focus_ips` / edges — never hardcoded `dmz-web` / `attacker` IDs |
| Lab Mode | Optional Containerlab deploy/destroy/workloads when `lab_mode: true` |
| Portability | Same model path for Containerlab **and** a real SPAN NIC via site YAML |

---

## Architecture

```text
[ Enterprise / lab hosts ]
          │
     SPAN / mirror
          ▼
┌─────────────────────────────────────┐
│ LOCAL CAPTURE  (telemetry/)         │
│ AF_PACKET → flow table → 2s windows │
│ JSONL stream (--no-inference)       │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ CONTROL BACKEND                     │
│  • topology_service → discovery graph│
│  • model_adapter → Dual-Branch+DeepOP│
│  • site_config → CIDRs / lab_mode   │
│  • commands → Lab Mode (optional)   │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ WEB DASHBOARD  (web_dashboard/)     │
│ Live graph + forecasts + controls   │
└─────────────────────────────────────┘
```

### Live ML stack (authoritative)

```text
SPAN 5-tuple flows
        │
        ▼
Data unification → UnifiedFlowRecord  (labels never used live)
        │
        ▼
TGNE-TA (BiTA) → 12-D host latent H
        ├──────────────────┐
        ▼                  ▼
  Branch A LSTM      Branch B WDT
  technique/risk     K-step latent rollout
        │                  │
        └────────┬─────────┘
                 ▼
          DeepOP / CWA
     future ATT&CK technique sequence
                 ▼
        PredictionEvent → dashboard
```

| Contract | Value |
|----------|-------|
| Latent size | 12 |
| Temporal attrs | 15 |
| Model input (Branch A) | **27-D** (12 + 15) |
| Window \(\Delta t\) | **2.0 s** |
| History | **5** steps (live) |
| Forecast horizon \(K\) | **8** steps (~16 s) |
| Label leakage | Forbidden on live path |

Retired: root `model/` V3.1 72-D PCAP transformer is **not** the live path.

---

## Repository layout

| Path | Role |
|------|------|
| `telemetry/` | Packets → windows → flows (no ML) |
| `control_backend/` | FastAPI, WS bus, topology, Dual-Branch adapter, Lab commands |
| `web_dashboard/` | React SOC UI (Vite) |
| `config/sites/` | Per-deployment YAML (CIDRs, sensor iface, `lab_mode`) |
| `bita/` | TGNE-TA encoder |
| `branch_a_gnn_lstm/` | Multi-task LSTM |
| `branch_b_world_model/` | World Dynamics Transformer + risk head |
| `deepop_decoder/` | CWA forecast decoder |
| `data_unification/` | UnifiedFlowRecord + temporal features |
| `saved_models/` | Production checkpoints (Branch A/B, DeepOP) |
| `containerlab/` | Optional enterprise cyber-range |
| `scripts/` | Deploy, destroy, telemetry, dashboard helpers |
| `docs/LOCAL_SPAN_RUNBOOK.md` | Lab + generic NIC bring-up |

---

## Requirements

- **OS**: Linux (Fedora / RHEL / Ubuntu / Debian)
- **Python**: 3.10+
- **Node.js**: v18+ and `npm` (required to build the web dashboard)
- **For Lab Mode**: Podman (or Docker) + Containerlab ≥ 0.50, `iproute2` / `tc`
- **For live capture**: `CAP_NET_RAW` (or root) on the mirror interface; promiscuous mode often required

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd cyberworld
   ```

2. **Install Python dependencies:**
   We recommend using a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install Frontend dependencies:**
   Build the React SOC dashboard.
   ```bash
   cd web_dashboard
   npm install
   npm run build
   cd ..
   ```

---

## Site configuration

Profiles live under `config/sites/`:

| Profile | `lab_mode` | Use |
|---------|------------|-----|
| `containerlab-enterprise` | `true` | Cyber-range + orchestration UI |
| `local-default` | `false` | Real / generic local SPAN |

Override with:

```bash
export CYBERWORLD_SITE=local-default
# or
export CYBERWORLD_SITE_CONFIG=/path/to/site.yaml
export CYBERWORLD_SENSOR_IFACE=eth1
```

> The older lowercase spellings (`cyberworld_SITE`, …) still work but are deprecated — the backend
> logs a warning naming the variable. Prefer the uppercase form.

Minimal fields: `enterprise_cidrs`, `sensor.interface`, `topology.node_ttl_sec` / `edge_ttl_sec` / `max_nodes`, optional `assets_of_interest`.

---

## Quickstart

### A. SOC dashboard (recommended)

```bash
# Lab profile
python run_dashboard.py --site containerlab-enterprise

# Local SPAN (needs privileges on the mirror NIC)
sudo -E python run_dashboard.py --site local-default --interface eth1
```

Opens `http://localhost:8000` (builds `web_dashboard` if `dist/` is missing).

**Lab UI flow:** START NETWORK → START SENSOR → START ML  

**Local UI flow:** START SENSOR → START ML (no deploy/destroy)

### B. Containerlab range only

```bash
./scripts/build.sh
./scripts/deploy.sh          # topology + SPAN mirrors → eth_sensor
./scripts/healthcheck.sh
./scripts/destroy.sh         # teardown
```

### C. Capture-only telemetry (ML stays in backend)

```bash
# From dashboard: START SENSOR
# Or manually (lab sensor netns):
./scripts/run_telemetry.sh --record-state /tmp/cyberworld_live_stream.jsonl --no-inference
```

Capture remains **`--no-inference`**. Dual-Branch / DeepOP runs in `control_backend`.

More detail: [docs/LOCAL_SPAN_RUNBOOK.md](docs/LOCAL_SPAN_RUNBOOK.md).

---

## Checkpoints

| Component | Path |
|-----------|------|
| TGNE-TA (BiTA) | `bita/saved_models/bita_bigru_transformer-warden_alerts.pth` |
| Branch A LSTM | `saved_models/branch_a/branch_a_lstm.pt` |
| Branch B WDT | `saved_models/branch_b/host_wdt.pt` |
| DeepOP CWA | `saved_models/deepop/cwa_forecast_decoder.pt` |

`scripts/ensure_checkpoints.py` can verify presence.

---

## Dynamic topology rules

1. Nodes/edges come only from observed flows.
2. No permanent attacker node — outside IPs appear as `role: external` when traffic is seen, then expire after TTL.
3. Internal vs external from **site CIDRs**, not topology fixtures.
4. Predictions highlight **IPs / edges** via `focus_ips` / `focus_edges`.

API: `GET /api/topology` · WS event: `topology_update` · Site: `GET /api/site` · Status: `GET /api/status` (includes `lab_mode`).

---

## Lab Mode vs production site

| | Lab Mode (`lab_mode: true`) | Local SPAN (`lab_mode: false`) |
|--|-----------------------------|-------------------------------|
| Deploy / destroy / workloads | Shown & allowed | Hidden & rejected by API |
| “Online” | Containerlab container count | Sensor streaming / window heartbeat |
| Sensor | `clab-enterprise-sensor` / `eth_sensor` | Configured NIC (`eth1`, …) |

---

## Normal enterprise workloads (Lab)

| Node | Profile |
|------|---------|
| `ws-office` | General office browsing / DNS / idle |
| `ws-web` | High-frequency HTTP to DMZ + internal |
| `ws-file` | Bursty file transfer to `srv-file` |
| `ws-app` | API traffic to `srv-app` (+ backend tiers) |

External campaigns are **operator-driven** (ARM EXTERNAL). The UI does not inject synthetic attack packets as topology truth.

---

## Security posture (lab firewall)

Typical Containerlab policy (see `docs/ARCHITECTURE.md` for diagrams):

- Outside range can reach public DMZ services; east-west to Users/Servers is blocked at the edge.
- Database accepts only the app tier.
- SPAN is a **mirror** (`tc mirred`) — sensor failure does not affect forwarding.

---

## Tests

```bash
python -m pytest tests/test_site_config.py tests/test_topology_service.py tests/test_control_backend.py -q
```

Covers CIDR classification, primary-host selection, topology TTL/caps, Lab Mode gating, Dual-Branch smoke.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Empty topology | Sensor running? Any flows in the mirror? Wait for conversations. |
| Sensor won’t start (lab) | `./scripts/deploy.sh` so `clab-enterprise-sensor` is up. |
| Sensor won’t start (local) | Root/`CAP_NET_RAW`, correct `--interface`, interface exists. |
| Lab buttons missing | `lab_mode: false` — expected for `local-default`. |
| `start_network` rejected | Site is not Lab Mode — switch to `containerlab-enterprise`. |
| No packets on tap | `podman exec clab-enterprise-sensor tcpdump -c 10 -ni eth_sensor` |
| Frontend missing | `cd web_dashboard && npm install && npm run build` |
| Checkpoint errors | Ensure `saved_models/` and `bita/saved_models/` files exist |

---

## License / notes

Research and competition-oriented cyber-range + predictive SOC observation stack. Blocking/SOAR actions in the UI are **recorded intents** unless wired to real enforcement.
