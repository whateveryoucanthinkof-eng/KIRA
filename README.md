# cyberworld — Predictive Network SOC (SPAN → Topology → Dual-Branch / DeepOP)

cyberworld turns a **SPAN / port-mirror feed** into a **live host graph** and **ATT&CK-aware risk forecasts** for operators.

> Passively watch live traffic → discover who is talking to whom → score each host and forecast the next few minutes → show it on a SOC console.

It is **not** an inline firewall/IPS, and it does **not** rely on a hardcoded attacker glyph or a fixed 15-node cartoon topology. Topology is **discovery-first**: nodes and edges appear only when SPAN observes them.

**The risk score, technique and alert on the dashboard are the model's output, unmodified.** An optional,
hand-written SOC rule layer exists (`CYBERWORLD_ENABLE_RULES=1`, off by default). When enabled it is shown
*beside* the model as an advisory opinion and never replaces the model's risk, technique or alert. The
ARM EXTERNAL button only labels the console; it is never an input to scoring.

It is **near-term forecasting, not campaign forecasting.** The models see 30 s of history and forecast at most
150 s ahead. Longer history reaches the model only through the graph encoder's memory. See
[Scope and limitations](#scope-and-limitations).

Each model implements a published method: **BiTA** (encoder), **GNN-LSTM** (Branch A) and **DeepOP** (forecast
decoder). [docs/PAPER_CONFORMANCE.md](docs/PAPER_CONFORMANCE.md) maps every equation to code and lists each
deliberate deviation with its reason.

**Evaluation protocol:** trained and tuned on CIC-IDS-2018 (from PCAP), scored once on CIC-IDS-2017, with
classes 2018 never contained reported separately. Which encoder to use (Warden, CIC-2018, or Warden
fine-tuned on CIC-2018) is decided by a measured comparison. See
[docs/CROSS_YEAR_PROTOCOL.md](docs/CROSS_YEAR_PROTOCOL.md).

---

## One-sentence product

**cyberworld turns a SPAN/mirror feed into a live host graph and predictive ATT&CK-aware risk forecasts for SOC operators.**

---

## What you get

| Capability | Behavior |
|------------|----------|
| Capture | Local AF_PACKET sniff on the mirror NIC (lab sensor netns or real interface) |
| Topology | Empty until traffic; fills from observed IPs/edges; external hosts appear only when seen |
| ML | Dual-Branch + DeepOP on 2.0s flow windows (inference in control backend, not on the sniffer). The displayed verdict is the model's output |
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

### Live ML stack

```text
SPAN 5-tuple flows, grouped into 2 s windows
        │
        ▼
Data unification → UnifiedFlowRecord  (labels never used live)
        │
        ▼  interaction graph of the window (nodes = IPs, edges = flows)
TGNE-TA (BiTA): graph attention over the window + TGN memory updated by the
                BiGRU-Transformer aggregator (carries history across windows)
        │
        ▼  12-D host latent z(t)  ⊕  15 host attributes a(t)
s(t) ∈ R^27  ─────────────┬─────────────────────┐
        ▼                                        ▼
  Branch A (GNN-LSTM)                      Branch B (world model)
  LSTM over s(t-14..t)                     s(t-14..t) → ŝ(t+1..t+5)
  risk · technique · gradation                   │
        │ technique per window                   │ predicted states
        ▼                                        ▼
  DeepOP encoder (observed sequence) ──► DeepOP decoder (causal window attention)
                                                 │
                                                 ▼
                                  next ATT&CK techniques → PredictionEvent → dashboard
```

The shipped checkpoints predate this wiring and load in their earlier architectures (12-D Branch B,
decoder-only DeepOP, memoryless encoder). See "Reproducing" in
[docs/PAPER_CONFORMANCE.md](docs/PAPER_CONFORMANCE.md) for the retrain order.

| Contract | Value |
|----------|-------|
| Latent size | 12 |
| Temporal attrs | 15 |
| Model input (Branch A) | **27-D** (12 + 15) |
| Window \(\Delta t\) | **2.0 s** |
| History | **15** steps (30 s) |
| Forecast horizon \(K\) | **5** steps; see the v4 section for step size |
| Label leakage | Forbidden on live path |

The temporal contract has **one** source, `cyberworld_v4/config.py`. The old `config/temporal_contract.json`
(5 / 8 / 16 s) was read by nothing and has been deleted.

Retired: root `model/` V3.1 72-D PCAP transformer is **not** the live path.

---

## CyberWorld v4

v4 is the in-progress rework of the learned system: future-dated targets instead of nowcasting,
separate heads with matching loss semantics, a distributional world model, real conformal
prediction, post-hoc calibration, and mandatory baselines.

**v4 temporal contract** — single source of truth, `cyberworld_v4/config.py`:

| | Value |
|---|---|
| Window `Δt` | **2.0 s** |
| History `L` | **15** steps (30 s) |
| Forecast `K` | **5** steps × **30 s** = **150 s** |
| Model input | **27-D** (12-D TGNE-TA latent + 15 flow attributes) |

**The shipped checkpoints do not match this contract yet.** Branch A, Branch B and DeepOP in
`saved_models/` were trained with 15 × 2 s history and 5 × **2 s** forecast steps (10 s ahead), before
the forecast step was coarsened to 30 s. The serving adapter refuses to load them unless
`CYBERWORLD_ALLOW_CONTRACT_MISMATCH=1`. Each checkpoint's real contract, metrics and warnings are in its
`*.manifest.json`, generated from the weights by `python scripts/write_model_manifests.py`.

| Where | What |
|---|---|
| `cyberworld_v4/` | config, contract, identity, targets, splits, models, conformal, benchmark, manifest, `metrics/`, `baselines/` |
| `docs/ARCHITECTURE.md` | the 2-page ML architecture document |
| `docs/CYBER_RANGE.md` | Containerlab range, SPAN tap, capture path |
| `claude_latest_analysis/` | verified audits; `28_accuracy_changes.md` is the current-state record |

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
- **Node.js**: v20+ and `npm` to build the web dashboard — on `PATH`, or inside a toolbox named by
  `CYBERWORLD_NPM_TOOLBOX` (default `claude-dev`). The launcher builds the UI itself when needed.
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

3. **Frontend:** nothing to do by hand. `run_dashboard.py` installs the UI's dependencies and builds
   it on first run, and rebuilds whenever its sources change. To do it manually:
   ```bash
   cd web_dashboard
   npm ci
   npm run build        # the console      -> dist/
   npm run build:demo   # the demo dashboard -> dist-demo/
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

### A. SOC console — K.I.R.A. (recommended)

```bash
# Lab profile
python run_dashboard.py --site containerlab-enterprise

# Local SPAN (needs privileges on the mirror NIC)
sudo -E python run_dashboard.py --site local-default --interface eth1

# Replay a capture through the real sensor path
python run_dashboard.py --site local-default --replay captures/live.pcap
```

Opens `http://localhost:8000`. This is the real console: real traffic, real models, Containerlab
controls, and no attack buttons — attacks are run against the lab from outside it. Without trained
checkpoints it still runs (sensor, topology, lab controls) and says **Models not loaded**.

### Demo dashboard — only to see the UI

```bash
python run_dashboard.py --demo
```

Opens `http://localhost:8443`: the same UI on sample data, with no backend, sensor or models, and
in-app attack-scenario buttons. Nothing on it is a model output, and the header says **DEMO DATA**.
Use the console above for anything real. Panel-by-panel sources: `docs/DASHBOARD_INTEGRATION.md`.

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

Serving reads **only** `saved_models/`. Per-epoch encoder snapshots go to `.spill/encoder_epochs/`
(scratch, gitignored); promote one with `scripts/select_best_encoder.py --copy`. The old top-level
`saved_checkpoints/` folder has been removed: a retrain writing to the wrong one of two
similar-looking folders has already happened once.

| Component | Path |
|-----------|------|
| TGNE-TA (BiTA) | `saved_models/bita_bigru_transformer-unified_final.pth` (falls back to the legacy `bita/saved_models/bita_bigru_transformer-warden_alerts.pth`) |
| Branch A LSTM | `saved_models/branch_a/branch_a_lstm.pt` |
| Branch B WDT | `saved_models/branch_b/host_wdt.pt` |
| DeepOP CWA | `saved_models/deepop/cwa_forecast_decoder.pt` |

`scripts/ensure_checkpoints.py` can verify presence. Each `*.manifest.json` states the checkpoint's
temporal contract, the metrics it carries (and which expected ones it does not), and its credibility
verdict. Regenerate them after every retrain with `python scripts/write_model_manifests.py`.

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

External campaigns are **operator-driven** (ARM EXTERNAL). The UI does not inject synthetic attack packets as topology truth,
and arming changes **no** score: it is echoed on the event for display only.

SOAR actions (isolate, block IP/port, revoke) are **recorded, not enforced**. The model keeps scoring the traffic it
actually sees after a mitigation is recorded. If flows that the recorded block should have stopped are still on the
wire, the verdict panel shows **Mitigation not effective** instead of reporting the host as quiet.

---

## Security posture (lab firewall)

Typical Containerlab policy (see `docs/CYBER_RANGE.md` for diagrams):

- Outside range can reach public DMZ services; east-west to Users/Servers is blocked at the edge.
- Database accepts only the app tier.
- SPAN is a **mirror** (`tc mirred`) — sensor failure does not affect forwarding.

---

## Access and exposure

The console shows the internal host map and can record mitigations, so it is **loopback-only by default**
(`127.0.0.1`). CORS allows only the local console origins (`CYBERWORLD_CORS_ORIGINS` adds more).

To expose it deliberately:

```bash
python run_dashboard.py --host 0.0.0.0                            # prints a URL with a generated access token
CYBERWORLD_API_TOKEN=... python run_dashboard.py --host 0.0.0.0   # or bring your own
```

With a token set, every request and the WebSocket must carry it (`Authorization: Bearer`, or open
`/?token=…` once, which sets an HttpOnly cookie).

`python run_dashboard.py --demo` (or `npm run demo` for a dev server) runs the UI on scripted fixtures
(`web_dashboard/src/api/mock.ts`) with no backend. The header shows **DEMO DATA** in that mode so it
cannot be mistaken for the live system, and the live build contains none of those fixtures.

---

## Scope and limitations

Stated here so nobody has to discover them.

**What the models can and cannot see**

- **Near-term only.** 30 s of history (15 × 2 s windows); forecast at most 150 s ahead under the current
  contract, and 10 s for the checkpoints actually shipped. If reconnaissance happened three days ago the
  model cannot know. This is a scope gap against "attack forecasting", recorded in
  `claude_latest_analysis/28_scope_and_feature_concerns.md`.
- **Long-range history lives only in the encoder's memory.** The encoder attends over the current 2 s
  window's graph; what happened earlier reaches the embedding through the TGN memory that BiTA's
  aggregator updates (12-D per host, GRU-gated). That is BiTA's design, and it is on by default for
  every new encoder. It makes training time-ordered (no batch shuffling) and serving stateful (memory
  is per session and cleared on reset). **The shipped encoder was trained with memory off**, which meant
  its BiTA aggregator never ran; it must be retrained to be a BiTA encoder at all.
- **Campaign correlation is a heuristic.** `correlation/` links alerts with hand-set kill-chain priors
  (every constant is in `causal_edge_scorer.HEURISTIC_PARAMS`; none were fitted). It is bounded to the
  models' evidence horizon (history + forecast) and splits campaigns on time gaps. The live backend runs
  it over the scored windows (`control_backend/correlation_service.py`) to fill the Campaign and
  Incidents pages; the payload labels it as heuristic.
- **Branch B may not be learning.** Its first run flatlined at epoch 1. Retraining now records its skill
  against persistence (copying the last embedding forward) in the checkpoint, and DeepOP refuses to train
  on a Branch B that does not beat it. The shipped Branch B predates that check.
- **Edge-feature ablation has not been run.** `dst_port_norm_65535` is a shortcut risk: a model that can
  read the port can learn "port ⇒ class" instead of behaviour. The ablation
  (`CYBERWORLD_ABLATE_EDGE_FEATURES`) is now recorded in the encoder config and enforced at load, but the
  comparison needs an encoder retrain that has not been done.

**Training data problems that cannot be fixed in code**

- 9 of 10 CIC-IDS-2018 CSV days fabricate host IPs from the row number, so host trajectories built from
  them are synthetic. The PCAP path fixes this and is not yet wired into training.
- All 158,930 CIC-IDS-2017 PortScan (recon) records come from a single host, `172.16.0.1`.
- The held-out validation split has 2 technique classes against 7 in training, so performance on the
  other 5 cannot be measured.
- The corpus is ~82.5% Benign, so accuracy cannot visibly fail on it. **Quote macro-F1, AUC and Brier,
  not accuracy.** The shipped Branch A checkpoint records only accuracy, and its manifest says so.

---

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_verdict_is_the_model.py` pins the behaviour above: external traffic is not forced to
ELEVATED, internal risk is not damped, the ARM button does not move the score, rules are advisory, and a
recorded mitigation does not hide ongoing traffic.

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
| Frontend missing | `python run_dashboard.py --rebuild`, or `cd web_dashboard && npm ci && npm run build` |
| "Models not loaded" in the header | No loadable checkpoints in `saved_models/`; the tooltip and Controls page give the reason. Retrain, then press Start Inference. |
| Checkpoint errors | Ensure `saved_models/` and `bita/saved_models/` files exist |

---

## License / notes

Research and competition-oriented cyber-range + predictive SOC observation stack. Blocking/SOAR actions in the UI are **recorded intents** unless wired to real enforcement.
                                                                                                   