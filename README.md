<div align="center">

<img src="web_dashboard/public/kira-mark.png" alt="K.I.R.A." width="96" />

# K.I.R.A.

### Kinetic Intrusion Risk Anticipator

**AI-based network attack forecasting from network traffic: a world model for predictive cyber defence**

▶ [Product demo](https://youtu.be/pUHwS8UNEDU) · ▶ [ML model explained](https://youtu.be/21zskAGmkDk)

Smart India Hackathon 2026 · Problem Statement **26153** (NTRO) · Theme: Blockchain & Cybersecurity

</div>

---

K.I.R.A. watches a network through a **SPAN / port-mirror feed**, learns how each host's state evolves
over time, and **forecasts** where an attack is heading: the probability of compromise over the next
150 seconds, the next MITRE ATT&CK stages, and the traffic features driving each prediction, all on a
live SOC console.

It is a world model, not a static classifier. A temporal graph network encodes the evolving network
state, a transformer learns the state-transition dynamics `P(S(t+1) | S(t))` as a predictive
distribution, and a decoder turns the rolled-out future into ATT&CK stages.

> [!IMPORTANT]
> **The models are still in active development.** The full pipeline (data ingestion, training,
> evaluation, serving and the console) is complete and covered by 1,296 automated tests. Retraining on
> the corrected labelling and training pipeline is in progress. Until a retrained model is promoted, the
> console runs with the sensor and topology live and reports **Models not loaded**. The **demo
> dashboard** shows the full interface on sample data.

> [!NOTE]
> **Linux only.** Live capture uses Linux `AF_PACKET` sockets on the mirror interface, the cyber range
> runs on Containerlab, and training jobs run under systemd memory limits. Network sensors and SOC
> servers run Linux, and so does K.I.R.A. Only the demo dashboard runs anywhere Python and Node.js do.

## Contents

[Deliverables](#deliverables) ·
[What it does](#what-it-does) ·
[Architecture](#architecture) ·
[Repository layout](#repository-layout) ·
[Setup](#setup) ·
[Datasets](#datasets) ·
[Train the models](#train-the-models) ·
[Run the console](#run-the-console) ·
[Live capture over SPAN](#live-capture-over-span) ·
[Tests](#tests) ·
[Status and limitations](#status-and-limitations)

## Deliverables

| Deliverable | Where |
|---|---|
| Source code | this repository |
| README with setup instructions | this file |
| Architecture document (2 pages) | [`docs/ARCHITECTURE.pdf`](docs/ARCHITECTURE.pdf) · [Markdown](docs/ARCHITECTURE.md) |
| Demo video (2 min) | ▶ [Product demo](https://youtu.be/pUHwS8UNEDU) · ▶ [ML model explained](https://youtu.be/21zskAGmkDk) on YouTube; the files are in the repository root ([`SIH_FULL_VIDEO.mp4`](SIH_FULL_VIDEO.mp4), [`Product_Demo.mp4`](Product_Demo.mp4), [`ML_Model_explained.mp4`](ML_Model_explained.mp4), Git LFS) |
| Technical presentation (5 slides) | repository root |
| Problem statement | [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) |

## What it does

| The problem statement asks for | K.I.R.A. |
|---|---|
| Flow-level **and** packet-level features | 12 flow features per edge (bytes, packets, duration, rates, protocol, port, direction) and 15 flow attributes per host; 30 packet-level attributes per host (TTL mean and variance, TCP window, retransmissions, inter-arrival statistics, payload sizes, SYN/RST patterns, vertical and horizontal scan signatures) parsed from PCAP in Rust |
| Network state as a graph | every 2 s window is an interaction graph: hosts are nodes, flows are edges |
| Learn `P(S(t+1) \| S(t))` | **Branch B**, a world-model transformer: a Gaussian predictive distribution over the next 5 host states, with per-step uncertainty |
| K-step forward simulation | autoregressive rollout, 5 steps × 30 s = 150 s ahead |
| Infiltration probability over time | hazard-based risk: probability of attack onset within the horizon, per step |
| ATT&CK stage mapping | **Branch A** (technique and kill-chain stage now) and **DeepOP** (next ATT&CK techniques along the rollout) |
| Driving features | Input × Gradient attribution over every input window and feature, grouped (TCP flags, scan signature, timing, volume, …), plus which windows mattered |
| Logistic-regression benchmark | trained on identical inputs and target; precision, recall, F1, false-positive rate and AUC on the held-out test, overall and for **early warning** (hosts not yet under attack) |
| Offline interface with PCAP input | the console replays a PCAP through the real sensor path; fully offline |

## Architecture

```text
 SPAN / mirror port ──► telemetry/  (AF_PACKET → flows → 2 s windows)
                                     │
                                     ▼
              ┌───────────────────────────────────────────────┐
              │ TGNE encoder (BiTA)                  bita/    │
              │ temporal graph attention + TGN memory,        │
              │ updated by a BiGRU-Transformer aggregator     │
              └──────────────────────┬────────────────────────┘
                                     │  host state s(t) = 12-D latent ⊕ host attributes
                    ┌────────────────┴───────────────┐
                    ▼                                ▼
     Branch A (GNN-LSTM)                  Branch B (world model)
     risk · technique · stage now         P(s(t+1..t+5) | s(t-14..t))
                    │                                │ predicted states + uncertainty
                    └──────────────┬─────────────────┘
                                   ▼
                     DeepOP decoder ──► next ATT&CK stages
                                   │
                                   ▼
          control_backend/ (FastAPI) ──► K.I.R.A. console (web_dashboard/)
          risk timeline · forecast tree · kill chain · explanations · campaigns
```

Each model follows a published method: **BiTA** for the encoder, **GNN-LSTM** for Branch A and
**DeepOP** for the decoder. [`docs/PAPER_CONFORMANCE.md`](docs/PAPER_CONFORMANCE.md) maps their
equations to the code. The two-page design summary is [`docs/ARCHITECTURE.pdf`](docs/ARCHITECTURE.pdf).

**Evaluation protocol.** Trained on CSE-CIC-IDS2018 (from raw PCAP) and CTU-13, tuned on held-out 2018
days, and scored **once** on CIC-IDS2017, a different year, network and attack mix. The numbers then
measure generalisation rather than memorised signatures
([`docs/CROSS_YEAR_PROTOCOL.md`](docs/CROSS_YEAR_PROTOCOL.md)).

## Repository layout

```text
KIRA/
├── train.sh                  one command: check data → train every model → promote to serving
├── run_dashboard.py          one command: the live console, or --demo
├── requirements.txt
├── data/                     your datasets (layout in data/README.md; git-ignored)
├── docs/                     architecture, evaluation protocol, cyber range, SPAN runbook
│
├── telemetry/                packet capture → flow table → 2 s windows (no ML)
├── data_unification/         dataset adapters (CIC-2017/2018, CTU-13, PCAP), labels, features, splits
├── rust/                     pcap_fast (PCAP parser) and tgn_host (neighbour sampler), bit-exact ports
├── bita/                     TGNE encoder: temporal graph network + BiTA aggregator
├── branch_a_gnn_lstm/        Branch A: LSTM risk / technique / stage heads + logistic baseline
├── branch_b_world_model/     Branch B: world-dynamics transformer + infiltration risk head
├── deepop_decoder/           DeepOP: ATT&CK forecast decoder
├── cyberworld_v4/            temporal contract, targets, metrics, conformal prediction, training guard
├── correlation/              campaign correlation over alerts (heuristic)
├── explainability/           feature attribution and explanation payloads
├── control_backend/          FastAPI backend: sensor, topology, model serving, WebSocket bus
├── web_dashboard/            the K.I.R.A. SOC console (React + Vite)
├── saved_models/             checkpoints the console serves
│
├── containerlab/  nodes/  workloads/  config/    enterprise cyber range and site profiles
├── captures/                 sample PCAP for replay
├── scripts/                  training, evaluation, dataset checks, range helpers
├── world_model/              research baselines (logistic regression, LSTM, static GCN)
└── tests/                    1,296 automated tests
```

## Setup

| Requirement | Needed for |
|---|---|
| Linux (Fedora, Ubuntu, Debian, RHEL) | everything except the demo |
| Python 3.10+ (3.12 tested) | everything |
| Node.js 20+ and npm | building the console UI (automatic on first launch) |
| NVIDIA GPU with CUDA, 8 GB+ | training (CPU works, far slower); inference runs on CPU |
| Rust (`cargo`) | training from PCAP (12× faster parsing; a Python fallback exists) |
| root or `CAP_NET_RAW` | live capture on a SPAN interface |
| Podman or Docker + Containerlab ≥ 0.50 | the optional enterprise cyber range |

```bash
git clone git@github.com:SIH-2026-SSSVBT/KIRA.git
cd KIRA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The console UI is built automatically on first launch. If `npm` lives inside a toolbox rather than on
`PATH`, set `CYBERWORLD_NPM_TOOLBOX=<toolbox name>`.

## Datasets

All public. Download them and place them under `data/` exactly as described in
**[`data/README.md`](data/README.md)**:

```text
data/cic2018/pcap/<day>_pcap/…        CSE-CIC-IDS2018 raw PCAPs            (training)
data/cic2018/csv/<day>_csv.csv        CSE-CIC-IDS2018 labels for the PCAPs (training)
data/cic2017/TrafficLabelling/*.csv   CIC-IDS2017 labelled flows           (held-out test)
data/ctu13/<1..13>/*.binetflow        CTU-13 botnet scenarios              (training)
```

To keep the data elsewhere, symlink these folders, or set `PCAP_ROOT`, `CIC2018_CSV_DIR`,
`CIC2017_DIR` and `CTU_DIR`. Check the layout with `./train.sh check`.

## Train the models

```bash
./train.sh check      # datasets present and correctly named, toolchain OK
./train.sh dryrun     # optional: the whole pipeline on a tiny synthetic corpus, in minutes
./train.sh            # train everything
./train.sh summary    # results: mean and spread over seeds, held-out benchmark
./train.sh promote    # install the trained models into saved_models/ for the console
```

`./train.sh` runs, in order:

1. dataset preflight, then parsing every capture once into a cache (Rust);
2. the TGNE encoder, with and without IP-identity features, over three seeds;
3. Branch A and its logistic-regression benchmark;
4. Branch B, then DeepOP, but only if Branch B beats the copy-the-last-state baseline.

Every model trains under the same guard: learning-rate warm-up, gradient clipping, roll-back to the
best epoch at half the learning rate, and early stopping. A crashed job resumes from its last finished
epoch. Outputs, logs and comparison tables go to `results/training_plan/`.

A full run takes many hours and about 50 GB of free disk. Heavy jobs are capped at `MEM_MAX` (default
17G). Optional settings are listed in [`scripts/ops/plan_env.example.sh`](scripts/ops/plan_env.example.sh).

## Run the console

### Demo dashboard: see the interface

```bash
python run_dashboard.py --demo                 # → http://localhost:8443
```

The full K.I.R.A. interface on scripted sample data, with in-app attack scenarios. No backend, sensor
or model runs, and the header reads **DEMO DATA**.

### Real console

```bash
python run_dashboard.py --replay captures/live.pcap                      # replay a capture
sudo -E python run_dashboard.py --site local-default --interface eth1    # live SPAN capture
python run_dashboard.py --site containerlab-enterprise                   # the cyber range
```

Opens `http://localhost:8000`. Real traffic, real models, and no attack buttons: attacks are run
against the network from outside. In the console: **START SENSOR → START ML** (on the cyber range,
**START NETWORK** first). Every number on screen is a model output. Panels for capabilities no model
provides yet are marked *Under development* ([`docs/DASHBOARD_INTEGRATION.md`](docs/DASHBOARD_INTEGRATION.md)).

The console binds to `127.0.0.1`. `--host 0.0.0.0` exposes it and prints a generated access token.

## Live capture over SPAN

1. Mirror the traffic to watch onto a NIC of the K.I.R.A. host (switch SPAN / port-mirror, or a TAP).
2. Describe the site in [`config/sites/local-default.yaml`](config/sites/local-default.yaml): your
   internal CIDRs and the capture interface.
3. Start the console with capture privileges:

   ```bash
   sudo -E python run_dashboard.py --site local-default --interface eth1
   ```

4. In the console: **START SENSOR → START ML**.

K.I.R.A. is passive. It only reads the mirror, so a sensor failure never affects the network. Hosts and
connections appear as traffic is observed, and addresses outside the site's CIDRs are shown as external.

**Cyber range.** [`containerlab/`](containerlab/) builds a small enterprise network (DMZ, servers,
workstations, database, identity server) with its own SPAN mirror and realistic background workloads:

```bash
./scripts/build.sh && ./scripts/deploy.sh      # bring the range up
python run_dashboard.py --site containerlab-enterprise
./scripts/destroy.sh                           # tear it down
```

More: [`docs/LOCAL_SPAN_RUNBOOK.md`](docs/LOCAL_SPAN_RUNBOOK.md), [`docs/CYBER_RANGE.md`](docs/CYBER_RANGE.md).

## Tests

```bash
python -m pytest tests/ -q
```

1,296 tests, covering feature parity between training and serving, Rust/Python parser equivalence,
label scoping, leakage guards, metric correctness and the console's contracts. GPU-only tests skip
without CUDA, and dataset-dependent tests skip without `data/`.

## Status and limitations

- **Models in development.** The checkpoints in `saved_models/` predate the current feature schema and
  are not served. Retraining on the corrected pipeline is under way, and the console reports *Models
  not loaded* until `./train.sh promote` installs new ones.
- **Labels.** CSE-CIC-IDS2018 publishes attack *time windows*, and nine of its ten label files carry no
  IP addresses. Traffic in an attack window that cannot be attributed to a host is left out of
  training rather than guessed (`CYBERWORLD_UNSCOPED_LABELS`).
- **Horizon.** 30 s of history and 150 s of forecast. Longer context reaches the model only through the
  encoder's memory, so multi-day campaigns are out of scope.
- **Not modelled yet:** the lateral-movement and execution stages (no labelled training data), traffic
  volume forecasts, and per-alternative risk curves. The console marks these *Under development*.
- **Campaign correlation** is a hand-tuned heuristic over model outputs, and is labelled as one.
- **Packet-level features** are opt-in (`--packet-features`). The CIC-IDS2017 test set has flows only,
  so they need a PCAP-derived test set to be evaluated fairly.

## Acknowledgements

Datasets: CSE-CIC-IDS2018 and CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New
Brunswick); CTU-13 (Stratosphere Lab, Czech Technical University). Methods: BiTA (Makki Nayeri and
Rezvani); GNN-LSTM attack-vector reconstruction (Vitulyova et al., 2025); DeepOP (Zhang, Xue and Su,
2025); Temporal Graph Networks (Rossi et al., 2020); MITRE ATT&CK.
