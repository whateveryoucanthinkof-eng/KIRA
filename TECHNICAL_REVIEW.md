# CyberWorld Technical Review

## 1. Purpose and Scope

CyberWorld is a passive network-observation and predictive SOC system. It accepts traffic from a SPAN or port-mirror interface, discovers communicating hosts and edges, builds temporal flow features, runs a multi-model prediction stack, and presents forecasts and explanations in a web dashboard.

The repository supports two deployment modes:

| Mode | Purpose | Network source | Containerlab required | ML path |
|---|---|---|---:|---|
| Local SPAN | Observe a real mirror interface | Host AF_PACKET socket | No | Live Dual-Branch/DeepOP |
| Containerlab Lab Mode | Reproducible enterprise cyber range | Mirrored traffic from the `sensor` node | Yes | Live Dual-Branch/DeepOP |

The most important architectural rule is that capture is passive and inference is owned by the control backend:

```text
SPAN / mirror NIC
    -> AF_PACKET capture
    -> packet parsing and 5-tuple flow windows
    -> JSONL telemetry stream
    -> FastAPI control backend
    -> topology discovery + target selection
    -> TGNE-TA / BiTA host embedding
    -> Branch A risk and technique prediction
    -> Branch B latent rollout and infiltration risk
    -> DeepOP future ATT&CK token forecast
    -> WebSocket events
    -> React SOC dashboard
```

This document describes the current code, not the retired V3.1 72-D transformer design. Some historical terminology remains in source comments and architecture documents and is called out at the end.

## 2. Current Production Model Contract

The live model adapter is `control_backend/model_adapter.py`. Its promoted stack is:

1. **TGNE-TA / BiTA**: converts host flow histories into a 12-dimensional host latent.
2. **Branch A**: consumes the 12-D latent plus 15 observable temporal attributes, giving a 27-D per-step input to a multitask LSTM.
3. **Branch B**: autoregressively rolls the 12-D latent into future host states and produces future infiltration risk.
4. **DeepOP / CWA decoder**: conditions on future Branch B states and predicts future network-observable ATT&CK technique tokens.

The live temporal contract is approximately:

| Quantity | Live value |
|---|---:|
| Capture window | 2.0 seconds |
| Host latent dimension | 12 |
| Branch A input dimension | 27 = 12 latent + 15 temporal attributes |
| Branch A history | 5 steps |
| Branch B rollout horizon | 8 steps |
| Forecast duration | About 16 seconds at 2-second windows |
| Live labels | Not available; ground-truth labels are not used |

The 72-D state assembled by `telemetry/state/state_builder.py` is a capture and recording artifact inherited from the earlier packet-feature pipeline. The live adapter primarily consumes normalized flow records and host trajectories, not the old V3.1 checkpoint contract.

## 3. End-to-End Runtime

### 3.1 Launch

`run_dashboard.py` is the main launcher. It selects a site profile, applies environment overrides, builds or serves the frontend bundle when required, and starts the FastAPI application with Uvicorn.

Important environment variables include:

- `CYBERWORLD_SITE`: site profile name, normally `containerlab-enterprise` or `local-default`.
- `CYBERWORLD_SITE_CONFIG`: explicit YAML configuration path.
- `CYBERWORLD_SENSOR_IFACE`: override for the observed host interface.

The backend is started from `control_backend.main:app` and serves the built dashboard when `web_dashboard/dist` exists.

### 3.2 Backend lifecycle

`control_backend/main.py` owns the FastAPI application and lifecycle hooks:

- Initializes the event broker loop.
- Exposes status, site, topology, scenario, command, mitigation, and WebSocket endpoints.
- Stops sensor and ML services during shutdown.
- Computes online state differently for Lab Mode and Local SPAN mode.

Main endpoints:

| Endpoint | Role |
|---|---|
| `GET /api/status` | Runtime, sensor, ML, topology, site, and model status |
| `GET /api/site` | Active site profile and deployment properties |
| `GET /api/topology` | Current discovery graph |
| `GET /api/scenarios` | Available operator scenario metadata |
| `POST /api/command/{cmd_name}` | Execute an allowlisted operational command |
| `POST /api/mitigate` | Record a mitigation action such as IP blocking |
| `WS /ws` | Stream status, topology, prediction, attack, and command events |

### 3.3 Sensor and inference lifecycle

`control_backend/telemetry_service.py` manages the capture process and JSONL stream.

1. `start_sensor()` resets the stream and topology state.
2. In Lab Mode it verifies the `clab-enterprise-sensor` container.
3. It starts `scripts/run_telemetry.sh`, or starts `telemetry/run_telemetry.py` directly for Local SPAN.
4. A tail worker reads one completed JSON object per telemetry window.
5. Raw flow dictionaries are converted to `UnifiedFlowRecord` objects.
6. Mitigation filters are applied.
7. `select_primary_target()` chooses the most relevant internal host when needed.
8. `topology_service.apply_window()` updates the observed graph.
9. When ML is enabled, `model_adapter.predict_window()` creates a `PredictionEvent`.
10. Topology and prediction events are published through `event_broker.py`.

ML is deliberately a separate control action. Starting a sensor does not automatically load or run predictions for every packet.

## 4. Network Capture and Telemetry

### 4.1 AF_PACKET capture

`telemetry/capture/sniffer.py` opens a Linux raw `AF_PACKET` socket. It parses Ethernet traffic and supports the packet information needed by the flow and packet feature engines:

- Ethernet and VLAN headers.
- IPv4 and IPv6 addresses.
- TCP and UDP ports.
- Packet length and timestamps.
- TCP flags, sequence information, TTL, and payload metadata where available.

The process requires `CAP_NET_RAW` or root privileges. In a lab, the sensor process runs in the sensor network namespace. In Local SPAN mode it observes the configured host NIC directly.

### 4.2 Flow table

`telemetry/flow/flow_table.py` maintains active bidirectional 5-tuple flows. It canonicalizes the two directions so forward and backward traffic can be accumulated into one logical flow.

Tracked information includes:

- Source and destination IPs.
- Source and destination ports.
- Protocol.
- Forward and backward bytes.
- Forward and backward packet counts.
- First and last timestamps.
- Duration and inter-arrival statistics.
- TCP flag counters.
- Port and peer diversity.
- Fanout and internal graph indicators.
- Administrative-port activity.

The table emits flow dictionaries used by `flows_from_span_dicts()` and downstream `UnifiedFlowRecord` construction.

### 4.3 Packet behavioral features

`telemetry/packet/pcap_engine.py` computes packet-level behavioral features in memory. Examples include:

- TTL mean, spread, and diversity.
- TCP window statistics.
- Retransmission estimates.
- Packet inter-arrival moments.
- Packet length moments.
- TCP flag ratios.
- Handshake completion and unanswered SYN ratios.
- Fragmentation indicators.
- Internal graph expansion.
- Host and pair diversity.

It does not load a model checkpoint. It is a feature extractor only.

### 4.4 Window state builder

`telemetry/state/state_builder.py` closes non-overlapping two-second windows. It maintains:

- The current packet list.
- A live flow table.
- A packet feature engine.
- A rolling state buffer.
- Window identifiers and latency measurements.

It can emit raw and scaled state arrays and a list of active flows. `telemetry/run_telemetry.py` removes heavyweight arrays before writing the JSONL stream, so the backend receives practical flow-window records rather than a model inference result from the sensor.

The module still exposes 70, 72, and 73 feature modes and a 72-feature canonical array because of historical compatibility. Those arrays are not the current production model input contract.

### 4.5 JSONL stream

`telemetry/run_telemetry.py` is capture-only:

- Starts `StreamingPacketSniffer`.
- Feeds packets into `LiveStateBuilder`.
- Closes a window every two seconds.
- Optionally records a reduced JSONL state record.
- Optionally records raw packets to PCAP.
- Prints flow and pipeline-latency summaries.

Inference does not run in this process. The `--no-inference` option is retained as the explicit operating mode. The old replay path exits rather than replaying through the retired V3.1 package.

## 5. Dynamic Topology

`control_backend/topology_service.py` implements a discovery-first graph.

### Rules

1. A node exists only after its IP appears in an observed flow.
2. An edge exists only after communication is observed.
3. CIDR membership determines `internal` versus `external` role.
4. Nodes and edges expire after configured TTLs.
5. Maximum node and edge limits bound noisy sites.
6. Predictions identify observed IPs and edges, not fixed dashboard node names.

`control_backend/schema.py` defines the topology event models, including nodes, edges, statistics, and `topology_update` events. `control_backend/event_broker.py` retains the latest status, topology, and prediction event for newly connected WebSocket clients.

## 6. Data Unification

The `data_unification/` package provides the common record contract used by training and live inference.

### 6.1 Canonical schema

`unified_schema.py` defines `UnifiedFlowRecord`, including:

- Flow identity and timestamps.
- IPs, ports, and protocol.
- Forward/backward traffic statistics.
- Raw label and label source.
- Coarse category.
- ATT&CK technique IDs.
- Dataset and provenance metadata.

For live SPAN records, ground-truth labels are unavailable. The live path uses an unlabeled source and must not read dataset attack labels.

### 6.2 Dataset adapters

- `cic2017_adapter.py`: CIC-IDS2017 flow records.
- `cic2018_adapter.py`: CIC-IDS2018 flow records.
- `ctu13_adapter.py`: CTU-13 records.
- `warden_adapter.py`: Warden data.
- `auth_log_adapter.py`: authentication events and optional identity context.

`multi_dataset_stream.py` combines normalized records into host windows and trajectories. `split_manager.py` produces disjoint scientific train, validation, and test partitions. `label_resolver.py` maps dataset-specific labels to common categories and ATT&CK techniques.

### 6.3 Feature and temporal helpers

- `flow_to_temporal_event.py`: converts flow records into temporal graph events.
- `tgne_features.py`: computes observable host and edge features for the graph encoder.
- `behavioral_fingerprint.py`: derives behavioral categories from flow patterns.
- `temporal_config.py`: defines live window, history, macro window, and forecast constants.
- `preprocessing_params.json`: stored preprocessing values.
- `label_maps/`: dataset label mapping tables.

## 7. ML Models

## 7.1 TGNE-TA / BiTA

The BiTA implementation is under `bita/`. It is a temporal graph network with memory and temporal message aggregation.

Core mechanism:

```text
observed host/edge event
    -> temporal message
    -> message aggregation
    -> node memory update
    -> temporal neighborhood/attention embedding
    -> host latent H in R^12
```

Important modules:

- `bita/model/tgn.py`: base temporal graph network.
- `bita/model/extentedtgn.py`: extended TGN variant used by the project.
- `bita/model/time_encoding.py`: encodes event time gaps.
- `bita/model/temporal_attention.py`: temporal attention operations.
- `bita/modules/memory.py`: node memory storage.
- `bita/modules/memory_updater.py`: recurrent memory updates.
- `bita/modules/message_aggregator.py`: message aggregation, including the BiGRU-Transformer aggregator.
- `bita/modules/message_function.py`: event-to-message transformations.
- `bita/modules/embedding_module.py`: graph and temporal embedding variants.
- `bita/utils/utils.py`: neighbor lookup, MLP, merge, and training utilities.

The live adapter loads `bita/saved_models/bita_bigru_transformer-warden_alerts.pth` and uses the resulting host embeddings as the shared representation for Branch A and Branch B.

### 7.2 Branch A multitask LSTM

`branch_a_gnn_lstm/lstm_multitask.py` defines `MultiTaskLSTM`.

Input for each host time step:

```text
[12 TGNE latent features, 15 observable temporal features] = 27 values
```

The model contains:

- A two-layer LSTM for temporal context.
- Temporal self-attention.
- A sigmoid risk head.
- A 14-class technique head.
- A four-class gradation/stage head.
- Focal loss support for imbalanced technique labels.
- Learned uncertainty weighting for multitask losses.

The live adapter uses Branch A risk and technique predictions. The model's additional gradation output exists for the multitask contract but is not necessarily surfaced as an independent dashboard field.

Supporting files:

- `attention.py`: temporal attention layer.
- `sequence_dataset.py`: host sequence samples, technique vocabulary, and gradation labels.
- `train_branch_a.py`: training entrypoint and TGNE checkpoint loading.
- `saved_models/branch_a/branch_a_lstm.pt`: trained Branch A checkpoint.
- `saved_models/branch_a/branch_a.manifest.json`: checkpoint metadata.

### 7.3 Branch B host world dynamics transformer

`branch_b_world_model/rollout_encoder_decoder.py` defines `HostWorldDynamicsTransformer`.

Its purpose is not direct classification. It predicts how a host's latent state may evolve:

```text
recent H_t history
    -> project 12-D states into 64-D transformer space
    -> causal temporal attention
    -> continuous time encoding
    -> residual latent delta prediction
    -> autoregressive future H states
```

The model includes a multi-host interaction layer. Active communication edges allow predicted host states to exchange messages, which helps represent distributed behavior and possible lateral propagation.

The live horizon is eight steps. The model applies stabilization to prevent long autoregressive rollouts from diverging.

`branch_b_world_model/infiltration_head.py` defines `InfiltrationRiskHead`. It maps each predicted future latent through an MLP, produces one risk value per future step, and aggregates the trajectory into a cumulative future risk.

Supporting files:

- `train_branch_b.py`: trains the rollout transformer and infiltration head.
- `saved_models/branch_b/host_wdt.pt`: trained Branch B checkpoint.
- `saved_models/branch_b/branch_b.manifest.json`: checkpoint metadata.

Branch B imports `ContinuousTimeEncoding` from `world_model/models/time_encoding.py`. This shared import is why the otherwise offline `world_model/` package is still partially coupled to live Branch B.

### 7.4 DeepOP / CWA forecast decoder

`deepop_decoder/forecast_decoder.py` defines `DeepOPForecastDecoder`.

The decoder conditions future technique prediction on:

- Branch B future latent states.
- The current observed technique token.
- Causal temporal context.
- Optional prototype and future-state gates.

`deepop_decoder/cwa.py` implements Causal Window Attention. The causal mask prevents a future token from attending to later tokens during autoregressive decoding.

`deepop_decoder/joint_vocab.py` defines the compact network-observable technique/stage vocabulary, including benign, reconnaissance, initial access, credential access, command and control, exfiltration, impact, and BOS/EOS/PAD tokens.

Supporting files:

- `train_cwa_decoder.py`: primary decoder training.
- `train_balanced_cwa.py`: balanced-data decoder experiment.
- `saved_models/deepop/cwa_forecast_decoder.pt`: decoder checkpoint.
- `saved_models/deepop/deepop.manifest.json`: checkpoint metadata.

## 8. Model Adapter and Prediction Contract

`control_backend/model_adapter.py` is the live orchestration layer. It:

1. Loads and validates the four promoted checkpoints.
2. Builds host trajectories from flow windows.
3. Selects a target host based on observed activity and site configuration.
4. Runs TGNE-TA to produce a 12-D latent.
5. Runs Branch A for current risk and technique probabilities.
6. Runs Branch B for future latent states and future infiltration risk.
7. Runs DeepOP for future ATT&CK technique forecasts.
8. Computes explainability data.
9. Binds the prediction to `focus_ips`, `focus_edges`, and the selected target IP.
10. Emits a Pydantic `PredictionEvent` for the dashboard.

The prediction event can contain:

- Current risk.
- Maximum future risk.
- Forecast points.
- Current technique/stage information.
- Future technique tokens.
- Early-warning state.
- Latency metrics.
- Model metadata.
- Explainability groups.
- Focus IPs and edges.

The adapter explicitly prioritizes repository modules and removes ambient modules named `model` to avoid accidentally importing an unrelated external or retired implementation.

## 9. Explainability

The live adapter computes input saliency for the Branch A risk output. Gradients are grouped into meaningful feature families:

- TGNE latent.
- Connectivity.
- Volume.
- Timing.
- General.

`explainability/unified_explanation.py` defines a broader explanation object and asynchronous queue for combining trajectory and campaign explanations. It is a reusable analytical layer, while the live adapter currently uses its local saliency implementation for low-latency dashboard output.

The offline `world_model/explainability/` modules provide a different set of tools:

- Attention extraction.
- Integrated gradients.
- Trajectory visualization.

Those tools target the offline research WDT, not the promoted live adapter.

## 10. Correlation and Campaign Analysis

The `correlation/` package is downstream analytical infrastructure:

- `trajectory_assembler.py`: builds host attack trajectories from temporal entries.
- `causal_edge_scorer.py`: scores likely causal transitions between events.
- `graph_compaction.py`: reduces detailed alert graphs while preserving important evidence.
- `campaign_merge.py`: merges related host trajectories into campaigns.

These modules are useful for future campaign-level SOC correlation, but the current telemetry-to-dashboard path primarily emits per-window topology and prediction events. They are not the source of live model inference.

## 11. Containerlab Lab Mode

### 11.1 Topology

`containerlab/enterprise.clab.yml` defines a 15-node enterprise range:

- Routers: `router-edge`, `router-firewall`, `router-core`.
- Workstations: `ws-office`, `ws-web`, `ws-file`, `ws-app`.
- Servers: `srv-dns`, `srv-id`, `srv-file`, `srv-app`, `srv-db`.
- DMZ: `dmz-web`.
- External lab traffic source: `attacker`.
- Passive sensor: `sensor`.

The attacker node is lab infrastructure only. It is not a permanent node in the production discovery graph.

### 11.2 Network setup

`containerlab/configs/setup_networking.sh` configures:

- Interface addresses and routes.
- User, server, and DMZ segments.
- Edge firewall policy.
- Workstation-to-database restrictions.
- `tc mirred` traffic mirroring from the core router to the sensor.
- Sensor promiscuous mode.

The mirror is passive. Sensor failure should not become an inline forwarding failure.

### 11.3 Services and workloads

`nodes/base/Dockerfile` creates the common node image.

Services under `nodes/services/`:

- `dns_server.py`: internal DNS behavior.
- `identity_server.py`: identity/authentication service.
- `file_server.py`: file service.
- `app_server.py`: application/API service.
- `db_server.py`: database service.
- `web_dmz.py`: public DMZ web service.

Workloads under `workloads/`:

- `user_workload.py`: normal office, web-heavy, file-heavy, and application traffic.
- `attacker_scenario.py`: controlled lab-only reconnaissance, probing, exploit-like requests, and lateral probing.

The dashboard's external attack controls arm monitoring state; they do not make synthetic attack traffic part of the observed topology truth.

## 12. Site Configuration

`control_backend/site_config.py` loads YAML site profiles and provides:

- Site identifier.
- Enterprise and external CIDRs.
- Zone names.
- Sensor mode and interface.
- Optional sensor container.
- Topology TTL and node/edge caps.
- Assets of interest.
- Lab-mode flag.

Profiles:

- `config/sites/containerlab-enterprise.yaml`: Lab Mode, Containerlab sensor, enterprise CIDRs, and lab assets.
- `config/sites/local-default.yaml`: non-lab Local SPAN profile.

`config/temporal_contract.json` records shared timing and shape expectations.

## 13. Commands and Event Transport

`control_backend/commands.py` implements an allowlisted command executor. Commands include:

- Build, deploy, start, stop, destroy, and healthcheck lab operations.
- Start and stop sensor.
- Start and stop ML inference.
- Start and stop normal workloads.
- Arm and disarm external attack monitoring.
- Mitigation actions.

Lab-only commands are rejected when the selected site has `lab_mode: false`.

`control_backend/event_broker.py`:

- Maintains WebSocket clients.
- Stores latest status, topology, and prediction events.
- Broadcasts command output and operational logs.
- Replays current state to newly connected clients.

`control_backend/schema.py` defines the Pydantic event contract shared by REST, WebSocket, model adapter, and frontend.

`control_backend/lab_config.py` centralizes repository paths, container names, and lab constants.

## 14. React Dashboard

The dashboard lives under `web_dashboard/` and is built with React and Vite.

### Application flow

`src/App.jsx`:

- Fetches initial status, site, and topology.
- Opens the WebSocket.
- Maintains prediction history and current events.
- Routes events to visual components.
- Sends commands and mitigation requests.

Important components:

- `components/Header.jsx`: system identity, connection, model, and runtime status.
- `components/ControlBar.jsx`: sensor, ML, Lab Mode, attack-arm, and mitigation controls.
- `components/NetworkTopology.jsx`: dynamic external/internal IP graph.
- `components/ThreatTrajectory.jsx`: risk and threat trajectory visualization.
- `components/ForecastCards.jsx`: future forecast summaries.
- `components/MitreStageCard.jsx`: ATT&CK stage and technique display.
- `components/ExplainabilityPanel.jsx`: saliency groups and explanations.
- `components/PredictionStoryBanner.jsx`: current prediction narrative.
- `components/ConsolePanel.jsx`: live command and telemetry logs.

The graph is discovery-driven. It does not generate a fixed 15-node topology or require an `attacker` node to render.

`src/App.css` and `src/index.css` contain layout, typography, state, graph, card, and responsive styles. `src/main.jsx` is the React bootstrap. `vite.config.js`, `package.json`, and `package-lock.json` define the frontend build and development environment.

## 15. Offline and Research World Model

The `world_model/` package is a separate research stack. It is not the promoted production model. It contains a larger latent-space world dynamics architecture, training utilities, baselines, evaluation, and explainability.

### Model modules

- `models/world_dynamics_transformer.py`: research WDT with causal attention, residual dynamics, and KV-cache support.
- `models/time_encoding.py`: continuous time-delta encoding; currently imported by live Branch B.
- `models/readout.py`: graph/latent readout.
- `models/state_decoder.py`: state reconstruction.
- `models/attack_decoder.py`: attack-stage decoding.
- `models/risk_head.py`: risk prediction.

### Data and training

- `data/ctu13_adapter.py`: CTU-13 preparation.
- `data/feature_schema.py`: research feature dimensions and labels.
- `data/graph_builder.py`: graph snapshot construction.
- `data/latent_dataset.py`: latent sequence datasets.
- `data/splits.py`: chronological and scenario splits.
- `training/train_world_model.py`: research WDT training.
- `training/train_downstream_heads.py`: downstream head training.
- `training/losses.py`: world-model and detection losses.
- `training/rollout.py`: rollout evaluation.
- `training/scheduled_sampling.py`: scheduled sampling.
- `training/checkpointing.py`: checkpoint persistence.

### Evaluation, baselines, and explanation

- `evaluation/run_evaluation.py`: full evaluation entrypoint.
- `evaluation/detection_metrics.py`: detection and attack-stage metrics.
- `evaluation/forecasting_metrics.py`: lead-time and forecast metrics.
- `evaluation/dynamics_metrics.py`: dynamics metrics.
- `evaluation/calibration.py`: calibration metrics and temperature scaling.
- `baselines/`: logistic, LSTM, and static-GCN comparisons.
- `scripts/generate_latent_dataset.py`: latent cache creation.
- `scripts/run_ablations.py`: ablation execution.
- `scripts/demo_explainability.py`: offline explanation demo.
- `DESIGN_DECISIONS.md`: research design rationale.
- `config/default_config.yaml`: research training configuration.

This stack should be treated as offline research unless a deliberate future integration replaces the current live adapter.

## 16. Scripts and Operations

| File | Role |
|---|---|
| `scripts/build.sh` | Build the common Podman node image |
| `scripts/deploy.sh` | Deploy Containerlab, configure routing/firewall/SPAN, start services and workloads |
| `scripts/destroy.sh` | Tear down the lab |
| `scripts/healthcheck.sh` | Validate node health, service reachability, segmentation, and SPAN visibility |
| `scripts/run_telemetry.sh` | Start capture in the lab sensor namespace or configured interface |
| `scripts/verify_telemetry.sh` | Verify that telemetry windows are produced |
| `scripts/run_control_panel.sh` | Start backend and optional Vite development server |
| `scripts/run_demo.sh` | Convenience dashboard launcher |
| `scripts/ensure_checkpoints.py` | Check checkpoint presence and loadability |
| `run_dashboard.py` | Primary dashboard/backend launcher |

The recommended focused test command is:

```bash
python -m pytest tests/test_site_config.py tests/test_topology_service.py tests/test_control_backend.py -q
```

The frontend build is run from `web_dashboard/` with the package scripts, normally `npm run build`.

## 17. Tests

- `tests/test_control_backend.py`: backend API, command allowlists, ML transitions, external monitoring state, and model smoke behavior.
- `tests/test_site_config.py`: site loading, CIDR classification, target selection, and site/topology API contracts.
- `tests/test_topology_service.py`: discovery graph creation, external/internal roles, TTL eviction, caps, API output, and Lab Mode gates.
- `bita/test_pipeline.py`: BiTA pipeline test placeholder.
- `world_model/models/test_wdt.py`: research WDT shape, masking, cache equivalence, and parameter tests.

The production tests focus on contracts at the backend boundary rather than packet capture against a live interface.

## 18. File-by-File Inventory

### Root files

- `README.md`: current product description, setup, architecture, checkpoints, and troubleshooting.
- `implementation.md`: local implementation plan and design decisions; intentionally ignored from Git.
- `run_dashboard.py`: primary application launcher.
- `.gitignore`: generated files, datasets, caches, secrets, frontend dependencies, and local planning files.
- `config/temporal_contract.json`: temporal contract metadata.
- `captures/live.pcap`: local capture artifact.
- `captures/states.jsonl`: local state-stream artifact.

### `control_backend/`

- `__init__.py`: package marker.
- `main.py`: FastAPI app and lifecycle/routes.
- `commands.py`: operational command allowlist and executor.
- `event_broker.py`: WebSocket event distribution and latest-event replay.
- `lab_config.py`: repository and lab constants.
- `model_adapter.py`: live model orchestration and prediction creation.
- `schema.py`: Pydantic API/event schemas.
- `site_config.py`: YAML profiles, CIDR classification, and target selection.
- `telemetry_service.py`: sensor process, JSONL tailing, topology and ML coordination.
- `topology_service.py`: discovery graph with TTL and caps.

### `telemetry/`

- `run_telemetry.py`: capture-only process and JSONL/PCAP recording.
- `capture/sniffer.py`: raw socket packet capture and decoding.
- `flow/flow_table.py`: bidirectional flow aggregation.
- `packet/pcap_engine.py`: packet behavioral features.
- `state/state_builder.py`: two-second windows and state records.

### `bita/`

- `README.md`: BiTA research and model notes.
- `train.py`: BiTA/TGN training entrypoint.
- `model/tgn.py`: base temporal graph network.
- `model/extentedtgn.py`: extended TGN.
- `model/time_encoding.py`: temporal encoding.
- `model/temporal_attention.py`: temporal attention.
- `modules/embedding_module.py`: graph embedding variants.
- `modules/memory.py`: node memory.
- `modules/memory_updater.py`: recurrent memory update.
- `modules/message_aggregator.py`: temporal message aggregation.
- `modules/message_function.py`: message construction.
- `utils/utils.py`: shared training and graph utilities.
- `Bipartite Graph Construction stage/BipartiteConstruction.py`: bipartite graph preparation.
- `Compute Time Statistics stage/ComputeTimeStatistics.py`: time-statistics preparation.
- `Data Loading and Preprocessing stage/DataLoading.py`: source loading/preprocessing.
- `Data Splitting stage/DataSplitting.py`: dataset splitting.
- `Node and Edge Feature Representation stage/NodeEdgeFeatureRepre.py`: graph feature representation.
- `Training stage/train_self_supervised.py`: self-supervised training.
- `evaluation/eval_edge_prediction_with_categories.py`: edge/category evaluation.
- `saved_models/bita_bigru_transformer-warden_alerts.pth`: live TGNE-TA checkpoint.
- `test_pipeline.py`: test placeholder.

### `branch_a_gnn_lstm/`

- `__init__.py`: package marker.
- `attention.py`: temporal self-attention.
- `lstm_multitask.py`: Branch A architecture and losses.
- `sequence_dataset.py`: host sequence dataset and vocabularies.
- `train_branch_a.py`: Branch A training and TGNE loading.

### `branch_b_world_model/`

- `__init__.py`: package exports.
- `rollout_encoder_decoder.py`: live latent rollout transformer.
- `infiltration_head.py`: per-step and cumulative future risk.
- `train_branch_b.py`: Branch B training.

### `deepop_decoder/`

- `__init__.py`: package exports.
- `cwa.py`: causal window attention.
- `forecast_decoder.py`: future technique decoder.
- `joint_vocab.py`: token vocabulary.
- `train_cwa_decoder.py`: primary decoder training.
- `train_balanced_cwa.py`: balanced decoder experiment.

### `data_unification/`

- `__init__.py`: package exports.
- `unified_schema.py`: canonical flow record.
- `cic2017_adapter.py`, `cic2018_adapter.py`, `ctu13_adapter.py`, `warden_adapter.py`: dataset adapters.
- `auth_log_adapter.py`: authentication event adapter.
- `label_resolver.py`: label and ATT&CK normalization.
- `flow_to_temporal_event.py`: temporal event conversion.
- `multi_dataset_stream.py`: host windows and trajectories.
- `behavioral_fingerprint.py`: flow behavior fingerprints.
- `tgne_features.py`: observable graph features.
- `split_manager.py`: scientific splits.
- `temporal_config.py`: timing constants.
- `preprocessing_params.json`: preprocessing parameters.
- `label_maps/`: dataset label maps.

### `correlation/` and `explainability/`

- `correlation/trajectory_assembler.py`: host trajectory assembly.
- `correlation/causal_edge_scorer.py`: causal edge scoring.
- `correlation/graph_compaction.py`: alert graph compaction.
- `correlation/campaign_merge.py`: campaign merging.
- `correlation/__init__.py`: package exports.
- `explainability/unified_explanation.py`: unified explanation object and queue.
- `explainability/__init__.py`: package exports.

### `world_model/`

The package contains the offline research WDT, data adapters, feature schemas, graph builders, latent datasets, training losses, checkpointing, evaluation metrics, baselines, ablation scripts, and explanation tools described in Section 15. Its only direct live dependency is the shared time encoder imported by Branch B.

### `containerlab/`, `nodes/`, and `workloads/`

- `containerlab/enterprise.clab.yml`: 15-node Lab Mode topology.
- `containerlab/configs/setup_networking.sh`: lab addressing, routes, firewall, bridges, and SPAN.
- `containerlab/clab-enterprise/`: generated Containerlab state and inventories; not authoritative source.
- `nodes/base/Dockerfile`: common node image.
- `nodes/services/*.py`: lab DNS, identity, file, app, database, and DMZ services.
- `workloads/user_workload.py`: normal traffic profiles.
- `workloads/attacker_scenario.py`: controlled lab-only external traffic generator.

### `web_dashboard/`

- `src/main.jsx`: React bootstrap.
- `src/App.jsx`: global state, REST bootstrap, WebSocket routing, and command calls.
- `src/components/*.jsx`: dashboard views and controls.
- `src/App.css`, `src/index.css`: frontend styling.
- `src/assets/`: frontend assets.
- `public/`: static icons and favicon.
- `package.json`, `package-lock.json`: dependencies and scripts.
- `vite.config.js`: Vite configuration.
- `README.md`: frontend notes.
- `dist/`: generated production bundle when built.

## 19. Known Technical Debt and Stale Material

### Retired V3.1 terminology

Some historical files still describe the removed 72-D V3.1 transformer. The current live adapter does not use that model. In particular, older architecture documentation and comments may mention:

- A root `model/` package.
- `CyberWorldModel V3.1-PCAP`.
- 72-D model input.
- A 15-step, 30-second inference history.
- A five-step forecast.
- `telemetry/inference/adapter.py`.

These references should not be used to configure the current runtime.

### Capture-state versus live-ML contract

The telemetry state builder still exposes a 72-feature state because it is useful for packet and flow recording and compatibility. The live model contract is flow-window to host trajectory to 12-D latent, not the old state-vector-to-transformer contract. This distinction should be preserved in future refactors.

### Shared time encoder coupling

Branch B imports `world_model/models/time_encoding.py`. Moving or re-exporting this small module under `branch_b_world_model/` would make the live/offline boundary clearer, but should be done only with tests for both import paths.

### Checkpoint verification

`scripts/ensure_checkpoints.py` verifies checkpoint presence and loadability. Its expected-key metadata should be reviewed against the current model attribute names so a future checkpoint mismatch cannot be hidden by incomplete key validation.

### Generated artifacts

`containerlab/clab-enterprise/`, `web_dashboard/dist/`, caches, PCAPs, JSONL captures, and temporary result files are generated or local artifacts. They should not be treated as authoritative source code.

## 20. Operational Summary

For a lab run:

```bash
python run_dashboard.py --site containerlab-enterprise
```

Then use the dashboard sequence:

```text
START NETWORK -> START SENSOR -> START ML
```

For a real or local mirror interface:

```bash
sudo -E python run_dashboard.py --site local-default --interface eth1
```

Then use:

```text
START SENSOR -> START ML
```

The lab topology is optional. The live product boundary is the passive sensor, the FastAPI control backend, the promoted Dual-Branch/DeepOP checkpoint stack, and the discovery-driven dashboard.