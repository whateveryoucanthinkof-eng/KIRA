# Dashboard integration — the K.I.R.A. console on our backend

The SOC console in `web_dashboard/` is the Cyber World (K.I.R.A.) frontend, wired to **this** repo's
backend (`control_backend/`) and models. This document says, panel by panel, where every number comes
from, and which panels are waiting on a model that does not exist yet.


## Two ways to run it

| Command | What it is | Data |
|---|---|---|
| `python run_dashboard.py` | **The console. Use this.** FastAPI backend + sensor + models + UI on `:8000`. | Real: your SPAN feed, Containerlab range or a replayed capture, scored by the loaded checkpoints. |
| `python run_dashboard.py --demo` | **Demo dashboard**, only for seeing the UI. A static bundle on `:8443`, no backend, no sensor, no models. | Sample data (`web_dashboard/src/api/mock.ts`). Nothing on it is a model output; the header says **DEMO DATA**. |

Common forms of the real console:

```bash
python run_dashboard.py --site containerlab-enterprise           # Containerlab Lab Mode
sudo -E python run_dashboard.py --site local-default --interface eth1   # local SPAN
python run_dashboard.py --site local-default --replay captures/live.pcap # replay a capture
```

Both modes build their UI on demand (`web_dashboard/dist/`, `web_dashboard/dist-demo/`) when the build is
missing or older than the sources. That needs `npm` on `PATH`, or a toolbox that has it
(`CYBERWORLD_NPM_TOOLBOX`, optional). `--rebuild` forces a build.

### What differs between the two builds

| | Real console | Demo dashboard |
|---|---|---|
| Attack buttons | **None.** Controls → "Watch for an attack" arms external-traffic monitoring (`start_attack`/`stop_attack`, display only, never a model input) and tracks what the model catches. Attacks are run against the lab from outside the console. | Five scenario buttons (recon, credential stuffing, exploit, C2 beaconing, flood). Each is a technique our heads can emit. |
| Containerlab controls | All of them: build, start/stop network, health check, verify telemetry, sensor, inference, workloads, reset. Lab-only commands are disabled when the site is not in lab mode. | Present, acting on the sample stream. |
| Sample data in the bundle | None. Live builds resolve `src/api/mock.ts` to the inert `src/api/mock.live.ts` (`vite.config.ts:liveBuildDropsDemoFixtures`). | All of it. |
| Fields no model produces yet | Shown as **Under development**. | Filled with sample values, so the full layout can be seen. |

## When there are no trained models

The backend starts anyway (`control_backend/model_adapter.py:get_model_adapter`). Sensor, topology,
flows and every lab control work; the header shows **Models not loaded** (the reason is in its tooltip and
on the Controls page); **Start Inference** is refused with the loader's reason. Drop trained checkpoints
into `saved_models/` and press Start Inference: the load is retried, no restart needed.

Today that is the state of the repo: there are no Branch A / Branch B / DeepOP checkpoints, and the only
encoder checkpoint is on feature schema 1.0.0 while the tree is 2.0.0.

## Where every panel's data comes from

Wire contract: `control_backend/schema.py` ↔ `web_dashboard/src/api/types.ts` (change both together).
Every `/ws` prediction passes through `web_dashboard/src/api/normalize.ts`, and the demo stream goes
through the same function.

| Panel / field | Source | Status |
|---|---|---|
| Risk, alert, alert level, threshold | Branch A risk head; threshold fitted on validation (`_adopt_risk_semantics`) | Real |
| Technique, tactic, stage probabilities | Branch A technique head (`TECHNIQUE_VOCAB`, 14 classes) | Real |
| Kill-chain lane (`tactic_lane`) | `control_backend/tactics.py`, via DeepOP's `consolidate_network_technique` | Real |
| Forecast risk per step | Branch B rollout → infiltration risk head, 5 × 30 s | Real |
| Forecast band | Split-conformal half-widths stored in the Branch B checkpoint (`forecast_band.py`) | Real when the checkpoint carries them; otherwise no band is drawn (never invented) |
| Forecast technique per step | DeepOP greedy decoding | Real |
| Early warning / lead time | First forecast step over the threshold (`model_adapter.py`) | Real |
| Alternate futures A/B/C | DeepOP: top-3 first tokens, each decoded greedily (`forecast_branches.py`); A *is* the served forecast | Real |
| ↳ hosts a branch passes through, packets, bytes, per-branch peak risk | — | **Under development**: forecasts are per host and Branch B rolls out one future |
| State vector (all 27 dims) + attribution | TGNE latent + 15 host attributes; Input × Gradient on Branch A (`_explain_full`) | Real |
| Top features, attribution groups | Same | Real |
| Flow table / flow evidence | Sensor's per-window flow export with TCP flag counts (`evidence.py`, `flow_table.py`) | Real |
| Host graph, Network page, 3D view | Discovery from observed flows (`topology_service.py`), zones/names from the site YAML | Real |
| Campaign graph | `correlation/` pipeline run live (`correlation_service.py`) | Real, **heuristic** (hand-set parameters, nothing learned) |
| Incidents | Model alerts grouped per host (`correlation_service.py`) | Real; status/assignee/notes are the analyst's, kept in the browser |
| Attention matrix (Forecast Stage → Attention) | TGNE `TemporalAttentionLayer` weights | **Under development**: computed every window, not yet read out |
| Kill-chain lanes Execution, Lateral Movement; ATT&CK cell T1021 | — | **Under development**: no head has these classes |
| Replay page | `/api/replay` (upload) and `/api/replay/samples` (files in `captures/`), scored on a private adapter, rules off | Real (needs models) |
| Throughput, packet "loss", latency, sensor drops | Measured by the sensor (`telemetry_service.py`, `capture_accounting.py`) | Real |
| Mitigation status | Recorded intent; flows a block should have stopped are counted, never hidden | Real (recorded, **not enforced**) |
| Rule-layer opinion | `advisory_rule_opinion`, only with `CYBERWORLD_ENABLE_RULES=1`; never changes `risk` | Real, off by default |

Model gaps behind the "Under development" marks are listed in README.md, "Status and limitations".

## What the console needs from the backend, and where it lives

| Need | Backend |
|---|---|
| Temporal contract for the timeline (window, steps, step seconds) | `/api/status` → `contract` (checkpoints' when loaded, `cyberworld_v4/config.py`'s when not). The UI never hardcodes it (`src/types/timeline.ts:setContract`). |
| Site geometry: what is internal, zones, asset names | `/api/site`. The UI never assumes `10.x` (`src/design/site.ts`). |
| Whether external monitoring is armed | `/api/status` and every status push → `attack_armed`. (On the bus `attack` means "the model is alerting".) |
| Model availability | `/api/status` → `model_loaded`, `model_error` |

## Follow-up: reading out TGNE attention

The served encoder (`graph_attention`, 2 heads) computes attention weights on every window and discards
them in `GraphAttentionEmbedding.aggregate`. Getting them to the console without editing model code:

1. a forward hook on `embedding_module.attention_models[0]` to capture the head-averaged weights
   `[n_hosts, n_neighbors]`;
2. the neighbour ids for the same call, from the window's `_WindowedNeighborFinder` (sampling is
   `most_recent`, so re-querying it is deterministic);
3. node id → IP from the extractor's `FlowToTemporalEventAdapter.ip_to_id` (stored when encoder memory
   is on, which is the default; per call otherwise);
4. collapse neighbour slots by host into the `AttentionMatrix` shape in `src/types/attention.ts`.

Per-cell saliency (Input × Gradient on the attention logit) is a separate, heavier step. Neither has
been done, because nothing can be verified against real weights until the encoder is retrained.

## Verifying a change

```bash
python -m pytest tests/test_dashboard_evidence.py tests/test_control_backend.py -q   # backend payloads
cd web_dashboard && npx tsc --noEmit -p . && npm run build && npm run build:demo     # both UIs
```
