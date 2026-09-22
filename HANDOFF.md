# Handoff — repo map

What each part is for, and which parts are live. Last updated 2026-09-22.

## The model pipeline (four models, in order)

```
flows ──> TGNE encoder ──> Branch A   (risk / technique / stage, now)
                      └──> Branch B ──> DeepOP   (ATT&CK token sequence, next 10s)
```

| directory | what |
|---|---|
| `bita/` | **TGNE encoder.** Temporal graph net, 12-D host embeddings. Trained by `bita/train.py`. Everything downstream consumes its output. |
| `branch_a_gnn_lstm/` | **Branch A.** LSTM + attention, three heads: risk, ATT&CK technique, attack stage. |
| `branch_b_world_model/` | **Branch B.** World model: predicts the next 5 host embeddings from the last 15. |
| `deepop_decoder/` | **DeepOP.** Transformer decoder, emits (category, technique) tokens from Branch B's rollout. |

**Dependencies:** Branch A and Branch B are independent — both just consume TGNE. DeepOP depends on Branch B. Retraining TGNE invalidates all three; retraining Branch B forces a DeepOP retrain; retraining Branch A affects nothing else.

## Supporting code

| directory | what |
|---|---|
| `data_unification/` | Adapters for CIC-2017/2018, CTU-13, PCAP; the 12-D edge features; `TrajectoryStore` (columnar host trajectories). Where data correctness lives. |
| `scripts/` | Training entry points. `retrain_branch_a_live.py` and `retrain_future_models_live.py` are the two that matter. |
| `control_backend/` | Live serving. `model_adapter.py` composes all four models and is what the dashboard calls. |
| `correlation/` | Host-trajectory assembly for serving. `campaign_merge.py` is aspirational — see caveats. |
| `cyberworld_v4/` | The **temporal contract** (2s windows / 15 history / 5 forecast) plus a separate scientific-benchmark track. `config.py` is authoritative — never hardcode these numbers. |
| `tests/` | 686 tests. Run `python -m pytest tests/ -q` before any commit. |
| `web_dashboard/`, `telemetry/`, `workloads/`, `nodes/` | Demo dashboard, live capture, traffic generators, lab topology. |
| `world_model/` | A self-contained parallel track, not in the live pipeline. |

## Where things are

- **Checkpoints:** `saved_models/` — `branch_a/`, `branch_b/`, `deepop/`, plus the encoder at the top. These paths are what serving loads; a retrain that writes elsewhere is a no-op.
- **Reports:** `claude_latest_analysis/` — numbered investigation docs. `27_STATE_*.md` is the current state; `28_*.md` holds two open reviewer concerns.
- **Run logs:** `logs/`. **Scratch:** `.spill/` (gitignored; multi-GB, never commit).
- **Launcher:** `run.sh` starts a full retrain under systemd with memory caps.

## Git

| remote | role |
|---|---|
| **`origin`** (`Sanyam-Ahuja/cyber_network_predictor-…`) | **The working repo.** `testing-prod` tracks it. Commit and push everything here. |
| **`sih`** (`SIH-2026-SSSVBT/Cyber_World`) | **Team repo — parked.** Deliberately rolled back to `995ebe4` ("harden MVP") on 2026-09-22. **Do not push to it.** Pre-rollback state is preserved on `backup/*-before-rollback-20260922` branches. |

## Rules that are not obvious

1. **Always cap heavy jobs.** `systemd-run --user --scope -p MemoryMax=NG -p MemorySwapMax=0`. An uncapped job froze this machine twice — 22 GiB total.
2. **Never dilute the data.** No stride, no row caps, no subsampling. `data_unification/density.py` enforces it.
3. **Never `git add -A` unfiltered** — it swept 500 MB `.spill` blocks into history and made the repo unpushable (GitHub rejects >100 MB). Use `git add -A ':!saved_models'`.
4. **Don't edit model code while it trains.** The result gets thrown away.
5. **Quote macro F1, not accuracy.** The corpus is ~82.5% Benign, so accuracy cannot fail visibly.

## Caveats a newcomer will otherwise trip on

- **The horizon is 30 seconds of history, 10 seconds ahead**, with `use_memory=false`. Multi-day campaign reasoning is structurally impossible today. `correlation/campaign_merge.py` has no time-horizon logic at all.
- **The 12 edge features have never been ablated.** `dst_port_norm_65535` is a shortcut-leak risk. Harness exists (`CYBERWORLD_ABLATE_EDGE_FEATURES`); the sweep has not been run.
- **9 of 10 CIC-2018 days fabricate host IPs by row index**, so CSV-path Branch B / DeepOP results are pipeline validation, not science. The PCAP path fixes this and is unwired.
- **Recon has one carrier host; the held-out test split has 2 technique classes against 7 in training.**
