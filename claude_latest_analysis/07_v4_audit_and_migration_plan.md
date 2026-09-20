# CyberWorld v4 — Audit and Migration Plan

Deliverables **A–J** required by `CyberWorld_Final_Architecture_Implementation_Prompt.md` §68 before
implementation. Every finding below was verified against the code at the commit this document lands on.

**Authoritative contract for v4** (spec §64; supersedes the live 5/8 contract):

```yaml
window_seconds: 2
history_steps: 15     # 30 s causal history
forecast_steps: 5     # 10 s horizon
```

> **This invalidates all four existing checkpoints**, which were trained at `history_steps=5,
> forecast_steps=8`. v4 requires retraining. That is a deliberate consequence of adopting §64, not an
> accident.

---

## A. Current architecture map — the *promoted* path

Only this path produced the shipped checkpoints. Everything else is dead or research-only.

```
CIC-2018 CSV  ─┐
CTU-13 .binetflow ─┤
               ├─► scripts/retrain_branch_a_live.py      (T=5,  2.0s)  ─► branch_a_lstm.pt
               └─► scripts/retrain_future_models_live.py (T=5, K=8)    ─► host_wdt.pt
                                                                        ─► cwa_forecast_decoder.pt
Warden IDEA CSV ─► bita/train.py ─► bita_bigru_transformer-warden_alerts.pth

LIVE:
SPAN ─► telemetry/ ─► flows ─► control_backend/model_adapter.py
        ─► TGNE (12-D) ⊕ host temporal attrs (15) = 27-D
        ─► Branch A (risk/technique/stage)
        ─► Branch B rollout K=8
        ─► DeepOP CWA decoder
        ─► heuristic override (model_adapter.py:350-362)   ← not model output
        ─► WebSocket ─► dashboard
```

**Dead / non-promoted:**

| Path | Status |
|---|---|
| `branch_a_gnn_lstm/train_branch_a.py` | data roots missing → trains on **zero records**, silently no-ops |
| `branch_b_world_model/train_branch_b.py` | same |
| `deepop_decoder/train_cwa_decoder.py`, `train_balanced_cwa.py` | same |
| `world_model/` (29 files) | cannot import — `world_model.models.*`, `world_model.data.*` never committed |
| `world_model/training/scheduled_sampling.py` | zero importers |
| `branch_a_gnn_lstm/technique_vocab.py` | dead; live vocab is `sequence_dataset.py:17-32` |
| `data_unification/host_attributes.py` | dead; advertises TCP-flag features the model never sees |
| `telemetry/packet/pcap_engine.py` | restored, still unwired |
| `MultiHostInteractionLayer`, `rollout_multi_host` | defined, never trained or called |

**Correction to the audit that prompted this document:** it identified "60 s training vs 2 s deployment"
as the single biggest problem, reading `train_branch_a.py` etc. Those scripts are dead. The shipped
checkpoints carry `window_seconds: 2.0`, and `retrain_future_models_live.py:46-47` trains `T=5, K=8` —
deployment-matched. The same applies to its "K=4 trained / K=8 deployed" claim. The real defect is that
the repo ships two trainer sets and the plausible-looking ones are broken.

---

## B. Current-vs-required discrepancy matrix

| # | Requirement (§) | Current | Required | Sev |
|---|---|---|---|---|
| 1 | Contract 2s/15/5 (§64) | 2s/5/8 live; 60s in dead scripts | single enforced contract | **CRIT** |
| 2 | Future targets (§6,7) | `target_snap = window_slice[-1]` — nowcast | `y_{t+1..t+5}` | **CRIT** |
| 3 | Risk semantics (§8,9) | `TACTIC_BASE_SEVERITY` lookup | P(current), hazard, cumulative, severity | **CRIT** |
| 4 | Multilabel ATT&CK (§13) | `technique_ids[0]` | `{0,1}^N` | **CRIT** |
| 5 | Unknown labels (§15) | `("Unknown", [], True if norm_key else False)` | explicit UNKNOWN, excluded | **CRIT** |
| 6 | Host identity (§18) | raw IP; CIC-2018 IPs **fabricated** 9/10 days | (dataset,scenario,capture,host) | **CRIT** |
| 7 | Stable IDs (§19) | Python `hash()` | SHA-256 | HIGH |
| 8 | Held-out test (§36) | `get_heldout_test_records()` never called | train/val/calib/test | **CRIT** |
| 9 | Conformal (§34) | hardcoded ±0.05; radii discarded | split conformal + ACI | **CRIT** |
| 10 | Calibration (§33,35) | temperature inside loss | post-hoc on calib split; ECE/Brier/NLL | HIGH |
| 11 | DeepOP input (§26) | oracle future + noise | WDT-predicted states | **CRIT** |
| 12 | Lead time (§30) | `forecast_steps × window` = horizon | `t_onset − t_alert` | HIGH |
| 13 | Rules engine (§41) | inside `predict_window` | outside research model | **CRIT** |
| 14 | Armed flag (§42) | feeds risk computation | never in benchmark | **CRIT** |
| 15 | Baselines (§39) | none | persistence/Markov/LR/GBDT | **CRIT** |
| 16 | Ablations (§40) | none | A0–A5 | HIGH |
| 17 | Seeds/CIs (§47,48) | single run, no CI | 5 seeds, group bootstrap | HIGH |
| 18 | Offline/live parity (§43) | two implementations | one processor + equality test | HIGH |
| 19 | Manifests (§45,46) | partial, contradictory | immutable, in-checkpoint | HIGH |
| 20 | Time gaps (§20) | consecutive snapshots | Δt stored, gaps split | HIGH |
| 21 | Multi-host (§24,25) | per-host only | claim-or-implement | MED |
| 22 | Explainability (§50) | 1-step, train mode | real input, eval mode | MED |
| 23 | Dashboard labels (§51) | synthetic shown as measured | labelled | MED |
| 24 | Deps | `pyarrow`/`pandas`/`sklearn`/`matplotlib` undeclared | declared | MED |

---

## C. Temporal-contract violations

| Location | Value | Action |
|---|---|---|
| `config/temporal_contract.json` | live 2s/5/8; macro 60s/5/4 | replace with v4 contract |
| `data_unification/temporal_config.py:21-25` | `DEFAULT_ROLLOUT_HORIZON_LIVE=8`, `DEFAULT_HISTORY_STEPS=5` | 5 / 15 |
| `train_branch_a.py:150`, `train_branch_b.py:92`, `train_cwa_decoder.py:106`, `train_balanced_cwa.py:31` | `window_size_sec=60.0` | delete scripts |
| `retrain_branch_a_live.py:112`, `retrain_future_models_live.py:106` | 2.0s, T=5, K=8 | migrate to 15/5 |
| `rollout_encoder_decoder.py:194` | `max_context_len=10` | ≥15 |
| `model_adapter.py:88-91` | `history_steps=5`, `forecast_steps=8` | from config |
| `forecast_decoder.py` | `pos_embed[:, :S, :]` | size to K=5 |
| `Predictions.tsx:188` | renders `+{i+1}m` for 2 s steps | render real seconds |

**No module may define these constants locally.** Single source: `cyberworld_v4/config.py`.

---

## D. Data / label leakage audit

**D1 — CIC-2018 fabricated identity (CRITICAL, affects shipped weights).**
Only `tue_20_csv.csv` (84 cols) carries `Src IP`/`Dst IP`. The other nine are 80 cols with none.
`cic2018_adapter.py:65,70` invents them:
```python
src_ips = np.array([f"192.168.10.{i % 250 + 1}" for i in range(len(chunk))])
dst_ips = np.array([f"172.16.0.{i % 100 + 1}"  for i in range(len(chunk))])
```
Source ports randomised at `:75`. A "host trajectory" is every 250th row. **The host graph is synthetic
for 9 of 10 training days.** Fix: source identity from the PCAPs (one file per host, IP in filename) or
restrict host-level work to `tue_20`.

**D2 — CTU-13 topology encodes the label (CRITICAL).**
`ctu13_adapter.py:62` — `dst_ip = "147.32.84.180" if is_attack else "147.32.80.1"`; ports from the
malware-family string at `:64`. The graph edge *is* the label. In `parse_parquet`, not the shipped path,
but live and referenced by `train_branch_a.py:48-50`.

**D3 — Unknown → attack.** `label_resolver.py:101`.

**D4 — Cross-dataset IP collision.** Trajectories group by raw IP; private ranges recur across datasets.

**D5 — Overlapping windows.** Stride-1 sequences; 35,993 "samples" are not independent. Bootstrap at
capture/host level.

**D6 — Row-prefix sampling. MEASURED: it makes forecasting vacuous.**

`--rows-per-file` is pandas `nrows` — a prefix, not a sample. The CSVs are time-ordered and attacks
occur in contiguous blocks, so a prefix is nearly single-label:

| File | first 60k rows |
|---|---|
| `fri_16_csv.csv` | **99.8% Impact** |
| `thu_1_csv.csv` | **100.0% Benign** |
| `fri_23_csv.csv` | 99.1% Benign |
| `thu_15_csv.csv` | 87.5% Impact |

Compounded with D1's round-robin fabricated IPs, every synthetic host's trajectory carries one label
for its entire length. A full v4 training run on 1.38M prefix records measured:

```
label churn (fraction where A_t+K != A_t) : 0.0000
persistence forecast PR-AUC by step        : [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Zero churn means the forecasting task has no content.** `A_{t+K}` is always `A_t`, so a model that
ignores the future entirely is perfect, and the 0.98 "forecast PR-AUC" that run produced is nowcasting
under a different name. It would have been reportable as a forecasting result by anyone not checking.

This is why `scripts/train_v4.py` refuses to present such a run as a result: it prints the churn figure
and a DEGENERATE TASK warning whenever persistence exceeds 0.99.

Fix applied: `--stride` samples every Nth row so the set spans the file's full time range. On
`fri_16_csv.csv` that moves the majority class from 99.8% to 63.9%.

**D7 — Armed-attack oracle.** `attack_active` reaches risk computation.

**Not leakage (checked):** TGNE neighbour lookup respects the cutoff timestamp. No future-neighbour
leak. The prompting audit was right to decline this accusation.

---

## E. Train/serve mismatch

| # | Train | Serve | Fix |
|---|---|---|---|
| E1 | DeepOP on observed future + noise | WDT-predicted states | train on WDT output |
| E2 | `stabilize_horizon=True` default, never overridden → `Δh *= 0.95^k` **during training** | same damping | disable in training; keep as inference option only |
| E3 | Offline `HostTrajectoryExtractor` | live `model_adapter._build_embedding` | one `CanonicalFlowProcessor` + equality test |
| E4 | Branch B free-running (no `detach`) | free-running | consistent — no action |
| E5 | TGNE pretrained on Warden bipartite | live shared IP namespace | retrain on production schema |
| E6 | Risk target = severity lookup | displayed risk = heuristic | both replaced |

**E7 — TGNE embedded on a different graph at train vs serve time. FOUND LATE, FIXED.**

Neither this audit nor the review that prompted it caught this; it surfaced only when the spec-43
parity check was actually built and run.

TGNE is a *graph* encoder — it aggregates over a host's neighbourhood, so the graph handed to it
determines the embedding. The two paths handed it different graphs:

| | Graph passed |
|---|---|
| Offline trainers | `extract_trajectories(all_records)` — the full window |
| Live adapter | `_build_embedding(target, host_flows)` — only edges touching the target |

Measured on a replayed window holding 16 flows of which 2 touched the target host, the two embeddings
differed by **1.5e-2 per dimension**, and the gap grows with cross-host traffic. The model was fitted on
full-graph embeddings and served subgraph ones.

Fixed: the live path now passes the full window to the encoder while the temporal *attributes* stay
host-scoped, which is correct since those are per-host aggregates. `scripts/verify_offline_live_parity.py`
now reports **0.000e+00** deviation.

The lesson is the spec's own: parity between offline and live has to be *asserted by a test*, not
inferred from the fact that both call the same class. Both paths did call the same extractor — they just
fed it different data.

---

E2 is the subtle one: Branch B is trained with its own deltas damped toward zero, then validated against
a **persistence** baseline — biased toward the baseline it is measured against.

---

## F. Files requiring modification

**Contract:** `config/temporal_contract.json`, `data_unification/temporal_config.py`, `model_contract.py`
**Data:** `cic2018_adapter.py` (D1), `ctu13_adapter.py` (D2), `label_resolver.py` (D3),
`multi_dataset_stream.py` (identity, Δt, severity), `unified_schema.py` (namespaced id, flags, IAT),
`split_manager.py` (calibration split, env path), `flow_to_temporal_event.py`
**Targets:** `sequence_dataset.py` (future targets, multilabel)
**Models:** `lstm_multitask.py` (heads, loss, calibration), `rollout_encoder_decoder.py` (damping,
context, NLL head), `forecast_decoder.py` (K=5, pos_embed), `infiltration_head.py`
**Serving:** `model_adapter.py` (remove override, explainability, contract), `telemetry_service.py`
**Frontend:** `adapter.ts`, `types.ts`, `Overview.tsx`, `Predictions.tsx`
**Docs:** `docs/ARCHITECTURE.md` (retired V3.1 content), `README.md`, `TECHNICAL_REVIEW.md`
**Packaging:** `requirements.txt`

## G. Files safe to delete

> **Corrected after verification.** The first version of this list was wrong. It reasoned "these
> trainers are dead, therefore the files are dead" — but those files also hold shared
> dataset-building code that live and working code imports. Checked with an explicit importer scan
> rather than assumed:

| File | Importers | Verdict |
|---|---|---|
| `branch_a_gnn_lstm/train_branch_a.py` | `control_backend/model_adapter.py:123` (`build_or_load_tgne_ta`) — **live path** | **KEEP** |
| `branch_b_world_model/train_branch_b.py` | `scripts/retrain_future_models_live.py:21` (`HostRolloutDataset`, `create_rollout_samples`) — the trainer that produced the shipped weights | **KEEP** |
| `deepop_decoder/train_cwa_decoder.py` | `scripts/retrain_future_models_live.py:27` (`CWASequenceDataset`, `create_cwa_training_samples`) | **KEEP** |
| `data_unification/host_attributes.py` | `data_unification/tgne_features.py:23` | **KEEP** (still misleading; fix its contents, do not delete) |
| `deepop_decoder/train_balanced_cwa.py` | none | **DELETED** |
| `branch_a_gnn_lstm/technique_vocab.py` | none | **DELETED** |

Only two of the six were genuinely orphaned. For the four that stay, the defect was never the file —
it was the hardcoded `window_size_sec=60.0`, which is now bound to `get_contract().window_seconds` so
these trainers can reproduce what they claim to produce.

**Lesson worth keeping:** "this code path is dead" does not imply "this file is dead." A module can be
unreachable as a script and still be a library for something that is very much alive.

`world_model/` is **kept** — it is the only code aimed at the PS benchmark deliverable, and its
`evaluation/` metrics are salvageable. It must not be cited as runnable until `models/` and `data/`
exist.

## H. Migration plan (dependency order)

```
P1  cyberworld_v4/{config,contract,identity}.py        foundation, no deps
P2  targets.py, splits.py                              needs P1
P3  metrics/, baselines/, conformal.py                 needs P1-P2
P4  adapter + dataset-builder fixes (D1-D7, E3)        needs P1-P2
P5  model changes (heads, K=5, damping, DeepOP)        needs P1-P4
P6  training + calibration + manifests                 needs P1-P5
P7  serving: remove override, wire explainability      needs P5
P8  frontend + docs                                    needs P7
```

P3 is deliberately early: baselines and metrics existing *before* the models change means every
subsequent change is measured rather than asserted.

## I. Evaluation implementation plan

Splits: chronological + scenario-held-out, with a **calibration** split distinct from validation; test
untouched until the model is frozen.

Detection: PR-AUC, ROC-AUC, precision, recall, F1, false alarms/hour.
Forecasting: PR-AUC@k, Brier@k, NLL@k for k=1..5, plus the horizon-degradation curve.
Calibration: ECE, Brier, NLL, reliability — before and after post-hoc fitting.
Early warning: `L = t_onset − t_first_valid_alert`, median/mean/p10/p90, recall at 2/4/6/8/10 s.
Techniques: micro/macro-F1, mAP, P@1, P@3, R@3 — split by seen/unseen family and dataset.
Baselines: persistence, last-label, first-order Markov, logistic regression, GBDT, plain LSTM.
Ablations: A0 features+GBDT → A5 full+rules.
Statistics: 5 seeds (42, 123, 2024, 3407, 9001); bootstrap CIs at capture/host level, never over
overlapping windows.

## J. Reproducibility plan

Immutable manifest per run — experiment id, git commit, dataset manifest hash, seed, full config,
dependency versions — embedded in the checkpoint and written to `results/experiments/`.
Contract validated at load; refuse to serve a checkpoint whose contract differs from runtime.
Declare `pandas`, `scikit-learn`, `pyarrow`, `matplotlib`. Move `CIC2018_DIR` to an env var.
Deterministic offline/live parity test asserting identical features from identical flows.
