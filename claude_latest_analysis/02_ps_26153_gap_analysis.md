# SIH 26153 (NTRO) — Requirement-by-Requirement Gap Analysis

**Date:** 2026-09-19
**Repo:** `cyber_network_predictor`, branch `testing-prod`
**PS:** "AI based Network Attack Forecasting from Network Traffic Data" (`ps.md`)
**Status:** All findings marked *verified* were independently re-checked. One claim toned down (see §F).

---

## Headline

The modelling core is real and better than most SIH entries. The **evidence layer** and the **demo layer** are where this loses.

Three things a judge can find without reading much code would be fatal:
1. The dashboard's explainability panel does not show the model's explainability.
2. The headline risk number is a hand-written heuristic.
3. There is not a single F1 / precision / recall / FPR number anywhere in the repo.

Tests pass 27/27 but they are contract/infra tests only — **zero ML-accuracy assertions**.

---

## A. Verdict table

| # | PS requirement | Verdict |
|---|---|---|
| 1a | Flow-level features (NetFlow/IPFIX) | **PARTIAL** |
| 1b | Packet-level features (PCAP-derived) | **MISSING** |
| 1c | "Combination of both levels is required" | **MISSING** |
| 2 | Learn P(S_t+1 \| S_t), not static classification | **PARTIAL** |
| 3 | K-step forward simulation → probability timeline | **MET** |
| 4 | MITRE ATT&CK stage mapping (5 named phases) | **PARTIAL** |
| 5 | Explainability (attention / SHAP / attribution) | **PARTIAL** (backend) / **MISSING** (demo) |
| 6 | Generalise to unseen attack patterns | **MISSING** (unproven) |
| 7 | Demo accepting PCAP **or CSV**, fully offline | **PARTIAL** |
| 8 | Benchmark F1/precision/recall/FPR vs logistic regression | **MISSING** |
| 9 | Training scripts + weights + reproducible config | **PARTIAL** |
| 10a | README setup | **MET** (app) / **PARTIAL** (PS) |
| 10b | Architecture doc ≤2 pages | **MISSING** (wrong subject, 2× length) |
| 10c | Demo video ≤2 min | **MISSING** |
| 10d | Technical presentation ≤5 slides | **MISSING** |
| — | State as feature vector / graph | **MET** |
| — | Sequence model / GNN / latent state model | **MET** |

---

## B. The three critical findings

### B1. The explainability panel is disconnected — *verified*

Backend computes real Input×Gradient attributions (`model_adapter.py:193-251`) and ships them on the wire (`schema.py:111`, `model_adapter.py:511`).

`grep -rn "explainability|top_features|mitre_tactic|predicted_stage" web_dashboard/src/` returns **nothing**.

The panel renders `signals`, hardcoded at `web_dashboard/src/api/adapter.ts:243-247`:

```ts
signals: [
  { name: "Hazard Score",    weight: p.prediction.hazard_score    || 0, direction: "positive" },
  { name: "Forecast Error",  weight: p.prediction.forecast_error  || 0, direction: "negative" },
  { name: "Max Future Risk", weight: p.prediction.max_future_risk || 0, direction: "positive" }
]
```

Three risk scores relabelled as features and drawn as percentage bars. `forecast_error` is never set by `predict_window`, so that bar is **permanently zero**.

`mitre_tactic`, `mitre_technique`, `predicted_stage`, `stage_probabilities` are all produced and **never rendered**. `ForecastPoint` in `types.ts:51-57` has no stage field — so PS's "attack stage annotations" is unmet in the UI too.

**Fix (~40 lines):** map `p.explainability.top_features` / `.groups` into `signals`; add `mitre_tactic` / `predicted_stage` to `types.ts` and render. The data is already on the wire.

### B2. Heuristic override on the live path — *verified verbatim*

`control_backend/model_adapter.py:350-362`:

```python
if ext_count > 0 or attack_active:
    threat_boost = min(0.65, (max(1, ext_count) / 75.0) * 0.50 + 0.25)
    obs_risk = min(0.96, max(raw_risk, 0.40) + threat_boost)
    ports_seen = {getattr(r, "dst_port", 0) for r in ext_flows}
    if len(ports_seen) >= 5:      obs_technique = "PortScan"
    elif any(... in (80, 443, 8080) ...): obs_technique = "WebAttack"
    else:                         obs_technique = "Exploit"
else:
    obs_risk = max(0.05, raw_risk * 0.4)
```

Whenever any external flow is present — i.e. during every attack demo — displayed risk is `max(model_output, 0.40) + f(external_flow_count)`. The model's `raw_risk` is discarded when below 0.40, multiplied by 0.4 otherwise.

The override strings `"PortScan"` / `"WebAttack"` / `"Exploit"` are **not in `TECHNIQUE_VOCAB`** and not keys in `TECHNIQUE_TO_MITRE`, so `:430` falls back to `("Unknown", …)` and `:390-393` to `"Recon"`. That rule-based label is then fed to DeepOP as the observed token (`:394-401`) — the forecast is conditioned on the heuristic rather than the classifier.

Branch B's rollout, the DeepOP forecast and per-step future risks *are* genuinely model-driven. Branch A's observed risk and technique — the two most visible numbers — are not.

**Fix (~1 h):** delete these 16 lines.

### B3. No detection metrics exist — *verified*

`results/training_metrics.csv` in full:

```
epoch,train_loss,val_ap,val_auc,val_accuracy,val_mrr,inductive_val_ap,inductive_val_auc,inductive_val_accuracy,epoch_time
0,4.635,0.6021,0.6131,0.0,0.0246,0.4947,0.4850,0.0,2.004
```

One epoch. Accuracy 0.0. Inductive AUC 0.485 — **below chance**. This is the BiTA/TGNE link-prediction curve and it is the sole quantitative evidence in the repository.

- **There is no logistic regression.** `world_model/baselines/logistic_regression.py:15` is a `LinearDynamicsBaseline` (`nn.Linear` next-latent regressor) — wrong model class, wrong task, wrong metrics.
- `world_model/baselines/run_baseline_comparison.py` cannot import (`world_model/data/` and `world_model/models/` never committed). Verified: `ModuleNotFoundError: No module named 'world_model.data'`.
- The metric functions are fine and salvageable: `world_model/evaluation/detection_metrics.py:25` (precision/recall/F1/AUROC, `fpr_at_95_tpr` at `:74`) and `forecasting_metrics.py:14` (mean/median lead time). They depend only on two missing constants (`ATTACK_STAGE_NAMES`, `N_ATTACK_STAGES`).

**Fix (~1 d):** standalone `scripts/run_benchmark.py` — do **not** revive `world_model/`. Fit `sklearn.linear_model.LogisticRegression` on the **same 27-D features** (that phrase in `ps.md:50` matters), score Branch A on the identical split, emit to `results/benchmark.json` + README table.

---

## C. Requirement detail

### 1a. Flow-level features — PARTIAL

PS demands TCP flag bitmask and IAT stats. Neither reaches a model.
- 12-D edge features (`tgne_features.py:28-41`): bytes, packets, duration, rates, is_tcp/udp/icmp, dst_port_norm, asymmetry. **No flags, no IAT.**
- 15 host attrs (`multi_dataset_stream.py:190-204`): counts, bytes, packets, unique_peers, unique_dst_ports, tcp/udp ratio, avg duration, rates, density. **No flags, no IAT.**

Two aggravating facts:
- **The schema file lies.** `data_unification/host_attributes.py:6-22` declares attrs including `tcp_flags_syn/ack/fin/rst`, and `tgne_features.py:44` publishes it as the "Authoritative 15-D Temporal Host Attribute Names". It bears no relation to what the extractor produces, and is imported by one file and used by zero code paths.
- **Rich features computed then discarded.** `telemetry/flow/flow_table.py:13-26` computes 42 window features including `syn_count, ack_count, rst_count, flow_iat_mean/std/max, dst_port_entropy, max_pair_sequential_port_score` — nearly the whole PS list. `state_builder` consumes exactly one scalar (`active_flows_count`); the other 41 are dropped.
- Structural bottleneck: `UnifiedFlowRecord` (`unified_schema.py:36-54`) has no flag or IAT field, so even CIC-IDS CSVs (which carry both) get stripped at the adapter boundary.

**Fix:** add `tcp_flags` + `iat_*` to `UnifiedFlowRecord`, populate in `snapshot_flows()` (already tracked on `FlowRecord`, `flow_table.py:63-68`) and the adapters, append 6-8 to the attr vector, 15→~22, retrain.

### 1b/1c. Packet-level features — MISSING

PS `ps.md:22` names six; `ps.md:24` says **"The combination of both levels is required"** — the only sentence in the PS using the word "required", and the requirement the repo is furthest from. See `01_training_data_pcap_audit.md` for full detail.

`pcap_engine.py` (restored, commit `0a01138`) covers five of six — **IP fragment flags are not computed**.

### 2. Learn P(S_t+1 | S_t) — PARTIAL

Genuinely present: `HostWorldDynamicsTransformer.forward_1step` (`rollout_encoder_decoder.py:158-186`) predicts residual delta `h_next = h_seq[:,-1,:] + delta_h` under a causal mask; `rollout()` (`:188-236`) iterates autoregressively. Training is next-state regression with discounted multi-horizon MSE (`train_branch_b.py:139-149`), and validation compares against a **persistence baseline** (`:173-174`) — a good sign.

Caveats:
- It is **deterministic**, not a distribution. PS asks for "the probability distribution over the next network state". No variance head, no sampling, no NLL. `rollout_with_uncertainty` (`:238`) uses **hardcoded** empirical radii `[0.01890, 0.03661, 0.05327, 0.06904]` — a confidence band, not a learned P(S_t+1|S_t).
- `stabilize_horizon` damping (`:228-230`, `delta_h *= 0.95**k`) shrinks every step past k=0 toward "no change" — a hand-tuned stabiliser that flattens long-horizon rollouts by construction.
- Branch A's risk **target** is a lookup keyed on the ground-truth tactic (`multi_dataset_stream.py:290-311`: `TACTIC_BASE_SEVERITY`). So Branch A's "risk regression" is a relabelled attack classifier for the *current* window. **Branch B is the dynamics model; Branch A is not** — despite being what the dashboard headlines.

**Fix (~15 lines):** add a log-variance head and train with Gaussian NLL. Converts a deterministic predictor into a literal P(S_t+1|S_t) and enables calibration reporting.

### 3. K-step forward simulation — MET

`model_adapter.py:374-388` → `risk_head.forward_trajectory(h_future)` → `ForecastPoint(horizon_seconds=(i+1)*window_seconds, risk=…)` at `:442-452`.

**But the horizon is 16 seconds** — *verified*: `LIVE_WINDOW_SIZE_SEC = 2.0`, `DEFAULT_ROLLOUT_HORIZON_LIVE = 8` (`temporal_config.py:21,24`).

**And the UI mislabels it by 30×** — *verified*: `web_dashboard/src/pages/Predictions.tsx:188` renders `+{i + 1}m`, so the table reads "+1m … +8m". A `macro_batch` mode (60 s × 4 = 240 s) exists in `config/temporal_contract.json` with no checkpoint trained for it.

**Fix:** render `f.horizon_seconds`; train a `macro_batch` checkpoint for the offline/PCAP mode.

### 4. MITRE ATT&CK stage mapping — PARTIAL

The live vocab is `branch_a_gnn_lstm/sequence_dataset.py:17-32` — **not** `technique_vocab.py`, which is dead code imported by nothing.

14 T-codes: `Benign, T1046, T1595, T1110, T1190, T1189, T1071, T1071.001, T1568.001, T1204, T1005, T1498, T1498.001, T1020`.

**There is no Lateral Movement technique — no T1021, no T1570** *(verified)*.

The 4 stage classes (`sequence_dataset.py:35-45`) — *verified*:
```
0=Benign, 1=Recon, 2=InitialAccess/Execution,
3=C2 / LateralMovement / Exfiltration / Impact
```
**Three of the PS's five named phases are collapsed into one class**, indistinguishable by construction.

DeepOP is worse: `joint_vocab.py:20-28` has 7 tokens, no Lateral Movement, and `consolidate_network_technique` (`:31-74`) has no LateralMovement branch — such a record falls through to the fallback at `:74` and is **labelled Benign**.

Compounding: zero Lateral Movement rows in `data_unification/label_maps/*.csv`. Meanwhile the demo's own attack script has a `LATERAL_PIVOT_STORM` stage (`workloads/attacker_scenario.py:629-703`) written "so model lead-time can be evaluated" — the demo stages a phase the model provably cannot name.

Related: `multi_dataset_stream.py:334` injects `T1059`, which is not in `TECHNIQUE_VOCAB`, so `sequence_dataset.py:122` silently maps it to index 0 = **Benign**.

**Fix:** split gradation head 4→6; add `T1021` to the vocab and a matching branch in `consolidate_network_technique`; add LateralMovement rows to the CIC map; delete the dead `technique_vocab.py`.

### 5. Explainability — backend PARTIAL, demo MISSING

Beyond B1, the backend implementation has three defects:
- **It explains a different input than the prediction.** `_explain` builds `x` as `[1, 1, 27]` (`:194-199`) while the scored input is `[1, 5, 27]` (`:317`). On a temporal model that is not an explanation of the prediction.
- **It runs with dropout on.** `:202` calls `self.branch_a.train()` (a cuDNN-RNN-backward workaround) with `dropout=0.2` active — attributions are stochastic and irreproducible.
- **It explains the wrong number** — attributes `raw_risk`, while the operator sees the heuristically overridden `obs_risk`.
- **Attention weights are computed and never surfaced.** `lstm_multitask.py:188` returns `attention_weights`; only `explainability/unified_explanation.py:137,157` consumes them, and that module is not on the live path. The PS's first-named mechanism is unused.
- No SHAP anywhere.

### 6. Generalise to unseen attacks — MISSING

`split_manager.py:13-17` documents a genuinely good held-out design (CIC-2017 Tue/Thu, CIC-2018 Infiltration/SQLi/XSS, Warden 16-17 Mar, CTU-13 late offset — cross-year, cross-scenario). `get_heldout_test_records()` is defined at `:116` and **called from nowhere**. No held-out evaluation, no zero-shot result, no number.

**Fix (~80 lines):** one script using `get_heldout_test_records()`, dumping per-family F1/precision/recall/FPR. Discharges most of requirement 8 simultaneously.

### 7. Demo: PCAP or CSV, offline — PARTIAL

- **Offline: clean.** No `http://`, `requests`, cloud SDK or CDN in `control_backend/` or `web_dashboard/src/`.
- **PCAP: yes, CLI-only.** `run_dashboard.py:60 --replay`, `run_telemetry.py:116`. *(Note: this path was silently broken — no window ever closed — and was fixed in commit `7b297a1`.)*
- **CSV: absent.** No upload endpoint, no `UploadFile`, no file input in the frontend. The CIC/CTU CSV adapters exist but are reachable only from training scripts. PS names CSV first at `ps.md:45`.
- Also missing per `ps.md:49`: "flagged flows" and "attack stage annotations" in the UI.

**Fix (~60 lines):** `POST /api/replay` taking `.pcap` or `.csv`; for CSV route through `CIC2018Adapter.parse_file()` → `UnifiedFlowRecord` → the same `predict_window` loop.

### 9. Reproducibility — PARTIAL

Good: all four checkpoints committed (~1.3 MB, explicitly re-included in `.gitignore:98-111`); per-model manifests with architecture/split/optimiser/lr/seed; `scripts/ensure_checkpoints.py` verifies keys.

Broken:
- **Training scripts do not reproduce the shipped checkpoints.** Manifests declare `window_size_sec: 2.0`, but `train_branch_a.py:150`, `train_branch_b.py:92`, `train_cwa_decoder.py:106`, `train_balanced_cwa.py:31` all construct `HostTrajectoryExtractor(window_size_sec=60.0)`. Only the undocumented `scripts/retrain_*_live.py` use 2.0. Running the named scripts yields models `validate_temporal_contract` would reject.
- **Hardcoded Windows path**: `split_manager.py:29` `CIC2018_DIR = r"C:\SIH_DATA\dump\..."`.
- **No seeding** in `train_branch_a.py` despite the manifest claiming `random_seed: 42`.
- **`requirements.txt` incomplete**: missing `pandas`, `scikit-learn`, `matplotlib`. A clean install then import fails.
- `bita/saved_models/bita_config.json` sets `"use_memory": false` — the TGN memory module is disabled. Defensible, but do not oversell as a memory-augmented TGN.

### 10. Deliverables

- **README** — clear for the app; zero mention of the PS, datasets, training, or benchmarks.
- **Architecture doc — MISSING.** `docs/ARCHITECTURE.md` is 233 lines / **1,576 words** (≈3-4 pages) and is about the *cyber range* — Topology, Workload Generation, SPAN Mirroring, Zero-Disk Telemetry. **There is no ML architecture section at all.** *(A banner marking it as the retired V3.1 design was added in commit `0a01138`; it still needs a rewrite.)*
- **Demo video — MISSING.** No `.mp4`.
- **Presentation — MISSING.** No slides.

---

## D. Ranked gaps — impact × effort

| Rank | Gap | Impact | Effort | Leverage |
|---|---|---|---|---|
| 1 | Explainability + MITRE stage not rendered in UI | Critical | ~0.5 d | **Extreme** |
| 2 | No F1/precision/recall/FPR vs logistic regression | Critical | ~1 d | **Extreme** |
| 3 | Heuristic risk/technique override on live path | Critical | ~1 h | **Extreme** |
| 4 | Held-out generalisation never evaluated | High | ~0.5 d | Very high |
| 5 | Architecture doc: wrong subject, 2× length | High | ~2 h | Very high |
| 6 | Demo video + 5 slides absent | High | ~0.5 d | Very high |
| 7 | Packet-level features absent (PS says "required") | Critical | ~2-3 d | High |
| 8 | No Lateral Movement in any vocab; 3 phases collapsed | High | ~1 d + retrain | High |
| 9 | TCP flags / IAT computed then discarded | High | ~1-2 d | High |
| 10 | CSV demo input missing | Medium | ~0.5 d | Medium |
| 11 | Training scripts don't reproduce checkpoints (60 s vs 2 s) | Medium | ~1 h | Medium |
| 12 | `requirements.txt` incomplete; Windows path | Medium | ~30 min | Medium |
| 13 | `_explain` uses 1-step input in train mode | Medium | ~1 h | Medium |
| 14 | README has no PS/dataset/training/benchmark section | Medium | ~2 h | Medium |
| 15 | 16 s horizon marketed as minutes in UI | Medium | ~15 min | Medium |
| 16 | `world_model/` tree non-importable | Low (if not cited) | ~3 d | Low |
| 17 | Deterministic rollout, not a distribution | Medium | ~1 d | Low |
| 18 | `host_attributes.py` advertises features that don't exist | Low | ~10 min | Low |

**If `world_model/` is cited in the submission as evidence of research depth, #16 becomes Critical** — a judge who clones and runs `python -m world_model.baselines.run_baseline_comparison` gets an immediate `ModuleNotFoundError`. Either commit the missing halves, or keep it for provenance and don't point at it.

---

## E. Top 3 highest-leverage fixes

**1. Wire the real explainability and ATT&CK stage into the dashboard — and delete the heuristic override.** (~1 day.)
`adapter.ts:243-247` → map `p.explainability.top_features` and `.groups`; add `mitre_tactic`/`predicted_stage` to `types.ts:51-57`, render in `Overview.tsx:527` and `Predictions.tsx:185`; fix `+{i+1}m` at `:188`. Then delete `model_adapter.py:350-362`, and pass the real `x_tensor` (`:317`) into `_explain`. Converts requirement 5 from "not acceptable" to satisfied and removes the two artefacts most likely to read as fabricated.

**2. Produce the benchmark table.** (~1 day.)
Standalone `scripts/run_benchmark.py`. `sklearn` LogisticRegression on the same 27-D vectors, Branch A on the same split, precision/recall/F1/FPR → `results/benchmark.json` + README table. Run on `get_heldout_test_records()` and discharge requirements 8 **and** 6 in one artefact. Reuse `detection_metrics.compute_detection_metrics` with two inlined constants. Delete or clearly caption `results/training_metrics.csv`.

**3. Restore minimal packet-level features.** (~2-3 days incl. retrain.)
`pcap_engine.py` is already restored. Call it from `state_builder.close_window()` (packets already buffered), carry six PS-named columns plus a one-line IP-fragment counter into the attr vector, 27→34, retrain via `scripts/retrain_branch_a_live.py`. Consider sourcing from `~/Documents/SIH/processed/network_states_2s_pcap.parquet` instead — the features are already computed and joined at 2 s windows.

**Also fix today (~2 h):** rewrite `docs/ARCHITECTURE.md` as a ≤2-page ML architecture doc; add `pandas`/`scikit-learn`/`matplotlib` to `requirements.txt`; change four `60.0` → `2.0`; move `CIC2018_DIR` to an env var; delete dead `technique_vocab.py` and `host_attributes.py`.

---

## F. Correction to the original agent report

The agent described the dashboard panel caption as reading **"SHAP-style"** to users. *Verified:* the visible title is `"ML Explainability — Feature Impact"` (`Overview.tsx:504`); the string "SHAP-style" appears only in a **code comment** at `:501`.

The panel is still disconnected and still renders hardcoded risk scores — but it does not advertise SHAP on screen.
