# ML review: do this in the next session

This file lists what is already known to need an ML-engineering review. It was written on 2026-09-25, while the full-scale
run was paused, after epoch 1 exposed a category-head collapse. Review everything below
as an ML engineer before trusting any metric from the run on branch v5.5o.

## Status after the 2026-10-02 code review (branch `fix/ml-review`)

Diagnosed from code only: no training, no real data (being re-fetched). Verified by
unit tests and the synthetic end-to-end dry run. The full run (7 epochs, not 1) collapsed
every epoch: val predicted InitialAccess for every edge, macro-F1 0.05-0.08, test Hits@3 ≈ 1.0.

**Root causes of §1, fixed:**

1. **Labels were the clock.** `config/attack_participants.json` ships empty and 9 of 10
   CIC-2018 CSVs have no IP columns, so every flow of every captured host (~445/day)
   inside an attack interval got the attack label. That gave 9.0M InitialAccess training
   edges, against ~0.5M InitialAccess rows in the CSVs. Now: unattributable intervals are
   `UNKNOWN` (excluded from every supervised target; legacy:
   `CYBERWORLD_UNSCOPED_LABELS=time_only`). Scoped intervals label attacking
   **pairs**, not "any flow touching a participant", which marked a victim server's
   clients as attackers. `apply_participants` had **no caller**; it does now.
2. **Training order ran capture after capture** (CTU-13, then the CIC days in date
   order), so the head ended each epoch fitted to the last day. Edges are now
   interleaved by within-capture progress, which is exact for TGN memory because nodes
   are per capture.
3. **Selection let a collapsed head win** (arithmetic mean, AP-dominated). Now the
   harmonic mean of inductive AP and macro-F1.
4. Per-class TRAIN accuracy is logged every epoch.

**Other items:**

| item | status |
|---|---|
| §2 selection metric | fixed (harmonic mean) |
| §2 mid-epoch validation | not needed: an epoch is ~18 min at level 4, not 70 |
| §2 UndefinedMetricWarning | fixed: single-class AUC slices are NaN; absent classes NaN, not "0.0 recall" |
| §2 leakage audit | Branch A target is the window after the history (no overlap). Added the **onset** slice (last input window benign), because continuations are predicted by persistence alone |
| §3 label scoping | fixed (above). Per-flow attack labels on the no-IP days need data: CSVs with Src/Dst IP, or a verified participant map |
| §3 CIC-2017 CSV test vs PCAP train | **open (data decision)**: different flow extractors. Using PCAP-derived CIC-2017 flows would also unlock packet features on test |
| §3 absent classes | handled: NaN per class; UNKNOWN never a class |
| §4 time-ordered batches | fixed (interleaving) |
| §4 inductive draw | UNKNOWN no longer drives class protection |
| §5 Branch A | benchmark added: **logistic regression** (PS-mandated, same inputs/target/threshold rule), persistence, FPR, onset slice |
| §5 Branch B | now a **distribution**: per-step Gaussian (variance head, NLL, 90% coverage), mean training unchanged |
| PS packet-level features | **plumbed, opt-in** (`--packet-features`, 27-D → 57-D, columnar path bit-identical, serving + explanations). Needs PCAP-derived test captures |
| §7 resume RNG | fixed |
| §7 stale serving fixture | open: needs the retrained encoder |
| §8 CredentialAccess in correlation | fixed (between Recon and InitialAccess, as the console draws it) |
| §6, rest of §8 | open (evtx logs, LateralMovement labels, per-alternative rollouts, attention saliency, campaign heuristic evaluation) |
| edge-feature ablation (`dst_port`) | open: needs a retrain |

**Second and third pass (same day), each fixed and tested:**

| area | bug | fix |
|---|---|---|
| encoder test | the best epoch's weights were tested on the **last** epoch's memory (epoch 4 weights + epoch 7 memory on 2026-09-25); a run resumed after its last epoch died with a NameError | keep the best epoch's post-validation memory; forward-only replay when none exists |
| hazard target | tau = 5 × 2 s = **10 s** while the contract forecasts 5 × 30 s = 150 s (the target was nearly a nowcast) | tau = `forecast_seconds` in both trainers |
| CIC-2017 test | Branch B risk scored against the severity column (store never converted to hazard) | converted like train/val |
| cross-year scoring | `rollout()` without elapsed times (model told every step is 2 s); padded steps counted; risk head never scored | real times, masked steps, risk MAE vs predict-zero |
| Branch B data | edge-padded future copies trained and scored as targets (free win for persistence) | `future_valid` mask in losses, validation, conformal |
| train/serve | serving scored new hosts with zero-padded histories that no model trained on; Branch B was zero-padded, DeepOP's rollouts edge-padded | Branch A trains with `--min-history-steps 1`; Branch B learns short edge-padded histories; serving edge-pads |
| train/serve | serving took flows by END time (long flows counted in every window); training buckets by START time | serving buckets by start time |
| host attributes | sent/received bytes and packets not oriented to the host (a flood's victim "sent" the flood) | oriented, both paths; schema 2.1.0 |
| Branch A selection | arithmetic mean of macro-F1 and overall AUC (dominated by continuations) | `early_warning`: harmonic mean of macro-F1 and onset Gini |
| Branch B alerts | forecast risk thresholded with Branch A's operating point | Branch B fits and ships its own |
| explanations | Input×Gradient read only the last of 15 windows; batch API defaulted to "input magnitude" | summed over all timesteps; gradient attribution by default |
| DeepOP | trained on clean history tokens, served Branch A's predictions; noise only removed tokens | + 10% substitution noise |
| rollout callers | standalone DeepOP trainer also omitted elapsed times | fixed; AST test covers every caller |
| CTU-13 labels | `Background` (unlabelled per the dataset authors) resolved to Benign | `CYBERWORLD_CTU_BACKGROUND=unknown` switch (default unchanged) |

**Fourth pass:**

| area | bug | fix |
|---|---|---|
| hazard target | windows just before dropped UNKNOWN traffic saw no "next attack" and got hazard 0 (false negatives on pre-attack windows) | column loading records the dropped spans; hazard is NaN (censored) where they could matter; every consumer masks NaN |
| edge features | log1p bytes/packets reached ~20 while every other encoder input is in [-1, 1] | scaled to [0, 1] (schema 2.1.0), all implementations bit-identical |
| ingest cache | `attack_participants.py` not in the parse-code hash (lazy import) | added |
| encoder promotion | `select_best_encoder.py` re-scored with the old arithmetic mean | harmonic, as the trainer |
| plan summary | reported overall AUC only | + onset AUC and the LR/persistence benchmark |

**Fifth pass: clean.** Verified: serving event adapter uses the canonical edge
features; encoder loader enforces schema, ablations and dimensions; conformal
order statistic; batch planner + fast path level 4 + capture interleaving
(GPU run); UNKNOWN class inside the CUDA-graph training step (GPU run).

Reviewed and left as is (by design or pinned by tests): guard gives one epoch after a
step-back; BiTA cross-edge context; TGN attention/neighbour finder (strictly before t);
DeepOP repetition penalty/continuity bonus (off for paper checkpoints).

## 1. Category head collapses at full scale (highest priority)

- **Observed:** epoch 1 at full scale (130.6M edges, 3.49M nodes, 4 classes). Both encoders
  (`full` and `cross_network`) predict **InitialAccess for every validation edge**:
  - per-class val accuracy: Benign 0, C2 0, Impact 0, InitialAccess 1.0;
  - CatAcc 0.0886, which equals the InitialAccess share of validation;
  - val macro-F1 0.054.
- **Link prediction is healthy:** val AUC 0.98 and AP 0.98.
- **The training category loss is low** (0.018 × `cat_loss_weight` 15), so the head fits in
  training but outputs a constant at validation.
- **Ruled out (measured or checked, 2026-09-25):**
  - the shared setup store: CTU-13 1+2 with and without it, both healthy;
  - the edge-feature gather at indices past 2^31 bytes (`np.take` with int64);
  - BatchNorm (there is none; the head is Linear→ReLU→Dropout→Linear).
- **Does NOT reproduce** on CTU-13 1+2 (2 classes, ~4.6M edges): Benign 0.976, C2 0.986.
- **Hypotheses to test:**
  - train→val label and feature shift from the per-capture 70/15/15 **time** split:
    CIC-2018 attacks sit in specific time windows, so val windows may hold different
    attack types or none;
  - focal loss (gamma 2) with inverse-frequency alpha (Benign 0.28, C2 5.0,
    Impact 0.97, InitialAccess 0.64) combined with `cat_loss_weight` 15;
  - memory or embedding saturation over a 59M-edge epoch;
  - edge-feature scaling (what `extract_canonical_edge_features` feeds the head,
    raw versus normalised);
  - whether training category accuracy is really high: log it per class during training;
  - the category head sees `[src_emb, dst_emb, edge_feat]`. Check how much of each it uses.

## 2. Evaluation and model selection

- **Selection score** = mean(inductive AP, val macro-F1) (about 0.52 at epoch 1). The link
  term dominates, so a collapsed category head still "IMPROVED". Decide whether that is
  the right selection metric.
- **Epoch length:** validation runs once per ~70-minute epoch (461k batches), and the guard's
  patience is 3 epochs. Consider mid-epoch validation checkpoints.
- **`UndefinedMetricWarning`** (single-class AUC slices): find which slices, and whether a
  metric silently excludes them.
- **Leakage fix (009f382):** the category head embedded twice, and the second pass saw the
  batch's own future. Audit the eval code, Branch A and downstream for any analogous
  within-batch or future leakage.

## 3. Data and labels

- **Train/test domain mismatch:** training flows come from PCAPs through our own flow
  extractor (Rust port of `telemetry/`). The CIC-2017 **test** flows come from CICFlowMeter
  CSVs. Different flow definitions mean a feature distribution shift. Quantify it, and
  decide whether the cross-year headline is measuring the model or the extractor.
- **Class coverage:** C2 exists only in CTU-13 (112k edges, against 46M Benign), and
  InitialAccess and Impact only in CIC-2018. Check which classes exist in val and test per
  corpus, and whether macro-F1 over absent classes is meaningful.
- **Label scoping:** PCAP labels come from time windows in the CSVs, scoped to known
  participants, and fall back to `time_only` when there are no participants. Measure how
  many edges carry time-only labels (noise).
- **263 corrupt PCAPs:** their days are truncated at the damaged record. Check whether
  attack windows fall in the missing parts, which would leave labels without traffic.
- **Node identity:** nodes are keyed per (capture, IP). The IP ablation (`cross_network`
  zeroes the octets and address-class flags) is the served variant. Confirm that no other
  identity shortcut remains (ports, per-capture node ids).

## 4. Training dynamics

- Batches are time-ordered and never shuffled (TGN memory requires it). Check the
  consequences for Adam, focal loss and the guard's LR step-backs.
- The inductive new-node draw holds out about 10% of the nodes seen in val/test, with a
  class-protection re-draw. Check its effect at this scale (Train=59M of 91M train-window
  edges).
- **Numerics:** fast path level 3 is bit-identical to level 2. Level 2 differs from the
  reference at fp32 rounding, and the reference's cuDNN GRU uses TF32 for weight gradients.
  Seeds: 3 planned (42, 123, 2024), and GPU atomics are nondeterministic.

## 5. Downstream (not yet run at PCAP scale)

- **Branch A:** check the hazard target, `hazard_tau`, the `budgeted_f1` operating point
  with alert budget 2.0, and the new columnar extraction path (bit-identical on tests, and
  5.5 GB for one full day).
- **Branch B:** check the credibility gate (must beat persistence by 2%) and what happens
  to DeepOP when it fails. The dry run shows Branch B is not credible on synthetic data,
  which is expected there.

## 6. Unused data worth a decision

- `DATA/logs`: Windows `.evtx` host event logs per host per day (1.8 GB), not used anywhere.
  They are a possible host-activity signal alongside the network graph.

## 7. Found while finishing the optimisation work

- **Resume doesn't save the validation samplers' RNG** (found by the batch-planner
  work). After a crash-resume, validation negatives restart from their seed, so validation
  metrics differ slightly from an uninterrupted run (ctu7: AUC 0.9211 vs 0.9215).
  Final weights are unaffected.
- **Serving tests fail on a stale saved checkpoint.** `test_serving_replay_isolation`,
  `test_control_backend`, `test_topology_service` and `test_site_config` load
  `saved_models/bita_bigru_transformer-unified_final.pth`, which is feature schema 1.0.0
  while the tree is 2.0.0. Promote the new encoder, or regenerate the fixture.

## 8. Model gaps found while wiring the K.I.R.A. dashboard (2026-10-01)

Found while mapping each dashboard panel to a real model output (branch
`feat/kira-dashboard`, see `docs/DASHBOARD_INTEGRATION.md`). The console shows each of these
as "Under development" until a model produces it.

- **No trained downstream checkpoints, and a stale encoder.** `saved_models/` has no
  `branch_a/`, `branch_b/` or `deepop/` files. The only encoder `.pth` is on feature schema
  1.0.0, but the tree is on 2.0.0. The backend now starts and serves without models, but nothing
  can be scored until the retrain lands.
- **No Lateral Movement or Execution output.** Branch A's `TECHNIQUE_VOCAB` (14 techniques)
  has no T1021, and DeepOP's joint vocabulary has 6 macro-techniques plus Benign, with no
  LateralMovement or Execution token. Pivoting between internal hosts, the core of a
  multi-host campaign, can't be named by any model. Decide whether to add these classes.
  (Is there labelled lateral-movement traffic in the corpora at all?)
- **`correlation/` doesn't know CredentialAccess.** `TACTIC_ORDER` in
  `causal_edge_scorer.py` has no `CredentialAccess`, yet DeepOP emits `CredentialAccess.T1110`
  (brute force). Those alerts get rank -1, so they score as "not an attack stage"
  (plausibility 0.1) and rarely link into a campaign. Not changed silently: decide where it
  sits in the order (the dashboard draws it between Recon and InitialAccess).
- **Forecasts are per host; nothing predicts *where* an attack goes next.** Branch B and
  DeepOP roll out one host's future. The forecast tree's "hops" (hosts a branch passes
  through) has no model behind it. Neither does a per-host attack path.
- **One rollout, not one per alternative.** Branch B produces a single future-state
  trajectory, so the risk head gives one risk curve. The three DeepOP alternatives shown as
  A/B/C (top-3 first tokens, then greedy) have no per-branch risk peak.
- **No traffic-volume forecast.** No model predicts future packets or bytes (the tree's
  per-branch packet/byte figures).
- **Attention saliency isn't computed.** TGNE's attention weights can be read out (2 heads,
  `graph_attention`), but the per-cell Input × Gradient breakdown over the key's inputs
  (12 edge features, time encoding, memory) isn't implemented.
- **Campaigns and incidents are heuristic.** They come from `correlation/` (hand-set
  parameters, nothing fitted) applied to model outputs, not from a learned model.
- **Evaluate the campaign heuristic before trusting the Campaign page.** As of 2026-10-01
  `control_backend/correlation_service.py` runs `correlation/` live (the only serving caller of
  `HeuristicCausalEdgeScorer`, allowlisted in `tests/test_serving_causal_edge_guard.py`).
  `HEURISTIC_PARAMS` has never been evaluated, and a big kill-chain jump (Recon → Impact) scores
  under the 0.35 link threshold, so those two episodes show as separate campaigns.
