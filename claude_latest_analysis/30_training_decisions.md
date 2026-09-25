# 30 — Design decisions for the next training run, and for the campaign model

**Date:** 2026-09-24
**How made:** every open choice was put to the project owner as a question, with the trade-offs.
The answers below are theirs. Where a choice changes behaviour, the enforcing code is named.
**Run it:** `scripts/run_training_plan.sh` (the plan in §1) implements every tactical decision here.
**Training policy and dry run:** see [32](32_training_guard_and_dry_run.md). Every trainer steps back after
2 flat epochs and stops at 3. The encoder's time encoding is fixed, and Branch A's discrete gradation
is read off the technique head.

---

## 1. Tactical models: encoder, Branch A, Branch B, DeepOP

| Decision | Chosen | Why | Enforced by |
|---|---|---|---|
| Training data | CIC-2018 **PCAP** + CTU-13 | 9 of 10 CIC-2018 CSV days fabricate host IPs; PCAP has real hosts | `--pcap-root`; CSV refused under cross-year |
| Evaluation | **Cross-year only**: tune on 2018, score once on all of CIC-2017 | the honest generalisation number | `--split-scheme cross_year_ctu` |
| CTU-13 under cross-year | **Train/val only, never tested** | C2 is ~10 hosts in CIC-2018; CTU-13 is botnet C2 | new scheme `cross_year_ctu` in `split_policy.py`; wired into all three trainers and the comparison harness |
| IP features | **Run both** (full, `cross_network`) and report both | `is_private` alone separates CIC-2018 attackers (AUC 1.0) | `run_encoder_comparison.py --ip-ablation` |
| Which variant is served / re-seeded | **`cross_network`, decided in advance** | picking on the 2017 test would make the headline optimistic; picking on 2018 val would pick the shortcut | `CHOSEN_IP` in `run_training_plan.sh` |
| Label role | Either endpoint | unchanged, comparable; covers internal lateral-movement sources | default `attack_role="either"` |
| Encoder arms | CIC-2018 encoder only | no Warden comparison this round | `--arms cic2018` |
| TGN memory | **On** | the BiTA paper's design; the aggregator only runs with memory | `--use_memory` (now the default); batches time-ordered, `backprop_every` 8 |
| Neighbours | 10, most-recent | matches shipped encoders. Raising it only moves the eviction threshold (analysis 29 §2.3) | `--n_degree 10`, carried to serving through the encoder config |
| Packet features (57-D) | Off | needs wiring through trainers + live sensor; TTL-type features can encode how CIC-2018 was staged. Its own A/B later | default |
| `dst_port` | Keep | real signal; cross-year partly tests port reliance | default |
| Branch A risk target | **Hazard + soft BCE** | a real forecast; the old severity target was detection with a one-step delay (analysis 27) | `--risk-target hazard --risk-objective soft_bce` |
| Branch A architecture | Paper (1×256, fixed 0.5/0.3/0.2) | default; gradient-conflict printout now shows whether fixed weights hurt | `--architecture paper` |
| Min history | Full 15 windows | no zero-padding masquerading as a quiet host | default |
| Alert threshold | Best F1 within 2× base-rate budget | a SOC has a finite alert budget | `--operating-point-criterion budgeted_f1 --alert-budget 2.0` |
| Forecast horizon | 5 × 30 s = 150 s | 10 s was a trivial task (persistence ~perfect) | contract default |
| Seeds | 42 for the comparison; **123, 2024** for the chosen variant | mean and spread on the headline without tripling everything | `SEEDS_EXTRA` in the plan |
| Branch B skill gate | Stop before DeepOP if < 2% over persistence | DeepOP cannot beat the rollouts it is trained on | default (`MIN_BRANCH_B_SKILL`) |
| Branch B damping | Keep 0.95^k | consistent train/serve; removes ~7% displacement by step 5 | default |
| Branch B risk target | Hazard | consistent with Branch A | `--risk-target hazard` |

### Built for this run

- **`cross_year_ctu`**: CTU-13's lock train scenarios go to train, and its val and test scenarios
  go to val. It is never tested. Separate from `cross_year` so the pure 2018 → 2017 protocol stays
  reproducible (`docs/CROSS_YEAR_PROTOCOL.md`). The downstream trainer's PCAP path previously read
  PCAP days only, so it would have silently trained Branch B on a different corpus than Branch A.
  It now reads CTU-13 under this scheme. Tests: `tests/test_cross_year_protocol.py`.
- **`scripts/run_training_plan.sh`**: preflight → compare → seeds → downstream → summary. Every
  heavy step runs under `systemd-run --scope -p MemoryMax=17G -p MemorySwapMax=0`, and the script
  refuses to run uncapped unless `ALLOW_UNCAPPED=1`. It never writes `saved_models/`; `summary`
  prints the promotion commands. Steps whose output exists are skipped on re-run.
- **Forecast uncertainty band.** The dashboard *invented* each step's band as
  `risk ± 20 × DeepOP confidence`, which is not a measurement and is backwards (more confidence
  gave a wider band). Branch B training now fits a per-step split-conformal half-width on
  validation residuals (`forecast_risk_conformal` in the checkpoint). The adapter serves it as
  `risk_lower`/`risk_upper` (`control_backend/forecast_band.py`), and the dashboard draws it only
  when present. Tests: `tests/test_forecast_band_is_measured.py`.
- **Sensor blind-spot tile** on the Overview KPI strip: incomplete + missed windows, and kernel
  drops (analysis 29 §2.2).
- **`wed_28` was trained as all-benign (fixed).** The 28 Feb PCAP day's CSV is named
  `wed_29_csv.csv`. The label resolver's weekday fallback took the first `wed_*_csv.csv` in sorted
  order, which is `wed_14` (14 Feb). Intervals are dated, so no 28 Feb packet matched: the
  infiltration day was labelled benign on every PCAP-path run. It is now an explicit alias
  (`training_sources.PCAP_LABEL_ALIASES`), and `check_label_day` raises if a label file's attacks
  fall on another date. The same fallback in the unused `load_pcap_records` was fixed too. The
  lock's *inventory* also recorded wed_14's attack fraction (0.3633) for `wed_28_pcap`. Its split
  (train) is unaffected, and the lock is left frozen. Tests: `tests/test_pcap_day_gets_its_own_labels.py`.
- **`scripts/check_datasets.py`**: verifies every capture by its lock name, pairs each PCAP day
  with its own label CSV, confirms CIC-2017 is the TrafficLabelling variant (has `Source IP`), and
  counts which known-corrupt PCAPs are still the corrupt copy. The launch script's preflight runs it.

### After the run

1. `scripts/run_training_plan.sh summary`: read both IP arms side by side, then the 3-seed mean
   and spread for `cross_network`.
2. Read Branch A's per-epoch `task gradients` line. Persistently negative cosines are the only
   evidence that would justify PCGrad.
3. Read the `neighbour exposure` line per split: how many attack windows the latent never saw.
4. Promote with the printed commands; serving must run with
   `CYBERWORLD_ABLATE_NODE_FEATURES=cross_network`, and the loader refuses a mismatch.

---

## 2. The campaign model (designed now, built in parallel with training)

The tactical models see 30 s and forecast 150 s. Nothing in the system can connect recon from days
ago to a login today (analysis 28, concern 2). This is the second layer that will.

| Decision | Chosen |
|---|---|
| Labels | Generated in the ContainerLab range, with **randomised multi-stage campaigns** (stages skipped/reordered, sampled dwell times), then tested once on a **public stage-labelled APT set** (candidates: DAPT 2020, Unraveled; availability and licence to be confirmed) |
| Dwell-time generation | Record several days of benign workload once; capture each attack stage separately on the same topology; **inject stages at randomised offsets**. Caveat: the model learns the dwell distribution we sample, which is why the public test set exists |
| Outputs | All four: current stage per host; next stage + time-to-next-stage hazard (hours to days); campaign linking; campaign-level risk |
| Inputs | Branch A's **calibrated** technique probabilities and risk, summarised per coarse window, **plus independent coarse traffic stats** (new peers, new ports, auth failures, bytes out), so slow recon below the tactical alert threshold is still visible |
| Memory | **Hidden semi-Markov model**: stages are hidden states with learned transition and dwell-time distributions; days of silence handled natively; interpretable |
| Stages | 6 network-visible: No-campaign, Recon, InitialAccess, Foothold (Execution + C2), LateralMovement, Objective (Exfiltration + Impact), mapped from `CoarseCategory` |
| Timescale | **5-minute windows, 7 days** per host, state persisted to disk across restarts |
| Unit | **Per-host state + learned linker** joining hosts into campaigns (attacker → victim, victim → new victim); campaign risk rolls up its hosts. Replaces the hand-set `correlation/causal_edge_scorer.py` |
| Fitting | **Supervised counts + expert priors** (Dirichlet on transitions following kill-chain order, Gamma on dwell), so an unseen transition is unlikely rather than impossible. No unsupervised EM refit: it can redefine stages, and an attacker who shapes traffic would shape the refit |
| Evaluation | Held-out range campaigns (unseen stage orders and dwell draws) for development; the public APT set **once** as the final test. Metrics: stage accuracy; next-stage hazard Brier at 1 h / 6 h / 24 h; lead time before each stage; linking pairwise F1. Baselines: "last seen stage persists" and the current heuristic linker |
| Alert power | **Escalate only, never suppress.** It can raise priority and group alerts; every tactical alert still reaches the analyst |
| Order | Tactical training runs now; the campaign pipeline is built in parallel. It depends on the tactical models only through Branch A's calibrated outputs, so keep temperature fitting on (the default) |

### Build order for the campaign layer

1. Range generator: extend `workloads/attacker_scenario.py` into per-stage captures with ground
   truth, plus a long benign recording; the injector samples campaigns from them.
2. 5-minute feature builder: Branch A output summaries + coarse stats per host.
3. HSMM with priors; supervised fit; per-host forward filtering for serving (state on disk).
4. Linker, then campaign risk roll-up.
5. Evaluation harness with the two baselines; the public APT test last, once.
6. Serving + dashboard: campaign stage and escalation, never suppression.
