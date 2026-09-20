# Analysis Reports — 2026-09-19

Assessment of this repo against **SIH PS 26153 (NTRO)** — *AI based Network Attack Forecasting from Network Traffic Data* (`../ps.md`).

| # | Report | Question it answers |
|---|---|---|
| 01 | [Training data / PCAP audit](01_training_data_pcap_audit.md) | Do the live models train on raw PCAP, or only flow records? |
| 02 | [PS 26153 gap analysis](02_ps_26153_gap_analysis.md) | Requirement-by-requirement: what's MET, PARTIAL, MISSING |
| 03 | [RL for training](03_rl_for_training.md) | Should reinforcement learning be applied at training time? |
| 04 | [RL at inference + MIRAS](04_rl_at_inference_and_miras.md) | Test-time adaptation, online learning, MIRAS |
| 05 | [CIC-2018 PCAP completeness](05_cic2018_pcap_completeness.md) | Is the claim "the PCAPs only hold 1–2 hours" true? |
| 06 | [PCAP corruption scan](06_pcap_corruption_scan.md) | Full census: which files are corrupt and need re-downloading |

## Corruption scan (report 06) — supersedes report 05's integrity claim

A full `capinfos` census of all 4,456 capture files found **263 files (5.9%, 56.2 GB) with severe
record-chain corruption** that need re-downloading, plus 75 with a merely truncated tail that need
nothing. Report 05 had concluded "only 4 files lost real coverage" — that was wrong, because its two
screens (`size % 4096`, tail-seek) are structurally blind to a bad length field mid-file: the tail bytes
are still valid records, so a tail-seek lands cleanly.

Re-download list: [`corrupt_files_redownload.txt`](corrupt_files_redownload.txt).

**After re-downloading a day, verify it:** [`verify_pcap_day.sh`](verify_pcap_day.sh)

```bash
toolbox run -c prism-dev ./claude_latest_analysis/verify_pcap_day.sh wed_14_pcap
```

It walks every record chain in that day and diffs against the baseline, reporting fixed / still-corrupt
/ newly-corrupt. If nothing was fixed, the damage is upstream and further re-downloading will not help.

Corruption clusters by day (1.1% on `tue_20` to 12.2% on `wed_14`) **and by host** — 19 hosts are corrupt
on 5+ of the 10 days, which hints the damage may be upstream rather than transfer-related. Verify one
file from the worst host before pulling 56 GB.

Report 05's other conclusions stand: all 10 days present, per-host split, ~9 h captures, snaplen 65535,
and the "only one or two hours" claim is still false.

## PCAP verdict (report 05)

**The "PCAPs only have 1-2 hours" claim is FALSE.** 4,457 files, ~560 GB, **all 10 official capture days
present** (dates confirmed from packet timestamps, not filenames). The corpus is split **per host, not per
time slice**: ~445 files per day, one per victim machine, each spanning the full ~9-hour capture day.
Across 143 probed files: 8.7-11.2 h, median ~9.1 h. Snaplen 65535 — full payloads, so every PS-required
packet feature is physically recoverable.

The claim came from ~8 fragment files (0.2%) that genuinely are short because they are *time-consecutive
pieces of one host's day* — `…69.13 part1` (1.71 h) + `part2` (0.53 h) + `part3` (6.90 h) reassemble to
9.22 h. They sort to conspicuous positions in a directory listing, which is exactly what a spot-check
opens first.

### The finding that matters more than the verdict

**Only `tue_20_csv.csv` (84 cols) carries `Src IP`/`Dst IP`. The other nine CSVs are 80 cols with no IP
columns at all** — verified. For those nine days `data_unification/cic2018_adapter.py:65,70` **fabricates
host IPs by row index**:

```python
src_ips = np.array([f"192.168.10.{i % 250 + 1}" for i in range(len(chunk))])
dst_ips = np.array([f"172.16.0.{i % 100 + 1}"  for i in range(len(chunk))])
```

Source ports are randomised at `:75`. This adapter is on the shipped path —
`scripts/retrain_branch_a_live.py:25,31,36` — so **the host graph the TGNE learned over is synthetic for
9 of 10 training days**. A "host trajectory" is every 250th row of a CSV. For a system whose thesis is
per-host trajectory forecasting, that is a correctness problem independent of any packet-feature gain,
and the PCAPs (one file per host) supply exactly what is missing.

## RL verdicts, in one line each

**Training-time RL: no.** The system's output does not change its input — a predicted ATT&CK stage doesn't alter the attacker's next packet. Transitions are exogenous, so the MDP collapses to a contextual bandit; and because the dataset is fully labelled, every action's reward is computable offline, making it a *full-information* bandit, which is supervised learning. In every canonical world-model system (Ha & Schmidhuber, DreamerV3, MuZero, TD-MPC2) RL trains the **controller**, not the dynamics model — and this repo has no controller because the PS doesn't ask for one.

**MIRAS: does not apply.** It is an architecture-design framework for building sequence-model *layers* (arXiv:2504.13173), where "test-time memorization" means a memory module updates its own memory parameters inside the forward pass. The outer weights stay frozen. It is not runtime adaptation bolted onto a deployed model, it targets contexts up to 10M tokens, and this system's context is 5 × 12 = 60 numbers. Adopting it means discarding all four checkpoints for a full retrain, with no benchmark in the repo that could show whether it helped.

**Online conformal prediction: the one concrete runtime win.** Label-free, no retrain, no checkpoint touched.

## The label ≠ outcome distinction

The sharpest idea in report 04. *Labels* (is this an attack?) are genuinely unavailable at runtime, forever. *Outcomes* are free: the rollout predicts `h_{t+K}`, and 16 s later the real `h_{t+K}` is already sitting in `h_state_history_by_target`. That splits the heads cleanly:

| Component | Runtime-supervisable? |
|---|---|
| Branch B (latent dynamics) | **yes** — outcome observed 16 s later |
| Branch A (risk / technique) | **no** — target is a lookup on ground-truth tactic |
| DeepOP | **no, and worse** — its "observed token" comes from the heuristic override, so self-training amplifies a hand-written rule |

Runtime adaptation is legitimate **only** for the latent dynamics and for calibration. Never for the risk score or the stage classifier.

## Verification status

Findings marked *verified* in these reports were independently re-checked against the code before being recorded. Two agent claims were **corrected** rather than passed through:

- **01 §6** — Branch A's negative training loss was flagged as evidence of a degenerate dataset. It is not: `MultiTaskUncertaintyLoss` is Kendall & Gal homoscedastic weighting (`L = exp(-s)·L + s`), where the bare `+ s` terms go negative by construction. The *99.94% technique accuracy* half of that finding does stand.
- **02 §F** — the dashboard panel was described as captioned "SHAP-style" to users. That string is in a code comment only; the visible title is "ML Explainability — Feature Impact". The panel is still disconnected.

## The short version

The modelling core is real — a genuine autoregressive latent world model with K-step rollout, ATT&CK mapping, and working gradient-based attribution. The losses are in the **evidence layer** and the **demo layer**.

Three findings are more dangerous than any missing feature, because a judge can reach them by opening two files:

1. **The dashboard never reads the model's explainability.** The backend computes real Input×Gradient attributions and puts them on the wire; the frontend renders three hardcoded risk scores relabelled as features, one of them permanently zero. (`web_dashboard/src/api/adapter.ts:243-247`)
2. **The headline risk number is a hand-written rule.** During any external-flow activity — i.e. every attack demo — displayed risk is `max(model_output, 0.40) + f(flow_count)`. (`control_backend/model_adapter.py:350-362`)
3. **No F1, precision, recall or FPR exists anywhere.** The only numeric artefact is a single-epoch TGNE curve with `val_accuracy=0.0` and inductive AUC *below chance*. (`results/training_metrics.csv`)

Highest-leverage fixes, in order: wire the real explainability + ATT&CK stage into the UI (~0.5 d), delete the heuristic override (~1 h), produce the logistic-regression benchmark on the held-out split (~1 d). Those three are roughly two days and move four requirements.

## Related repo changes made during this analysis

| Commit | What |
|---|---|
| `2490b6e` | Removed dead files; `CYBERWORLD_*` env vars with lowercase fallback; `utcnow()` → `utc_now_iso()` |
| `7b297a1` | Removed the dead 72-D capture vector; **fixed `--replay`**, which silently produced zero windows |
| `87e47bc` | Restored `world_model/` (research + benchmark track) |
| `0a01138` | Restored `pcap_engine.py`; marked `docs/ARCHITECTURE.md` as the retired V3.1 design |

Everything removed at any point is preserved on `archive/pre-cleanup-2026-09-19` (on both remotes).

---

## v4 build status

`cyberworld_v4/` is the rebuilt scientific core. 118 tests pass; pyflakes clean.

| Built | What it replaces |
|---|---|
| `config.py` / `contract.py` | contract scattered across modules; refuses mismatched checkpoints |
| `identity.py` | raw-IP grouping and `hash()` (salted per process) |
| `targets.py` | `target_snap = window_slice[-1]` — the nowcast defect |
| `splits.py` | missing calibration split; row-level splitting |
| `metrics/` | accuracy as headline; horizon = "lead time" |
| `conformal.py` | hardcoded ±0.05 called conformal |
| `models.py` | one "risk" scalar doing four jobs |
| `benchmark.py` | no baselines, no benchmark table |
| `manifest.py` | checkpoints with no reproducible origin |
| `scripts/train_v4.py` | trainers that could not reproduce their own outputs |
| `data_unification/pcap_adapter.py` | fabricated host IPs; unwired packet features |
| `POST /api/replay` | no CSV/PCAP demo input |

### Defects found by building it

Each was measured, not inferred:

1. **TGNE embedded on a different graph at train vs serve.** Trainers pass the full window; the live
   adapter passed only the target's edges. 1.5e-2 per dimension. Neither this audit nor the external
   review found it — only the parity test did. Now 0.000e+00.
2. **Prefix sampling makes forecasting vacuous.** `--rows-per-file` is pandas `nrows`; the first 60k
   rows of a day are 87–100% one label. Measured label churn: **0.0000**, persistence PR-AUC **1.000**
   at every horizon. The 0.98 "forecast PR-AUC" that produced was nowcasting relabelled.
3. **Lead time inflated by duplicate timestamps** — `list(times).index(t)` credited an earlier window.
4. **Benchmark compared two probability scales** — threshold from raw validation scores applied to
   calibrated test scores, yielding all-positive operating points.
5. **`MITRE Unknown` on the dashboard** — the rule layer emits labels that are not ATT&CK ids.
6. **React hook-ordering crash** in `Overview.tsx` — `useState` after an early return.

### The number worth remembering

With the SOC rules on, the console displayed **0.657**. The model output **0.197**. Both are now on the
wire as `ml_risk` / `rule_risk`, and `CYBERWORLD_DISABLE_RULES=1` gives model-only output.

### Still open

No trained v4 checkpoint that clears the degeneracy guard. The v4 contract invalidates all four v3
checkpoints by design, and the CSV path may not be able to support a forecasting claim at all — which
is why `pcap_adapter.py` exists. Packet features are extracted but not yet in a model's input space.
