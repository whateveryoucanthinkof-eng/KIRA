# Analysis Reports — 2026-09-19

Assessment of this repo against **SIH PS 26153 (NTRO)** — *AI based Network Attack Forecasting from Network Traffic Data* (`../ps.md`).

| # | Report | Question it answers |
|---|---|---|
| 01 | [Training data / PCAP audit](01_training_data_pcap_audit.md) | Do the live models train on raw PCAP, or only flow records? |
| 02 | [PS 26153 gap analysis](02_ps_26153_gap_analysis.md) | Requirement-by-requirement: what's MET, PARTIAL, MISSING |
| 03 | [RL for training](03_rl_for_training.md) | Should reinforcement learning be applied at training time? |
| 04 | [RL at inference + MIRAS](04_rl_at_inference_and_miras.md) | Test-time adaptation, online learning, MIRAS |

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
