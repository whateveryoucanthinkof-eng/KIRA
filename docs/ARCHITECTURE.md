# CyberWorld v4 — ML Architecture

**Scope:** the learned system — state, dynamics, heads, uncertainty, evaluation. Code:
`cyberworld_v4/`. Cyber range, SPAN and capture: `docs/CYBER_RANGE.md`. Audit:
`claude_latest_analysis/07_v4_audit_and_migration_plan.md`.

## 1. Formulation

A host is identified by `(dataset, scenario, capture, host)`, never by raw IP: private ranges recur
across captures. Traffic is aggregated into non-overlapping `Δ = 2 s` windows. Per host per window:

```
s(t) = [ z(t) ; a(t) ] ∈ R^27,   z ∈ R^12 TGNE-TA latent,  a ∈ R^15 flow attributes
```

`a` = flow count; forward/backward/total bytes and packets (log1p); unique peers; unique destination
ports; TCP and UDP ratios; mean duration; byte rate; packet rate; connection density.

**Temporal contract** (single source, `cyberworld_v4/config.py`): history `L = 15` windows (30 s),
horizon `K = 5` windows (10 s). The input is `s(t−L+1 … t)`; every target lies at `t+1` or later.
This supersedes the v3 live contract (`L=5, K=8`) and therefore invalidates all four existing
checkpoints. `contract.py` refuses a mismatched checkpoint at load rather than serving it.

| Output | Meaning | Loss semantics |
|---|---|---|
| `P(A_t)` | attack active now | BCE — nowcast, scored separately from forecasting |
| `P(A_{t+k})` | attack active at `t+k`, `k=1..K` | per-step BCE |
| `h_k` | `P(onset = t+k \| no onset before)` | BCE masked to the at-risk set (discrete survival) |
| `y_{t+k,j}` | ATT&CK technique `j` active at `t+k` | multilabel BCE over the full label set |
| `sev` | operator ranking aid | SmoothL1 — explicitly **not** a probability |
| `ŝ(t+k)` | next-state distribution | Gaussian NLL over `(μ, log σ²)` |

Cumulative onset is derived, never predicted and never `max()`:
`P(onset ≤ t+K) = 1 − Π_k (1 − h_k)`.

## 2. Component chain

```
 SPAN mirror  /  PCAP replay
        │ packets
        ▼
 2.0 s window ─► 5-tuple flow snapshot        (sensor only: no features, no inference)
        │
        ▼ ───────────────── control_backend  /  offline dataset builder ──────────────┐
 │  TGNE-TA (BiTA) temporal graph encoder ─► z ∈ R^12                                 │
 │  per-host flow aggregation             ─► a ∈ R^15      concat ─► s(t) ∈ R^27      │
 │                                   history buffer [1, 15, 27]                       │
 │                          ┌───────────────────┴───────────────────┐                 │
 │                          ▼                                       ▼                 │
 │            Branch A — LSTM forecaster            Branch B — world model            │
 │            ├ current attack        BCE           Transformer, residual delta:      │
 │            ├ future attack   × K   BCE             ŝ(t+1) = s(t) + δ(s)            │
 │            ├ hazard          × K   masked BCE    autoregressive K-step rollout     │
 │            ├ techniques  × K × N   multilabel    emits (μ, log σ²), Gaussian NLL   │
 │            └ severity              SmoothL1                      │                 │
 │                          │                                       ▼                 │
 │                          │                          DeepOP CWA decoder ─► future   │
 │                          │                          ATT&CK technique tokens        │
 │                          └───────────────────┬───────────────────┘                 │
 │           temperature scaling (calibration split) ─► split / adaptive conformal     │
 └────────────────────────────────────┬───────────────────────────────────────────────┘
                                      ▼
              risk curve + technique forecast + attribution ─► dashboard
```

**TGNE-TA** encodes the host-interaction graph into a 12-D latent; neighbour lookup respects the
window cutoff (verified: no future-neighbour leakage). **Branch A** is a deliberately plain 2-layer
LSTM — the ablation floor a graph or transformer encoder must beat before replacing it. **Branch B**
is the world model proper: it learns `P(s_{t+1} | s_t)` as a distribution, not a point estimate with
an error bar attached afterwards. **DeepOP** decodes predicted future states into ATT&CK tokens.
Heads emit logits; sigmoid is applied at the serving boundary so temperature scaling has logits.

**Uncertainty.** Temperature is fitted on a *calibration* split disjoint from validation.
`conformal.py` provides split conformal (finite-sample coverage under exchangeability) and ACI
(long-run coverage without it). Stated limit: ACI guarantees coverage, not detection — a patient
adversary who shapes the residual stream can widen the band, so `min_width` bounds it.

## 3. Evaluation protocol

Four splits, grouped by capture/host so no group crosses a boundary: train, validation
(selection and thresholds), calibration (temperature and conformal quantiles only), test (touched
once, after freezing). Two regimes: chronological, and scenario-held-out for unseen attack families.

Metrics: PR-AUC, ROC-AUC, precision, recall, F1, false alarms/hour; PR-AUC@k, Brier@k, NLL@k for
`k=1..5` plus the horizon-degradation curve; ECE/Brier/NLL before and after calibration; lead time
`t_onset − t_first_valid_alert` (not `K·Δ`); technique micro/macro-F1, mAP, P@1, P@3, R@3.
Mandatory baselines: persistence, last-label, first-order Markov, logistic regression, GBDT, plain
LSTM. Five seeds; confidence intervals by bootstrap over capture/host groups, never over
overlapping windows. If a baseline wins, that is reported as the result.

## 4. Limitations (stated, not omitted)

1. **No trained v4 checkpoint exists.** The contract change is intentional and invalidates the four
   v3 checkpoints. **No benchmark numbers exist yet**; the harness is built, unrun.
2. **Packet-level features are not wired.** `telemetry/packet/pcap_engine.py` computes ~30 features
   (TTL, TCP window, retransmission, payload distribution, vertical/horizontal scan scores). No
   model consumes them. PS 26153 asks for both levels; only the flow level is currently modelled.
3. **The multi-host world model is defined but never trained.** `MultiHostInteractionLayer` and
   `rollout_multi_host` exist and are never called. Dynamics are per-host today.
4. **Training-data identity defects.** CIC-2018 IPs are fabricated on 9 of 10 days, so host
   trajectories there are synthetic; CTU-13's parquet path encodes the label in the destination IP.
   Both must be fixed before any host-graph claim is defensible.
5. **v3 DeepOP was trained on oracle future states**; v4 requires training on world-model output.
6. **Explainability** (attention and feature attribution) is specified and partially present; it
   must run in `eval()` mode on real inputs to be reproducible.
