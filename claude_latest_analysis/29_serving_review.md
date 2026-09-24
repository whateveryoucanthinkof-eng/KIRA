# 29 — Serving-path review: the detector cannot alert on lateral movement

**Date:** 2026-09-22

## The finding that matters most

Internal-only traffic is scored `max(0.05, raw_risk * 0.4)`. `raw_risk` is
bounded at 1.0, so the displayed value **cannot exceed 0.40** — while the alert
threshold is **0.65**.

| model risk | displayed | alerts? |
|---:|---:|---|
| 0.30 | 0.1200 | no |
| 0.70 | 0.2800 | no |
| 0.90 | 0.3600 | no |
| 0.9965 | 0.3986 | no |
| **1.00** | **0.4000** | **no** |

You would need `raw_risk = 1.625` to clear the cut. **No internal-only window
can raise an alert at any model confidence.**

**Lateral movement is internal-only traffic by definition.** So the one rule
makes the system structurally unable to alert on the behaviour Branch B and
DeepOP exist to forecast. This is not a tuning choice — it is a ceiling below
the floor.

Rules are **on by default** (`CYBERWORLD_DISABLE_RULES` must be set to turn
them off).

**What was done:** the policy was NOT changed — suppressing internal chatter is
a defensible thing to want, and changing alerting behaviour is the operator's
call. `_check_alerting_is_reachable()` runs at adapter startup and logs
`ALERTING UNREACHABLE` with the arithmetic and three concrete remedies, so the
combination cannot ship unnoticed. Four tests pin it.

**Decision needed:** lower the threshold below 0.40, raise the suppression
factor above 0.65, or disable rules for any run whose results are quoted.

## The parity check was measuring itself

`verify_offline_live_parity.py` compared `off_attrs` and `live_attrs` — **the
same expression**, both calling `compute_host_temporal_attributes()` with
identical arguments. 15 of 27 dimensions were compared against themselves and
would have reported 0.0 however far the paths had drifted. The "PARITY,
0.000e+00 over 300 windows" previously reported was true of the 12 embedding
dimensions and vacuous for the other 15.

Fixed: the offline side now takes both halves from the snapshot the trainers
actually consume (`feats[row] = [embedding ; temporal_attrs]`). The two sides
share no code line.

**And my first fix was worse than the bug.** I clipped the live side to the
last `window_seconds` while leaving the offline side unclipped, and it reported
`DIVERGENT, 6.789` across 105 of 105 windows. Measured directly:

```
same flows to both paths   max |delta| 0.0000e+00   0/300 windows differ
live clipped, offline not  max |delta| 6.7894e+00   105/105 differ
```

The code paths agree exactly. Feeding them different inputs measures the
inputs. Now passes on all 27 dimensions over 300 windows, no windows skipped.

Still open: whether the adapter's window clip lines up with the extractor's
internal windowing. It needs a fixture spanning several windows; this one is
already grouped into single 2 s windows and cannot exercise it.

## Fixed in the review

- **Attribution was computed on an all-zero input.** `for e in obs_entries[-5:]` never read `e`; every timestep was zeros, so every host got the same narrative: `H_emb_0 (0.0%), H_emb_1 (0.0%), H_emb_2 (0.0%)`. The all-zero-features defect, still live in the explainability path.
- **Right-padding where every trainer left-pads.** Short hosts had real data first and zeros last; for Branch B the *last* history step — the seed for the residual rollout — was the zero vector, so those hosts were forecast from the origin.
- **`sequence_ready` and the timestamp both lied.** The h-state list was padded in place, mutating retained history (`sequence_ready=True, buffer_length=15, non-zero entries=1`). The timestamp was local time labelled `Z` — off by 5h30m from every other event on the bus.
- Hardcoded `>= 65` alert gate ignoring the fitted threshold; invented "live" packet-loss/throughput/latency metrics; `K=4, window=60.0` MACRO defaults no model is trained at; `confidence=max(0.4, 0.9-0.1*k)` that never consulted the model.

## Reported, not fixed

- **`flow_table.py` counters are cumulative since flow start, exported as per-window.** A steady 20 kB/2 s flow reports 20k, 40k, 60k, 80k, 100k across consecutive windows; training counts each flow once. A train/serve mismatch the parity fixture does not reach.
- **`causal_edge_scorer.py`** gates on `min_score_threshold=0.35` over an MLP **never loaded from any checkpoint** — randomly initialised weights.
- **`/api/replay` mutates the serving singleton** while the tail worker may be scoring on it; `finally` restores rules but not history.
- The rule layer's coefficients (0.40, 0.15, 0.4, 75.0) have no derivation anywhere.
