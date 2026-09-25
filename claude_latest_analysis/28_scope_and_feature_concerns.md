# 28 — Two reviewer concerns, assessed

**Date:** 2026-09-22
**Raised by:** the user's professor, in review.
**Verdict:** both valid. The second is structural, not a tuning problem.

---

## Concern 2 (taken first, because it is the serious one)

> *"If the recon was done three days ago and the person is continuing recon,
> how can we move to login and all those things?"*

### The contract the system actually runs on

| | |
|---|---|
| window | **2.0 s** |
| history the model sees | 15 steps = **30 seconds** |
| forecast horizon | 5 steps = **10 seconds** |
| encoder long-term memory | **`use_memory: false`** |

The model sees thirty seconds of history and predicts ten seconds ahead, and
the TGNE encoder carries no state between batches.

### The concern is correct, and stronger than "it performs poorly"

If reconnaissance happened three days ago, **there is nothing in this
architecture that could know**. The information is not weakly represented or
hard to recover — it is structurally absent. No amount of training fixes it.

Real campaigns unfold over hours to weeks:

```
recon ──(hours-days)──> initial access ──(hours)──> lateral movement ──> exfiltration
```

A 30 s / 10 s model cannot represent any of those transitions. What has
actually been built is a **near-real-time anomaly forecaster**, not an
attack-campaign progression predictor. "Attack forecasting" in PS 26153
implies the latter, so this is a **scope gap**.

### `use_memory=False` is the specific cause, and it was chosen for other reasons

Disabling the TGN memory module is what makes sample shuffling legal (no
serial batch dependency) and serving stateless — both genuinely valuable, and
both load-bearing for the throughput work done this week. But it is also
precisely what forecloses long-horizon reasoning. **That trade was never
written down.** It is recorded here.

### The campaign layer is aspirational, not functioning

`correlation/campaign_merge.py` exists. Measured:

- imported by **2** files, one of which is `correlation/__init__.py`
- contains **no time-horizon logic at all** — no hour, day, 3600, 86400, or
  gap constant anywhere in it
- not in any training path

So there is currently no component that spans more than 30 seconds.

### What would actually address it

A **multi-scale contract**: keep the 2 s tactical model for near-term alerting,
and add a coarse model (5–15 minute windows, hours-to-days of history) whose
job is stage transition. That is a design change, not a parameter change, and
it needs its own evaluation — campaign-level labels, not per-window ones.

---

## Concern 1: how were the features selected?

The 12 edge features, from the encoder config:

```
log1p_fwd_bytes, log1p_bwd_bytes, log1p_fwd_packets, log1p_bwd_packets,
duration_norm_300s, log1p_byte_rate_norm, log1p_packet_rate_norm,
is_tcp, is_udp, is_icmp, dst_port_norm_65535, directional_flow_asymmetry
```

### Where the concern lands

**There is no ablation study, no feature-importance analysis, and no
documented justification for this set anywhere in the repository.** They are
conventional NetFlow summaries chosen by convention. The question "why these
twelve?" currently has no answer beyond "they are standard".

### One nuance, stated precisely

*Causation* is the wrong bar for a detector — these models need features that
are **discriminative and non-leaking**, not causal. A feature can be purely
correlational and still be the right feature.

But the underlying instinct is sound, and it points at a concrete, testable
danger:

### `dst_port_norm_65535` is a shortcut-leak risk

In CIC-2018 several attack classes sit on fixed destination ports. A model can
learn *port ⇒ attack class* and score well without learning behaviour at all —
and that collapses the moment an attacker changes port. It would also inflate
every number reported so far.

**This is directly testable:** ablate that one feature, retrain the encoder,
compare inductive AUC. If the drop is small, the model was not leaning on it.
If it craters, much of the reported performance was port memorisation.

### Proposed work

1. **Per-feature ablation** — drop each of the 12 in turn, retrain, report
   ΔAUC and Δmacro-F1. `dst_port_norm_65535` first. This answers the
   professor's question with numbers instead of convention.
2. Record the result here, whichever way it goes.

---

## Status

Neither concern is addressed by code yet. Concern 1 has a concrete experiment
ready to run; concern 2 needs a design decision before any code. Both are
recorded so they are not lost, and so nobody later mistakes the current 30 s /
10 s system for a campaign-horizon one.
