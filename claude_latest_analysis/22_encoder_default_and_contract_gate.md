# 22 — The retrained encoder was not the default, and what that exposed

**Date:** 2026-09-21

## 1. Six call sites were still loading the broken encoder

`build_or_load_tgne_ta()` fell back to
`bita/saved_models/bita_bigru_transformer-warden_alerts.pth` whenever neither an
explicit `checkpoint_path` nor `TGNE_CHECKPOINT_PATH` was given.

Six call sites call it with **no argument**, including
`control_backend/model_adapter.py:169` — the live serving path. The Branch A
retrain only used the new encoder because its systemd unit sets
`TGNE_CHECKPOINT_PATH`. Anything launched without that env var — tests, the
serving adapter, `train_v4.py`, `train_cwa_decoder.py` — resolved to the Warden
checkpoint, the one whose category head was a single `Linear` over summed node
embeddings and collapsed to a constant prediction.

Since the head changed shape, this now fails loudly (`StaleEncoderArchitecture`)
rather than silently serving a broken encoder — but it still means training and
serving disagreed about which encoder exists.

**Fix.** Resolution order is now explicit, in `branch_a_gnn_lstm/train_branch_a.py`:

1. explicit `checkpoint_path` argument
2. `TGNE_CHECKPOINT_PATH`
3. `CANONICAL_TGNE_CHECKPOINT` — `saved_models/bita_bigru_transformer-unified_final.pth`
4. `LEGACY_TGNE_CHECKPOINT` — the Warden path, kept last so an install without
   the retrained file still gets the familiar error, not `FileNotFoundError`

Paths resolve from the repo root, so a different cwd still finds them. Verified:
the default now loads the edge-aware head (`Linear(36→36) → ReLU → Dropout →
Linear(36→5)`).

## 2. Branch A's contract test flipped

`test_served_checkpoints_match_the_contract[branch_a]` reported
**`XPASS(strict)`** — it started passing, which a strict xfail correctly treats
as a failure so a finished retrain cannot be quietly forgotten. That is the
mechanism working.

The marker covered all three branches at once, but they retrain one at a time.
It is now per-parameter: Branch A asserts for real; Branch B and DeepOP keep
`_AWAITING_RETRAIN` until `retrain_future_models_live.py` finishes. Test status
is documented in the module docstring.

## 3. Enabling the adapter tests exposed a real contract conflict

With the encoder default fixed, the three adapter test modules stopped being
skipped — and immediately errored at collection:

```
RuntimeError: Checkpoint contract conflict on history_steps:
branch_b says 5, an earlier checkpoint said 15.
Checkpoints trained under different temporal contracts cannot be composed.
```

**This is correct behaviour, not a regression.** Branch A is now v4
(history 15 / forecast 5); Branch B and DeepOP are still v3 (history 5 /
forecast 8). The adapter refuses to compose them. It should.

The problem was only that a collection *error* takes the whole suite down. The
`tests/conftest.py` gate previously asked one question ("does the encoder
load?"); it now reports *why* the adapter cannot be built, covering both
preconditions, and checks the **import** rather than the constructor —
`model_adapter.py:820` builds a module-level singleton, so the conflict raises
on import.

The skip prints its cause and clears itself once Branch B and DeepOP retrain:

```
SKIPPING adapter tests: Checkpoint contract conflict on history_steps:
branch_b says 5, an earlier checkpoint said 15. ...
These re-enable themselves once the staged retrain completes.
```

## 4. Checked and found *not* to be a bug

The adapter warns `branch_a checkpoint carries no credibility verdict; its
metrics are unvalidated`. That is expected mid-run: `retrain_branch_a_live.py`
stamps `ckpt["credibility"]` only after the final held-out test evaluation. The
per-epoch saves legitimately lack it. No change made.

---

**Suite after all of the above: 453 passed, 2 xfailed** (the two xfails are
Branch B and DeepOP, awaiting their retrain).
