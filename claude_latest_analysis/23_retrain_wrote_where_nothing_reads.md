# 23 — The downstream retrain wrote to a path nothing loads

**Date:** 2026-09-21
**Severity:** would have silently wasted the entire Branch B + DeepOP retrain.

## The defect

`scripts/retrain_future_models_live.py` saved its two checkpoints as:

```
<out_dir>/host_wdt.canonical-tgne.pt
<out_dir>/cwa_forecast_decoder.canonical-tgne.pt
```

`control_backend/model_adapter.py` loads:

```
saved_models/branch_b/host_wdt.pt              (line 187)
saved_models/deepop/cwa_forecast_decoder.pt    (line 202)
```

A repo-wide search found **no reader of `canonical-tgne.pt` anywhere** — the
only two occurrences were the two writes. It was a write-only artifact.

## Why it would not have been noticed

The downstream retrain would have run for hours, printed improving losses,
saved checkpoints, and exited 0. Meanwhile:

- `saved_models/branch_b/host_wdt.pt` would still be the **Sep 10 v3** file
- the adapter would keep raising `Checkpoint contract conflict on
  history_steps: branch_b says 5, an earlier checkpoint said 15`
- the three adapter test modules would stay skipped
- `scripts/verify_offline_live_parity.py` would stay blocked

...and every one of those symptoms is *already present today* for a legitimate
reason (Branch B genuinely has not been retrained yet), so the retrain finishing
without changing any of them would have looked exactly like the state before it
ran. There is no runtime check that catches this: two source files simply
disagreed about a filename.

Branch A's live script already writes straight to its served path, which is why
Branch A's checkpoint updated correctly and the others would not have.

## Fix

Outputs are now `out_dir/branch_b/host_wdt.pt` and
`out_dir/deepop/cwa_forecast_decoder.pt`, matching the adapter. Any existing
checkpoint is copied to `<stem>.superseded-<UTC>.pt` **before training starts** —
both trainers save on every improving epoch, so a backup taken later would
capture a partly-retrained model rather than the one being replaced.

`tests/test_retrain_writes_where_serving_reads.py` (5 tests) compares the path
literals in the two source files directly, since that is where the disagreement
lives:

- the adapter declares all three served checkpoints
- the retrain writes to `branch_b/host_wdt.pt` and `deepop/cwa_forecast_decoder.pt`
- no write-only `.canonical-tgne.pt` name comes back (comments excluded, since
  the explanatory comment legitimately names it)
- the backup happens before the first trainer call
- Branch A's served path still matches

## Also corrected

`claude_latest_analysis/16_downstream_retrain_runbook.md` named
`bita_bigru_transformer-unified_v5.pth`, which does not exist — the encoder is
`bita_bigru_transformer-unified_final.pth`. It also passed `--spill-dir` twice.
Both fixed, and the section now states what the contract conflict means and
when it clears.

## Chaining

`scripts/chain_downstream_retrain.sh` waits for `branch-a-retrain.service` to go
inactive, refuses to start if `logs/branch_a_full.out` has no `saved=` line (the
last thing a completed run prints, after the held-out test and the credibility
stamp), and otherwise launches the downstream retrain under
`MemoryMax=17G / MemorySwapMax=0`.

They are chained rather than run together deliberately: Branch A holds ~7.6 GiB
for its trajectory store and the downstream run builds a comparable one, which
does not fit alongside it on a 22 GiB machine.

Running as `chain-downstream.service` since 21:32:51.
