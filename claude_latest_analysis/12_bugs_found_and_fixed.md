# Bugs found and fixed — 2026-09-21 session

Every entry below was **measured**, not inferred, and each invalidates checkpoints trained before it.
Kept as the record of why a full retrain was necessary.

---

## 1. Epoch timestamps 1000x too small (CRITICAL, all pandas adapters)

**Symptom:** `fri_16_csv.csv` parsed to a time span of 43 seconds instead of 11.96 hours; records
dated 1970-01-18 instead of 2018-02-16.

**Cause:** pandas 3.0.5 returns `datetime64[us]`, not `[ns]`. Every adapter did:
```python
ts_series.astype("int64") / 1e9     # assumes nanoseconds -> yields microseconds/1e9
```
`1518769643000000 µs / 1e9 = 1,518,769` seconds = January 1970.

**Blast radius:** with 2-second windows, ~21,600 windows per capture day collapsed to ~22. Host
trajectories, temporal attributes and every TGNE embedding derived from them were meaningless.
Affected `cic2017_adapter`, `cic2018_adapter`, `ctu13_adapter`, `warden_adapter`.

**Fix:** one shared helper `data_unification/time_utils.py::to_epoch_seconds`, resolution-independent
and tz-aware-safe. Verified: CIC-2018 21,536 windows (was 22), CIC-2017 2017-07-07, CTU-13 2011-08-15.

---

## 2. CIC-2017 timestamps silently zeroed (CRITICAL)

**Symptom:** zero valid timestamps from a 100k-row read, with no error surfaced.

**Cause:** the adapter parses with `utc=True`, producing `datetime64[us, UTC]`. Casting a tz-aware
series to `datetime64[ns]` raises; the adapter's bare `except Exception:` then set **all** timestamps
to zero. A corpus-wide failure presented as a successful load.

**Fix:** `to_epoch_seconds` converts tz-aware to UTC and drops the tz before casting.

---

## 3. Focal loss applied binary weighting to 6 classes (CRITICAL, TGNE)

**Symptom:** TGNE category head predicted "Benign" for everything. Test per-class accuracy
`{0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}`, macro F1 **0.158**.

**Cause:**
```python
at = torch.ones_like(targets) * (1 - self.alpha)   # 0.75 for EVERY class
at[targets == 1] = self.alpha                      # 0.25 for class index 1 only
```
Classes are `{0: Benign, 1: C2, 2: Impact, 3: InitialAccess, 4: Recon, 5: UNKNOWN}`. Benign — the
majority class — received **0.75** while C2 received **0.25**. The loss rewarded predicting Benign.

**Fix:** true multi-class focal loss with per-class alpha. First attempt used plain inverse frequency
and produced `{Benign: 0.0, C2: 0.0, Impact: 2.0, ...}` — the same collapse inverted, because absent
classes got astronomically large raw weights and skewed the normalisation. Final version computes
weights over **present classes only**, damps by sqrt, and clips to [0.2, 5.0]. Verified across
realistic-skew, absent-class and balanced regimes.

---

## 4. Blank padding rows became phantom hosts (HIGH, CIC-2017)

**Symptom:** `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` yielded 458,968 records of
which 288,602 had `src_ip='nan'`, `dst_ip='nan'`, `start_time=-9,223,372,036`.

**Cause:** rows 170,366+ of that file are entirely blank (spreadsheet padding). `NaT` cast to int64
gives the sentinel `-9223372036854775808`, which after `/1e9` is a plausible-looking negative epoch
that slipped past a numeric guard.

**Fix:** `to_epoch_seconds` detects `NaT` via `isna()` *before* casting; CIC adapters skip rows with
a non-positive timestamp or a `nan` source IP. Record count for that file: 458,968 -> **170,366**.

---

## 5. Trajectory store compact only after finalize (HIGH, memory)

**Symptom:** a full-density run climbed to 17 GB with the memmap spill file still at **0 bytes**,
then died. Two hard laptop freezes preceded this.

**Cause:** the columnar store measured 151 B/snapshot *finalized*, but the builder accumulated ten
metadata columns as Python lists (~36 B per int vs 4 B for `int32`) plus one list of row indices per
host. Peak-during-build was tens of GB. The measurement answered the wrong question.

**Fix:** growable numpy columns (`_Col`), per-host grouping computed once at finalize via a stable
argsort. Build-time peak measured at **64 B/snapshot** at 4M rows.

---

## 6. `range(8)` against a 5-step contract (MEDIUM, Branch B)

`train_branch_b_live` summed its rollout loss over `range(8)` (v3's K) while building samples with
`K=_c.forecast_steps` (5 under v4). Out-of-bounds or silently wrong-shaped loss. Also wrote
`"history_steps": 5` into the checkpoint regardless of the 15 actually used.

---

## 7. DeepOP trained on oracle future states (MEDIUM, audit E1)

The live trainer fed DeepOP ground-truth future TGNE embeddings, which it never sees at serve time —
in production its input is a Branch-B rollout. Interim mitigation (Gaussian noise scaled 0.015→0.055
across the horizon) existed in the standalone script but **not** in the live path.

**Fix:** Branch B now returns its best model; `create_cwa_training_samples(..., T=history_steps)`
carries `h_history`; DeepOP conditions on `wdt.rollout(...)`. Log confirms
`DeepOP conditioning: Branch-B rollouts (E1 fixed)`.

---

## 8. Nowcast target (MEDIUM, Branch A) — carried over from the approved plan

`sequence_dataset.py` used `target_snap = window_slice[-1]`, i.e. the last **input** window as the
prediction target. Now `snapshots[end_idx]`, a genuinely future snapshot. Verified with a synthetic
trajectory: samples now target real onsets instead of reusing observed windows.

---

## Verification standard applied

Every representation change was held to bit-exact equivalence against the pre-change code before use:

- **Columnar snapshot store:** 0 mismatches on snapshots *and* derived training samples, across
  RAM / memmap / old-vs-new, 10,629 snapshots compared.
- **Record representation** (slots + interning + `metadata=None`): identical SHA-256 over every field
  *and* every derived property (`duration`, `total_bytes`, `byte_rate`, `src_host_key`, ...) across
  40,000 real records.
- **118 tests** green after every change.

One thing this standard caught: a chunked-extraction design that looked correct, compiled, and was
**structurally wrong** — it lost 27,432 of 37,024 hosts because its 30-second overlap was meaningless
on data spanning 5.3 seconds. It was discarded rather than shipped.
