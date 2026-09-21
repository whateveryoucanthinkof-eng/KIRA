# MASTER STATE — pipeline remediation loop

**Purpose:** this file is the context-recovery document. If the session is compacted or lost,
read *this file first*; it is written to be sufficient on its own. Every other report is
referenced from here.

**Last updated:** 2026-09-21
**Branch:** `testing-prod`
**Standing instruction from the user:** *perfect the pipeline and the model code; do not stop at
finding bugs — fix them; train, validate, iterate in a loop; the corrupted data is being
re-downloaded, so do not block on it; keep reports current so context loss costs nothing.*

---

## 0. The one-paragraph situation

Four models (TGNE encoder -> Branch A risk/technique/stage -> Branch B world model -> DeepOP
ATT&CK sequence decoder) train from three corpora (CIC-2018, CIC-2017, CTU-13) through a
unified adapter layer. The data on disk is partly damaged, but **the damage is upstream in the
published distributions** (proved in report 09) — re-downloading the CSVs changes nothing, so
every CSV defect must be handled in code. The user is separately re-downloading the 266 corrupt
PCAP files, which *is* a real fix. Our job until that lands: make the pipeline and the models
correct, and prove it.

---

## 1. Machine limits — respect these, they have frozen the laptop twice

| Resource | Value | Rule |
|---|---|---|
| RAM | 22 GiB total, ~12 GiB available | Cap heavy jobs at **17-18 GiB** under `systemd-run`. 19 GiB starved the desktop and froze it. |
| Swap | 8 GiB **zram only** | zram means OOM presents as *livelock*, not a clean kill. Always set `MemorySwapMax=0` on training units. |
| Disk | 952 GB, **79 GB free (92% used)** | Tight. The memmap spill (`.spill/`) and the 29 GB PCAP re-download both land here. Check before large writes. |
| GPU | RTX 4060 Laptop, 8 GiB | Currently 660 MiB used. Batch sizes are nowhere near the limit. |
| CPU | 16 cores | |

Launch pattern that works:
```
systemd-run --user --unit=<name> -p MemoryMax=17G -p MemorySwapMax=0 \
  --working-directory=<repo> \
  /var/home/samito/.pyenv/versions/3.12.14/bin/python <script> ...
```
Use the **absolute pyenv python path** — `bash -c python3` resolves to `/usr/bin/python3`, which
has no torch (this already cost one failed run).

**Never write outputs to `/tmp`** — it is tmpfs and was wiped by a reboot, losing completed
Branch B and DeepOP checkpoints. Everything goes to `saved_models/`.

---

## 2. The temporal contract (single source of truth)

`data_unification/temporal_contract.py` — v4:
```
window_seconds = 2.0 , history_steps = 15 , forecast_steps = 5 , STATE_DIM = 27
```
Feature layout: 12-D TGNE embedding + 15 host attributes = 27.

**Every model, checkpoint and dataset builder must agree with this.** Report 10 found four
places that did not; two are fixed, two remain (see §4).

---

## 3. Data reality — what the corpora actually are (from report 09)

| Corpus | Rows / files | Verdict |
|---|---|---|
| CIC-2018 CSV | 16,233,002 rows, 10 files | Complete vs. published total. 7 files capped at 2^20 **upstream**; 9 of 10 have **no IP columns**; 12-hour clock; 14 rows dated 1970 |
| CIC-2018 PCAP | 4,457 files, 600.7 GB | 4,191 clean; **266 corrupt, 28.8 GB unrecoverable** — user is re-downloading. 262 of 266 come from one capture agent (`capDESKTOP-AN3U28N`) |
| CTU-13 | 19,976,700 rows, 13 scenarios | **Clean.** 2.2261% botnet overall; 55.51% excluding Background |
| CIC-2017 5-tuple | 2,830,743 real rows, 8 files | Usable. 288,602 blank padding rows upstream; latin-1; 12-hour clock |
| CIC-2017 anonymized | 8 files | **Unusable** — 79 cols, no IP/port/protocol/timestamp. Keep only as a row-count oracle |

### 3.1 The 12-hour clock (verified independently, this session)

Hour histogram over the `Timestamp` column of **all ten** CIC-2018 files: hours **13-23 and 00
appear zero times**. Present hours are 01-05 and 08-12. A 12-hour dial uses `12`, never `00`,
so the absence of `00` is the confirming signature. `01:00:00` means **13:00:00**.

Repair rule: `if hour <= 7: hour += 12`. Hours 06/07 are near-empty except `tue_20`
(6,278 + 3,100 rows), which the rule maps to 18:00/19:00.

Consequence if unrepaired: afternoon traffic sorts *before* morning, so every host trajectory
carries one 12-hour backward discontinuity in the middle. Window *contents* stay correct (the
mapping is injective on this corpus — no real time collides with another), but ordering, time
deltas and any PCAP<->CSV label alignment are wrong for the afternoon half.

CIC-2017 has the identical defect.

---

## 4. Fix ledger — the actual work

Status: DONE = applied and verified. OPEN = not yet fixed.

| # | Defect | File | Status |
|---|---|---|---|
| 1 | `max_context_len` hardcoded 10 vs contract history 15 — Branch B never saw its 5 oldest steps | `branch_b_world_model/rollout_encoder_decoder.py` | **DONE** |
| 2 | DeepOP checkpoint omitted `history_steps` -> `int(None)` -> would be refused even after a clean retrain | `scripts/retrain_future_models_live.py:241` | **DONE** |
| 3 | `UNKNOWN` vs `Unknown` case mismatch -> unresolved labels silently graded Benign | `branch_a_gnn_lstm/sequence_dataset.py` | **DONE** |
| 4 | Nowcast bug — target reused the last *input* window | `branch_a_gnn_lstm/sequence_dataset.py:102` | **DONE** |
| 5 | Focal loss applied binary alpha to 6 classes -> collapse (macro F1 0.158) | `bita/train.py` | **DONE** |
| 6 | Timestamps 1000x too small (pandas returns `datetime64[us]`, adapters divided by 1e9) | `data_unification/time_utils.py` | **DONE** |
| 7 | CIC-2017 tz-aware cast raised; bare `except` zeroed every timestamp | same | **DONE** |
| 8 | CTU-13 label match `== "benign"` hit 0 of 19,976,700 rows -> reported 100% attack | `ctu13_adapter.py` | **DONE** |
| 9 | Blank padding rows -> 288,602 records at ts -9.2e9 | `cic2017_adapter.py` | **DONE** |
| 10 | **12-hour clock** — afternoon sorts before morning, corpus-wide | all three adapters | **OPEN** |
| 11 | Protocol-0 / port-0 junk rows and 14 rows dated 1970 | adapters | **OPEN** |
| 12 | `temporal_config.py:24-26` is a **second contract** (history=5, horizon=8) and it seeds the serving adapter | `data_unification/temporal_config.py` | **OPEN** |
| 13 | T=4 / K=4 defaults contradict the contract | `train_branch_b.py:48,81`; `train_cwa_decoder.py:49,113,332` | **OPEN** |
| 14 | Contract check writes to a log string; DeepOP never gets `_adopt_contract` | `control_backend/model_adapter.py:184,188-202` | **OPEN** |
| 15 | `host_attributes.py` — 15 names, **zero** match the computed values; no TCP flag computed anywhere | `host_attributes.py:6-22` | **OPEN** |
| 16 | `splits.lock.json` has **zero consumers**; five independent splits still exist | all trainers | **OPEN** |
| 17 | 3 of 6 checkpoints fail `validate_checkpoint` — and they are exactly the three the live adapter serves | `saved_models/` | **OPEN** |
| 18 | **TGNE inductive AUC 0.5043** (transductive 0.9981) — memorizes seen hosts, chance on unseen | `bita/` | **OPEN** |

Items 15 and 18 are the scientifically serious ones. 15 mislabels every feature attribution the
dashboard shows; 18 means the encoder may not generalize to a host it has not met, which is the
whole point of a forecasting deployment.

---

## 5. What is frozen and must not drift

`data_unification/splits.lock.json` — capture-level split, stratified by attack fraction 4:1:1:

| Corpus | train / val / test |
|---|---|
| CIC-2018 | 8 / 1 / 1 |
| CIC-2017 | 6 / 1 / 1 |
| CTU-13 | 9 / 2 / 2 |
| PCAP | 8 / 1 / 1 |

The lock exists and is correct. **Nothing reads it yet** (item 16). Until trainers consume it,
each script invents its own split and the lock is documentation, not enforcement.

---

## 6. Verification standard used throughout

- Bit-exact equivalence for refactors: identical SHA-256 over all fields and derived properties
  (40,000 records; 0 mismatches), RAM vs memmap vs old-vs-new on 10,629 snapshots.
- `pytest` suite: **118 tests**, green. Run after every phase.
- Credibility gate (`scripts/credibility_check.py`): label churn, base rate, host groups,
  persistence baseline, val size. Validated to flag all 8 pathologies on degenerate data and to
  pass healthy data.
- Numbers are self-checked at **peak**, not just final state (a builder once looked fine at
  151 B/snapshot finalized while hoarding tens of GB during the build).

---

## 7. Report index

| Report | Contents |
|---|---|
| `08_why_the_csv_path_cannot_benchmark.md` | Why three CSV-only runs were discarded |
| `09_data_integrity_audit.md` | Full corpus audit; the upstream-damage finding; acceptance checklist for new data |
| `10_contract_and_model_consistency.md` | Contract drift across models; 12-step remediation list |
| `11_full_retrain_execution_plan.md` | The retrain sequence to run when clean data lands |
| `12_bugs_found_and_fixed.md` | Running bug log with evidence |
| **`13_MASTER_STATE.md`** | **This file — start here** |

---

## 8. The loop

Repeat until the pipeline is defensible:

1. Pick the highest-severity OPEN item in §4.
2. Fix it in code. Not a note, not a TODO — an edit.
3. Run `pytest` (118 tests) plus any targeted check.
4. Update §4 status and append evidence to report 12.
5. When a fix touches training, retrain the affected stage and read the metrics.
6. Watch RAM/disk against §1 before launching anything heavy.
