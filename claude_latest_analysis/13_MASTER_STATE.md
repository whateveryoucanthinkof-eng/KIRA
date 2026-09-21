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
| RAM | 22 GiB total, ~12-17 GiB available | Cap heavy jobs at **16 GiB** (`MemoryHigh=14G`). 19 GiB froze the desktop twice; a stride-4 run reached **15.28 GiB** with only 2 GiB left system-wide and was stopped before it hit the cap. |
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

### Measured costs — use these, do not estimate

| Thing | Measured | How |
|---|---|---|
| `UnifiedFlowRecord` | **300 B** (was 479 before the interning fix) | current RSS delta over 200k records, after a 50k warm-up so pandas buffers are already allocated |
| Full corpus | 39M records (CIC-2018 16.2M + CIC-2017 2.8M + CTU-13 20.0M) | row counts from report 09 |
| stride 4 | ~9.7M records -> ~2.9 GB of records | 300 B x 9.7M |

**Do not measure memory with `ru_maxrss`.** It is *peak* RSS and will capture
pandas' chunk buffers, not your objects. It reported 5,103 B/record for a
record that actually costs 300 B — a 17x error that would have forced a
needless stride of 20. Read `/proc/self/statm` after `gc.collect()` instead.

**Never run two data-heavy jobs at once.** A smoke test launched while the
trainer was loading drove system-available memory to 0.4 GiB and the trainer
was OOM-killed at 14.3 GiB. The cgroup cap contained it -- the desktop
survived -- but the lesson stands: while a training unit is active, do only
work that costs megabytes (tests, edits, docs).

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

Status: DONE = applied, verified, and covered by a test. OPEN = not yet fixed.

### Fixed this session

| # | Defect | File | Evidence it was real |
|---|---|---|---|
| 1 | `max_context_len` hardcoded 10 vs contract history 15 | `branch_b_world_model/rollout_encoder_decoder.py` | Branch B never saw its 5 oldest history steps |
| 2 | DeepOP checkpoint omitted `history_steps` -> `int(None)` | `scripts/retrain_future_models_live.py:241` | Would be refused even after a clean retrain |
| 3 | `UNKNOWN` vs `Unknown` case mismatch | `branch_a_gnn_lstm/sequence_dataset.py` | Unresolved labels silently graded Benign |
| 4 | Nowcast bug — target reused the last *input* window | `branch_a_gnn_lstm/sequence_dataset.py:102` | Not a forecast at all |
| 5 | Focal loss applied binary alpha to 6 classes | `bita/train.py` | Collapse, macro F1 0.158 |
| 6 | Timestamps 1000x too small (`datetime64[us]`) | `data_unification/time_utils.py` | 12h day -> 43s; 21,600 windows -> 22 |
| 7 | CIC-2017 tz-aware cast raised; bare `except` zeroed all | same | Every CIC-2017 timestamp was 0 |
| 8 | CTU-13 label match `== "benign"` hit 0 of 19,976,700 rows | `ctu13_adapter.py` | Reported 100% attack; true rate 2.2261% |
| 9 | Blank padding rows -> 288,602 records at ts -9.2e9 | `cic2017_adapter.py` | Window grid stretched to 1677 AD |
| 10 | **12-hour clock, corpus-wide** | all three adapters | See §3.1. wed_29: hours {1-5,8-12} -> {8-17}; span 12.00h -> 9.47h |
| 11 | Protocol-0/port-0 TSO failures and epoch-1970 rows | `data_unification/row_guards.py` (new) | thu_22: exactly 9 rejected, matching the audit's independent count |
| 12 | `temporal_config.py` declared a **second contract** | `data_unification/temporal_config.py` | history 5 / forecast 8, and it seeded the serving adapter |
| 13 | T=4 / K=4 literals | `train_branch_b.py`, `train_cwa_decoder.py` | Contradicted the contract |
| 14 | DeepOP never got `_adopt_contract`; mismatch went to a log string | `control_backend/model_adapter.py` | Now refuses to serve, with an env escape hatch |
| 15 | `host_attributes.py` — 15 names, **zero** matched | `data_unification/host_attributes.py` | 4 named TCP flags that are never computed |
| 16 | `splits.lock.json` had **zero consumers** | `data_unification/split_manager.py` | Its own split contradicted the lock (Wednesday train vs test) |
| 17 | Branch B checkpoint recorded no contract at all | `train_branch_b.py` | `validate_checkpoint` could never accept it |
| 18 | Assembler defaulted to K=4, 60s windows, history 10/5 | `correlation/trajectory_assembler.py` | Truncated Branch A and B at serving time |
| 19 | `ScientificSplitManager` returned **ZERO records** for every split | `data_unification/split_manager.py` | Every path wrong, incl. literal `C:\SIH_DATA\...`; guarded by `os.path.exists` so nothing raised. Standalone Branch B/DeepOP trained on nothing |
| 20 | Prefix sampling was label-biased | `split_manager.py`, `bita/train.py` | Frozen val split measured **0% attack** at 2000 rows/capture |
| 21 | `stride` applied after loading, so peak memory was the full corpus | `bita/train.py:274` | Striding saved nothing at the peak |
| 22 | Hits@K / MRR were O(N x P x C) Python loops | `bita/evaluation/eval_edge_prediction_with_categories.py` | Eval took longer than training: 12+ min after a 4 min epoch. 288x / 72x faster |
| 23 | Node features were **all zero** -> inductive learning impossible | `bita/train.py`, `data_unification/ip_features.py` (new) | Transductive AUC 0.9981 vs inductive 0.5043 (chance) |
| 24 | Trainers invented 5 splits; none matched the lock | `retrain_branch_a_live.py`, `retrain_future_models_live.py`, `split_policy.py` | Branch A sliced a **sorted** list 70/15/15; 3 more 80/20 slices |
| 25 | PCAP test day would `KeyError` or be folded into train | `retrain_future_models_live.py` | Lock assigns `thu_1_pcap` to test; only train/val stores existed |
| 26 | **String interning never fired for IPs** | `data_unification/unified_schema.py` | Guard was `type(x) is str`; adapters emit `numpy.str_`. 479 -> 300 B/record |
| 27 | Edge features copied three times at peak | `bita/train.py` | list of N small arrays -> np.stack -> np.vstack |
| 28 | **TGNE trained on the held-out captures** | `bita/train.py` | Its category head uses labels; val/test captures leaked into the encoder, invisible to any branch-level check |
| 29 | Loader held ~8M record objects -> OOM-killed at 14.3 GiB | `bita/train.py` | Now streams into growable numpy columns; nothing in the output needs the objects |
| 30 | `NameError: train_records` on every non-credible run | `scripts/retrain_branch_a_live.py:255` | The credibility verdict was replaced by a traceback |

### Still open

| # | Defect | Why it matters |
|---|---|---|
| A | **TGNE inductive AUC 0.5043** (transductive 0.9981) | The encoder is at CHANCE on unseen hosts. A forecasting deployment meets new hosts constantly. **Highest-value open item.** |
| B | **TGNE category head collapsed** | CatAcc frozen at 0.3327 -> 0.3326 across two epochs, inductive 0.4342 -> 0.4340. Four-decimal-stable accuracy means a constant prediction. |
| C | 3 served checkpoints fail `validate_checkpoint` | They carry history 5 / forecast 8 — the deleted second contract's values. Tracked by 3 strict-xfail tests. Resolves on retrain. |
| D | Branch A / B / DeepOP not yet retrained under the fixed pipeline | Everything above changes their inputs. |
| E | PCAP bridge not wired into the live retrain scripts | `pcap_bridge.py` exists and works; `--pcap-dir` is not plumbed through. |
| F | 266 corrupt PCAP files (28.8 GB) | User is re-downloading. 262 of 266 come from one capture agent (`capDESKTOP-AN3U28N`). |
| G | `NeighborFinder` is handed an `adj_list` of Python **tuples** | 2 tuples per edge at ~156 B each = ~2.4 GB of transient peak at 7.8M edges, allocated right after loading. Not yet fixed; it is the remaining obstacle to full density. |

### A consequence of #10 worth stating separately

`bita/train.py:357` splits temporally at the 70/85 timestamp quantiles. With the
12-hour clock unrepaired, afternoon traffic sorted *before* morning, so that
"temporal" boundary did not separate past from future — **the TGNE train/val
split was leaking**. Every TGNE run before 2026-09-21 11:02 is affected.

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

## 8. Current training run

Unit `tgne-retrain`, log `logs/tgne_fixed_retrain.out`.

```
--stride 4 --train_splits train --n_epoch 30 --patience 5 --gpu 0
--data_name unified_v4
```

First run with the clock repair, IP node features, working interning, the
columnar loader, the frozen train-only split and the fast evaluation.

**Launch discipline learned the hard way:** run nothing else heavy while this
is loading. A concurrent smoke test drove system-available to 0.4 GiB and the
unit was OOM-killed at 14.3 GiB. The cgroup cap contained it -- the desktop did
not freeze -- which is exactly what the cap is for.

Archived logs, in order, none of them valid baselines:

| log | why it is not a baseline |
|---|---|
| `tgne_retrain_PRE_CLOCKFIX.out` | 12-hour clock unrepaired -> leaking temporal split |
| `tgne_retrain_PRE_NODEFEAT.out` | node features all zero |
| `tgne_retrain_PRE_LEAKFIX.out` | trained on val/test captures |
| `tgne_retrain_OOM_ATTEMPT.out`, `tgne_retrain_OOM2.out` | OOM-killed during load |

The collapse signature from the pre-clockfix run is the diagnostic record for
open items A and B:

| | Epoch 00 | Epoch 01 |
|---|---|---|
| Val AUC | 0.9981 | 0.9937 |
| Val CatAcc | 0.3327 | 0.3326 |
| Inductive Val AUC | 0.5043 | 0.5026 |
| Inductive CatAcc | 0.4342 | 0.4340 |

CatAcc stable to four decimals across two epochs is a constant prediction, not
learning.

---

## 9. The loop

Repeat until the pipeline is defensible:

1. Pick the highest-severity OPEN item in §4.
2. Fix it in code. Not a note, not a TODO — an edit.
3. Run `pytest` (223 passing, 3 xfailed as of this writing) plus a targeted check.
4. Update §4 status and append evidence to report 12.
5. When a fix touches training, retrain the affected stage and read the metrics.
6. Watch RAM/disk against §1 before launching anything heavy.

Never mark an item DONE without a test or a measurement attached to it.
