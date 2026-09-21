# Remediation session — 2026-09-21

**Mandate:** perfect the pipeline and the model code; fix, do not merely find;
iterate; do not block on the corrupted data, which is being re-downloaded.

**Result:** 17 defects fixed, each with a test or a measurement attached.
Test suite went from 118 to **256 passing** (+3 strict-xfail tracking the stale
checkpoints). Six commits on `testing-prod`.

Start here: [`13_MASTER_STATE.md`](13_MASTER_STATE.md) is the context-recovery
document. This file is the evidence log behind it.

---

## The three that would have invalidated results silently

### 1. Every CIC timestamp is a 12-hour clock with no AM/PM

Hours 13–23 appear **zero** times across all 16,233,002 CIC-2018 rows. Hour
`00` also appears zero times — and that is the confirming signature, not a
coincidence: a 12-hour dial writes noon as `12`, never `00`. The present hours
are 01–05 and 08–12, i.e. a working day where the afternoon was written on a
12-hour face.

Unrepaired, every afternoon flow parses twelve hours *before* the morning.

Measured on `wed_29` after the repair:

| | before | after |
|---|---|---|
| hours present | {1,2,3,4,5,8,9,10,11,12} | {8..17} |
| sorted span | **12.00 h** (the artifact) | 9.47 h |
| backward jumps > 6 h | 77,673 | 18,558 |

The residual backward jumps are CIC's non-chronological row order, which the
window bucketing does not depend on.

**The part that matters most:** `bita/train.py:357` splits train/val/test at the
70/85 **timestamp quantiles**. With afternoon sorting before morning, that
boundary did not separate past from future — the TGNE split was leaking. Every
TGNE run before this session is invalid on that ground alone.

Detection refuses to guess. Any hour in 13–23 or hour 00 proves a 24-hour clock;
no hour in 01–07 means nothing to repair; an afternoon-only file is ambiguous
and is refused. That last rule is why CIC-2017 detection pools a whole capture
*day*: its Thursday and Friday afternoon files hold only hours 01–05 and are
individually undecidable, but pooled with their morning siblings the day spans
01–05 and 08–12 and the verdict is decisive. All 8 CIC-2017 files now resolve
(Friday over 3 files, Thursday over 2).

The repair is idempotent by construction: it maps 01–07 onto 13–19, which is
outside the shift range.

*26 tests.*

### 2. `ScientificSplitManager` returned zero records for every split

Every dataset path in it was unreachable on this machine, including a literal
Windows path `C:\SIH_DATA\dump\cic2018(csv+pcap+logs)raw\CSV`. Each lookup sat
behind an `if os.path.exists(...)` guard, so nothing ever raised:
`get_train_records()`, `get_val_records()` and `get_heldout_test_records()` all
returned `[]`.

**The standalone Branch B and DeepOP trainers were training on nothing, and
saying nothing about it.**

Its hardcoded assignment also contradicted `splits.lock.json` — CIC-2017
Wednesday in train where the lock says test, Friday-Morning in val where the
lock says train. One working path away from test leakage.

*10 tests.*

### 3. Node features were all zero, so inductive learning was impossible

`bita/train.py` built `node_features = np.zeros((total_nodes, 12))`.

In TGN a node's embedding is a function of its memory, its node features and
its neighbours. For a node never seen during training the memory is zero too,
so an unseen host carried **no signal whatsoever** and the link decoder could
only score it at chance. That is exactly what was measured:

```
transductive val AUC   0.9981
inductive   val AUC    0.5043      <- chance
```

The encoder had memorised the hosts it had seen and learned nothing
transferable — a lookup table, not a model. For a system whose purpose is
forecasting on a live network where unseen hosts arrive constantly, that is
the whole ballgame.

`data_unification/ip_features.py` supplies 12 intrinsic features per address.
The octets are the point: two hosts in the same /24 share their first three
octets, so an unseen host in a known subnet arrives close to its neighbours in
feature space. The structure is real here — CIC-2018 is 172.31.x.x, CIC-2017 is
192.168.10.x, external traffic is public.

Every value is a pure function of the IP string, so there is no label leakage
and no temporal leakage across the quantile split.

*13 tests.*

---

## Contract unification

`data_unification/temporal_config.py` declared a **second contract** — horizon
8, history 5 — against the authoritative `forecast_steps=5, history_steps=15`.
`control_backend/model_adapter.py` imported its serving defaults from there, so
the live path requested an 8-step rollout with 5 steps of history from models
trained for 5 and 15. Shapes are permissive enough that nothing raised.

It was not theoretical. The three checkpoints the adapter serves carry exactly
those phantom values:

| checkpoint | window | history | forecast |
|---|---|---|---|
| contract | 2.0 | **15** | **5** |
| branch_a | 2.0 | 5 | – |
| branch_b | 2.0 | 5 | 8 |
| deepop | 2.0 | – | 8 |

Everything now derives from `get_contract()`. `MACRO_BATCH` raises rather than
diverging — no model here is trained at 60 s granularity. Also fixed:

- `correlation/trajectory_assembler.py` defaulted to K=4, 60 s windows and
  history 10/5 while feeding the real checkpoints.
- `train_branch_b.py` / `train_cwa_decoder.py` had literal `T=4, K=4`.
- Branch B's checkpoint recorded **no contract at all**, so
  `validate_checkpoint` could never accept it.
- DeepOP was the only checkpoint never passed to `_adopt_contract`, though its
  horizon defines the served forecast length.
- The extractor was built from the *provisional* window before any checkpoint
  was read, so features were bucketed on a different grid than the models were
  trained on.
- A contract mismatch went to a **log string**. It now refuses to serve, with
  `CYBERWORLD_ALLOW_CONTRACT_MISMATCH=1` as an explicit, loud escape hatch.

---

## The frozen split finally has consumers

The lock was correct and **nothing read it**. Five independent splits existed:

| where | what it did |
|---|---|
| `retrain_branch_a_live.py` | sliced a **sorted** file list 70/15/15 — "train" meant "alphabetically first" |
| `retrain_future_models_live.py` | 80/20 slice of sorted CSVs |
| same, `load_pcap_records` | 80/20 slice of sorted PCAP days |
| same, `iter_pcap_day_records` | another 80/20 slice |
| `split_manager.py` | its own hardcoded assignment |

All now resolve through `split_policy.partition_paths`. Verified against the
real corpus: **17 train / 3 val / 3 test** for CIC-2018 + CTU-13, and all 10
PCAP day directories at **8/1/1** — including the two the corpus itself
misspells `_pacap`, and `wed_28_pcap`, whose CSV counterpart is confusingly
named `wed_29_csv.csv`.

One consequence had to be handled: the lock assigns `thu_1_pcap` to test, but
`_pcap_trajectories_per_day` built stores for train and val only. That would
have raised `KeyError` — or, had the split defaulted, folded a held-out capture
into training. It now builds all three and keeps test separate.

*12 tests.*

---

## Prefix sampling was quietly destroying the validation set

Passing `max_rows` to an adapter takes the **first** N rows, and these captures
are chronological: benign in the morning, attacks later.

Measured on the frozen val split at 2,000 rows per capture: **0% attack**. A
validation set with no positives at all — every early-stopping decision and
every threshold derived from it was meaningless, and nothing said so.

With `stride=200` over the same captures:

| split | records | attack | categories |
|---|---:|---:|---|
| train | 55,445 | 20.41% | Benign, Impact, InitialAccess, Recon, C2 |
| val | 7,393 | **5.59%** | Benign, InitialAccess, C2 |
| test | 9,650 | 31.39% | Benign, C2, Impact |

`records_for()` now warns loudly if a cap is requested with `stride=1`.

**Known and accepted:** the category *mixes* differ across splits — train has
Impact and Recon, val has neither, test has no InitialAccess. That is inherent
to capture-level splitting over few captures and is the honest price of zero
leakage, not something to shuffle away.

---

## Performance and memory

### Evaluation was slower than training

The eval phase took 12+ minutes after a 4-minute epoch. Three Python loops over
every evaluation sample. The worst, per-class MRR, ran
`np.where(ranks_c == idx)[0][0]` — a full O(N) scan of the ranking — once per
positive sample, inside a loop over classes: O(N × P × C).

That expression is just the inverse permutation of the argsort.

| | N=20,000 | N=100,000 |
|---|---|---|
| Hits@K + MRR | 0.241 s → 0.0008 s (**288×**) | 1.222 s → 0.0042 s (**293×**) |
| per-class MRR | 0.023 s → 0.0009 s (24×) | 0.469 s → 0.0066 s (**72×**) |

Per-class MRR is bit-identical. Hits@K and MRR differ only in tie handling,
unreachable with real softmax outputs. *21 tests pin the new code against the
original loops kept verbatim as the reference.*

### Interning never fired for the fields that mattered

`UnifiedFlowRecord`'s docstring calls interning "load-bearing at corpus scale".
The guard was `type(x) is str`, which is False for `numpy.str_` — precisely what
the pandas adapters produce for `src_ip` and `dst_ip`. So it silently skipped
the two highest-cardinality, highest-volume fields while appearing to work.

On `wed_29`, `raw_label` collapsed to **one** object across 5,000 records while
`src_ip` kept **5,000** objects for 250 unique values.

Fixed: **479 → 300 bytes per record** (37%), `src_ip` at 250 objects for 250
values.

Also removed three redundant copies of the edge-feature block (list of N small
arrays → `np.stack` → `np.vstack`), now one preallocated array.

*8 tests.*

### A measurement error, recorded honestly

The first measurement of record size reported **5,103 bytes**. That used
`ru_maxrss`, which is *peak* RSS and had captured pandas' chunk buffers rather
than the records. Re-measured with current RSS after `gc.collect()`: **479**.

A 17× error. Acting on it would have forced a stride of 20 and needlessly
diluted the training data — the opposite of the mandate. No decision was taken
on the wrong number. **Do not use `ru_maxrss` for object sizing.**

---

## Still open

| item | status |
|---|---|
| TGNE inductive AUC | fix applied (node features); awaiting the current run |
| TGNE category head collapse | CatAcc frozen at 0.3327 → 0.3326 across two epochs. Likely the same root cause; awaiting data |
| 3 checkpoints fail `validate_checkpoint` | resolves on retrain; 3 strict-xfail tests track it |
| Branch A / B / DeepOP retrain | blocked on TGNE finishing |
| 266 corrupt PCAP files (28.8 GB) | user re-downloading |

---

## Commits

```
418c2d4  fix(contract): one contract, and host attribute names that describe real values
da05d6f  fix(data): repair the 12-hour clock and drop upstream-broken rows
0c988fa  fix(splits): the frozen lock is now the only split, and it resolves real data
b87deca  perf(eval): vectorise Hits@K and MRR; they were stalling every epoch
1a07775  fix(ingest): stride during ingestion, not after; document full state
a3ca4ba  fix(tgne): give nodes real features; all-zero features made inductive learning impossible
a6d10cf  fix(splits): every trainer now reads the frozen lock; five splits become one
1b9a6c9  perf(memory): interning never fired for IPs; 479 -> 300 bytes per record
```
