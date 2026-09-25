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

---

## Addendum — finding the real OOM

The TGNE run was OOM-killed three times. Two fixes were applied before the
cause was actually located, and it is worth recording that they were not it.

| attempt | hypothesis | outcome |
|---|---|---|
| 1 | the ~8M-record Python list | real (479 → 300 B/record via the interning fix) but **not the OOM** |
| 2 | three copies of the edge-feature block | real (list → `np.stack` → `np.vstack` collapsed to one preallocated array) but **not the OOM** |
| 3 | the 12-hour-clock detection pre-pass | **this was it** |

What isolated it was comparing **peak against final**: peak 13.75 GiB, final
1.01 GiB. A large gap means the consumer is transient, so measuring the
accumulated structures was never going to find it. Measuring each component
alone:

| component | peak RSS |
|---|---|
| adapter streaming a whole file, every record discarded | 0.22 GiB, flat |
| final loader state (columns + edge features + node features) | 1.01 GiB |
| **`detect_12h_clock_in_file` on `tue_20` alone** | **13.91 GiB** |

`pd.read_csv(path, usecols=[ts], low_memory=False)` forces pandas to parse the
whole file as one block; the C tokenizer buffers every column before `usecols`
is applied. On 7.9M rows × 84 columns that is 13.91 GiB for one column of
timestamps. Chunked: **0.23 GiB**, identical verdict, same runtime. Full
CIC-2018 train loader: **13.75 → 1.16 GiB**, byte-identical output.

Chunking then regressed two CIC-2017 files from True to False. Mixed dtypes
(guaranteed by the blank padding rows) send pandas down its DtypeWarning path,
where `usecols` + `chunksize` hit a pandas bug — `_concatenate_chunks` indexes
`column_names` by the original column position while that list holds only the
selected column — raising IndexError. A bare `except` turned that into "no
hours" → "not a 12-hour clock" → **file left unrepaired**, the same
silent-failure shape as the bug the module exists to fix. Now `dtype=str`
(skipping inference entirely) plus an ERROR log on any detection failure.

All 18 verdicts (10 CIC-2018 + 8 CIC-2017) confirmed True afterwards.

### Also in this addendum

- `~/Downloads/DATA` audited: **byte-identical** to the training copy (MD5 match
  on all ten, independently spot-checked on three), `sum(wc -l − 1)` =
  **16,233,002**, the published total exactly. The 2^20 cap is upstream, as
  report 09 concluded. Its `logs/` holds 1.8 GB of Windows `.evtx` victim-host
  logs (451 IPs) which cannot restore per-flow IPs — no join key, victim side
  only, and a strict subset of the 33,176 IPs `tue_20` already has. Full detail
  in [`15_downloads_csv_audit.md`](15_downloads_csv_audit.md).
- `scripts/retrain_branch_a_live.py` smoke-tested end to end. The credibility
  gate fires correctly: it refused a run where model accuracy 0.7221 did not
  beat the 0.9666 persistence baseline.

---

## Addendum 2 — full density, and what it exposed

Removing the stride was not just a volume change. It changed which defects were
visible.

### The eleven places data was being thinned

The user caught `--stride 4` on the TGNE run. Auditing for it found ten more,
three of them introduced the same day by me:

| where | mechanism | discarded |
|---|---|---:|
| `split_manager` train/val/test *(mine)* | `stride=20` | **95%** |
| TGNE invocation *(mine)* | `--stride 4` | **75%** |
| `retrain_branch_a_live.py` | `--rows-per-file 1000` | most of a capture |
| `retrain_future_models_live.py` | `--rows-per-file 1000` | most of a capture |
| `build_tgne_live_dataset.py` | `--rows-per-file 1000` | most of a capture |
| `train_branch_b.py` | `max_per_source` 600 / 200 | most of a capture |
| `train_cwa_decoder.py` | `max_per_source 1000` | most of a capture |
| `flow_table.snapshot_flows` | `max_flows=256` | flows past the 256th |
| `pcap_bridge` ×2, `pcap_adapter` ×2 | `max_flows=256` | flows past the 256th |
| `retrain_future_models_live.py` | `max_packets_per_host=20000` | rest of the capture |

`snapshot_flows(max_flows=256)` was the worst of them: silent structural
truncation inside PCAP ingestion, with nothing in any log to indicate flows had
been dropped.

`data_unification/density.py` now enforces the rule. `require_full_density()`
runs at the top of all three trainers and raises on any active cap or stride;
`CYBERWORLD_ALLOW_SUBSAMPLING=1` is the only bypass and logs
"SUBSAMPLED RUN, NOT A RESULT".

### What made full density possible

| fix | before | after |
|---|---|---|
| clock detection pre-pass | 13.91 GiB | 0.23 GiB |
| neighbour finder (`adj_list` of Python tuples → CSR) | 293 B/edge = **9.3 GiB** | 48 B/edge = **1.5 GiB** |
| record interning (`numpy.str_` skipped the guard) | 479 B/record | 300 B/record |
| edge features (list → stack → vstack) | 3 copies | 1 preallocated array |

Measured on the real run: **34,152,542 records, 2,186,177 hosts, peak ~9.6 GiB
against an 18 GiB ceiling.** No memmap required. Load 12.7 min, epoch ~13 min
(92,132 batches at ~114 batch/s).

### Two defects only full density could reveal

**1. The temporal split was a split by corpus.** CTU-13 is 2011, CIC-2017 is
2017, CIC-2018 is 2018 — disjoint ranges. A global
`np.quantile(ts, [.70, .85])` trained on 2011 Czech university botnet traffic
and tested on 2018 AWS enterprise traffic. Now cut per corpus:

```
CIC-2017   1,691,081 edges   val cut 2017-07-07
CIC-2018  14,357,106 edges   val cut 2018-02-20
CTU-13    18,104,355 edges   val cut 2011-08-16
```

**2. The focal alpha collapsed four of five classes onto the clip floor.**

```
{Benign: 0.2, C2: 0.2, Impact: 0.2, InitialAccess: 0.2, Recon: 4.911}
```

Pinned classes are weighted identically, so the loss was doing no balancing at
all between the four classes carrying the data. The weights were normalised by
the arithmetic mean; they are multiplicative, and Recon is ~5 orders of
magnitude rarer than Benign, so that one outlier set the scale for everyone.
Geometric-mean centring gives
`{0.2, 0.381, 0.466, 0.851, 5.0}` — four distinct weights where there was one.

This function has now failed twice in opposite directions (plain 1/frequency
previously drove the majority classes to 0.0), both times because the
normaliser was a statistic one extreme class controlled. It now logs the class
counts and warns when more than half the classes pin at a bound.

### Reading the diagnostics

| observation | meaning |
|---|---|
| focal weight exactly **1.0** | that class had **zero** training samples |
| more than half the classes at a clip bound | the loss is not balancing them |
| transductive AUC ~0.99 alone | says nothing; it was true while the model was a lookup table |
| inductive AUC ~0.50 | chance — the encoder cannot generalise to unseen hosts |
| CatAcc stable to 4 decimals | a constant prediction, not learning |
