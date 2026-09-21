# Session log — while you were away

Short entries. Newest at the bottom. Started ~15:40.

---

### 15:41 — Credibility verdict now saved with the checkpoint
**Was:** the gate computed a verdict, printed it, threw it away. A model its
own training run judged unsound could be served with nothing recording that.
**Now:** checkpoint carries `{credible, problems, stats}`; the serving adapter
logs ERROR when it loads an unsound one, and flags a checkpoint with no
verdict at all. Gate stays advisory (a demo may want a weak model) — you just
can't serve one unknowingly. *6 tests.*

### 15:48 — Experiment A finished, and it found something big
Epoch 0: Val AUC 0.8168, **Inductive Val AUC 0.7961**, 1022 s/epoch.
Per-class: `{Benign 0.2, C2 0.0, Impact 0.0, InitialAccess 1.0, Recon 0.0}`
→ **the category head predicts InitialAccess for everything.**
Also means the earlier "good" CatAcc 0.8351 was *also* a collapse — onto
Benign, which is ~80% of the data. Both constant predictors.

### 15:50 — Root cause: the head couldn't see the flow *(biggest fix today)*
```python
combined = src_emb + dst_emb          # 12-D
logits   = nn.Linear(12, n_classes)(combined)
```
The label (Benign/C2/Impact/InitialAccess/Recon) is a property of the **flow**
— ports, bytes, duration, protocol. The head only saw node embeddings, so a
benign flow and an attack flow *between the same host pair* were identical
inputs. Unseparable in principle → collapse. Addition also erased direction
(A→B == B→A).
**Now:** `[src ; dst ; edge_features] → Linear → ReLU → Dropout → Linear`.
Verified: logits change when only the edge changes. *7 tests, two of which
prove the old design had the flaw so the new ones can't pass vacuously.*

**Consequence:** Sep-10 checkpoints can't load (head shape changed). They
already failed `validate_checkpoint`. Loader now raises
`StaleEncoderArchitecture` with a plain explanation instead of a state_dict
diff; 3 adapter test modules skip with a stated reason until retrain.

### 15:52 — Final TGNE run launched
Edge-aware head + per-capture split + full density (34,152,542 records) +
parallel ingest + vectorised sampler. 30 epochs, patience 5, seed 0.
`logs/tgne_final.out`

**Dropped the strict perf A/B.** A was a baseline for an architecture we just
replaced, so it's no longer valid for comparison. Both speedups are already
proven lossless by 25 + 6 equivalence tests; I'll read wall-clock off this run
instead. Recorded the reasoning in `17_perf_experiments.md`.

### What to check at epoch 0
**Per-class accuracy with more than one class above zero.** Aggregate CatAcc
proves nothing — 0.8351 and 0.2115 were both collapses.

### 15:56 — Parallel ingest confirmed on the real corpus
**231.9 s** vs **12.5 min** serial for the same 34,152,542 records → **3.23x**,
matching the 3.14x measured in isolation. ~8.6 min saved per run, and it
applies to every downstream retrain too. Output is bit-identical (6 tests).

### 16:00 — Reviewed Branch A. Architecture is sound; one thing to flag
Branch A's input is 27-D = 12 TGNE embedding **+ 15 behavioural attributes**
(bytes, packets, ports, ratios), so unlike the encoder's category head it does
see host behaviour. 2-layer LSTM + temporal attention, three heads with proper
hidden layers. **No equivalent defect.** Target construction is also correct
(target strictly after the input window — the nowcast bug is fixed).

**Flag for you, not a bug — `risk_score` is derived from the label.**
`multi_dataset_stream.py:345-347`:
```python
risk = min(1.0, max(0.20, base_sev + 0.04*atk_density + 0.04*vol_scale))  # attack
risk = 0.0                                                                # benign
```
Benign is *exactly* 0.0; attack is *at least* 0.20. No sample lands in
between. So the "risk regression" head is binary classification wearing a
regression costume — its MAE will look good while adding nothing over
`is_attack`. Corroborated by the smoke run: `risk_mean 0.428, risk_std 0.468`
(std ≈ mean = a two-point distribution).

Not leakage in the train/test sense, and deriving a risk target from severity
is a normal thing to do. But **risk MAE should not be reported as a separate
capability** from technique/gradation accuracy — it is the same signal.

Also a missed opportunity for a *forecaster*: a host under reconnaissance that
is not yet labelled attack gets risk exactly 0.0, so the target carries no
early-warning gradient. Making risk graded would be a research change to the
target definition, so I am flagging it rather than changing it unilaterally.

### 16:05 — Checked train/serve config consistency. Clean, but a correction
`use_memory` is **False** everywhere — training default, loader default, and
the config JSON written beside the checkpoint all agree. No mismatch.
Downstream load will also work with the new head: the config records
`edge_feat_dim: 12`, so both sides build a 12+12+12=36 input. Verified.

**Correction to something I said earlier.** When explaining the chance-level
inductive AUC I said a new node has "zero memory and zero features". The
memory half was wrong — the memory module is *disabled* in this
configuration, so there is no memory for anyone, seen or unseen. The operative
cause was the all-zero node features alone. The fix was right; my reasoning
for it was half wrong.

**Candidate experiment (queued):** turn the memory module ON. TGN's per-node
memory is the paper's central contribution and is exactly the "host state
accumulating over time" an attack forecaster wants. Running without it makes
this a temporal graph attention net, not really a TGN. Worth an A/B once the
current run lands — it is an accuracy lever, so it runs alone.

### 15:58 — Class-starvation guard fired in production and worked
Log: `Inductive draw removed class(es) [4] from training entirely; re-drawing`
(class 4 = Recon). After the re-draw:

```
counts  = {Benign 13,466,285, C2 46,408, Impact 1,300,101,
           InitialAccess 156,990, Recon 137,877}
weights = {0.2, 3.099, 0.586, 1.685, 1.798}
```

**Recon: 0 -> 137,877.** All five weights distinct, none clipped, none at
exactly 1.0 (the zero-samples signature). First run where all five classes
have real training data.

The full progression of this one class, each stage a different bug:

| split | Recon train samples |
|---|---:|
| global time quantile | 0 (3 of 5 classes absent — split by corpus) |
| per-corpus quantile | 7 (cut at 13:13, PortScan runs 13:00-15:59) |
| per-capture quantile | 0 (its single carrier host lost the inductive draw) |
| per-capture + guard | **137,877** |

**Still true and unchanged:** those 137,877 records all come from ONE host
(172.16.0.1). The class is now *trainable* but not *generalisable* — a good
Recon score means the model recognises that host. Do not report Recon as
detection performance.

### 16:10 — Cumulative horizon risk was taking the peak, not accumulating
`InfiltrationRiskHead` returned `max(step_risks)` as `cumulative_risk`,
contradicting its own docstring and the v4 metrics module (which has a test
arguing peak is wrong). The worked example:

| step risks | peak | cumulative |
|---|---|---|
| `[0.50, 0, 0, 0, 0]` | 0.50 | 0.500 |
| `[0.30, 0.30, 0.30, 0, 0]` | 0.30 | **0.657** |

Peak ranks the single spike above the sustained threat. This value feeds
`HostAttackTrajectory.cumulative_forecast_risk` — **how hosts get ordered for
an analyst** — so a host under persistent pressure was being ranked below one
with a noisy spike. Now `1 - prod(1-r)` in log space; `peak_risk()` kept
separately since it answers a different question.
Training unaffected (every trainer discarded this value). *7 tests.*

### 16:00 — Vectorised sampler measured in the real training loop
| | Exp A (reference sampler) | Final (vectorised) |
|---|---|---|
| throughput | 113 batch/s | **158.4 batch/s** |
| batches/epoch | 115,693 | 118,029 *(more data)* |
| epoch time | ~17.0 min | **~12.4 min** |

**+40% throughput**, and that is *despite* the new category head being a
bigger MLP and there being more training edges. My earlier estimate from the
isolated benchmark (~11%) was conservative — in the real loop the Python
sampling loop was costing more than the microbenchmark suggested.

Combined with parallel ingest (12.5 min -> 3.9 min), a full 30-epoch run goes
from roughly 8.7 h to **6.3 h**.

### 16:15 — Found two more copies of the attribute-name list; both had drifted
Same defect as this morning's `host_attributes.py`, in two more places:

- **`explainability/unified_explanation.py`** said `active_conn_density` for
  a value that is `unique_peers / flow_count` (peer fan-out). These strings
  sit next to attribution scores an operator reads.
- **`control_backend/model_adapter.py`** used `avg_flow_duration` and
  `active_conn_density` as dict **keys** in `FEATURE_GROUP_MAP`. Neither name
  exists, so both lookups silently missed and two of fifteen attributes had
  no group in the dashboard UI.

Both now derive from `HOST_ATTRIBUTES`; model_adapter raises at import if the
two ever disagree. An ast-based test now fails if any module uses a stale name
as a real string literal (docstrings excluded — they legitimately quote the
old names when explaining the bug).

*A duplicated name list drifts. The fix is to not have one.* *2 tests.*

### 16:14 — Head fix helped, but three classes still at zero. Found why.
Epoch 0 with the edge-aware head:

| | old head | new head |
|---|---|---|
| Val CatAcc | 0.2115 | **0.4409** |
| Val MRR | 0.5957 | **0.7103** |
| Benign recall | 0.20 | **0.449** |
| Inductive CatAcc | 0.3557 | **0.5318** |
| C2 / Impact / Recon | 0.0 | **still 0.0** |

Real improvement, and no longer a single-class collapse — but Impact has
**1.3M training samples** and 0.0 recall. That is not undertraining.

**Ruled out by measurement, not guesswork:**
- a gradient-boosted tree separates these classes from the 12 edge features at
  **0.9997** balanced accuracy → the signal is there;
- the *same architecture* as the shipped head reaches **0.998** on those
  features standalone, raw or standardised → the head is capable, and the 20x
  feature-scale spread is not the problem.

**Actual cause — the category loss was contributing almost nothing.** Edge and
category losses were summed with equal weight and never logged separately:

| | value |
|---|---|
| edge loss at AUC ~0.82 | 0.5754 |
| category focal loss, mediocre | 0.0420 (**14x smaller**) |
| category focal loss, confident | 0.0001 (**~5700x smaller**) |

Focal γ=2 down-weights easy examples — but 89% of this corpus is one easy
class, so it drives the whole *term* to irrelevance beside a co-summed task.
C2 compounds it at 0.3% of data: a batch of 128 holds ~0.4 C2 examples.

**Fix:** `--cat_loss_weight` (default 15, so 0.042×15 ≈ 0.63 sits alongside the
0.575 edge loss rather than dominating it), and **both terms now print in the
epoch line** so this can never again be invisible. `1.0` reproduces the old
behaviour. *5 tests, one of which establishes the premise by measurement.*

### 16:17 — Restarted with the weighted loss
`logs/tgne_final.out`. Previous run archived as `tgne_final_PRE_CATWEIGHT.out`.
Restarted at 1.2 epochs in — cheap, and the category task is what injects
attack-class information into the embeddings Branch A consumes.

### 16:25 — Branch A's multi-task loss could run away
Kendall & Gal uncertainty weighting is **unbounded below**. For a task with
loss L the optimum is `log_var = log(L)` worth `1 + log(L)`, so as L→0 the
total dives to −∞ and that task's precision explodes. Measured with
`risk_loss=1e-4`:

| step | total | prec_risk | prec_tech |
|---|---|---|---|
| 1 | 2.100 | 1.0 | 1.000 |
| 300 | −6.133 | **9993.5** | 0.833 |

A **12,000× imbalance** — and the logged loss falls the whole time, so it
looks like training is going well.

Not hypothetical here: `risk_score` is derived from `is_attack`, so risk is
the **easiest and least informative** of the three tasks — exactly the one
that runs away, at the expense of technique and gradation, which are the ones
that matter.

Clamped `log_var` to [−3, 3] (precision 0.05–20). Same scenario now settles at
**24×**. Checked the clamp doesn't disable the mechanism: equal losses still
give equal weights, and a genuinely easier task still earns more precision.
*7 tests.*

### 16:30 — Label-mapping coverage was tracked but never reported
`LabelResolver.unresolved_report()` had **zero callers**. Unmapped labels
become `UNKNOWN`/`is_attack=False` — the right default (asserts nothing) but
silent label noise if the rate is material.

**Measured: 0.0% unresolved across 450,000 records, all three corpora.** So
this is a safeguard, not a fix. It matters when your fresh CIC data arrives
with label strings the maps have not seen — precisely the case nobody would
think to check. Trainer now logs coverage, and WARNs with the offending
labels if any appear.

Third instance of the same pattern today (computed → discarded): credibility
verdict, the two loss terms, and now this.

### 16:39 — Weighted loss ruled out its own hypothesis. Found the real cause.
Epoch 0 with `cat_loss_weight=15`:

| | before | now |
|---|---|---|
| Val AUC | 0.8212 | **0.9374** |
| Inductive Val AUC | 0.7911 | **0.9187** |
| Val AP | 0.8376 | **0.9524** |
| C2/Impact/Recon recall | 0.0 | **still 0.0** |

Link prediction improved a lot. The category head did not — and the new loss
logging **disproved my hypothesis**: `edge 0.1154, cat 0.0188 ×15 = 0.282`.
The category term now dominates 2.4:1 and three classes still sit at zero. It
was never gradient starvation.

**Ruled out, by measurement:**
- val genuinely contains all five classes (Benign 92.1%, C2 1.66%, **Impact
  3.41%**, IA 2.70%, Recon 0.12%) — so 0.0 on Impact is a real failure, not an
  absent class;
- the arithmetic matches exactly: `0.9211×0.251 + 0.027×0.997 = 0.258` = the
  observed CatAcc, i.e. it predicts Benign ~25% and InitialAccess for nearly
  everything else;
- no train/eval `n_neighbors` mismatch.

**Actual cause: 58.6% of training batches contain a single class.** TGN slices
batches contiguously in time and attacks are time-localised, so most gradient
steps see 128 samples with one label — "predict X for all of these", no
contrast. The head oscillates and never learns a discriminative rule, while a
tree on the same features *shuffled* gets 0.9997.

**Fix: permute samples each epoch. 58.6% → 0.0% single-class batches.**

Two things worth recording:
- It must shuffle **samples, not batch order**. I wrote the batch-order
  version first and it would have done nothing — reordering contiguous blocks
  leaves every block single-class. Caught it before it shipped; a test now
  pins the distinction.
- Valid **only** because the memory module is off. With memory on, batch N
  depends on batch N−1 and shuffling would train on states that never
  existed. That combination now raises.

### 16:49 — Restarted (v3). `logs/tgne_final.out`
Previous logs archived: `tgne_final_PRE_CATWEIGHT.out`, `..._PRE_SHUFFLE.out`.

### 17:00 — Branch A would not have fitted at full density. Fixed.
Measured the snapshot volume before launching the retrain rather than
discovering it mid-run:

- snapshot ratios: **1.99 per record** (CIC-2018), **0.72** (CTU-13)
- Branch A's 32.7M training records → **~42M snapshots**
- `create_host_sequence_samples` materialises a `[15, 27]` float32 array per
  sample = **2,053 bytes each** → **80.3 GiB**

On a 22 GiB machine that is a hard blocker, and thinning is not an option.

**Fix:** `LazyHostSequenceDataset` keeps two int32 columns (~8 B/sample,
**336 MB** at 42M) and gathers each window from the memmapped feature block in
`__getitem__` — which is what DataLoader workers are for. Semantics identical;
7 tests, the main one asserting every sample matches the eager path field for
field. Credibility gate switched to `evaluate_store`, which already existed
for this purpose.

Branch A also needs `--spill-dir` at full density (store is ~6.3 GB on disk,
~1 GB resident).

### 17:12 — The shuffle fix worked. First good epoch of the project.

| metric | v2 (time-ordered) | **v3 (shuffled)** |
|---|---|---|
| Val AUC | 0.9374 | **0.9970** |
| Inductive Val AUC | 0.9187 | **0.9923** |
| Val CatAcc | 0.2581 | **0.9758** |
| Inductive CatAcc | 0.4594 | **0.9718** |
| Val MRR | 0.6185 | **0.9878** |

```
per-class val: {Benign 0.994, C2 0.677, Impact 1.000, InitialAccess 0.505, Recon 0.995}
inductive:     {Benign 0.995, C2 0.325, Impact 1.000, InitialAccess 0.509, Recon 0.0}
```

Three classes went from **exactly 0.0** to learning. The single-class-batch
diagnosis was right.

**Two caveats, so the numbers are not oversold:**

1. **Inductive Recon = 0.0 is CORRECT, not a failure.** Recon comes from one
   host (172.16.0.1), so evaluating it on *unseen hosts* is meaningless —
   there are no other Recon hosts to generalise to. The documented limitation
   surfacing exactly where it should.

2. **CatAcc is probably reading the edge features, not the embeddings.** A
   tree on those 12 features alone gets 0.9997, and the category loss sits at
   0.0093 — the task is easy once you can see the flow. So 0.9758 must not be
   read as "the embeddings encode attack class".

**What actually validates the encoder is link prediction:** inductive 0.9923
vs transductive 0.9970 — a **0.005 gap**, against the 0.49 gap when it was
memorising hosts (0.9981 vs 0.5043). That is the number that says it learned
transferable structure.

Weakest classes: InitialAccess 0.505 and C2 0.677/0.325. Worth watching
across epochs.

### 17:30 — Epoch 1. Learning, and C2 exposes a second data limitation.

| class | ep 0 | ep 1 |
|---|---|---|
| InitialAccess | 0.505 | **0.714** |
| C2 | 0.677 | 0.674 |
| C2 inductive | 0.325 | 0.319 |
| Benign / Impact / Recon | 0.994 / 1.0 / 0.995 | 0.993 / 1.0 / 0.994 |

CatAcc 0.9758 → 0.9803, inductive AUC steady at 0.9918.

**C2's val-vs-inductive gap (0.67 vs 0.32) is the tell.** C2 is botnet C&C and
comes almost entirely from CTU-13, where I measured **1–2 bot source hosts per
scenario** (only scenario 9 has 10). So the model can recognise C2 on hosts it
has seen and cannot transfer to new ones — the same structural limitation as
Recon, less extreme.

**Two of five classes are therefore host-diversity-limited, not model-limited:**
Recon (1 host) and C2 (10 hosts across all of CTU-13). No amount of training
fixes that; it needs a corpus with more distinct attackers (CIDDS-001 was the
candidate identified earlier).

### 17:35 — All three downstream stages had the same 80/38/52 GiB blocker
Checked Branch B and DeepOP rather than waiting to hit it:

| stage | materialised | lazy |
|---|---|---|
| Branch A | **80.3 GiB** | 336 MB |
| Branch B | **37.9 GiB** | ~280 MB |
| DeepOP | **~52 GiB** | ~340 MB |

None of them would have run at full density. Each keeps int32 index columns
and gathers windows from the store's memmapped block in `__getitem__`.

DeepOP needed one extra care: it 2x-oversamples non-Benign windows to stop the
decoder collapsing to the quiescent sequence. That is preserved by listing
those positions twice in the index, and the index is built from the store's
`cat_id` column directly so no row is materialised to decide it.

*22 tests across the three, each asserting equivalence with the eager path
rather than just "it runs".*

### 17:50 — Early stopping was selecting on a saturated metric
`val_ap` (link prediction, seen hosts) converges in **one epoch** here and then
flatlines, while the classification head keeps improving:

| | ep 0 | ep 1 | ep 2 |
|---|---|---|---|
| val_ap | 0.9971 | 0.9968 | 0.9968 |
| inductive AP | 0.9926 | 0.9921 | 0.9918 |
| C2 inductive | 0.325 | 0.319 | **0.517** |

With patience 5 the run would stop and **restore epoch 0**, discarding the only
thing still getting better. Confirmed by feeding the real sequence to
`EarlyStopMonitor`: it picks epoch 0.

**Now selects on `0.5 × inductive AP + 0.5 × val macro-F1`:**
- *inductive*, because deployment meets unseen hosts and the transductive
  figure has no discriminative power left at 0.9968;
- *macro* F1, because Benign is ~90% of val — a head predicting Benign for
  everything scores 0.90 accuracy and under 0.25 macro-F1. Aggregate accuracy
  is precisely the metric that hid the collapse all day.

**Deliberately not restarting for this.** The current run will stop around
epoch 5 and show whether the category head plateaus by itself — that is the
data needed to know whether the change matters. Applying it to the next run.

### 17:55 — InitialAccess: an optimisation gap, not a data limit
Investigated the oscillation (0.505 → 0.714 → 0.578). Two hypotheses tested:

1. **Heterogeneous class?** No. In CIC-2018 InitialAccess is 99%
   `Infilteration` (93,063 of ~94,000); Brute-Force-Web/XSS and SQL Injection
   contribute under 1,000 between them. Not a taxonomy problem.
2. **Unlearnable class?** No. The earlier separability test measured
   **0.9997** balanced accuracy for Benign vs InitialAccess with a tree on the
   same 12 edge features.

So the ceiling is ~1.0 and the model sits at 0.58. **That is an optimisation
gap, not a corpus gap** — unlike Recon (1 carrier host) and C2 (10), which are
genuinely data-limited.

Distinguishing these matters: Recon and C2 need a different corpus, whereas
InitialAccess should be recoverable with training changes. Not acting on it
yet — the run is still improving and I do not want to change two things at
once.

### 18:00 — CORRECTION to the 17:55 entry on InitialAccess
I said InitialAccess had a **0.9997 ceiling** and was therefore an
"optimisation gap, not a data limit". **That was wrong.** The separability
test sampled the first six CIC-2018 files alphabetically, which **excludes
`wed_14`**, and I generalised from a subset dominated by the easy behaviour.

Measured per file, both labelled InitialAccess:

| file | content | tree balanced acc | IA recall |
|---|---|---|---|
| `wed_14` | FTP/SSH brute force | **1.0000** | 1.000 |
| `thu_1` | **Infilteration** | **0.6322** | **0.278** |

The class conflates a trivially separable behaviour with one a
gradient-boosted tree can barely detect. Infiltration being undetectable from
flow features is a documented property of CSE-CIC-IDS2018 — the compromise is
host-level and the network flows look benign.

**Revised: the model's 0.58 on InitialAccess is plausibly near its practical
ceiling**, not an optimisation failure. So all three weak classes are
data-limited, in three different ways:

| class | limit |
|---|---|
| Recon | 1 carrier host — cannot generalise |
| C2 | 10 hosts across all of CTU-13 — improving but capped |
| InitialAccess | conflates brute force (easy) with Infiltration (≈undetectable from flows) |

**Method note:** the first test was cheap and I trusted it too quickly. A
per-file breakdown would have caught it immediately — averaging over a corpus
whose files have different attack types hides exactly this.

### 18:05 — Epoch 3 breaks the "plateau", and corrects me again

| | ep 0 | ep 1 | ep 2 | **ep 3** |
|---|---|---|---|---|
| Val AP | 0.9971 | 0.9968 | 0.9968 | **0.9973** |
| Inductive AP | 0.9926 | 0.9921 | 0.9918 | **0.9935** |
| Inductive AUC | 0.9923 | 0.9918 | 0.9916 | **0.9936** |
| C2 val | 0.677 | 0.674 | 0.761 | **0.779** |
| C2 inductive | 0.325 | 0.319 | 0.517 | **0.565** |
| InitialAccess | 0.505 | 0.714 | 0.578 | **0.612** |

**Correction to the 17:50 entry.** I called `val_ap` "saturated" from three
points (0.9971, 0.9968, 0.9968) and argued the run would stop and restore
epoch 0. Epoch 3 beat all three. It was **noise, not saturation** — three
points is not a trend, and I should not have drawn one from them.

Consequences:
- Early stopping now selects epoch 3, so the "restores a worse checkpoint"
  concern is gone. **No restart needed.**
- The selection-metric change is still worth keeping (macro-F1 catches
  minority collapse that aggregate accuracy hides), but my *justification*
  for it was premature.

**C2 inductive has nearly doubled (0.325 → 0.565)**, which weakens the
host-diversity explanation I gave for it — 10 distinct bot hosts appear to be
enough to learn transferable structure. Recon (1 host) remains the only class
that is structurally unable to generalise.

### 18:12 — Branch A validated end to end against the new encoder
Ran it under a 4 GB cap so it could not disturb training. **Exit 0, peak 2.6 GB,
72 s.**

| check | result |
|---|---|
| new TGNE architecture loads downstream | ✓ |
| lazy dataset wiring | ✓ 6,644 / 856 / 882 samples |
| frozen split honoured | ✓ **17 train / 3 val / 3 test** |
| store-based credibility gate | ✓ **CREDIBLE** |
| held-out test scored once | ✓ `tech_accuracy 0.756` vs persistence 0.679 |
| **credibility verdict persisted** | ✓ `{checked: True, credible: True, problems: []}` |
| **contract persisted** | ✓ `window 2.0s, history 15, forecast 5` — matches v4 |

Both of today's checkpoint-integrity fixes work in practice, and a real
retrain should flip the three strict-xfail contract tests to passing.

Numbers are from a deliberately tiny config (250 rows/capture, stride 400)
and are **not results** — the point was to prove the path, not to measure the
model.

One thing to watch at full density: `val_traj_len_median = 1` and only 6 hosts
with 16+ snapshots at this stride. Branch B needs T+1 = 16 snapshots per host,
so at low density it would have almost nothing to train on. Full density
should fix it, but worth checking when Branch B runs.

### 18:30 — Quantified why Branch B needs the PCAP path
Checked host-trajectory lengths before running Branch B, since it needs
T+1 = 16 snapshots per host:

| source | hosts | real identity? | ≥16 snapshots |
|---|---|---|---|
| CIC-2018 ×9 days | **350** | ✗ fabricated | 100% |
| CIC-2018 `tue_20` | 37 | ✓ real | 11 |
| CTU-13 scen 9 | 127,730 | ✓ real | **0.6%** |

**The 350 is the tell.** `cic2018_adapter` fabricates
`192.168.10.{i%250+1}` and `172.16.0.{i%100+1}` — exactly 250 + 100. Every
long, clean trajectory on those days is an artefact of assigning host identity
by row index, not network behaviour.

So on the CSV path Branch B would train almost entirely on a synthetic
350-host graph, while the genuinely-identified sources give either very few
hosts (tue_20: 37) or very few sustained ones (CTU-13: 0.6% reach 16).

**Plan (pragmatic, and stated honestly rather than hidden):**
1. Run Branch B/DeepOP on the CSV path first, to validate the pipeline
   end-to-end and produce loadable checkpoints.
2. **Label those results as fabricated-identity for 9 of 10 CIC-2018 days.**
   They demonstrate the pipeline, not the science.
3. The scientifically valid run needs `--pcap-root`. Rough cost estimate:
   ~1.2B packets across 600 GB, so several hours of parsing — an overnight
   job, and better done once the 266 corrupt files are replaced.

This confirms the original plan's Phase 2/3 rationale with numbers rather than
argument.

### 18:56 — Epoch 6 is the new best. Not stopping training.

| epoch | ind_AP | macro recall | selection |
|---|---|---|---|
| 3 | 0.9935 | 0.8760 | 0.9347 |
| 5 | 0.9934 | 0.8488 | 0.9211 |
| **6** | **0.9945** | **0.8768** | **0.9357** |

C2 inductive: 0.325 → 0.319 → 0.517 → 0.565 → 0.586 → 0.480 → **0.692**.
More than doubled since epoch 0, and still climbing.

**I had intended to stop after epoch 7 and run the downstream models so there
would be a complete pipeline tonight. That was the wrong call** — the encoder
is still gaining materially, and cutting it short would trade real model
quality for a tidier status report.

Both criteria now agree on epoch 6, so the earlier divergence has resolved as
well.

**Plan: let it converge on its own (patience 5).** Downstream runs after.
Everything for that transition is already staged and verified:
- `select_best_encoder.py` — picks the epoch, no restart needed
- `write_encoder_config.py` — the config an earlier epoch would otherwise lack
- epoch checkpoint verified to load (category head 36 → 5)
- Branch A validated end to end at 2.6 GB
- all three lazy datasets in place
