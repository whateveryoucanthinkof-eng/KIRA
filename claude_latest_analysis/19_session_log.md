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
