# While you were out — the short version

Detail in [`19_session_log.md`](19_session_log.md). Nothing is broken; nothing
is waiting on you.

---

## The headline

**The encoder went from useless to working.**

| | this morning | now |
|---|---|---|
| Inductive AUC *(unseen hosts)* | **0.5043** — chance | **0.9946** |
| Val CatAcc | 0.3327 — collapsed | **0.9792** |
| classes learning | 2 of 5 | **5 of 5** |

The number that matters is the **gap**: inductive 0.9946 vs transductive
0.9978 = **0.003**. It was 0.49 when the model was memorising hosts. A small
gap is what generalisation looks like; a high transductive score alone means
nothing.

Tests: **118 → 440 passing.**

---

## The three fixes that actually mattered

**1. The category head could not see what it was classifying.**
It was `Linear(12)` on `src_emb + dst_emb`. The label is a property of the
*flow* — ports, bytes, duration — so a benign flow and an attack flow between
the same two hosts were **identical inputs**. Unseparable in principle. Now
`[src ; dst ; edge_features]`.

**2. 58.6% of training batches contained a single class.**
TGN batches contiguously in time and attacks are time-localised, so most
gradient steps saw 128 samples with one label — nothing to discriminate
against. Shuffling samples took that to 0.0%. This was the single biggest
lever: three classes went from **exactly 0.0** to learning.

**3. All three downstream models would have run out of memory.**
Branch A needed **80 GiB**, Branch B **38 GiB**, DeepOP **52 GiB** — they
materialised every training window. Now built on demand from the memmapped
store: **~1 GB total**. Found by measuring before launching, not by crashing.

---

## What I found that you cannot fix with code

Three classes are limited by the *data*, not the model:

| class | limit |
|---|---|
| **Recon** | all 158,930 records come from **one host** (172.16.0.1). Inductive recall is 0.0 and always will be — there is nothing to generalise to. |
| **C2** | ~10 bot hosts across all of CTU-13. Still reached 0.692 inductive, better than I expected. |
| **InitialAccess** | conflates brute force (a tree gets **1.000**) with Infiltration (a tree gets **0.278**). Infiltration is not detectable from flow features — a known CIC-2018 property. |

**Also:** on the CSV path, 9 of 10 CIC-2018 days fabricate exactly **350
hosts** (`i % 250` / `i % 100`). Branch B trained there learns a synthetic
host graph. Its results will be **pipeline validation, not science**, until
the PCAP path runs. Your fresh PCAPs are the fix.

---

## Also fixed (each with tests)

Cumulative risk returned the *peak* instead of accumulating, so hosts were
ranked wrong for the analyst · the multi-task loss could run away 12,000× ·
two more drifted copies of the attribute names, one silently breaking
dashboard grouping · inference fed the encoder **zero** node features
(would have cancelled the whole inductive gain) · spill files never
reclaimed on a 90%-full disk · credibility verdicts computed then discarded ·
early stopping selecting on a saturated metric.

**Speed, all lossless:** ingest 12.5 min → 3.9 min (3.2×), training
throughput +40%.

---

## Things I got wrong and corrected

Worth knowing, since I stated some of these confidently before checking:

- Claimed InitialAccess had a 0.9997 ceiling → I had measured the wrong
  files; it is ~half unlearnable.
- Called `val_ap` "saturated" from three noisy points → epoch 3 beat all of
  them.
- Wrote a batch-*order* shuffle that would have done nothing → caught before
  shipping.
- Added smoothing to the epoch selector that picked a **worse** checkpoint →
  reverted to raw score plus a reported tie set.

---

## The full pipeline runs end to end

Validated all four stages against the new encoder (small smoke configs, so the
numbers are **not results** — the point was proving every integration point):

| stage | result |
|---|---|
| TGNE encoder | inductive AUC **0.9946**, epoch 6 selected |
| Branch A | exit 0, 2.6 GB peak, **CREDIBLE**, held-out test scored once |
| Branch B | exit 0, 3.4 GB peak, **CREDIBLE**, val_loss 0.3916 |
| DeepOP | val_loss 0.5397, conditioned on **Branch-B rollouts** (audit E1, not oracle futures) |

Both downstream checkpoints carry the correct v4 contract, and Branch A's
carries its credibility verdict. So the real retrain has no unknowns left in
it.

## Where it stands right now

Training is **still improving**, so I did not stop it — I had planned to, to
have a complete pipeline ready for you, and that would have traded real
quality for a tidier report.

Everything for the downstream handoff is staged and verified: best-epoch
selection, config generation, checkpoint load, Branch A validated end to end.
When the encoder converges the rest is a few commands with no unknowns.
