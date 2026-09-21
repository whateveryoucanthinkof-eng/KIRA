# Performance experiments — protocol and results

**Question:** the machine is idle (1 core of 16, GPU at 3W of 80W) while an
epoch takes 13.5 minutes. Can that be recovered **without paying accuracy**?

**Design rule:** one variable per experiment. A performance change and an
accuracy change never move in the same run, or neither result means anything.

**Bar for a "lossless" change:** metrics must come back **identical**, not
similar. Both changes under test are provably equivalent at unit level, so any
end-to-end drift is a bug to chase, not a tradeoff to accept.

All runs: seed 0, full density (34,152,542 records), frozen train captures,
per-capture temporal split, 18 GiB cap.

---

## The bottleneck, measured

| resource | during training | headroom |
|---|---|---|
| CPU | **97.9%** = 1 core of 16 | 94% idle |
| GPU | 0–12%, **3W of 80W**, P8 | ~95% idle |
| GPU memory | 67 MiB of 8,188 | 99% free |
| per batch | 8.8 ms | — |

Three compounding causes:

1. **TGN is architecturally sequential.** The memory module makes batch *N*
   depend on batch *N−1*. No shuffling, no data-parallel processes. This is a
   hard floor, not a tuning miss.
2. **Batches are too small for the GPU.** `batch_size=128`, `n_degree=10` →
   ~4,200×12 tensors. The 4060 finishes in microseconds and waits. TGN issues
   ~100–300 small kernels per batch; the cost is launch overhead and Python
   dispatch, not compute. 3W of 80W is the tell.
3. **Sampling was a Python loop** — 384 iterations per batch, ~35M per epoch.

### A correction

An earlier estimate put sampling at **71%** of batch time. That was measured
against a toy MLP standing in for the real TGN model and was **wrong**. Against
the real 8.8 ms/batch, sampling is ~1.15 ms — **~13%**. The remaining ~87% is
the TGN model itself. Recorded because it changed which lever matters.

---

## Experiment A — baseline

```
TGNE_REFERENCE_SAMPLER=1  --ingest_workers 0
```
Original per-node sampler, serial ingest.

## Experiment B — lossless performance changes

```
--ingest_workers 6        (vectorised sampler is the default)
```

| change | unit-level result | tests |
|---|---|---|
| Vectorised `get_temporal_neighbor` | 4.7× at batch 128, **13.3× at 512** | 25, vs the original kept verbatim as oracle |
| Parallel capture ingest | **3.14×** (42.5s → 13.5s) | 6, bit-identical output |

The sampler speedup **scales with batch size**, so it compounds with any later
batch-size change.

**Prediction:** epoch-0 metrics identical to A; wall time lower.

### The trap parallel ingest had to avoid

Node ids are assigned in **encounter order**. Workers finishing out of order
would give the same host a different id — changing its node-feature row, the
negative sampler's draws, and which nodes are inductive. The graph stays
isomorphic but reproducibility dies, silently invalidating this very
comparison. Workers therefore return **local** vocabularies and the parent
merges in fixed capture order.

## Experiment C — batch size (accuracy lever, runs alone)

Only after A/B settles. Within a TGN batch every interaction sees the *same*
memory snapshot, so a **smaller** batch means fresher memory and a temporally
more faithful model — but noisier gradients and more of the launch overhead
that already dominates. Genuinely bidirectional; needs measurement, not a
guess.

---

## Results

| run | load | epoch 0 | Val AUC | Val CatAcc | Ind. AUC | Ind. CatAcc |
|---|---|---|---|---|---|---|
| A (baseline) | *pending* | | | | | |
| B (perf) | | | | | | |

---

## What is NOT worth doing

**Multiple training processes / DDP.** Data-parallel training assumes batches
are independent. TGN's memory makes them a serial dependency chain — each
process would read a memory state that never existed. Fast, and wrong.
