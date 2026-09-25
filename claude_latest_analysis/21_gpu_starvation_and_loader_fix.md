# 21 — "Why CPU and not GPU": the loader was starving the GPU

**Date:** 2026-09-21
**Trigger:** observed GPU at 12–16% / 22 W of 85 W while Branch A trained, CPU at 338%.

---

## The short answer

Training *was* on the GPU (`device=cuda`). The GPU was **starved**, not unused.

Two causes, both mine, both in code I wrote earlier this session:

1. **`DataLoader` was created with no `num_workers`** — PyTorch defaults to `0`,
   which loads every batch in the main process, serialised with GPU compute.
   When I wrote the lazy datasets I justified the per-item cost with "that is
   what a DataLoader worker is for", and then never enabled one.
2. **`__getitem__` called `store._materialize(row)`** to build a whole
   `HostWindowSnapshot` — both feature slices, the technique list, the host
   string — and then read four scalars off it.

The lazy datasets trade memory for per-item work (80.3 GiB → 336 MB for Branch
A). That trade only pays if the work is *overlapped*. It wasn't.

---

## A measurement I got wrong, and the correction

My first benchmark said `_materialize` cost **86 µs/sample**, i.e. 29.6 min per
epoch, i.e. **87×** slower than reading columns. That was wrong.

`_materialize` does `from data_unification.multi_dataset_stream import
HostWindowSnapshot` *inside the function*. The first call in my timing loop paid
the module's ~1.7 s import cost, and I divided it across 20,000 iterations —
manufacturing ~85 µs of phantom per-call cost.

With a warmup, the honest figures are:

| | per sample | per epoch (20,664,255 samples) |
|---|---|---|
| `_materialize` + field reads | 3.12 µs | 1.1 min |
| `target_fields` (direct columns) | 0.86 µs | 0.3 min |
| | **3.6×** | **saves ~0.8 min/epoch** |

So the snapshot building was real but minor. I recorded the 87× claim before
checking it; the profile below is what actually decided the work.

## What the profiler said

`py-spy record`, 60 s, 5,350 samples, against the live run:

| bucket | share of wall clock |
|---|---|
| **data loading** (dataloader + dataset + memmap) | **35.7%** |
| torch compute (fwd/bwd/optim) | 35.3% |
| other / unattributed | rest |

Data loading costs ≈1.0× compute time and was fully serialised with it. That
puts the ceiling from workers at ~1.55×, not the 5–10× the bad benchmark
implied.

**Measured end-to-end** (same model, same batches, 1.08M-row memmap-backed
store, 1,500 batches, while the live job competed for cores):

```
num_workers=0 (current)     168.2 batch/s
num_workers=4 (fixed)       215.8 batch/s     1.28x
```

---

## Changes

| file | change | why |
|---|---|---|
| `data_unification/trajectory_store.py` | new `target_fields(row)` | four columns instead of a whole snapshot |
| `branch_a_gnn_lstm/sequence_dataset.py` | uses `target_fields`; caches per-host row arrays | drops a dict hash and a dataclass build per sample |
| `branch_b_world_model/train_branch_b.py` | `risk_future` is one fancy-index | it built **K** snapshots per sample to read one float from each |
| `scripts/retrain_branch_a_live.py` | `--num-workers` (4), `pin_memory`, `persistent_workers`, `prefetch_factor=4`, `non_blocking` H2D | overlap loading with compute |
| `scripts/retrain_future_models_live.py` | same, via `_loader_kwargs`, on all four loaders | Branch B and DeepOP had the identical defect |
| both scripts | loss accumulated on-device, read once per epoch | `loss.item()` per batch forced ~161k host↔device syncs per epoch |
| both scripts | progress line every 2,000 batches with rate + ETA | an epoch is ~161k batches and printed **nothing** until it ended |
| `scripts/retrain_branch_a_live.py` | `_evaluate` accumulates on device | it synced 3× per batch and grew a 1.02M-element Python list |
| `scripts/retrain_future_models_live.py` | Branch B now reports `train_loss` | it was never tracked at all |

### A bug the benchmark caught before the run did

`compute_loss` returns shape `[1]`, not a 0-dim scalar, so
`_loss_sum += loss.detach().double()` raised
`output with shape [] doesn't match the broadcast shape [1]`. It would have
killed the next run on batch 1. Fixed with `.sum()`, which accepts both shapes.

---

## Why I did *not* restart the in-flight run

I first read "50 minutes elapsed, no epoch printed" as a 50-minute epoch. Wrong:
extraction ran 20:29:57 → 21:07:35 (**37.6 min**) and epoch 1 had only been
training for 17.6 min.

Epoch 1 finished at 21:25:13 — **17.6 min/epoch**, against the 16 min the
benchmark predicted. Eight epochs ≈ 2.3 h.

Restarting would have cost 37.6 min of re-extraction to buy 1.28× on ~2.3 h of
training: a wash. The fixes land on Branch B and DeepOP, which had not started.

`epoch=1 train_loss=-4.1490 val_loss=1.8819 risk_mae=0.2229 tech_accuracy=0.879`
(negative train loss is expected: the Kendall & Gal homoscedastic multi-task
loss is unbounded below; `log_var` is clamped to [-3, 3].)

---

## Correctness

`tests/test_target_fields_equivalence.py` (new, 4 tests):

- `target_fields` matches `_materialize` on **every row** of a mixed store
- the Branch A `Dataset` yields identical `risk` / `technique` / `gradation` /
  `window_idx` / `host_ip` to the snapshot path
- Branch B's `risk_future` matches element-for-element, padding included
- `target_fields` stays faster than `_materialize` (guards the regression)

`_evaluate` returns **identical** metrics at `num_workers=0` and `3`
(`loss=3.770992 risk_mae=0.330889`). All three trainers smoke-tested end to end
with workers on.

Full suite: **453 passed, 2 xfailed**.
