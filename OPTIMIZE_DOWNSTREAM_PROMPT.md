# Prompt: optimise Branch A, Branch B and DeepOP training (the way the encoder was optimised)

You are taking over performance work on the CyberWorld training pipeline (repo
`cyber_network_predictor-…`, branch `v5.5o`). The TGN **encoder** (`bita/train.py`)
has already been optimised from ~1 batch/s to ~230 batch/s (single lane) on the
full 130.6M-edge corpus, with every change proven equivalent. The three
**downstream** trainers were never touched and are now the bottleneck. Your job
is to give them the same treatment.

Read this whole file before writing code. The rules in section 1 are not
negotiable; the playbook in section 4 is what worked.

---

## 1. Non-negotiables (user's standing rules)

1. **No quality trade-offs for speed.** Batch sizes stay as they are (Branch A
   128, Branch B 128, DeepOP 64). No data thinning, no strides, no row caps, no
   fewer epochs, no smaller models, no mixed precision that changes results.
2. **Every change is proven equivalent before it is used.**
   - Pure data-structure / bookkeeping / scheduling changes must be
     **bit-identical** (`np.array_equal` / `torch.equal`, not `allclose`):
     same samples, same order, same losses, gradients, final weights, metrics.
   - Only changes that reorder floating-point math (fused kernels, CUDA graphs
     over different shapes) may differ, and only at fp32-rounding level. You
     must report the measured magnitudes, and it must be smaller than the
     code's own run-to-run noise (GPU atomics). Those changes are **opt-in
     flags**, never default, and the user approves them.
3. **Profile first, then fix what the profile shows.** Do not guess. Every
   claimed speedup is a measured, interleaved A/B on the same input.
4. **Installs only in the Fedora toolbox `cyberworld_claude`**
   (`toolbox run -c cyberworld_claude …`). Never install on the host. The host
   already has torch 2.13 + CUDA 13, Triton, gcc, cargo/rustc, numpy 1.26.4
   (pinned: the Rust PCAP parity depends on it; do not upgrade numpy).
5. **Commit every working milestone** on your branch with a clear message
   (end with the `Co-Authored-By:` line your harness specifies). Sessions get
   cut off by usage limits; uncommitted work is lost work. Push to `origin`
   only, never to the `sih` remote.
6. If a tool call is rejected, **stop and report** what you have with your last
   commit. Do not wait silently.

---

## 2. The machine (this has frozen before; respect it)

- Laptop: 16 cores, **22 GiB RAM, zram-only swap**, RTX 4060 Laptop **8 GB**.
- A job that spills into zram livelocks the whole desktop (happened twice, no
  OOM-kill ever fired). Run anything heavy under
  `systemd-run --user --scope -p MemorySwapMax=0 -- <cmd>`. You do not need a
  `MemoryMax` cap unless asked, but `MemorySwapMax=0` is mandatory.
- **`/tmp` is tmpfs = RAM.** Never write large files there. Scratch goes in
  `~/.local/state/cyberworld/scratch/<your-task>/`.
- `~/Documents/SIH` is an **older, separate git repo**. Do not create files
  there. Data dirs under it (`DATA/`, `CTU-13-Dataset/`, `test/`) are read-only.
- Disk is tight (~60 GB free). Clean up your scratch.
- In `pkill`/`pgrep`, anchor patterns to the interpreter
  (`'^[^ ]*python[0-9.]* (-u )?scripts/retrain_'`). An unanchored pattern
  matches your own shell and kills it (exit 144).

---

## 3. Where things are

| What | Where |
|---|---|
| Encoder trainer (already optimised) | `bita/train.py`, `bita/model/{tgn,extentedtgn}.py`, `bita/modules/*`, `bita/fast/*` |
| **Branch A trainer** | `scripts/retrain_branch_a_live.py` (loop ~L1795-1860, loaders ~L1576-1595, eval `_evaluate` ~L840-1060) |
| Branch A model / data | `branch_a_gnn_lstm/lstm_multitask.py` (`MultiTaskLSTM`), `branch_a_gnn_lstm/sequence_dataset.py` (`LazyHostSequenceDataset`, `__getitem__` ~L329) |
| Branch A extraction / cache | `data_unification/capture_columns.py`, `data_unification/parallel_extract.py`, `data_unification/multi_dataset_stream.py` (`HostTrajectoryExtractor`), `data_unification/trajectory_store.py` |
| **Branch B + DeepOP trainer** | `scripts/retrain_future_models_live.py` (`train_branch_b_live` ~L221, `_precompute_rollouts` ~L527, `train_deepop_live` ~L603, `main` ~L988) |
| Training guard (all four models) | `cyberworld_v4/training_guard.py` (has a sync-free `backward_step_deferred` the encoder uses; the downstream trainers still use the syncing `backward_step`) |
| Plan orchestration | `scripts/run_training_plan.sh`, `scripts/run_encoder_comparison.py`, launcher `final_runner.sh` |
| Synthetic end-to-end corpus | `scripts/dry_run_plan.py` (`build_corpus`) |
| Machine settings | `~/.config/cyberworld/plan_env.sh` (template `scripts/ops/plan_env.example.sh`) |
| Watchdog | `scripts/ops/watchdog.py` |
| Existing tests | `tests/test_branch_a_*`, `tests/test_branch_b_*`, `tests/test_deepop_*`, `tests/test_lazy_sequence_dataset.py`, `tests/test_capture_columns.py` |

**Plan order (LANES=1):** encoder(full, seed 42) → Branch A → encoder(cross_network,
seed 42) → Branch A → encoder(seed 123) → Branch A → encoder(seed 2024) → Branch A →
Branch B → DeepOP (once, on the seed-42 cross_network encoder). So Branch A runs
**four times**: its speed matters four times over.

**Do not edit these** (doing so invalidates the 8.4 GB encoder ingest cache,
keyed by a content hash of the parse code): `data_unification/{training_sources,
pcap_bridge,pcap_adapter,attack_windows,label_resolver,tgne_features,unified_schema,
parallel_ingest}.py`, `telemetry/**`, `rust/pcap_fast/**`. Put new code in new
modules. Do not touch `results/training_plan/` (live run state) or any systemd
unit you did not start.

**Caches that already exist, reuse them:**
- `results/training_plan/.ingest_cache/` (8.4 GB): parsed encoder input.
- `results/training_plan/.shared_setup/` (21 GB): encoder setup store.
- `results/training_plan/.spill/capture_cache/` (~21 GB): **Branch A's parsed
  captures, all 31 (train/val/test) already built.** Branch A runs do not need
  to re-parse anything.

---

## 4. What was done for the encoder (the playbook)

Measured on the full corpus (130.6M edges, 3.49M nodes, 461,389 batches/epoch):

| Step | Change | Result | Equivalence |
|---|---|---|---|
| 0 | Baseline | 0.95 batch/s; GPU idle, 1 core pinned | — |
| 1 | `memory.clone()` instead of `get_memory(list(range(n_nodes)))` per batch (an O(nodes) Python list every step) | 13x | bit-identical (values + grads tested) |
| 2 | **Profile with cProfile + `torch.profiler`** to split host time, GPU time, kernel counts, syncs | found: ~6,400 GPU ops/batch, 450 `.item()` + ~500 stream syncs/batch, GPU busy 9 ms of a 100 ms step | — |
| 3 | Pending messages as GPU tensors instead of per-node Python lists; vectorised grouping; batched host→device uploads in one pinned buffer | ~4-5x | forward bit-identical |
| 4 | Triton BiGRU kernel (fwd+bwd) | 1.2x more | fp32-rounding (and *more* accurate than cuDNN's TF32 default) |
| 5 | **Remove per-batch host↔device syncs**: accumulate losses/metrics on GPU, read once per log interval / epoch | 94 → 108 batch/s | bit-identical |
| 6 | Sync-free guard step (`backward_step_deferred`: Adam step on GPU, undone on GPU if non-finite, outcome read one step later) | 181 → 219 | bit-identical incl. NaN cases |
| 7 | `index_select` instead of advanced indexing in a gather (its backward was 37% of GPU time) | GPU 2.4 → 1.7 ms/batch | bit-identical |
| 8 | CUDA graphs over fixed-shape parts (level 3) | 1.4x | bit-identical |
| 9 | Hot host-side lookup moved to Rust (`rust/tgn_host`, C-ABI via ctypes, numpy fallback) | +7% | bit-identical |
| 10 | **Batch planner**: a forked process precomputes all value-independent per-batch work (indices, negatives, grouping) into a shared-memory ring; trainer only replays | +18% | bit-identical end to end incl. crash-resume |
| 11 | Out-of-core: per-edge arrays on disk (`np.memmap`), only each batch's rows go to GPU | RAM 38 GB → 2.4 GB; GPU no longer grows with edges | bit-identical |
| 12 | One shared read-only setup store for all runs on the same data | ~7 min + 20 GB saved per run | bit-identical |

**What did NOT work / was rejected:**
- `torch.compile`: slower (dynamic shapes → recompiles). Measure before adopting.
- Raising batch size: rejected by the user (quality rule).
- Two training processes sharing the 8 GB GPU at the graph-heavy level: OOM.
  Production runs **one lane**.

---

## 5. Method (follow this order for each trainer)

1. **Reproduce the slowness small.** Build a fixed input you can run in
   1-3 minutes: the synthetic corpus (`scripts/dry_run_plan.build_corpus`) and
   one real small capture (CTU-13 scenario 7,
   `~/Documents/SIH/CTU-13-Dataset/7/capture20110816-2.binetflow`, 114k records),
   using the already-built capture cache where possible.
2. **Profile** with three tools, and write the numbers down:
   - `py-spy dump`/`record` (in the toolbox) on the live process: main process
     *and* each DataLoader worker. Who is waiting on whom?
   - `torch.profiler` with CUDA activity over ~50 steps: kernel count per
     batch, GPU busy time vs wall time, `cudaStreamSynchronize` count.
   - System: per-process `%CPU` and state (`R` vs `D` = waiting on disk), GPU
     util, `vmstat 1` (`wa` column = I/O wait), page-fault rate
     (`/proc/<pid>/stat` majflt).
3. **State the bottleneck in one sentence with numbers** before changing code.
4. **Fix the largest item. Keep each change small and separate.**
5. **Prove equivalence** (section 6) for that change.
6. **Measure interleaved A/B** (old, new, old, new on the same input) and
   report batch/s and host ms/batch vs GPU ms/batch.
7. **Commit**, then go back to step 2. Stop when the GPU is the limit (host
   ms/batch < GPU ms/batch) or the remaining gains are below ~5%.

---

## 6. Equivalence harness (build this first, per trainer)

- Run the trainer twice on the fixed input, **on CPU, single-threaded**
  (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, fixed seed), old code vs new
  code. CPU single-thread is deterministic; the GPU is not (atomics).
- Compare and require exact equality for: the order and content of every
  batch the loader yields, per-step loss sequence, final `state_dict` (every
  tensor), optimizer state, validation metrics, and the guard's recorded
  decisions.
- For fp32-rounding changes: also run on GPU, report max abs/rel diff of loss,
  gradients and weights after N steps, and compare with the old code's own
  run-to-run diff on GPU. The new diff must not exceed it.
- Turn the harness into pytest tests that run in under a minute.
- Run the existing test suites for the touched trainer plus the dry run
  (`python scripts/dry_run_plan.py`) before declaring done.

---

## 7. Targets

### 7.1 Branch A (first priority: runs 4x per plan)

**Measured at full scale (2026-09-26, production run, before it was stopped):**
- 529,723 batches/epoch (batch 128 → ~67.8M training samples), **~29 batch/s**,
  ~5 h per epoch, up to 15 epochs.
- **GPU: 246 MiB, 2% util. Main process 7.6% CPU, each of 4 DataLoader
  workers 14.5% CPU, system load 5.4 on 16 cores.** Nothing is busy, which
  means something is *waiting*.
- Main process RSS ~57% of RAM.

**Hypotheses to test (unverified; the profile decides):**
1. **Random reads from a memmap bigger than free page cache.** `shuffle=True`
   over ~68M samples; each `__getitem__` gathers `rows[start:end]` from
   `store.feats` (a memmap) at random positions. If the feature block does not
   fit in page cache, each sample is several major page faults, so workers sit
   in `D` state. Check majflt and `vmstat` `wa`.
2. **Per-sample Python overhead in `LazyHostSequenceDataset.__getitem__`**:
   five `torch.tensor` constructions, a dict, a string (`host_ip`), padding via
   `np.concatenate`, then default collate of 128 dicts in the worker and pickling
   to the main process. Fix direction: a batch-level `__getitems__` (PyTorch
   supports it) or a custom `BatchSampler` + one vectorised numpy gather per
   batch, keeping the **exact same sample order** (bit-identical).
3. **`model.task_gradient_conflict(...)` once per epoch and the syncing
   `guard.backward_step`**: switch to `backward_step_deferred` as the encoder did
   (bit-identical, proven there).
4. Tiny LSTM steps → kernel-launch bound once data is fixed: CUDA graphs over
   the fixed `[128, seq_len, 27]` shape.

Also check `_evaluate` (validation over ~1M+ samples each epoch) and the
calibration/conformal/operating-point stages after training: they ran on the
full validation and test splits and may have their own per-sample loops.

### 7.2 Branch B (`train_branch_b_live`)

Never run at full scale; nothing is measured. Batch 128, `shuffle=True`,
`num_workers` from the CLI, guard `backward_step` (syncing). Profile it the
same way. Likely the same dataset/loader pattern as Branch A.

### 7.3 DeepOP (`train_deepop_live` + `_precompute_rollouts`)

Never run at full scale. Two phases: `_precompute_rollouts` runs Branch B's
world model over every train/val sample (batch 1024, spills to disk), then
DeepOP trains at batch 64. Profile both phases separately. The rollout
precompute is pure inference: it can use larger inference batches without
changing any result (inference batch size does not affect outputs if the model
has no batch-dependent layers; **verify that bit-for-bit** before relying on
it), `torch.inference_mode`, and overlapped I/O.

---

## 8. Lessons and traps from the encoder work

- The GPU "utilisation %" lies: it reports any kernel running. Use profiler
  GPU busy time vs wall time.
- `.item()`, `float(tensor)`, `.cpu()`, `print(tensor)`, `if tensor:` and
  `torch.from_numpy(x).to(cuda)` (without pinned memory) each **synchronise**.
  One per batch is enough to serialise CPU and GPU.
- A spawned/forked worker's code hash or import side effects can differ from
  the parent's (a cache key once silently included `bita/train.py` because the
  hashing ran in a spawn child that re-imported the caller). Test that keys
  match across call sites.
- Two processes building the same on-disk cache entry raced and deleted each
  other's committed files. Use a per-entry file lock (see
  `data_unification/capture_columns.py::_entry_lock`).
- `git merge` silently dropped an `argparse` line once. After every merge,
  run the dry run.
- Parity tests catch what reasoning misses: numpy's `log1p` (SVML) differs
  from glibc's on ~25% of inputs, and numpy's mean/std sum in blocks of 8192.
  Compare outputs, do not assume.
- Resume files do not save every RNG (validation samplers restart from their
  seed after a resume). If you add state, add it to the resume file and test
  crash-resume.

---

## 9. Report (your final message, and a copy in your branch as `PERF_REPORT.md`)

For each trainer:
1. Profile findings before (numbers, one-sentence bottleneck).
2. Each change: what, the measured interleaved speedup (batch/s and host vs
   GPU ms/batch), and its equivalence level with evidence (bit-identical, or
   the max diffs vs the old code's own noise).
3. Final CPU-vs-GPU split per batch, and projected epoch time at full scale.
4. How to enable anything opt-in.
5. Risks, open issues, and what you would do next.

Branch name and last commit hash at the top. Do not push to anything but
`origin`, and do not merge into `v5.5o` yourself: the reviewer merges.

---

## 10. Context you should know but not act on

- **The encoder's category head is broken at full scale** (predicts one class
  for nearly every validation edge; link prediction is fine, val AUC ~0.99).
  That is an ML-quality issue tracked in `do_this_in_next_session_ml_review.md`.
  It is **not** your task. Branch A reads only the encoder's link-prediction
  embeddings (`get_host_embeddings`), not the category head.
- Fast-path level 4 + two lanes OOMs the GPU; production is one lane
  (`final_runner.sh` forces `LANES=1`).
