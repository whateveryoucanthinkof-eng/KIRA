# Downstream performance report (Branch A, Branch B, DeepOP, extraction)

Branch `perf/downstream-optimise` (from `v5.5o` @ 0173d2e). Not merged, not
pushed; for review. Last commit: see `git log -1` (this file is updated in it).

## Summary

| Stage | Before | After | How |
|---|---|---|---|
| Branch A training loop, full scale | 24.5-29 batch/s (~5 h/epoch) | **532 batch/s** (~17 min/epoch) | host-major batched loader, CUDA-graphed whole step, packed batches, 8 workers |
| Branch A data prep (extraction) | 2.5 h + 52 min | ~30-35 min (projected) | encoder fast path in extraction, one queue for all splits |
| Branch B loop | 47.8 batch/s (RAM) / loader 19.5 at full scale | 82 batch/s, GPU-bound | batched loader, graphs |
| DeepOP loop | 78 batch/s (RAM) / loader 37 at full scale | 298 batch/s, GPU-bound | batched loader, vectorised index, whole-step graph |
| Encoder | ~366-395 batch/s | unchanged | checked; no bit-identical change worth its risk |

Everything is bit-identical to the baseline (section "How it was measured").

**Rule kept throughout:** no quality trade-off. Batch sizes, data, epochs and
models are unchanged, and every change below is **bit-identical** to the
baseline (proof listed per change). Nothing that only matches to fp32
rounding is turned on.

## How it was measured

* **Equivalence:** the real trainers (`scripts/retrain_branch_a_live.py`,
  `scripts/retrain_future_models_live.py`) run end to end on the dry-run
  corpus (`scripts/dry_run_plan.build_corpus`) with the production flags and
  the production encoder, **baseline checkout vs this branch**. Checkpoints are
  compared leaf by leaf with exact equality: every weight, buffer, metric,
  per-class table, operating point, calibration, guard history and cross-year
  score. Done on CPU single-thread (0 and 2 loader workers) and on GPU (0 and
  2 workers). Result: **0 differences** for all three models
  (1,294-1,385 / 268 / 314-325 leaves). The GPU happens to be run-to-run
  deterministic for these models, so GPU old-vs-new is exact too.
* **Speed:** interleaved A/B on the same input. A full-scale synthetic store
  (`scripts/perf/synth_store.py`, 77.5M rows, 3.49M hosts, 68.2M Branch A
  samples, 8.4 GB memmap; its trajectory-length distribution matches the
  2026-09-26 train split: 67.8M kept / 6.2M dropped) and, for page-cache
  pressure, an 11 GB cgroup cap like the production trainer had. Real cached
  captures for extraction.
* **Cache safety:** the ingest-cache and capture-cache code hashes
  (`parse_code_hash`) are identical to the baseline's: nothing here
  invalidates the 8.4 GB / 21 GB caches.

## 1. Branch A

**Before (profile):** 529,723 batches/epoch at ~29 batch/s in production,
GPU 2 %. Reproduced at full scale: **24.5 batch/s, 1,660 major page faults per
batch** -- each sample's 15 rows were scattered over the 8.4 GB memmap (~13
random 4 KiB disk reads per sample). Once that was fixed, one Python thread
was pegged dispatching ~250 kernels per step (3.4 ms host vs ~1 ms GPU,
SM ~21 % at idle clocks).

| Change | Speed (interleaved) | Equivalence |
|---|---|---|
| Host-major copy of the feature block (`data_unification/host_major.py`) + batched vectorised gather (`gather_batch`) | loader 24.5 -> **349 batch/s** at full scale (majflt 1,660 -> 128) | `tests/test_branch_a_batched_loader.py` (every tensor of every batch, order, RNG, padding/gap/hazard) + end to end |
| `PermutationBatchSampler`: RandomSampler's order without `randperm(68M).tolist()` (~2.4 GB Python list per epoch) | memory | same seed draw, same batches, RNG state equal |
| Sync-free training step (`backward_step_deferred`), `effective_temperature` on device (2 syncs/step), cached `_tech_level` (sync H2D copy/step) | -- | end to end |
| Validation without syncs (`cyberworld_v4/device_hist.py` instead of bincount/boolean masks) | 327 -> 358 batch/s | `tests/test_branch_a_eval_hist.py` |
| Forward+backward as a CUDA graph (`cyberworld_v4/graphed_step.py`; warm-up RNG restored, partial batches eager) | step 3.39 -> 1.26 ms; loop 219 -> 374 | `tests/test_graphed_step.py` (dropout live) + end to end (87 graphed steps) |
| Guard snapshot/undo as CUDA graphs (`TrainingGuard(graph_undo=True)`) | 374 -> 418 batch/s | NaN-skip + optimizer-reload test + end to end |
| **Whole step as one CUDA graph** (`TrainingGuard.deferred_step_graphed` + `WholeStepGraph`: forward, loss, backward, clipping, finite checks, snapshot; only Adam eager) | 422 -> 518 batch/s | 150-step test with inf-injected batches, partial batch, epoch boundary, optimizer reload: identical losses, weights, counters; end to end |
| **One buffer per batch** (`host_major.pack_batch` / `unpack_batch`: one shm handle, one pin, one H2D copy) | 509 -> **770 batch/s** | pack/pin/unpack test + loader order test + end to end |

Branch A loop overall (store in RAM, same input, interleaved): **211 -> 770
batch/s (3.6x)**; at full scale the old loop was I/O-bound at 24.5-29 batch/s,
so **~25x**. Host 1.30 ms/batch vs ~1 ms GPU: close to the GPU.

**Measured at full scale** (77.5M-row synthetic store, 8.4 GB memmap, 11 GB
cgroup like the production trainer, the whole loop: loader + graphed step):
4 workers 414, **8 workers 532**, 12 workers 540 batch/s. The plan now defaults
to `NUM_WORKERS=8` (worker count never changes batches or order). A
529,723-batch epoch: **~17 min** (was ~5 h); in RAM the loop reaches 770 batch/s,
so the remaining limit at full scale is memmap reads (page cache).

## 2. Branch B

**Before:** never run at scale. Loader at full scale **19.5 batch/s** (2,094
major faults/batch); training loop 47.8 batch/s even in RAM.

| Change | Speed | Equivalence |
|---|---|---|
| Host-major batched gather (20 contiguous rows per sample) | loader 19.5 -> **274.6 batch/s** at full scale | `tests/test_branch_b_deepop_batched_loader.py` + end to end |
| Sync-free step, sync-free validation histograms, CUDA-graphed step, graphed guard undo | loop 47.8 -> **81.6 batch/s** | end to end |

**Now GPU-bound:** 11.5 ms of GPU work per step; half of it is the fp32
memory-efficient attention kernel (15 attention calls per step: 3 layers x 5
autoregressive rollout steps, no KV cache), backward dominant. Faster
attention kernels or a KV cache would change rounding and/or dropout masks:
**not done** (opt-in candidates, need your approval).

## 3. DeepOP

**Before:** loader at full scale **37.0 batch/s** (1,081 major faults/batch);
the index build had a per-window Python loop (309 s at full scale);
`target_token_histogram` looped over every sample (~110M) and is called three
times per run; ~21 `vocab.encode` per sample in the loader; the loss had two
syncs per step (`bool(sup.any())`, boolean column mask).

| Change | Speed | Equivalence |
|---|---|---|
| Host-major batched gather + per-row token table | loader 37.0 -> **467 batch/s** | batched-loader tests + end to end |
| Oversampling mask from a cumulative sum; vectorised histogram | index 309 s -> 99 s; histogram minutes -> seconds | compared with the old loops in the tests |
| `smoothed_and_plain_ce(support_index=...)` | removes 2 syncs/step, makes the loss capturable | values and gradients equal (test) |
| CUDA-graphed step + graphed guard undo | loop 78.0 -> 257 batch/s | end to end (77 graphed steps) |
| Whole step as one CUDA graph | 257 -> **295 batch/s** | end to end (76 of 78 steps graphed) |
| Validation: `forecast_sequence(decode_names=False)` (it decoded every token to a name with one `.cpu()` per sample, then discarded them), sync-free scorer | ~300k val batches/epoch at full scale no longer sync 64x each | end to end + 98 DeepOP tests |

**Now GPU-bound:** a py-spy profile shows the main thread waiting on the
previous step's GPU event (31 %); packed batches change nothing (296 -> 298).

## 4. Extraction (data prep before every Branch A / B / DeepOP run)

**Before:** train extraction took **2.5 h** and validation **52 min** in the
2026-09-26 Branch A run, and Branch B/DeepOP repeat it. cProfile: the encoder
ran its *reference* streaming code -- 45 % in a per-node Python loop
(`message_aggregator.aggregate`), ~200k small `torch.stack` calls.

| Change | Speed | Equivalence |
|---|---|---|
| Extraction encoder on the encoder's own fast path, level 1 (`data_unification/fast_extract.py`) | CTU-13 #7 8.8 -> 5.1 s; CTU-13 #4 114.6 -> 55.6 s; PCAP fri_16 (12M records) **1,010 -> 352 s** | `tests/test_fast_extraction.py` (all formats + real CTU-13, chained) + end to end |

| Train, val and test extracted through ONE parallel queue (`extract_parallel(exposure_for=...)`) | workers no longer idle at each split's tail | end to end, store/exposure log lines identical |

**Projected:** ~12,500 CPU-s of extraction -> ~4,800 CPU-s; with the default
3 workers ~30 min for all three splits (was 2.5 h + 52 min + test).

Rejected after measuring: more torch threads per extraction worker
(2 threads: 2.3e-6 differences, only 10 % faster).

## 5. Encoder (checked, as asked)

Full corpus, production flags (level 4 + batch planner), one lane: ~300
batch/s and still climbing at 116k batches; **GPU SM ~41 %, main process
pegged at ~97 % of one core** -- host-bound like the downstream trainers were.
Main-thread profile: memory-overlay reads (`fast_read`) ~15 %, waiting on the
planner ~9 %, the guard's per-tensor undo loop ~5 %, eager BiTA transformer.

Tried: the guard's snapshot/undo graphs for the encoder. Interleaved at full
scale (production flags, 30k-batch steady-state windows): off 365.9 / 394.7,
on 361.4 / 365.9 batch/s -- no gain (the encoder steps its optimizer once per
8 batches, `backprop_every=8`, so the guard is not on its critical path).
**Reverted: the encoder runs exactly as before.** Its remaining host costs
(memory-overlay reads, planner hand-off, eager BiTA transformer) are inside the
previously tuned fast path; I did not find a change there that is both
bit-identical and worth its risk. The encoder baseline is not GPU-deterministic
(two baseline runs differ by up to 2.2e-5 in weights from atomics), which is
worth knowing for any future change to it.

## 6. Second round: validation, disk, scheduling

| Change | Measured | Equivalence |
|---|---|---|
| Validation passes as CUDA graphs (`graphed_step.GraphedBody`: forward + every accumulator per batch; host counters outside; DeepOP scorer split into `update_device` / `update_host`) | Branch A 480 -> **1,540** batch/s; Branch B 154 -> **281**; DeepOP 47.9 -> **269.5** (full scale: ~1.7 h -> ~19 min per DeepOP epoch) | end to end, CPU + GPU, 0 + 2 workers: identical in every leaf |
| One feature block on disk: after the host-major copy the store reads rows through it (`RowMappedFeats`) and the original spill block is released | -8.4 GB disk per downstream / Branch A run | every access pattern tested; end to end with memmap-forced stores identical |
| DeepOP rollout cache computes/stores each distinct window once (oversampled duplicates mapped), each at its original batch size; RNG draw preserved | dry-run corpus -24 % rollouts; at full scale roughly the attack-window share of the train set, tens of GB -> less | dedup cache == original for every sample, RNG state equal; end to end identical |

**Disk:** the DeepOP stage at full scale (rollout cache ~44-50 GB train +
~10 GB val, plus spill + host-major copy) needed ~72 of ~75 GB free. With the
two disk changes it is roughly 50-55 GB.

**Measured and rejected:**
* Running Branch B / DeepOP alongside the later encoder runs. Full-scale encoder
  + Branch B at once: encoder ~330 -> ~160-215 batch/s, Branch B 79 -> 47 --
  ~10 % more combined throughput, both runs much longer, more RAM risk. At
  steady state the encoder keeps the GPU ~72-78 % busy (the 41 % measured
  earlier was its warm-up), so the two compete for the GPU.
* Deeper encoder work. Its remaining host costs are spread thin (memory-overlay
  reads ~15 %, planner hand-off ~9 %, guard ~5 % of the main thread) with the
  GPU ~75 % busy: single-digit gains at best, inside a fast path that cannot
  run on CPU and is not GPU-deterministic, so bit-identity could not be proven.

## Estimated total plan time

Early stopping decides the epoch counts; typical 5-8 epochs assumed.

| Stage | Per epoch | Per run | Runs | Total |
|---|---|---|---|---|
| Encoder | ~20 min + validation | ~3-4 h | 4 | ~12-16 h |
| Branch A | ~17 min train + ~1.5 min validation; extraction ~35 min | ~2-3 h | 4 | ~8-11 h |
| Branch B | ~1.85 h train + ~9 min validation; extraction ~35 min | ~8-13 h | 1 | ~8-13 h |
| DeepOP | ~1.2 h train + ~19 min validation; rollout precompute ~45 min | ~6-10 h | 1 | ~6-10 h |
| **Total** | | | | **~34-50 h** |

## How to switch things off

All on by default (all bit-identical):

* `--legacy-loader` (both trainers): the old per-sample loaders (no packing).
* `CYBERWORLD_EXTRACT_ONE_QUEUE=0`: Branch A extracts split by split.
* `NUM_WORKERS=4` in the plan env: the previous loader worker count.
* `CYBERWORLD_CUDA_GRAPH=0` also turns off the validation graphs.
* `--no-cuda-graph` (Branch A) or `CYBERWORLD_CUDA_GRAPH=0` (all): eager steps.
* `CYBERWORLD_GUARD_SYNC=1`: the old syncing guard step.
* `CYBERWORLD_FAST_EXTRACT=0`: the reference extraction path.

## Tried and rejected (measured)

* Background prefetch thread for the batch handoff: 412 -> 395 batch/s (GIL).
* Guard snapshot/undo graphs for the encoder: no gain (it steps once per 8 batches).
* More loader workers beyond 4-8: no gain once the main thread is the limit.
* Multi-threaded extraction workers: not bit-identical.

## Verification run at the end

* `scripts/dry_run_plan.py` (the whole plan on the synthetic corpus: 4
  encoders, 4 Branch A, Branch B, DeepOP): **DRY RUN PASSED**.
* Full test suite: **1,207 passed**, 4 failed + 3 modules fail to collect.
  All 7 also fail on the baseline: `test_split_manager_uses_the_lock` (2) only
  because the suite ran as root with `HOME=/root` (pass with the real HOME);
  `test_site_config` (2), `test_control_backend`, `test_serving_replay_isolation`,
  `test_topology_service`: the served encoder `saved_models/...unified_final.pth`
  is feature schema 1.0.0, the tree is 2.0.0 (already in
  `do_this_in_next_session_ml_review.md`). The fix is promoting a schema-2.0.0
  encoder, which the plan leaves to a human on purpose (run_training_plan.sh
  prints the copy commands); the only 2.0.0 encoder on disk is the epoch-1 one
  with the collapsed category head, so it was not promoted.
* Parse-cache code hashes unchanged (ingest `6fd03474...`, columns `b056d0ee...`).

## Open issues / notes for the reviewer

* **Shared-setup key mismatch (pre-existing):** with the current code (baseline
  and this branch alike) the encoder's shared-setup key is `3680d22b...`, not
  the `82b65a30...` on disk, so the next production encoder run rebuilds the
  setup store once (~7 min, ~20 GB). The parse-code hash is unchanged, and
  none of the key's source files differ from 0173d2e; the store on disk was
  built from a state that is not the committed one.
* Branch B and DeepOP are GPU-bound in fp32 kernels (attention). Faster
  kernels exist (other SDPA backends, KV cache, lower precision) but change
  floating-point results or dropout masks. **Not done, by your rule: no
  quality loss of any kind, fp32 rounding included.**
* Extraction worker memory: peak RssAnon 3.4 GB on a 12M-record PCAP day
  (largest days ~17M). Three workers already use most of the 22 GB laptop;
  more workers would need the 32 GB machine.
