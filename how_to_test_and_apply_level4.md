# How to test fast-path level 4 and switch production to it

Production runs `--fast_step --fast_step_level 3 --batch_planner`. Level 3 is bit-identical to level 2.
Level 4 is roughly 2× faster (1.5M-node input, with the planner: ~236 batch/s at level 3
against ~471 at level 4). It is NOT enabled yet, because it has only been tested on small
inputs.

## What level 4 changes

- The BiTA aggregator and the GRU memory updater also run as replayed CUDA graphs, on
  padded shapes (message sequences up to 32; longer batches take the normal path).
- Numerics: fp32 rounding, not bit-identical. Measured over 38 steps plus validation:
  loss rel 1.6e-7, gradients 6e-8, weights 5.9e-6, validation equal to 6 digits.
  That is smaller than level 2's own distance from the reference, and
  `tests/test_fast_tgn_graphs.py` asserts it.

## Already tested (2026-09-25)

- Rounding-level equivalence on ctu7 and on the 1.5M-node synthetic input (`bita/fast/bench.py --compare`).
- Bit-identity with the batch planner at level 4 (`tests/test_batch_planner.py`).
- GPU test suite (65 tests), including pool growth and compaction at level 4.

## NOT tested yet: do these before applying

1. **GPU memory at production scale, two lanes, through a validation peak.**
   At level 3, each encoder reaches ~3.5-3.8 GB at the epoch-end validation (8 GB GPU).
   Level 4 adds graph pools for BiTA and the updater.
   - Run the real plan at level 4 for one full epoch (~35-40 min at level 4) with both lanes,
     through the end-of-epoch validation.
   - Watch `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` every 30 s, and
     `scripts/ops/watchdog.py`, which alerts at 5.5, 6.5 and 7.4 GB total.
   - Pass: the peak total stays under ~7.4 GB, with no `OutOfMemoryError`. If it fails, run
     one lane at level 4 (a single level-4 process is about as fast as two level-3 lanes).
2. **The illegal-memory-access issue.** The optimisation work hit
   `CUDA error: an illegal memory access` when graphs of several models were captured
   and destroyed in one process, together with `torch.cuda.empty_cache()`. The workaround
   (capture without `empty_cache`, and raise if a weight is re-allocated) is in, and
   production runs one model per process, but the root cause is unknown.
   - Run `CUDA_LAUNCH_BLOCKING=1 python -m pytest tests/test_fast_tgn_graphs.py -x` 5 times,
     and a 2-epoch ctu7 run at level 4 with `compute-sanitizer --tool memcheck`
     (from the CUDA toolkit; install inside the `cyberworld_claude` toolbox).
   - Pass: no memcheck errors, and every repeat passes.
3. **Long sequences.** Batches whose per-edge message sequences exceed 32 fall back
   to the eager path. At full scale, hub nodes have long sequences.
   - During the step-1 run, log how many batches fell back (add a counter in
     `bita/fast/graphs.py`), and confirm the equivalence harness exercised both paths.
4. **End-to-end numerics at scale.** After the one-epoch level-4 run, compare its epoch-1
   validation metrics against the level-3 run's epoch-1 metrics. They should agree to
   seed-noise level (differences well below the 3-seed spread).

## Applying it

1. Stop at an epoch boundary: the plan resumes from the last finished epoch, so
   nothing trained is lost.
2. In `~/.config/cyberworld/plan_env.sh`, set
   `PLAN_ENCODER_EXTRA="--fast_step --fast_step_level 4 --batch_planner"`.
3. Relaunch with `./final_runner.sh`. The level isn't part of the resume fingerprint,
   so the encoders continue from their checkpoints.
4. Watch the first validation peak. If the GPU runs out of memory, switch back to level 3.
