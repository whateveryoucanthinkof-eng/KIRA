# 32 — The training guard, and what the end-to-end dry run found

**Date:** 2026-09-24
**Why:** each training run costs hours and the owner cannot afford a run lost to a bug that only
shows after extraction. Two things were added before training: one policy that all four trainers
follow, and a dry run that exercises the whole plan on real-format data. The dry run found seven
defects that would have damaged the real run. All of them are fixed and pinned by tests.

---

## 1. The training policy (`cyberworld_v4/training_guard.py`)

The encoder, Branch A, Branch B and DeepOP all train under the same `TrainingGuard`:

| When | What happens |
|---|---|
| every step | linear LR warmup over the first ~5% of epoch 1 (at most 500 steps); gradient-norm clipping; a non-finite loss or gradient is **skipped** (no optimizer step) and counted |
| epoch improves | snapshot of weights **and** optimizer state, kept in memory |
| **2** epochs without improvement | **step back**: restore the best snapshot, halve the LR, log a diagnosis, continue |
| **3** epochs without improvement | stop. The best epoch is what is saved |
| unstable epoch (NaN score, or > 1% of steps skipped) | step back immediately |

After a step-back the model gets one epoch at the halved LR. If that epoch also fails to improve,
patience 3 stops the run. The step-back limit is 2, and the LR floor is 1e-6.

### What the log tells you

One line per epoch:
```
[encoder guard] epoch 4: score=0.61 best=0.63 (epoch 2) | lr 5.00e-05 | grad norm mean 58 max 97 clipped 0% | skipped 0/812 | -> STEP_BACK (restored epoch 2, lr -> 5.00e-05)
```
On a step-back or stop, the guard also writes a diagnosis. It covers:
- whether training loss is still falling (over-fitting), flat (not learning) or rising (steps too large);
- gradients that are zero or too large;
- **which tensor carries the gradient** (`largest gradients: ...`);
- each trainer's own health checks, for example "technique head predicts 1 of 3 classes" or
  "world model is not better than persistence".

Every checkpoint stores the guard's full history: `*_training_guard.json` for the encoder, and
`ckpt["training_guard"]` for the others.

### Settings by model

| Model | Selection score | Clip | LR | Epoch ceiling |
|---|---|---|---|---|
| Encoder | 0.5 · inductive AP + 0.5 · val macro-F1 (max) | 100 (a healthy step's norm is ~60) | 1e-4 | 50 |
| Branch A | `--select-on` score, composite by default (max) | 1.0 | 1e-3 | 15 |
| Branch B | validation loss (min) | 1.0 | 1e-3 | 12 |
| DeepOP | free-running macro-F1, else −validation CE (max) | 1.0 | 5e-4 | 12 |

Epoch counts are ceilings. The guard usually ends the run earlier.

### Crash recovery (`ResumePoint`, same module)

A crash, power cut or OOM used to send every trainer back to epoch 0. Each trainer now writes a
resume point after every epoch. It holds:
- the weights and the optimizer state;
- the guard, including its best snapshot;
- the RNG states.

The write is atomic (temp file, then `os.replace`), so a crash during the write leaves the previous
epoch's file intact. **Re-running the same command continues after the last finished epoch.** A
successful run deletes its resume point.

| Trainer | Resume file |
|---|---|
| Encoder | `<save_dir>/<prefix>-<data>_resume.pt` |
| Branch A | `<output stem>_resume.pt` |
| Branch B, DeepOP | `branch_b/host_wdt_resume.pt` and `deepop/cwa_forecast_decoder_resume.pt`, kept until the whole downstream run finishes, so a DeepOP crash does not retrain Branch B |

Rules:
- A resume point from a run with **different arguments** is set aside as `.stale` and never
  loaded. The epoch ceiling and the worker count may change between attempts.
- An unreadable file is set aside as `.corrupt`.
- `--no_resume` (encoder) or `--no-resume` (the others) starts fresh.
- Feature extraction re-runs on a restart; the finished epochs do not.

Verified by killing each real trainer right after epoch 1 on the dry-run corpus
(`CYBERWORLD_TEST_CRASH_AFTER_EPOCH=1`) and re-running it. All four resumed at epoch 2 and finished.
That test found a bug in the downstream cleanup: a `NameError` on the last line of a successful
run. In a unit test, a run resumed after epoch 2 ends with the same weights as an uninterrupted one
(`tests/test_training_resume.py`).

The encoder keeps only its best and current per-epoch checkpoints; it used to keep all 50.

### Disk space

Preflight refuses to start with less than `MIN_FREE_GB` (default 50) GiB free under `OUT`. That
directory holds the spill stores, checkpoints and resume points.

## 2. The dry run (`scripts/dry_run_plan.py`, also the `dryrun` stage of the plan)

The dry run writes a synthetic corpus in the exact on-disk formats the plan reads:
- libpcap captures (Ethernet/IPv4/TCP);
- CIC-2018 label CSVs on the 12-hour clock with the UTC+4 offset;
- CIC-2017 TrafficLabelling CSVs;
- CTU-13 `.binetflow` files, named as `splits.lock.json` names them.

It then runs `scripts/run_training_plan.sh` with only the epoch counts shrunk: preflight, both encoder
arms, Branch A, the extra seeds, Branch B, DeepOP and summary. It fails if any stage fails, if any
checkpoint lacks its guard record, or if any log holds a traceback. The numbers it prints mean nothing
because the data is tiny and synthetic. What it proves is that the plan runs end to end.

`run_training_plan.sh all` runs it first. Set `SKIP_DRY_RUN=1` to skip it.

## 3. What it found

| # | Symptom in the dry run | Cause | Fix |
|---|---|---|---|
| 1 | Encoder validation crashed: "Trying to update memory to time in the past" | nodes were keyed by IP across captures, so a host's memory clock came from another day's capture | nodes keyed by `(capture, ip)` (`bita/train.py`, both loader paths) |
| 2 | `time_encoder.w` held 100% of the gradient norm (~1e7) and every step was clipped | a node's first memory message encoded the **absolute** Unix time (memory starts at `last_update = 0`); negatives came from any capture or year, e.g. a 2011 CTU host scored at a 2018 timestamp | first contact encodes dt = 0 (`tgn.get_raw_messages`); negatives come from the positive edge's own capture (`RandEdgeSampler(node_group=...)`) |
| 3 | A 30 s gap computed as 0 | float32 steps are **128 s** at 1.5e9 | absolute times are float64 end to end: neighbour finder, `memory.last_update`, message times, BiTA aggregator. Only differences go to float32 |
| 4 | After #3, `time_encoder.w` held **91–97%** of the squared gradient norm, so the global clip set the step size of the whole model | d/dw cos(w·dt) = −dt·sin(w·dt) grows with dt in seconds, and real captures have deltas of hours | **fixed time encoding** (GraphMixer, ICLR 2023): the TGAT frequencies are frozen. `--learn_time_encoding` restores training them. After: gradient norm mean 31, max 61, 0% clipped (was mean 93–128, max 361, 50% clipped) |
| 5 | Inductive AUC and AP were **exactly 0.5000** every epoch, and that is half the encoder's selection score | in a capture the new-node pool was one server, so every negative *was* the positive destination | the sampler never returns the positive destination; inductive pools use the whole capture. Inductive AP now moves: 0.70 → 0.75 over two epochs |
| 6 | Branch A gradation predicted "1 Recon/Unknown" for 100% of held-out test, while the technique head separated 3 of 3 classes | the paper's scalar gradation is MSE-trained and was decoded to the nearest level. MSE predicts the mean, and an uncertain window between Benign (0) and attack (2/3, 1) lands on 1/3 | the discrete level is the **technique head's probabilities summed per level** (`TECHNIQUE_GRADATION`, checked against every label map). The scalar head and its 0.2-weighted MSE loss are unchanged. The state dict is unchanged, so serving still loads strictly |
| 7 | Branch A crashed under `cross_year_ctu` | `capture_namespace()` was given a `Capture` object, not a path | accepts both; the downstream CTU extraction also got capture namespaces |

Also fixed:
- the encoder saved its **last** epoch rather than its best;
- the comparison harness returned 0 when an arm failed;
- each arm's metrics CSV and curves overwrote the previous arm's.

## 4. What was removed or changed, and why

- **Removed: the trainable time-encoding frequencies** (#4). A fixed encoding cannot run away, and
  published results show it is at least as good.
- **Removed: the nearest-level decode of the scalar gradation** (#6). Gradation is a fixed function
  of the technique label, so a separate decode could only disagree with the technique head.

Considered and **kept**:
- **TGN memory, on.** This is the owner's decision (analysis 30). Validation no longer crashes (#1),
  and the guard's health check reports a category head that collapses onto one class.
- **Clip 1.0 for Branch A, Branch B and DeepOP.** In the final dry run:
  - Branch B clipped 67% of steps in epoch 1 and 0% in epoch 2;
  - DeepOP clipped 15% in epoch 1 and 0% in epoch 2;
  - Branch A clipped 0% in epoch 1 and 43% in epoch 2, with a mean norm of 1.12, just above the clip.

  Clipping at this level under Adam rescales steps but does not slow learning. If a real run logs
  `largest gradients:` every epoch, the guard names the tensor responsible.

## 5. Status

- Tests: 1034 pass. The failures are the same 3 environmental ones as before, plus one
  collection error. `test_serving_replay_isolation` needs a retrained encoder, which this run
  produces.
- Dry run: **passes** end to end, covering both IP arms and three seeds for the encoder and Branch A,
  then Branch B and DeepOP. Every model improved in epoch 2 except one Branch A arm (seed 42,
  `cross_network`), whose score fell 0.585 → 0.566. The guard kept epoch 1 for it, as designed. Every
  checkpoint carries its guard record. The numbers are from 2-epoch runs on synthetic data and say
  nothing about real performance.
