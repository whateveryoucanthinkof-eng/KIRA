# 26 — The downstream retrain was thrashing, not running

**Date:** 2026-09-22
**Found by:** asking how long training had left, and not being able to answer.

## Symptom

`downstream-retrain.service` had been "running" for 38 minutes with no output
since the record-loading line. The tells:

```
memory.current  = 17G       (== memory.max)
memory.events   high 983985
memory.pressure full avg10=57.68
CPU             38.3%
```

Throttled 983,985 times and **fully stalled on memory 57.7% of the time**,
pressed against the hard limit. It was not slow; it was not going to finish.

## Cause

The CSV path did this:

```python
train_records, val_records = load_records(...)          # every record, one list
train_traj = extractor.extract_trajectories(train_records)
```

`load_records` returns an entire split as a Python list. At full density that
is **32,461,461 records (~9.7 GiB at ~300 B each)**, and the trajectory store
was then built on top of it, so both were resident together.

`py-spy` confirmed the phase (`extract_trajectories -> process_records`), and
`/proc/<pid>/fd` showed **no spill file open at all** — `--spill-dir` was
accepted and passed to the extractor, but the CSV path never reached a builder
that used it.

Branch A already loads per capture for exactly this reason. That change was
made earlier in the session and never ported here.

## Fix

`main()` now streams: `split_capture_files()` resolves the frozen split, and
`_store_per_capture()` parses one capture, feeds the shared
`TrajectoryStoreBuilder(spill_dir=...)`, frees the records and collects, then
finalises. Peak becomes the largest single capture rather than a whole split,
and the feature block goes to the spill memmap.

Per-capture extraction is also more faithful to the data: a CIC-2018 day and a
CTU-13 scenario are unrelated networks, and one merged TGNE neighbour graph
would make their hosts each other's temporal neighbours.

## Verified identical, then measured

The streaming path must not change *what* is trained on:

| | Branch A (already streaming) | downstream (now streaming) |
|---|---|---|
| frozen split | 17 train / 3 val / 3 test | 17 train / 3 val |
| records | 32,461,461 train + 1,035,053 val | 32,461,461 + 1,035,053 |

Identical. One capture read in isolation: `wed_29_csv.csv`, 602,331 records,
**0.98 GiB peak**.

After relaunch:

| | before | after |
|---|---|---|
| RSS | 17.0 GiB (at hard cap) | **1.45 GiB** |
| throttle events | 983,985 | **0** |
| fully stalled | 57.7% | **0.00%** |
| CPU | 38% | **247%** |

It is now CPU-bound, which is what it should have been all along.

## Note on the lost 38 minutes

The thrashed attempt's log is kept at `logs/downstream_retrain.thrashed-attempt.out`.
Nothing was written to `saved_models/` — the run never reached training, so the
v3 Branch B and DeepOP checkpoints are untouched and no backup was consumed.
