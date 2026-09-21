# Autonomous session plan — 2026-09-21 ~15:40 onward

User is away ~4-5 hours with instruction: **do not stop; keep improving,
training, retraining, experimenting.** This file is the resume point if
context is lost. Read [`13_MASTER_STATE.md`](13_MASTER_STATE.md) first.

## Running order

| # | stage | unit name | log | done when |
|---|---|---|---|---|
| 1 | Exp A baseline | `tgne-expA` | `logs/expA_baseline.out` | epoch 0 recorded |
| 2 | Exp B perf | `tgne-expB` | `logs/expB_perf.out` | epoch 0 == A's |
| 3 | Exp C batch size | `tgne-expC*` | `logs/expC_bs*.out` | epoch 0 compared |
| 4 | Full TGNE | `tgne-final` | `logs/tgne_final.out` | early stop / 30 epochs |
| 5 | Branch A | `branch-a-retrain` | `logs/branch_a.out` | held-out test scored |
| 6 | Branch B + DeepOP | `future-models-retrain` | `logs/future_models.out` | both saved |
| 7 | Validate + parity | — | — | xfails flip, parity passes |

## Launch template — one heavy job at a time

```
systemd-run --user --unit=<NAME> -p MemoryMax=18G -p MemoryHigh=16G \
  -p MemorySwapMax=0 -p WorkingDirectory=$PWD -p Environment="PYTHONPATH=$PWD" \
  -p StandardOutput=append:$PWD/logs/<LOG> -p StandardError=append:$PWD/logs/<LOG> \
  /var/home/samito/.pyenv/versions/3.12.14/bin/python3 -u <script> ...
```

`systemctl --user reset-failed <NAME>` before relaunching a unit that died.
Absolute pyenv path — `python3` resolves to `/usr/bin/python3`, no torch.
Never write outputs to `/tmp` (tmpfs; a reboot already destroyed checkpoints).

### Exp B
```
--train_splits train --ingest_workers 6 --gpu 0 --n_epoch 2 --patience 5 --seed 0
--data_name expB
```
(vectorised sampler is the default; A set `TGNE_REFERENCE_SAMPLER=1`)

### Exp C — one batch size per run, sequentially
`--batch_size 64` then `--batch_size 256`, everything else as B.

### Stage 4 — full run
As B but `--n_epoch 30`, `--data_name unified_final`.

### Stages 5-6
See [`16_downstream_retrain_runbook.md`](16_downstream_retrain_runbook.md).
No `--stride`, no `--rows-per-file`: `require_full_density()` refuses.
Branch A takes `TGNE_CHECKPOINT_PATH=<ckpt>` as an env var, not a flag.

## Gate before spending hours downstream

From epoch 0 of stage 4:
- **inductive val AUC > 0.65** (0.5043 was chance; 0.8330 measured once)
- **CatAcc must move between epochs** (0.3327 -> 0.3326 was a constant prediction)
- **focal alpha: no weight exactly 1.0** — that means a class has ZERO
  training samples
- **per-class val acc** — aggregate CatAcc is ~80% Benign and hides collapse

If the gate fails, diagnose rather than proceeding.

## Known limitations — do not try to fix these

- **Recon is not evaluable.** One source host (172.16.0.1) carries all 158,930
  records. The code warns each run. CIDDS-001 would fix it; not now.
- **9 of 10 CIC-2018 days have no IP columns.** User is bringing fresh PCAPs.
- 266 corrupt PCAPs — user's side.

## If everything above completes

Useful experiments, in value order:
1. `n_degree` sweep (10 -> 20): more neighbours per node, accuracy vs time.
2. Prefetch/pipeline the sampler so it overlaps GPU compute (lossless).
3. `torch.compile` on the TGN forward — attacks the ~87% of batch time that is
   launch overhead. Verify equivalence before trusting.
4. Wire the PCAP bridge into Branch B/DeepOP (`--pcap-root`) and compare
   against the CSV path — real per-host trajectories vs fabricated identity on
   9 of 10 days.
5. Re-check `verify_offline_live_parity.py` end to end.

**Always: one heavy job at a time; measure before and after; a
performance change must be provably lossless or it is an accuracy change.**
