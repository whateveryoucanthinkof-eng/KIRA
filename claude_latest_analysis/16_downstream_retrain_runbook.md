# Downstream retrain runbook — run this the moment TGNE finishes

**Prerequisite:** `tgne-retrain` completes and writes
`saved_models/bita_bigru_transformer-unified_final.pth`.

Everything below assumes the fixes committed on 2026-09-21 are in place: the
12-hour clock repair, IP node features, the frozen split wired into every
trainer, streaming clock detection, and the contract unification. See
[`13_MASTER_STATE.md`](13_MASTER_STATE.md).

---

## 0. Launch discipline — read before running anything

**One heavy job at a time.** The TGNE run was OOM-killed once because a smoke
test ran alongside it. Check `free -g` first and confirm at least 10 GiB
available.

```
systemd-run --user --unit=<name> -p MemoryMax=16G -p MemoryHigh=14G \
  -p MemorySwapMax=0 -p WorkingDirectory=$PWD -p Environment="PYTHONPATH=$PWD" \
  -p StandardOutput=append:$PWD/logs/<name>.out \
  -p StandardError=append:$PWD/logs/<name>.out \
  /var/home/samito/.pyenv/versions/3.12.14/bin/python3 -u <script> ...
```

Absolute pyenv path — `bash -c python3` resolves to `/usr/bin/python3`, which
has no torch. Never write outputs to `/tmp`; it is tmpfs here and a reboot
already destroyed one set of finished checkpoints.

---

## 1. Gate: is the encoder worth building on?

Before spending hours downstream, read the TGNE result:

```bash
grep -E "Epoch [0-9]+ \[|Early stopping|FINAL TEST" logs/tgne_fixed_retrain.out | tail -20
```

Two numbers decide it.

| metric | before the fixes | what to require |
|---|---|---|
| **Inductive val AUC** | 0.5043 (chance) | **> 0.65.** Below ~0.60 the IP node features did not deliver and the encoder still cannot generalise to unseen hosts. Stop and diagnose rather than training three models on it. |
| **Val CatAcc** | frozen 0.3327 → 0.3326 | must **move between epochs**. A value stable to four decimals is a constant prediction, i.e. the head is still collapsed. |

Transductive val AUC being ~0.99 means little on its own — that was true while
the model was a lookup table.

If the gate fails, the likely next moves are: raise the category-head loss
weight relative to the edge loss, check the focal alpha actually spans the five
classes present (`grep "focal-loss per-class" logs/...`), or widen the node
features beyond the 12 intrinsic IP ones.

---

## 1b. Pick the encoder epoch — do NOT just take the final checkpoint

```bash
python scripts/select_best_encoder.py logs/tgne_final.out \
    --copy saved_models/bita_bigru_transformer-unified_final.pth
```

The two objectives diverge on this corpus. Measured over six epochs:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| val_ap | .9971 | .9968 | .9968 | .9973 | .9974 | **.9975** |
| inductive AP | .9926 | .9921 | .9918 | .9935 | **.9936** | .9934 |
| C2 inductive | .325 | .319 | .517 | .565 | **.586** | .480 |
| InitialAccess | .505 | **.714** | .578 | .612 | .561 | .510 |

`val_ap` rises to the end while classification degrades after epoch 4, so the
final checkpoint is **not** the best one. Every epoch is saved, so the choice
is made after the fact: the script scores
`0.5 × inductive AP + 0.5 × macro recall` and prints what `val_ap` alone would
have picked, so the gap is visible.

Then export `TGNE_CHECKPOINT_PATH` to the winning checkpoint for stages 2-3,
and make sure a matching `<checkpoint>_config.json` sits beside it --
`build_or_load_tgne_ta` reads it and will refuse a mismatched one.

## 2. Branch A

```bash
TGNE_CHECKPOINT_PATH=saved_models/bita_bigru_transformer-unified_final.pth \
python scripts/retrain_branch_a_live.py \
  --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
  --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
  --output saved_models/branch_a/branch_a_lstm.pt \
  --epochs 8 --batch-size 128 --spill-dir .spill \
  --spill-dir .spill
```

- The checkpoint comes from `TGNE_CHECKPOINT_PATH`, **not** a `--tgne` flag —
  this script has no such flag.
- The split is read from `splits.lock.json` (17 train / 3 val / 3 test
  captures). It is no longer a sorted 70/15/15 slice.
- **No `--stride`, no `--rows-per-file`.** Full density is the default and
  `require_full_density()` refuses to start otherwise. If you genuinely need a
  smoke run, set `CYBERWORLD_ALLOW_SUBSAMPLING=1` — and the numbers from it are
  not a result.
- Expect the **credibility gate** to print a verdict. It is advisory, not
  fatal. Treat "model accuracy does not beat persistence" as a real failure —
  a smoke run already produced 0.7221 against a 0.9666 persistence baseline.
- The held-out test split is scored **once**, on the restored best checkpoint.

## 3. Branch B + DeepOP

```bash
python scripts/retrain_future_models_live.py \
  --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
  --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
  --tgne saved_models/bita_bigru_transformer-unified_final.pth \
  --out-dir saved_models \
  --epochs 6 --spill-dir .spill --num-workers 4
```

`--out-dir saved_models` puts the two checkpoints exactly where
`control_backend/model_adapter.py` loads them:
`saved_models/branch_b/host_wdt.pt` and
`saved_models/deepop/cwa_forecast_decoder.pt`. Any existing file is copied to
`<stem>.superseded-<UTC>.pt` before training starts. (They previously went to
`host_wdt.canonical-tgne.pt`, which nothing ever read -- see report 23.)

Until both finish, the adapter refuses to load at all --
`Checkpoint contract conflict on history_steps: branch_b says 5, an earlier
checkpoint said 15` -- because Branch A is already v4. That is expected. It is
also what keeps the three adapter test modules skipped and
`scripts/verify_offline_live_parity.py` blocked; all three clear themselves
when this run completes.

This one **does** take `--tgne`. To use real per-host PCAP trajectories instead
of the CSV path (which fabricates host identity on 9 of 10 CIC-2018 days):

```bash
  --pcap-root /var/home/samito/Documents/SIH/DATA/pcap \
  --cic2018-csv-dir /var/home/samito/Documents/SIH/DATA/CSV \
```

`--pcap-root` requires `--cic2018-csv-dir`: PCAP packets carry no label of
their own, so labels are derived from the paired CSV day. Note the CSV-derived
attack windows inherit the 12-hour clock repair, which is what makes that
alignment correct — before the repair the afternoon half was misaligned by
exactly 12 hours against PCAP's true epoch time.

The frozen PCAP day split is 8 train / 1 val (`fri_2_pcap`) / 1 test
(`thu_1_pcap`). The test store is built and kept separate; it is never handed
to the trainers.

## 4. Validate the checkpoints

```bash
python -m pytest tests/test_served_checkpoint_contracts.py -q
```

Three cases are currently `xfail(strict=True)` because `saved_models/` holds v3
weights (history 5 / forecast 8 — the values the deleted second contract
declared). **After a successful retrain they should start passing, which makes
the strict xfail FAIL.** That is the signal to delete the marker, and the
reason it is strict.

Then remove the opt-in in `tests/conftest.py`
(`CYBERWORLD_ALLOW_CONTRACT_MISMATCH`) and confirm the backend tests still
import — that env var exists only to keep them running against stale weights.

## 5. Parity and the full suite

```bash
python scripts/verify_offline_live_parity.py
python -m pytest tests/ -q     # 270 passing, 3 xfailed as of 2026-09-21
```

---

## 6. What to report

For each model: the held-out **test** number (scored once), the persistence
baseline it must beat, and the credibility verdict. A number without its
baseline is not a result — the gate exists because three earlier training runs
produced numbers that looked fine and meant nothing.


---

## 7. What the Branch B / DeepOP numbers will and will not mean

Measured host-trajectory lengths, which decide what those models can learn:

| source | hosts | real identity? | ≥16 snapshots (Branch B needs T+1) |
|---|---|---|---|
| CIC-2018 ×9 days | **350** | ✗ fabricated `i%250` / `i%100` | 100% |
| CIC-2018 `tue_20` | 37 | ✓ | 11 |
| CTU-13 scen 9 | 127,730 | ✓ | **0.6%** |

On the CSV path Branch B trains almost entirely on a **synthetic 350-host
graph** — those long clean trajectories are an artefact of assigning host
identity by row index. Report CSV-path Branch B/DeepOP results as **pipeline
validation, not science**.

The scientifically valid run needs `--pcap-root`, which gives real per-host
captures. Rough cost: ~1.2B packets over 600 GB, several hours — an overnight
job, and better after the 266 corrupt files are replaced.
