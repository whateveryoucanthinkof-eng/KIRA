# results/

## `tgne_pretrain_curve.csv`

A **single-epoch TGNE/BiTA link-prediction curve**, not a detection result.

```
epoch 0 | val_ap 0.602 | val_auc 0.613 | val_accuracy 0.0 | inductive_val_auc 0.485
```

Read it for what it is: one epoch of self-supervised graph pretraining, with accuracy at zero and
inductive AUC *below chance*. It says nothing about whether the system detects or forecasts attacks,
and it must not be cited as evidence that it does. It was previously named `training_metrics.csv`,
which invited exactly that misreading.

## `v4_benchmark.json` (produced by `scripts/train_v4.py`)

The real evaluation: nowcast and forecast scored separately, per-horizon PR-AUC/Brier/NLL, baselines
trained on identical features, group-bootstrap intervals, calibration before and after, and the full
experiment manifest. That file — not the curve above — is what a benchmark claim rests on.

Regenerate with:

```bash
python scripts/train_v4.py \
  --cic-dir ~/Documents/SIH/DATA/CSV \
  --ctu-dir ~/Documents/SIH/CTU-13-Dataset \
  --rows-per-file 60000 --epochs 10
```
