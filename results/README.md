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

## `training_metrics.csv`

A **second, contradictory** encoder curve: 12 epochs, val AUC 0.9977, inductive val AUC 0.9943.
`tgne_pretrain_curve.csv` above is described as a *rename* of this file, but both are present with
different numbers, so neither can be attributed to a run and nothing here identifies which encoder
produced the shipped checkpoint. Do not cite either until one is regenerated with a manifest.

## `v4_benchmark.json` (produced by `scripts/train_v4.py`)

The shape a benchmark claim should take: nowcast and forecast scored separately, per-horizon
PR-AUC/Brier/NLL, baselines trained on identical features, group-bootstrap intervals, calibration
before and after, lead time measured against onset, conformal coverage, label coverage, and the full
experiment manifest.

**The file currently committed is not a result.** It predates the credibility gate (it carries no
`credible` key), and every gate would have fired on it:

| | |
|---|---|
| test split | **1 host**, `n_groups: 1`, so `ci_low` / `ci_high` are `NaN` |
| test base rate | **0.9997** — PR-AUC 0.9993 is what a constant predictor scores |
| recall / F1 | **0.000** at the chosen threshold: 14,973 false negatives, 0 true positives |
| vs. baselines | model nowcast ROC-AUC 0.690 against gradient boosting at **0.780** |
| vs. persistence | model Brier ~0.99 per step against persistence at **0.00067** |
| timestamps | `time_spans` near −9.22e9, i.e. NaT sentinels — the chronological order was meaningless |
| calibration | temperature 0.05 fitted on 154 samples that are 100% positive; the ECE of 1e-7 is an artefact of that |

It is kept as the record of a run that must not be repeated, not as evidence of anything.

Regenerate with the current trainer, which builds full `(dataset, scenario, capture, host)`
identities, splits on captures from `data_unification/splits.lock.json`, drops labels the ontology
cannot name, reports lead time and conformal coverage, and refuses to present numbers that fail the
gate:

```bash
python scripts/train_v4.py \
  --cic-dir ~/Documents/SIH/DATA/CSV \
  --ctu-dir ~/Documents/SIH/CTU-13-Dataset \
  --split frozen --rows-per-file 60000 --epochs 10
```
