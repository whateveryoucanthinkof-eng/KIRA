# 24 — Branch A's accuracy could not fail visibly

**Date:** 2026-09-21
**Found by:** reading the epoch metrics against the credibility report.

## The problem

Branch A's `_evaluate` reported exactly three numbers: `loss`, `risk_mae`,
`tech_accuracy`. Over five epochs `tech_accuracy` read:

```
epoch 1  0.879      epoch 4  0.880
epoch 2  0.880      epoch 5  0.878
epoch 3  0.881
```

Flat from the first epoch. The credibility report for the same run says
`val_positive_rate = 0.1752` — **82.5% of validation samples are Benign**. A
model that predicts Benign for every input scores ~0.825.

So 0.880 is **5.5 points above a constant classifier**, and nothing in the
output could tell the difference between a technique head that learned a little
and one that collapsed onto the majority class. That is not a hypothetical
failure mode in this repo: the TGNE category head scored 0.83 while emitting a
single class for every input, and it was caught only when macro F1 was
computed. The same blind spot was still present one model downstream.

## Fix

`_evaluate` now accumulates a full confusion matrix on-device — one
`torch.bincount` per batch, so no extra host-device sync — and returns:

| key | why |
|---|---|
| `tech_macro_f1` | unweighted over present classes; collapses to ~0.23 when one class is always predicted |
| `tech_majority_baseline` | what "always predict the most common class" scores |
| `tech_lift_over_baseline` | accuracy − baseline; **0.000 for a collapsed head** |
| `tech_classes_predicted` / `tech_classes_present` | `1/4` is a collapse, stated outright |
| `tech_per_class` | precision, recall, f1, support, predicted-count per class |
| `gradation_accuracy` | the third head was never reported at all |

`_warn_if_head_collapsed()` prints a named warning at every epoch and on the
held-out test when the head predicts a single class, when lift is under 1
point, or when macro F1 is under 0.2 across more than two classes. The held-out
test also prints a per-class table sorted by support.

## `--eval-only`

The run in flight tonight started before any of this existed, so its checkpoint
will carry only aggregate accuracy. Retraining purely to get better metrics
would waste ~2.5 h, and extraction (~37 min) is the expensive part either way.

`--eval-only CKPT` loads a checkpoint, scores it on the validation split with
the full metric set, prints the per-class table and exits without training or
writing anything. `-` means "the path in `--output`".

```bash
python scripts/retrain_branch_a_live.py \
  --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
  --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
  --output saved_models/branch_a/branch_a_lstm.pt \
  --eval-only - --spill-dir .spill --num-workers 4
```

**Queued to run after the downstream retrain**, so the two do not contend for
memory.

## Tests

`tests/test_branch_a_eval_detects_collapse.py` (6 tests) builds an
82.5%-majority dataset and runs `_evaluate` against two stub models:

- a constant predictor scores `tech_accuracy > 0.80` while
  `tech_classes_predicted == 1`, `lift == 0.000` and `macro_f1 < 0.30`
- an oracle scores 1.0 on both accuracy and macro F1, with lift > 0.15
- `tech_macro_f1` **matches `sklearn.f1_score(average="macro")` exactly**
- per-class supports sum to the dataset size
- the warning fires for the collapse case and stays silent for the healthy one

## What this does not yet tell us

Whether Branch A's head has *actually* collapsed. The numbers are consistent
with either a modest real signal or a near-collapse; that is the whole point of
the gap. `--eval-only` answers it, and the answer goes in report 25.

Suite: **464 passed, 2 xfailed**.
