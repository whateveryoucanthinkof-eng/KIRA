#!/usr/bin/env python3
"""Pick the best TGNE epoch checkpoint from a training log.

## Why this exists

Early stopping originally selected on `val_ap` -- link prediction over hosts
the model has already seen. On this corpus the two objectives DIVERGE:

    epoch      0       1       2       3       4       5
    val_ap    .9971   .9968   .9968   .9973   .9974   .9975   <- still rising
    ind AP    .9926   .9921   .9918   .9935   .9936   .9934
    C2 ind    .325    .319    .517    .565    .586    .480    <- peaks at 4
    IA        .505    .714    .578    .612    .561    .510

`val_ap` keeps creeping up while classification degrades after epoch 4, so
selecting on it alone picks epoch 5 and gives away the better classifier.

This scores every epoch on

    0.5 * inductive AP  +  0.5 * macro recall over the per-class accuracies

* **inductive**, because deployment meets unseen hosts constantly and the
  transductive figure sits at 0.997 with no discriminative power left;
* **macro**, because Benign is ~90% of validation -- a head predicting Benign
  for everything scores ~0.9 aggregate accuracy, which is exactly the metric
  that hid a collapsed head for most of this project.

Every epoch checkpoint is written during training, so the choice can be made
after the fact without retraining.

Usage:
    python scripts/select_best_encoder.py logs/tgne_final.out \
        [--prefix bita_bigru_transformer-unified_final] [--copy DEST]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

EPOCH_RE = re.compile(r"Epoch (\d+) \[.*?Inductive Val AUC: ([\d.]+), AP: ([\d.]+)")
PERCLASS_RE = re.compile(r"per-class val acc: \{([^}]*)\}")


def score_log(log_text: str):
    """[(epoch:int, inductive_ap, macro_recall, selection)] in epoch order."""
    epochs = EPOCH_RE.findall(log_text)
    per_class = PERCLASS_RE.findall(log_text)
    out = []
    for (ep, _auc, ap), pc in zip(epochs, per_class):
        vals = [float(v) for v in re.findall(r": ([\d.]+)", pc)]
        if not vals:
            continue
        macro = sum(vals) / len(vals)
        out.append((int(ep), float(ap), macro, 0.5 * float(ap) + 0.5 * macro))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path)
    p.add_argument("--prefix", default="bita_bigru_transformer-unified_final")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("saved_checkpoints"))
    p.add_argument("--copy", type=Path, default=None,
                   help="Copy the winning checkpoint here (e.g. saved_models/...pth)")
    a = p.parse_args()

    rows = score_log(a.log.read_text())
    if not rows:
        print(f"no completed epochs found in {a.log}")
        return 1

    print(f"{'epoch':>5}  {'ind_AP':>7}  {'macro':>7}  {'selection':>9}")
    for ep, ap, macro, sel in rows:
        print(f"{ep:>5}  {ap:>7.4f}  {macro:>7.4f}  {sel:>9.4f}")

    best = max(rows, key=lambda r: r[3])
    ckpt = a.checkpoint_dir / f"{a.prefix}-{best[0]}.pth"
    print(f"\nbest epoch: {best[0]}  (selection {best[3]:.4f}, "
          f"inductive AP {best[1]:.4f}, macro recall {best[2]:.4f})")
    print(f"checkpoint: {ckpt}  {'[exists]' if ckpt.exists() else '[MISSING]'}")

    # What val_ap alone would have chosen, so the difference is visible.
    # NOTE: the pipe matters. "Val AUC: ..., AP: ..." also appears inside
    # "Inductive Val AUC: ..., AP: ...", so without it the two series
    # interleave and the comparison is nonsense.
    val_ap = re.findall(r"\| Val AUC: [\d.]+, AP: ([\d.]+)", a.log.read_text())
    if val_ap:
        alt = max(range(len(val_ap)), key=lambda i: float(val_ap[i]))
        if alt != best[0]:
            alt_row = next((r for r in rows if r[0] == alt), None)
            if alt_row:
                print(f"\nNOTE: selecting on val_ap alone would pick epoch {alt} "
                      f"(selection {alt_row[3]:.4f}) -- a worse classifier.")

    if a.copy:
        if not ckpt.exists():
            print(f"refusing to copy: {ckpt} does not exist")
            return 1
        a.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, a.copy)
        print(f"copied -> {a.copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
