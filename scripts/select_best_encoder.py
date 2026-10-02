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

    harmonic mean of inductive AP and macro recall over the per-class accuracies

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


def smooth(values, window: int):
    """Centred moving average, clipped at the ends. INFORMATIONAL ONLY.

    An earlier version of this script SELECTED on the smoothed score. That was
    wrong: smoothing identifies a good REGION of training, but we deploy one
    specific checkpoint, and that checkpoint's quality is its own raw score,
    not its neighbours' average. It picked epoch 2 (raw 0.9286) over epoch 6
    (raw 0.9357) -- a demonstrably worse checkpoint.

    The real concern it was reaching for is the winner's curse: with C2
    inductive recall at std 0.119 across epochs, the maximum is partly luck
    and will regress. The honest remedy is to report which epochs are
    statistically tied, not to deliberately choose a lower-scoring one.
    """
    if window <= 1:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


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
        out.append((int(ep), float(ap), macro, _harmonic(float(ap), macro)))
    return out


def _harmonic(ap: float, macro: float) -> float:
    """The trainer's selection rule (bita/train.py::selection_score): the
    harmonic mean, so a collapsed head cannot be promoted on AP alone. This
    used 0.5*AP + 0.5*macro and could promote a different epoch than the
    trainer kept."""
    return 0.0 if ap + macro <= 0 else 2.0 * ap * macro / (ap + macro)


#: The encoder build_or_load_tgne_ta() prefers. Relative to the repo root.
SERVED_ENCODER = "saved_models/bita_bigru_transformer-unified_final.pth"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path)
    p.add_argument("--prefix", default="bita_bigru_transformer-unified_final")
    p.add_argument("--checkpoint-dir", type=Path, default=Path(".spill/encoder_epochs"),
                   help="Where bita/train.py wrote per-epoch snapshots (scratch).")
    p.add_argument("--smooth", type=int, default=3,
                   help="Window for the informational smoothed column. Selection "
                        "always uses the raw score; see smooth().")
    p.add_argument("--tie-tolerance", type=float, default=0.01,
                   help="Epochs within this of the best are reported as tied, "
                        "because minority-class recall is noisy (C2 std 0.119).")
    p.add_argument("--copy", type=Path, default=None,
                   help="Copy the winning checkpoint here (e.g. saved_models/...pth)")
    a = p.parse_args()

    rows = score_log(a.log.read_text())
    if not rows:
        print(f"no completed epochs found in {a.log}")
        return 1

    sm = smooth([r[3] for r in rows], a.smooth)
    print(f"{'epoch':>5}  {'ind_AP':>7}  {'macro':>7}  {'selection':>9}  {'smoothed':>9}")
    for (ep, ap, macro, sel), sv in zip(rows, sm):
        print(f"{ep:>5}  {ap:>7.4f}  {macro:>7.4f}  {sel:>9.4f}  {sv:>9.4f}")

    # Select on the RAW score -- that is the deployed checkpoint's own quality.
    best = max(rows, key=lambda r: r[3])
    tied = [r[0] for r in rows if r[3] >= best[3] - a.tie_tolerance and r[0] != best[0]]
    if tied:
        print(f"\nStatistically tied within {a.tie_tolerance}: epochs "
              f"{', '.join(str(t) for t in tied)}. Minority-class recall is noisy "
              f"(C2 inductive std 0.119 across epochs), so the winner is partly "
              f"luck and any of these is a defensible choice.")
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
    else:
        print()
        print(f"NOTHING WAS PROMOTED. Serving loads {SERVED_ENCODER}; the "
              f"per-epoch snapshots are never read. Re-run with "
              f"--copy {SERVED_ENCODER} (and write its _config.json) to serve it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
