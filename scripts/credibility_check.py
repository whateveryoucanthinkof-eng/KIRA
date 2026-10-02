#!/usr/bin/env python3
"""Credibility gate for the three-branch pipeline.

scripts/train_v4.py refuses to present a run whose numbers cannot mean anything,
and that gate is the only reason three earlier runs reporting PR-AUC 0.9998 were
caught and discarded. The Branch A / Branch B / DeepOP trainers never adopted it
(they import cyberworld_v4.config for the contract but not its splits, metrics or
gate), so their numbers have always gone unchecked. This applies the same tests to
the sample sets those trainers actually build.

The checks, and why each one invalidates a headline number:

  label churn      the fraction of samples whose target label differs from the
                   last observed input window. Near zero means "predict no
                   change" is already perfect and there is nothing to forecast --
                   a forecast score is then a nowcast score wearing a new name.
  base rate        a split that is ~all one class gives a near-1.0 PR-AUC for any
                   ranking, including random noise.
  host groups      confidence intervals must bootstrap over hosts, not over
                   overlapping windows. One or two hosts means no interval exists.
  persistence      if "same as last window" scores what the model scores, the
                   model has learned nothing the baseline did not.
  val size         a small validation split makes best-epoch selection noise.

Exit code is 1 when the run is not credible, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np


def evaluate_samples(train_samples: Sequence[Dict[str, Any]],
                     val_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Diagnostics over the sample dicts create_host_sequence_samples produces."""
    out: Dict[str, Any] = {}

    def split_stats(samples, name):
        n = len(samples)
        hosts = {s.get("host_ip", "") for s in samples}
        tech = np.asarray([int(s["technique"]) for s in samples]) if n else np.zeros(0, int)
        grad = np.asarray([int(s["gradation"]) for s in samples]) if n else np.zeros(0, int)
        risk = np.asarray([float(s["risk"]) for s in samples]) if n else np.zeros(0)
        # "attack" here = any non-Benign technique (index 0 is Benign)
        pos = float(np.mean(tech != 0)) if n else 0.0
        out[f"{name}_n"] = n
        out[f"{name}_hosts"] = len(hosts)
        out[f"{name}_positive_rate"] = pos
        out[f"{name}_technique_classes"] = int(len(set(tech.tolist()))) if n else 0
        out[f"{name}_gradation_hist"] = dict(Counter(grad.tolist())) if n else {}
        out[f"{name}_risk_mean"] = float(np.nanmean(risk)) if n and np.isfinite(risk).any() else 0.0
        out[f"{name}_risk_std"] = float(np.nanstd(risk)) if n and np.isfinite(risk).any() else 0.0
        return pos

    split_stats(train_samples, "train")
    val_pos = split_stats(val_samples, "val")

    # Persistence: the last observed input step already carries the host's
    # features; if the target technique equals the majority technique of that
    # host's other samples, "no change" is a strong baseline. Approximated
    # per-host as the modal target, which is the best a no-change rule can do.
    by_host: Dict[str, List[int]] = {}
    for s in val_samples:
        by_host.setdefault(s.get("host_ip", ""), []).append(int(s["technique"]))
    correct = total = 0
    for _h, labels in by_host.items():
        if not labels:
            continue
        modal = Counter(labels).most_common(1)[0][1]
        correct += modal
        total += len(labels)
    out["val_persistence_accuracy"] = (correct / total) if total else 0.0

    # Label churn: fraction of hosts whose target label is not constant.
    churn_hosts = sum(1 for labels in by_host.values() if len(set(labels)) > 1)
    out["val_hosts_with_label_change"] = churn_hosts
    out["val_label_churn"] = (churn_hosts / len(by_host)) if by_host else 0.0
    return out


def evaluate_store(train_store, val_store) -> Dict[str, Any]:
    """Diagnostics straight off a TrajectoryStore, so the same gate applies to
    Branch A, Branch B and DeepOP -- they build different sample shapes but all
    derive from these stores."""
    out: Dict[str, Any] = {}

    def stats(store, name):
        n = store.n_snapshots
        out[f"{name}_n"] = n
        out[f"{name}_hosts"] = len(store)
        if n == 0:
            out[f"{name}_positive_rate"] = 0.0
            out[f"{name}_technique_classes"] = 0
            return
        out[f"{name}_positive_rate"] = float(np.mean(store.is_attack))
        out[f"{name}_technique_classes"] = int(len(store.techniques))
        out[f"{name}_categories"] = list(store.categories)
        out[f"{name}_risk_mean"] = float(np.nanmean(store.risk_score)) if np.isfinite(store.risk_score).any() else 0.0

    stats(train_store, "train")
    stats(val_store, "val")

    # per-host label change over the host's own trajectory
    changed = 0
    total_hosts = 0
    lens = []
    for h in val_store.keys():
        rows = val_store._rows_by_host[h]
        lens.append(len(rows))
        total_hosts += 1
        lab = val_store.is_attack[rows]
        if lab.size and bool(lab.max() != lab.min()):
            changed += 1
    out["val_hosts_with_label_change"] = changed
    out["val_label_churn"] = (changed / total_hosts) if total_hosts else 0.0
    if lens:
        arr = np.asarray(lens)
        out["val_traj_len_median"] = int(np.median(arr))
        out["val_traj_len_max"] = int(arr.max())
        out["val_hosts_with_16plus"] = int((arr >= 16).sum())
    # a no-change rule's accuracy on is_attack
    if val_store.n_snapshots:
        p = float(np.mean(val_store.is_attack))
        out["val_persistence_accuracy"] = max(p, 1.0 - p)
    return out


def gate(stats: Dict[str, Any], model_accuracy: float | None = None) -> List[str]:
    problems: List[str] = []
    if stats.get("val_label_churn", 0.0) < 0.01:
        problems.append(
            f"label churn {stats['val_label_churn']:.4f}: almost no host ever changes label, "
            "so there is nothing to forecast")
    pr = stats.get("val_positive_rate", 0.0)
    if pr > 0.95 or pr < 0.05:
        problems.append(
            f"val base rate {pr:.4f} is extreme: any ranking scores near-perfect here")
    if stats.get("val_hosts", 0) < 5:
        problems.append(
            f"val split has {stats.get('val_hosts')} host group(s): no usable confidence interval")
    if stats.get("val_n", 0) < 500:
        problems.append(
            f"val split has {stats.get('val_n')} samples: best-epoch selection is noise")
    if stats.get("val_technique_classes", 0) < 2:
        problems.append(
            f"val split contains {stats.get('val_technique_classes')} technique class(es): "
            "accuracy is meaningless")
    persistence = stats.get("val_persistence_accuracy", 0.0)
    if persistence > 0.99:
        problems.append(
            f"persistence baseline scores {persistence:.4f}: a no-change rule is already perfect")
    if model_accuracy is not None and persistence > 0:
        lift = model_accuracy - persistence
        stats["model_minus_persistence"] = lift
        if lift <= 0.0:
            problems.append(
                f"model accuracy {model_accuracy:.4f} does not beat persistence {persistence:.4f}")
    # train/val imbalance makes val-based selection unreliable even if valid
    if stats.get("train_n", 0) and stats.get("val_n", 0):
        ratio = stats["train_n"] / max(1, stats["val_n"])
        stats["train_val_ratio"] = ratio
        if ratio > 25:
            problems.append(
                f"train:val ratio is {ratio:.0f}:1 -- validation is too thin relative to train "
                "for stable checkpoint selection")
    return problems


def report(stats: Dict[str, Any], problems: List[str]) -> None:
    print("=" * 68)
    print("CREDIBILITY CHECK")
    print("=" * 68)
    for k in sorted(stats):
        print(f"  {k:32s} {stats[k]}")
    print("=" * 68)
    if problems:
        print("NOT CREDIBLE — do not report these numbers")
        for p in problems:
            print(f"  - {p}")
    else:
        print("CREDIBLE — no disqualifying condition found")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-json", type=str, required=True,
                    help="JSON produced by evaluate_samples (or a training run)")
    ap.add_argument("--model-accuracy", type=float, default=None)
    a = ap.parse_args()
    stats = json.loads(open(a.stats_json).read())
    problems = gate(stats, a.model_accuracy)
    report(stats, problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
