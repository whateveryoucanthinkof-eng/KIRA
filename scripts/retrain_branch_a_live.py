"""Retrain Branch A on supplied 2-second SIH/CTU live-telemetry data."""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from branch_a_gnn_lstm.sequence_dataset import (
    format_history_report as _fmt_history,
    TECHNIQUE_VOCAB,
    TECH_TO_IDX,
    GRADATION_LEVELS,
    HostSequenceDataset,
    create_host_sequence_samples,
    LazyHostSequenceDataset,
    BatchedSequenceView,
    PermutationBatchSampler,
    collate_prebatched,
)
from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.density import require_full_density
from data_unification.split_policy import is_cross_year
from cyberworld_v4.training_guard import (IMPROVED, STOP, ResumePoint, TrainingGuard,
                                          default_warmup_steps, run_fingerprint)
from cyberworld_v4.config import get_contract, DEFAULT_CONFIG
from cyberworld_v4.graphed_step import GraphedLoss, graphs_enabled
from cyberworld_v4.manifest import ExperimentManifest, set_all_seeds


def _make_samples(records, extractor, seq_len: int, min_history_steps=None):
    """Samples for one record batch, with the history floor applied and reported.

    `min_history_steps=None` requires a fully observed window. The old call
    passed `min_trajectory_len=1` and let the rest be zero-padded, which on
    this corpus means the median sample was 14/15 padding -- see
    `create_host_sequence_samples`.
    """
    from branch_a_gnn_lstm.sequence_dataset import format_history_report
    trajectories = extractor.extract_trajectories(records)
    rep = {}
    out = create_host_sequence_samples(
        trajectories,
        seq_len=seq_len,
        min_trajectory_len=1,
        min_history_steps=min_history_steps,
        report=rep,
    )
    print("  " + format_history_report(rep), flush=True)
    return out


def _selection_score(metrics, mode):
    """Higher is better. Which epoch's weights we keep.

    ## Why the default is not validation loss

    The objective is Kendall & Gal homoscedastic uncertainty weighting:

        L = sum_i [ exp(-s_i) * L_i + s_i ]

    where every `s_i` is a **learned parameter that moves during training**.
    Two epochs therefore do not report the same quantity: the loss is computed
    under a different weighting each time, and the bare `+ s_i` terms add a
    drifting constant. The 2026-09-21 checkpoint ended with
    s_risk = -3.0090 and s_tech = -3.0243 against an initialisation of 0, so
    that constant drifted from 0 to -7.04 over the run. A large part of the
    apparent fall in validation loss was the offset moving, not the model
    improving -- and picking the single lowest reading then selected epoch 6
    (0.6601, about half the median of the other epochs), which scored worst on
    the held-out test.

    A selection metric has to mean the same thing at every epoch. Macro F1 and
    risk MAE do; the weighted loss does not.

    - ``composite`` (default): ``0.5 * macro_f1 + 0.5 * risk_auc`` -- balances
      the two heads that carry the task, both bounded in [0, 1] and both
      independent of the loss weighting. AUC replaces the risk MAE that was
      here first: MAE is median-seeking, so on a target that is 0 for 82.5% of
      samples it rewards a head for predicting nothing, which is the opposite
      of what selection should reward.
    - ``macro_f1``: technique head only.
    - ``val_loss``: the previous behaviour, kept so a run can be reproduced.
      Negated here because this function is maximised.
    """
    if mode == "val_loss":
        return -float(metrics["loss"])
    if mode == "macro_f1":
        return float(metrics.get("tech_macro_f1", 0.0))
    f1 = float(metrics.get("tech_macro_f1", 0.0))
    auc = float(metrics.get("risk_auc", float("nan")))
    if auc != auc:                      # NaN: one class only in this split
        auc = 0.5                       # chance -- contributes nothing either way
    return 0.5 * f1 + 0.5 * auc


def _flag_outlier_selection(history, best):
    """Warn when the winning epoch's val loss is far off the run's own trend.

    The 2026-09-21 run produced val losses 1.88, 2.08, 1.39, 1.24, 1.39, 0.66,
    1.41, 1.21. Epoch 6 at 0.66 is roughly half the median of every other
    epoch, and picking the single best reading selected exactly that outlier --
    which then scored 26.36 on the held-out test against 0.66 on validation.

    A one-off low reading on a multi-task loss with learned log-variance terms
    is at least as likely to be noise as a genuinely better model. This does
    not change the selection -- that would be a modelling decision, not a
    reporting one -- it states when the result deserves suspicion.
    """
    if not history or not best or len(history) < 4:
        return
    losses = [h["loss"] for h in history if "loss" in h]
    chosen = best.get("loss")
    if chosen is None or len(losses) < 4:
        return
    others = sorted(l for l in losses if l != chosen)
    if not others:
        return
    median = others[len(others) // 2]
    if median <= 0:
        return
    if chosen < 0.6 * median:
        print(
            f"\nNOTE: the selected epoch ({best.get('epoch')}) had validation "
            f"loss {chosen:.4f}, against a median of {median:.4f} across the "
            f"other epochs -- {median / max(chosen, 1e-9):.1f}x lower. A single "
            f"outlier reading may be noise rather than a better model; compare "
            f"the held-out test result before quoting this checkpoint. Every "
            f"epoch's metrics are in ckpt['epoch_history'].",
            flush=True)


#: score bins for the risk histogram. 2000 gives AUC to ~5e-4 and a
#: 0.05%-wide calibration bin, at 16 KB of device memory.
RISK_BINS = 2000


def _label_columns(store):
    """Per-row (technique index, gradation level) for a whole store, vectorised.

    Both are derived from interned columns the store already holds, via a
    lookup table per distinct string -- so this is two fancy-index gathers
    over `n_snapshots`, not 20.7M dict lookups. The defaults match
    `LazyHostSequenceDataset.__getitem__` exactly: no technique recorded means
    Benign, an unknown coarse category means level 0.
    """
    tech_map = np.array(
        [TECH_TO_IDX.get(t, TECH_TO_IDX["Benign"]) for t in store.techniques],
        dtype=np.int64)
    cat_map = np.array(
        [GRADATION_LEVELS.get(c, 0) for c in store.categories], dtype=np.int64)
    tech_off = np.asarray(store.tech_off)
    lo = tech_off[:-1]
    has = tech_off[1:] > lo
    tech = np.full(store.n_snapshots, TECH_TO_IDX["Benign"], dtype=np.int64)
    if has.any() and tech_map.size:
        tech[has] = tech_map[np.asarray(store.tech_flat)[lo[has]]]
    grad = cat_map[np.asarray(store.cat_id)] if cat_map.size else \
        np.zeros(store.n_snapshots, dtype=np.int64)
    return tech, grad


def target_label_counts(dataset):
    """(technique_counts, gradation_counts) over a LazyHostSequenceDataset.

    The focal alpha needs the class histogram of all 20.7M training targets
    before the first batch, so how this is computed matters.

    Not through `__getitem__`: that gathers a [15, 27] feature window from the
    memmap and builds four tensors per sample, none of which a label count
    uses. Not by materialising the target-row indices either -- that is a
    166 MB int64 array in a process that is already holding three stores.

    Instead, by complement. A sample's target is row `rows[end]` with `end`
    running 1..n-1 per host, so the multiset of target rows is *every row in
    the store except each host's first*. Counting all rows and subtracting the
    per-host first rows is two bincounts and one gather of `len(hosts)`
    indices, and it is exact rather than approximate.

    The identity only holds when every store row belongs to a host the dataset
    kept (true at `min_trajectory_len <= 1`, which is what Branch A uses), so
    it is checked against the dataset's own sample count rather than assumed;
    a mismatch falls back to the direct per-host count.
    """
    store = dataset.store
    tech, grad = _label_columns(store)
    n_hosts = len(dataset._rows)
    complement_ok = (len(dataset._pos) == store.n_snapshots - n_hosts)

    if complement_ok:
        first = np.fromiter((int(r[0]) for r in dataset._rows),
                            dtype=np.int64, count=n_hosts)
        t_counts = (np.bincount(tech, minlength=len(TECHNIQUE_VOCAB))
                    - np.bincount(tech[first], minlength=len(TECHNIQUE_VOCAB)))
        g_counts = (np.bincount(grad, minlength=4)
                    - np.bincount(grad[first], minlength=4))
        return t_counts, g_counts

    t_counts = np.zeros(len(TECHNIQUE_VOCAB), dtype=np.int64)
    g_counts = np.zeros(4, dtype=np.int64)
    for rows in dataset._rows:
        r = np.asarray(rows)[1:]
        if r.size:
            t_counts += np.bincount(tech[r], minlength=len(TECHNIQUE_VOCAB))
            g_counts += np.bincount(grad[r], minlength=4)
    return t_counts, g_counts


#: Metric keys that must never reach the checkpoint or `epoch_history`.
#: `_evaluate` returns the raw score histograms and (optionally) the collected
#: technique logits so a caller can fit an operating point and a temperature
#: without a second pass over the split. Those are working data, not results:
#: the logits alone are 57 MB at the validation split's size, and a
#: checkpoint that carried them per epoch would be gigabytes.
BULKY_METRIC_KEYS = (
    "risk_pos_hist", "risk_neg_hist", "risk_resid_hist",
    "technique_logits", "technique_labels", "tech_confusion",
)


def slim(metrics, drop_per_class=False):
    """A copy of a metrics dict safe to store, print or log."""
    drop = set(BULKY_METRIC_KEYS)
    if drop_per_class:
        drop |= {"tech_per_class", "gradation_per_class"}
    return {k: v for k, v in metrics.items() if k not in drop}


def conformal_halfwidth_from_histogram(resid_hist, alpha=0.05, bins=RISK_BINS):
    """Split-conformal half-width from a binned residual distribution.

    The same finite-sample correction as
    `cyberworld_v4.conformal.conformal_quantile`: the k-th smallest residual
    with k = ceil((n+1)(1-alpha)), not the plain empirical quantile -- that
    correction is what makes the coverage guarantee exact rather than
    approximate.

    Binning rounds the quantile UP to the bin's upper edge, so the interval is
    at worst one bin (1/(bins-1) = 5e-4) wider than the exact one. That
    direction is deliberate: a conformal interval rounded up over-covers,
    which is a weaker claim, while rounding down would over-state coverage.
    `empirical_coverage` is computed from the same histogram so the reported
    number is the one the width actually achieves, never the target alone.
    """
    h = np.asarray(resid_hist, dtype=np.int64)
    n = int(h.sum())
    if n == 0:
        return {"fitted": False, "n": 0, "reason": "no residuals"}
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return {"fitted": False, "n": n,
                "reason": (f"{n} calibration points cannot support "
                           f"{(1 - alpha) * 100:.1f}% coverage")}
    cum = np.cumsum(h)
    b = int(np.searchsorted(cum, k, side="left"))
    q = min(float(b + 1) / (bins - 1), 1.0)
    covered = float(cum[b] / n)
    return {
        "fitted": True, "alpha": float(alpha),
        "target_coverage": 1.0 - float(alpha),
        "half_width": q,
        "empirical_coverage": covered,
        "n": n,
        "bin_width": 1.0 / (bins - 1),
        "method": "split conformal, ceil((n+1)(1-alpha)) order statistic, "
                  "binned and rounded up",
    }


def _print_conformal(c, where):
    if not c.get("fitted"):
        print(f"  conformal [{where}]: NOT FITTED -- {c.get('reason')}", flush=True)
        return
    print(f"  conformal [{where}]: half-width {c['half_width']:.4f} for "
          f"{c['target_coverage'] * 100:.0f}% target coverage "
          f"(empirical {c['empirical_coverage'] * 100:.2f}% on n={c['n']:,})",
          flush=True)
    if c["half_width"] > 0.4:
        # Not a failure -- the honest answer for a near-binary target. Said
        # out loud because the number it replaces (a hardcoded 0.05 described
        # as "guaranteeing 95% coverage") was small enough to look useful.
        print(f"    NOTE: a {c['target_coverage'] * 100:.0f}% prediction "
              f"interval on a near-binary outcome is wide by construction -- "
              f"to contain y in {{0,1}} it needs |p - y| <= half-width, so "
              f"anything short of a near-certain prediction fails. This is "
              f"the measured answer; the +/-0.05 it replaces was fitted to "
              f"nothing and its real coverage was whatever it happened to be.",
              flush=True)


def calibrate_and_fit_operating_point(model, metrics, args, where="validation"):
    """Everything that must happen AFTER training, on held-out data, frozen.

    Three post-hoc fits, all from one already-completed evaluation pass:

    * the technique temperature, by NLL on the collected raw logits;
    * the risk head's conformal half-width, from the residuals;
    * the served alert threshold, from the risk score histograms.

    They are grouped because they share a precondition that is easy to lose:
    the model's parameters must already be final. A temperature fitted while
    training continues is not a temperature scaling, which is exactly the
    defect this replaces.

    ## What this inherits, and what it would take to not inherit it

    These are fitted on VALIDATION, the same split that chose the checkpoint,
    so they carry that selection's optimism. The honest remedy is a fourth
    split -- `cyberworld_v4.splits` already defines CALIBRATION as disjoint
    from VALIDATION for precisely this reason -- but `splits.lock.json` is
    frozen three-way and is not Branch A's file to change. Carving a
    calibration split out of the three validation captures by hand would be
    worse than the optimism it removes: the captures are one CIC-2018 day and
    two CTU-13 scenarios, and CTU-13 scenarios are ~100% attack, so any
    capture-disjoint carve-out gives a calibration set whose base rate is
    nothing like the one serving will see. A temperature or a threshold fitted
    on an unrepresentative split is a worse number than a mildly optimistic
    one. Recorded in the checkpoint as `split: "validation"` so the claim is
    never stronger than the evidence.
    """
    out = {}

    if not args.no_fit_temperature:
        logits = metrics.get("technique_logits")
        labels = metrics.get("technique_labels")
        if logits is None or labels is None:
            out["temperature"] = {"fitted": False,
                                  "reason": "no logits collected"}
        else:
            rep = model.fit_temperature(logits, labels)
            rep["split"] = where
            out["temperature"] = rep
            if rep.get("fitted"):
                print(f"  temperature [{where}]: T={rep['temperature']:.4f} "
                      f"NLL {rep['nll_before']:.4f} -> {rep['nll_after']:.4f}, "
                      f"top-label ECE {rep['ece_before']:.4f} -> "
                      f"{rep['ece_after']:.4f} (n={rep['n_calibration']:,})",
                      flush=True)
                if rep["ece_after"] > rep["ece_before"]:
                    print(f"  WARNING [{where}]: temperature scaling lowered NLL "
                          f"but RAISED top-label ECE "
                          f"({rep['ece_before']:.4f} -> {rep['ece_after']:.4f}). "
                          f"NLL and ECE disagree when the errors are not a pure "
                          f"confidence miscalibration; the fit is recorded, and "
                          f"it should not be described as improving calibration.",
                          flush=True)
            else:
                print(f"  temperature [{where}]: NOT FITTED -- {rep.get('reason')}",
                      flush=True)

    conf = conformal_halfwidth_from_histogram(
        metrics["risk_resid_hist"], alpha=args.conformal_alpha)
    _print_conformal(conf, where)
    if conf.get("fitted"):
        model.set_risk_conformal_halfwidth(conf["half_width"])
    out["risk_conformal"] = conf

    op = fit_operating_point(
        metrics["risk_pos_hist"], metrics["risk_neg_hist"],
        criterion=args.operating_point_criterion,
        alert_budget=args.alert_budget,
    )
    if op.get("fitted"):
        # What the historical 0.65 cut does to this same head. It is the
        # number that decides whether carrying it across would be silent.
        curve = _pr_curve_from_histograms(metrics["risk_pos_hist"],
                                          metrics["risk_neg_hist"])
        b65 = int(round(0.65 * (RISK_BINS - 1)))
        op["legacy_0_65"] = _point(curve, b65)
        op["risk_objective"] = args.risk_objective
        op["risk_target"] = args.risk_target
        op["positive_event"] = (
            "risk_score > 0 (this window contains attack traffic)"
            if args.risk_target == "severity" else
            f"hazard >= exp(-1) (an attack within one forecast horizon, "
            f"tau={metrics.get('risk_positive_above', 0):.4f} cut)")
    _print_operating_point(op, where)
    out["operating_point"] = op
    return out


def _print_class_counts(counts, names, label):
    """The counts the weights were built from.

    A weight of exactly 1.0 means a class had ZERO training samples, and that
    is how a broken split was found before (3 of 5 encoder categories absent
    because the temporal cut was splitting by corpus). Printing the histogram
    next to the weights is what makes that visible.
    """
    total = max(int(np.sum(counts)), 1)
    print(f"  {label} class counts (train):", flush=True)
    for i, c in enumerate(counts):
        if c == 0:
            continue
        print(f"    {names.get(i, f'class_{i}'):<28} {int(c):>12,} "
              f"({100.0 * c / total:5.2f}%)", flush=True)
    absent = [names.get(i, f"class_{i}") for i, c in enumerate(counts) if c == 0]
    if absent:
        print(f"    ABSENT from train ({len(absent)}): {', '.join(absent)}",
              flush=True)


def _pr_curve_from_histograms(pos_hist, neg_hist, bins=RISK_BINS):
    """Exact TP/FP/FN/precision/recall/F1/alert-rate at every bin boundary.

    `_evaluate` already accumulates the risk scores into `pos_hist`/`neg_hist`
    on the device, so the whole precision-recall curve is two reversed cumulative
    sums away -- no second pass over 1.02M validation windows and no 1.02M-float
    score array on the host.

    The binning is exact, not approximate. A score lands in bin
    `floor(p * (bins - 1))`, and `floor(p * (bins-1)) >= b` iff `p >= b/(bins-1)`,
    so the threshold that bin boundary `b` represents is exactly `b/(bins-1)` and
    the counts above it are exactly the tail sums. Verified against
    `sklearn.metrics.precision_recall_curve` on 200k scores quantised onto this
    grid: maximum precision and recall difference **0.0**, and TP/FP/alert-rate
    reproduced exactly by brute force at eight thresholds
    (tests/test_branch_a_operating_point.py).

    Precision where nothing is predicted positive is defined as 1.0, which is
    sklearn's convention; `alert_rate` is 0 there, so such a point can never be
    selected by a criterion that requires alerting.
    """
    ph = np.asarray(pos_hist, dtype=np.float64)
    nh = np.asarray(neg_hist, dtype=np.float64)
    P, N = float(ph.sum()), float(nh.sum())
    tp = np.cumsum(ph[::-1])[::-1]          # tp[b] = positives in bins >= b
    fp = np.cumsum(nh[::-1])[::-1]
    pred_pos = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1.0), 1.0)
        recall = (tp / P) if P > 0 else np.zeros_like(tp)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return {
        "threshold": np.arange(bins, dtype=np.float64) / (bins - 1),
        "tp": tp, "fp": fp, "fn": P - tp, "tn": N - fp,
        "precision": precision, "recall": recall, "f1": f1,
        "alert_rate": pred_pos / max(P + N, 1.0),
        "n_positive": P, "n_negative": N,
        "base_rate": P / max(P + N, 1.0),
    }


def _point(curve, b):
    """One operating point off the curve, as plain floats."""
    return {
        "alert_threshold": float(curve["threshold"][b]),
        "precision": float(curve["precision"][b]),
        "recall": float(curve["recall"][b]),
        "f1": float(curve["f1"][b]),
        "alert_rate": float(curve["alert_rate"][b]),
        "tp": int(curve["tp"][b]), "fp": int(curve["fp"][b]),
        "fn": int(curve["fn"][b]), "tn": int(curve["tn"][b]),
    }


def _downsample_pr_curve(curve, n_points=101):
    """~100 points along the recall axis, for the report.

    Sampled by recall rather than by threshold: recall is the axis an operator
    reasons about ("what fraction of attacks do I see?"), and threshold-uniform
    sampling wastes most of its points on the flat high-threshold tail where
    almost nothing changes. Recall is non-increasing in threshold, so for each
    target recall the highest threshold that still reaches it is also the
    highest-precision way to reach it.
    """
    rec = curve["recall"]
    out, seen = [], set()
    for target in np.linspace(0.0, 1.0, n_points):
        ok = np.nonzero(rec >= target)[0]
        if ok.size == 0:
            continue
        b = int(ok[-1])
        if b in seen:
            continue
        seen.add(b)
        out.append(_point(curve, b))
    return sorted(out, key=lambda d: d["alert_threshold"])


def fit_operating_point(pos_hist, neg_hist, *, criterion="budgeted_f1",
                        alert_budget=2.0, bins=RISK_BINS):
    """Choose the threshold Branch A's risk head alerts at, on VALIDATION.

    ## Why this has to be fitted at all

    `--risk-objective bce` changed what `risk_score` means: it is now
    P(next window is an attack window), not the old severity magnitude
    (0.0 benign, 0.50-0.96 by tactic). Serving's historical cut was 0.65,
    chosen against the severity scale. A calibrated probability on a ~17.5%
    base rate almost never reaches 0.65 -- measured on a synthetic head with
    this corpus's base rate, it fires on 0.9% of windows and recalls 5.4% of
    attacks. Carrying the old cut across would turn the detector off, and the
    failure presents to an operator as "no attacks today".

    ## Why the default is not max-F1

    Max-F1 is the obvious criterion and it is the wrong default alone, because
    it prices a false positive and a false negative the same and is blind to
    volume. A SOC has a finite alert budget; a threshold that produces 10,000
    alerts a shift produces zero read alerts. F1 cannot see that -- it has no
    term for how many windows fire.

    So the default is **max F1 subject to an alert-rate budget**:

        alert_rate <= alert_budget * base_rate      (default 2x)

    A budget expressed as a multiple of the base rate rather than an absolute
    number is scale-free: at perfect precision the alert rate IS the base rate,
    so 2x is "at most one false alert for every true one, at full recall", and
    the constraint keeps its meaning on a corpus with a different attack
    density. It is a guard rail rather than a thumb on the scale -- on the
    synthetic check in tests/test_branch_a_operating_point.py the unconstrained
    max-F1 point (alert rate 0.172) sits inside a 0.348 budget and the budget
    does not bind at all.

    `max_f1` (unconstrained) and `max_recall_at_budget` (highest recall inside
    the budget, for a deployment that would rather chase false positives than
    miss an intrusion) are both available. Whichever is used, the unconstrained
    max-F1 point is also recorded, so the cost of the budget is visible rather
    than implied.

    Degenerate points are excluded: a threshold that alerts on everything, or
    on nothing, is not an operating point. Same rule as
    `cyberworld_v4.benchmark.choose_threshold`.

    Returns None when the split cannot support a threshold (one class only),
    because a fabricated threshold would be silently served.
    """
    curve = _pr_curve_from_histograms(pos_hist, neg_hist, bins=bins)
    P, N = curve["n_positive"], curve["n_negative"]
    if P <= 0 or N <= 0:
        return {
            "fitted": False,
            "reason": (f"validation split has {int(P)} attack and {int(N)} benign "
                       f"windows; a threshold needs both classes"),
            "base_rate": curve["base_rate"],
        }

    base_rate = curve["base_rate"]
    budget = float(alert_budget) * base_rate
    usable = (curve["alert_rate"] > 0.0) & (curve["alert_rate"] < 1.0)
    if not usable.any():
        # Every threshold alerts on everything or on nothing. That happens
        # when the head emits one score for every window -- the collapsed
        # case -- and there is then no operating point to choose. Returning
        # the least-bad degenerate one would put "alert on everything" into
        # serving under the name of a fitted threshold.
        return {
            "fitted": False,
            "reason": ("every threshold is degenerate: the risk head's scores "
                       "do not separate the two classes at any cut, which "
                       "means the head is emitting a constant"),
            "base_rate": base_rate,
            "n_positive": int(P), "n_negative": int(N),
        }
    within = usable & (curve["alert_rate"] <= budget)

    b_max_f1 = int(np.argmax(np.where(usable, curve["f1"], -1.0)))
    max_f1_point = _point(curve, b_max_f1)

    if criterion == "max_f1":
        b, why = b_max_f1, "max F1 over all non-degenerate thresholds"
    elif criterion in ("budgeted_f1", "max_recall_at_budget"):
        if not within.any():
            # Every usable threshold is over budget. Say so and fall back to
            # max-F1 rather than returning the most extreme in-budget point,
            # which would be no point at all.
            b, why = b_max_f1, (
                f"max F1; NO threshold met the alert budget "
                f"{budget:.4f} ({alert_budget}x base rate {base_rate:.4f}), "
                f"so the budget was reported and not applied")
        elif criterion == "budgeted_f1":
            b = int(np.argmax(np.where(within, curve["f1"], -1.0)))
            why = (f"max F1 subject to alert_rate <= {budget:.4f} "
                   f"({alert_budget}x the {base_rate:.4f} validation base rate)")
        else:
            b = int(np.argmax(np.where(within, curve["recall"], -1.0)))
            why = (f"max recall subject to alert_rate <= {budget:.4f} "
                   f"({alert_budget}x the {base_rate:.4f} validation base rate)")
    else:
        raise ValueError(f"unknown operating-point criterion {criterion!r}")

    op = _point(curve, b)
    op.update({
        "fitted": True,
        "criterion": why,
        "criterion_name": criterion,
        "alert_budget_multiple": float(alert_budget),
        "alert_budget": budget,
        "base_rate": base_rate,
        "n_positive": int(P), "n_negative": int(N),
        "split": "validation",
        "score_bins": int(bins),
        "unconstrained_max_f1": max_f1_point,
        "curve": _downsample_pr_curve(curve),
    })
    return op


def _print_operating_point(op, where):
    if not op:
        return
    if not op.get("fitted"):
        print(f"  operating point [{where}]: NOT FITTED -- {op.get('reason')}",
              flush=True)
        return
    m = op["unconstrained_max_f1"]
    print(f"  operating point [{where}]: threshold={op['alert_threshold']:.4f} "
          f"precision={op['precision']:.4f} recall={op['recall']:.4f} "
          f"f1={op['f1']:.4f} alert_rate={op['alert_rate']:.4f} "
          f"(base rate {op['base_rate']:.4f})", flush=True)
    print(f"    criterion: {op['criterion']}", flush=True)
    print(f"    unconstrained max-F1 would be threshold={m['alert_threshold']:.4f} "
          f"f1={m['f1']:.4f} recall={m['recall']:.4f} "
          f"alert_rate={m['alert_rate']:.4f}", flush=True)
    # What the historical cut would have done to the SAME head. This is the
    # number that says whether carrying 0.65 across would have been silent.
    legacy = op.get("legacy_0_65")
    if legacy:
        print(f"    the legacy 0.65 cut on this head: recall={legacy['recall']:.4f} "
              f"alert_rate={legacy['alert_rate']:.4f} f1={legacy['f1']:.4f}",
              flush=True)


#: Gradation is a 4-level severity ladder, not a technique. GRADATION_LEVELS
#: maps several coarse categories onto the same level (InitialAccess and
#: Execution both -> 2; C2, LateralMovement, Exfiltration and Impact all -> 3),
#: so a level's name has to say which categories it covers or a per-class table
#: is unreadable.
GRADATION_NAMES = {
    0: "0 Benign",
    1: "1 Recon/Unknown",
    2: "2 InitialAccess/Exec",
    3: "3 C2/Lateral/Exfil/Impact",
}


def _print_per_class(per_class, where, names=None, label="technique"):
    """Per-class precision/recall/f1/support, largest class first."""
    if not per_class:
        return
    inv = names if names is not None else {v: k for k, v in TECH_TO_IDX.items()}
    print(f"  per-class {label} ({where}):", flush=True)
    print(f"    {label:<28} {'prec':>6} {'recall':>7} {'f1':>6} "
          f"{'support':>9} {'predicted':>10}", flush=True)
    for c, m in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
        print(f"    {inv.get(c, f'class_{c}'):<28} {m['precision']:>6.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>6.3f} {m['support']:>9,} "
              f"{m['predicted']:>10,}", flush=True)


def _branch_a_health(metrics):
    """Problems the training guard repeats in its diagnosis (empty = healthy).

    The same conditions the _warn_if_* helpers print, as short strings the
    guard can carry into its step-back / stop report.
    """
    out = []
    t_pred = metrics.get("tech_classes_predicted", 0)
    t_present = metrics.get("tech_classes_present", 0)
    if t_present > 1 and t_pred <= 1:
        out.append(f"technique head predicts {t_pred} of {t_present} classes: "
                   f"collapsed onto the majority class")
    elif t_present > 1 and (metrics.get("tech_macro_f1_lift") or 0.0) <= 0.0:
        out.append("technique macro-F1 is no better than always predicting the majority class")
    auc = float(metrics.get("risk_auc", float("nan")))
    if auc == auc and auc < 0.55:
        out.append(f"risk head AUC {auc:.3f}: at or near chance")
    g_pred = metrics.get("gradation_classes_predicted", 0)
    g_present = metrics.get("gradation_classes_present", 0)
    if g_present > 1 and g_pred <= 1:
        out.append(f"gradation head predicts {g_pred} of {g_present} levels: collapsed")
    return out


def _warn_if_risk_head_useless(metrics, where):
    """Say so when the risk head is not beating its own base rate.

    The 2026-09-21 head scored MAE 0.2268 where the smooth-L1-optimal constant
    scores 0.2279 -- it had learned the unconditional mean and nothing else,
    and no reported number said so. AUC at 0.5 is chance; a Brier score at or
    above base_rate*(1-base_rate) means predicting the base rate for every
    host would do as well.
    """
    auc = float(metrics.get("risk_auc", float("nan")))
    brier = float(metrics.get("risk_brier", float("nan")))
    base = float(metrics.get("risk_brier_baseline", float("nan")))
    if auc == auc and auc < 0.55:
        print(f"  WARNING [{where}]: risk AUC {auc:.4f} is at or near chance -- "
              f"the head is not ranking attack windows above benign ones.",
              flush=True)
    if brier == brier and base == base and brier >= base:
        print(f"  WARNING [{where}]: risk Brier {brier:.4f} is no better than "
              f"predicting the base rate {metrics.get('risk_base_rate', 0):.4f} "
              f"for every host ({base:.4f}).", flush=True)


def _metrics_from_confusion(cm):
    """Accuracy, macro F1, majority baseline, lift and per-class, from one cm.

    Shared by the technique and gradation heads. It was written once for
    technique; the gradation head had **no evaluation at all** before
    2026-09-22 beyond aggregate accuracy, which on a ladder whose level 0 is
    82.5% of the corpus is the same metric that could not fail for technique.
    Two heads with the same class-prior problem get the same instrument.
    """
    cm = np.asarray(cm)
    support = cm.sum(axis=1)             # true count per class
    predicted = cm.sum(axis=0)           # predicted count per class
    tp = np.diag(cm)
    present = support > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    total = int(support.sum())
    accuracy = float(tp.sum() / max(total, 1))
    baseline = float(support.max() / max(total, 1)) if total else 0.0

    # The macro F1 a majority-class predictor would score, exactly from the
    # support: it gets recall 1.0 and precision equal to that class's share on
    # the majority class, and F1 = 0 on every other present class.
    #
    # Comparing macro F1 against the ACCURACY baseline is a category error,
    # and it made this file's own warning misleading. On epoch 1 of the
    # 2026-09-22 run the technique head scored accuracy 0.791 against a 0.900
    # majority share, so the warning called it "adding almost nothing" --
    # while its macro F1 of 0.397 beat the majority predictor's 0.316 by
    # +0.081. Giving up majority-class accuracy to gain minority recall is
    # exactly what focal loss is for, so judging these heads on accuracy
    # punishes them for working as intended.
    n_present = int(present.sum())
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    macro_baseline = (float((2 * baseline / (baseline + 1.0)) / n_present)
                      if n_present and total else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_baseline": baseline,
        "lift_over_baseline": accuracy - baseline,
        "macro_f1_baseline": macro_baseline,
        "macro_f1_lift": macro_f1 - macro_baseline,
        "classes_present": int(present.sum()),
        "classes_predicted": int((predicted > 0).sum()),
        "per_class": {
            int(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
                "predicted": int(predicted[c]),
            }
            for c in range(cm.shape[0]) if support[c] > 0 or predicted[c] > 0
        },
        "n": total,
    }


def _warn_if_gradation_collapsed(metrics, where):
    """The gradation head gets the same scrutiny as the technique head.

    It had never been scored beyond `gradation_accuracy`, which was added on
    2026-09-22. That single number cannot fail on this corpus for exactly the
    reason the technique accuracy could not: GRADATION_LEVELS sends every
    benign window to level 0, so a head that emits 0 for everything scores
    ~0.825 and looks like it works.
    """
    pred = metrics.get("gradation_classes_predicted", 0)
    present = metrics.get("gradation_classes_present", 0)
    lift = metrics.get("gradation_lift_over_baseline", 0.0)
    f1 = metrics.get("gradation_macro_f1", 0.0)
    acc = metrics.get("gradation_accuracy") or 0.0
    if present <= 1:
        return
    if pred <= 1:
        print(f"  WARNING [{where}]: gradation head predicts a SINGLE level for "
              f"every input ({present} levels present). Its accuracy {acc:.3f} "
              f"is the class prior, not a result.", flush=True)
    elif metrics.get("gradation_macro_f1_lift") is not None and \
            metrics["gradation_macro_f1_lift"] <= 0.0:
        # Macro F1, not accuracy -- see the technique warning for why.
        print(f"  WARNING [{where}]: gradation macro F1 {f1:.3f} is at or below "
              f"what a majority-level predictor scores "
              f"({metrics.get('gradation_macro_f1_baseline', 0):.3f}) -- the head "
              f"is adding nothing over a constant.", flush=True)
    elif f1 < 0.2 and present > 2:
        print(f"  WARNING [{where}]: gradation macro F1 {f1:.3f} over {present} "
              f"levels -- accuracy {acc:.3f} is carried by level 0 while the "
              f"escalation levels are largely missed. Consider "
              f"--gradation-class-weights.", flush=True)


def _warn_if_head_collapsed(metrics, where):
    """Say plainly when accuracy is coming from the class prior, not the model.

    On this corpus 82.5% of validation samples are Benign, so a constant
    predictor scores ~0.825. Accuracy alone therefore cannot fail visibly. The
    TGNE category head scored 0.83 while emitting one class for every input and
    went unnoticed until macro F1 was computed -- these checks exist so that
    cannot happen quietly a second time.
    """
    pred = metrics.get("tech_classes_predicted", 0)
    present = metrics.get("tech_classes_present", 0)
    lift = metrics.get("tech_lift_over_baseline", 0.0)
    f1 = metrics.get("tech_macro_f1", 0.0)

    if pred <= 1 and present > 1:
        print(f"  WARNING [{where}]: technique head predicts a SINGLE class for "
              f"every input ({present} classes present in the data). Its "
              f"accuracy {metrics.get('tech_accuracy', 0):.3f} is the class "
              f"prior, not a result.", flush=True)
    elif metrics.get("tech_macro_f1_lift") is not None and \
            metrics["tech_macro_f1_lift"] <= 0.0 and present > 1:
        # Judged on macro F1, never on accuracy. A focal-loss head trades
        # majority-class accuracy for minority recall by design, so accuracy
        # below the majority share is expected and is not evidence of
        # failure -- macro F1 at or below what a constant predictor scores is.
        print(f"  WARNING [{where}]: technique macro F1 "
              f"{metrics.get('tech_macro_f1', 0):.3f} is at or below what a "
              f"majority-class predictor scores "
              f"({metrics.get('tech_macro_f1_baseline', 0):.3f}) -- the head is "
              f"adding nothing over a constant.", flush=True)
    elif f1 < 0.2 and present > 2:
        print(f"  WARNING [{where}]: macro F1 {f1:.3f} over {present} classes -- "
              f"accuracy {metrics.get('tech_accuracy', 0):.3f} is carried by the "
              f"majority class while minority classes are largely missed.",
              flush=True)


from cyberworld_v4.device_hist import device_hist as _hist  # noqa: E402  (no CUDA sync)


def _evaluate(model, loader, device, num_techniques=None, num_gradations=4,
              risk_positive_above=0.0, collect_logits=False):
    """Validation pass.

    Accumulators live on the device and are read once at the end. An earlier
    version called `.item()` for the loss, `.cpu().numpy()` for the risk errors
    and `.item()` for the technique hits -- three host-device syncs per batch --
    and grew a Python list to one float per validation sample (1.02M of them).

    ## Why this reports more than accuracy

    82.5% of validation samples are Benign (val_positive_rate 0.1752). A model
    that predicts Benign for everything scores ~0.825 aggregate accuracy, so
    `tech_accuracy=0.880` is only 5.5 points above a constant classifier and
    cannot, by itself, distinguish a working head from a collapsed one. That is
    not hypothetical: the TGNE category head scored 0.83 while predicting a
    single class, and was only caught when macro F1 was computed.

    So this also returns macro F1, the majority-class baseline, the lift over
    it, how many distinct classes the model actually predicts, and per-class
    precision/recall/support. A confusion matrix is accumulated on-device with
    a single bincount per batch, which adds no host-device sync.
    """
    model.eval()
    nb = 0
    non_blocking = (device == "cuda")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    abs_err_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_risk = 0
    conf_hist = torch.zeros(RISK_BINS, device=device, dtype=torch.float64)
    pos_hist = torch.zeros(RISK_BINS, device=device, dtype=torch.long)
    neg_hist = torch.zeros(RISK_BINS, device=device, dtype=torch.long)
    brier_sum = torch.zeros((), device=device, dtype=torch.float64)
    prob_sum = torch.zeros((), device=device, dtype=torch.float64)
    # |prediction - target|, binned. The conformal half-width is a quantile of
    # this, and a quantile needs the distribution, not a running mean. A fixed
    # histogram gives it to within one bin (5e-4) for 16 KB, where the exact
    # route would either hold 1.02M floats or cost a second pass.
    resid_hist = torch.zeros(RISK_BINS, device=device, dtype=torch.long)

    task_sums = {k: torch.zeros((), device=device, dtype=torch.float64)
                 for k in ("loss_risk", "loss_tech", "loss_grad",
                           "weight_risk", "weight_tech", "weight_grad")}

    C = int(num_techniques) if num_techniques else len(TECHNIQUE_VOCAB)
    # confusion[t * C + p] -- flat so one bincount per batch suffices
    confusion = torch.zeros(C * C, device=device, dtype=torch.long)

    grad_correct = torch.zeros((), device=device, dtype=torch.long)
    grad_total = 0
    G = int(num_gradations) if num_gradations else 4
    grad_confusion = torch.zeros(G * G, device=device, dtype=torch.long)

    # Raw technique logits + labels, for post-hoc temperature scaling. Held on
    # the host because the device copy would be [N, C] float32 -- 57 MB at the
    # 1.02M-sample validation split, which is fine, but a calibration pass is
    # the only caller and it runs once.
    keep_logits, keep_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device, non_blocking=non_blocking)
            targets = {
                "risk": batch["risk"].to(device, non_blocking=non_blocking),
                "technique": batch["technique"].to(device, non_blocking=non_blocking),
                "gradation": batch["gradation"].to(device, non_blocking=non_blocking),
            }
            # Real elapsed time between the history steps -- see
            # MultiTaskLSTM.time_proj. Validation must see what training sees.
            t_hist = (batch["t_history"].to(device, non_blocking=non_blocking)
                      if "t_history" in batch else None)
            predictions = model(x, t_history=t_hist)
            loss, parts = model.compute_loss(predictions, targets)
            loss_sum += loss.detach().double().sum()
            # Per-task losses and their learned weights. Without these the
            # total is uninterpretable: the 2026-09-21 run reported a
            # validation loss of 0.66 against a test loss of 26.36 with no way
            # to see that two of the three weights had saturated at exp(3)=20
            # and were multiplying everything.
            for _k, _acc in task_sums.items():
                _v = parts.get(_k)
                if _v is not None:
                    _acc += _v.double().reshape(())
            nb += 1
            err = (predictions["risk_score"] - targets["risk"]).abs()
            abs_err_sum += err.double().sum()
            n_risk += int(err.numel())
            _rb = (err.clamp(0, 1) * (RISK_BINS - 1)).long().clamp_(0, RISK_BINS - 1)
            resid_hist += _hist(_rb.reshape(-1), RISK_BINS)

            # Risk as a probability: AUC, Brier and calibration, accumulated
            # from a fixed histogram so 1.02M samples cost O(bins) memory and
            # no host-device sync. MAE alone cannot judge this head -- on a
            # target that is 0 for 82.5% of samples the MAE-optimal constant
            # is 0, so a well-fit head can still "lose" to predicting nothing.
            # Which windows count as positive for AUC / Brier / ECE / the
            # operating point. With the severity target the event is "this is
            # an attack window", i.e. risk > 0. With the hazard target
            # (exp(-dt/tau)) risk > 0 degenerates to "this host is attacked at
            # SOME later point", which discards the timing the hazard exists
            # to carry; the operationally meaningful event there is "an attack
            # occurs within one forecast horizon", i.e. hazard >= exp(-1).
            # `risk_positive_above` carries whichever the caller means.
            _p = predictions["risk_score"].clamp(0, 1).reshape(-1)
            _y = (targets["risk"] > risk_positive_above).reshape(-1)
            brier_sum += ((_p - _y.to(_p.dtype)) ** 2).double().sum()
            _b = (_p * (RISK_BINS - 1)).long().clamp_(0, RISK_BINS - 1)
            pos_hist += _hist(_b, RISK_BINS, _y.long())
            neg_hist += _hist(_b, RISK_BINS, (~_y).long())
            # Sum of the predicted probabilities per bin, so ECE can use each
            # bin's ACTUAL mean confidence rather than its nominal centre.
            conf_hist += _hist(_b, RISK_BINS, _p.double())
            prob_sum += _p.double().sum()

            pred_t = predictions["technique_logits"].argmax(dim=-1)
            true_t = targets["technique"]
            # No separate hit counter: the confusion matrix's trace IS the
            # number correct, so accumulating it twice bought one more
            # device tensor and an extra `.item()` sync at the end.
            confusion += _hist(true_t * C + pred_t, C * C)

            if "gradation_logits" in predictions:
                pred_g = predictions["gradation_logits"].argmax(dim=-1)
                true_g = targets["gradation"]
                grad_correct += (pred_g == true_g).sum()
                grad_total += int(true_g.numel())
                grad_confusion += _hist(true_g * G + pred_g, G * G)

            if collect_logits:
                _raw = predictions.get("technique_logits_raw")
                if _raw is None:
                    _raw = predictions["technique_logits"]
                keep_logits.append(_raw.detach().float().cpu())
                keep_labels.append(true_t.detach().cpu())

    cm = confusion.reshape(C, C).cpu().numpy()
    tech = _metrics_from_confusion(cm)
    macro_f1 = tech["macro_f1"]
    baseline = tech["majority_baseline"]
    accuracy = tech["accuracy"]
    per_class = tech["per_class"]

    # The gradation head, scored the same way. Until 2026-09-22 nothing
    # measured it at all; then it got aggregate accuracy, which on a ladder
    # that is 82.5% level 0 is the metric that already failed to fail for
    # technique.
    grad = _metrics_from_confusion(grad_confusion.reshape(G, G).cpu().numpy())

    # AUC exactly from the histogram: for each score bin, every negative in a
    # strictly lower bin is a win and every negative in the same bin is a tie.
    ph = pos_hist.double().cpu().numpy()
    nh = neg_hist.double().cpu().numpy()
    ch = conf_hist.cpu().numpy()
    P, N = float(ph.sum()), float(nh.sum())
    if P > 0 and N > 0:
        neg_below = np.concatenate([[0.0], np.cumsum(nh)[:-1]])
        risk_auc = float((ph * (neg_below + 0.5 * nh)).sum() / (P * N))
        # ECE with each bin's MEASURED mean confidence.
        #
        # This used the nominal bin centre `(i + 0.5) / RISK_BINS`, which
        # disagreed with how scores are binned everywhere else in this file:
        # a score lands in `floor(p * (RISK_BINS - 1))`, so bin i spans
        # [i/(B-1), (i+1)/(B-1)) and its centre is (i+0.5)/(B-1), not
        # (i+0.5)/B. The mismatch reached 5.0e-4 at the top of the range --
        # negligible against an ECE of 0.07, but up to 25% of one at 0.002,
        # and ECE at that scale is exactly where a calibration claim is made.
        #
        # Summing the probabilities per bin removes the approximation rather
        # than correcting it: this is the textbook definition, and it is exact
        # whatever the binning.
        cnt = ph + nh
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_in_bin = np.where(cnt > 0, ph / np.maximum(cnt, 1), 0.0)
            conf = np.where(cnt > 0, ch / np.maximum(cnt, 1), 0.0)
        risk_ece = float((cnt * np.abs(acc_in_bin - conf)).sum() / max(cnt.sum(), 1))
    else:
        # AUC and ECE need both classes; the BASE RATE does not, and reporting
        # it as 0.0 on an all-attack split (every CTU-13 scenario is one) said
        # the opposite of the truth and drove `risk_brier_baseline` to 0, which
        # made the "no better than the base rate" warning fire unconditionally.
        risk_auc, risk_ece = float("nan"), float("nan")
    base_rate = P / (P + N) if (P + N) > 0 else 0.0
    n_prob = max(int(P + N), 1)
    out_tasks = {k: (float((v / nb).item()) if nb else 0.0) for k, v in task_sums.items()}
    return {
        "loss": float((loss_sum / nb).item()) if nb else 0.0,
        "risk_mae": float((abs_err_sum / n_risk).item()) if n_risk else 0.0,
        "tech_accuracy": accuracy,
        # -- risk as a probability, judged the way a probability must be --
        "risk_auc": risk_auc,
        "risk_brier": float((brier_sum / n_prob).item()),
        "risk_brier_baseline": base_rate * (1 - base_rate),   # always predict the base rate
        "risk_ece": risk_ece,
        "risk_base_rate": base_rate,
        "risk_mean_prediction": float((prob_sum / n_prob).item()),
        **out_tasks,
        # -- the metrics that can tell a working head from a collapsed one --
        "tech_macro_f1": macro_f1,
        "tech_macro_f1_baseline": tech["macro_f1_baseline"],
        "tech_macro_f1_lift": tech["macro_f1_lift"],
        "tech_majority_baseline": baseline,
        "tech_lift_over_baseline": accuracy - baseline,
        "tech_classes_present": tech["classes_present"],
        "tech_classes_predicted": tech["classes_predicted"],
        "tech_per_class": per_class,
        # Full matrix, so a cross-dataset test can show what an UNSEEN class
        # was predicted as (cyberworld_v4/cross_dataset.py).
        "tech_confusion": cm,
        # -- the gradation head, judged the same way as the technique head --
        "gradation_accuracy": (float(grad_correct.item()) / grad_total) if grad_total else None,
        "gradation_macro_f1": grad["macro_f1"],
        "gradation_macro_f1_baseline": grad["macro_f1_baseline"],
        "gradation_macro_f1_lift": grad["macro_f1_lift"],
        "gradation_majority_baseline": grad["majority_baseline"],
        "gradation_lift_over_baseline": grad["lift_over_baseline"],
        "gradation_classes_present": grad["classes_present"],
        "gradation_classes_predicted": grad["classes_predicted"],
        "gradation_per_class": grad["per_class"],
        # Raw histograms, so a caller can fit an operating point without a
        # second pass over the split. 2 x 2000 int64 -- 32 KB.
        "risk_pos_hist": pos_hist.cpu().numpy(),
        "risk_neg_hist": neg_hist.cpu().numpy(),
        "risk_resid_hist": resid_hist.cpu().numpy(),
        "risk_positive_above": float(risk_positive_above),
        "technique_logits": (torch.cat(keep_logits) if keep_logits else None),
        "technique_labels": (torch.cat(keep_labels) if keep_labels else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cic-dir", type=Path, default=None,
                        help="CIC-IDS-2018 CSV directory. Refused under --split-scheme "
                             "cross_year: 9 of its 10 days fabricate host IPs; use --pcap-root.")
    parser.add_argument("--ctu-dir", type=Path, default=None, help="CTU-13 directory")
    parser.add_argument("--cic2017-dir", type=Path, default=None,
                        help="CIC-IDS-2017 CSV directory (TrafficLabelling). Parsed with the "
                             "CIC-2017 adapter, which repairs its 12-hour clock.")
    parser.add_argument("--pcap-root", type=Path, default=None,
                        help="CIC-IDS-2018 PCAP root: one <day>_pcap directory per day. "
                             "Real host addresses; labels come from --cic2018-csv-dir.")
    parser.add_argument("--cic2018-csv-dir", type=Path, default=None,
                        help="CIC-2018 <day>_csv.csv files used ONLY as labels for --pcap-root")
    parser.add_argument("--pcap-max-windows-per-day", type=int, default=None)
    parser.add_argument("--pcap-window-stride", type=int, default=1)
    parser.add_argument("--allow-cic2018-csv", action="store_true",
                        help="Accept CIC-2018 CSVs under cross_year despite their synthetic IPs")
    parser.add_argument("--split-scheme", choices=("frozen", "cross_year", "cross_year_ctu"),
                        default="frozen",
                        help="'frozen': splits.lock.json as is. 'cross_year': train on CIC-2018 "
                             "(its lock train days), tune on its other days, test ONCE on all "
                             "of CIC-2017; CTU-13 unused. 'cross_year_ctu': the same, with CTU-13 "
                             "added to train/val only (pass --ctu-dir). See "
                             "data_unification/split_policy.py.")
    parser.add_argument("--tgne", type=Path, default=None,
                        help="Encoder checkpoint to extract host states with (default: the "
                             "served one). The encoder comparison passes each arm's encoder.")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="Also write validation/test metrics, including the unseen-class "
                             "report, to this JSON file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=None,
                        help="Cap records kept per capture. Default None = FULL DENSITY.")
    parser.add_argument("--spill-dir", type=Path, default=None, help="Write the bulk trajectory feature block here instead of RAM (np.memmap)")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth record across a wider read, instead of a plain file-prefix (applied in data_unification/training_sources.read_capture)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--risk-objective", choices=("bce", "soft_bce", "smooth_l1"),
                        default="bce",
                        help="How the risk head is trained. 'bce' predicts "
                             "P(next window is an attack window) -- calibrated, "
                             "and judged by AUC/Brier which match the loss. "
                             "'soft_bce' is the same proper scoring rule "
                             "against a CONTINUOUS target in [0,1]; it is the "
                             "right pairing for --risk-target hazard, which "
                             "'bce' would binarise and throw away. "
                             "'smooth_l1' is the original scalar regression, "
                             "kept to reproduce the 2026-09-21 run; against "
                             "the severity target it cannot beat a constant on "
                             "MAE by construction.")
    parser.add_argument("--risk-target", choices=("severity", "hazard"),
                        default="severity",
                        help="What the risk head is asked to predict. "
                             "'severity' (default, unchanged) is the "
                             "base_severity(tactic)+density+volume score "
                             "written during extraction -- it describes THIS "
                             "window, so predicting it is detection with a "
                             "one-step delay rather than forecasting. "
                             "'hazard' is exp(-dt/tau), seconds until the "
                             "host's next attack window: continuous, monotone "
                             "in time and independent of the tactic label. The "
                             "default is deliberately unchanged -- swapping the "
                             "target changes what the model learns and deserves "
                             "a measured A/B, not a silent default.")
    parser.add_argument("--hazard-tau", type=float, default=None,
                        help="Decay scale for --risk-target hazard, in seconds. "
                             "Default forecast_steps * window_seconds (= 10.0s "
                             "under v4), so a host exactly one forecast horizon "
                             "from an attack scores exp(-1) = 0.368 -- which is "
                             "also the cut that defines the positive class for "
                             "AUC/Brier/the operating point under this target.")
    parser.add_argument("--architecture", choices=("paper", "legacy"), default="paper",
                        help="'paper' = Vitulyova et al. 2025 (Computers 14:301): 1 x 256 "
                             "LSTM, heads on the last hidden state H_t, linear heads, scalar "
                             "gradation trained with MSE, fixed loss weights 0.5/0.3/0.2, "
                             "class-weighted CE for techniques. 'legacy' = the 2 x 64 "
                             "attention/MLP/uncertainty-weighted model earlier checkpoints "
                             "were trained as.")
    parser.add_argument("--focal-gamma", type=float, default=None,
                        help="Focusing exponent of the technique focal loss. "
                             "2.0 is Lin et al.'s value and the one this head "
                             "has always used; exposed because gamma and the "
                             "per-class alpha both correct imbalance and their "
                             "combination has never been swept on this corpus. "
                             "Default: 0 with --architecture paper (alpha-weighted CE), "
                             "2 with legacy.")
    parser.add_argument("--no-focal-alpha", action="store_true",
                        help="Train the technique head with a FLAT focal alpha. "
                             "The default is the damped, clipped, geometric-mean "
                             "centred inverse-frequency weighting computed from "
                             "the training split -- the class balancing the "
                             "loss's own docstring has always claimed and, "
                             "before 2026-09-22, never did (alpha was None).")
    parser.add_argument("--gradation-class-weights", action="store_true",
                        help="Weight the gradation cross-entropy by inverse "
                             "class frequency. OFF by default: the head had "
                             "never been evaluated at all until 2026-09-22, and "
                             "re-weighting a head before measuring it is "
                             "guessing. Turn it on when the per-class table "
                             "shows it collapsing onto level 0.")
    parser.add_argument("--operating-point-criterion",
                        choices=("budgeted_f1", "max_f1", "max_recall_at_budget"),
                        default="budgeted_f1",
                        help="How the served alert threshold is chosen from the "
                             "validation precision-recall curve. See "
                             "fit_operating_point for why max-F1 is not the "
                             "default.")
    parser.add_argument("--alert-budget", type=float, default=2.0,
                        help="Alert-rate budget as a MULTIPLE of the validation "
                             "base rate. 2.0 means at most one false alert per "
                             "true one at full recall. Scale-free, so it keeps "
                             "its meaning on a corpus with a different attack "
                             "density.")
    parser.add_argument("--no-fit-temperature", action="store_true",
                        help="Skip post-hoc temperature scaling of the technique "
                             "logits. Fitting is on by default and happens after "
                             "training, on validation, with the model frozen.")
    parser.add_argument("--conformal-alpha", type=float, default=0.05,
                        help="Miscoverage rate for the risk head's split-conformal "
                             "interval, fitted on validation residuals. On a "
                             "binary target a 95%% interval is wide by "
                             "construction; that is the honest answer, and the "
                             "reason the hardcoded +/-0.05 was not one.")
    parser.add_argument("--max-gap-seconds", type=float, default=None,
                        help="Cut each host's trajectory where consecutive active "
                             "windows are further apart than this, so no sample's "
                             "history or target crosses the gap. Off by default. "
                             "Rows are ACTIVE windows, not clock ticks: a CTU-13 "
                             "host's next window is a median 736 s away, against a "
                             "10 s contract horizon. Enabling this drops samples -- "
                             "see claude_latest_analysis/30_time_gaps.md.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after N epochs without improving --select-on. "
                             "The best checkpoint is already written, so this "
                             "cannot cost quality.")
    parser.add_argument("--step-back-after", type=int, default=2,
                        help="After N epochs without improvement, restore the best "
                             "weights and halve the LR before trying again "
                             "(cyberworld_v4/training_guard.py). Must be < --patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore a resume point left by a crashed run of the same "
                             "command and start from epoch 1")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="Gradient-norm clip; 0 disables clipping (norms still logged)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear LR warmup; default ~5%% of the first epoch, at most 500")
    parser.add_argument("--select-on", choices=("composite", "macro_f1", "val_loss"),
                        default="composite",
                        help="Which validation metric picks the kept checkpoint. "
                             "Default 'composite' = 0.5*macro_f1 + 0.5*risk_auc. "
                             "(It read '0.5*(1-risk_mae)' until 2026-09-22; that "
                             "was the first version and the help text outlived "
                             "it. MAE is median-seeking, so on a target that is "
                             "0 for 82.5%% of samples it rewards predicting "
                             "nothing -- which is why it was replaced by AUC.) "
                             "'val_loss' was the previous default but is not "
                             "comparable across epochs: the uncertainty-weighted "
                             "loss contains learned log-variance terms that drift "
                             "(0 -> -7.04 over the 2026-09-21 run), so part of its "
                             "fall is the weighting moving rather than the model "
                             "improving.")
    parser.add_argument("--eval-only", type=str, default=None,
                        metavar="CKPT",
                        help="Score an existing checkpoint and exit; no "
                             "training. The post-hoc fits ARE written back "
                             "into it -- the operating point, the technique "
                             "temperature and the conformal half-width -- "
                             "because those need only a frozen model and a "
                             "validation pass, and a finished checkpoint "
                             "should not need a retrain to acquire them. "
                             "Weights are never modified. '-' means the path "
                             "given by --output.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes. 0 loads in the main "
                             "process, which serialises data loading with GPU "
                             "compute -- measured at 35.7%% of wall clock on the "
                             "full-density run, i.e. the GPU idled for a third "
                             "of every epoch. Default 4 is deliberately modest: "
                             "loading costs ~1.0x compute, so 2 already hide it, "
                             "and this box has ~9 GiB free under a 17 GiB cap.")
    parser.add_argument("--capture-cache", type=str, default=None,
                        help="Directory caching each capture parsed to compact columns "
                             "(data_unification/capture_columns.py), shared by every "
                             "Branch A / downstream run with the same inputs and parsing "
                             "code. Default: $CYBERWORLD_CAPTURE_CACHE, else "
                             "<--spill-dir>/capture_cache. 'off' reads records as before.")
    parser.add_argument("--ingest-workers", type=int, default=3,
                        help="Captures parsed at once, each in its own process (a PCAP "
                             "day takes ~30 min on one core and ~1-2 GB while parsing). "
                             "Extraction itself stays sequential and in capture order.")
    parser.add_argument("--extract-workers", type=int, default=3,
                        help="Captures extracted at once, each in its own CPU process "
                             "(data_unification/parallel_extract.py); 1 = in this process. "
                             "Needs the capture cache. Each worker holds one capture "
                             "(~1.2 GB for a CIC-2018 PCAP day).")
    parser.add_argument("--log-every", type=int, default=2000,
                        help="Print a progress line every N batches. At full "
                             "density an epoch is ~161k batches; with no "
                             "progress line a run is unobservable for an hour.")
    parser.add_argument("--min-history-steps", type=int, default=None,
                        help="a sample must have at least this many REAL history "
                             "steps; default is the full window. Pass 1 to restore "
                             "the old zero-padding behaviour (the median host on "
                             "this corpus has one window, so the default drops a lot "
                             "and says so).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="Run the training step's forward/backward eagerly instead of "
                             "replaying it as a CUDA graph (cyberworld_v4/graphed_step.py; "
                             "bit-identical, ~2.7x less host time per step). Also "
                             "CYBERWORLD_CUDA_GRAPH=0.")
    parser.add_argument("--legacy-loader", action="store_true",
                        help="Per-sample __getitem__ + default collate (the pre-2026-10 "
                             "loader). The default batched loader yields bit-identical "
                             "batches in the same order from a host-major feature copy "
                             "(tests/test_branch_a_batched_loader.py); this is a fallback.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _cfg = DEFAULT_CONFIG
    _manifest = ExperimentManifest.create(
        f"branch_a_v4_seed{args.seed}_{args.split_scheme}",
        seed=args.seed,
        config=_cfg,
        repo=Path(__file__).resolve().parent.parent,
        dataset_sources=[str(x) for x in (args.cic_dir, args.ctu_dir, args.cic2017_dir,
                                          args.pcap_root) if x],
    )
    set_all_seeds(args.seed)

    require_full_density(
        'Branch A retrain',
        stride=args.stride,
        rows_per_file=args.rows_per_file,
        pcap_window_stride=args.pcap_window_stride,
        pcap_max_windows_per_day=args.pcap_max_windows_per_day,
    )
    if args.pcap_root and not args.cic2018_csv_dir:
        parser.error("--pcap-root needs --cic2018-csv-dir: PCAP packets carry no labels")

    # Three-way, capture-disjoint, and READ FROM THE FROZEN LOCK.
    #
    # This used to slice a *sorted* file list 70/15/15, so "train" meant
    # "alphabetically first" and the assignment shifted whenever a file was
    # added or renamed -- results were not comparable across retrains, which is
    # the entire reason data_unification/splits.lock.json exists. The lock had
    # zero consumers; now it has this one.
    #
    # Test is scored once, after the model is frozen, and never influences
    # training.
    from data_unification.training_sources import (
        describe as _describe_captures, discover_captures, read_capture)
    partition = discover_captures(
        scheme=args.split_scheme,
        cic2017_dir=args.cic2017_dir,
        cic2018_dir=args.cic_dir,
        ctu13_dir=args.ctu_dir,
        pcap2018_root=args.pcap_root,
        allow_cic2018_csv=args.allow_cic2018_csv,
    )
    train_files, val_files, test_files = (
        partition["train"], partition["val"], partition["test"],
    )
    for _name, _files in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not _files:
            raise RuntimeError(
                f"split '{_name}' ({args.split_scheme}) matched no captures. "
                f"Refusing to train on a split that does not exist.")
    print(f"{args.split_scheme} split -- {_describe_captures(partition)}", flush=True)
    if is_cross_year(args.split_scheme):
        print("  every tuning decision (early stopping, threshold, temperature, conformal "
              "width) uses CIC-2018 validation days only; CIC-2017 is scored once, at the end",
              flush=True)
    import time
    tgn = build_or_load_tgne_ta(checkpoint_path=str(args.tgne) if args.tgne else None)
    # Contract-bound (v4). Previously 2.0s / seq_len=5 hardcoded, which matched
    # the v3 contract by coincidence rather than by construction. Under v4 this
    # produces history_steps=15, so it yields a v4 checkpoint, not a v3 one.
    _c = get_contract()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=_c.window_seconds,
                                        spill_dir=str(args.spill_dir) if args.spill_dir else None)
    # Load -> extract -> free, one split at a time. Holding both record lists
    # at once costs an extra ~2.8 GB at full density for no reason: the val
    # records are not needed until the train split has already been reduced to
    # samples.
    record_counts: Dict[str, int] = {}
    neighbor_exposure: Dict[str, Dict[str, Any]] = {}

    # Captures as compact cached columns (data_unification/capture_columns.py)
    # rather than ~1 KB/record UnifiedFlowRecord lists: a CIC-2018 PCAP day is
    # ~12M records, ~12 GB as objects, so one day alone breaks the 17 GiB cap.
    # Same records, same order, same snapshots -- tests/test_capture_columns.py.
    from data_unification.capture_columns import ColumnSpec, columns_plan, iter_capture_columns
    _cplan = columns_plan(args.capture_cache, args.spill_dir, extractor)
    print(_cplan.describe(args.ingest_workers), flush=True)

    def _iter_capture_inputs(files):
        """(capture, records or None, columns or None, coverage), in capture order."""
        if not _cplan.enabled:
            for f in files:
                cov: List[dict] = []
                recs = read_capture(
                    f, window_seconds=_c.window_seconds, pcap_label_dir=args.cic2018_csv_dir,
                    rows_per_file=args.rows_per_file, stride=args.stride,
                    pcap_max_windows=args.pcap_max_windows_per_day,
                    pcap_window_stride=args.pcap_window_stride, coverage=cov)
                yield f, recs, None, cov[0]
            return
        specs = [ColumnSpec.for_read_capture(
                     f, window_seconds=_c.window_seconds, pcap_label_dir=args.cic2018_csv_dir,
                     rows_per_file=args.rows_per_file, stride=args.stride,
                     pcap_max_windows=args.pcap_max_windows_per_day,
                     pcap_window_stride=args.pcap_window_stride) for f in files]
        for f, (_spec, cols) in zip(files, iter_capture_columns(
                specs, cache_dir=_cplan.cache_dir, scratch_dir=_cplan.scratch_dir,
                workers=args.ingest_workers)):
            yield f, None, cols, cols.coverage


    def _store_per_capture(files, label):
        """Load -> extract -> free, one capture file at a time.

        Holding the whole split resident costs ~15.7 GB at full density (30.4M
        records x ~518 B measured), which does not fit alongside the snapshot
        store and adapter arrays. Per-file keeps peak at the largest single
        capture (~4.7M records, ~2.4 GB).

        Processing per file also builds the TGNE neighbour graph per capture
        rather than across all of them. That is more faithful, not less: a
        CIC-2018 capture day and a CTU-13 botnet scenario are unrelated
        networks, and a merged graph would make hosts from different captures
        each other's temporal neighbours, which they never were.
        """
        from data_unification.trajectory_store import TrajectoryStoreBuilder, capture_namespace
        from data_unification.label_filter import (
            format_unresolved_report, merge_unresolved_reports)
        shared = TrajectoryStoreBuilder(spill_dir=str(args.spill_dir) if args.spill_dir else None)
        widx_base = 0
        total_recs = 0
        coverage: List[dict] = []
        if _cplan.enabled and _cplan.cache_dir is not None and args.extract_workers > 1:
            # Captures are independent (memory and host ids reset per capture),
            # so extract several at once and append them in capture order:
            # the same store -- see data_unification/parallel_extract.py.
            from data_unification.parallel_extract import extract_parallel
            caps = list(files)

            def _items():
                for i, (f, _r, cols, cov) in enumerate(_iter_capture_inputs(caps)):
                    coverage.append(cov)
                    yield i, cols, capture_namespace(f)
            items = _items()
            for i, info in extract_parallel(
                    items, extractor=extractor, builder_for=lambda _k: shared,
                    tgne=str(args.tgne) if args.tgne else None,
                    part_dir=Path(args.spill_dir) / f"parts_{label}",
                    workers=args.extract_workers):
                total_recs += info["n_records"]
                print(f"  [{label} {i+1}/{len(caps)}] {caps[i].label}: {info['n_records']} recs, "
                      f"store={shared._n} snaps, extract {info['seconds']:.1f}s "
                      f"(in a worker), merge {info['merge_seconds']:.1f}s", flush=True)
        else:
          for i, (f, recs, cols, cov) in enumerate(_iter_capture_inputs(files)):
            t = time.time()
            n_recs = len(cols) if cols is not None else len(recs)
            total_recs += n_recs
            coverage.append(cov)
            # One trajectory per (host, capture): see TrajectoryStoreBuilder.
            # set_namespace. Without it, the fabricated CIC-2018 days merged
            # all 350 of their hosts across days, and 2018 rows preceded 2011
            # rows in merged CTU-13 hosts.
            shared.set_namespace(capture_namespace(f))
            if cols is not None:
                extractor.extract_trajectories_columns(cols, builder=shared,
                                                       window_idx_base=widx_base)
            else:
                extractor.extract_trajectories(recs, builder=shared, window_idx_base=widx_base)
            widx_base = shared.next_window_base()
            print(f"  [{label} {i+1}/{len(files)}] {f.label}: {n_recs} recs, "
                  f"store={shared._n} snaps, {time.time()-t:.1f}s", flush=True)
            del recs, cols
            gc.collect()
        store = shared.finalize()
        print(format_unresolved_report(merge_unresolved_reports(coverage), where=label), flush=True)
        print(f"{label}: {total_recs} records -> {store.n_snapshots} snapshots "
              f"over {len(store)} hosts | {store.memory_report()}", flush=True)
        # How much of each host's traffic its 12-D latent never saw, and how
        # many attack windows it saw NONE of the attack in (flooding evasion).
        from data_unification.multi_dataset_stream import format_neighbor_exposure
        neighbor_exposure[label] = extractor.neighbor_exposure_report(reset=True)
        print(format_neighbor_exposure(neighbor_exposure[label], where=label), flush=True)
        # The record count is returned, not just printed: the run summary below
        # used to reference `train_records`/`val_records`, which the
        # load -> extract -> free refactor had already deleted. Every
        # non-credible run therefore died with NameError instead of reporting
        # the credibility verdict it had just computed.
        record_counts[label] = total_recs
        # Return the STORE, not materialised samples.
        #
        # create_host_sequence_samples builds a [15, 27] float32 array per
        # sample -- 2,053 bytes each. At full corpus density Branch A produces
        # roughly 42M samples (measured snapshot ratios: 1.99 per record for
        # CIC-2018, 0.72 for CTU-13), which is **80.3 GiB**. It does not fit,
        # and thinning the data is not an option.
        #
        # LazyHostSequenceDataset keeps two int32 columns (~8 B/sample, 336 MB
        # at 42M) and gathers each window from the memmapped feature block on
        # __getitem__ -- which is what DataLoader workers are for. Verified to
        # produce identical samples.
        return store

    # tau defaults to the horizon the model is actually asked about, so a host
    # exactly one forecast horizon from an attack scores exp(-1) = 0.368.
    hazard_tau = (args.hazard_tau if args.hazard_tau is not None
                  else _c.forecast_steps * _c.window_seconds)
    #: The cut that makes a window "positive" for AUC / Brier / ECE / the
    #: operating point. For the severity target, any non-zero score means the
    #: window contains attack traffic. For the hazard target, `> 0` would mean
    #: "this host is attacked at some later point in this split" -- nearly
    #: constant, and it discards exactly the timing the hazard encodes. The
    #: meaningful event is "an attack within one forecast horizon", which is
    #: hazard >= exp(-tau/tau) = exp(-1). The epsilon makes the comparison
    #: inclusive of a host sitting exactly on the horizon.
    risk_positive_above = (0.0 if args.risk_target == "severity"
                           else math.exp(-1.0) - 1e-6)

    def _apply_risk_target(store, label):
        """Swap in the hazard target, on every split or on none.

        Applying it to some splits and not others would train against one
        distribution and score against another, and the metrics would look
        fine while meaning nothing -- so this is called from one place for all
        three stores rather than at each call site.
        """
        if args.risk_target != "hazard":
            return
        summary = store.use_hazard_target(hazard_tau)
        print(f"  [{label}] risk target -> hazard(tau={hazard_tau:.1f}s): "
              f"zero fraction {summary['zero_fraction_before']:.4f} -> "
              f"{summary['zero_fraction_after']:.4f} | {summary}", flush=True)

    import gc
    t0 = time.time()
    train_store = _store_per_capture(train_files, "train")
    _apply_risk_target(train_store, "train")
    _rep_train = {}
    train_ds = LazyHostSequenceDataset(train_store, seq_len=_c.history_steps, min_trajectory_len=1,
        min_history_steps=args.min_history_steps, report=_rep_train,
        max_gap_seconds=args.max_gap_seconds)
    print(f"  [train] " + _fmt_history(_rep_train), flush=True)
    print(f"train done in {time.time()-t0:.1f}s ({len(train_ds)} samples)", flush=True)
    t0 = time.time()
    val_store = _store_per_capture(val_files, "val")
    _apply_risk_target(val_store, "val")
    _rep_val = {}
    val_ds = LazyHostSequenceDataset(val_store, seq_len=_c.history_steps, min_trajectory_len=1,
        min_history_steps=args.min_history_steps, report=_rep_val,
        max_gap_seconds=args.max_gap_seconds)
    print(f"  [val] " + _fmt_history(_rep_val), flush=True)
    print(f"val done in {time.time()-t0:.1f}s ({len(val_ds)} samples)", flush=True)
    t0 = time.time()
    test_store = _store_per_capture(test_files, "test")
    _apply_risk_target(test_store, "test")
    _rep_test = {}
    test_ds = LazyHostSequenceDataset(test_store, seq_len=_c.history_steps, min_trajectory_len=1,
        min_history_steps=args.min_history_steps, report=_rep_test,
        max_gap_seconds=args.max_gap_seconds)
    print(f"  [test] " + _fmt_history(_rep_test), flush=True)
    if args.max_gap_seconds is not None:
        for _n, _d in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            _tot = len(_d) + _d.n_dropped_by_gap
            print(f"gap segmentation [{_n}] at {args.max_gap_seconds}s: kept {len(_d):,} of "
                  f"{_tot:,} samples, dropped {_d.n_dropped_by_gap:,} "
                  f"({100.0 * _d.n_dropped_by_gap / max(_tot, 1):.1f}%) whose target lay beyond a gap",
                  flush=True)
    print(f"test done in {time.time()-t0:.1f}s ({len(test_ds)} samples)", flush=True)

    # Pairing guard. `bce` binarises at risk > 0; under the hazard target that
    # question is "is this host ever attacked later", which is not what the
    # target encodes and is nearly constant on this corpus. Loud, not fatal --
    # it is a legitimate ablation, just not a sensible default.
    if args.risk_target == "hazard" and args.risk_objective == "bce":
        print("\nWARNING: --risk-target hazard with --risk-objective bce "
              "binarises exp(-dt/tau) at > 0, which discards the timing the "
              "hazard target exists to carry and leaves a near-constant label. "
              "--risk-objective soft_bce is the proper scoring rule for a "
              "continuous target in [0, 1].\n", flush=True)
    if args.risk_target == "severity" and args.risk_objective == "soft_bce":
        print("\nWARNING: --risk-objective soft_bce against the severity "
              "target regresses a bimodal variable (82.5% exactly 0, the rest "
              "0.50-0.96) with a mean-seeking loss -- the same failure mode as "
              "smooth_l1.\n", flush=True)

    # Gate the data before training on it.
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from credibility_check import evaluate_store, gate as _gate, report as _report
        _stats = evaluate_store(train_store, val_store)
        _report(_stats, _gate(_stats))
    except Exception as _e:
        print(f"credibility check skipped: {_e}", flush=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("The 2-second pipeline produced no train/validation samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Workers matter more than anything else in this loop: profiling the
    # full-density run showed 35.7% of wall clock inside the dataset/collate
    # path, all of it in the main process and therefore serialised with the
    # GPU. The dataset is lazy by design (336 MB of indices instead of an
    # 80.3 GiB materialised array), which trades memory for per-item work --
    # that trade only pays if the work is overlapped.
    _loader_kw = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"))
    if args.num_workers > 0:
        _loader_kw.update(persistent_workers=True, prefetch_factor=4)
    def _make_loader(ds, shuffle):
        """One loader policy for train, validation and test.

        The batched path builds each batch with one vectorised gather from a
        host-major copy of the feature block (data_unification/host_major.py)
        instead of 128 per-sample memmap gathers, each of which was ~13 random
        disk reads at full scale. Same batches, same order, same RNG draws as
        DataLoader(ds, shuffle=...): tests/test_branch_a_batched_loader.py.
        """
        if args.legacy_loader:
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **_loader_kw)
        if not hasattr(ds, "_flat"):
            _t = time.time()
            ds.enable_batched(spill_dir=str(args.spill_dir) if args.spill_dir else None)
            print(f"  batched loader ready in {time.time() - _t:.1f}s "
                  f"(host-major features: {ds._feats_hm is not None})", flush=True)
        if shuffle:
            return DataLoader(BatchedSequenceView(ds),
                              batch_sampler=PermutationBatchSampler(len(ds), args.batch_size),
                              collate_fn=collate_prebatched, **_loader_kw)
        return DataLoader(BatchedSequenceView(ds), batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_prebatched, **_loader_kw)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)

    # Per-class focal alpha, computed from the TRAINING split only.
    #
    # Before 2026-09-22 this head was constructed as
    # `MultiClassFocalLoss(gamma=2.0)` -- alpha=None -- so it did no class
    # balancing whatsoever, while its own docstring described per-class
    # weighting as the reason it existed. The geometric-mean-normalised
    # weighting that the TGNE encoder's loss already used
    # (bita/train.py::FocalLoss) had never been carried across to Branch A.
    #
    # Counted from the store's columns, not by iterating the dataset: 20.7M
    # targets, and a label count needs none of the feature windows.
    from branch_a_gnn_lstm.lstm_multitask import MultiClassFocalLoss
    _t_counts, _g_counts = target_label_counts(train_ds)
    _inv_tech = {v: k for k, v in TECH_TO_IDX.items()}
    _print_class_counts(_t_counts, _inv_tech, "technique")
    _print_class_counts(_g_counts, GRADATION_NAMES, "gradation")

    focal_alpha = None
    if not args.no_focal_alpha:
        focal_alpha = MultiClassFocalLoss.alpha_from_counts(
            _t_counts, len(TECHNIQUE_VOCAB))
        print(f"  focal alpha: "
              f"{ {_inv_tech.get(i, i): round(float(w), 3) for i, w in enumerate(focal_alpha) if _t_counts[i] > 0} }",
              flush=True)
        _pinned = int(((focal_alpha <= 0.2 + 1e-9) | (focal_alpha >= 5.0 - 1e-9))
                      .logical_and(torch.as_tensor(_t_counts > 0)).sum())
        _present = int((_t_counts > 0).sum())
        if _pinned > _present // 2:
            print(f"  WARNING: {_pinned} of {_present} present classes are "
                  f"pinned at a clip bound, so the loss does no balancing "
                  f"between them. Class frequencies span "
                  f"{_t_counts[_t_counts > 0].max() / max(_t_counts[_t_counts > 0].min(), 1):.0f}x.",
                  flush=True)

    grad_weights = None
    if args.gradation_class_weights:
        grad_weights = MultiClassFocalLoss.alpha_from_counts(_g_counts, 4)
        print(f"  gradation class weights: "
              f"{ {GRADATION_NAMES.get(i, i): round(float(w), 3) for i, w in enumerate(grad_weights)} }",
              flush=True)

    _arch = dict(MultiTaskLSTM.PAPER_ARCH if args.architecture == "paper"
                 else MultiTaskLSTM.LEGACY_ARCH)
    if args.focal_gamma is not None:
        _arch["focal_gamma"] = args.focal_gamma
    print(f"Branch A architecture: {args.architecture} {_arch}", flush=True)
    model = MultiTaskLSTM(
        input_dim=27,
        num_techniques=len(TECHNIQUE_VOCAB),
        num_gradations=4,
        risk_objective=args.risk_objective,
        gradation_class_weights=(grad_weights.to(device)
                                 if grad_weights is not None else None),
        **_arch,
    ).to(device)
    if focal_alpha is not None:
        model.tech_focal_loss.alpha = focal_alpha.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss = float("-inf")       # _selection_score is maximised
    _history: List[Dict[str, float]] = []
    best_metrics: Dict[str, float] = {}

    print(
        f"train_records={record_counts.get('train', 0)} "
        f"val_records={record_counts.get('val', 0)} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} device={device}"
    )
    _n_train_batches = len(train_loader)
    # One policy for every trainer (cyberworld_v4/training_guard.py): warmup,
    # clipping, non-finite steps skipped, step back to the best weights at
    # half the LR after --step-back-after flat epochs, stop at --patience.
    guard = TrainingGuard(
        "branch_a", [model], optimizer, mode="max",
        patience=args.patience, step_back_after=args.step_back_after,
        warmup_steps=(args.warmup_steps if args.warmup_steps is not None
                      else default_warmup_steps(_n_train_batches)),
        clip_norm=args.clip_norm or None,
        log=lambda m: print(m, flush=True))
    if args.eval_only:
        # Re-score an existing checkpoint without retraining.
        #
        # The full-density Branch A run of 2026-09-21 completed before
        # _evaluate reported macro F1, the majority-class baseline or a
        # per-class breakdown, so its checkpoint carries only aggregate
        # accuracy -- which on an 82.5%-Benign corpus cannot distinguish a
        # working technique head from a collapsed one. Extraction is the
        # expensive part (~37 min) and is identical either way, so re-scoring
        # costs a fraction of a retrain.
        _src = args.eval_only if str(args.eval_only) != "-" else args.output
        print(f"EVAL-ONLY: scoring {_src} (no training)", flush=True)
        _ck = torch.load(_src, map_location=device, weights_only=False)
        # The checkpoint's own architecture, not the one these flags describe.
        model = MultiTaskLSTM.from_checkpoint(_ck, device=device)
        print(f"checkpoint epoch={_ck.get('epoch')} "
              f"recorded_metrics={_ck.get('metrics')}", flush=True)

        # Score BOTH splits. The 2026-09-21 run selected on validation loss and
        # then scored 0.6601 on validation against 26.36 on the held-out test --
        # a 40x gap. Reporting only one split cannot show whether that is a
        # collapsed head, a distribution shift between capture days, or a
        # checkpoint picked on an outlier epoch, so both are printed side by
        # side with the same metric set.
        _tl = _make_loader(test_ds, shuffle=False)
        _fits = None
        for _name, _ldr in (("validation", val_loader), ("held-out test", _tl)):
            # Logits are collected on validation only: that is the split a
            # temperature may honestly be fitted on, and collecting them on
            # test would invite exactly the mistake.
            _m = _evaluate(model, _ldr, device, num_techniques=len(TECHNIQUE_VOCAB),
                           risk_positive_above=risk_positive_above,
                           collect_logits=(_name == "validation"
                                           and not args.no_fit_temperature))
            _pc = _m.pop("tech_per_class", {})
            _gpc = _m.pop("gradation_per_class", {})
            print(f"\n{_name.upper()}: "
                  f"loss={_m['loss']:.4f} risk_mae={_m['risk_mae']:.4f} "
                  f"acc={_m['tech_accuracy']:.4f} "
                  f"macro_f1={_m['tech_macro_f1']:.4f} "
                  f"baseline={_m['tech_majority_baseline']:.4f} "
                  f"lift={_m['tech_lift_over_baseline']:+.4f} "
                  f"classes_pred={_m['tech_classes_predicted']}"
                  f"/{_m['tech_classes_present']} "
                  f"| gradation acc={_m['gradation_accuracy']} "
                  f"macro_f1={_m['gradation_macro_f1']:.4f} "
                  f"baseline={_m['gradation_majority_baseline']:.4f} "
                  f"lift={_m['gradation_lift_over_baseline']:+.4f} "
                  f"levels_pred={_m['gradation_classes_predicted']}"
                  f"/{_m['gradation_classes_present']} "
                  f"| risk auc={_m['risk_auc']:.4f} "
                  f"brier={_m['risk_brier']:.4f} ece={_m['risk_ece']:.4f}",
                  flush=True)
            _warn_if_head_collapsed(_m, _name)
            _warn_if_gradation_collapsed(_m, _name)
            _warn_if_risk_head_useless(_m, _name)
            _print_per_class(_pc, _name)
            _print_per_class(_gpc, _name, names=GRADATION_NAMES, label="gradation")
            if _name == "validation":
                # Fit the post-hoc parameters on validation and write them
                # back into the checkpoint that was just scored. This is the
                # whole reason --eval-only exists for a finished run: a
                # trained Branch A on disk can get an operating point without
                # paying for another retrain, and extraction (the expensive
                # part, ~37 min) is identical either way.
                _fits = calibrate_and_fit_operating_point(model, _m, args)
        if _fits is not None:
            _ck["operating_point"] = _fits["operating_point"]
            _ck["technique_calibration"] = _fits.get("temperature")
            _ck["risk_conformal"] = _fits.get("risk_conformal")
            _ck["model_state_dict"] = model.state_dict()
            _tc = dict(_ck.get("training_contract") or {})
            # Never re-stamp an objective onto weights that were trained with
            # a different one.
            #
            # `--eval-only` does not train, so `args.risk_objective` describes
            # THIS invocation, not the run that produced the checkpoint. The
            # checkpoint on disk from 2026-09-21 predates the flag entirely
            # and was trained with smooth_l1, so writing the current default
            # (bce) into its contract would tell serving that its severity
            # magnitude is a probability -- and serving bands alerts on
            # exactly that distinction. An existing value wins; an absent one
            # is recorded with a warning that it came from the command line.
            _existing = _tc.get("risk_objective")
            if _existing and _existing != args.risk_objective:
                print(f"\nNOTE: {_src} records risk_objective="
                      f"{_existing!r}; --risk-objective {args.risk_objective!r} "
                      f"describes this scoring run only and is NOT written "
                      f"into the contract. The stored value is what the "
                      f"weights were trained with.", flush=True)
            elif not _existing:
                print(f"\nWARNING: {_src} carries no risk_objective, so it "
                      f"predates the flag. Recording {args.risk_objective!r} "
                      f"from the command line -- if these weights were NOT "
                      f"trained that way, re-run --eval-only with the right "
                      f"--risk-objective, because serving bands alerts on "
                      f"this field.", flush=True)
                _tc["risk_objective"] = args.risk_objective
            _tc.setdefault("risk_target", args.risk_target)
            _ck["training_contract"] = _tc
            torch.save(_ck, _src)
            print(f"\nwrote fitted operating point / calibration back to {_src}",
                  flush=True)
        return

    # Crash recovery (cyberworld_v4/training_guard.ResumePoint). Extraction
    # re-runs on a restart; the finished epochs do not.
    resume = ResumePoint(args.output.with_name(args.output.stem + "_resume.pt"),
                         run_fingerprint(args, ignore=("epochs", "num_workers", "capture_cache", "ingest_workers", "extract_workers", "legacy_loader", "no_cuda_graph")),
                         enabled=not args.no_resume, log=lambda m: print(m, flush=True))
    first_epoch = 1
    _rp = resume.load()
    if _rp is not None:
        model.load_state_dict(_rp["model"])
        optimizer.load_state_dict(_rp["optimizer"])
        guard.load_state_dict(_rp["guard"])
        best_loss, best_metrics = _rp["best_loss"], _rp["best_metrics"]
        _history[:] = _rp["history"]
        first_epoch = _rp["done_epochs"] + 1
        if guard.should_stop():
            first_epoch = args.epochs + 1    # it had already stopped; just finish

    _graphed_loss = None
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        # Loss is accumulated as a GPU tensor and read once at the end of the
        # epoch. `float(loss.item())` per batch forces a host-device sync on
        # every one of ~161k batches, which serialises the CPU against the GPU
        # and defeats the prefetching the workers above are there to provide.
        # The reported value is unchanged: still the unweighted mean over
        # batches.
        _nb = 0
        _loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        # Steps taken, counted on the device: with the deferred guard step the
        # verdict of a step is a device bool, read by the guard one step later.
        _nb_dev = torch.zeros((), device=device, dtype=torch.long)
        _k = 0
        # "no step taken yet this epoch" (the old `_nb == 0`). Until the first
        # successful step the guard steps synchronously and returns a Python
        # bool, so this is known on the host exactly when the old test was.
        _stepped = False
        _t_epoch = time.time()
        _non_blocking = (device == "cuda")
        # No host-device sync per step (TrainingGuard.backward_step_deferred):
        # the step is taken on the device and undone there if the loss or the
        # gradient is not finite. Same decisions, counters and weights bit for
        # bit (tests/test_training_guard_one_sync.py); CYBERWORLD_GUARD_SYNC=1
        # restores the syncing step.
        _guard_step = (guard.backward_step_deferred if guard.deferred_supported()
                       else guard.backward_step)
        # forward + loss + backward replayed as a CUDA graph for full batches
        # (built once, on the first one); identical losses and weights.
        if _graphed_loss is None:
            def _train_loss(x_, t_, risk_, tech_, grad_):
                p_ = model(x_, t_history=t_)
                return model.compute_loss(
                    p_, {"risk": risk_, "technique": tech_, "gradation": grad_})[0]
            _graphed_loss = GraphedLoss(
                _train_loss, [model],
                enabled=(device == "cuda" and not args.no_cuda_graph and graphs_enabled()))
        for batch in train_loader:
            _k += 1
            x = batch["features"].to(device, non_blocking=_non_blocking)
            targets = {
                "risk": batch["risk"].to(device, non_blocking=_non_blocking),
                "technique": batch["technique"].to(device, non_blocking=_non_blocking),
                "gradation": batch["gradation"].to(device, non_blocking=_non_blocking),
            }
            # Real elapsed time between history steps. The model was blind to
            # it: none of the 15 temporal attributes spans windows, so fifteen
            # steps 2 s apart and fifteen spread over hours looked identical.
            t_hist = (batch["t_history"].to(device, non_blocking=_non_blocking)
                      if "t_history" in batch else None)
            if not _stepped:
                # Once per epoch, before the step: do the tasks fight over the
                # shared LSTM? Measured instead of assumed (see
                # MultiTaskLSTM.task_gradient_conflict); writes no .grad.
                _conflict = model.task_gradient_conflict(x, targets, t_history=t_hist)
                _c3 = _conflict["cosine"]
                print(f"  task gradients epoch={epoch}: cos(risk,tech)={_c3['risk_vs_tech']:+.3f} "
                      f"cos(risk,grad)={_c3['risk_vs_grad']:+.3f} "
                      f"cos(tech,grad)={_c3['tech_vs_grad']:+.3f} "
                      f"dominant={_conflict['dominant_task']}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if t_hist is not None:
                loss = _graphed_loss(x, t_hist, targets["risk"], targets["technique"],
                                     targets["gradation"])
            else:
                predictions = model(x, t_history=t_hist)
                loss, _ = model.compute_loss(predictions, targets)
            # backward + clip + step. A non-finite loss is skipped and counted,
            # never stepped on; the guard turns a run of them into a step back.
            _ok = _guard_step(loss)
            if _ok is False:
                continue
            # Keep the log-variances in range in the saved weights too: a step
            # can leave one epsilon outside the bound, and that is the value a
            # checkpoint written this epoch would record. (After a step the
            # device undid, the weights are the previous, already-projected
            # ones, and the clamp leaves them unchanged.)
            model.uncertainty_loss.project_()
            _stepped = True
            if _ok is True:
                _loss_sum += loss.detach().double().sum()
                _nb_dev += 1
            else:
                # adds exactly 0.0 / 0 for a skipped step, as `continue` did
                _loss_sum += torch.where(_ok, loss.detach().double().sum(),
                                         torch.zeros((), device=device, dtype=torch.float64))
                _nb_dev += _ok.to(torch.long)
            if args.log_every and _k % args.log_every == 0:
                _el = time.time() - _t_epoch
                _rate = _k / max(_el, 1e-9)
                _eta = (_n_train_batches - _k) / max(_rate, 1e-9)
                print(f"  epoch={epoch} batch={_k}/{_n_train_batches} "
                      f"({100.0 * _k / max(_n_train_batches, 1):.1f}%) "
                      f"{_rate:.1f} batch/s elapsed={_el / 60:.1f}m "
                      f"eta={_eta / 60:.1f}m", flush=True)
        guard.flush()
        _nb = int(_nb_dev)
        if _graphed_loss is not None and _graphed_loss.enabled:
            print(f"  cuda graph: {_graphed_loss.n_graphed} graphed steps, "
                  f"{_graphed_loss.n_eager} eager (other batch shapes)", flush=True)
            _graphed_loss.n_graphed = _graphed_loss.n_eager = 0

        metrics = _evaluate(model, val_loader, device,
                            num_techniques=len(TECHNIQUE_VOCAB),
                            risk_positive_above=risk_positive_above)
        metrics["epoch"] = epoch
        metrics["task_gradients"] = _conflict if _nb else None
        metrics["train_loss"] = float((_loss_sum / max(_nb, 1)).item())
        metrics["epoch_seconds"] = float(time.time() - _t_epoch)
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} risk_mae={metrics['risk_mae']:.4f} "
            f"tech_accuracy={metrics['tech_accuracy']:.3f} "
            f"tech_macro_f1={metrics['tech_macro_f1']:.3f} "
            f"lift_acc={metrics['tech_lift_over_baseline']:+.3f} "
            f"lift_f1={metrics['tech_macro_f1_lift']:+.3f} "
            f"classes_pred={metrics['tech_classes_predicted']}/{metrics['tech_classes_present']} "
            f"sel[{args.select_on}]={_selection_score(metrics, args.select_on):.4f} "
            f"| task_loss risk={metrics['loss_risk']:.4f} tech={metrics['loss_tech']:.4f} "
            f"grad={metrics['loss_grad']:.4f} "
            f"| weight risk={metrics['weight_risk']:.2f} tech={metrics['weight_tech']:.2f} "
            f"grad={metrics['weight_grad']:.2f} "
            f"| risk auc={metrics['risk_auc']:.4f} brier={metrics['risk_brier']:.4f} "
            f"(base {metrics['risk_brier_baseline']:.4f}) ece={metrics['risk_ece']:.4f} "
            f"| gradation acc={metrics['gradation_accuracy']} "
            f"macro_f1={metrics['gradation_macro_f1']:.3f} "
            f"lift={metrics['gradation_lift_over_baseline']:+.3f} "
            f"levels_pred={metrics['gradation_classes_predicted']}"
            f"/{metrics['gradation_classes_present']} "
            f"wall={metrics['epoch_seconds'] / 60:.1f}m"
        )
        _warn_if_head_collapsed(metrics, f"epoch {epoch}")
        _warn_if_gradation_collapsed(metrics, f"epoch {epoch}")
        _warn_if_risk_head_useless(metrics, f"epoch {epoch}")
        # Keep every epoch's validation metrics. Only the best-scoring weights
        # are written, so without this the other epochs are unrecoverable and
        # a selection decision cannot be revisited without a full retrain.
        _history.append(slim(metrics, drop_per_class=True))
        _score = _selection_score(metrics, args.select_on)
        metrics["selection_score"] = _score
        metrics["selection_metric"] = args.select_on
        _action = guard.end_epoch(_score, train_loss=metrics["train_loss"],
                                  health=_branch_a_health(metrics))
        metrics["guard_action"] = _action
        _history[-1]["guard_action"] = _action
        if _action == IMPROVED:
            best_loss = _score
            # `slim` strips the score histograms and the collected logits.
            # They are working data for the post-hoc fits, not results, and a
            # checkpoint that carried them per epoch would be gigabytes.
            best_metrics = slim(metrics)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    # Architecture kwargs, so serving rebuilds this exact model.
                    "arch": model.arch_config(),
                    # Which technique classes training contained, so any later
                    # evaluation on another dataset can report unseen classes
                    # separately (cyberworld_v4/cross_dataset.py).
                    "train_technique_counts": [int(x) for x in _t_counts],
                    "epoch": epoch,
                    "metrics": metrics,
                    "epoch_history": list(_history),
                    # Written from the contract actually in force, not from
                    # literals. These were hardcoded to 2.0/5 regardless of what
                    # the run used, so the metadata could describe a model that
                    # was never trained.
                    "training_contract": {
                        "window_seconds": _c.window_seconds,
                        "window_size_sec": _c.window_seconds,  # v3 key, kept readable
                        "history_steps": _c.history_steps,
                        "forecast_steps": _c.forecast_steps,
                        "feature_dim": _cfg.state_dim,
                        "sources": [str(x) for x in (args.cic_dir, args.ctu_dir, args.cic2017_dir, args.pcap_root) if x],
                        "split_scheme": args.split_scheme,
                        # What `risk_score` MEANS. Serving reads this to decide
                        # whether the number it is thresholding is a severity
                        # magnitude or a probability -- and to say so loudly
                        # when it is a probability with no fitted threshold.
                        # It was not written before, so the adapter could only
                        # guess, and its guess was the old severity scale.
                        "risk_objective": args.risk_objective,
                        "risk_target": args.risk_target,
                        "hazard_tau_seconds": (hazard_tau
                                               if args.risk_target == "hazard"
                                               else None),
                        "focal_gamma": model.tech_focal_loss.gamma,
                    },
                    "focal_alpha": (focal_alpha.detach().cpu().tolist()
                                    if focal_alpha is not None else None),
                    "train_class_counts": {
                        "technique": {_inv_tech.get(i, i): int(c)
                                      for i, c in enumerate(_t_counts)},
                        "gradation": {GRADATION_NAMES.get(i, i): int(c)
                                      for i, c in enumerate(_g_counts)},
                    },
                    "config": _cfg.to_dict(),
                    "manifest": _manifest.to_dict(),
                    "fingerprint": _manifest.fingerprint(),
                },
                args.output,
            )
        resume.save(epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    guard=guard.state_dict(), best_loss=best_loss,
                    best_metrics=best_metrics, history=list(_history))
        if _action == STOP:
            # The best weights are already written, so stopping here cannot
            # cost quality -- it only stops paying for epochs that do nothing.
            print(f"early stop at epoch {epoch}: {guard.stop_reason} (best epoch "
                  f"{best_metrics.get('epoch')}, {args.select_on}={best_loss:.4f})",
                  flush=True)
            break

    # Held-out test: scored once, on the restored best checkpoint, after
    # training is finished. This is the only number that is a generalisation
    _flag_outlier_selection(_history, best_metrics)

    # estimate rather than a selection artefact.
    ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Post-hoc calibration, and the served operating point.
    #
    # Order matters and is the point of doing it here: the weights are the
    # FINAL selected ones, restored above, and nothing after this line trains.
    # A temperature fitted before or during training is not temperature
    # scaling; the previous implementation trained one jointly with the loss
    # and then dropped it at inference.
    #
    # Validation is re-scored rather than reusing the best epoch's metrics,
    # because those came from whatever the weights were at the end of that
    # epoch and the technique logits were not kept. One extra pass over
    # validation buys the fits for all three post-hoc parameters.
    print("\nfitting post-hoc calibration and the operating point on "
          "validation, model frozen", flush=True)
    _val_metrics = _evaluate(model, val_loader, device,
                             num_techniques=len(TECHNIQUE_VOCAB),
                             risk_positive_above=risk_positive_above,
                             collect_logits=not args.no_fit_temperature)
    fits = calibrate_and_fit_operating_point(model, _val_metrics, args)
    # The temperature and the conformal width are buffers, so the state_dict
    # written below has to be re-taken AFTER the fit -- otherwise serving
    # loads a calibrated checkpoint whose weights say it was never calibrated.
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["operating_point"] = fits["operating_point"]
    ckpt["technique_calibration"] = fits.get("temperature")
    ckpt["risk_conformal"] = fits.get("risk_conformal")
    ckpt["validation_metrics_at_fit"] = slim(_val_metrics)

    test_loader = _make_loader(test_ds, shuffle=False)
    test_metrics = _evaluate(model, test_loader, device,
                             num_techniques=len(TECHNIQUE_VOCAB),
                             risk_positive_above=risk_positive_above)
    _per_class = test_metrics.pop("tech_per_class", {})
    _grad_per_class = test_metrics.pop("gradation_per_class", {})
    # Seen vs unseen classes. Under cross_year, CIC-2017 has PortScan and
    # Heartbleed and CIC-2018 has neither; averaging them in as ordinary misses
    # would hide that the model cannot name what it never saw.
    from cyberworld_v4.cross_dataset import format_unseen_report, unseen_class_report
    _unseen = unseen_class_report(
        _t_counts, test_metrics["tech_confusion"],
        {v: k for k, v in TECH_TO_IDX.items()})
    print(format_unseen_report(_unseen, "held-out test"), flush=True)
    # `slim` keeps the histograms out of the printed line and out of the
    # checkpoint; they are 2 x 2000 arrays of working data.
    print(f"HELD-OUT TEST (best epoch {ckpt.get('epoch')}): "
          f"{slim(test_metrics)}", flush=True)
    _warn_if_head_collapsed(test_metrics, "held-out test")
    _warn_if_gradation_collapsed(test_metrics, "held-out test")
    _warn_if_risk_head_useless(test_metrics, "held-out test")
    _print_per_class(_per_class, "held-out test")
    _print_per_class(_grad_per_class, "held-out test", names=GRADATION_NAMES,
                     label="gradation")
    # What the fitted threshold actually delivers on data it was not chosen
    # on. The operating point is fitted on validation and must never be
    # re-fitted here -- that would make it a test-set artefact -- but scoring
    # the SAME threshold on test is the only honest statement of what an
    # operator will see.
    _op = fits["operating_point"]
    if _op.get("fitted"):
        _tcurve = _pr_curve_from_histograms(test_metrics["risk_pos_hist"],
                                            test_metrics["risk_neg_hist"])
        _tb = int(round(_op["alert_threshold"] * (RISK_BINS - 1)))
        _op["held_out_test"] = _point(_tcurve, _tb)
        _t = _op["held_out_test"]
        print(f"  operating point on HELD-OUT TEST (threshold "
              f"{_op['alert_threshold']:.4f}, fitted on validation, not "
              f"re-fitted): precision={_t['precision']:.4f} "
              f"recall={_t['recall']:.4f} f1={_t['f1']:.4f} "
              f"alert_rate={_t['alert_rate']:.4f} "
              f"(test base rate {_tcurve['base_rate']:.4f})", flush=True)
    test_metrics = slim(test_metrics)
    test_metrics["tech_per_class"] = _per_class
    test_metrics["gradation_per_class"] = _grad_per_class
    test_metrics["unseen_class_report"] = _unseen
    test_metrics["split_scheme"] = args.split_scheme
    # The credibility verdict travels WITH the checkpoint.
    #
    # It used to be computed, printed to stdout, and thrown away. The
    # checkpoint kept `test_metrics` but nothing recording that those numbers
    # might be meaningless, so a non-credible model could be loaded and served
    # with no trace -- and its metrics quoted as results. That is the same
    # "looks fine, means nothing" failure mode as a collapsed head scoring
    # 0.83 CatAcc, or a class with seven training samples.
    credibility = {"checked": False}
    try:
        from credibility_check import evaluate_store as _es, gate as _g, report as _r
        _ts = _es(train_store, test_store)
        _problems = _g(_ts, model_accuracy=test_metrics.get("tech_accuracy"))
        _r(_ts, _problems)
        credibility = {
            "checked": True,
            "credible": not _problems,
            "problems": list(_problems),
            "stats": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                      for k, v in _ts.items()},
        }
    except Exception as _e:
        credibility = {"checked": False, "error": str(_e)}
        print(f"test credibility check skipped: {_e}", flush=True)

    ckpt["test_metrics"] = test_metrics
    ckpt["credibility"] = credibility
    ckpt["split_scheme"] = args.split_scheme
    ckpt["encoder"] = str(args.tgne) if args.tgne else "served default"
    # Per split: how much traffic the 12-D latent never saw (neighbour cut-off).
    ckpt["neighbor_exposure"] = neighbor_exposure
    # Every epoch's score, LR, gradient norms, skipped steps and guard action.
    ckpt["training_guard"] = guard.summary()
    torch.save(ckpt, args.output)
    if args.results_json:
        import json as _json
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(_json.dumps({
            "split_scheme": args.split_scheme,
            "encoder": ckpt["encoder"],
            "branch_a_checkpoint": str(args.output),
            "best_epoch": ckpt.get("epoch"),
            "validation": {k: v for k, v in slim(best_metrics, True).items()
                           if isinstance(v, (int, float, str))},
            "test": {k: v for k, v in test_metrics.items()
                     if isinstance(v, (int, float, str, dict))},
            "operating_point": ckpt.get("operating_point"),
            "credibility": credibility,
        }, indent=2, default=str))
        print(f"results written to {args.results_json}", flush=True)
    if credibility.get("checked") and not credibility.get("credible"):
        print("WARNING: checkpoint saved but marked NOT CREDIBLE -- "
              "its metrics must not be reported as results.", flush=True)

    print(f"saved={args.output} best_metrics={slim(best_metrics, True)} "
          f"test_metrics={slim(test_metrics, True)}")
    resume.clear()


if __name__ == "__main__":
    main()