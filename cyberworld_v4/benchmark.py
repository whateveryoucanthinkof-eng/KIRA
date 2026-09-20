"""Benchmark harness (spec 39, 40, 55; PS 26153 "Expected Solution").

Produces the table the problem statement requires — F1, precision, recall and
false-positive rate against a logistic-regression baseline trained on the same
features — plus the horizon curve, calibration evidence and group-bootstrap
intervals that make the numbers defensible.

Two rules enforced here rather than left to discipline:

  * The test split is touched once, after everything is frozen. Thresholds are
    chosen on VALIDATION (spec 66: never optimise a threshold against the test
    set), the temperature is fitted on CALIBRATION, and only then is test
    scored.

  * If a baseline wins, that is the result and it is reported (spec 62). A
    harness that can only confirm the preferred model is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CyberWorldConfig, DEFAULT_CONFIG
from .metrics.bootstrap import format_ci, group_bootstrap_ci, multi_seed_summary
from .metrics.calibration import TemperatureScaler, calibration_report
from .metrics.detection import detection_metrics
from .metrics.forecasting import horizon_metrics


@dataclass
class EvalSet:
    """One split, already featurised."""

    X: np.ndarray                  # [N, L, D]
    y_current: np.ndarray          # [N]
    y_future: np.ndarray           # [N, K]
    groups: np.ndarray             # [N] capture/host id — the bootstrap unit
    hours_observed: Optional[float] = None

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class ModelEntry:
    """Anything with fit/predict_proba. Baselines and the full model alike."""

    name: str
    fit: Callable[[np.ndarray, np.ndarray], Any]
    predict_proba: Callable[[np.ndarray], np.ndarray]
    is_baseline: bool = True


def choose_threshold(
    y_true: np.ndarray, y_score: np.ndarray, *, objective: str = "f1"
) -> float:
    """Pick a threshold on VALIDATION only.

    Choosing it on test inflates every downstream number and is the most common
    way an honest-looking benchmark becomes dishonest.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5

    best_t, best_v = 0.5, -np.inf
    for t in np.unique(np.round(np.quantile(y_score, np.linspace(0.01, 0.99, 99)), 6)):
        m = detection_metrics(y_true, y_score, threshold=float(t))

        # Reject degenerate thresholds. On a weak classifier with a high
        # positive rate, F1 is maximised by predicting everything positive:
        # recall 1.0, FPR 1.0, and an F1 that looks respectable while the
        # detector is useless. A threshold that never says "benign" is not an
        # operating point, so it is excluded rather than silently chosen.
        if m["tp"] + m["fp"] == 0 or m["tn"] + m["fn"] == 0:
            continue

        v = m["f1"] if objective == "f1" else m["balanced_accuracy"]
        if objective == "f1":
            # Break ties toward a usable false-positive rate.
            v -= 0.05 * m["fpr"]
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def run_benchmark(
    models: Sequence[ModelEntry],
    train: EvalSet,
    validation: EvalSet,
    calibration: EvalSet,
    test: EvalSet,
    *,
    config: CyberWorldConfig = DEFAULT_CONFIG,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit on train, tune on validation, calibrate on calibration, score test once."""
    results: Dict[str, Any] = {
        "contract": config.temporal.to_dict(),
        "split_sizes": {
            "train": len(train), "validation": len(validation),
            "calibration": len(calibration), "test": len(test),
        },
        "positive_rate": {
            "train": float(train.y_current.mean()),
            "test": float(test.y_current.mean()),
        },
        "models": {},
    }

    for entry in models:
        entry.fit(train.X, train.y_current)

        # Order matters. Calibrate FIRST, then pick the threshold on calibrated
        # validation scores, so the threshold lives on the same scale it will be
        # applied to at test time. Choosing it on raw probabilities and applying
        # it to temperature-scaled ones compares two different scales and can
        # silently produce an all-positive operating point.
        cal_p = np.asarray(entry.predict_proba(calibration.X), dtype=float)
        scaler = TemperatureScaler().fit_from_probs(cal_p, calibration.y_current)

        val_p = scaler.transform_probs(
            np.asarray(entry.predict_proba(validation.X), dtype=float)
        )
        threshold = choose_threshold(validation.y_current, val_p)

        # --- test touched here, once ---
        test_p = scaler.transform_probs(
            np.asarray(entry.predict_proba(test.X), dtype=float)
        )

        det = detection_metrics(
            test.y_current, test_p, threshold=threshold, hours_observed=test.hours_observed
        )
        if det["tp"] + det["fp"] == 0 or det["tn"] + det["fn"] == 0:
            det["degenerate"] = True
            det["note"] = (
                "threshold predicts a single class on test; treat precision/recall/F1 "
                "as uninformative and read PR-AUC instead"
            )

        idx = np.arange(len(test))
        def ap(sub: Sequence[Any]) -> float:
            i = np.asarray([s for s in sub], dtype=int)
            yy = test.y_current[i]
            if len(np.unique(yy)) < 2:
                return float("nan")
            from sklearn.metrics import average_precision_score
            return float(average_precision_score(yy, test_p[i]))

        ci = group_bootstrap_ci(
            list(idx), lambda i: str(test.groups[i]), ap, n_resamples=n_resamples, seed=seed
        )

        results["models"][entry.name] = {
            "is_baseline": entry.is_baseline,
            "threshold_from_validation": threshold,
            "detection": det,
            "pr_auc_ci": ci,
            "calibration": {
                "temperature": scaler.temperature,
                "before": scaler.report()["before"],
                "after": scaler.report()["after"],
            },
        }

    results["comparison"] = _compare(results["models"])
    return results


def _compare(models: Dict[str, Any]) -> Dict[str, Any]:
    """State plainly whether the full model beat the baselines."""
    base = {k: v for k, v in models.items() if v["is_baseline"]}
    full = {k: v for k, v in models.items() if not v["is_baseline"]}
    if not base or not full:
        return {"verdict": "insufficient entries to compare"}

    best_b = max(base.items(), key=lambda kv: kv[1]["detection"].get("pr_auc", -1))
    best_f = max(full.items(), key=lambda kv: kv[1]["detection"].get("pr_auc", -1))
    bp = best_b[1]["detection"].get("pr_auc", float("nan"))
    fp = best_f[1]["detection"].get("pr_auc", float("nan"))
    lo = best_f[1]["pr_auc_ci"].get("ci_low", float("nan"))

    beaten = bool(np.isfinite(fp) and np.isfinite(bp) and fp > bp)
    decisive = bool(np.isfinite(lo) and np.isfinite(bp) and lo > bp)
    return {
        "best_baseline": best_b[0],
        "best_baseline_pr_auc": bp,
        "best_model": best_f[0],
        "best_model_pr_auc": fp,
        "delta": (fp - bp) if np.isfinite(fp) and np.isfinite(bp) else float("nan"),
        "model_beats_baseline": beaten,
        # The honest bar: the model's interval must clear the baseline point,
        # not merely its point estimate.
        "beats_baseline_outside_ci": decisive,
        "verdict": (
            "model beats best baseline, CI excludes it" if decisive
            else "model ahead but within CI — not yet decisive" if beaten
            else "baseline wins; report this (spec 62)"
        ),
    }


def markdown_table(results: Dict[str, Any]) -> str:
    """The PS deliverable table."""
    rows = [
        "| Model | PR-AUC [95% CI] | ROC-AUC | Precision | Recall | F1 | FPR | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in sorted(
        results["models"].items(), key=lambda kv: -(kv[1]["detection"].get("pr_auc") or -1)
    ):
        d = r["detection"]
        tag = "" if r["is_baseline"] else " **(full)**"
        rows.append(
            f"| {name}{tag} | {format_ci(r['pr_auc_ci'])} | {d.get('roc_auc', float('nan')):.3f} | "
            f"{d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['fpr']:.3f} | "
            f"{r['calibration']['after']['ece']:.4f} |"
        )
    c = results.get("comparison", {})
    rows += ["", f"**Verdict:** {c.get('verdict', 'n/a')}"]
    if "delta" in c and np.isfinite(c.get("delta", float("nan"))):
        rows.append(
            f"Best baseline `{c['best_baseline']}` PR-AUC {c['best_baseline_pr_auc']:.3f} · "
            f"best model `{c['best_model']}` {c['best_model_pr_auc']:.3f} · Δ {c['delta']:+.3f}"
        )
    return "\n".join(rows)


def save(results: Dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=float))
    return p
