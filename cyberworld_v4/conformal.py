"""Conformal prediction (spec 34).

What v3 called conformal was a hardcoded interval:

    conformal_error: float = 0.05
    risk_lower = clamp(risk - 0.05); risk_upper = clamp(risk + 0.05)

and Branch B's rollout_with_uncertainty carried baked-in radii
[0.01890, 0.03661, 0.05327, 0.06904] whose steps 5+ came from an invented
0.015*sqrt(step-3) formula. A fixed number is not a coverage guarantee; nothing
in it responds to the data, and in the live adapter the radii were discarded
anyway.

Two procedures here, both real:

  SplitConformal   quantile of calibration nonconformity scores. Finite-sample
                   coverage under exchangeability. Honest and simple.

  AdaptiveConformal (ACI, Gibbs & Candes 2021)
                   alpha_{t+1} = alpha_t + gamma*(alpha - 1{Y_t not in C_t}).
                   Long-run coverage WITHOUT exchangeability, which matters
                   because network telemetry is neither i.i.d. nor stationary,
                   and because this is the one runtime adaptation the system can
                   perform with no labels: the realized outcome arrives
                   forecast_seconds later for free.

Security note (deliberate, not incidental): ACI widens whenever the model is
surprised. An adversary who shapes traffic shapes the residual stream, so a
patient attacker can widen the band until the real attack falls inside it. ACI
guarantees *coverage*, not detection. `min_width` exists for that reason: the
band may tighten below the offline baseline but never loosen past it without
being visible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1-alpha) quantile.

    Uses ceil((n+1)(1-alpha))/n, not the plain empirical quantile — that
    correction is what makes the coverage guarantee exact rather than
    approximate.
    """
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")  # too few calibration points for this alpha; say so
    return float(s[k - 1])


def halfwidth_from_histogram(hist, alpha: float = 0.05) -> Dict[str, Any]:
    """Split-conformal half-width from a histogram of |residuals| in [0, 1].

    For residuals too many to hold (a full-corpus validation split), binned on
    device as `floor(r * (bins - 1))`. Same order statistic as
    `conformal_quantile`: the ceil((n+1)(1-alpha))-th smallest residual. Binning
    rounds it UP to the bin's upper edge, so the interval over-covers by at
    most one bin -- the safe direction; rounding down would over-state
    coverage. `empirical_coverage` is what that rounded width achieves on the
    same residuals, reported rather than assumed.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    bins = h.size
    n = int(h.sum())
    if n == 0:
        return {"fitted": False, "n": 0, "reason": "no residuals"}
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return {"fitted": False, "n": n,
                "reason": f"{n} points cannot support {(1 - alpha) * 100:.1f}% coverage"}
    cum = np.cumsum(h)
    b = int(np.searchsorted(cum, k, side="left"))
    return {
        "fitted": True,
        "alpha": float(alpha),
        "target_coverage": 1.0 - float(alpha),
        "half_width": min(float(b + 1) / (bins - 1), 1.0),
        "empirical_coverage": float(cum[b] / n),
        "n": n,
        "bin_width": 1.0 / (bins - 1),
    }


@dataclass
class SplitConformal:
    """Split conformal intervals from calibration residuals.

    Fit on the CALIBRATION split only — never on train or validation.
    """

    alpha: float = 0.05
    quantile_: float = float("nan")
    n_calibration_: int = 0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformal":
        s = np.abs(np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel())
        self.quantile_ = conformal_quantile(s, self.alpha)
        self.n_calibration_ = int(s.size)
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(y_pred, dtype=float)
        return p - self.quantile_, p + self.quantile_

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Empirical coverage vs target. Report both; never claim the target alone.

        `groups` -- pass the label. Split conformal guarantees MARGINAL coverage
        only. On a target that is 0 for most windows the quantile is set by the
        benign majority, so 95% overall coverage can sit on top of far lower
        coverage for the attack windows -- the ones that matter. With `groups`
        the per-group coverage is reported instead of being averaged away.
        """
        lo, hi = self.interval(y_pred)
        y = np.asarray(y_true, dtype=float).ravel()
        covered = (y >= lo.ravel()) & (y <= hi.ravel())
        out = {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "median_width": float(np.median(hi - lo)),
            "quantile": self.quantile_,
            "n_calibration": self.n_calibration_,
            "n_test": int(y.size),
        }
        if groups is not None:
            out.update(coverage_by_group(covered, groups))
        return out


def coverage_by_group(covered: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """Coverage within each group, plus the worst group.

    Keys are str(group) so the result survives a JSON round trip unchanged.
    """
    covered = np.asarray(covered, dtype=bool).ravel()
    groups = np.asarray(groups).ravel()
    if groups.size != covered.size:
        raise ValueError(f"groups has {groups.size} entries but there are "
                         f"{covered.size} predictions")
    per: Dict[str, Dict[str, float]] = {}
    for g in np.unique(groups):
        m = groups == g
        key = str(g.item() if hasattr(g, "item") else g)
        per[key] = {"n": int(m.sum()), "empirical_coverage": float(covered[m].mean())}
    worst = min(per, key=lambda k: per[k]["empirical_coverage"]) if per else None
    return {
        "coverage_by_group": per,
        "worst_group": worst,
        "worst_group_coverage": per[worst]["empirical_coverage"] if worst is not None else float("nan"),
    }


@dataclass
class LabelConditionalConformal:
    """Label-conditional (Mondrian) conformal prediction sets for a classifier.

    Vovk, Gammerman & Shafer, "Algorithmic Learning in a Random World" (2005),
    Sec. 4.5; Sadinle, Lei & Wasserman, JASA 2019. One quantile per class,
    fitted only on the calibration points OF THAT CLASS:

        s(x, c) = 1 - p_c(x)
        q_c     = conformal_quantile({s(x_i, c) : y_i = c}, alpha)
        C(x)    = {c : s(x, c) <= q_c}

    Guarantee: P(y in C(x) | y = c) >= 1 - alpha for EVERY class c, under
    exchangeability within that class. Marginal split conformal gives no such
    thing for the minority class: on a ~17% attack base rate its 95% can be
    met almost entirely by benign windows. Here the attack class gets its own
    95%, paid for with larger (ambiguous) sets where the model cannot separate
    the classes -- `ambiguous_rate` says how often.

    A class with too few calibration points for `alpha` gets q_c = inf: it is
    in every set, because no guarantee can be given without data. That shows
    up in `quantiles_` and in `mean_set_size` rather than silently.

    Like every split method this assumes the calibration windows represent
    test. An attacker who shifts their traffic away from the calibration
    distribution breaks that assumption, which is why coverage is measured
    on test by `evaluate` and never taken from the target.
    """

    alpha: float = 0.05
    quantiles_: Dict[int, float] = field(default_factory=dict)
    n_calibration_: Dict[int, int] = field(default_factory=dict)

    @staticmethod
    def _as_proba(proba: np.ndarray) -> np.ndarray:
        p = np.asarray(proba, dtype=float)
        if p.ndim == 1:  # binary, given as P(class 1)
            p = np.stack([1.0 - p, p], axis=1)
        if p.ndim != 2:
            raise ValueError(f"expected [N] or [N, C] probabilities, got shape {p.shape}")
        return p

    def fit(self, y_true: np.ndarray, proba: np.ndarray) -> "LabelConditionalConformal":
        p = self._as_proba(proba)
        y = np.asarray(y_true).astype(int).ravel()
        if y.size != p.shape[0]:
            raise ValueError(f"{y.size} labels for {p.shape[0]} predictions")
        self.quantiles_, self.n_calibration_ = {}, {}
        for c in range(p.shape[1]):
            s = 1.0 - p[y == c, c]
            self.quantiles_[c] = conformal_quantile(s, self.alpha)
            self.n_calibration_[c] = int(s.size)
        return self

    def predict_sets(self, proba: np.ndarray) -> np.ndarray:
        """[N, C] boolean: is class c in the prediction set of sample n."""
        if not self.quantiles_:
            raise RuntimeError("fit() on the calibration split first")
        p = self._as_proba(proba)
        q = np.array([self.quantiles_.get(c, float("inf")) for c in range(p.shape[1])])
        return (1.0 - p) <= q[None, :]

    def evaluate(self, y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
        sets = self.predict_sets(proba)
        y = np.asarray(y_true).astype(int).ravel()
        if y.size != sets.shape[0]:
            raise ValueError(f"{y.size} labels for {sets.shape[0]} predictions")
        covered = sets[np.arange(y.size), y]
        by_class = coverage_by_group(covered, y)
        for c, rec in by_class["coverage_by_group"].items():
            rec["quantile"] = self.quantiles_.get(int(c), float("inf"))
            rec["n_calibration"] = self.n_calibration_.get(int(c), 0)
        size = sets.sum(axis=1)
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "coverage_by_class": by_class["coverage_by_group"],
            "worst_class": by_class["worst_group"],
            "worst_class_coverage": by_class["worst_group_coverage"],
            "mean_set_size": float(size.mean()),
            "empty_set_rate": float((size == 0).mean()),
            "ambiguous_rate": float((size > 1).mean()),
            "n_test": int(y.size),
        }


@dataclass
class AdaptiveConformal:
    """Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021).

    Online, label-free: it needs only the realized outcome, which this system
    produces for free `forecast_seconds` after each prediction.

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    `min_width` floors the interval at the offline conformal width so online
    adaptation can tighten but not silently loosen past the accredited baseline.
    """

    alpha: float = 0.05
    gamma: float = 0.01
    min_width: float = 0.0
    max_width: float = float("inf")
    window: int = 500

    alpha_t: float = field(init=False)
    scores_: Deque[float] = field(init=False)
    history_: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha
        self.scores_ = deque(maxlen=self.window)

    def warm_start(self, residuals: Sequence[float]) -> "AdaptiveConformal":
        """Seed from offline calibration residuals so step 1 is not a guess."""
        for r in residuals:
            self.scores_.append(float(abs(r)))
        return self

    def _half_width(self) -> float:
        if not self.scores_:
            return self.min_width
        q = conformal_quantile(np.asarray(self.scores_), np.clip(self.alpha_t, 1e-4, 0.999))
        if not np.isfinite(q):
            q = float(max(self.scores_))
        return float(np.clip(q, self.min_width, self.max_width))

    def predict_interval(self, y_pred: float) -> tuple[float, float]:
        w = self._half_width()
        return float(y_pred - w), float(y_pred + w)

    def update(self, y_true: float, y_pred: float) -> Dict[str, float]:
        """Observe the realized outcome and adapt. This is the whole method."""
        lo, hi = self.predict_interval(y_pred)
        err = 0.0 if (lo <= y_true <= hi) else 1.0

        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err), 1e-4, 0.999))
        self.scores_.append(float(abs(y_true - y_pred)))

        rec = {"alpha_t": self.alpha_t, "half_width": (hi - lo) / 2.0, "error": err}
        self.history_.append(rec)
        return rec

    def running_coverage(self, last: Optional[int] = None) -> float:
        h = self.history_[-last:] if last else self.history_
        return float(1.0 - np.mean([r["error"] for r in h])) if h else float("nan")

    def drift_signal(self, short: int = 50, long: int = 500) -> Dict[str, float]:
        """Sustained widening is itself a detection signal (and an attack surface).

        A monotone rise in half-width means the world model can no longer
        predict this host — which is information — but is also exactly what a
        patient adversary would induce. Alarm on it; do not only trust it.
        """
        if len(self.history_) < short:
            return {"ratio": float("nan"), "short": float("nan"), "long": float("nan")}
        s = float(np.mean([r["half_width"] for r in self.history_[-short:]]))
        l = float(np.mean([r["half_width"] for r in self.history_[-long:]]))
        return {"ratio": (s / l) if l > 0 else float("nan"), "short": s, "long": l}
