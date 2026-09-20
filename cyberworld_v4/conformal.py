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

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Empirical coverage vs target. Report both; never claim the target alone."""
        lo, hi = self.interval(y_pred)
        y = np.asarray(y_true, dtype=float)
        covered = (y >= lo) & (y <= hi)
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "median_width": float(np.median(hi - lo)),
            "quantile": self.quantile_,
            "n_calibration": self.n_calibration_,
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
