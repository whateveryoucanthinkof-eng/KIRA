"""Learned baselines on the SAME features as the full model (spec 39).

PS 26153 requires a logistic-regression comparison "trained on the same
features". That phrase is load-bearing: comparing a transformer on 27-D
temporal sequences against logistic regression on some weaker hand-built
representation would be rigged. These flatten the identical [L, D] tensor the
neural path consumes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def flatten_history(X: np.ndarray) -> np.ndarray:
    """[N, L, D] -> [N, L*D]. The same information the sequence model receives."""
    X = np.asarray(X, dtype=float)
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


class LogisticBaseline:
    """L2 logistic regression. The PS-mandated comparison."""

    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42, **kw: Any) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=seed,
                class_weight=kw.pop("class_weight", "balanced"),
                **kw,
            ),
        )
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        y = np.asarray(y).astype(int).ravel()
        if len(np.unique(y)) < 2:
            self.fitted_ = False
            self._const = float(y.mean()) if y.size else 0.0
            return self
        self.model.fit(flatten_history(X), y)
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = flatten_history(X)
        if not self.fitted_:
            return np.full(Xf.shape[0], getattr(self, "_const", 0.0), dtype=float)
        return self.model.predict_proba(Xf)[:, 1]


class GradientBoostingBaseline:
    """Gradient-boosted trees — the strong tabular baseline (spec 39 A0).

    Often the hardest thing to beat on flow features. If the full architecture
    cannot, that is the finding, and spec 62 says report it.
    """

    name = "gradient_boosting"

    def __init__(self, seed: int = 42, **kw: Any) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            random_state=seed,
            max_iter=kw.pop("max_iter", 200),
            learning_rate=kw.pop("learning_rate", 0.1),
            **kw,
        )
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBaseline":
        y = np.asarray(y).astype(int).ravel()
        if len(np.unique(y)) < 2:
            self.fitted_ = False
            self._const = float(y.mean()) if y.size else 0.0
            return self
        self.model.fit(flatten_history(X), y)
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = flatten_history(X)
        if not self.fitted_:
            return np.full(Xf.shape[0], getattr(self, "_const", 0.0), dtype=float)
        return self.model.predict_proba(Xf)[:, 1]
