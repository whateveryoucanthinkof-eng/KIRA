"""Zero- and low-parameter baselines. The floor every model must clear."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class PersistenceBaseline:
    """h_{t+k} = h_t. The baseline a latent world model must beat.

    Highly competitive at short horizons, which is exactly why it is the honest
    comparison for a 10-second forecast.
    """

    name = "persistence"

    def predict_states(self, history: np.ndarray, K: int) -> np.ndarray:
        """history [N, L, D] -> [N, K, D], repeating the last observed state."""
        last = np.asarray(history)[:, -1, :]
        return np.repeat(last[:, None, :], K, axis=1)

    def predict_proba(self, current_attack: np.ndarray, K: int) -> np.ndarray:
        """Assume the current state persists over the horizon."""
        p = np.asarray(current_attack, dtype=float).ravel()
        return np.repeat(p[:, None], K, axis=1)


class LastLabelBaseline:
    """Predict the last observed technique for every future step."""

    name = "last_label"

    def predict(self, last_labels: np.ndarray, K: int) -> np.ndarray:
        l = np.asarray(last_labels)
        return np.repeat(l[:, None], K, axis=1)


class MajorityClassBaseline:
    """Always predict the training majority. Exposes accuracy as meaningless
    under imbalance — if the model's accuracy matches this, it learned nothing."""

    name = "majority_class"

    def __init__(self) -> None:
        self.majority_: int = 0
        self.rate_: float = 0.0

    def fit(self, y: np.ndarray) -> "MajorityClassBaseline":
        y = np.asarray(y).astype(int).ravel()
        self.rate_ = float(y.mean()) if y.size else 0.0
        self.majority_ = int(self.rate_ >= 0.5)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.rate_, dtype=float)

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.majority_, dtype=int)


class MarkovBaseline:
    """First-order Markov chain over discrete states (spec 39).

    For technique forecasting this is the baseline that matters: it captures
    "attacks follow attacks" without any learned representation.
    """

    name = "markov"

    def __init__(self, n_states: int, smoothing: float = 1.0) -> None:
        self.n_states = n_states
        self.smoothing = smoothing
        self.T_: Optional[np.ndarray] = None

    def fit(self, sequences: Sequence[Sequence[int]]) -> "MarkovBaseline":
        counts = np.full((self.n_states, self.n_states), self.smoothing, dtype=float)
        for seq in sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                if 0 <= a < self.n_states and 0 <= b < self.n_states:
                    counts[a, b] += 1.0
        self.T_ = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, current: np.ndarray, K: int) -> np.ndarray:
        """[N] current states -> [N, K, n_states] by powering the chain."""
        if self.T_ is None:
            raise RuntimeError("MarkovBaseline.fit must be called first")
        cur = np.asarray(current).astype(int).ravel()
        out = np.zeros((cur.size, K, self.n_states), dtype=float)
        dist = np.eye(self.n_states)[np.clip(cur, 0, self.n_states - 1)]
        for k in range(K):
            dist = dist @ self.T_
            out[:, k, :] = dist
        return out
