"""
cyberworld_v4.targets — future targets with explicit statistical semantics.

This module replaces the v3 target construction, whose defect was structural:

    target_snap = window_slice[-1]        # sequence_dataset.py:116

The label came from the last element *of the input window*, so the model was
trained to name what it had already seen. That is nowcasting. Reported accuracy
under it says nothing about early warning, which is the claim the project makes.

Four distinct quantities are produced here, and they are never conflated
(spec 7, 8, 9, 10):

  current_attack     A_t              nowcasting target, kept but labelled
  future_attack[k]   A_{t+k}          k = 1..K, strictly after the input
  hazard[k]          P(onset = t+k | no onset before, X<=t)
  severity           operator ranking aid — NOT a probability, never a BCE target

Cumulative probability follows from the hazard, and is not max():

    P(onset <= t+K) = 1 - prod_k (1 - h_k)

Techniques are multilabel (spec 13): v3 kept `technique_ids[0]` and discarded
the rest.

Trajectories are split on gaps (spec 20): a 102-minute gap between two
snapshots is not a 2-second transition, and feeding it to an autoregressive
model as one step is simply wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import CyberWorldConfig, DEFAULT_CONFIG
from .identity import HostId

# Sentinel for "no onset within the horizon".
NO_ONSET = -1


@dataclass
class ForecastSample:
    """One training example. Inputs end at t; every target is at t+1 or later."""

    host: HostId
    t_index: int
    t_end: float                    # wall-clock end of the last input window

    features: np.ndarray            # [history_steps, state_dim]

    # --- nowcasting (spec 38 Task A) — separate head, separate metric ---
    current_attack: int

    # --- forecasting (spec 38 Task B/C) — strictly future ---
    future_attack: np.ndarray       # [K] int
    future_techniques: np.ndarray   # [K, n_techniques] multilabel {0,1}
    future_states: np.ndarray       # [K, latent_dim] Branch B regression target

    # --- hazard (spec 39) ---
    onset_step: int                 # 1..K, or NO_ONSET
    hazard_target: np.ndarray       # [K] 1 at the onset step, else 0
    at_risk: np.ndarray             # [K] 1 while still at risk; masks the likelihood
    onset_censored: bool            # True when already under attack at t

    # --- operator aid, explicitly not a probability ---
    severity: float = 0.0

    def cumulative_probability(self, hazard: np.ndarray) -> float:
        """P(onset <= t+K) from predicted per-step hazards. Never max()."""
        h = np.clip(np.asarray(hazard, dtype=np.float64), 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - h))


def split_on_gaps(
    snapshots: Sequence[Any],
    max_gap_seconds: float,
    *,
    time_attr: str = "window_end",
) -> List[List[Any]]:
    """Split a host's snapshots into contiguous segments.

    Spec 20. Any gap wider than `max_gap_seconds` ends the segment: the two
    sides are not a single temporal transition and must not be modelled as one.
    """
    segments: List[List[Any]] = []
    current: List[Any] = []
    prev_t: Optional[float] = None

    for snap in snapshots:
        t = float(getattr(snap, time_attr))
        if prev_t is not None and (t - prev_t) > max_gap_seconds:
            if current:
                segments.append(current)
            current = []
        current.append(snap)
        prev_t = t

    if current:
        segments.append(current)
    return segments


def _multilabel(techniques: Iterable[str], vocab_index: Dict[str, int], n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for t in techniques or ():
        idx = vocab_index.get(t)
        if idx is not None:
            v[idx] = 1.0
    return v


def _hazard_targets(future_attack: np.ndarray, current_attack: int) -> Tuple[int, np.ndarray, np.ndarray, bool]:
    """Discrete-time hazard targets for attack onset.

    Onset is a benign -> attack transition. A host already under attack at t is
    *censored* for this task: it cannot have an onset, and including it as a
    negative would teach the model that ongoing attacks are safe. Its at_risk
    mask is all zeros, so it contributes nothing to the hazard likelihood while
    remaining a valid example for the other heads.
    """
    K = len(future_attack)
    hazard = np.zeros(K, dtype=np.float32)
    at_risk = np.zeros(K, dtype=np.float32)

    if current_attack == 1:
        return NO_ONSET, hazard, at_risk, True

    onset = NO_ONSET
    for k in range(K):
        at_risk[k] = 1.0          # still event-free entering step k+1
        if future_attack[k] == 1:
            hazard[k] = 1.0
            onset = k + 1
            break                 # after the event the host leaves the risk set

    return onset, hazard, at_risk, False


def build_samples(
    snapshots: Sequence[Any],
    host: HostId,
    technique_vocab: Sequence[str],
    config: CyberWorldConfig = DEFAULT_CONFIG,
    *,
    stride: int = 1,
    require_full_history: bool = True,
) -> List[ForecastSample]:
    """Build forecast samples for one host.

    The input window is snapshots[i-L+1 .. i]; every target is drawn from
    snapshots[i+1 .. i+K]. There is no overlap, by construction — which is the
    property v3 lacked.
    """
    L = config.temporal.history_steps
    K = config.temporal.forecast_steps
    vocab_index = {t: i for i, t in enumerate(technique_vocab)}
    n_tech = len(technique_vocab)

    samples: List[ForecastSample] = []

    for segment in split_on_gaps(snapshots, config.max_gap_seconds):
        n = len(segment)
        if n < L + K:
            continue  # cannot form a full history AND a full horizon

        start = L - 1 if require_full_history else 0
        for i in range(start, n - K, stride):
            hist = segment[max(0, i - L + 1) : i + 1]
            if require_full_history and len(hist) < L:
                continue

            feats = np.stack(
                [np.concatenate([s.embedding, s.temporal_attrs]) for s in hist]
            ).astype(np.float32)
            if feats.shape[0] < L:  # left-pad only when explicitly allowed
                pad = np.zeros((L - feats.shape[0], feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([pad, feats], axis=0)

            future = segment[i + 1 : i + 1 + K]
            fut_attack = np.array([int(bool(s.is_attack)) for s in future], dtype=np.int64)
            fut_tech = np.stack(
                [_multilabel(s.technique_ids, vocab_index, n_tech) for s in future]
            )
            fut_states = np.stack(
                [np.asarray(s.embedding, dtype=np.float32) for s in future]
            )

            cur = int(bool(segment[i].is_attack))
            onset, hazard, at_risk, censored = _hazard_targets(fut_attack, cur)

            samples.append(
                ForecastSample(
                    host=host,
                    t_index=i,
                    t_end=float(segment[i].window_end),
                    features=feats,
                    current_attack=cur,
                    future_attack=fut_attack,
                    future_techniques=fut_tech,
                    future_states=fut_states,
                    onset_step=onset,
                    hazard_target=hazard,
                    at_risk=at_risk,
                    onset_censored=censored,
                    severity=float(getattr(segment[i], "risk_score", 0.0) or 0.0),
                )
            )

    return samples


def cumulative_from_hazard(hazard: np.ndarray) -> np.ndarray:
    """Cumulative onset probability at each horizon, from per-step hazards.

    Returns [K] where entry k is P(onset <= t+k+1) = 1 - prod_{j<=k}(1-h_j).
    """
    h = np.clip(np.asarray(hazard, dtype=np.float64), 0.0, 1.0)
    return (1.0 - np.cumprod(1.0 - h, axis=-1)).astype(np.float32)


def describe_targets(samples: Sequence[ForecastSample]) -> Dict[str, Any]:
    """Class balance and censoring summary. Report this beside any accuracy."""
    if not samples:
        return {"n": 0}
    K = len(samples[0].future_attack)
    fut = np.stack([s.future_attack for s in samples])
    return {
        "n": len(samples),
        "hosts": len({s.host.key for s in samples}),
        "current_attack_rate": float(np.mean([s.current_attack for s in samples])),
        "future_attack_rate_by_step": [float(fut[:, k].mean()) for k in range(K)],
        "onset_within_horizon": float(
            np.mean([s.onset_step != NO_ONSET for s in samples])
        ),
        "censored_already_attacking": float(np.mean([s.onset_censored for s in samples])),
    }
