"""Early-warning evaluation (spec 30, 31, 32).

v3 reported `forecast_steps * window_seconds` as "lead time". That is the
forecast *horizon* — a constant of the architecture, identical whether the
model works or not. It cannot be a result.

True lead time is measured against the event:

    L = t_onset - t_first_valid_alert

and is only defined for episodes the model actually caught. Missed episodes are
reported separately; averaging lead time over caught episodes while hiding the
miss rate is how a model that fires on 5% of attacks looks excellent.

A valid alert (spec 31) must precede onset, clear the threshold, and — with
`persistence` > 1 — do so on consecutive windows, which is what suppresses
single-window noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Episode:
    """One attack episode with its ground-truth onset."""

    episode_id: str
    onset_time: float
    times: np.ndarray    # alert-opportunity timestamps, ascending
    scores: np.ndarray   # model score at each


def first_valid_alert(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    persistence: int = 1,
    before: Optional[float] = None,
) -> Optional[float]:
    """Timestamp of the first alert clearing `threshold` for `persistence` windows."""
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if times.size == 0:
        return None

    order = np.argsort(times)
    times, scores = times[order], scores[order]
    if before is not None:
        m = times < before
        times, scores = times[m], scores[m]
        if times.size == 0:
            return None

    # Track the run's start POSITION. Resolving it by value with
    # list(times).index(t) returns the first index holding that timestamp, so a
    # duplicated timestamp credits an earlier window than the one that actually
    # began the qualifying run — inflating reported lead time, which is the
    # precise overstatement this module exists to prevent.
    run = 0
    run_start = 0
    for i, sc in enumerate(scores):
        if sc >= threshold:
            if run == 0:
                run_start = i
            run += 1
        else:
            run = 0
        if run >= persistence:
            return float(times[run_start])
    return None


def lead_time_report(
    episodes: Sequence[Episode],
    *,
    threshold: float,
    persistence: int = 1,
    hours_observed: Optional[float] = None,
    recall_at_seconds: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0),
) -> Dict[str, Any]:
    """Lead-time distribution plus the miss rate that gives it meaning."""
    leads: List[float] = []
    missed: List[str] = []

    for ep in episodes:
        t_alert = first_valid_alert(
            ep.times, ep.scores, threshold=threshold, persistence=persistence, before=ep.onset_time
        )
        if t_alert is None:
            missed.append(ep.episode_id)
        else:
            leads.append(float(ep.onset_time - t_alert))

    n = len(episodes)
    arr = np.asarray(leads, dtype=float)
    out: Dict[str, Any] = {
        "episodes": n,
        "detected": len(leads),
        "missed": len(missed),
        "detection_rate": float(len(leads) / n) if n else 0.0,
        "missed_ids": missed[:50],
        "threshold": float(threshold),
        "persistence": int(persistence),
    }

    if arr.size:
        out.update(
            median_lead_seconds=float(np.median(arr)),
            mean_lead_seconds=float(arr.mean()),
            p10_lead_seconds=float(np.percentile(arr, 10)),
            p90_lead_seconds=float(np.percentile(arr, 90)),
            min_lead_seconds=float(arr.min()),
            max_lead_seconds=float(arr.max()),
        )
        # "warned at least S seconds ahead", over ALL episodes, not just caught ones
        out["recall_at_lead"] = {
            f"{s:g}s": float(np.sum(arr >= s) / n) if n else 0.0 for s in recall_at_seconds
        }
    else:
        out["note"] = "no episode detected before onset; lead time undefined"

    if hours_observed and hours_observed > 0:
        out["hours_observed"] = float(hours_observed)

    return out
