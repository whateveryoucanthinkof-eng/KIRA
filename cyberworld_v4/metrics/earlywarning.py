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


# ---------------------------------------------------------------------------
# Adapter: ForecastSample sequences -> Episode list
# ---------------------------------------------------------------------------
#
# `lead_time_report` above is the metric; this is the missing wire between it
# and what a trainer actually holds. Without it the module had no caller, and
# every "lead time" the project reported was `forecast_steps * window_seconds`
# -- the horizon the model was ASKED about, which is a constant and says
# nothing about whether a single onset was ever warned about.


def episodes_from_series(
    episode_prefix: str,
    times: np.ndarray,
    current_attack: np.ndarray,
    scores: np.ndarray,
) -> List[Episode]:
    """Split one host's window series into attack episodes at each 0 -> 1 onset.

    Alert opportunities for an episode are the event-free windows since the end
    of the previous episode, never the whole history: crediting an alert raised
    during an earlier, unrelated attack would report lead time for a warning
    that was about something else.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(current_attack, dtype=int)
    sc = np.asarray(scores, dtype=float)
    if t.size == 0:
        return []

    order = np.argsort(t, kind="stable")
    t, a, sc = t[order], a[order], sc[order]

    out: List[Episode] = []
    window_start = 0          # first index of the current event-free run
    k = 0
    for i in range(t.size):
        if a[i] == 1 and (i == 0 or a[i - 1] == 0):
            if i > window_start:
                out.append(Episode(
                    episode_id=f"{episode_prefix}#{k}",
                    onset_time=float(t[i]),
                    times=t[window_start:i],
                    scores=sc[window_start:i],
                ))
                k += 1
        elif a[i] == 0 and i > 0 and a[i - 1] == 1:
            window_start = i  # previous episode ended here
    return out


def lead_time_from_samples(
    samples: Sequence[Any],
    future_probabilities: np.ndarray,
    *,
    threshold: float,
    persistence: int = 1,
    window_seconds: float = 2.0,
    recall_at_seconds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Lead-time report for a split of `cyberworld_v4.targets.ForecastSample`.

    `future_probabilities` is [N, K], aligned with `samples`. The alert score
    for a window is `max_k P(A_{t+k})`: the model's strongest claim that an
    attack is coming inside the horizon. A per-step probability cannot be
    compared against a single threshold, and the hazard curve answers a
    different question (when, given not yet), so the max is the quantity an
    operator's "warn me" switch actually reads.
    """
    p = np.asarray(future_probabilities, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    score = p.max(axis=1)

    by_host: Dict[str, List[int]] = {}
    for i, smp in enumerate(samples):
        by_host.setdefault(smp.host.key, []).append(i)

    episodes: List[Episode] = []
    span = 0.0
    for host, idxs in by_host.items():
        t = np.array([samples[i].t_end for i in idxs], dtype=float)
        a = np.array([samples[i].current_attack for i in idxs], dtype=int)
        episodes.extend(episodes_from_series(host, t, a, score[idxs]))
        if t.size > 1:
            span += float(t.max() - t.min())

    K = p.shape[1]
    if recall_at_seconds is None:
        # Report at every horizon the model was trained to predict, so the
        # numbers line up with the per-step forecast table beside them.
        recall_at_seconds = tuple(window_seconds * (k + 1) for k in range(K))

    report = lead_time_report(
        episodes, threshold=threshold, persistence=persistence,
        hours_observed=span / 3600.0 if span else None,
        recall_at_seconds=recall_at_seconds,
    )
    report["hosts"] = len(by_host)
    report["score"] = "max_k P(A_t+k)"
    report["horizon_seconds"] = float(window_seconds * K)
    return report
