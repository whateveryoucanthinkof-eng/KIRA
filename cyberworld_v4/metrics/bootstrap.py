"""Group-level bootstrap confidence intervals (spec 48, 49).

Resampling individual rows is wrong here. Sliding windows overlap by
history_steps-1 states, and a host's consecutive windows are strongly
autocorrelated, so 35,993 "samples" carry far less information than 35,993
independent draws. Row bootstrap would produce intervals that are too narrow —
confidently wrong.

Resample whole groups instead (capture, scenario, host-session). The group is
the independent unit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def group_bootstrap_ci(
    items: Sequence[Any],
    group_of: Callable[[Any], str],
    statistic: Callable[[Sequence[Any]], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Percentile bootstrap over groups.

    Returns the point estimate, the interval, and `n_groups` — the number that
    actually determines the interval's width, and which should be reported
    alongside it.
    """
    buckets: Dict[str, List[Any]] = defaultdict(list)
    for it in items:
        buckets[group_of(it)].append(it)

    keys = list(buckets)
    n_groups = len(keys)
    point = float(statistic(items))

    if n_groups < 2:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_groups": n_groups,
            "note": "fewer than 2 groups; interval undefined",
        }

    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(n_resamples):
        picked = rng.integers(0, n_groups, size=n_groups)
        sample: List[Any] = []
        for i in picked:
            sample.extend(buckets[keys[i]])
        try:
            v = float(statistic(sample))
        except Exception:
            continue
        if np.isfinite(v):
            stats.append(v)

    if len(stats) < 10:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_groups": n_groups,
            "note": "statistic undefined on too many resamples",
        }

    arr = np.asarray(stats)
    return {
        "point": point,
        "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_groups": n_groups,
        "n_resamples": len(stats),
        "alpha": alpha,
    }


def multi_seed_summary(values: Sequence[float]) -> Dict[str, Any]:
    """mean +/- sd across seeds (spec 47). Report with n; 5 seeds is not many."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],
    }


def format_ci(d: Dict[str, Any], digits: int = 3) -> str:
    if not np.isfinite(d.get("ci_low", float("nan"))):
        return f"{d['point']:.{digits}f} (CI undefined, n_groups={d.get('n_groups')})"
    return f"{d['point']:.{digits}f} [{d['ci_low']:.{digits}f}, {d['ci_high']:.{digits}f}]"
