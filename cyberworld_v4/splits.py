"""
cyberworld_v4.splits — chronological and scenario-held-out splits.

Spec 36. Four splits, not three:

    TRAIN        fit parameters
    VALIDATION   model selection, early stopping, hyperparameters
    CALIBRATION  post-hoc temperature / conformal quantiles ONLY
    TEST         untouched until the model is frozen

CALIBRATION must be disjoint from VALIDATION. Fitting a temperature or a
conformal quantile on data used for model selection inherits that selection's
optimism, and the resulting coverage claim is not honest.

Splitting is never random over rows. Rows are not independent: sliding windows
overlap, and a host's windows are strongly autocorrelated. The unit is a
*group* — a capture, scenario, or host — so that no group appears on both sides
of a boundary (spec 24, 25).

Two regimes:

  chronological     hold out later time. Answers "does it work tomorrow?"
  scenario_held_out hold out whole scenarios/attack families. Answers
                    "does it work on an attack it never saw?" — the claim
                    PS 26153 asks about, and the harder of the two.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SplitName = str
TRAIN, VAL, CALIB, TEST = "train", "validation", "calibration", "test"
SPLIT_ORDER = (TRAIN, VAL, CALIB, TEST)


@dataclass
class SplitAssignment:
    """Which group went where, plus an audit trail."""

    groups: Dict[SplitName, List[str]] = field(default_factory=dict)
    strategy: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def of(self, group: str) -> Optional[SplitName]:
        for name, gs in self.groups.items():
            if group in gs:
                return name
        return None

    def counts(self) -> Dict[SplitName, int]:
        return {k: len(v) for k, v in self.groups.items()}

    def assert_disjoint(self) -> None:
        seen: Dict[str, SplitName] = {}
        for name, gs in self.groups.items():
            for g in gs:
                if g in seen:
                    raise ValueError(
                        f"group {g!r} appears in both {seen[g]!r} and {name!r}; "
                        "splits must be disjoint"
                    )
                seen[g] = name

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "groups": self.groups, "notes": self.notes}


def chronological_split(
    items: Sequence[Any],
    time_of: Callable[[Any], float],
    group_of: Callable[[Any], str],
    *,
    fractions: Tuple[float, float, float, float] = (0.6, 0.15, 0.10, 0.15),
) -> SplitAssignment:
    """Split by time, cutting only at group boundaries.

    A group is assigned by its earliest timestamp, so a capture is never torn
    across a boundary. Fractions are over groups, not rows.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")

    first_seen: Dict[str, float] = {}
    for it in items:
        g = group_of(it)
        t = float(time_of(it))
        if g not in first_seen or t < first_seen[g]:
            first_seen[g] = t

    ordered = [g for g, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]
    n = len(ordered)
    n_tr = int(n * fractions[0])
    n_va = int(n * fractions[1])
    n_ca = int(n * fractions[2])

    assign = SplitAssignment(
        groups={
            TRAIN: ordered[:n_tr],
            VAL: ordered[n_tr : n_tr + n_va],
            CALIB: ordered[n_tr + n_va : n_tr + n_va + n_ca],
            TEST: ordered[n_tr + n_va + n_ca :],
        },
        strategy="chronological",
        notes={"n_groups": n, "fractions": list(fractions), "ordered_by": "first timestamp"},
    )
    assign.assert_disjoint()
    return assign


def scenario_held_out_split(
    groups: Sequence[str],
    *,
    test_groups: Sequence[str],
    calibration_groups: Sequence[str],
    validation_groups: Sequence[str],
) -> SplitAssignment:
    """Hold out named scenarios/captures wholesale.

    The strongest generalisation test (spec 38): the test scenarios are attack
    families or captures the model has never seen in any form.
    """
    test = list(dict.fromkeys(test_groups))
    calib = list(dict.fromkeys(calibration_groups))
    val = list(dict.fromkeys(validation_groups))
    held = set(test) | set(calib) | set(val)
    train = [g for g in dict.fromkeys(groups) if g not in held]

    assign = SplitAssignment(
        groups={TRAIN: train, VAL: val, CALIB: calib, TEST: test},
        strategy="scenario_held_out",
        notes={"held_out": sorted(held)},
    )
    assign.assert_disjoint()
    return assign


def partition(
    items: Sequence[Any],
    assignment: SplitAssignment,
    group_of: Callable[[Any], str],
) -> Dict[SplitName, List[Any]]:
    """Apply an assignment to items. Items in unassigned groups are dropped."""
    out: Dict[SplitName, List[Any]] = {k: [] for k in SPLIT_ORDER}
    for it in items:
        name = assignment.of(group_of(it))
        if name is not None:
            out[name].append(it)
    return out


def leakage_report(
    parts: Dict[SplitName, Sequence[Any]],
    host_of: Callable[[Any], str],
    time_of: Optional[Callable[[Any], float]] = None,
) -> Dict[str, Any]:
    """Check the split actually holds. Run this before trusting any metric.

    Reports host overlap between splits, and — for chronological splits —
    whether test time genuinely follows train time.
    """
    hosts = {k: {host_of(i) for i in v} for k, v in parts.items()}
    overlaps: Dict[str, int] = {}
    names = [n for n in SPLIT_ORDER if n in parts]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = hosts[a] & hosts[b]
            if shared:
                overlaps[f"{a}&{b}"] = len(shared)

    report: Dict[str, Any] = {
        "sizes": {k: len(v) for k, v in parts.items()},
        "unique_hosts": {k: len(v) for k, v in hosts.items()},
        "host_overlap": overlaps,
        "clean": not overlaps,
    }

    if time_of is not None:
        spans = {}
        for k, v in parts.items():
            if v:
                ts = [float(time_of(i)) for i in v]
                spans[k] = (min(ts), max(ts))
        report["time_spans"] = spans
        if TRAIN in spans and TEST in spans:
            report["test_after_train"] = spans[TEST][0] >= spans[TRAIN][1]

    return report
