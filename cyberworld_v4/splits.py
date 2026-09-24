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

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    sizes: Dict[str, int] = defaultdict(int)
    for it in items:
        g = group_of(it)
        t = float(time_of(it))
        sizes[g] += 1
        if g not in first_seen or t < first_seen[g]:
            first_seen[g] = t

    ordered = [g for g, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]
    total = sum(sizes.values())

    # Fill each split by SAMPLE count, not by group count. Splitting the group
    # list by count assumes groups are similar sizes; hosts are not. Measured on
    # a real run, 14 hosts split 8/2/1/2 by count produced train 8,524 /
    # validation 50 / calibration 357 / test 15,260 — a validation set too small
    # to pick a threshold from and a test set larger than train.
    #
    # Groups are still never split, so the boundary guarantee is unchanged; only
    # where the boundary falls changes.
    # Two constraints, in priority order:
    #   1. every split gets at least one group (an empty split is unusable)
    #   2. sample counts land as close to `fractions` as group boundaries allow
    #
    # Constraint 1 comes first because a 0-sample validation split cannot pick a
    # threshold and a 0-sample test split cannot be scored. When one group holds
    # most of the data, no sample-balanced split exists at all; that is reported
    # in `notes` rather than silently producing a lopsided result.
    n_splits = len(SPLIT_ORDER)
    if len(ordered) < n_splits:
        raise ValueError(
            f"{len(ordered)} groups cannot fill {n_splits} splits; "
            f"need at least one group per split"
        )

    # Splits are contiguous in time, so the assignment is three cut points in
    # the time-ordered group list. Choose them greedily against the cumulative
    # sample targets, then repair any empty split by moving a boundary.
    #
    # Repair is required, not cosmetic: when one host holds most of the samples
    # the greedy pass puts everything before the first cut and leaves later
    # splits empty. A 0-sample validation split cannot pick a threshold and a
    # 0-sample test split cannot be scored.
    cum_targets = np.cumsum([f * total for f in fractions])[: n_splits - 1]

    cuts: List[int] = []
    acc = 0.0
    ti = 0
    for i, g in enumerate(ordered):
        acc += sizes[g]
        while ti < len(cum_targets) and acc >= cum_targets[ti]:
            cuts.append(i + 1)
            ti += 1
    while len(cuts) < n_splits - 1:
        cuts.append(len(ordered))

    # Repair: enforce strictly increasing cuts leaving >=1 group per split.
    n = len(ordered)
    for i in range(len(cuts)):
        lo = i + 1                       # at least one group before this cut
        hi = n - (len(cuts) - i)         # leave one group for each later split
        cuts[i] = max(lo, min(cuts[i], hi))
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1

    bounds = [0] + cuts + [n]
    buckets: Dict[SplitName, List[str]] = {
        SPLIT_ORDER[i]: ordered[bounds[i] : bounds[i + 1]] for i in range(n_splits)
    }

    achieved = {k: sum(sizes[g] for g in v) for k, v in buckets.items()}
    largest = max(sizes.values()) if sizes else 0
    dominated = largest > 0.5 * total

    assign = SplitAssignment(
        groups=buckets,
        strategy="chronological",
        notes={
            "n_groups": len(ordered),
            "fractions_requested": list(fractions),
            "fractions_achieved": {k: round(v / total, 4) for k, v in achieved.items()},
            "samples_per_split": achieved,
            "ordered_by": "first timestamp",
            "balanced_by": "sample count, one group per split guaranteed",
            "largest_group_share": round(largest / total, 4) if total else 0.0,
            "dominated_by_one_group": dominated,
            "warning": (
                "one group holds >50% of samples; no sample-balanced split exists "
                "and the achieved fractions will be far from requested"
                if dominated else None
            ),
        },
    )
    assign.assert_disjoint()
    return assign


def frozen_capture_split(
    items,
    group_of,
    lock,
    *,
    calibration_fraction: float = 0.15,
    time_of=None,
) -> SplitAssignment:
    """Apply `data_unification/splits.lock.json` to v4 samples.

    The lock is the repository's existing answer to "which capture is in which
    split": 41 captures, stratified by measured attack fraction and dealt
    round-robin 4:1:1 so no split clusters all the quiet days together. Every
    v3 trainer reads it. The v4 trainer did not -- it recomputed a split at
    training time from bare IPs, which is both a weaker design and a different
    answer, so v3 and v4 numbers were never comparable.

    The lock names three splits and v4 needs four. CALIBRATION is carved out of
    the LOCK'S TRAIN captures -- never out of validation, which would inherit
    model-selection optimism, and never out of test. It takes the last
    `calibration_fraction` of train captures in time order, so calibration data
    is also the most recent train data, which is the closest stand-in for
    deployment conditions.

    `lock` is `{dataset: {capture_name: split}}`, i.e. the output of
    `data_unification.split_policy.load_lock()`. A group the lock does not name
    is an error: dropping it silently would shrink a split, and assigning it on
    the fly would make the split unreproducible.
    """
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError(f"calibration_fraction must be in [0, 1), got {calibration_fraction}")

    flat = {f"{ds}|{cap}": split for ds, m in lock.items() for cap, split in m.items()}

    first_seen: Dict[str, float] = {}
    sizes: Dict[str, int] = defaultdict(int)
    order: List[str] = []
    for idx, it in enumerate(items):
        g = group_of(it)
        if g not in sizes:
            order.append(g)
        sizes[g] += 1
        t = float(time_of(it)) if time_of is not None else float(idx)
        if g not in first_seen or t < first_seen[g]:
            first_seen[g] = t

    unknown = [g for g in order if g not in flat]
    if unknown:
        raise KeyError(
            f"{len(unknown)} capture(s) are not in the frozen split lock: "
            f"{sorted(unknown)[:5]}{'...' if len(unknown) > 5 else ''}. Add them "
            f"with scripts/freeze_splits.py rather than assigning them here."
        )

    buckets: Dict[SplitName, List[str]] = {k: [] for k in SPLIT_ORDER}
    lock_train = []
    for g in order:
        where = flat[g]
        if where == "train":
            lock_train.append(g)
        elif where == "val":
            buckets[VAL].append(g)
        elif where == "test":
            buckets[TEST].append(g)
        else:
            raise ValueError(f"lock assigns {g} to unknown split {where!r}")

    # Carve calibration off the END of train, in time order.
    lock_train.sort(key=lambda g: (first_seen[g], g))
    n_cal = int(round(calibration_fraction * len(lock_train)))
    n_cal = max(1, n_cal) if lock_train and calibration_fraction > 0 else n_cal
    n_cal = min(n_cal, max(0, len(lock_train) - 1))   # never empty train
    buckets[CALIB] = lock_train[len(lock_train) - n_cal:] if n_cal else []
    buckets[TRAIN] = lock_train[: len(lock_train) - n_cal]

    achieved = {k: sum(sizes[g] for g in v) for k, v in buckets.items()}
    total = sum(sizes.values())
    assign = SplitAssignment(
        groups=buckets,
        strategy="frozen_capture_lock",
        notes={
            "lock_captures": len(flat),
            "groups_present": len(order),
            "samples_per_split": achieved,
            "fractions_achieved": {k: round(v / total, 4) for k, v in achieved.items()} if total else {},
            "calibration_fraction_requested": calibration_fraction,
            "calibration_from": "last train captures in time order",
            "unit": "capture",
        },
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
