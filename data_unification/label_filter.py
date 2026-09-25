"""Keep unresolved labels out of supervised training and out of benchmarks.

`label_resolver` already does the hard part correctly. A raw label its maps do
not recognise resolves to the third state `UNKNOWN` with `is_attack=False`, and
the module says plainly what a caller is then supposed to do:

    UNKNOWN is now a third state (spec 15). is_attack is False so nothing is
    asserted, and callers building a benchmark must exclude these rows rather
    than treat them as benign.

No caller did. `is_unresolved()` and `LabelResolver.unresolved_report()` had
zero call sites in the repository, so every label the maps missed -- a new
attack name, a typo, a renamed category, a corpus the maps were never written
for -- arrived at the loss function as a confident negative. That is the same
defect the third state was introduced to fix, moved one layer downstream: the
resolver stopped asserting, and the trainer asserted on its behalf.

This module is the missing call site. It is deliberately a filter over records
rather than a flag on each adapter, because the decision belongs to whoever is
building a training set, not to whoever is reading a CSV.

Usage:

    records, report = drop_unresolved(records)
    print(format_unresolved_report(report))

`report` belongs beside any headline metric: it says how much of the corpus the
ontology could not name, which bounds what the metric can mean.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from data_unification.label_resolver import UNKNOWN_CATEGORY, is_unresolved

__all__ = [
    "count_unresolved",
    "drop_unresolved",
    "format_unresolved_report",
    "merge_unresolved_reports",
    "partition_unresolved",
]


def _category(record: Any) -> str:
    return str(getattr(record, "coarse_category", "") or "")


def partition_unresolved(records: Iterable[Any]) -> Tuple[List[Any], List[Any]]:
    """(resolved, unresolved). Order is preserved within each list."""
    resolved: List[Any] = []
    unresolved: List[Any] = []
    for r in records:
        (unresolved if is_unresolved(_category(r)) else resolved).append(r)
    return resolved, unresolved


def count_unresolved(records: Sequence[Any], *, top: int = 20) -> Dict[str, Any]:
    """Coverage report without dropping anything.

    `raw_labels` names the actual strings that failed to map, which is what
    someone fixing the label maps needs; a rate alone does not tell them which
    entry to add.
    """
    total = len(records)
    bad = [r for r in records if is_unresolved(_category(r))]
    raw = Counter(str(getattr(r, "raw_label", "")) for r in bad)
    by_source = Counter(str(getattr(r, "raw_label_source", "")) for r in bad)
    return {
        "total_records": total,
        "unresolved_records": len(bad),
        "unresolved_rate": (len(bad) / total) if total else 0.0,
        "resolved_rate": (1.0 - len(bad) / total) if total else 0.0,
        "distinct_unresolved_labels": len(raw),
        "top_unresolved_labels": raw.most_common(top),
        "unresolved_by_source": dict(by_source),
        "unknown_category": UNKNOWN_CATEGORY,
    }


def drop_unresolved(records: Sequence[Any], *, top: int = 20) -> Tuple[List[Any], Dict[str, Any]]:
    """Records the ontology could name, plus the report for the ones it could not."""
    report = count_unresolved(records, top=top)
    resolved, _ = partition_unresolved(records)
    return resolved, report


def format_unresolved_report(report: Dict[str, Any], *, where: str = "") -> str:
    """One short block, printable beside a training header.

    It is loud when the rate is high because an unresolved rate above a few
    percent means the benchmark is scoring a corpus the label maps do not
    cover, and that has to be visible without anyone going looking.
    """
    tag = f" [{where}]" if where else ""
    n, k = report["total_records"], report["unresolved_records"]
    rate = report["unresolved_rate"]
    lines = [
        f"label coverage{tag}: {n - k}/{n} resolved ({report['resolved_rate']:.4f}), "
        f"{k} dropped as {report['unknown_category']} ({rate:.4f})"
    ]
    if k:
        shown = ", ".join(f"{lbl!r}x{c}" for lbl, c in report["top_unresolved_labels"][:8])
        lines.append(f"  {report['distinct_unresolved_labels']} distinct unmapped labels: {shown}")
        if report["unresolved_by_source"]:
            lines.append(f"  by source: {report['unresolved_by_source']}")
    if rate > 0.05:
        lines.append(
            f"  WARNING: {rate:.1%} of this split could not be named by "
            f"data_unification/label_maps/. Any metric computed on it describes "
            f"the mapped subset only -- add the labels above to the maps before "
            f"reporting a number."
        )
    return "\n".join(lines)


def merge_unresolved_reports(reports: Sequence[Dict[str, Any]], *, top: int = 20) -> Dict[str, Any]:
    """Fold per-capture reports into one split-level report.

    The streaming trainers load one capture at a time and free it, so no single
    call ever sees the whole split. Coverage still has to be reported for the
    split as a whole, or a corpus that is 40% unmapped hides behind ten
    individually unremarkable lines.
    """
    labels: Counter = Counter()
    sources: Counter = Counter()
    total = unresolved = 0
    for r in reports:
        total += int(r.get("total_records", 0))
        unresolved += int(r.get("unresolved_records", 0))
        labels.update(dict(r.get("top_unresolved_labels", ())))
        sources.update(r.get("unresolved_by_source", {}) or {})
    return {
        "total_records": total,
        "unresolved_records": unresolved,
        "unresolved_rate": (unresolved / total) if total else 0.0,
        "resolved_rate": (1.0 - unresolved / total) if total else 0.0,
        "distinct_unresolved_labels": len(labels),
        "top_unresolved_labels": labels.most_common(top),
        "unresolved_by_source": dict(sources),
        "unknown_category": UNKNOWN_CATEGORY,
        "merged_from": len(reports),
    }
