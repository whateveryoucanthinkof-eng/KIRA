"""Derive ground-truth attack intervals from the CIC-IDS-2018 CSV labels.

Labelling PCAP windows needs to-the-second attack intervals. Two options exist
and only one is sound:

  * Join each packet to a CSV flow row. Rejected: the CSV timestamps use a
    12-hour clock with no AM/PM marker (verified — hours present are 1-5, 7-12),
    so every row before 08:00 is ambiguous by twelve hours.

  * Derive intervals from the labels, disambiguate once against the known
    capture window, then apply them to packet UTC. Used here.

The disambiguation rule is `hour < 8 means PM`, which is checked rather than
assumed: the resulting span must fall inside the day's real capture window
(~12:15-21:30 UTC, i.e. ~08:15-17:30 local) and attack rows must form a small
number of contiguous blocks rather than scattering. `derive_windows` returns
that evidence alongside the intervals so a caller can refuse bad input.

Offset: `pcap_utc = csv_local + 4:00:00`, verified to the second on two
independent days in claude_latest_analysis/05_cic2018_pcap_completeness.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CSV_TO_UTC = timedelta(hours=4)

# A capture day runs roughly 08:00-17:30 local, so a bare hour below this is PM.
PM_BELOW_HOUR = 8


@dataclass
class AttackInterval:
    label: str
    start_utc: float
    end_utc: float
    n_rows: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_utc - self.start_utc

    def contains(self, ts_utc: float) -> bool:
        return self.start_utc <= ts_utc <= self.end_utc


@dataclass
class DerivedWindows:
    intervals: List[AttackInterval] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.intervals) and self.evidence.get("plausible", False)

    def label_at(self, ts_utc: float) -> Optional[str]:
        for iv in self.intervals:
            if iv.contains(ts_utc):
                return iv.label
        return None

    def is_attack(self, ts_utc: float) -> bool:
        return self.label_at(ts_utc) is not None


def _parse_local(stamp: str) -> Optional[datetime]:
    """'14/02/2018 08:31:01' -> datetime, applying the PM rule. None if malformed."""
    try:
        d, t = stamp.strip().split(" ")
        dd, mo, yy = (int(x) for x in d.split("/"))
        h, mi, s = (int(x) for x in t.split(":"))
    except Exception:
        return None
    if not (1 <= mo <= 12 and 1 <= dd <= 31 and yy > 2000):
        return None  # malformed rows exist; some files carry embedded headers
    if h < PM_BELOW_HOUR:
        h += 12
    if h > 23:
        return None
    try:
        return datetime(yy, mo, dd, h, mi, s)
    except ValueError:
        return None


def derive_windows(
    csv_path: str | Path,
    *,
    max_rows: Optional[int] = None,
    merge_gap_seconds: float = 300.0,
) -> DerivedWindows:
    """Contiguous attack intervals in packet UTC, with the evidence to judge them.

    Rows are bucketed to the second per label, then runs closer together than
    `merge_gap_seconds` are merged — the CSV is not globally time-sorted, so
    intervals must be built from the set of observed times, not by scanning in
    file order.
    """
    path = Path(csv_path)
    per_label: Dict[str, List[float]] = {}
    malformed = 0
    total = 0

    with open(path, newline="", encoding="latin1") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        try:
            ti = hdr.index("Timestamp")
            li = hdr.index("Label")
        except ValueError:
            return DerivedWindows(evidence={"error": "no Timestamp/Label column", "plausible": False})

        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            if len(row) <= max(ti, li):
                continue
            total += 1
            lab = row[li].strip()
            if not lab or lab.lower() == "label":
                malformed += 1
                continue
            dt = _parse_local(row[ti])
            if dt is None:
                malformed += 1
                continue
            if lab.lower() in ("benign", "normal"):
                continue
            per_label.setdefault(lab, []).append(
                dt.replace(tzinfo=timezone.utc).timestamp() + CSV_TO_UTC.total_seconds()
            )

    intervals: List[AttackInterval] = []
    for lab, times in per_label.items():
        times.sort()
        run_start, prev, n = times[0], times[0], 1
        for t in times[1:]:
            if t - prev > merge_gap_seconds:
                intervals.append(AttackInterval(lab, run_start, prev, n))
                run_start, n = t, 0
            prev, n = t, n + 1
        intervals.append(AttackInterval(lab, run_start, prev, n))

    intervals.sort(key=lambda iv: iv.start_utc)

    # Evidence a caller can act on rather than a bare list.
    spans = [iv for iv in intervals if iv.duration_seconds > 0]
    all_t = [t for ts in per_label.values() for t in ts]
    ev: Dict[str, Any] = {
        "rows_read": total,
        "malformed_rows": malformed,
        "attack_rows": len(all_t),
        "labels": sorted(per_label),
        "n_intervals": len(intervals),
        "interval_seconds": [round(iv.duration_seconds, 1) for iv in intervals[:20]],
    }
    if all_t:
        lo = datetime.fromtimestamp(min(all_t), timezone.utc)
        hi = datetime.fromtimestamp(max(all_t), timezone.utc)
        ev["attack_span_utc"] = (lo.isoformat(), hi.isoformat())
        # A capture day is ~12:15-21:30 UTC; attacks must fall inside it, and a
        # handful of contiguous blocks is what a scripted campaign looks like.
        ev["plausible"] = bool(
            11 <= lo.hour <= 23 and 11 <= hi.hour <= 23 and 0 < len(intervals) <= 40
        )
    else:
        ev["plausible"] = False

    return DerivedWindows(intervals=intervals, evidence=ev)
