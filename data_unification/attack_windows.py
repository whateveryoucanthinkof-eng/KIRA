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
    #: Addresses observed on the attack rows of this interval: the attacker
    #: and its victims. EMPTY means "not known", not "nobody".
    #:
    #: Without this the interval is a pure time range, and `pcap_bridge`
    #: stamped it on every host active in the window -- ~445 hosts per
    #: CIC-2018 day, of which one or two were actually involved. A model
    #: trained on that can score well by learning what time of day it is,
    #: which is why it has to be scoped wherever the corpus allows.
    participants: frozenset = field(default_factory=frozenset)

    @property
    def duration_seconds(self) -> float:
        return self.end_utc - self.start_utc

    @property
    def scoped(self) -> bool:
        """True when this interval knows who took part."""
        return bool(self.participants)

    def contains(self, ts_utc: float) -> bool:
        return self.start_utc <= ts_utc <= self.end_utc

    def involves(self, *ips: str) -> bool:
        """Whether any of `ips` took part. Unscoped intervals answer True:
        an unknown participant set cannot exclude anyone, and silently
        excluding everyone would delete the day's labels."""
        if not self.participants:
            return True
        return any(ip in self.participants for ip in ips)


@dataclass
class DerivedWindows:
    intervals: List[AttackInterval] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.intervals) and self.evidence.get("plausible", False)

    def interval_at(self, ts_utc: float) -> Optional[AttackInterval]:
        for iv in self.intervals:
            if iv.contains(ts_utc):
                return iv
        return None

    def label_at(self, ts_utc: float) -> Optional[str]:
        iv = self.interval_at(ts_utc)
        return iv.label if iv else None

    def is_attack(self, ts_utc: float) -> bool:
        return self.label_at(ts_utc) is not None

    @property
    def scoped(self) -> bool:
        """True when every interval knows its participants, i.e. labels can be
        attributed to hosts rather than only to instants."""
        return bool(self.intervals) and all(iv.scoped for iv in self.intervals)

    def participant_summary(self) -> Dict[str, Any]:
        return {
            "intervals": len(self.intervals),
            "scoped_intervals": sum(1 for iv in self.intervals if iv.scoped),
            "participants_by_label": {
                iv.label: sorted(iv.participants)[:12] for iv in self.intervals if iv.scoped
            },
        }


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
    per_label_ips: Dict[str, set] = {}
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

        # Src/Dst IP where the corpus has them. Nine of the ten CIC-2018 CSVs
        # ship with 80 columns and no addresses at all (see
        # claude_latest_analysis/15_downloads_csv_audit.md), so this is
        # opportunistic: when the columns exist the interval can name its
        # participants, and when they do not the interval stays a pure time
        # range and says so.
        _norm = {c.strip().lower(): i for i, c in enumerate(hdr)}
        si = _norm.get("src ip", _norm.get("source ip"))
        di = _norm.get("dst ip", _norm.get("destination ip"))

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
            if si is not None and di is not None and len(row) > max(si, di):
                ips = per_label_ips.setdefault(lab, set())
                a, b = row[si].strip(), row[di].strip()
                if a:
                    ips.add(a)
                if b:
                    ips.add(b)

    intervals: List[AttackInterval] = []
    for lab, times in per_label.items():
        times.sort()
        # Participants are collected per LABEL, not per merged run: the run
        # boundaries come from `merge_gap_seconds`, while the campaign they
        # belong to is the label.
        who = frozenset(per_label_ips.get(lab, ()))
        run_start, prev, n = times[0], times[0], 1
        for t in times[1:]:
            if t - prev > merge_gap_seconds:
                intervals.append(AttackInterval(lab, run_start, prev, n, who))
                run_start, n = t, 0
            prev, n = t, n + 1
        intervals.append(AttackInterval(lab, run_start, prev, n, who))

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
        "has_ip_columns": bool(si is not None and di is not None),
        "scoped_intervals": sum(1 for iv in intervals if iv.scoped),
    }
    if intervals and not ev["has_ip_columns"]:
        ev["label_scope_warning"] = (
            "this CSV has no Src/Dst IP columns, so the intervals are time-only. "
            "Every host active during an attack window will be labelled attacked "
            "unless a participant map is supplied "
            "(data_unification/attack_participants.py)."
        )
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
