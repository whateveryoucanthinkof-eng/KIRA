"""Who took part in each attack, so a label can name hosts and not just instants.

## The problem this exists for

`pcap_bridge.iter_day_records` labelled a window like this:

    raw_label = attack_windows.label_at(mid_ts) or "BENIGN"
    coarse, technique_ids, is_attack = resolver.resolve(raw_label, ...)
    for host_ip, (flows, _pf) in per_host.items():
        for f in flows:
            rec = _flow_dict_to_record(f, is_attack=is_attack, ...)

`is_attack` is computed once per window from the clock and then stamped on
every flow of every host. A CIC-2018 day directory holds ~445 per-host
captures, so during the DDoS hour all 445 hosts are labelled under attack when
one or two were involved. The target becomes almost a function of the
timestamp, and a model can score well on it by learning the attack schedule.

`derive_windows` now recovers participants directly from the CSV wherever the
CSV has Src/Dst IP columns. On this corpus that is CIC-2017 (all files) and
CIC-2018 `tue_20` -- the other nine CIC-2018 days ship with 80 columns and no
addresses at all, which is the same defect that makes `cic2018_adapter` invent
`192.168.10.{i % 250 + 1}`.

## What this module adds

An explicit, auditable participant map for the days the CSV cannot answer for,
plus the code to apply it. Nothing here guesses: a day with no entry stays
unscoped and is reported as time-only, which is strictly the old behaviour plus
a warning. Inventing addresses would create new label errors on top of the ones
being fixed.

## Filling it in

The CSE-CIC-IDS2018 release documents an attacker/victim pair per attack.
Transcribe it into a JSON file:

    {
      "provenance": "CSE-CIC-IDS2018 published attack schedule, transcribed <date> by <who>",
      "days": {
        "wed_14": {
          "FTP-BruteForce": {"attackers": ["..."], "victims": ["..."]},
          "SSH-Bruteforce": {"attackers": ["..."], "victims": ["..."]}
        }
      }
    }

Then verify before trusting it -- `scripts/verify_attack_participants.py`
checks any day the corpus can answer for on its own (one with IP columns) and
reports agreement. A map that disagrees with the data is worse than no map.

Matching on the label is by normalised substring, so "DoS attacks-Hulk" in the
CSV finds a "Hulk" entry.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

LOG = logging.getLogger(__name__)

DEFAULT_MAP_PATH = Path(__file__).resolve().parent.parent / "config" / "attack_participants.json"

__all__ = [
    "DEFAULT_MAP_PATH",
    "ParticipantMap",
    "apply_participants",
    "load_participant_map",
    "participants_from_derived",
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


class ParticipantMap:
    """{day: {attack_label: set(ips)}}, matched leniently on the label."""

    def __init__(self, days: Dict[str, Dict[str, Set[str]]], provenance: str = ""):
        self.days = days
        self.provenance = provenance

    def __bool__(self) -> bool:
        return bool(self.days)

    def for_day(self, day: str) -> Dict[str, Set[str]]:
        key = _norm(day)
        for name, table in self.days.items():
            if _norm(name) == key:
                return table
        # A pcap directory is named e.g. "thu_15_pacap" for the day "thu_15".
        for name, table in self.days.items():
            n = _norm(name)
            if n and (n in key or key in n):
                return table
        return {}

    def lookup(self, day: str, label: str) -> Set[str]:
        table = self.for_day(day)
        if not table:
            return set()
        want = _norm(label)
        if want in {_norm(k) for k in table}:
            return {ip for k, v in table.items() if _norm(k) == want for ip in v}
        # substring either way: "DoS attacks-Hulk" <-> "Hulk"
        hits: Set[str] = set()
        for k, v in table.items():
            nk = _norm(k)
            if nk and (nk in want or want in nk):
                hits |= set(v)
        return hits


def load_participant_map(path: Optional[Path] = None) -> ParticipantMap:
    """Load the map, or an empty one when the file is absent.

    Absence is a normal state, not an error: it means every day stays
    time-only, which is what the pipeline did before this existed.
    """
    path = Path(path) if path else DEFAULT_MAP_PATH
    if not path.exists():
        return ParticipantMap({}, provenance=f"no map at {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Never fail silently here: a broken map reads as "no participants",
        # which silently restores the time-only labelling this file exists to
        # replace.
        LOG.error("participant map %s is unreadable (%s: %s); days will stay "
                  "time-only -- fix it rather than ignoring this",
                  path, type(exc).__name__, exc)
        return ParticipantMap({}, provenance=f"unreadable: {path}")

    days: Dict[str, Dict[str, Set[str]]] = {}
    for day, table in (doc.get("days") or {}).items():
        out: Dict[str, Set[str]] = {}
        for label, entry in (table or {}).items():
            ips: Set[str] = set()
            if isinstance(entry, dict):
                for key in ("attackers", "victims", "hosts", "ips"):
                    ips |= {str(x).strip() for x in (entry.get(key) or ()) if str(x).strip()}
            elif isinstance(entry, (list, tuple, set)):
                ips |= {str(x).strip() for x in entry if str(x).strip()}
            if ips:
                out[label] = ips
        if out:
            days[day] = out
    return ParticipantMap(days, provenance=str(doc.get("provenance", "")))


def participants_from_derived(derived: Any) -> Dict[str, Set[str]]:
    """{label: ips} recovered from a DerivedWindows, for days that had IPs."""
    out: Dict[str, Set[str]] = {}
    for iv in getattr(derived, "intervals", ()):
        if iv.participants:
            out.setdefault(iv.label, set()).update(iv.participants)
    return out


def apply_participants(
    derived: Any,
    day: str,
    mapping: Optional[ParticipantMap] = None,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Attach participants from `mapping` to any unscoped interval of `derived`.

    Intervals that already know their participants -- because the CSV carried
    addresses -- are left alone unless `overwrite`, since data beats a
    transcription. Returns a short report for the training log.
    """
    mapping = mapping if mapping is not None else load_participant_map()
    scoped_before = sum(1 for iv in derived.intervals if iv.scoped)
    filled, unmatched = 0, []

    for iv in derived.intervals:
        if iv.scoped and not overwrite:
            continue
        ips = mapping.lookup(day, iv.label)
        if ips:
            iv.participants = frozenset(ips)
            filled += 1
        elif not iv.scoped:
            unmatched.append(iv.label)

    report = {
        "day": day,
        "intervals": len(derived.intervals),
        "scoped_from_csv": scoped_before,
        "scoped_from_map": filled,
        "time_only_labels": sorted(set(unmatched)),
        "fully_scoped": derived.scoped,
        "map_provenance": mapping.provenance,
    }
    if unmatched:
        LOG.warning(
            "%s: %d interval(s) have no known participants (%s). Every host "
            "active in those windows will be labelled attacked. Add them to "
            "%s to scope the labels to the hosts actually involved.",
            day, len(unmatched), ", ".join(report["time_only_labels"][:6]),
            DEFAULT_MAP_PATH,
        )
    return report
