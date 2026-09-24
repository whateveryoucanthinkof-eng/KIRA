#!/usr/bin/env python3
"""Check config/attack_participants.json against the corpus itself.

A transcribed attacker/victim table is only useful if it is right, and a wrong
one is worse than none: it would scope labels to the wrong hosts, which is a
new label error on top of the one it was meant to fix.

Most CIC-2018 CSVs cannot check it -- nine of ten ship with 80 columns and no
Src/Dst IP at all. But `tue_20_csv.csv` has 84 columns including addresses, and
every CIC-2017 file has them, so those days can answer for themselves. This
script derives participants from whichever files do have IP columns and reports
agreement with the map.

    python scripts/verify_attack_participants.py ~/Documents/SIH/DATA/CSV/tue_20_csv.csv
    python scripts/verify_attack_participants.py --map config/attack_participants.json FILE...

Exit code is 1 when the map contradicts the data, so it can gate a run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_unification.attack_participants import (  # noqa: E402
    DEFAULT_MAP_PATH,
    load_participant_map,
    participants_from_derived,
)
from data_unification.attack_windows import derive_windows  # noqa: E402


def day_of(path: Path) -> str:
    """The day key a pcap directory and a CSV share: 'tue_20_csv.csv' -> 'tue_20'."""
    return path.stem.replace("_csv", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", type=Path, nargs="+",
                    help="corpus CSVs; only those with Src/Dst IP columns can be checked")
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    mapping = load_participant_map(args.map)
    print(f"map: {args.map}")
    print(f"  provenance: {mapping.provenance or '(none recorded)'}")
    print(f"  days: {sorted(mapping.days) or '(empty)'}\n")

    checkable = 0
    disagreements = 0

    for csv in args.csvs:
        if not csv.exists():
            print(f"{csv}: missing")
            continue
        derived = derive_windows(csv, max_rows=args.max_rows)
        ev = derived.evidence
        day = day_of(csv)
        print(f"{csv.name}  ({day})")
        print(f"  intervals {ev.get('n_intervals', 0)}  labels {ev.get('labels')}")

        if not ev.get("has_ip_columns"):
            print("  no Src/Dst IP columns -- this day cannot verify itself.")
            claimed = {lab: sorted(mapping.lookup(day, lab))[:6]
                       for lab in (ev.get("labels") or ())}
            covered = {k: v for k, v in claimed.items() if v}
            print(f"  map covers {len(covered)}/{len(claimed)} of its labels: {covered or '{}'}")
            print()
            continue

        checkable += 1
        observed = participants_from_derived(derived)
        for label, ips in sorted(observed.items()):
            claimed = mapping.lookup(day, label)
            if not claimed:
                print(f"  {label}: {len(ips)} host(s) observed, no map entry "
                      f"(nothing to contradict)")
                continue
            extra = claimed - ips
            missing = ips - claimed
            verdict = "OK" if not extra else "DISAGREES"
            if extra:
                disagreements += 1
            print(f"  {label}: {verdict}  map={len(claimed)} observed={len(ips)}")
            if extra:
                print(f"    in the map but never seen on an attack row: {sorted(extra)[:8]}")
            if missing:
                print(f"    seen on attack rows but absent from the map: {sorted(missing)[:8]}")
        print()

    if not checkable:
        print("Nothing was verifiable: none of these files carry IP columns. "
              "The map remains unverified -- treat scoped labels from it as an "
              "assumption, not as data.")
        return 0
    if disagreements:
        print(f"{disagreements} label(s) name hosts the data never shows on an "
              f"attack row. Fix the map before training on it.")
        return 1
    print(f"Verified against {checkable} file(s); no contradictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
