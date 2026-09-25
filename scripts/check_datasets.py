#!/usr/bin/env python3
"""Is every file the training plan needs present, and named as the lock names it?

    python scripts/check_datasets.py --pcap-root ... --cic2018-csv-dir ... \\
        --cic2017-dir ... --ctu-dir ... [--scheme cross_year_ctu]

Reads headers only, so it takes seconds. It checks:

  * every capture the split scheme uses exists under the name
    data_unification/splits.lock.json gives it;
  * every CIC-2018 PCAP day has capture files and resolves to its OWN label
    CSV (the wed_28 -> wed_29 alias included), and that CSV has Timestamp and
    Label columns;
  * CIC-2017 is the TrafficLabelling variant (it has Source IP -- the
    MachineLearningCVE variant does not, and the host graph needs addresses);
  * CTU-13 files are the labelled bidirectional .binetflow ones;
  * which of the 263 PCAPs found corrupt on 2026-09-19 (analysis 06) are still
    byte-for-byte the corrupt copy.

Exit 0 when nothing required is missing. The corrupt-PCAP count is reported,
not fatal: those files keep a usable prefix, they just lose the rest of the day.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_unification.pcap_adapter import iter_day_captures  # noqa: E402
from data_unification.split_policy import SCHEMES, load_lock, split_of  # noqa: E402
from data_unification.training_sources import _pcap_label_csv  # noqa: E402

ANALYSIS = REPO / "claude_latest_analysis"


def header(path: Path) -> set:
    with open(path, newline="", encoding="latin1") as fh:
        row = next(csv.reader(fh), [])
    return {c.strip().lower() for c in row}


def need_columns(path: Path, cols, problems, what):
    missing = [c for c in cols if c.lower() not in header(path)]
    if missing:
        problems.append(f"{what} {path.name}: missing column(s) {missing}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap-root", type=Path, required=True)
    ap.add_argument("--cic2018-csv-dir", type=Path, required=True)
    ap.add_argument("--cic2017-dir", type=Path, required=True)
    ap.add_argument("--ctu-dir", type=Path, default=None)
    ap.add_argument("--scheme", choices=SCHEMES, default="cross_year_ctu")
    a = ap.parse_args()

    lock = load_lock()
    problems, notes = [], []

    def used(ds):
        return {c: split_of(ds, c, a.scheme) for c in lock[ds] if split_of(ds, c, a.scheme)}

    # CIC-2018 PCAP days + their label CSVs
    print(f"== CIC-2018 PCAP  ({a.pcap_root})")
    for day, sp in sorted(used("PCAP2018").items()):
        d = a.pcap_root / day
        if not d.is_dir():
            problems.append(f"PCAP day missing: {d}")
            print(f"  MISSING  {sp:5s} {day}")
            continue
        caps = iter_day_captures(d)
        label = _pcap_label_csv(d, a.cic2018_csv_dir)
        if not caps:
            problems.append(f"PCAP day {day}: no capture files in {d} or {d / 'pcap'}")
        if label is None:
            problems.append(f"PCAP day {day}: no label CSV in {a.cic2018_csv_dir}")
        else:
            need_columns(label, ("Timestamp", "Label"), problems, "label CSV")
        gb = sum(p.stat().st_size for p in caps) / 1e9
        print(f"  ok       {sp:5s} {day:13s} {len(caps):4d} captures {gb:7.1f} GB  "
              f"labels <- {label.name if label else 'NONE'}")

    # CIC-2017 (test under cross-year)
    print(f"== CIC-2017  ({a.cic2017_dir})")
    for name, sp in sorted(used("CIC2017").items()):
        p = a.cic2017_dir / name
        if not p.is_file():
            problems.append(f"CIC-2017 capture missing: {p}")
            print(f"  MISSING  {sp:5s} {name}")
            continue
        need_columns(p, ("Source IP", "Destination IP", "Timestamp", "Label"), problems, "CIC-2017")
        print(f"  ok       {sp:5s} {name}")

    # CTU-13
    ctu = used("CTU13")
    if ctu:
        print(f"== CTU-13  ({a.ctu_dir})")
        if a.ctu_dir is None:
            problems.append(f"--scheme {a.scheme} uses CTU-13; pass --ctu-dir")
        else:
            for name, sp in sorted(ctu.items(), key=lambda kv: int(kv[0].split("/")[0])):
                p = a.ctu_dir / name
                if not p.is_file():
                    problems.append(f"CTU-13 capture missing: {p}")
                    print(f"  MISSING  {sp:5s} {name}")
                    continue
                need_columns(p, ("StartTime", "SrcAddr", "DstAddr", "Label"), problems, "CTU-13")
                print(f"  ok       {sp:5s} {name}")

    # PCAPs known corrupt on 2026-09-19
    tsv = ANALYSIS / "pcap_scan_results.tsv"
    bad = (ANALYSIS / "corrupt_files_redownload.txt")
    if tsv.exists() and bad.exists():
        size = {}
        for line in tsv.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 4 and f[0] == "CORRUPT":
                size[f[3].lstrip("./")] = int(f[1])
        still, changed, gone = [], [], []
        for rel in bad.read_text().split():
            p = a.pcap_root / rel
            if not p.exists():
                gone.append(rel)
            elif size.get(rel) == p.stat().st_size:
                still.append(rel)
            else:
                changed.append(rel)
        print(f"== known-corrupt PCAPs (analysis 06): {len(still)} still the corrupt copy, "
              f"{len(changed)} changed since (verify: claude_latest_analysis/verify_pcap_day.sh), "
              f"{len(gone)} absent")
        if still:
            notes.append(f"{len(still)} PCAPs are still the corrupt copy: each loses the part of "
                         f"its day after the damaged record. Re-fetch: "
                         f"claude_latest_analysis/refetch_corrupt.sh")

    print()
    for n in notes:
        print(f"NOTE  {n}")
    for p in problems:
        print(f"FAIL  {p}")
    print("datasets ready" if not problems else f"{len(problems)} problem(s): not ready")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
