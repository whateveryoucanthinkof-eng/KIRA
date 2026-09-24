"""Which captures a trainer reads, from where, and how -- one implementation.

Branch A, Branch B/DeepOP and the encoder each used to discover and read their
own inputs. Only the Branch B/DeepOP trainer could read CIC-2018 from PCAP, and
Branch A read every *.csv with the CIC-2018 parser, CIC-2017 files included.

Why PCAP for CIC-2018
---------------------
Nine of the ten CIC-2018 CSV days carry host addresses fabricated from the row
number (claude_latest_analysis/08_why_the_csv_path_cannot_benchmark.md). A
host graph built from them is synthetic, so any encoder or trajectory model
trained on it learns a graph that never existed. The PCAPs carry the real
addresses; the paired CSV supplies only the attack windows, applied by
timestamp (attack_windows.derive_windows). Under the cross_year scheme CIC-2018
CSVs are refused unless explicitly allowed.

Why CIC-2017 needs its own parser
---------------------------------
Its CSVs use different column names and a 12-hour clock without AM/PM that
data_unification/cic2017_adapter.py repairs. Parsing them with the CIC-2018
adapter silently misreads timestamps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data_unification.split_policy import SCHEMES, capture_name_for_path, split_of

#: PCAP day directories: "<day>_pcap". Two days are misspelled "_pacap" in the
#: corpus itself (fri_23_pacap, thu_15_pacap).
_PCAP_DAY = re.compile(r"^(?P<day>.+)_pa?cap$")


@dataclass(frozen=True)
class Capture:
    dataset: str        # CIC2017 | CIC2018 | CTU13 | PCAP2018
    name: str           # the capture's key in splits.lock.json
    path: Path
    split: str          # train | val | test, under the scheme it was discovered with

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.name}"


class CorpusRefused(ValueError):
    pass


def discover_captures(
    *,
    scheme: str = "frozen",
    cic2017_dir: Optional[Path] = None,
    cic2018_dir: Optional[Path] = None,
    ctu13_dir: Optional[Path] = None,
    pcap2018_root: Optional[Path] = None,
    allow_cic2018_csv: bool = False,
) -> Dict[str, List[Capture]]:
    """{split: [Capture, ...]} for every capture the scheme uses.

    A capture the lock does not name is an error, as in split_policy: silently
    dropping or assigning it would change a split without saying so.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; expected one of {SCHEMES}")
    if cic2018_dir and pcap2018_root:
        raise CorpusRefused(
            "pass CIC-2018 as --pcap-root (real host addresses) OR as CSVs, not both: "
            "the same days would be read twice")
    if scheme == "cross_year" and cic2018_dir and not allow_cic2018_csv:
        raise CorpusRefused(
            "cross_year trains on CIC-2018, and 9 of its 10 CSV days fabricate host IPs "
            "from the row number, so the host graph would be synthetic. Use the PCAPs "
            "(--pcap-root with the CSV label directory), or pass --allow-cic2018-csv to "
            "accept that knowingly.")

    found: List[Tuple[str, str, Path]] = []
    if cic2017_dir:
        found += [("CIC2017", p.name, p) for p in sorted(Path(cic2017_dir).glob("*.csv"))]
    if cic2018_dir:
        found += [("CIC2018", p.name, p) for p in sorted(Path(cic2018_dir).glob("*.csv"))]
    if ctu13_dir:
        found += [("CTU13", f"{p.parent.name}/{p.name}", p)
                  for p in sorted(Path(ctu13_dir).glob("*/*.binetflow"))]
    if pcap2018_root:
        for p in sorted(Path(pcap2018_root).iterdir()):
            if p.is_dir() and _PCAP_DAY.match(p.name):
                found.append(("PCAP2018", p.name, p))

    out: Dict[str, List[Capture]] = {"train": [], "val": [], "test": []}
    unknown = []
    for dataset, name, path in found:
        if dataset in ("CIC2017", "CIC2018", "CTU13"):
            # Cross-check the naming against the lock's own mapping.
            dataset, name = capture_name_for_path(path)
        try:
            sp = split_of(dataset, name, scheme)
        except KeyError:
            unknown.append(f"{dataset}/{name}")
            continue
        if sp is not None:
            out[sp].append(Capture(dataset, name, path, sp))
    if unknown:
        raise KeyError(
            f"{len(unknown)} capture(s) are not in the frozen split: {unknown[:5]}. "
            f"Add them via scripts/freeze_splits.py rather than assigning them on the fly.")
    return out


def _pcap_label_csv(day_dir: Path, label_dir: Path) -> Optional[Path]:
    day = _PCAP_DAY.match(day_dir.name).group("day")
    csv_path = Path(label_dir) / f"{day}_csv.csv"
    if csv_path.exists():
        return csv_path
    # The PCAP and CSV corpora name one day differently ("wed_28" vs "wed_29",
    # verified in the corpus): fall back to the weekday prefix.
    prefix = day.rsplit("_", 1)[0]
    cands = sorted(Path(label_dir).glob(f"{prefix}_*_csv.csv"))
    return cands[0] if cands else None


def iter_pcap_day_windows(day_dir: Path, label_dir: Path, window_seconds: float,
                          max_windows: Optional[int] = None, window_stride: int = 1,
                          max_packets_per_host: Optional[int] = None) -> Iterator[list]:
    """Per-window record lists for one PCAP day, labelled from its CSV."""
    from data_unification.attack_windows import derive_windows
    from data_unification.pcap_bridge import iter_day_records

    day_dir = Path(day_dir)
    csv_path = _pcap_label_csv(day_dir, label_dir)
    if csv_path is None:
        print(f"skipping {day_dir.name}: no label CSV under {label_dir}", flush=True)
        return
    dw = derive_windows(str(csv_path))
    if not dw.ok:
        print(f"skipping {day_dir.name}: implausible attack windows ({dw.evidence})", flush=True)
        return
    day = _PCAP_DAY.match(day_dir.name).group("day")
    n = 0
    for w_idx, (_s, _e, recs) in enumerate(iter_day_records(
            day_dir, dw, scenario_id=day, window_seconds=window_seconds,
            max_packets_per_host=max_packets_per_host)):
        if window_stride > 1 and (w_idx % window_stride):
            continue
        yield recs
        n += 1
        if max_windows is not None and n >= max_windows:
            break


def read_capture(
    cap: Capture,
    *,
    window_seconds: float,
    pcap_label_dir: Optional[Path] = None,
    rows_per_file: Optional[int] = None,
    stride: int = 1,
    pcap_max_windows: Optional[int] = None,
    pcap_window_stride: int = 1,
    coverage: Optional[list] = None,
) -> list:
    """All records of one capture, unmappable labels removed."""
    from data_unification.label_filter import drop_unresolved

    if cap.dataset == "PCAP2018":
        if pcap_label_dir is None:
            raise ValueError("PCAP captures need the CIC-2018 CSV label directory")
        recs: list = []
        for window in iter_pcap_day_windows(cap.path, pcap_label_dir, window_seconds,
                                            pcap_max_windows, pcap_window_stride):
            recs.extend(window)
    else:
        cap_rows = None if rows_per_file is None else rows_per_file * stride
        if cap.dataset == "CIC2017":
            from data_unification.cic2017_adapter import CIC2017Adapter
            gen = CIC2017Adapter().parse_file(str(cap.path), max_rows=cap_rows)
        elif cap.dataset == "CIC2018":
            from data_unification.cic2018_adapter import CIC2018Adapter
            gen = CIC2018Adapter().parse_file(str(cap.path), max_rows=cap_rows)
        else:
            from data_unification.ctu13_adapter import CTU13Adapter
            gen = CTU13Adapter().parse_netflow_csv(str(cap.path), max_rows=cap_rows)
        recs = _strided(gen, stride, rows_per_file)
    recs, cov = drop_unresolved(recs)
    if coverage is not None:
        coverage.append(cov)
    return recs


def _strided(gen, stride: int, want=None) -> list:
    """Every `stride`-th record, at most `want` of them. `want=None` = all.

    None must mean "no cap": full density is the default, and treating None as
    0 would silently return nothing (this crashed the Branch A retrain twice).
    """
    out = []
    for i, r in enumerate(gen):
        if i % stride == 0:
            out.append(r)
            if want is not None and len(out) >= want:
                break
    return out


def describe(captures: Dict[str, List[Capture]]) -> str:
    parts = []
    for sp in ("train", "val", "test"):
        by_ds: Dict[str, int] = {}
        for c in captures[sp]:
            by_ds[c.dataset] = by_ds.get(c.dataset, 0) + 1
        parts.append(f"{sp}: " + (", ".join(f"{n} {d}" for d, n in sorted(by_ds.items())) or "none"))
    return " | ".join(parts)
