"""The single, frozen train/val/test assignment for every model in this repo.

Why this file exists
--------------------
Three trainers previously used three different, independently-computed splits
(Branch A: file-disjoint 70/15/15; Branch B/DeepOP: day-disjoint 80/20 with no
test at all; TGNE: chronological quantiles plus an inductive node holdout).
Numbers from different branches were therefore not comparable, and any change
to file ordering silently reshuffled what "test" meant. This freezes one
assignment so results stay comparable and nothing has to be retrained because
the split moved.

Design decisions, and the evidence behind them
----------------------------------------------
* **The split unit is a capture** (a CSV day, a CTU-13 scenario, a PCAP day).
  Flows inside one capture share hosts, timing and topology, so splitting below
  capture level leaks: the audit's D5 notes that stride-1 sliding windows make
  "samples" that are not independent. Capture-level assignment is the coarsest
  unit that removes that leak.

* **Test captures are held out entirely** -- never trained on, never used for
  checkpoint selection, scored once. Branch A/B/DeepOP previously reported
  validation numbers from the split used to pick the checkpoint.

* **Every split must contain attacks.** Measured attack fractions span 0.00%
  (CIC-2017 Monday) to 100% (all CTU-13). Assigning by name or at random can
  put an all-benign capture in test, which is the extreme-base-rate condition
  that makes PR-AUC ~1.0 for any ranking -- the exact failure that invalidated
  three earlier runs. Assignment below is therefore stratified by attack
  fraction: captures are ordered by attack rate and dealt round-robin, so each
  split gets a mix of quiet and busy captures.

* **CTU-13 is split by scenario**, which holds out whole botnet families. That
  is a harder and more honest generalisation test than mixing scenarios.

* **Chronology is preserved where it is meaningful.** Within CIC-2018 and
  CIC-2017 the later captures go to test where stratification allows, so the
  model is not trained on the future and tested on the past.

Changing any assignment here invalidates comparability with previously reported
numbers. Treat it as frozen; add new captures rather than reshuffling old ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal

Split = Literal["train", "val", "test"]

POLICY_VERSION = "1.0.0"
_LOCK = Path(__file__).with_name("splits.lock.json")


def load_lock() -> Dict[str, Dict[str, str]]:
    """{dataset: {capture_name: split}} as frozen on disk."""
    if not _LOCK.exists():
        raise FileNotFoundError(
            f"{_LOCK} missing -- run scripts/freeze_splits.py to create it once.")
    doc = json.loads(_LOCK.read_text())
    return doc["assignment"]


def split_of(dataset: str, capture_name: str) -> Split:
    a = load_lock()
    try:
        return a[dataset][capture_name]  # type: ignore[return-value]
    except KeyError:
        raise KeyError(
            f"{dataset}/{capture_name} is not in the frozen split. Add it to "
            f"{_LOCK} rather than assigning it on the fly.")


def captures_for(split: Split, dataset: str | None = None) -> List[str]:
    a = load_lock()
    out: List[str] = []
    for ds, m in a.items():
        if dataset and ds != dataset:
            continue
        out.extend(name for name, s in m.items() if s == split)
    return sorted(out)


# ---------------------------------------------------------------------------
# Path -> split resolution
# ---------------------------------------------------------------------------
#
# Trainers hold filesystem paths, not the capture names the lock records. They
# were each inventing their own split instead (scripts/retrain_branch_a_live.py
# sliced a SORTED file list 70/15/15, so "train" meant "alphabetically first").
# These helpers let every trainer ask the lock directly.


def capture_name_for_path(path) -> tuple:
    """(dataset, capture_name) for a corpus file, named as the lock names it.

    CTU-13 captures are keyed `<scenario>/<file>.binetflow`; CIC files by
    basename. Raises rather than guessing when a path matches nothing.
    """
    from pathlib import Path as _P

    p = _P(path)
    if p.suffix == ".binetflow":
        return "CTU13", f"{p.parent.name}/{p.name}"
    if p.name.endswith("pcap_ISCX.csv"):
        return "CIC2017", p.name
    if p.suffix == ".csv":
        return "CIC2018", p.name
    raise ValueError(f"cannot map {path} to a corpus the frozen split knows")


def split_of_path(path) -> Split:
    dataset, capture = capture_name_for_path(path)
    return split_of(dataset, capture)


def partition_paths(paths) -> Dict[str, List]:
    """Group real file paths into {'train': [...], 'val': [...], 'test': [...]}
    according to the frozen lock.

    Any path the lock does not name is an error: silently dropping it would
    shrink a split without saying so, and silently assigning it would make the
    split unreproducible.
    """
    out: Dict[str, List] = {"train": [], "val": [], "test": []}
    unknown = []
    for p in paths:
        try:
            out[split_of_path(p)].append(p)
        except (KeyError, ValueError):
            unknown.append(str(p))
    if unknown:
        raise KeyError(
            f"{len(unknown)} file(s) are not in the frozen split: {unknown[:5]}"
            f"{' ...' if len(unknown) > 5 else ''}. Add them to splits.lock.json "
            f"via scripts/freeze_splits.py rather than assigning them on the fly."
        )
    return out
