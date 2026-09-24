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
from typing import Dict, List, Literal, Optional

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


# ---------------------------------------------------------------------------
# Split schemes
# ---------------------------------------------------------------------------
#
# "frozen"      the lock above, unchanged: every corpus split 4:1:1 by capture.
#
# "cross_year"  train on CIC-IDS-2018, test on CIC-IDS-2017. A DERIVED view of
#               the lock, not a second lock, so it is reproducible from the
#               same file:
#                 * CIC-2018 (CSV or PCAP): the lock's train days -> train; its
#                   val AND test days -> val. Every tuning decision (early
#                   stopping, alert threshold, temperature, conformal width)
#                   is therefore made on 2018 alone.
#                 * CIC-2017: every capture -> test, scored once, after the
#                   model is frozen. Nothing trained or tuned has seen 2017.
#                 * CTU-13: excluded (None).
#               A different year, network (AWS 172.31/16 vs lab 192.168.10/24),
#               toolset and CICFlowMeter build: a generalisation test, not an
#               in-distribution one. Expect lower numbers than "frozen", and
#               read them as the honest ones.
SCHEMES = ("frozen", "cross_year")


def _scheme_split(dataset: str, locked: str, scheme: str) -> Optional[Split]:
    if scheme == "frozen":
        return locked  # type: ignore[return-value]
    if scheme == "cross_year":
        if dataset == "CIC2017":
            return "test"
        if dataset in ("CIC2018", "PCAP2018"):
            return "train" if locked == "train" else "val"
        return None
    raise ValueError(f"unknown split scheme {scheme!r}; expected one of {SCHEMES}")


def split_of(dataset: str, capture_name: str, scheme: str = "frozen") -> Optional[Split]:
    """The split a capture belongs to under `scheme`; None = not used at all."""
    a = load_lock()
    try:
        locked = a[dataset][capture_name]
    except KeyError:
        raise KeyError(
            f"{dataset}/{capture_name} is not in the frozen split. Add it to "
            f"{_LOCK} rather than assigning it on the fly.")
    return _scheme_split(dataset, locked, scheme)


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


def split_of_path(path, scheme: str = "frozen") -> Optional[Split]:
    dataset, capture = capture_name_for_path(path)
    return split_of(dataset, capture, scheme)


def partition_paths(paths, scheme: str = "frozen") -> Dict[str, List]:
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
            sp = split_of_path(p, scheme)
        except (KeyError, ValueError):
            unknown.append(str(p))
            continue
        if sp is not None:          # None: this scheme does not use the corpus
            out[sp].append(p)
    if unknown:
        raise KeyError(
            f"{len(unknown)} file(s) are not in the frozen split: {unknown[:5]}"
            f"{' ...' if len(unknown) > 5 else ''}. Add them to splits.lock.json "
            f"via scripts/freeze_splits.py rather than assigning them on the fly."
        )
    return out


# ---------------------------------------------------------------------------
# The two splits, and which one is frozen
# ---------------------------------------------------------------------------
#
# There are two, they are easy to confuse, and only one of them is this file's
# lock:
#
# 1. THE FROZEN CAPTURE SPLIT -- splits.lock.json, above.
#    Which whole captures are train / val / test for the PIPELINE.
#    CIC-2018 8/1/1, CIC-2017 6/1/1, CTU-13 9/2/2, PCAP 8/1/1.
#    Frozen 2026-09-21T04:55:11Z. **Never changes.** Changing it invalidates
#    comparability with every number previously reported.
#
# 2. THE ENCODER'S INTERNAL SUB-SPLIT -- bita/train.py::split_data.
#    TGNE trains only on the 23 captures that (1) assigns to train, and needs
#    its own train/val/test inside them for early stopping and for inductive
#    (unseen-host) evaluation. It cannot reach the captures (1) holds out --
#    they are never loaded, which `--train_splits train` enforces and the
#    load log states ("N captures read, 8 held out from the encoder").
#
# Policy (2) is now FIXED as well, after three iterations that each starved a
# class in a different way:
#
#   global time quantile  -> split by CORPUS (disjoint years), 3 of 5 classes
#                            absent from training
#   per-corpus quantile   -> Recon 7 samples: CIC-2017's cut fell at 13:13 and
#                            all PortScan runs 13:00-15:59
#   per-capture quantile  -> Recon 0 samples: its single carrier host
#                            (172.16.0.1) was held out by the inductive draw
#
# SETTLED POLICY: cut each CAPTURE at its own 70/85 time quantiles, then
# guarantee no class is deleted by the inductive node draw. Rationale: it is
# the finest grain that still preserves causality (train precedes val precedes
# test within every capture) while letting every attack class appear on all
# three sides. Do not change it again without a measured reason recorded here.

ENCODER_SUBSPLIT_POLICY = {
    "granularity": "capture",
    "quantiles": (0.70, 0.85),
    "guard": "no class may be emptied from train by the inductive node draw",
    "settled": "2026-09-21",
}
