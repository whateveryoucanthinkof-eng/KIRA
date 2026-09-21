"""Canonical on-disk locations of the three corpora.

One definition, imported by both `scripts/freeze_splits.py` (which writes the
frozen split) and `data_unification/split_manager.py` (which reads it). They
must agree: a capture named in the lock but unreachable at load time silently
shrinks a split.

Before this module existed, `split_manager.py` carried its own paths, and every
one of them was wrong on this machine -- including a literal Windows path
`C:\\SIH_DATA\\dump\\...`. The result was that all three splits returned ZERO
records, so the standalone Branch B and DeepOP trainers were training on
nothing at all, silently.

Paths can be overridden with environment variables so a different machine does
not need a code change.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Note the TRAILING SPACE in the CIC-2017 directory name -- it is real.
CIC2017_DIR = Path(os.environ.get(
    "CIC2017_DIR",
    os.path.expanduser("~/Documents/SIH/test/extracted_flows/TrafficLabelling "),
))

CIC2018_DIR = Path(os.environ.get(
    "CIC2018_DIR",
    os.path.expanduser("~/Documents/SIH/DATA/CSV"),
))

CTU13_DIR = Path(os.environ.get(
    "CTU13_DIR",
    os.path.expanduser("~/Documents/SIH/CTU-13-Dataset"),
))

PCAP2018_DIR = Path(os.environ.get(
    "PCAP2018_DIR",
    os.path.expanduser("~/Documents/SIH/DATA/pcap"),
))

GLOB = {
    "CIC2017": str(CIC2017_DIR / "*.csv"),
    "CIC2018": str(CIC2018_DIR / "*.csv"),
    "CTU13": str(CTU13_DIR / "*" / "*.binetflow"),
    "PCAP2018": str(PCAP2018_DIR / "*"),
}

ROOT = {
    "CIC2017": CIC2017_DIR,
    "CIC2018": CIC2018_DIR,
    "CTU13": CTU13_DIR,
    "PCAP2018": PCAP2018_DIR,
}


def resolve(dataset: str, capture: str) -> Path:
    """Absolute path of a capture named the way the frozen lock names it.

    CTU-13 captures are recorded as `<scenario>/<file>.binetflow`; the other
    corpora use a bare basename.
    """
    return ROOT[dataset] / capture


def missing_roots() -> dict:
    """{dataset: path} for every root that does not exist. Empty means healthy."""
    return {d: str(p) for d, p in ROOT.items() if not p.exists()}
