"""The frozen capture split must never change.

Two splits exist and are easy to confuse. This file guards the one that is
locked: which whole captures are train/val/test for the pipeline. The encoder's
internal sub-split (bita/train.py::split_data) operates entirely INSIDE the
locked train captures and is a different thing.

Changing the lock invalidates comparability with every previously reported
number, which is the entire reason it was frozen.
"""

import json
from pathlib import Path

from data_unification.split_policy import load_lock

LOCK = Path("data_unification/splits.lock.json")

# Recorded when the split was frozen, 2026-09-21T04:55:11Z.
EXPECTED = {
    "CIC2017": {"train": 6, "val": 1, "test": 1},
    "CIC2018": {"train": 8, "val": 1, "test": 1},
    "CTU13": {"train": 9, "val": 2, "test": 2},
    "PCAP2018": {"train": 8, "val": 1, "test": 1},
}

HELD_OUT = {
    "CIC2017": {"Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv"},
    "CIC2018": {"wed_29_csv.csv", "fri_2_csv.csv"},
}


def test_the_frozen_timestamp_has_not_moved():
    doc = json.loads(LOCK.read_text())
    assert doc["frozen_utc"].startswith("2026-09-21T04:55:11"), (
        "splits.lock.json was re-frozen; previously reported numbers are no "
        "longer comparable"
    )


def test_per_dataset_counts_match_what_was_frozen():
    lock = load_lock()
    for ds, expected in EXPECTED.items():
        got = {s: sum(1 for v in lock[ds].values() if v == s)
               for s in ("train", "val", "test")}
        assert got == expected, f"{ds} split changed: {got} != {expected}"


def test_the_specific_held_out_captures_are_still_held_out():
    """Naming them, so a reshuffle that preserved the counts still fails."""
    lock = load_lock()
    for ds, names in HELD_OUT.items():
        for name in names:
            assert lock[ds][name] in ("val", "test"), (
                f"{ds}/{name} moved into training -- it is held-out data"
            )


def test_no_capture_is_unassigned():
    for ds, mapping in load_lock().items():
        for cap, split in mapping.items():
            assert split in ("train", "val", "test"), f"{ds}/{cap} -> {split!r}"
