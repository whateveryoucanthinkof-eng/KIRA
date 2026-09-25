"""Trainers must resolve file paths through the frozen lock, never invent a split.

Before this, three scripts each rolled their own:

* `retrain_branch_a_live.py` sliced a SORTED file list 70/15/15, so "train"
  meant "alphabetically first" and the assignment moved whenever a file was
  added or renamed.
* `retrain_future_models_live.py` took an 80/20 slice of sorted CSVs, and two
  more 80/20 slices of sorted PCAP day directories.

None agreed with `splits.lock.json`, and none were reproducible across a change
to the corpus. Results were not comparable between retrains -- the exact thing
freezing a split is for.
"""

from pathlib import Path

import pytest

from data_unification.split_policy import (
    capture_name_for_path,
    load_lock,
    partition_paths,
    split_of,
    split_of_path,
)


def test_ctu13_paths_key_on_scenario_and_file():
    ds, cap = capture_name_for_path(Path("/data/CTU-13-Dataset/7/capture20110816-2.binetflow"))
    assert (ds, cap) == ("CTU13", "7/capture20110816-2.binetflow")


def test_cic2017_paths_key_on_basename():
    ds, cap = capture_name_for_path(Path("/x/TrafficLabelling /Monday-WorkingHours.pcap_ISCX.csv"))
    assert (ds, cap) == ("CIC2017", "Monday-WorkingHours.pcap_ISCX.csv")


def test_cic2018_paths_key_on_basename():
    ds, cap = capture_name_for_path(Path("/x/CSV/wed_14_csv.csv"))
    assert (ds, cap) == ("CIC2018", "wed_14_csv.csv")


def test_an_unmappable_path_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        capture_name_for_path(Path("/x/notes.txt"))


def test_split_of_path_agrees_with_the_lock():
    lock = load_lock()
    for capture, expected in lock["CIC2018"].items():
        assert split_of_path(Path("/anywhere/CSV") / capture) == expected
    for capture, expected in lock["CTU13"].items():
        assert split_of_path(Path("/anywhere/CTU-13-Dataset") / capture) == expected


def test_partition_covers_every_path_exactly_once():
    lock = load_lock()
    paths = [Path("/c") / n for n in lock["CIC2018"]]
    paths += [Path("/t") / n for n in lock["CTU13"]]
    part = partition_paths(paths)
    assert sum(len(v) for v in part.values()) == len(paths)
    flat = [p for v in part.values() for p in v]
    assert len(set(flat)) == len(flat)


def test_partition_matches_the_frozen_counts():
    """CIC2018 8/1/1 + CTU13 9/2/2 = 17/3/3."""
    lock = load_lock()
    paths = [Path("/c") / n for n in lock["CIC2018"]]
    paths += [Path("/t") / n for n in lock["CTU13"]]
    part = partition_paths(paths)
    assert (len(part["train"]), len(part["val"]), len(part["test"])) == (17, 3, 3)


def test_an_unknown_capture_is_an_error_not_a_silent_drop():
    """Dropping it would shrink a split without saying so; assigning it would
    make the split unreproducible."""
    with pytest.raises(KeyError):
        partition_paths([Path("/c/brand_new_day_csv.csv")])


def test_partition_never_places_a_capture_in_two_splits():
    lock = load_lock()
    paths = [Path("/c") / n for n in lock["CIC2018"]]
    part = partition_paths(paths)
    names = [[p.name for p in v] for v in part.values()]
    assert not (set(names[0]) & set(names[1]))
    assert not (set(names[0]) & set(names[2]))
    assert not (set(names[1]) & set(names[2]))


# --------------------------------------------------------------- PCAP days

def test_pcap_day_split_handles_the_misspelled_directories():
    """Two day dirs are `_pacap`, not `_pcap`, in the corpus itself. A naive
    endswith("_pcap") silently drops them."""
    assert split_of("PCAP2018", "fri_23_pacap") == "train"
    assert split_of("PCAP2018", "thu_15_pacap") == "train"


def test_pcap_days_are_frozen_eight_one_one():
    lock = load_lock()["PCAP2018"]
    counts = {s: sum(1 for v in lock.values() if v == s) for s in ("train", "val", "test")}
    assert counts == {"train": 8, "val": 1, "test": 1}


def test_wed_28_pcap_is_named_despite_having_no_matching_csv_day():
    """The PCAP day is wed_28 while its CSV is wed_29_csv.csv -- a real corpus
    quirk that must not drop the day from the split."""
    assert split_of("PCAP2018", "wed_28_pcap") in ("train", "val", "test")
