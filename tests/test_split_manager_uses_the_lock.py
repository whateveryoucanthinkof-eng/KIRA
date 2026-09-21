"""The frozen split must be the ONLY split, and it must actually resolve data.

Two failures this pins, both of which were live:

1. `splits.lock.json` had zero consumers. `ScientificSplitManager` carried its
   own hardcoded assignment that CONTRADICTED the lock -- it put CIC-2017
   Wednesday in train where the lock puts it in test, and Friday-Morning in val
   where the lock puts it in train. Had its paths worked, that was test leakage.

2. Its paths did not work. Every one was unreachable on this machine, including
   a literal Windows path. Each lookup was guarded by `os.path.exists`, so
   nothing raised and all three splits returned zero records -- the standalone
   Branch B and DeepOP trainers trained on nothing, silently.
"""

import pytest

from data_unification.split_manager import EmptySplitError, ScientificSplitManager
from data_unification.split_policy import load_lock

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def sm():
    return ScientificSplitManager()


def test_the_manager_reads_the_frozen_lock(sm):
    assert sm.assignment == load_lock()


def test_every_capture_lands_in_exactly_one_split(sm):
    seen = {}
    for split in SPLITS:
        for dataset, capture in sm.captures_in(split):
            key = f"{dataset}/{capture}"
            assert key not in seen, f"{key} is in both {seen.get(key)} and {split}"
            seen[key] = split
    assert seen, "no captures at all"


def test_splits_are_disjoint_and_cover_the_lock(sm):
    """No capture may be dropped on the floor either."""
    from_lock = {
        f"{ds}/{cap}"
        for ds, m in load_lock().items()
        if ds != "PCAP2018"
        for cap in m
    }
    from_manager = {
        f"{ds}/{cap}" for split in SPLITS for ds, cap in sm.captures_in(split)
    }
    assert from_manager == from_lock


def test_split_sizes_match_the_frozen_four_to_one_to_one(sm):
    """CIC2018 8/1/1, CIC2017 6/1/1, CTU13 9/2/2 -> 23/4/4."""
    assert len(sm.captures_in("train")) == 23
    assert len(sm.captures_in("val")) == 4
    assert len(sm.captures_in("test")) == 4


def test_no_capture_is_assigned_an_unknown_split(sm):
    for _ds, mapping in sm.assignment.items():
        for capture, split in mapping.items():
            assert split in SPLITS, f"{capture} has bogus split {split!r}"


def test_pcap_captures_are_frozen_but_not_served_as_csv(sm):
    """PCAP is in the lock, but it is consumed through pcap_bridge, not here."""
    assert "PCAP2018" in sm.assignment
    for split in SPLITS:
        assert not any(ds == "PCAP2018" for ds, _c in sm.captures_in(split))


def test_an_empty_split_raises_instead_of_returning_nothing(monkeypatch, sm):
    """Silently training on zero records is the failure this class must prevent."""
    m = ScientificSplitManager(strict=True)
    monkeypatch.setattr(m, "captures_in", lambda split: [])
    with pytest.raises(EmptySplitError):
        m.records_for("train")


def test_strict_false_downgrades_to_a_logged_error(monkeypatch):
    m = ScientificSplitManager(strict=False)
    monkeypatch.setattr(m, "captures_in", lambda split: [])
    assert m.records_for("train") == []


# ------------------------------------------------------------- prefix bias

def test_stride_beats_a_chronological_prefix_for_label_coverage(sm):
    """A prefix sample of the val split measured 0% attack: these captures are
    benign in the morning. Stride must recover positives."""
    prefix = sm.records_for("val", max_per_source=300, stride=1)
    strided = sm.records_for("val", max_per_source=300, stride=200)

    prefix_attacks = sum(1 for r in prefix if r.is_attack)
    strided_attacks = sum(1 for r in strided if r.is_attack)

    assert strided_attacks > prefix_attacks, (
        f"stride sampling found {strided_attacks} attacks vs {prefix_attacks} "
        f"for a prefix -- stride is supposed to span the whole capture"
    )


def test_a_prefix_capped_request_warns_about_bias(sm, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="data_unification.split_manager"):
        sm.records_for("val", max_per_source=50, stride=1)
    assert any("PREFIX" in r.message or "prefix" in r.message.lower()
               for r in caplog.records), "silent prefix sampling must warn"
