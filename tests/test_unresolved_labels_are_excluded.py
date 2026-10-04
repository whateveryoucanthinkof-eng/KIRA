"""Unmapped labels must not reach a loss function as confident negatives.

`label_resolver` returns UNKNOWN with `is_attack=False` for a label its maps do
not recognise, and its own comment says callers "must exclude these rows rather
than treat them as benign". Before `data_unification/label_filter.py` existed,
`is_unresolved` and `unresolved_report` had no call sites anywhere in the repo,
so every unmapped label was trained as a negative and counted as a true
negative at evaluation time.
"""

import pytest

from data_unification.label_filter import (
    count_unresolved,
    drop_unresolved,
    format_unresolved_report,
    partition_unresolved,
)
from data_unification.label_resolver import UNKNOWN_CATEGORY, LabelResolver
from data_unification.unified_schema import LabelSource, UnifiedFlowRecord


def _rec(raw_label, coarse, is_attack=False, source=LabelSource.CIC2018.value):
    return UnifiedFlowRecord(
        src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80, protocol=6,
        start_time=0.0, end_time=1.0, fwd_bytes=1, bwd_bytes=1,
        fwd_packets=1, bwd_packets=1,
        raw_label=raw_label, raw_label_source=source,
        is_attack=is_attack, coarse_category=coarse, attck_technique_ids=[],
    )


BENIGN = _rec("BENIGN", "Benign")
ATTACK = _rec("DDoS", "Impact", is_attack=True)
UNKNOWN = _rec("Some-New-Attack-2027", UNKNOWN_CATEGORY)


def test_the_resolver_really_does_emit_unknown_for_an_unmapped_label():
    """Guards the premise: if this changes, the filter below is pointless."""
    coarse, techs, is_attack = LabelResolver().resolve(
        "Some-New-Attack-2027", source=LabelSource.CIC2018.value)
    assert coarse == UNKNOWN_CATEGORY
    assert techs == []
    assert is_attack is False, "an unmapped label must not assert an attack either"


def test_unknown_records_are_dropped_not_kept_as_benign():
    kept, report = drop_unresolved([BENIGN, ATTACK, UNKNOWN])
    assert UNKNOWN not in kept
    assert kept == [BENIGN, ATTACK]
    assert report["unresolved_records"] == 1
    assert report["unresolved_rate"] == pytest.approx(1 / 3)


def test_partition_preserves_order_and_loses_nothing():
    recs = [BENIGN, UNKNOWN, ATTACK, UNKNOWN, BENIGN]
    good, bad = partition_unresolved(recs)
    assert good == [BENIGN, ATTACK, BENIGN]
    assert bad == [UNKNOWN, UNKNOWN]
    assert len(good) + len(bad) == len(recs)


def test_report_names_the_labels_that_failed_to_map():
    """A rate alone does not tell anyone which map entry to add."""
    report = count_unresolved([UNKNOWN, UNKNOWN, BENIGN])
    assert report["distinct_unresolved_labels"] == 1
    assert report["top_unresolved_labels"][0] == ("Some-New-Attack-2027", 2)
    assert report["unresolved_by_source"] == {LabelSource.CIC2018.value: 2}


def test_counting_does_not_drop_anything():
    recs = [BENIGN, UNKNOWN]
    count_unresolved(recs)
    assert len(recs) == 2


def test_empty_input_does_not_divide_by_zero():
    kept, report = drop_unresolved([])
    assert kept == []
    assert report["unresolved_rate"] == 0.0
    assert report["resolved_rate"] == 0.0


def test_a_high_unresolved_rate_is_stated_loudly():
    text = format_unresolved_report(count_unresolved([UNKNOWN] * 9 + [BENIGN]))
    assert "WARNING" in text
    assert "Some-New-Attack-2027" in text


def test_a_clean_corpus_reports_without_a_warning():
    text = format_unresolved_report(count_unresolved([BENIGN, ATTACK]), where="train")
    assert "WARNING" not in text
    assert "[train]" in text
    assert "2/2 resolved" in text


def test_the_filter_is_actually_wired_into_the_trainers():
    """The defect was never the helper -- it was that nothing called it."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    for script in ("data_unification/training_sources.py",):
        src = (repo / script).read_text(encoding="utf-8")
        assert "drop_unresolved" in src, f"{script} still trains on UNKNOWN as benign"
    # Branch A reads every capture through training_sources.read_capture.
    assert "read_capture(" in (repo / "scripts/retrain_branch_a_live.py").read_text(encoding="utf-8")
