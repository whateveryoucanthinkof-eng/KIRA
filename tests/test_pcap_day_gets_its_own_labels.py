"""Each CIC-2018 PCAP day is labelled from its OWN day's CSV.

The 28 Feb capture (`wed_28_pcap`, the infiltration day) has its CSV named
`wed_29_csv.csv` in this corpus. The resolver fell back to the first
`wed_*_csv.csv` in sorted order -- `wed_14_csv.csv`, 14 Feb. Label intervals
are dated, so no 28 Feb packet fell inside a 14 Feb interval and the whole day
trained as benign, with no error anywhere.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_unification.training_sources import (
    PCAP_LABEL_ALIASES,
    _pcap_label_csv,
    check_label_day,
)

REPO = Path(__file__).resolve().parents[1]
DAYS = ["wed_14", "wed_21", "wed_29", "thu_15", "thu_22", "thu_1", "fri_16", "fri_23", "fri_2", "tue_20"]


@pytest.fixture
def label_dir(tmp_path):
    for d in DAYS:
        (tmp_path / f"{d}_csv.csv").write_text("Timestamp,Label\n")
    return tmp_path


@pytest.mark.parametrize("pcap_day,expected", [
    ("wed_28_pcap", "wed_29_csv.csv"),   # the alias, not wed_14
    ("wed_14_pcap", "wed_14_csv.csv"),
    ("thu_15_pacap", "thu_15_csv.csv"),  # misspelled day directory in the corpus
    ("fri_23_pacap", "fri_23_csv.csv"),
    ("thu_1_pcap", "thu_1_csv.csv"),
])
def test_each_pcap_day_resolves_to_its_own_csv(label_dir, pcap_day, expected):
    assert _pcap_label_csv(Path(pcap_day), label_dir).name == expected


def test_no_weekday_guessing(label_dir):
    # A day with no CSV and no alias gets nothing rather than another day's labels.
    assert _pcap_label_csv(Path("wed_7_pcap"), label_dir) is None
    assert PCAP_LABEL_ALIASES == {"wed_28": "wed_29"}


def _iv(y, m, d, h=14, n=100):
    ts = datetime(y, m, d, h, tzinfo=timezone.utc).timestamp()
    return SimpleNamespace(start_utc=ts, end_utc=ts + 600, n_rows=n)


def test_labels_from_another_day_are_refused():
    with pytest.raises(ValueError, match="another capture day"):
        check_label_day("wed_28", [_iv(2018, 2, 14), _iv(2018, 2, 14, 16)])


def test_labels_from_the_right_day_pass():
    check_label_day("wed_28", [_iv(2018, 2, 28), _iv(2018, 2, 28, 19)])
    check_label_day("thu_1", [_iv(2018, 3, 1)])
    # a handful of stray rows on another date do not outvote the day itself
    check_label_day("fri_16", [_iv(2018, 2, 16, n=5000), _iv(2018, 2, 17, n=3)])
    check_label_day("tue_20", [])


def test_both_pcap_readers_use_the_shared_resolver():
    src = (REPO / "scripts" / "retrain_future_models_live.py").read_text(encoding="utf-8")
    assert "_pcap_label_csv(day_dir, csv_label_dir)" in src
    assert 'glob(f"{prefix}_*_csv.csv")' not in src
    ts = (REPO / "data_unification" / "training_sources.py").read_text(encoding="utf-8")
    assert "check_label_day(day, dw.intervals)" in ts
