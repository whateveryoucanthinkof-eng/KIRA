"""Clock detection must stream, and must not fail silently.

Two regressions this pins, both introduced while fixing the 12-hour clock:

1. `pd.read_csv(..., usecols=[ts], low_memory=False)` forces pandas to parse
   the whole file as one block; the C tokenizer buffers every column before
   `usecols` is applied. On tue_20 (7.9M rows x 84 cols) that single call
   peaked at **13.91 GiB** and was the sole reason the TGNE loader kept being
   OOM-killed. Chunked, the same call peaks at 0.23 GiB with an identical
   verdict, and the whole CIC-2018 loader went from 13.75 GiB to 1.16 GiB.

2. Chunking alone then broke two CIC-2017 files. A column with mixed types --
   which the blank padding rows guarantee -- sends pandas down its
   DtypeWarning path, where `usecols` + `chunksize` hit a pandas bug
   (`_concatenate_chunks` indexes `column_names` by the ORIGINAL column
   position while that list holds only the selected column) and raise
   IndexError. A bare `except` turned that into "no hours", i.e. "not a
   12-hour clock", i.e. a genuinely broken file left unrepaired -- the exact
   silent-failure shape of the original bug this module exists to fix.
"""

import logging

import pandas as pd
import pytest

from data_unification.time_utils import (
    dial_needs_repair,
    detect_12h_clock_in_file,
    hours_present_in_file,
)

ROWS = 3000


def _write(tmp_path, name, rows, extra_cols=0, blanks=0, encoding="latin1"):
    """A CSV shaped like the corpus: a Timestamp column among many others."""
    data = {"Timestamp": rows}
    for c in range(extra_cols):
        data[f"c{c}"] = ["x"] * len(rows)
    df = pd.DataFrame(data)
    if blanks:
        pad = pd.DataFrame({k: [None] * blanks for k in df.columns})
        df = pd.concat([df, pad], ignore_index=True)
    p = tmp_path / name
    df.to_csv(p, index=False, encoding=encoding)
    return str(p)


def _twelve_hour_rows(n=ROWS):
    """Hours 01-05 and 08-12 only -- the real corpus signature."""
    hours = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
    return [f"14/02/2018 {hours[i % len(hours)]:02d}:{i % 60:02d}:00" for i in range(n)]


def test_detects_the_twelve_hour_signature(tmp_path):
    f = _write(tmp_path, "a.csv", _twelve_hour_rows())
    assert detect_12h_clock_in_file(f, "Timestamp")


def test_verdict_is_independent_of_chunk_boundaries(tmp_path):
    """The hour SET and the valid count both compose across chunks, so a file
    read in one chunk and in many must give the same answer."""
    f = _write(tmp_path, "b.csv", _twelve_hour_rows(ROWS))
    hours, n = hours_present_in_file(f, "Timestamp")
    assert n == ROWS
    assert hours == {1, 2, 3, 4, 5, 8, 9, 10, 11, 12}


def test_a_mixed_dtype_column_does_not_crash_detection(tmp_path):
    """The CIC-2017 Thursday regression: blank padding rows make the column
    mixed-dtype, which used to raise IndexError inside pandas and be swallowed."""
    f = _write(tmp_path, "c.csv", _twelve_hour_rows(), extra_cols=6, blanks=2000)
    hours, n = hours_present_in_file(f, "Timestamp")
    assert n == ROWS, "blank rows must be skipped, not abort the scan"
    assert hours == {1, 2, 3, 4, 5, 8, 9, 10, 11, 12}
    assert detect_12h_clock_in_file(f, "Timestamp") is True


def test_blank_padding_rows_do_not_count_as_valid(tmp_path):
    f = _write(tmp_path, "d.csv", _twelve_hour_rows(1000), blanks=5000)
    _hours, n = hours_present_in_file(f, "Timestamp")
    assert n == 1000


def test_a_genuine_24h_file_is_still_refused(tmp_path):
    rows = [f"14/02/2018 {h:02d}:{i % 60:02d}:00" for h in range(24) for i in range(60)]
    f = _write(tmp_path, "e.csv", rows)
    assert detect_12h_clock_in_file(f, "Timestamp") is False


def test_a_failure_is_logged_loudly_and_never_silent(tmp_path, caplog):
    """A swallowed error returns "no hours" -> "not 12-hour" -> file left
    unrepaired. That must be impossible to miss in a log."""
    missing = str(tmp_path / "does_not_exist.csv")
    with caplog.at_level(logging.ERROR, logger="data_unification.time_utils"):
        hours, n = hours_present_in_file(missing, "Timestamp")
    assert (hours, n) == (set(), 0)
    assert any("clock detection failed" in r.message for r in caplog.records), (
        "a detection failure must be logged at ERROR, not swallowed"
    )


def test_a_missing_column_is_reported_not_silently_false(tmp_path, caplog):
    f = _write(tmp_path, "g.csv", _twelve_hour_rows())
    with caplog.at_level(logging.ERROR, logger="data_unification.time_utils"):
        assert detect_12h_clock_in_file(f, "NoSuchColumn") is False
    assert any("clock detection failed" in r.message for r in caplog.records)


def test_wide_files_are_handled(tmp_path):
    """tue_20 is 84 columns; usecols must still isolate the timestamp."""
    f = _write(tmp_path, "h.csv", _twelve_hour_rows(), extra_cols=83)
    assert detect_12h_clock_in_file(f, "Timestamp") is True


def test_pooled_day_verdict_still_composes(tmp_path):
    """Afternoon-only and morning-only files are each ambiguous; together they
    are decisive. This is what rescues CIC-2017 Thursday and Friday."""
    assert not dial_needs_repair({1, 2, 3, 4, 5}, 10_000)
    assert not dial_needs_repair({8, 9, 10, 11, 12}, 10_000)
    assert dial_needs_repair({1, 2, 3, 4, 5} | {8, 9, 10, 11, 12}, 20_000)
