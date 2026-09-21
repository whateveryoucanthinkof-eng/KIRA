"""Epoch-second conversion that does not depend on pandas' parsed resolution.

Every adapter previously did `series.astype("int64") / 1e9`, which assumes the
parsed dtype is datetime64[ns]. pandas >= 2 picks the smallest sufficient unit,
and on this corpus returns datetime64[us] -- so that expression produced epoch
seconds 1000x too small. A 12-hour capture day collapsed to 43 apparent
seconds, turning ~21,600 two-second windows into ~22 and making every host
trajectory meaningless. The CIC-2017 adapter additionally parses with utc=True,
which yields a timezone-aware dtype that cannot be cast straight to
datetime64[ns] at all; its except-branch silently zeroed every timestamp.

One helper, used by all adapters, so the assumption lives in a single place.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def to_epoch_seconds(series: pd.Series, *, nan: float = 0.0) -> np.ndarray:
    """Epoch seconds as float64, for tz-aware or tz-naive datetime series.

    NaT is detected with isna() *before* the int64 cast. Casting NaT yields
    -9223372036854775808, which becomes -9.223e9 after dividing -- a plausible
    looking negative epoch that slipped past a numeric threshold guard. One
    real CIC-2017 file (Thursday WebAttacks) is 63% blank padding rows, and
    those produced 288,602 records at that sentinel timestamp, which would
    stretch the window grid from 1677 AD to 2017.
    """
    s = pd.Series(series)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    missing = s.isna().to_numpy()
    out = s.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    out = np.where(missing, nan, out)
    return np.nan_to_num(out, nan=nan, posinf=nan, neginf=nan)


# ---------------------------------------------------------------------------
# 12-hour clock repair
# ---------------------------------------------------------------------------
#
# Every CIC-2018 and CIC-2017 CSV records its Timestamp on a 12-hour dial with
# no AM/PM marker. Verified over all 16,233,002 CIC-2018 rows: hours 13-23
# appear ZERO times, and hour 00 also appears zero times -- a 12-hour dial
# writes noon/midnight as `12`, never `00`, so the absence of `00` is the
# confirming signature rather than a coincidence.
#
# The capture days run roughly 08:00-18:00 local, so the recorded hours are
# 08,09,10,11,12 (morning) and 01,02,03,04,05 (= 13:00-17:00). Parsed naively,
# every afternoon flow lands twelve hours BEFORE the morning, which is why
# per-file backward-time-step counts run to hundreds of thousands.
#
# This damage is upstream, baked into the published distribution -- confirmed by
# mtimes untouched since the 2018 release and by row totals matching the
# published 16,233,002 exactly. Re-downloading reproduces it byte for byte, so
# the repair has to live here.
#
# Note the mapping is INJECTIVE on this corpus: because no real traffic occurs
# in 01:00-07:59 local, no repaired timestamp collides with an existing one.
# Window *contents* were therefore always correct; what was wrong was ordering,
# inter-window time deltas, and any PCAP<->CSV alignment (PCAP carries true
# epoch time, so the afternoon half was misaligned by exactly 12 hours).

TWELVE_HOURS_SECONDS = 12 * 3600.0

#: Hours a 12-hour dial can never emit. Seeing any of these proves a 24h clock.
_IMPOSSIBLE_ON_12H_DIAL = frozenset(range(13, 24)) | {0}

#: Hours that must be shifted forward when the dial is confirmed 12-hour.
_PM_HOURS = frozenset(range(1, 8))


def hour_of_day(epoch_seconds: np.ndarray) -> np.ndarray:
    """UTC hour-of-day for each epoch second. Naive local times parse as UTC,
    so this returns the hour exactly as it was written in the CSV."""
    return ((np.asarray(epoch_seconds, dtype=np.float64) // 3600) % 24).astype(np.int64)


def dial_needs_repair(present_hours, n_valid: int, *, min_valid_rows: int = 500) -> bool:
    """Verdict from the SET of hours observed, pooled over however much data the
    caller could see.

    Deliberately conservative -- it must never "repair" a genuine 24-hour clock:

      * any hour in 13..23, or hour 00  -> a real 24h clock, refuse.
      * no hour in 01..07               -> nothing to repair, refuse.
      * no hour in 08..12               -> a 12-hour dial and a genuine
                                            early-morning capture are
                                            indistinguishable, so refuse.
      * too few rows to judge           -> refuse.

    The third rule is why detection must pool a whole capture DAY, not one file.
    CIC-2017 splits Thursday and Friday into morning/afternoon files: the
    afternoon files hold only hours 01-05 and are individually ambiguous, but
    pooled with their morning sibling the day spans 01-05 and 08-12 and the
    verdict is decisive.
    """
    present = set(int(h) for h in present_hours)
    if n_valid < min_valid_rows:
        return False
    if present & _IMPOSSIBLE_ON_12H_DIAL:
        return False
    if not (present & _PM_HOURS):
        return False
    if not (present & frozenset(range(8, 13))):
        return False
    return True


def looks_like_12h_clock(epoch_seconds: np.ndarray, *, min_valid_rows: int = 500) -> bool:
    """`dial_needs_repair` over one array of epoch seconds."""
    e = np.asarray(epoch_seconds, dtype=np.float64)
    valid = e > 0
    n = int(valid.sum())
    if n == 0:
        return False
    return dial_needs_repair(np.unique(hour_of_day(e[valid])).tolist(), n,
                             min_valid_rows=min_valid_rows)


def repair_12h_clock(epoch_seconds: np.ndarray) -> np.ndarray:
    """Shift 01:00-07:59 forward twelve hours. Caller must have confirmed the
    dial over the whole capture day, never one chunk -- a single chunk can
    legitimately hold only afternoon rows."""
    e = np.asarray(epoch_seconds, dtype=np.float64).copy()
    shift = (e > 0) & np.isin(hour_of_day(e), list(_PM_HOURS))
    e[shift] += TWELVE_HOURS_SECONDS
    return e


def hours_present_in_file(
    filepath: str,
    ts_col: str,
    *,
    encoding: str = "latin1",
    dayfirst: bool = True,
    utc: bool = False,
):
    """(set_of_hours, n_valid_rows) from one cheap single-column read."""
    # Read in CHUNKS. `low_memory=False` was catastrophic here: it forces pandas
    # to parse the whole file as one block, and the C tokenizer buffers every
    # column before `usecols` is applied. On tue_20 (7.9M rows x 84 columns)
    # this single call peaked at 13.91 GiB -- it was the entire reason the TGNE
    # loader was being OOM-killed, not the records it was accumulating.
    #
    # Only the SET of hours and a valid-row count are needed, and both compose
    # across chunks, so peak is now one chunk.
    hours: set = set()
    n_valid = 0
    try:
        # dtype=str is required, not cosmetic. Without it pandas infers dtypes
        # per chunk, and a column with mixed types -- which the 288,602 blank
        # padding rows in CIC-2017 Thursday-Morning-WebAttacks guarantee --
        # sends it down the DtypeWarning path, where `usecols` + `chunksize`
        # together hit a pandas bug: _concatenate_chunks indexes column_names
        # by the ORIGINAL column position while that list holds only the
        # selected column, raising IndexError. Reading the column as text
        # skips inference entirely, and we parse it as a date ourselves anyway.
        reader = pd.read_csv(
            filepath, usecols=[ts_col], encoding=encoding,
            dtype={ts_col: str}, chunksize=500_000,
        )
        for chunk in reader:
            parsed = pd.to_datetime(chunk[ts_col], dayfirst=dayfirst, utc=utc,
                                    errors="coerce")
            e = to_epoch_seconds(parsed)
            valid = e > 0
            k = int(valid.sum())
            if k:
                n_valid += k
                hours |= set(np.unique(hour_of_day(e[valid])).tolist())
            # Once a 24-hour clock is proven there is nothing left to learn.
            if hours & _IMPOSSIBLE_ON_12H_DIAL:
                break
    except Exception as exc:
        # NEVER fail silently here. A swallowed error returns "no hours", which
        # reads as "not a 12-hour clock", which leaves a genuinely broken file
        # unrepaired -- the exact silent-failure shape of the original bug this
        # module exists to fix (a bare `except` that zeroed every timestamp).
        logging.getLogger(__name__).error(
            "clock detection failed on %s (%s: %s); the file will NOT be "
            "repaired -- investigate rather than ignoring this",
            filepath, type(exc).__name__, exc,
        )
        return set(), 0
    return hours, n_valid


def detect_12h_clock_in_file(
    filepath: str,
    ts_col: str,
    *,
    encoding: str = "latin1",
    dayfirst: bool = True,
    utc: bool = False,
) -> bool:
    """Decide the dial from a single file. Correct for a file spanning a whole
    capture day; for morning/afternoon-split corpora use
    `detect_12h_clock_in_group` instead, or afternoon-only files come back
    ambiguous and go unrepaired."""
    hours, n = hours_present_in_file(filepath, ts_col, encoding=encoding,
                                     dayfirst=dayfirst, utc=utc)
    return dial_needs_repair(hours, n)


def detect_12h_clock_in_group(
    files_and_cols,
    *,
    encoding: str = "latin1",
    dayfirst: bool = True,
    utc: bool = False,
) -> bool:
    """Pool the hours of every file in one capture day, then decide once.

    `files_and_cols` is an iterable of (filepath, ts_col). The verdict applies
    to every file in the group -- they are one capture split across files, so
    they share a dial.
    """
    pooled, total = set(), 0
    for path, col in files_and_cols:
        hours, n = hours_present_in_file(path, col, encoding=encoding,
                                         dayfirst=dayfirst, utc=utc)
        pooled |= hours
        total += n
    return dial_needs_repair(pooled, total)
