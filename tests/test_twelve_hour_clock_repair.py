"""CIC-2018 and CIC-2017 timestamps are a 12-hour dial with no AM/PM.

Verified over all 16,233,002 CIC-2018 rows: hours 13-23 appear zero times, and
hour 00 also appears zero times -- a 12-hour dial writes noon as `12`, never
`00`, so the missing `00` is the confirming signature rather than a coincidence.
Unrepaired, every afternoon flow parses twelve hours BEFORE the morning.

The defect is upstream in the published distribution (file mtimes untouched
since the 2018 release; row totals match the published figure exactly), so
re-downloading reproduces it and the repair has to live in our code.

These tests pin both halves: the repair fires where it should, and -- more
important -- it refuses to fire anywhere it might corrupt a genuine 24h clock.
"""

import numpy as np
import pytest

from data_unification.time_utils import (
    dial_needs_repair,
    hour_of_day,
    looks_like_12h_clock,
    repair_12h_clock,
)

BASE = 1518566400.0  # 2018-02-14 00:00:00 UTC


def mk(hours, per_hour=60):
    """Epoch seconds covering `per_hour` minutes inside each listed hour."""
    return np.array([BASE + h * 3600 + m * 60 for h in hours for m in range(per_hour)],
                    dtype=np.float64)


# --------------------------------------------------------------------------
# It must fire on the real signature
# --------------------------------------------------------------------------

def test_detects_the_real_cic_signature():
    """Hours 01-05 and 08-12, nothing else -- exactly what all ten files show."""
    assert looks_like_12h_clock(mk([1, 2, 3, 4, 5, 8, 9, 10, 11, 12]))


def test_repair_turns_the_dial_into_a_real_working_day():
    before = mk([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
    after = repair_12h_clock(before)
    assert sorted(set(hour_of_day(after).tolist())) == [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def test_repair_shifts_exactly_twelve_hours_and_only_the_pm_half():
    before = mk([3, 10])
    after = repair_12h_clock(before)
    pm = hour_of_day(before) <= 7
    assert np.all(after[pm] - before[pm] == 12 * 3600.0)
    assert np.all(after[~pm] == before[~pm]), "morning rows must not move"


def test_repair_removes_the_systematic_backward_seam():
    """Afternoon sorted before morning is the whole defect; it must be gone."""
    morning, afternoon = mk([9]), mk([2])   # 02:00 is really 14:00
    before = np.concatenate([morning, afternoon])
    assert before[len(morning)] < before[0], "precondition: afternoon sorts first"
    after = repair_12h_clock(before)
    assert after[len(morning)] > after[len(morning) - 1], "afternoon must now follow morning"


def test_the_mapping_is_injective_on_this_corpus():
    """No real traffic occurs 01:00-07:59, so no repaired stamp collides."""
    e = mk([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
    r = repair_12h_clock(e)
    assert len(np.unique(r)) == len(np.unique(e)) == len(e)


def test_repair_is_idempotent():
    """A second application is a no-op, so a double-repair cannot corrupt data.

    This holds structurally rather than by accident: the shift maps 01-07 onto
    13-19, and 13-19 is outside the shift range, so nothing moves twice. It is
    the property that makes the repair safe to apply defensively.
    """
    once = repair_12h_clock(mk([1, 3, 5, 7]))
    twice = repair_12h_clock(once)
    assert np.array_equal(once, twice)


def test_a_repaired_column_no_longer_detects_as_twelve_hour():
    """Detection must not re-fire on already-repaired data."""
    repaired = repair_12h_clock(mk([1, 2, 3, 4, 5, 8, 9, 10, 11, 12]))
    assert not looks_like_12h_clock(repaired)


# --------------------------------------------------------------------------
# It must NOT fire anywhere else -- these are the dangerous cases
# --------------------------------------------------------------------------

def test_never_fires_on_a_genuine_24_hour_clock():
    assert not looks_like_12h_clock(mk(range(24), per_hour=30))


@pytest.mark.parametrize("hour", list(range(13, 24)))
def test_a_single_afternoon_hour_proves_a_24h_clock(hour):
    """One hour in 13..23 is conclusive; a 12-hour dial cannot emit it."""
    assert not looks_like_12h_clock(mk([9, 10, 3, hour]))


def test_hour_zero_proves_a_24h_clock():
    """A 12-hour dial writes midnight as 12, never 00."""
    assert not looks_like_12h_clock(mk([0, 1, 2, 3, 9, 10, 11]))


def test_refuses_when_there_is_nothing_to_repair():
    assert not looks_like_12h_clock(mk([8, 9, 10, 11, 12]))


def test_refuses_on_an_ambiguous_afternoon_only_file():
    """01-05 alone could be a genuine early-morning capture. Refuse, don't guess.

    This is precisely why CIC-2017 detection pools a whole capture day: its
    afternoon-split files are individually ambiguous.
    """
    assert not looks_like_12h_clock(mk([1, 2, 3, 4, 5]))


def test_refuses_on_too_little_data():
    assert not looks_like_12h_clock(mk([9, 3], per_hour=10))


# --------------------------------------------------------------------------
# Day-level pooling is what rescues the split files
# --------------------------------------------------------------------------

def test_pooling_a_split_capture_day_resolves_the_ambiguity():
    """CIC-2017 Friday = morning + DDos + PortScan. Alone, two are ambiguous."""
    assert not dial_needs_repair({3, 4, 5}, 10_000)
    assert not dial_needs_repair({1, 2, 3}, 10_000)
    assert not dial_needs_repair({8, 9, 10, 11, 12}, 10_000)
    pooled = {8, 9, 10, 11, 12} | {3, 4, 5} | {1, 2, 3}
    assert dial_needs_repair(pooled, 30_000), "the pooled day must be decisive"


def test_pooling_still_refuses_a_24h_day():
    assert not dial_needs_repair({8, 9, 10} | {14, 15, 16}, 10_000)


def test_zero_and_negative_timestamps_are_left_alone():
    """NaT sentinels and blank padding rows must not be shifted into validity."""
    e = np.array([0.0, -1.0, BASE + 3 * 3600], dtype=np.float64)
    r = repair_12h_clock(e)
    assert r[0] == 0.0 and r[1] == -1.0
    assert r[2] == e[2] + 12 * 3600.0
