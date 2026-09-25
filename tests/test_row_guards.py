"""Row guards must drop exactly the upstream-broken rows and nothing else.

A guard that silently discards real traffic is worse than the defect it fixes,
so every rule here is pinned from both sides: it fires on the documented
signature, and it leaves legitimate rows alone.
"""

import numpy as np
import pytest

from data_unification.row_guards import (
    MAX_PLAUSIBLE_EPOCH,
    MIN_PLAUSIBLE_EPOCH,
    blank_host_mask,
    invalid_timestamp_mask,
    rejection_breakdown,
    tso_failure_mask,
    valid_row_mask,
)

GOOD_TS = 1519286537.0   # 2018-02-22 08:22:17 UTC
CTU_TS = 1319000000.0    # 2011 -- CTU-13 era, must survive


# ---------------------------------------------------------------------- time

def test_epoch_1970_rows_are_rejected_although_positive():
    """The real defect: `10/01/1970 03:04:26` is ~2.3e4 -- POSITIVE, so the old
    `<= 0` check let all fourteen through. One stretches a 2s grid over 48 years."""
    ts = np.array([23066.0])
    assert ts[0] > 0, "precondition: the sentinel is positive"
    assert invalid_timestamp_mask(ts).all()


def test_nat_sentinel_is_rejected():
    """Casting NaT yields -9.223e9, a plausible-looking negative epoch."""
    assert invalid_timestamp_mask(np.array([-9.223372036854776e9])).all()


@pytest.mark.parametrize("ts", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_degenerate_timestamps_are_rejected(ts):
    assert invalid_timestamp_mask(np.array([ts])).all()


@pytest.mark.parametrize("ts", [GOOD_TS, CTU_TS, MIN_PLAUSIBLE_EPOCH, MAX_PLAUSIBLE_EPOCH])
def test_real_corpus_timestamps_survive(ts):
    """CTU-13 is 2011-2013, so the floor must not exclude it."""
    assert not invalid_timestamp_mask(np.array([ts])).any()


def test_far_future_is_rejected():
    assert invalid_timestamp_mask(np.array([MAX_PLAUSIBLE_EPOCH + 1.0])).all()


# ----------------------------------------------------------------------- TSO

def test_tso_signature_requires_both_fields():
    """protocol==0 AND port==0 together. Either alone is not the defect."""
    assert tso_failure_mask(np.array([0]), np.array([0])).all()
    assert not tso_failure_mask(np.array([0]), np.array([80])).any()
    assert not tso_failure_mask(np.array([6]), np.array([0])).any()


def test_ordinary_flows_are_not_tso_failures():
    proto = np.array([6, 17, 1, 6])
    port = np.array([80, 53, 0, 443])
    assert not tso_failure_mask(proto, port).any()


# ---------------------------------------------------------------------- host

@pytest.mark.parametrize("bad", ["nan", "NaN", "", "  ", "none", "0"])
def test_blank_hosts_are_rejected(bad):
    assert blank_host_mask(np.array([bad])).all()


@pytest.mark.parametrize("ok", ["10.0.0.1", "172.16.0.5", "192.168.10.50"])
def test_real_hosts_survive(ok):
    assert not blank_host_mask(np.array([ok])).any()


# -------------------------------------------------------------------- joined

def test_valid_row_mask_keeps_only_the_clean_row():
    e = np.array([GOOD_TS, 23000.0, -9.2e9, GOOD_TS, GOOD_TS])
    p = np.array([6, 6, 6, 0, 6])
    d = np.array([80, 80, 80, 0, 80])
    s = np.array(["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "nan"])
    assert valid_row_mask(e, p, d, s).tolist() == [True, False, False, False, False]


def test_a_fully_clean_chunk_loses_nothing():
    n = 100
    e = np.full(n, GOOD_TS)
    p = np.full(n, 6)
    d = np.full(n, 443)
    s = np.array([f"10.0.0.{i % 250}" for i in range(n)])
    assert valid_row_mask(e, p, d, s).all()
    b = rejection_breakdown(e, p, d, s)
    assert b["invalid_timestamp"] == b["tso_failure"] == b["blank_host"] == 0


def test_rejection_breakdown_attributes_each_drop():
    e = np.array([GOOD_TS, 23000.0, GOOD_TS])
    p = np.array([6, 6, 0])
    d = np.array([80, 80, 0])
    s = np.array(["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    b = rejection_breakdown(e, p, d, s)
    assert b == {"invalid_timestamp": 1, "tso_failure": 1, "blank_host": 0, "total_rows": 3}
