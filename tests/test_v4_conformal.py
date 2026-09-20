"""v4 conformal prediction guarantees.

v3's "conformal" was `risk +/- 0.05` and a table of baked-in radii. A fixed
number responds to nothing and guarantees nothing. These tests check that the
replacements do the two things a fixed number cannot:

  SplitConformal     finite-sample coverage under exchangeability, via the
                     ceil((n+1)(1-alpha))/n correction
  AdaptiveConformal  long-run coverage WITHOUT exchangeability — the headline
                     property, asserted against a SplitConformal that fails on
                     the same shifted stream
"""

import numpy as np
import pytest

from cyberworld_v4.conformal import (
    AdaptiveConformal,
    SplitConformal,
    conformal_quantile,
)


# =========================================================================
# conformal_quantile
# =========================================================================

def test_quantile_uses_the_finite_sample_correction_not_the_plain_quantile():
    """k = ceil((n+1)(1-alpha)); the +1 is what makes coverage exact."""
    s = np.arange(1.0, 100.0)                       # n = 99, values 1..99
    k = int(np.ceil((99 + 1) * 0.9))                # = 90
    assert conformal_quantile(s, 0.1) == pytest.approx(s[k - 1]) == 90.0

    s100 = np.arange(100.0)                         # n = 100, values 0..99
    corrected = conformal_quantile(s100, 0.1)
    assert corrected == pytest.approx(s100[int(np.ceil(101 * 0.9)) - 1]) == 90.0
    # strictly more conservative than the uncorrected empirical quantile
    assert corrected > float(np.quantile(s100, 0.9))


def test_quantile_ignores_input_order_and_shape():
    rng = np.random.default_rng(0)
    s = rng.random(200)
    assert conformal_quantile(s, 0.1) == conformal_quantile(rng.permutation(s), 0.1)
    assert conformal_quantile(s.reshape(20, 10), 0.1) == conformal_quantile(s, 0.1)


def test_quantile_is_inf_when_there_are_too_few_points_for_the_alpha():
    """ceil((n+1)(1-alpha)) > n means the sample cannot support the request.

    Saying inf is honest; silently returning the max would claim coverage the
    data does not provide.
    """
    # n=10, alpha=0.05 -> ceil(11*0.95) = 11 > 10
    assert conformal_quantile(np.arange(10.0), 0.05) == float("inf")
    assert conformal_quantile(np.array([]), 0.1) == float("inf")

    # n=20 is exactly enough: ceil(21*0.95) = 20
    assert np.isfinite(conformal_quantile(np.arange(20.0), 0.05))
    assert conformal_quantile(np.arange(20.0), 0.05) == 19.0

    # a looser alpha is satisfiable with the same 10 points
    assert np.isfinite(conformal_quantile(np.arange(10.0), 0.2))


# =========================================================================
# SplitConformal
# =========================================================================

def test_split_conformal_empirical_coverage_matches_the_target():
    rng = np.random.default_rng(0)
    cal_true, cal_pred = rng.normal(0, 1, 2000), np.zeros(2000)
    sc = SplitConformal(alpha=0.05).fit(cal_true, cal_pred)

    test_true, test_pred = rng.normal(0, 1, 4000), np.zeros(4000)
    ev = sc.evaluate(test_true, test_pred)

    assert ev["target_coverage"] == 0.95
    assert ev["empirical_coverage"] == pytest.approx(0.95, abs=0.02)
    assert ev["n_calibration"] == 2000 and ev["n_test"] == 4000
    assert ev["median_width"] == pytest.approx(2 * sc.quantile_)
    # the interval is centred on the prediction
    lo, hi = sc.interval(np.array([3.0]))
    assert lo[0] == pytest.approx(3.0 - sc.quantile_) and hi[0] == pytest.approx(3.0 + sc.quantile_)


def test_split_conformal_widens_for_a_smaller_alpha():
    rng = np.random.default_rng(1)
    y, p = rng.normal(0, 1, 1500), np.zeros(1500)
    q90 = SplitConformal(alpha=0.10).fit(y, p).quantile_
    q99 = SplitConformal(alpha=0.01).fit(y, p).quantile_
    assert q99 > q90 > 0


# =========================================================================
# AdaptiveConformal — the headline property
# =========================================================================

@pytest.fixture(scope="module")
def shifted_stream():
    """Residual scale jumps 1 -> 5 halfway through. Calibration saw only scale 1."""
    rng = np.random.default_rng(1)
    calibration = rng.normal(0, 1, 1000)
    n_pre, n_post = 800, 2000
    stream = np.concatenate([rng.normal(0, 1, n_pre), rng.normal(0, 5, n_post)])
    return calibration, stream, n_pre


def test_aci_holds_coverage_under_shift_where_split_conformal_collapses(shifted_stream):
    """Split conformal's quantile is frozen at fit time, so when the world moves
    its coverage falls off a cliff. ACI adapts with no labels beyond the realized
    outcome, which this system gets for free forecast_seconds later.

    Both sides are asserted: the adaptive one must hold AND the static one must
    fail, otherwise the comparison proves nothing.
    """
    calibration, stream, n_pre = shifted_stream
    preds = np.zeros_like(stream)

    static = SplitConformal(alpha=0.05).fit(calibration, np.zeros_like(calibration))
    lo, hi = static.interval(preds)
    static_covered = (stream >= lo) & (stream <= hi)

    aci = AdaptiveConformal(alpha=0.05, gamma=0.02, window=500).warm_start(calibration)
    aci_errors = np.array([aci.update(float(y), 0.0)["error"] for y in stream])

    # both behave before the shift
    assert float(1 - aci_errors[:n_pre].mean()) == pytest.approx(0.95, abs=0.04)
    assert float(static_covered[:n_pre].mean()) == pytest.approx(0.95, abs=0.04)

    # after the shift (allowing the adaptive window to turn over) they diverge
    after = slice(n_pre + 500, None)
    aci_after = float(1 - aci_errors[after].mean())
    static_after = float(static_covered[after].mean())

    assert aci_after == pytest.approx(0.95, abs=0.03), f"ACI lost coverage: {aci_after}"
    assert static_after < 0.70, f"SplitConformal was supposed to fail here: {static_after}"
    assert aci_after - static_after > 0.20

    assert aci.running_coverage() == pytest.approx(float(1 - aci_errors.mean()))
    assert aci.running_coverage(last=500) == pytest.approx(0.95, abs=0.05)
    assert len(aci.history_) == stream.size


def test_aci_alpha_update_follows_the_gibbs_candes_recursion():
    """alpha_{t+1} = alpha_t + gamma * (alpha - err_t).

    Covered -> alpha_t rises -> band tightens. Missed -> alpha_t falls -> widens.
    """
    gamma, alpha = 0.05, 0.10

    covered = AdaptiveConformal(alpha=alpha, gamma=gamma, window=500).warm_start(np.linspace(0, 1, 200))
    rec = covered.update(0.0, 0.0)                       # residual 0, certainly inside
    assert rec["error"] == 0.0
    assert covered.alpha_t == pytest.approx(alpha + gamma * alpha)

    missed = AdaptiveConformal(alpha=alpha, gamma=gamma, window=500).warm_start(np.linspace(0, 1, 200))
    rec = missed.update(1e6, 0.0)                        # far outside any band
    assert rec["error"] == 1.0
    assert missed.alpha_t == pytest.approx(alpha + gamma * (alpha - 1.0))
    assert missed.alpha_t < covered.alpha_t              # a miss widens, a hit tightens


def test_min_width_floors_the_interval():
    """Online adaptation may tighten below the accredited offline band, but it
    must not silently loosen past it — an adversary shaping residuals would
    otherwise widen the band until the real attack fits inside."""
    tiny = np.full(300, 0.001)

    floored = AdaptiveConformal(alpha=0.05, min_width=2.0).warm_start(tiny)
    lo, hi = floored.predict_interval(1.0)
    assert (hi - lo) / 2.0 == pytest.approx(2.0)
    assert (lo, hi) == pytest.approx((-1.0, 3.0))

    unfloored = AdaptiveConformal(alpha=0.05).warm_start(tiny)
    ulo, uhi = unfloored.predict_interval(1.0)
    assert (uhi - ulo) / 2.0 < 0.01, "without min_width the band should collapse"

    capped = AdaptiveConformal(alpha=0.05, max_width=0.5).warm_start(np.full(300, 9.0))
    clo, chi = capped.predict_interval(0.0)
    assert (chi - clo) / 2.0 == pytest.approx(0.5)

    # no history and no warm start: the floor is all there is
    bare = AdaptiveConformal(alpha=0.05, min_width=0.75)
    blo, bhi = bare.predict_interval(0.0)
    assert (bhi - blo) / 2.0 == pytest.approx(0.75)


def test_drift_signal_ratio_rises_when_residuals_grow():
    """A sustained widening is itself information — and an attack surface."""
    rng = np.random.default_rng(3)
    aci = AdaptiveConformal(alpha=0.05, gamma=0.02, window=200)
    aci.warm_start(rng.normal(0, 1, 200))

    for _ in range(500):
        aci.update(float(rng.normal(0, 1)), 0.0)
    calm = aci.drift_signal(short=50, long=500)
    assert calm["ratio"] == pytest.approx(1.0, abs=0.25)
    assert calm["ratio"] == pytest.approx(calm["short"] / calm["long"])

    for _ in range(120):
        aci.update(float(rng.normal(0, 8)), 0.0)
    drifting = aci.drift_signal(short=50, long=500)

    assert drifting["ratio"] > 1.5, drifting
    assert drifting["ratio"] > calm["ratio"]
    assert drifting["short"] > drifting["long"] > 0


def test_drift_signal_is_undefined_before_the_short_window_fills():
    aci = AdaptiveConformal(alpha=0.05).warm_start(np.linspace(0.1, 1.0, 100))
    for _ in range(10):
        aci.update(0.2, 0.0)
    d = aci.drift_signal(short=50, long=500)
    assert np.isnan(d["ratio"]) and np.isnan(d["short"]) and np.isnan(d["long"])
    assert np.isnan(AdaptiveConformal().running_coverage())


def test_aci_falls_back_when_the_window_cannot_support_alpha_t():
    """Too few scores for the requested alpha -> conformal_quantile is inf, and
    the band uses the observed maximum rather than an infinite interval."""
    aci = AdaptiveConformal(alpha=0.05, window=500)
    aci.warm_start([0.5, 1.0, 2.0])                 # n=3, far too few for alpha=0.05
    assert conformal_quantile(np.asarray(aci.scores_), 0.05) == float("inf")
    lo, hi = aci.predict_interval(0.0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert (hi - lo) / 2.0 == pytest.approx(2.0)
