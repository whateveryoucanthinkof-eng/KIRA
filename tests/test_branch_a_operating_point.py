"""The alert threshold has to be fitted, and fitted correctly.

`--risk-objective bce` changed what `risk_score` means: it is now
P(next window is an attack window), not the severity magnitude (0.0 benign,
0.50-0.96 by tactic) the serving cut of 0.65 was chosen against. A calibrated
probability against a ~17.5% base rate almost never reaches 0.65, so carrying
the old cut across turns the detector off -- silently, and to an operator it
looks like "no attacks today".

So the threshold is fitted on validation and stored in the checkpoint. These
tests hold that machinery to two standards:

1. The precision/recall computed from the 2000-bin score histograms must equal
   what sklearn computes from the raw scores. Not "close" -- equal. The
   histogram is an exact re-encoding of the scores, not an approximation of
   them, so any discrepancy is a bug rather than quantisation.
2. The chosen point must be defensible: inside its alert budget, never
   degenerate, and recorded with the criterion that produced it.
"""
import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("bra_op", "scripts/retrain_branch_a_live.py")
bra = importlib.util.module_from_spec(spec)
sys.modules["bra_op"] = bra
spec.loader.exec_module(bra)

BINS = bra.RISK_BINS


def _synthetic(n=200_000, base_rate=0.1752, seed=7):
    """Scores quantised onto the histogram grid, so the comparison is exact.

    Quantising is not a convenience: an unquantised score and its bin differ
    by up to one bin width, and comparing against sklearn on raw scores would
    therefore measure the binning, not the arithmetic. Putting the scores on
    the grid isolates the thing under test.
    """
    rng = np.random.default_rng(seed)
    y = rng.random(n) < base_rate
    p = np.where(y, rng.beta(4, 6, n), rng.beta(1, 12, n))
    b = np.clip((p * (BINS - 1)).astype(np.int64), 0, BINS - 1)
    pq = b / (BINS - 1)
    pos = np.bincount(b[y], minlength=BINS)
    neg = np.bincount(b[~y], minlength=BINS)
    return y, pq, pos, neg


def test_precision_and_recall_match_sklearn_exactly():
    sk = pytest.importorskip("sklearn.metrics")
    y, pq, pos, neg = _synthetic()
    curve = bra._pr_curve_from_histograms(pos, neg)

    sp, sr, st = sk.precision_recall_curve(y.astype(int), pq)
    idx = np.clip(np.rint(st * (BINS - 1)).astype(int), 0, BINS - 1)
    assert np.abs(curve["precision"][idx] - sp[:-1]).max() == 0.0
    assert np.abs(curve["recall"][idx] - sr[:-1]).max() == 0.0


def test_counts_match_brute_force_at_the_bin_boundaries():
    """`floor(p*(B-1)) >= b` iff `p >= b/(B-1)` -- the identity the whole
    histogram shortcut rests on. If it is off by one, everything downstream
    is off by one bin and nothing else would notice."""
    y, pq, pos, neg = _synthetic()
    curve = bra._pr_curve_from_histograms(pos, neg)
    for t in (0.0, 0.05, 0.1752, 0.3, 0.5, 0.65, 0.9, 0.999):
        b = int(round(t * (BINS - 1)))
        thr = curve["threshold"][b]
        assert int(((pq >= thr) & y).sum()) == int(curve["tp"][b])
        assert int(((pq >= thr) & ~y).sum()) == int(curve["fp"][b])
        assert float((pq >= thr).mean()) == pytest.approx(
            float(curve["alert_rate"][b]), abs=1e-12)


def test_f1_matches_sklearn_at_the_chosen_point():
    sk = pytest.importorskip("sklearn.metrics")
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg, criterion="max_f1")
    t = op["alert_threshold"]
    assert op["f1"] == pytest.approx(
        sk.f1_score(y.astype(int), (pq >= t).astype(int)), abs=1e-12)


# --- the failure this exists to prevent ------------------------------------

def test_the_legacy_065_cut_would_be_nearly_silent():
    """The measured reason a fitted threshold is not optional."""
    y, pq, pos, neg = _synthetic()
    curve = bra._pr_curve_from_histograms(pos, neg)
    b = int(round(0.65 * (BINS - 1)))
    assert curve["recall"][b] < 0.15, "sanity: 0.65 should miss most attacks here"
    assert curve["alert_rate"][b] < 0.02
    # and the fitted point recovers most of them
    op = bra.fit_operating_point(pos, neg)
    assert op["recall"] > 5 * curve["recall"][b]


def test_the_chosen_point_respects_its_alert_budget():
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg, criterion="budgeted_f1", alert_budget=0.5)
    assert op["alert_rate"] <= 0.5 * op["base_rate"] + 1e-12, op


def test_the_budget_is_a_guard_rail_not_a_thumb_on_the_scale():
    """At the 2x default the unconstrained max-F1 point is already inside the
    budget on a well-separated head, so the budget changes nothing. A
    constraint that binds on a healthy model would be the wrong constraint."""
    y, pq, pos, neg = _synthetic()
    budgeted = bra.fit_operating_point(pos, neg, criterion="budgeted_f1", alert_budget=2.0)
    unconstrained = bra.fit_operating_point(pos, neg, criterion="max_f1")
    assert budgeted["alert_threshold"] == unconstrained["alert_threshold"]


def test_a_tight_budget_trades_recall_for_volume_and_says_so():
    y, pq, pos, neg = _synthetic()
    loose = bra.fit_operating_point(pos, neg, criterion="max_recall_at_budget",
                                    alert_budget=2.0)
    tight = bra.fit_operating_point(pos, neg, criterion="max_recall_at_budget",
                                    alert_budget=0.5)
    assert tight["recall"] < loose["recall"]
    assert tight["alert_rate"] < loose["alert_rate"]
    assert "max recall subject to" in tight["criterion"]


def test_max_recall_at_budget_beats_budgeted_f1_on_recall():
    y, pq, pos, neg = _synthetic()
    f1p = bra.fit_operating_point(pos, neg, criterion="budgeted_f1", alert_budget=2.0)
    recp = bra.fit_operating_point(pos, neg, criterion="max_recall_at_budget",
                                   alert_budget=2.0)
    assert recp["recall"] >= f1p["recall"]
    assert recp["precision"] <= f1p["precision"]


def test_the_unconstrained_max_f1_is_always_recorded():
    """So the cost of the budget is visible rather than implied."""
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg, criterion="budgeted_f1", alert_budget=0.4)
    m = op["unconstrained_max_f1"]
    assert m["f1"] >= op["f1"]
    assert m["alert_rate"] > op["alert_rate"]


def test_a_degenerate_threshold_is_never_chosen():
    """Alerting on everything gives recall 1.0 and a respectable-looking F1
    on a high base rate. It is not an operating point."""
    y, pq, pos, neg = _synthetic(base_rate=0.6, seed=3)
    op = bra.fit_operating_point(pos, neg, criterion="max_f1")
    assert 0.0 < op["alert_rate"] < 1.0
    assert op["tn"] > 0 and op["tp"] > 0


def test_a_single_class_split_refuses_rather_than_inventing_a_threshold():
    pos = np.zeros(BINS, dtype=np.int64)
    neg = np.bincount(np.full(1000, 3), minlength=BINS)
    op = bra.fit_operating_point(pos, neg)
    assert op["fitted"] is False
    assert "both classes" in op["reason"]
    assert "alert_threshold" not in op


def test_the_criterion_is_recorded_in_words():
    """`criterion` has to survive into the checkpoint, or the number becomes a
    mystery the next time someone asks why the detector fires when it does."""
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg, criterion="budgeted_f1", alert_budget=2.0)
    assert op["criterion_name"] == "budgeted_f1"
    assert "max F1 subject to alert_rate" in op["criterion"]
    assert f"{op['base_rate']:.4f}" in op["criterion"]


def test_an_impossible_budget_falls_back_loudly_rather_than_returning_nothing():
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg, criterion="budgeted_f1",
                                 alert_budget=1e-6)
    assert op["fitted"] is True
    assert "NO threshold met the alert budget" in op["criterion"]


def test_the_curve_is_downsampled_monotone_and_small():
    y, pq, pos, neg = _synthetic()
    op = bra.fit_operating_point(pos, neg)
    curve = op["curve"]
    assert 40 <= len(curve) <= 110, len(curve)
    thr = [c["alert_threshold"] for c in curve]
    rec = [c["recall"] for c in curve]
    assert thr == sorted(thr)
    assert rec == sorted(rec, reverse=True), "recall must fall as threshold rises"
    assert all(set(c) >= {"precision", "recall", "f1", "alert_rate"} for c in curve)


def test_precision_is_one_where_nothing_is_predicted_positive():
    """sklearn's convention. It cannot be selected -- alert_rate is 0 there --
    but it must not be NaN, because a NaN would propagate into argmax."""
    y, pq, pos, neg = _synthetic()
    curve = bra._pr_curve_from_histograms(pos, neg)
    empty = curve["alert_rate"] == 0.0
    if empty.any():
        assert np.all(curve["precision"][empty] == 1.0)
    assert np.isfinite(curve["precision"]).all()
    assert np.isfinite(curve["f1"]).all()


# --- the conformal half-width ----------------------------------------------

def test_conformal_from_histogram_matches_the_exact_order_statistic():
    """The binned quantile must equal the exact one to within one bin, and must
    never be SMALLER -- rounding a conformal interval down over-states
    coverage, which is the one direction that is not allowed."""
    rng = np.random.default_rng(11)
    resid = np.abs(rng.beta(2, 5, 50_000))
    b = np.clip((resid * (BINS - 1)).astype(np.int64), 0, BINS - 1)
    hist = np.bincount(b, minlength=BINS)

    for alpha in (0.01, 0.05, 0.1, 0.25):
        got = bra.conformal_halfwidth_from_histogram(hist, alpha=alpha)
        n = resid.size
        k = int(np.ceil((n + 1) * (1 - alpha)))
        exact = np.sort(b / (BINS - 1))[k - 1]
        assert got["half_width"] >= exact - 1e-12, (alpha, got, exact)
        assert got["half_width"] - exact <= 1.0 / (BINS - 1) + 1e-12
        assert got["empirical_coverage"] >= 1 - alpha - 1e-9


def test_conformal_refuses_when_there_are_too_few_points():
    hist = np.zeros(BINS, dtype=np.int64)
    hist[5] = 10                       # n=10 cannot support 99% coverage
    out = bra.conformal_halfwidth_from_histogram(hist, alpha=0.01)
    assert out["fitted"] is False and "cannot support" in out["reason"]


def test_a_bernoulli_target_gives_a_wide_interval_and_that_is_the_honest_answer():
    """The hardcoded +/-0.05 claimed 95% coverage. Measure what it delivers."""
    rng = np.random.default_rng(2)
    n = 100_000
    y = (rng.random(n) < 0.1752).astype(float)
    p = np.where(y > 0, rng.beta(4, 6, n), rng.beta(1, 12, n))
    resid = np.abs(p - y)
    hist = np.bincount(np.clip((resid * (BINS - 1)).astype(int), 0, BINS - 1),
                       minlength=BINS)
    got = bra.conformal_halfwidth_from_histogram(hist, alpha=0.05)
    assert got["half_width"] > 0.5, (
        "a 95% interval on a near-binary target is wide by construction")
    # what the old constant actually covered
    assert float((resid <= 0.05).mean()) < 0.60, (
        "the hardcoded +/-0.05 was described as 95% coverage; it is nowhere near")
