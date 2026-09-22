"""Alert banding, and what happens when risk_score changes meaning.

Branch A's risk head has two objectives:

  smooth_l1  risk_score is a severity magnitude -- 0.0 benign, 0.50-0.96 by
             tactic for attack. The historical 0.65 alerting cut was chosen
             against that scale.
  bce        risk_score is P(next window is an attack window). This became the
             default once the severity regression was shown to score worse
             than predicting zero.

Both live on [0, 1] but they are not the same quantity. A calibrated
probability against a ~17.5% base rate seldom exceeds 0.65, so carrying the old
cut across would quietly stop the detector alerting -- a silent failure that
presents as "no attacks today".
"""
import pytest


def _bands(threshold):
    """Mirror of ModelAdapter._alert_level, as a pure function."""
    t = threshold
    def level(risk):
        if risk >= t + (1.0 - t) * 0.60:
            return "CRITICAL"
        if risk >= t + (1.0 - t) * 0.25:
            return "ELEVATED"
        if risk >= t:
            return "WARNING"
        return "NOMINAL"
    return level


def test_warning_band_is_reachable():
    """The regression: ELEVATED used the literal 0.65 and alert_threshold was
    also 0.65, so nothing could ever be WARNING."""
    level = _bands(0.65)
    assert level(0.65) == "WARNING", "the band at exactly the threshold must be WARNING"
    assert level(0.70) == "WARNING"
    assert "WARNING" in {level(x / 100) for x in range(0, 101)}


def test_bands_are_ordered_and_cover_the_range():
    for t in (0.05, 0.25, 0.5, 0.65, 0.9):
        level = _bands(t)
        assert level(0.0) == "NOMINAL"
        assert level(1.0) == "CRITICAL"
        seen = [level(x / 1000) for x in range(1001)]
        order = {"NOMINAL": 0, "WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
        ranks = [order[s] for s in seen]
        assert ranks == sorted(ranks), f"bands not monotonic at threshold {t}"


def test_bands_follow_a_low_threshold():
    """With a fitted probability threshold the bands must move with it, not
    stay anchored to the old severity scale."""
    level = _bands(0.20)
    assert level(0.10) == "NOMINAL"
    assert level(0.20) == "WARNING"
    assert level(0.40) == "ELEVATED"
    assert level(0.90) == "CRITICAL"


def test_a_probability_scale_would_be_silent_under_the_old_cut():
    """Why this matters: score a calibrated head against the legacy 0.65."""
    import numpy as np
    rng = np.random.default_rng(0)
    base_rate = 0.1752
    y = rng.random(200_000) < base_rate
    # a good but calibrated model: P(attack) concentrated well below 0.65
    p = np.where(y, rng.beta(4, 6, 200_000), rng.beta(1, 12, 200_000))
    fired_old = (p >= 0.65).mean()
    assert fired_old < 0.02, "sanity: a calibrated head rarely exceeds 0.65 here"
    # recall at the legacy cut is what an operator actually loses
    recall_old = (p[y] >= 0.65).mean()
    assert recall_old < 0.15, (
        f"the legacy cut would miss most attacks (recall {recall_old:.3f}); "
        f"this is the failure the fitted operating point prevents")


def test_adapter_exposes_the_risk_objective():
    """Serving must be able to say which quantity it is thresholding."""
    pytest.importorskip("torch")
    import inspect
    from control_backend import model_adapter as ma
    src = inspect.getsource(ma.AntigravityModelAdapter._adopt_risk_semantics)
    assert "risk_objective" in src
    assert "operating_point" in src, "must read a fitted threshold when present"
