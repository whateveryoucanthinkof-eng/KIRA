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


# --- the rule layer must not make alerting arithmetically impossible -------

def test_internal_only_traffic_cannot_alert_under_the_shipped_settings():
    """The defect, stated as arithmetic.

    Internal-only windows are scored max(0.05, raw_risk * 0.4). raw_risk is
    bounded at 1.0, so the displayed value cannot exceed 0.40 -- while the
    alert threshold is 0.65. No internal-only window can raise an alert at ANY
    model confidence: a model output of 0.9965 displays as 0.3986, and you
    would need raw_risk = 1.625 to clear the cut.

    Lateral movement is internal-only traffic by definition, so this silently
    makes the system unable to alert on the behaviour the pipeline exists to
    forecast.
    """
    factor, threshold = 0.4, 0.65
    ceiling = max(0.05, 1.0 * factor)
    assert ceiling < threshold
    assert threshold / factor > 1.0, "would need a risk above 1.0 to alert"
    for ml in (0.30, 0.50, 0.70, 0.90, 0.9965, 1.00):
        assert max(0.05, ml * factor) < threshold


def test_the_adapter_warns_when_alerting_is_unreachable(caplog):
    """It must be impossible to ship this combination unnoticed."""
    pytest.importorskip("torch")
    import logging
    from control_backend.model_adapter import AntigravityModelAdapter
    a = AntigravityModelAdapter.__new__(AntigravityModelAdapter)
    a.rules_enabled = True
    a.alert_threshold = 0.65
    a.INTERNAL_SUPPRESSION_FACTOR = 0.4
    with caplog.at_level(logging.WARNING, logger="antigravity.model_adapter"):
        ceiling = AntigravityModelAdapter._check_alerting_is_reachable(a)
    assert ceiling == pytest.approx(0.40)
    assert "ALERTING UNREACHABLE" in caplog.text
    assert "Lateral movement is internal by definition" in caplog.text


def test_no_warning_once_the_settings_are_consistent():
    pytest.importorskip("torch")
    import logging
    from control_backend.model_adapter import AntigravityModelAdapter
    a = AntigravityModelAdapter.__new__(AntigravityModelAdapter)
    a.rules_enabled = True
    a.alert_threshold = 0.30          # below the 0.40 ceiling
    a.INTERNAL_SUPPRESSION_FACTOR = 0.4
    import io as _io, contextlib
    logger = logging.getLogger("antigravity.model_adapter")
    buf = _io.StringIO()
    h = logging.StreamHandler(buf); logger.addHandler(h)
    try:
        AntigravityModelAdapter._check_alerting_is_reachable(a)
    finally:
        logger.removeHandler(h)
    assert "UNREACHABLE" not in buf.getvalue()


def test_disabling_rules_removes_the_ceiling():
    pytest.importorskip("torch")
    import logging
    from control_backend.model_adapter import AntigravityModelAdapter
    a = AntigravityModelAdapter.__new__(AntigravityModelAdapter)
    a.rules_enabled = False
    a.alert_threshold = 0.65
    a.INTERNAL_SUPPRESSION_FACTOR = 0.4
    import io as _io
    logger = logging.getLogger("antigravity.model_adapter")
    buf = _io.StringIO(); h = logging.StreamHandler(buf); logger.addHandler(h)
    try:
        AntigravityModelAdapter._check_alerting_is_reachable(a)
    finally:
        logger.removeHandler(h)
    assert "UNREACHABLE" not in buf.getvalue()
