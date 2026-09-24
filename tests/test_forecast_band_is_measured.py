"""The forecast's uncertainty band is fitted, not invented.

The dashboard drew each forecast step's band as risk +/- 20 * DeepOP token
confidence -- no measurement behind it, and backwards (more confidence, wider
band). Branch B's training now fits a per-step split-conformal half-width on
validation, the checkpoint carries it, the adapter serves it, and with no fit
there is no band.
"""

from pathlib import Path

import numpy as np
import pytest

from control_backend.forecast_band import forecast_band, forecast_band_halfwidths
from cyberworld_v4.conformal import conformal_quantile, halfwidth_from_histogram

REPO = Path(__file__).resolve().parents[1]
BINS = 2000


def _hist(resid):
    b = np.clip((np.clip(resid, 0, 1) * (BINS - 1)).astype(int), 0, BINS - 1)
    return np.bincount(b, minlength=BINS)


def test_histogram_halfwidth_matches_the_exact_quantile_within_one_bin():
    rng = np.random.default_rng(0)
    resid = np.abs(rng.normal(0, 0.12, 50_000)).clip(0, 1)
    exact = conformal_quantile(resid, 0.05)
    got = halfwidth_from_histogram(_hist(resid), 0.05)
    assert got["fitted"]
    assert exact <= got["half_width"] <= exact + 1.0 / (BINS - 1) + 1e-12, "must round UP, by < 1 bin"
    assert got["empirical_coverage"] >= 0.95


def test_histogram_halfwidth_refuses_too_few_points():
    got = halfwidth_from_histogram(_hist(np.array([0.1, 0.2])), 0.05)
    assert not got["fitted"] and "cannot support" in got["reason"]
    assert not halfwidth_from_histogram(np.zeros(BINS), 0.05)["fitted"]


def test_the_trainer_fits_one_band_per_step():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rfml", REPO / "scripts" / "retrain_future_models_live.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(1)
    # Later steps are harder: residual spread grows with k.
    hists = np.stack([_hist(np.abs(rng.normal(0, 0.03 * (k + 1), 20_000))) for k in range(5)])
    conf = mod._forecast_risk_conformal(hists)
    hw = conf["half_width_by_step"]
    assert len(hw) == 5 and all(h is not None for h in hw)
    assert hw == sorted(hw), "a band should widen with the horizon when the error does"
    assert conf["fitted_on"] == "validation" and conf["alpha"] == 0.05


def test_the_trainer_saves_the_band_with_the_checkpoint():
    src = (REPO / "scripts" / "retrain_future_models_live.py").read_text(encoding="utf-8")
    assert '"forecast_risk_conformal":_conf' in src


def test_serving_reads_the_band_and_clips_it():
    ckpt = {"forecast_risk_conformal": {"half_width_by_step": [0.05, 0.1, None, float("nan"), 0.3]}}
    hw = forecast_band_halfwidths(ckpt)
    assert hw == [0.05, 0.1, None, None, 0.3]
    assert forecast_band(0.5, hw, 0) == {"risk_lower": 0.45, "risk_upper": 0.55}
    assert forecast_band(0.02, hw, 4) == {"risk_lower": 0.0, "risk_upper": 0.32}
    assert forecast_band(0.9, hw, 4) == {"risk_lower": 0.6, "risk_upper": 1.0}
    assert forecast_band(0.5, hw, 2) == {"risk_lower": None, "risk_upper": None}
    assert forecast_band(0.5, hw, 9) == {"risk_lower": None, "risk_upper": None}


@pytest.mark.parametrize("ckpt", [{}, {"forecast_risk_conformal": {}},
                                  {"forecast_risk_conformal": {"half_width_by_step": [None, None]}}])
def test_no_fitted_band_means_no_band(ckpt):
    assert forecast_band_halfwidths(ckpt) is None
    assert forecast_band(0.4, None, 0) == {"risk_lower": None, "risk_upper": None}


def test_the_wire_schema_and_dashboard_carry_it_and_nothing_is_invented():
    from control_backend.schema import ForecastPoint
    fp = ForecastPoint(horizon_seconds=30.0, risk=0.4, risk_lower=0.3, risk_upper=0.5)
    assert fp.risk_lower == 0.3 and ForecastPoint(horizon_seconds=30.0, risk=0.4).risk_upper is None
    app = (REPO / "web_dashboard" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "conf * 20" not in app, "the dashboard is inventing a band again"
    assert "f.risk_lower" in app and "f.risk_upper" in app
