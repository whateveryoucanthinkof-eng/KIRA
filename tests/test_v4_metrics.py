"""v4 evaluation-metric guarantees: detection, forecasting, calibration,
bootstrap and early warning.

Each module under cyberworld_v4/metrics/ exists to stop a specific way of
lying with numbers. These tests enforce the property, not the code path:

  detection    an undefined AUC must say so, never default to 0.5
  forecasting  per-horizon reporting, and cumulative onset as 1-prod(1-h)
  calibration  temperature scaling must actually recover a temperature
  bootstrap    the independent unit is the group, so intervals widen
  earlywarning lead time is measured against onset, and misses are counted
"""

import numpy as np
import pytest

from cyberworld_v4.metrics.bootstrap import (
    format_ci,
    group_bootstrap_ci,
    multi_seed_summary,
)
from cyberworld_v4.metrics.calibration import (
    TemperatureScaler,
    calibration_report,
    expected_calibration_error,
)
from cyberworld_v4.metrics.detection import detection_metrics, multilabel_metrics
from cyberworld_v4.metrics.earlywarning import (
    Episode,
    first_valid_alert,
    lead_time_report,
)
from cyberworld_v4.metrics.forecasting import (
    cumulative_onset_metrics,
    horizon_metrics,
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =========================================================================
# detection.py
# =========================================================================

@pytest.mark.parametrize("const", [0, 1])
def test_auc_is_nan_with_a_single_class_never_a_silent_half(const):
    """A single-class split makes PR-AUC and ROC-AUC undefined.

    Emitting 0.5 there is the classic way a useless evaluation looks like
    coin-flip performance instead of an error.
    """
    y = np.full(40, const, dtype=int)
    s = np.linspace(0.0, 1.0, 40)
    m = detection_metrics(y, s)

    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"]) and np.isnan(m["brier"])
    assert m["pr_auc"] != 0.5 and m["roc_auc"] != 0.5
    assert "note" in m and "undefined" in m["note"]


def test_false_alarms_per_hour_is_fp_over_hours():
    """The operational rate an analyst feels, which precision alone hides."""
    # 6 negatives scored above threshold -> 6 false alarms.
    y = np.array([0] * 10 + [1] * 4)
    s = np.array([0.9] * 6 + [0.1] * 4 + [0.9] * 4)
    m = detection_metrics(y, s, threshold=0.5, hours_observed=3.0)

    assert m["fp"] == 6 and m["tn"] == 4 and m["tp"] == 4 and m["fn"] == 0
    assert m["false_alarms_per_hour"] == pytest.approx(6 / 3.0)
    assert m["fpr"] == pytest.approx(6 / 10)


def test_false_alarms_per_hour_absent_without_an_observation_window():
    y = np.array([0, 0, 1, 1])
    assert "false_alarms_per_hour" not in detection_metrics(y, np.array([0.1, 0.9, 0.9, 0.9]))
    assert "false_alarms_per_hour" not in detection_metrics(
        y, np.array([0.1, 0.9, 0.9, 0.9]), hours_observed=0.0
    )


def test_perfect_classifier_scores_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = detection_metrics(y, y.astype(float), threshold=0.5)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["fp"] == 0 and m["fn"] == 0


@pytest.mark.parametrize("base_rate", [0.2, 0.1])
def test_random_classifier_pr_auc_equals_the_base_rate(base_rate):
    """PR-AUC's floor is the positive rate — that is why it leads, not accuracy."""
    rng = np.random.default_rng(5)
    n = 4000
    y = (rng.random(n) < base_rate).astype(int)
    m = detection_metrics(y, rng.random(n))

    assert m["pr_auc"] == pytest.approx(base_rate, abs=0.03)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.05)
    # accuracy looks fine while the detector is worthless — the whole point
    assert m["accuracy"] > 0.4


def test_multilabel_precision_and_recall_at_k_on_a_hand_checked_example():
    """Worked by hand:

    sample 0 true = {0, 1}, ranked 0 > 1 > 2 > 3
    sample 1 true = {3},    ranked 0 > 1 > 2 > 3

    top-1 hits = (1, 0)  -> P@1 = (1+0)/2/1 = 0.5   R@1 = (1/2 + 0/1)/2 = 0.25
    top-3 hits = (2, 0)  -> P@3 = (2+0)/2/3 = 1/3   R@3 = (2/2 + 0/1)/2 = 0.5
    """
    y_true = np.array([[1, 1, 0, 0], [0, 0, 0, 1]])
    y_score = np.array([[0.90, 0.80, 0.20, 0.10], [0.95, 0.85, 0.75, 0.05]])
    m = multilabel_metrics(y_true, y_score, threshold=0.5)

    assert m["precision_at_1"] == pytest.approx(0.5)
    assert m["recall_at_1"] == pytest.approx(0.25)
    assert m["precision_at_3"] == pytest.approx(1 / 3)
    assert m["recall_at_3"] == pytest.approx(0.5)
    assert m["label_support"] == [1, 1, 0, 1]


def test_multilabel_map_skips_labels_with_no_support():
    """Label 2 never occurs; including it would make mAP undefined."""
    y_true = np.array([[1, 1, 0, 0], [0, 0, 0, 1]])
    y_score = np.array([[0.90, 0.80, 0.20, 0.10], [0.95, 0.85, 0.75, 0.05]])
    m = multilabel_metrics(y_true, y_score)
    assert np.isfinite(m["mAP"]) and 0.0 <= m["mAP"] <= 1.0

    empty = multilabel_metrics(np.zeros((3, 4), dtype=int), np.random.default_rng(0).random((3, 4)))
    assert np.isnan(empty["mAP"])


# =========================================================================
# forecasting.py
# =========================================================================

def _degrading_forecast(seed=0, n=800, k_steps=5):
    """A forecaster whose signal is progressively drowned out by horizon."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    y = np.zeros((n, k_steps), dtype=int)
    s = np.zeros((n, k_steps))
    for k in range(k_steps):
        y[:, k] = (rng.random(n) < sigmoid(z - 1.0)).astype(int)
        s[:, k] = sigmoid(z + rng.normal(0, 0.3 + 1.2 * k, n))
    return y, s


def test_horizon_metrics_has_one_entry_per_step_with_the_right_seconds():
    y, s = _degrading_forecast()
    window = 2.0
    out = horizon_metrics(y, s, window_seconds=window, threshold=0.5, hours_observed=4.0)

    assert len(out["per_step"]) == y.shape[1]
    for k, m in enumerate(out["per_step"]):
        assert m["step"] == k + 1
        assert m["horizon_seconds"] == pytest.approx((k + 1) * window)
        assert np.isfinite(m["nll"])
        assert m["false_alarms_per_hour"] == pytest.approx(m["fp"] / 4.0)

    assert out["horizon_seconds"] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert len(out["pr_auc_by_step"]) == len(out["brier_by_step"]) == len(out["nll_by_step"]) == 5


def test_degradation_is_positive_when_the_forecaster_really_degrades():
    """The headline number for a forecasting claim: how much is lost by step K."""
    y, s = _degrading_forecast()
    out = horizon_metrics(y, s, window_seconds=2.0)

    pr = out["pr_auc_by_step"]
    assert out["degradation_pr_auc"] == pytest.approx(pr[0] - pr[-1])
    assert out["degradation_pr_auc"] > 0.05, f"expected genuine decay, got {pr}"
    assert pr[0] > pr[-1]


def test_degradation_undefined_for_a_single_step():
    y, s = _degrading_forecast(k_steps=1)
    assert np.isnan(horizon_metrics(y, s, window_seconds=2.0)["degradation_pr_auc"])


def test_horizon_metrics_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        horizon_metrics(np.zeros((4, 5), dtype=int), np.zeros((4, 3)), window_seconds=2.0)


def test_cumulative_onset_uses_one_minus_product_not_max():
    """Row B has a lower peak hazard than row A but a higher accumulated one.

    max(h) ranks A first and gets it wrong; 1-prod(1-h) ranks B first and gets
    it right. Only the product formula scores 1.0 here.
    """
    h = np.array([
        [0.50, 0.00, 0.00, 0.00, 0.00],   # peak 0.50, cumulative 0.500
        [0.30, 0.30, 0.30, 0.00, 0.00],   # peak 0.30, cumulative 0.657  <- onset
        [0.05, 0.05, 0.05, 0.05, 0.05],
        [0.02, 0.02, 0.02, 0.02, 0.02],
    ])
    onset_step = np.array([-1, 5, -1, -1])

    out = cumulative_onset_metrics(onset_step, h, window_seconds=2.0)
    assert out["final_cumulative_pr_auc"] == pytest.approx(1.0)

    y_final = np.array([0, 1, 0, 0])
    peak_pr_auc = detection_metrics(y_final, h.max(axis=1))["pr_auc"]
    assert peak_pr_auc < out["final_cumulative_pr_auc"], "max(h) must not reproduce the result"


def test_cumulative_onset_per_step_is_monotone_and_matches_the_formula():
    rng = np.random.default_rng(11)
    n, k_steps = 120, 5
    h = rng.uniform(0.02, 0.45, size=(n, k_steps))
    onset_step = rng.choice([-1, 1, 2, 3, 4, 5], size=n)

    out = cumulative_onset_metrics(onset_step, h, window_seconds=2.0)
    cum = 1.0 - np.cumprod(1.0 - h, axis=1)

    # the derived curve itself never goes down
    assert np.all(np.diff(cum, axis=1) >= -1e-12)

    for k, m in enumerate(out["per_step"]):
        assert m["step"] == k + 1
        assert m["horizon_seconds"] == pytest.approx((k + 1) * 2.0)
        y_k = ((onset_step != -1) & (onset_step <= k + 1)).astype(int)
        expected = detection_metrics(y_k, cum[:, k])
        # pins the scored quantity to 1-prod(1-h), step by step
        assert m["pr_auc"] == pytest.approx(expected["pr_auc"])
        assert m["brier"] == pytest.approx(expected["brier"])

    # the target itself is cumulative, so its positive rate cannot fall
    rates = [m["positive_rate"] for m in out["per_step"]]
    assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))


def test_cumulative_onset_clips_hazards_into_probability_range():
    out = cumulative_onset_metrics(
        np.array([1, -1, 2, -1]),
        np.array([[2.0, 0.1], [-1.0, 0.0], [0.3, 0.9], [0.0, 0.0]]),
        window_seconds=2.0,
    )
    assert len(out["per_step"]) == 2


# =========================================================================
# calibration.py
# =========================================================================

@pytest.mark.parametrize("true_T", [0.5, 1.0, 3.0])
def test_temperature_scaler_recovers_an_injected_temperature(true_T):
    """The key correctness test.

    y ~ Bernoulli(sigmoid(z)), so sigmoid(z) is the true probability. Feeding
    logits z*T means the optimal temperature is exactly T; if the fit does not
    find it, the scaler is not doing post-hoc calibration at all.
    """
    rng = np.random.default_rng(7)
    n = 3000
    z = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < sigmoid(z)).astype(float)

    scaler = TemperatureScaler().fit(z * true_T, y)

    assert scaler.fitted
    rel_err = abs(scaler.temperature - true_T) / true_T
    assert rel_err < 0.10, f"T={true_T}, recovered {scaler.temperature}"

    # and the recovered scaler reproduces the true probabilities
    assert np.max(np.abs(scaler.transform_logits(z * true_T) - sigmoid(z))) < 0.05


def test_ece_drops_substantially_after_fitting_miscalibrated_scores():
    rng = np.random.default_rng(11)
    n = 4000
    z = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < sigmoid(z)).astype(float)

    scaler = TemperatureScaler().fit(z * 3.0, y)   # heavily overconfident input
    rep = scaler.report()

    assert rep["before"]["ece"] > 0.08
    assert rep["after"]["ece"] < rep["before"]["ece"] / 3.0
    assert rep["ece_improvement"] == pytest.approx(rep["before"]["ece"] - rep["after"]["ece"])
    assert rep["ece_improvement"] > 0.05
    assert rep["after"]["nll"] <= rep["before"]["nll"]


def test_ece_is_near_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(3)
    n = 8000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)

    ece, _ = expected_calibration_error(y, p, n_bins=15)
    assert ece < 0.02, ece

    rep = calibration_report(y, p)
    assert rep["mean_predicted"] == pytest.approx(rep["observed_rate"], abs=0.02)


def test_reliability_curve_bin_counts_sum_to_n():
    rng = np.random.default_rng(1)
    n = 500
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)

    ece, curve = expected_calibration_error(y, p, n_bins=15)
    assert sum(curve["count"]) == n
    assert len(curve["count"]) == len(curve["confidence"]) == len(curve["accuracy"])
    assert len(curve["count"]) == len(curve["bin_center"]) <= 15
    assert 0.0 <= ece <= 1.0

    # out-of-range probabilities are clipped in, not dropped
    _, clipped = expected_calibration_error(np.zeros(4), np.array([-1.0, 0.5, 1.5, 0.5]), n_bins=5)
    assert sum(clipped["count"]) == 4


def test_unfitted_scaler_is_the_identity():
    s = TemperatureScaler()
    assert not s.fitted and s.temperature == 1.0
    assert s.report()["ece_improvement"] is None
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(s.transform_probs(p), p, atol=1e-5)


# =========================================================================
# bootstrap.py
# =========================================================================

def _grouped_items(seed=4, n_groups=10, per_group=50):
    """Items with strong per-group structure: 500 rows, 10 independent units."""
    rng = np.random.default_rng(seed)
    items = []
    for g in range(n_groups):
        mu = rng.normal(0.0, 1.0)              # group effect dominates
        for i in range(per_group):
            items.append({"group": f"grp{g}", "row": f"{g}-{i}", "v": mu + rng.normal(0, 0.05)})
    return items


def _mean_v(xs):
    return float(np.mean([x["v"] for x in xs])) if len(xs) else float("nan")


def test_group_bootstrap_is_wider_than_row_bootstrap():
    """The entire reason the module exists.

    Row resampling treats 500 autocorrelated rows as 500 independent draws and
    produces an interval that is confidently wrong. Resampling the 10 groups
    reflects the information that is actually there.
    """
    items = _grouped_items()
    grouped = group_bootstrap_ci(items, lambda x: x["group"], _mean_v, n_resamples=400, seed=1)
    rows = group_bootstrap_ci(items, lambda x: x["row"], _mean_v, n_resamples=400, seed=1)

    assert grouped["n_groups"] == 10
    assert rows["n_groups"] == 500

    grouped_width = grouped["ci_high"] - grouped["ci_low"]
    row_width = rows["ci_high"] - rows["ci_low"]
    assert grouped_width > 3 * row_width, (grouped_width, row_width)

    # both are centred on the same point estimate; only the width differs
    assert grouped["point"] == pytest.approx(rows["point"])
    assert grouped["ci_low"] < grouped["point"] < grouped["ci_high"]


def test_group_bootstrap_reports_n_groups_and_notes_a_single_group():
    items = _grouped_items(n_groups=1, per_group=30)
    out = group_bootstrap_ci(items, lambda x: x["group"], _mean_v, n_resamples=50, seed=1)

    assert out["n_groups"] == 1
    assert np.isnan(out["ci_low"]) and np.isnan(out["ci_high"])
    assert "note" in out and "2 groups" in out["note"]
    assert np.isfinite(out["point"])           # it reports rather than crashing


def test_group_bootstrap_notes_a_statistic_that_is_mostly_undefined():
    items = _grouped_items(n_groups=3, per_group=5)
    out = group_bootstrap_ci(
        items, lambda x: x["group"], lambda xs: float("nan"), n_resamples=50, seed=1
    )
    assert np.isnan(out["ci_low"]) and "note" in out


def test_multi_seed_summary_mean_and_std():
    vals = [0.10, 0.20, 0.30, 0.40, 0.50]
    out = multi_seed_summary(vals)

    assert out["n"] == 5
    assert out["mean"] == pytest.approx(np.mean(vals))
    assert out["std"] == pytest.approx(np.std(vals, ddof=1))   # sample sd, not population
    assert out["min"] == pytest.approx(0.10) and out["max"] == pytest.approx(0.50)
    assert out["values"] == pytest.approx(vals)

    assert multi_seed_summary([0.4])["std"] == 0.0
    assert multi_seed_summary([])["n"] == 0
    assert multi_seed_summary([1.0, float("nan"), 3.0])["n"] == 2   # non-finite dropped


def test_format_ci_declares_an_undefined_interval():
    undefined = {"point": 0.5, "ci_low": float("nan"), "ci_high": float("nan"), "n_groups": 1}
    assert "CI undefined" in format_ci(undefined) and "n_groups=1" in format_ci(undefined)
    assert format_ci({"point": 0.5, "ci_low": 0.4, "ci_high": 0.6}) == "0.500 [0.400, 0.600]"


# =========================================================================
# earlywarning.py
# =========================================================================

TIMES = np.array([0.0, 2.0, 4.0, 6.0, 8.0])


def test_lead_time_is_onset_minus_first_valid_alert():
    """Not forecast_steps * window_seconds — that is the horizon, not a result."""
    ep = Episode("e1", onset_time=10.0, times=TIMES, scores=np.array([0.1, 0.9, 0.1, 0.1, 0.1]))
    assert first_valid_alert(ep.times, ep.scores, threshold=0.5, before=ep.onset_time) == 2.0

    r = lead_time_report([ep], threshold=0.5)
    assert r["detected"] == 1 and r["missed"] == 0
    assert r["mean_lead_seconds"] == pytest.approx(10.0 - 2.0)
    assert r["median_lead_seconds"] == pytest.approx(8.0)


def test_persistence_requires_consecutive_windows_and_credits_the_run_start():
    """scores over threshold at t=0 (isolated), then t=4,6 (a run of two).

    persistence=1 -> t=0     persistence=2 -> t=4 (start of the run, not t=6)
    persistence=3 -> no qualifying run at all
    """
    times = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9])

    assert first_valid_alert(times, scores, threshold=0.5, persistence=1, before=14.0) == 0.0
    assert first_valid_alert(times, scores, threshold=0.5, persistence=2, before=14.0) == 4.0
    assert first_valid_alert(times, scores, threshold=0.5, persistence=3, before=14.0) is None

    # the isolated spike alone must not qualify under persistence=2
    spike = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert first_valid_alert(times, spike, threshold=0.5, persistence=2, before=14.0) is None

    # and the credited time drives the reported lead
    ep = Episode("p", 14.0, times, scores)
    assert lead_time_report([ep], threshold=0.5, persistence=2)["mean_lead_seconds"] == 10.0
    assert lead_time_report([ep], threshold=0.5, persistence=1)["mean_lead_seconds"] == 14.0


def test_an_alert_at_or_after_onset_is_not_a_warning():
    ep = Episode("late", onset_time=4.0, times=TIMES, scores=np.array([0.1, 0.1, 0.9, 0.9, 0.9]))
    assert first_valid_alert(ep.times, ep.scores, threshold=0.5, before=ep.onset_time) is None

    r = lead_time_report([ep], threshold=0.5)
    assert r["missed"] == 1 and r["detected"] == 0
    assert "note" in r and "undefined" in r["note"]
    assert r["detection_rate"] == 0.0


def test_missed_episodes_are_counted_not_dropped_and_rates_use_all_episodes():
    """Averaging lead time over caught episodes while hiding the miss rate is
    how a model that fires on a handful of attacks looks excellent."""
    episodes = [
        Episode("a", 10.0, TIMES, np.array([0.1, 0.9, 0.1, 0.1, 0.1])),   # alert t=2, lead 8
        Episode("b", 10.0, TIMES, np.array([0.1, 0.1, 0.1, 0.1, 0.9])),   # alert t=8, lead 2
        Episode("c", 10.0, TIMES, np.array([0.1, 0.1, 0.1, 0.1, 0.1])),   # never alerts
        Episode("d", 10.0, TIMES, np.array([0.0, 0.0, 0.0, 0.0, 0.0])),   # never alerts
    ]
    r = lead_time_report(
        episodes, threshold=0.5, recall_at_seconds=(2.0, 4.0, 8.0, 10.0), hours_observed=1.5
    )

    assert r["episodes"] == 4                       # every episode is accounted for
    assert r["detected"] == 2 and r["missed"] == 2
    assert sorted(r["missed_ids"]) == ["c", "d"]    # named, not silently dropped
    assert r["detection_rate"] == pytest.approx(0.5)

    # lead stats are over the caught episodes ...
    assert r["mean_lead_seconds"] == pytest.approx(5.0)
    assert r["min_lead_seconds"] == pytest.approx(2.0)
    assert r["max_lead_seconds"] == pytest.approx(8.0)
    # ... but recall-at-lead is over ALL four, which is what makes it honest
    assert r["recall_at_lead"]["2s"] == pytest.approx(2 / 4)
    assert r["recall_at_lead"]["4s"] == pytest.approx(1 / 4)
    assert r["recall_at_lead"]["8s"] == pytest.approx(1 / 4)
    assert r["recall_at_lead"]["10s"] == pytest.approx(0.0)
    assert r["hours_observed"] == pytest.approx(1.5)


def test_first_valid_alert_sorts_and_handles_empty_input():
    scores = np.array([0.1, 0.1, 0.9, 0.9, 0.1])
    shuffled = np.argsort([3, 1, 4, 0, 2])
    assert first_valid_alert(
        TIMES[shuffled], scores[shuffled], threshold=0.5, persistence=2, before=10.0
    ) == 4.0
    assert first_valid_alert(np.array([]), np.array([]), threshold=0.5) is None
    # `before` filtering can empty the episode
    assert first_valid_alert(TIMES, scores, threshold=0.5, before=-1.0) is None
    assert lead_time_report([], threshold=0.5)["detection_rate"] == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG in earlywarning.first_valid_alert: it resolves the run start with "
        "list(times).index(t), which finds the FIRST index holding that timestamp "
        "value. With a duplicated timestamp it credits an earlier window than the "
        "one that started the qualifying run, inflating reported lead time — the "
        "exact overstatement this module exists to prevent."
    ),
)
def test_duplicate_timestamps_must_not_inflate_lead_time():
    # the run of two over-threshold windows starts at the FIRST t=4.0 entry
    times = np.array([0.0, 2.0, 4.0, 4.0, 6.0])
    scores = np.array([0.1, 0.1, 0.9, 0.9, 0.1])
    assert first_valid_alert(times, scores, threshold=0.5, persistence=2, before=10.0) == 4.0
