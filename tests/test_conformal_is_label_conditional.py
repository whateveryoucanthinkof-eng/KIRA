"""Conformal coverage must be reported, and guaranteed, per label.

Split conformal promises MARGINAL coverage. On a target that is 0 for ~83%
of windows the calibration quantile is set by the benign majority, so the
interval can meet 95% overall while missing a large share of the attack
windows. `SplitConformal.evaluate(groups=label)` exposes that;
`LabelConditionalConformal` fixes it with one quantile per class.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cyberworld_v4.conformal import (
    LabelConditionalConformal,
    SplitConformal,
    coverage_by_group,
)

REPO = Path(__file__).resolve().parents[1]


def _stream(n, seed, base_rate=0.17):
    """A model that is confident on benign windows and mediocre on attacks."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(int)
    p = np.where(y == 1, rng.beta(2.0, 3.0, n), rng.beta(1.0, 8.0, n))
    return y, p


@pytest.fixture(scope="module")
def cal_test():
    return _stream(20000, seed=0), _stream(20000, seed=1)


def test_marginal_coverage_hides_the_attack_windows(cal_test):
    (yc, pc), (yt, pt) = cal_test
    rep = SplitConformal(alpha=0.05).fit(yc, pc).evaluate(yt, pt, groups=yt)

    assert abs(rep["empirical_coverage"] - 0.95) < 0.01, "premise: marginal target is met"
    attack = rep["coverage_by_group"]["1"]["empirical_coverage"]
    benign = rep["coverage_by_group"]["0"]["empirical_coverage"]
    assert attack < 0.85, f"premise: attack windows should be under-covered, got {attack:.3f}"
    assert benign > 0.99
    assert rep["worst_group"] == "1"
    assert rep["worst_group_coverage"] == attack


def test_label_conditional_sets_cover_every_class(cal_test):
    (yc, pc), (yt, pt) = cal_test
    rep = LabelConditionalConformal(alpha=0.05).fit(yc, pc).evaluate(yt, pt)
    for c in ("0", "1"):
        cov = rep["coverage_by_class"][c]["empirical_coverage"]
        assert cov >= 0.94, f"class {c} coverage {cov:.3f} below target"
    assert rep["worst_class_coverage"] >= 0.94
    # The price: sets that contain both labels where the model cannot tell.
    assert 0.0 < rep["ambiguous_rate"] < 1.0
    assert 1.0 <= rep["mean_set_size"] <= 2.0


def test_binary_vector_equals_two_column_probabilities(cal_test):
    (yc, pc), (yt, pt) = cal_test
    a = LabelConditionalConformal(alpha=0.1).fit(yc, pc)
    b = LabelConditionalConformal(alpha=0.1).fit(yc, np.stack([1 - pc, pc], axis=1))
    assert a.quantiles_ == b.quantiles_
    np.testing.assert_array_equal(a.predict_sets(pt), b.predict_sets(np.stack([1 - pt, pt], 1)))


def test_a_class_without_enough_calibration_points_is_always_in_the_set():
    y = np.array([0] * 50 + [1] * 3)
    p = np.concatenate([np.full(50, 0.1), np.full(3, 0.9)])
    lcc = LabelConditionalConformal(alpha=0.05).fit(y, p)
    assert lcc.quantiles_[1] == float("inf"), "3 points cannot support alpha=0.05"
    sets = lcc.predict_sets(np.array([0.0, 0.5, 1.0]))
    assert sets[:, 1].all(), "no guarantee without data -> the class stays in"


def test_multiclass_is_supported():
    rng = np.random.default_rng(3)
    n, C = 6000, 4
    y = rng.integers(0, C, n)
    logits = rng.normal(size=(n, C)) + 2.0 * np.eye(C)[y]
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    lcc = LabelConditionalConformal(alpha=0.1).fit(y[:3000], p[:3000])
    rep = lcc.evaluate(y[3000:], p[3000:])
    assert set(rep["coverage_by_class"]) == {"0", "1", "2", "3"}
    assert rep["worst_class_coverage"] >= 0.86


def test_groups_must_align_with_predictions():
    with pytest.raises(ValueError):
        coverage_by_group(np.ones(4, bool), np.zeros(3))
    with pytest.raises(ValueError):
        LabelConditionalConformal().fit(np.zeros(3), np.zeros(4))


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        LabelConditionalConformal().predict_sets(np.zeros(2))


def test_report_survives_a_json_round_trip(cal_test):
    (yc, pc), (yt, pt) = cal_test
    rep = SplitConformal().fit(yc, pc).evaluate(yt, pt, groups=yt)
    rep["label_conditional"] = LabelConditionalConformal().fit(yc, pc).evaluate(yt, pt)
    back = json.loads(json.dumps(rep, default=float))
    assert back["coverage_by_group"].keys() == rep["coverage_by_group"].keys()
    assert back["label_conditional"]["coverage_by_class"]["1"]["n"] == int((yt == 1).sum())


def test_the_benchmark_reports_and_gates_per_label_coverage():
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert "groups=y_fut.ravel().astype(int)" in src, "interval coverage is not split by label"
    assert "LabelConditionalConformal(alpha=args.conformal_alpha)" in src
    assert "worst_group_coverage" in src and "worst_class_coverage" in src, (
        "the credibility gate does not look at the worst label")
