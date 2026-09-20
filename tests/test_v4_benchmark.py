"""v4 benchmark-harness guarantees.

The harness exists to make two rules mechanical rather than a matter of
discipline:

  * the test split is touched once, after fitting, calibration and threshold
    selection are frozen (spec 66)
  * if a baseline wins, that is the result and it is reported (spec 62)

A harness that can only confirm the preferred model is not evidence, so these
tests deliberately run a benchmark in which the "full" model loses.
"""

import numpy as np
import pytest

from cyberworld_v4.benchmark import (
    EvalSet,
    ModelEntry,
    _compare,
    choose_threshold,
    markdown_table,
    run_benchmark,
    save,
)
from cyberworld_v4.metrics.calibration import TemperatureScaler
from cyberworld_v4.metrics.detection import detection_metrics


L, D = 4, 3


def make_split(n, seed):
    """X[:, -1, 0] carries the signal; column 2 is noise the weak model uses."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, L, D))
    p = 1.0 / (1.0 + np.exp(-(2.0 * X[:, -1, 0] - 0.8)))
    y = (rng.random(n) < p).astype(int)
    groups = np.array([f"g{i % 8}" for i in range(n)])
    return EvalSet(
        X=X, y_current=y, y_future=np.zeros((n, 5), dtype=int),
        groups=groups, hours_observed=2.0,
    )


def strong_scores(X):
    return 1.0 / (1.0 + np.exp(-(2.0 * X[:, -1, 0] - 0.8)))


def weak_scores(X):
    return 1.0 / (1.0 + np.exp(-(0.2 * X[:, -1, 2])))


class Recorder:
    """A model entry that logs every call, so call ORDER can be asserted."""

    def __init__(self, score_fn, log):
        self.score_fn, self.log = score_fn, log

    def fit(self, X, y):
        self.log.append(("fit", len(X)))
        return self

    def predict_proba(self, X):
        self.log.append(("predict", len(X)))
        return self.score_fn(X)


# =========================================================================
# choose_threshold
# =========================================================================

def test_choose_threshold_refuses_a_degenerate_operating_point():
    """With a 90% positive rate, raw F1 is maximised by predicting everything
    positive: recall 1.0, FPR 1.0, F1 0.947, and a detector that never says
    "benign". That is not an operating point, so it must be excluded even though
    it scores best.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([1] * 180 + [0] * 20)
    scores = rng.choice([0.2, 0.5], size=n)          # 0.2 is the minimum -> all-positive

    all_positive = detection_metrics(y, scores, threshold=0.2)
    assert all_positive["tn"] + all_positive["fn"] == 0, "fixture no longer degenerate"

    chosen = choose_threshold(y, scores)
    picked = detection_metrics(y, scores, threshold=chosen)

    assert chosen != 0.2
    assert picked["tp"] + picked["fp"] > 0, "never predicts positive"
    assert picked["tn"] + picked["fn"] > 0, "never predicts negative"
    # the rejected point really did score higher on raw F1
    assert all_positive["f1"] > picked["f1"]


def test_chosen_threshold_is_non_degenerate_on_ordinary_scores():
    rng = np.random.default_rng(1)
    n = 600
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    scores = 1 / (1 + np.exp(-(z + rng.normal(0, 0.5, n))))

    for objective in ("f1", "balanced_accuracy"):
        t = choose_threshold(y, scores, objective=objective)
        m = detection_metrics(y, scores, threshold=t)
        assert m["tp"] + m["fp"] > 0 and m["tn"] + m["fn"] > 0
        assert scores.min() <= t <= scores.max()


def test_choose_threshold_falls_back_on_a_single_class():
    rng = np.random.default_rng(2)
    assert choose_threshold(np.ones(50, dtype=int), rng.random(50)) == 0.5
    assert choose_threshold(np.zeros(50, dtype=int), rng.random(50)) == 0.5


def test_choose_threshold_prefers_the_lower_false_alarm_rate_on_a_tie():
    """The f1 objective subtracts 0.05 * fpr to break ties toward usability."""
    y = np.array([1] * 10 + [0] * 10)
    # two thresholds give the same F1; the one with fewer false alarms must win
    scores = np.concatenate([np.linspace(0.6, 0.9, 10), np.linspace(0.1, 0.4, 10)])
    t = choose_threshold(y, scores, objective="f1")
    assert detection_metrics(y, scores, threshold=t)["fpr"] == 0.0


# =========================================================================
# run_benchmark — a run in which the "full" model loses
# =========================================================================

@pytest.fixture(scope="module")
def losing_run():
    log = []
    strong, weak = Recorder(strong_scores, log), Recorder(weak_scores, log)
    train, validation, calibration, test = (
        make_split(400, 1), make_split(250, 2), make_split(260, 3), make_split(300, 4)
    )
    entries = [
        ModelEntry("strong_baseline", strong.fit, strong.predict_proba, is_baseline=True),
        ModelEntry("noise_model", weak.fit, weak.predict_proba, is_baseline=False),
    ]
    results = run_benchmark(
        entries, train, validation, calibration, test, n_resamples=60, seed=42
    )
    return results, log, {"train": train, "validation": validation,
                         "calibration": calibration, "test": test}


def test_test_split_is_scored_only_after_fitting_and_tuning(losing_run):
    """Fit on train, calibrate on calibration, tune on validation, then test.

    Sizes are distinct so the call log identifies which split each call saw.
    """
    _, log, splits = losing_run
    n = {k: len(v) for k, v in splits.items()}
    assert len({n["train"], n["validation"], n["calibration"], n["test"]}) == 4

    per_model = [log[:4], log[4:]]
    for calls in per_model:
        kinds = [k for k, _ in calls]
        sizes = [s for _, s in calls]
        assert kinds == ["fit", "predict", "predict", "predict"]
        assert sizes[0] == n["train"]
        assert sizes[1] == n["calibration"]
        assert sizes[2] == n["validation"]
        assert sizes[3] == n["test"], "test was not the last split touched"


def test_reported_threshold_is_the_one_chosen_on_validation(losing_run):
    """Recomputed independently: calibrate on calibration, then choose the
    threshold on temperature-scaled VALIDATION scores. Test never enters."""
    results, _, splits = losing_run
    validation, calibration = splits["validation"], splits["calibration"]

    for name, score_fn in (("strong_baseline", strong_scores), ("noise_model", weak_scores)):
        scaler = TemperatureScaler().fit_from_probs(
            score_fn(calibration.X), calibration.y_current
        )
        val_p = scaler.transform_probs(score_fn(validation.X))
        expected = choose_threshold(validation.y_current, val_p)

        assert results["models"][name]["threshold_from_validation"] == pytest.approx(expected)
        assert results["models"][name]["calibration"]["temperature"] == pytest.approx(
            scaler.temperature
        )


def test_verdict_says_so_when_a_baseline_beats_the_full_model(losing_run):
    """Spec 62: if a baseline wins, that is the result and it is reported."""
    results, _, _ = losing_run
    c = results["comparison"]

    assert c["best_baseline"] == "strong_baseline"
    assert c["best_model"] == "noise_model"
    assert c["best_baseline_pr_auc"] > c["best_model_pr_auc"]
    assert c["delta"] == pytest.approx(c["best_model_pr_auc"] - c["best_baseline_pr_auc"])
    assert c["delta"] < 0
    assert c["model_beats_baseline"] is False
    assert c["beats_baseline_outside_ci"] is False
    assert c["verdict"] == "baseline wins; report this (spec 62)"
    assert "baseline wins" in markdown_table(results)


def test_benchmark_reports_contract_sizes_and_group_intervals(losing_run):
    results, _, splits = losing_run

    assert results["contract"]["window_seconds"] == 2.0
    assert results["contract"]["forecast_steps"] == 5
    assert results["split_sizes"] == {k: len(v) for k, v in splits.items()}
    assert results["positive_rate"]["test"] == pytest.approx(splits["test"].y_current.mean())

    for r in results["models"].values():
        ci = r["pr_auc_ci"]
        assert ci["n_groups"] == 8                          # the 8 capture groups, not 300 rows
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
        assert r["detection"]["false_alarms_per_hour"] == pytest.approx(
            r["detection"]["fp"] / 2.0
        )
        assert r["calibration"]["before"]["ece"] >= 0.0
        assert r["calibration"]["after"]["ece"] >= 0.0


def test_markdown_table_has_one_row_per_model(losing_run):
    results, _, _ = losing_run
    table = markdown_table(results)
    lines = table.splitlines()

    pipe_rows = [l for l in lines if l.startswith("|")]
    header, separator, body = pipe_rows[0], pipe_rows[1], pipe_rows[2:]

    assert "PR-AUC" in header and separator.startswith("|---")
    assert len(body) == len(results["models"]) == 2
    for name in results["models"]:
        assert any(line.startswith(f"| {name}") for line in body)

    # the full model is marked, baselines are not
    assert sum("**(full)**" in line for line in body) == 1
    assert "**Verdict:**" in table

    # it scales with the number of models
    one = {**results, "models": {"strong_baseline": results["models"]["strong_baseline"]}}
    assert len([l for l in markdown_table(one).splitlines() if l.startswith("|")]) == 3


def test_benchmark_results_are_json_serialisable(losing_run, tmp_path):
    import json

    results, _, _ = losing_run
    path = save(results, tmp_path / "nested" / "benchmark.json")
    assert path.exists()
    assert set(json.loads(path.read_text())["models"]) == set(results["models"])


# =========================================================================
# the other verdict branches
# =========================================================================

def test_verdict_calls_a_win_decisive_only_when_the_ci_clears_the_baseline():
    def entry(pr_auc, ci_low, is_baseline):
        return {
            "is_baseline": is_baseline,
            "detection": {"pr_auc": pr_auc},
            "pr_auc_ci": {"ci_low": ci_low, "ci_high": pr_auc + 0.1},
        }

    decisive = _compare({"b": entry(0.40, 0.30, True), "f": entry(0.80, 0.60, False)})
    assert decisive["model_beats_baseline"] and decisive["beats_baseline_outside_ci"]
    assert decisive["verdict"] == "model beats best baseline, CI excludes it"

    # ahead on the point estimate but the interval still contains the baseline
    marginal = _compare({"b": entry(0.40, 0.30, True), "f": entry(0.50, 0.35, False)})
    assert marginal["model_beats_baseline"] and not marginal["beats_baseline_outside_ci"]
    assert marginal["verdict"] == "model ahead but within CI — not yet decisive"

    assert _compare({"b": entry(0.8, 0.7, True)})["verdict"] == "insufficient entries to compare"
    assert _compare({"f": entry(0.8, 0.7, False)})["verdict"] == "insufficient entries to compare"


def test_full_model_win_is_reported_end_to_end():
    """The same machinery with the labels swapped: the strong scorer is now the
    full model, and the harness must be willing to say it won."""
    log = []
    strong, weak = Recorder(strong_scores, log), Recorder(weak_scores, log)
    results = run_benchmark(
        [
            ModelEntry("noise_baseline", weak.fit, weak.predict_proba, is_baseline=True),
            ModelEntry("full_model", strong.fit, strong.predict_proba, is_baseline=False),
        ],
        make_split(400, 1), make_split(250, 2), make_split(260, 3), make_split(300, 4),
        n_resamples=60, seed=42,
    )
    c = results["comparison"]
    assert c["best_model"] == "full_model" and c["model_beats_baseline"] is True
    assert c["delta"] > 0
    assert c["verdict"] == "model beats best baseline, CI excludes it"
    assert "**(full)**" in markdown_table(results)


def test_degenerate_test_operating_point_is_flagged_not_hidden():
    """A threshold that collapses to one class on test makes precision/recall/F1
    uninformative; the harness must say so instead of printing them plainly."""
    n = 240
    rng = np.random.default_rng(9)

    def flat(X):
        return np.full(X.shape[0], 0.8)

    class Flat:
        def fit(self, X, y):
            return self

        predict_proba = staticmethod(flat)

    m = Flat()
    splits = [make_split(k, s) for k, s in ((300, 11), (n, 12), (n, 13), (n, 14))]
    results = run_benchmark(
        [ModelEntry("constant", m.fit, m.predict_proba, is_baseline=True)],
        *splits, n_resamples=20, seed=7,
    )
    det = results["models"]["constant"]["detection"]
    assert det["tp"] + det["fp"] == 0 or det["tn"] + det["fn"] == 0
    assert det.get("degenerate") is True
    assert "PR-AUC" in det["note"]
