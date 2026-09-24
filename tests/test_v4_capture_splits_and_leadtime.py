"""The v4 trainer must split on captures and must report lead time.

Two defects this pins, both visible in the committed results/v4_benchmark.json
produced by the previous code:

  * The split group was the bare IP, because `scripts/train_v4.py` built every
    HostId as `offline_host("mixed", "mixed", "mixed", ip)`. That gave 11 train
    hosts / 1 validation / 1 calibration / 1 test, a test base rate of 0.9997,
    and `"ci_low": NaN, "n_groups": 1`. The repository already had a frozen,
    attack-fraction-stratified assignment over 41 captures that no v4 code read.

  * `cyberworld_v4/metrics/earlywarning.py` was written, tested and never
    called, so nothing in the benchmark measured warning before onset -- the
    one thing the product claims.
"""

import numpy as np
import pytest

from cyberworld_v4.metrics.earlywarning import (
    Episode,
    episodes_from_series,
    first_valid_alert,
    lead_time_from_samples,
)
from cyberworld_v4.splits import CALIB, TEST, TRAIN, VAL, frozen_capture_split, partition


# ---------------------------------------------------------------------------
# frozen_capture_split
# ---------------------------------------------------------------------------


class _Sample:
    """Just enough of ForecastSample for the split functions."""

    def __init__(self, capture, t_end, current_attack=0, host="h"):
        self.capture = capture
        self.t_end = float(t_end)
        self.current_attack = int(current_attack)

        class _H:
            key = f"{capture}|{host}"

        self.host = _H()


LOCK = {
    "CIC2018": {
        "a.csv": "train", "b.csv": "train", "c.csv": "train", "d.csv": "train",
        "e.csv": "val", "f.csv": "test",
    },
    "CTU13": {"1/x.binetflow": "train", "2/y.binetflow": "val", "3/z.binetflow": "test"},
}


def _group(s):
    return s.capture


def _corpus():
    out = []
    for t, cap in enumerate(
        ["CIC2018|a.csv", "CIC2018|b.csv", "CIC2018|c.csv", "CIC2018|d.csv",
         "CIC2018|e.csv", "CIC2018|f.csv",
         "CTU13|1/x.binetflow", "CTU13|2/y.binetflow", "CTU13|3/z.binetflow"]
    ):
        out.extend(_Sample(cap, t * 100 + i) for i in range(10))
    return out


def test_every_split_is_populated_and_disjoint():
    samples = _corpus()
    assign = frozen_capture_split(samples, _group, LOCK, time_of=lambda s: s.t_end)
    parts = partition(samples, assign, _group)
    for k in (TRAIN, VAL, CALIB, TEST):
        assert parts[k], f"{k} is empty"
    seen = set()
    for k in (TRAIN, VAL, CALIB, TEST):
        caps = {_group(s) for s in parts[k]}
        assert not (caps & seen), f"{k} shares a capture with an earlier split"
        seen |= caps


def test_validation_and_test_come_from_the_lock_unchanged():
    samples = _corpus()
    assign = frozen_capture_split(samples, _group, LOCK, time_of=lambda s: s.t_end)
    assert set(assign.groups[VAL]) == {"CIC2018|e.csv", "CTU13|2/y.binetflow"}
    assert set(assign.groups[TEST]) == {"CIC2018|f.csv", "CTU13|3/z.binetflow"}


def test_calibration_is_carved_out_of_train_never_out_of_validation():
    """Fitting a temperature on the model-selection split inherits its optimism."""
    samples = _corpus()
    assign = frozen_capture_split(samples, _group, LOCK, calibration_fraction=0.4,
                                  time_of=lambda s: s.t_end)
    lock_train = {"CIC2018|a.csv", "CIC2018|b.csv", "CIC2018|c.csv",
                  "CIC2018|d.csv", "CTU13|1/x.binetflow"}
    assert set(assign.groups[CALIB]) <= lock_train
    assert set(assign.groups[CALIB]).isdisjoint(assign.groups[VAL])
    assert set(assign.groups[CALIB]).isdisjoint(assign.groups[TEST])
    assert set(assign.groups[TRAIN]).isdisjoint(assign.groups[CALIB])


def test_calibration_takes_the_most_recent_train_captures():
    samples = _corpus()
    assign = frozen_capture_split(samples, _group, LOCK, calibration_fraction=0.2,
                                  time_of=lambda s: s.t_end)
    # CTU13|1/x.binetflow has the latest first-timestamp among lock-train captures
    assert assign.groups[CALIB] == ["CTU13|1/x.binetflow"]


def test_train_is_never_emptied_by_a_greedy_calibration_fraction():
    samples = _corpus()
    assign = frozen_capture_split(samples, _group, LOCK, calibration_fraction=0.99,
                                  time_of=lambda s: s.t_end)
    assert assign.groups[TRAIN], "calibration consumed the whole train split"


def test_a_capture_missing_from_the_lock_is_an_error_not_a_silent_drop():
    samples = _corpus() + [_Sample("CIC2018|brand_new.csv", 9999)]
    with pytest.raises(KeyError, match="frozen split lock"):
        frozen_capture_split(samples, _group, LOCK, time_of=lambda s: s.t_end)


def test_the_real_lock_on_disk_is_usable_as_four_splits():
    """The shipped lock must actually be able to fill all four v4 splits."""
    from data_unification.split_policy import load_lock

    lock = load_lock()
    samples, t = [], 0
    for ds, m in lock.items():
        for cap in m:
            samples.extend(_Sample(f"{ds}|{cap}", t + i) for i in range(5))
            t += 100
    assign = frozen_capture_split(samples, _group, lock, time_of=lambda s: s.t_end)
    counts = assign.counts()
    for k in (TRAIN, VAL, CALIB, TEST):
        assert counts[k] >= 1, f"{k} empty on the real lock: {counts}"
    # The whole point: enough test groups for a bootstrap interval to exist.
    assert counts[TEST] >= 5, (
        f"only {counts[TEST]} test captures; group_bootstrap_ci needs >= 5 to "
        f"return an interval instead of NaN"
    )


# ---------------------------------------------------------------------------
# lead time
# ---------------------------------------------------------------------------


def test_episodes_split_at_each_onset():
    t = np.arange(10, dtype=float)
    a = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    eps = episodes_from_series("host", t, a, np.zeros(10))
    assert [e.onset_time for e in eps] == [2.0, 7.0]


def test_an_episode_only_sees_windows_since_the_previous_episode_ended():
    """An alert raised during an earlier attack was about that attack."""
    t = np.arange(10, dtype=float)
    a = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    eps = episodes_from_series("host", t, a, np.arange(10, dtype=float))
    second = eps[1]
    assert second.times.min() >= 4.0, "second episode can see windows from the first"
    assert second.times.max() < second.onset_time


def test_an_attack_starting_at_the_first_window_is_not_an_episode():
    """There is no event-free history to have warned from."""
    t = np.arange(4, dtype=float)
    a = np.array([1, 1, 0, 0])
    assert episodes_from_series("host", t, a, np.zeros(4)) == []


def test_lead_time_is_measured_against_onset_not_taken_from_the_horizon():
    """The old reported 'lead time' was forecast_steps * window_seconds, a constant."""
    ep = Episode("e", onset_time=10.0, times=np.array([4.0, 6.0, 8.0]),
                 scores=np.array([0.1, 0.9, 0.9]))
    assert first_valid_alert(ep.times, ep.scores, threshold=0.5) == 6.0
    assert first_valid_alert(ep.times, ep.scores, threshold=0.5, persistence=2) == 6.0
    assert first_valid_alert(ep.times, ep.scores, threshold=0.99) is None


class _FSample:
    def __init__(self, host_key, t_end, current_attack):
        self.t_end = float(t_end)
        self.current_attack = int(current_attack)

        class _H:
            key = host_key

        self.host = _H()


def test_lead_time_from_samples_finds_a_real_warning():
    # one host: quiet, then rising forecast, then attack at t=8
    ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    cur = [0, 0, 0, 0, 1, 1]
    samples = [_FSample("ds|cap|h1", t, c) for t, c in zip(ts, cur)]
    p = np.array([[0.01], [0.02], [0.90], [0.95], [0.99], [0.99]])
    rep = lead_time_from_samples(samples, p, threshold=0.5, window_seconds=2.0)
    assert rep["episodes"] == 1
    assert rep["detected"] == 1
    assert rep["median_lead_seconds"] == pytest.approx(4.0)  # warned at t=4, onset t=8


def test_a_model_that_never_warns_reports_a_miss_not_an_empty_average():
    ts = [0.0, 2.0, 4.0, 6.0]
    cur = [0, 0, 0, 1]
    samples = [_FSample("ds|cap|h1", t, c) for t, c in zip(ts, cur)]
    p = np.zeros((4, 1))
    rep = lead_time_from_samples(samples, p, threshold=0.5, window_seconds=2.0)
    assert rep["episodes"] == 1 and rep["detected"] == 0
    assert rep["detection_rate"] == 0.0
    assert "median_lead_seconds" not in rep
    assert "note" in rep


def test_hosts_do_not_share_episodes():
    samples = (
        [_FSample("ds|cap|h1", t, c) for t, c in zip([0.0, 2.0, 4.0], [0, 0, 1])]
        + [_FSample("ds|cap|h2", t, c) for t, c in zip([0.0, 2.0, 4.0], [0, 0, 1])]
    )
    p = np.array([[0.9], [0.9], [0.9], [0.0], [0.0], [0.0]])
    rep = lead_time_from_samples(samples, p, threshold=0.5, window_seconds=2.0)
    assert rep["episodes"] == 2
    assert rep["detected"] == 1, "one host warned, the other did not"
    assert rep["hosts"] == 2


def test_the_trainer_actually_calls_it():
    """The metric existed and was tested for a whole version with no caller."""
    import ast
    import pathlib

    tree = ast.parse(
        (pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train_v4.py")
        .read_text(encoding="utf-8")
    )
    called = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    for fn in ("lead_time_from_samples", "SplitConformal", "frozen_capture_split",
               "drop_unresolved"):
        assert fn in called, f"train_v4 never calls {fn}"

    # Host identity must not be collapsed. Checked against CALLS, not source
    # text, so the comment explaining the old defect does not trip it.
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "offline_host":
            literals = [a.value for a in n.args if isinstance(a, ast.Constant)]
            assert "mixed" not in literals, (
                "host identity is still collapsed to 'mixed'; the dataset, scenario "
                "and capture are what stop two corpora sharing a private address"
            )
