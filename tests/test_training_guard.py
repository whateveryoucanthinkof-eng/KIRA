"""The shared training policy: learn every epoch, step back after 2, stop at 3."""

import math

import pytest

torch = pytest.importorskip("torch")

from cyberworld_v4.training_guard import (  # noqa: E402
    IMPROVED, STEP_BACK, STOP, WAIT, TrainingGuard, default_warmup_steps)


def _setup(lr=0.1, **kw):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    logs = []
    g = TrainingGuard("toy", [m], opt, log=logs.append, **kw)
    return m, opt, g, logs


def _train_step(m, g, scale=1.0):
    x = torch.randn(8, 4)
    loss = (m(x) ** 2).mean() * scale
    return g.backward_step(loss)


def test_improve_then_step_back_at_two_then_stop_at_three():
    m, opt, g, logs = _setup()
    _train_step(m, g)
    assert g.end_epoch(0.50, train_loss=1.0) == IMPROVED
    best_w = m.weight.detach().clone()

    _train_step(m, g)
    assert g.end_epoch(0.49, train_loss=0.9) == WAIT
    assert not torch.equal(m.weight, best_w), "premise: weights moved after the best epoch"

    _train_step(m, g)
    assert g.end_epoch(0.48, train_loss=0.8) == STEP_BACK
    assert torch.equal(m.weight, best_w), "step back must restore the best weights"
    assert g.current_lr() == pytest.approx(0.05), "and halve the learning rate"

    _train_step(m, g)
    assert g.end_epoch(0.47, train_loss=0.8) == STOP
    assert g.should_stop() and "patience 3" in g.stop_reason
    assert any("action: stopped" in line for line in logs)


def test_improving_after_a_step_back_resets_the_count():
    m, opt, g, _ = _setup()
    for s, want in [(0.5, IMPROVED), (0.4, WAIT), (0.4, STEP_BACK), (0.6, IMPROVED), (0.55, WAIT)]:
        _train_step(m, g)
        assert g.end_epoch(s, train_loss=1.0) == want
    assert g.best == 0.6 and g.best_epoch == 4


def test_optimizer_state_is_restored_with_the_weights():
    m, opt, g, _ = _setup()
    _train_step(m, g)
    g.end_epoch(0.5, train_loss=1.0)
    snap = {k: v.clone() for k, v in opt.state_dict()["state"][0].items() if torch.is_tensor(v)}
    for _ in range(2):
        _train_step(m, g)
        g.end_epoch(0.1, train_loss=1.0)
    now = opt.state_dict()["state"][0]
    for k, v in snap.items():
        assert torch.equal(now[k], v), f"optimizer {k} not restored"


def test_a_non_finite_loss_is_skipped_not_stepped():
    m, opt, g, _ = _setup()
    before = m.weight.detach().clone()
    assert g.backward_step(torch.tensor(float("nan"), requires_grad=True)) is False
    assert torch.equal(m.weight, before)
    assert g.n_nonfinite == 1


def test_an_unstable_epoch_steps_back_immediately():
    m, opt, g, logs = _setup(max_nonfinite_frac=0.01)
    _train_step(m, g)
    assert g.end_epoch(0.5, train_loss=1.0) == IMPROVED
    for _ in range(10):
        _train_step(m, g)
    g.backward_step(torch.tensor(float("inf"), requires_grad=True))
    assert g.end_epoch(0.6, train_loss=1.0) == STEP_BACK, "11% non-finite steps is unstable"
    assert any("numerically unstable" in line for line in logs)
    assert g.best == 0.5, "an unstable epoch must never become the best"


def test_a_nan_score_is_unstable():
    m, opt, g, _ = _setup()
    _train_step(m, g)
    g.end_epoch(0.5, train_loss=1.0)
    _train_step(m, g)
    assert g.end_epoch(float("nan"), train_loss=1.0) == STEP_BACK


def test_warmup_ramps_the_learning_rate():
    m, opt, g, _ = _setup(lr=1.0, warmup_steps=10)
    assert g.current_lr() == pytest.approx(0.1)
    lrs = []
    for _ in range(12):
        _train_step(m, g)
        lrs.append(g.current_lr())
    assert lrs == sorted(lrs) and lrs[-1] == pytest.approx(1.0)


def test_clipping_is_applied_and_measured():
    m, opt, g, _ = _setup(clip_norm=1e-3)
    for _ in range(4):
        _train_step(m, g, scale=100.0)
    assert g.n_clipped == 4 and g.grad_norm_max > 1e-3
    g.end_epoch(0.5, train_loss=1.0)
    assert g.history[-1]["clipped_frac"] == 1.0


def test_min_mode_for_losses():
    m, opt, g, _ = _setup(mode="min")
    for s, want in [(1.0, IMPROVED), (0.8, IMPROVED), (0.9, WAIT), (0.85, STEP_BACK)]:
        _train_step(m, g)
        assert g.end_epoch(s, train_loss=s) == want


def test_it_stops_once_the_step_backs_are_used_up():
    m, opt, g, _ = _setup(patience=4, step_back_after=2, max_step_backs=1)
    seq = [(0.5, IMPROVED), (0.4, WAIT), (0.4, STEP_BACK), (0.4, WAIT), (0.4, STOP)]
    for s, want in seq:
        _train_step(m, g)
        assert g.end_epoch(s, train_loss=1.0) == want


def test_diagnosis_distinguishes_overfitting_from_a_flat_loss_and_repeats_health():
    m, opt, g, logs = _setup()
    _train_step(m, g)
    g.end_epoch(0.5, train_loss=1.0)
    _train_step(m, g)
    g.end_epoch(0.4, train_loss=0.8)
    _train_step(m, g)
    g.end_epoch(0.4, train_loss=0.6, health=["technique head predicts 1 of 5 classes"])
    text = "\n".join(logs)
    assert "over-fitting" in text
    assert "technique head predicts 1 of 5 classes" in text

    m, opt, g, logs = _setup()
    for s in (0.5, 0.4, 0.4):
        _train_step(m, g)
        g.end_epoch(s, train_loss=1.0)
    assert "train loss flat" in "\n".join(logs)


def test_step_back_must_come_before_the_stop():
    m = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        TrainingGuard("x", [m], torch.optim.SGD(m.parameters(), lr=0.1), patience=2, step_back_after=2)


def test_summary_is_json_friendly():
    import json
    m, opt, g, _ = _setup()
    _train_step(m, g)
    g.end_epoch(0.5, train_loss=1.0)
    json.dumps(g.summary())


def test_default_warmup_steps():
    assert default_warmup_steps(100_000) == 500
    assert default_warmup_steps(200) == 10
    assert default_warmup_steps(3) == 1
