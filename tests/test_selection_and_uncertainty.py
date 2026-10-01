"""The two defects behind Branch A's 26.36 held-out test loss.

1. Checkpoint selection used the uncertainty-weighted validation loss, which
   is not comparable between epochs: every `s_i` in
   `sum_i [exp(-s_i) L_i + s_i]` is a learned parameter that moves, so the
   bare `+ s_i` terms add a drifting constant. Over the 2026-09-21 run that
   constant went from 0 to -7.04, and selecting the single lowest reading
   picked epoch 6 (0.6601, ~half the median of the others), which generalised
   worst.

2. `log_var.clamp(lo, hi)` passes no gradient outside the bounds, so the two
   parameters that drifted past -3 were frozen for the rest of training.
"""
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

from branch_a_gnn_lstm.lstm_multitask import MultiTaskUncertaintyLoss

spec = importlib.util.spec_from_file_location("bra_sel", "scripts/retrain_branch_a_live.py")
bra = importlib.util.module_from_spec(spec)
sys.modules["bra_sel"] = bra
spec.loader.exec_module(bra)


# --- 1. selection ----------------------------------------------------------

def test_composite_is_unaffected_by_the_loss_offset():
    """The whole point: a drifting log-variance must not move the score."""
    early = {"loss": 2.08, "tech_macro_f1": 0.40, "risk_auc": 0.80}
    # same model quality, but the loss has drifted by the -7.04 offset
    late = {"loss": 2.08 - 7.04, "tech_macro_f1": 0.40, "risk_auc": 0.80}
    assert bra._selection_score(early, "composite") == bra._selection_score(late, "composite")
    # whereas val_loss selection is moved entirely by the offset
    assert bra._selection_score(late, "val_loss") > bra._selection_score(early, "val_loss")


def test_composite_prefers_the_better_model_not_the_lower_loss():
    """Epoch 6 won on loss; a genuinely better epoch should win on composite."""
    ep6 = {"loss": 0.6601, "tech_macro_f1": 0.31, "risk_auc": 0.62}
    ep4 = {"loss": 1.2351, "tech_macro_f1": 0.47, "risk_auc": 0.78}
    assert bra._selection_score(ep6, "val_loss") > bra._selection_score(ep4, "val_loss")
    assert bra._selection_score(ep4, "composite") > bra._selection_score(ep6, "composite")


def test_selection_score_is_maximised_and_bounded():
    perfect = {"loss": 0.0, "tech_macro_f1": 1.0, "risk_auc": 1.0}
    useless = {"loss": 0.0, "tech_macro_f1": 0.0, "risk_auc": 0.0}
    assert bra._selection_score(perfect, "composite") == pytest.approx(1.0)
    assert bra._selection_score(useless, "composite") == pytest.approx(0.0)


def test_a_chance_risk_head_scores_exactly_half_of_the_risk_term():
    """AUC 0.5 is chance, so it should contribute 0.25 of a 0.5-weighted term
    -- not look like a good score and not look like a catastrophic one."""
    m = {"tech_macro_f1": 0.0, "risk_auc": 0.5}
    assert bra._selection_score(m, "composite") == pytest.approx(0.25)


def test_val_loss_mode_still_reproduces_the_old_behaviour():
    a = {"loss": 1.0, "tech_macro_f1": 0.9, "risk_auc": 0.9}
    b = {"loss": 2.0, "tech_macro_f1": 0.9, "risk_auc": 0.9}
    assert bra._selection_score(a, "val_loss") > bra._selection_score(b, "val_loss")


def test_macro_f1_mode_ignores_risk():
    a = {"loss": 9.0, "tech_macro_f1": 0.8, "risk_auc": 0.4}
    b = {"loss": 0.1, "tech_macro_f1": 0.2, "risk_auc": 0.99}
    assert bra._selection_score(a, "macro_f1") > bra._selection_score(b, "macro_f1")


# --- 2. the log-variance parameters ----------------------------------------

def _step(u, lr=0.1, n=1):
    opt = torch.optim.SGD(u.parameters(), lr=lr)
    for _ in range(n):
        opt.zero_grad()
        loss, _ = u(torch.tensor(0.01), torch.tensor(0.30), torch.tensor(0.50))
        loss.backward()
        opt.step()
    return u


def test_parameters_never_escape_the_bounds():
    """An optimiser step can push a parameter epsilon past the bound; project_
    is what a training loop calls to keep the SAVED value clean. That window is
    how the 2026-09-21 checkpoint came to store -3.0090."""
    u = MultiTaskUncertaintyLoss()
    _step(u, lr=5.0, n=40)
    u.project_()                      # what the training loop now does
    for name, p in u.named_parameters():
        v = float(p.detach())
        assert MultiTaskUncertaintyLoss.LOG_VAR_MIN - 1e-6 <= v <= MultiTaskUncertaintyLoss.LOG_VAR_MAX + 1e-6, \
            f"{name} escaped to {v}"


def test_a_parameter_pinned_at_the_bound_keeps_a_live_gradient():
    """The regression: at raw=-3.009 the old code gave grad exactly 0 forever."""
    u = MultiTaskUncertaintyLoss()
    with torch.no_grad():
        u.log_var_risk.fill_(-3.009)
        u.log_var_tech.fill_(-3.0243)
    loss, _ = u(torch.tensor(0.01), torch.tensor(0.30), torch.tensor(0.50))
    loss.backward()
    assert float(u.log_var_risk.grad) != 0.0
    assert float(u.log_var_tech.grad) != 0.0


def test_a_pinned_parameter_can_move_back_off_the_bound():
    """log_var_tech was frozen at -3.0243; with a large tech loss it must rise."""
    u = MultiTaskUncertaintyLoss()
    with torch.no_grad():
        u.log_var_tech.fill_(-3.0)
    opt = torch.optim.SGD(u.parameters(), lr=0.05)
    for _ in range(50):
        opt.zero_grad()
        loss, _ = u(torch.tensor(0.01), torch.tensor(5.0), torch.tensor(0.5))
        loss.backward()
        opt.step()
    assert float(u.log_var_tech.detach()) > -3.0 + 1e-3, "stayed pinned; gradient still dead"


def test_metrics_do_not_force_a_host_device_sync():
    """They were seven .item() calls per batch that nothing consumed."""
    u = MultiTaskUncertaintyLoss()
    _, m = u(torch.tensor(0.01), torch.tensor(0.3), torch.tensor(0.5))
    assert m, "metrics dict should not be empty"
    assert all(torch.is_tensor(v) for v in m.values())
    assert not any(v.requires_grad for v in m.values())


def test_a_step_can_leave_a_parameter_out_of_range_until_projected():
    """Pins the reason project_ is public rather than only called in forward."""
    u = MultiTaskUncertaintyLoss()
    _step(u, lr=5.0, n=40)
    out_of_range = [n for n, p in u.named_parameters()
                    if not (MultiTaskUncertaintyLoss.LOG_VAR_MIN <= float(p.detach())
                            <= MultiTaskUncertaintyLoss.LOG_VAR_MAX)]
    assert out_of_range, "expected a post-step parameter outside the bound"
    u.project_()
    assert all(MultiTaskUncertaintyLoss.LOG_VAR_MIN <= float(p.detach())
               <= MultiTaskUncertaintyLoss.LOG_VAR_MAX for p in u.parameters())


def test_forward_bounds_the_weights_even_if_a_parameter_is_out_of_range():
    """Whatever is stored, the weight actually applied is bounded."""
    import math
    u = MultiTaskUncertaintyLoss()
    with torch.no_grad():
        u.log_var_risk.fill_(-50.0)
    _, m = u(torch.tensor(0.01), torch.tensor(0.3), torch.tensor(0.5))
    assert float(m["weight_risk"]) <= math.exp(-MultiTaskUncertaintyLoss.LOG_VAR_MIN) + 1e-6


# --- 3. the risk head as a probability -------------------------------------

def test_composite_uses_auc_not_mae():
    """MAE is median-seeking: on an 82.5%-zero target it rewards predicting
    nothing. AUC rewards ranking attack windows above benign ones."""
    # a head that predicts 0 everywhere: perfect MAE, chance AUC
    useless = {"tech_macro_f1": 0.5, "risk_auc": 0.50, "risk_mae": 0.0}
    # a head that ranks well but is not MAE-optimal
    good = {"tech_macro_f1": 0.5, "risk_auc": 0.90, "risk_mae": 0.30}
    assert bra._selection_score(good, "composite") > bra._selection_score(useless, "composite")


def test_nan_auc_is_treated_as_chance_not_as_zero():
    """A split with one class gives NaN AUC; that must not silently zero the
    score and make every epoch look equally bad."""
    m = {"tech_macro_f1": 0.8, "risk_auc": float("nan")}
    assert bra._selection_score(m, "composite") == pytest.approx(0.5 * 0.8 + 0.5 * 0.5)


def test_risk_warning_fires_at_chance_auc(capsys):
    bra._warn_if_risk_head_useless(
        {"risk_auc": 0.503, "risk_brier": 0.10, "risk_brier_baseline": 0.145}, "unit")
    out = capsys.readouterr().out
    assert "near chance" in out and "unit" in out


def test_risk_warning_fires_when_brier_matches_the_base_rate(capsys):
    base = 0.1752 * (1 - 0.1752)
    bra._warn_if_risk_head_useless(
        {"risk_auc": 0.80, "risk_brier": base + 1e-6,
         "risk_brier_baseline": base, "risk_base_rate": 0.1752}, "unit")
    assert "no better than" in capsys.readouterr().out


def test_no_risk_warning_for_a_good_head(capsys):
    base = 0.1752 * (1 - 0.1752)
    bra._warn_if_risk_head_useless(
        {"risk_auc": 0.93, "risk_brier": base * 0.5,
         "risk_brier_baseline": base, "risk_base_rate": 0.1752}, "unit")
    assert "WARNING" not in capsys.readouterr().out


def test_bce_objective_targets_the_binarised_label():
    """The magnitude is ~a function of the category, which another head already
    predicts; the risk head is held to the part it can own."""
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    m = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1,
                      num_techniques=14, num_gradations=4, risk_objective="bce")
    x = torch.randn(64, 5, 27)
    # two targets with the same attack/benign pattern but different magnitudes
    pat = (torch.arange(64) % 5 == 0)
    t_a = {"risk": torch.where(pat, torch.full((64,), 0.72), torch.zeros(64)),
           "technique": torch.zeros(64, dtype=torch.long),
           "gradation": torch.zeros(64, dtype=torch.long)}
    t_b = dict(t_a, risk=torch.where(pat, torch.full((64,), 0.96), torch.zeros(64)))
    preds = m(x)
    _, pa = m.compute_loss(preds, t_a)
    _, pb = m.compute_loss(preds, t_b)
    assert float(pa["loss_risk"]) == pytest.approx(float(pb["loss_risk"]), abs=1e-9), \
        "bce must depend only on whether it is an attack, not on the severity value"


def test_smooth_l1_objective_still_sees_the_magnitude():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    m = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1,
                      num_techniques=14, num_gradations=4, risk_objective="smooth_l1")
    x = torch.randn(64, 5, 27)
    pat = (torch.arange(64) % 5 == 0)
    t_a = {"risk": torch.where(pat, torch.full((64,), 0.72), torch.zeros(64)),
           "technique": torch.zeros(64, dtype=torch.long),
           "gradation": torch.zeros(64, dtype=torch.long)}
    t_b = dict(t_a, risk=torch.where(pat, torch.full((64,), 0.96), torch.zeros(64)))
    preds = m(x)
    _, pa = m.compute_loss(preds, t_a)
    _, pb = m.compute_loss(preds, t_b)
    assert float(pa["loss_risk"]) != pytest.approx(float(pb["loss_risk"]), abs=1e-9)


def test_early_warning_selection_needs_both_heads_and_ignores_persistence():
    """Default selection: harmonic mean of macro-F1 and the onset Gini. A head
    that only copies the present (high overall AUC, chance on onset) or a
    collapsed technique head cannot win."""
    good = {"tech_macro_f1": 0.5, "risk_auc": 0.80, "risk_onset_auc": 0.80}
    persistence_like = {"tech_macro_f1": 0.5, "risk_auc": 0.99, "risk_onset_auc": 0.50}
    collapsed = {"tech_macro_f1": 0.02, "risk_auc": 0.95, "risk_onset_auc": 0.95}
    s = lambda m: bra._selection_score(m, "early_warning")   # noqa: E731
    assert s(persistence_like) == 0.0
    assert s(good) > s(collapsed) and s(good) > s(persistence_like)
    assert s({"tech_macro_f1": 1.0, "risk_onset_auc": 1.0}) == pytest.approx(1.0)
