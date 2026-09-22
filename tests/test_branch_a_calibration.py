"""Temperature scaling, and the interval that was never an interval.

## What `self.temperature` used to be

    self.temperature = nn.Parameter(torch.ones(1))      # in __init__
    ...
    temp = self.temperature.clamp(min=0.2, max=5.0)     # in compute_loss
    scaled_logits = predictions["technique_logits"] / temp

That is not temperature scaling. Guo et al. fit a single T by minimising NLL
on held-out data **with the model frozen**, precisely because a network is
overconfident on the data it was fitted to. Training T through the training
objective moves it toward whatever sharpness minimises the training loss --
the opposite correction. The shipped checkpoint settled at T = 0.9169, i.e. it
learned to sharpen.

And it was applied in the loss but dropped at inference: `forward` returned
raw logits, which `correlation/trajectory_assembler.py:163` softmaxes and
reports the max of as an operator-facing confidence. Trained under
logits/0.9169, served under logits.

## What `predict_calibrated_risk` used to be

    calibrated_risk = where(raw < 0.35, raw * 0.5, raw)
    lower, upper    = calibrated +/- 0.05     # "95% coverage for SOAR"

Three magic numbers, none fitted to anything, presented to an operator as a
calibrated estimate with a coverage guarantee.
"""
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM


def _model(**kw):
    """A small model in eval mode.

    eval() matters: the heads carry nn.Dropout(0.2), so two forward passes on
    the same input differ in train mode and every "the temperature did / did
    not change this" comparison would be measuring dropout noise.
    """
    kw.setdefault("input_dim", 27)
    kw.setdefault("hidden_dim", 16)
    kw.setdefault("num_layers", 1)
    kw.setdefault("num_techniques", 8)
    return MultiTaskLSTM(**kw).eval()


def _calibratable_logits(n=40_000, k=8, scale=2.0, inflate=1.0, seed=3):
    """Logits with labels genuinely drawn from them, scaled by `inflate`.

    Random labels against random logits have no finite NLL-optimal
    temperature -- the objective just wants T as large as the grid allows --
    so a fit on them correctly refuses, and a test built that way is testing
    the refusal, not the fit.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g) * scale
    labels = torch.multinomial(z.softmax(-1), 1, generator=g).squeeze(1)
    return z * inflate, labels


# --- temperature is no longer trained --------------------------------------

def test_temperature_is_not_a_trainable_parameter():
    """The regression: Adam updated it on every batch of the training loop."""
    m = _model()
    assert "temperature" not in dict(m.named_parameters())
    assert not any(p is m.temperature for p in m.parameters())


def test_the_training_loss_does_not_depend_on_the_temperature():
    """Even with a fitted T installed, the objective must see raw logits --
    otherwise T is an extra scale parameter on the head, and gradient descent
    moves it toward the training-optimal sharpness."""
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(16, 5, 27)
    batch = {"risk": torch.rand(16),
             "technique": torch.randint(0, 8, (16,)),
             "gradation": torch.randint(0, 4, (16,))}
    preds = m(x)
    before = float(m.compute_loss(preds, batch)[1]["loss_tech"])
    m.temperature.fill_(2.5)
    m.temperature_fitted.fill_(1.0)
    preds2 = m(x)
    after = float(m.compute_loss(preds2, batch)[1]["loss_tech"])
    assert before == pytest.approx(after, abs=1e-7)


def test_an_unfitted_temperature_is_inert():
    m = _model()
    m.temperature.fill_(0.9169)          # the legacy jointly-trained value
    out = m(torch.randn(4, 5, 27))
    assert torch.allclose(out["technique_logits"], out["technique_logits_raw"])


def test_a_fitted_temperature_reaches_the_served_logits():
    """Serving softmaxes `technique_logits` and cannot be asked to divide by a
    temperature it does not know about, so `forward` has to apply it."""
    m = _model()
    m.fit_temperature(*_calibratable_logits(inflate=2.0))
    assert float(m.temperature_fitted) == 1.0
    out = m(torch.randn(4, 5, 27))
    assert not torch.allclose(out["technique_logits"], out["technique_logits_raw"])
    assert torch.allclose(out["technique_logits"] * float(m.temperature),
                          out["technique_logits_raw"], atol=1e-5)


def test_temperature_never_changes_a_prediction():
    """T > 0 divides monotonically, so accuracy, macro F1 and the confusion
    matrix are all invariant. Only the probabilities move."""
    m = _model()
    x = torch.randn(64, 5, 27)
    raw = m(x)["technique_logits"].argmax(-1)
    m.temperature.fill_(3.7)
    m.temperature_fitted.fill_(1.0)
    assert torch.equal(m(x)["technique_logits"].argmax(-1), raw)


@pytest.mark.parametrize("true_T", [0.5, 1.5, 3.0])
def test_the_fit_recovers_an_injected_temperature(true_T):
    """Sample labels from softmax(z), then present the model with z*T. The
    NLL-optimal temperature is then exactly T; if the fit does not find it,
    it is not fitting anything."""
    torch.manual_seed(3)
    m = _model()
    z = torch.randn(60_000, 8) * 2.0
    labels = torch.multinomial(z.softmax(-1), 1).squeeze(1)
    rep = m.fit_temperature(z * true_T, labels)
    assert rep["fitted"]
    assert abs(rep["temperature"] - true_T) / true_T < 0.10, rep


def test_the_fit_lowers_nll():
    torch.manual_seed(4)
    m = _model()
    z = torch.randn(40_000, 8) * 2.0
    labels = torch.multinomial(z.softmax(-1), 1).squeeze(1)
    rep = m.fit_temperature(z * 2.5, labels)
    assert rep["nll_after"] <= rep["nll_before"]


def test_a_boundary_hit_is_refused_rather_than_recorded_as_a_fit():
    """T on the end of the grid means the objective wanted to go further and
    was stopped. Recording it would put an arbitrary boundary value into
    serving."""
    m = _model()
    # logits so extreme that the NLL keeps falling as T grows
    z = torch.randn(5000, 8) * 400.0
    labels = torch.randint(0, 8, (5000,))
    rep = m.fit_temperature(z, labels)
    assert rep["fitted"] is False
    assert "boundary" in rep["reason"]
    assert float(m.temperature_fitted) == 0.0, "must not install a boundary value"


def test_a_single_class_calibration_split_is_refused():
    m = _model()
    rep = m.fit_temperature(torch.randn(500, 8), torch.zeros(500, dtype=torch.long))
    assert rep["fitted"] is False and "one class" in rep["reason"]


def test_top_label_ece_is_zero_for_a_perfectly_calibrated_head():
    """A head that says 100% and is right 100% of the time has ECE 0."""
    probs = torch.zeros(1000, 4)
    probs[:, 0] = 1.0
    labels = torch.zeros(1000, dtype=torch.long)
    assert MultiTaskLSTM.top_label_ece(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_top_label_ece_measures_the_gap_it_claims_to():
    """Confidence 0.9 on every sample, correct on 50% of them -> ECE 0.4."""
    n = 1000
    probs = torch.full((n, 4), 0.1 / 3)
    probs[:, 0] = 0.9
    labels = torch.zeros(n, dtype=torch.long)
    labels[n // 2:] = 1
    assert MultiTaskLSTM.top_label_ece(probs, labels) == pytest.approx(0.4, abs=1e-6)


# --- backwards compatibility with what is on disk --------------------------

def test_a_checkpoint_without_the_new_buffers_still_loads_strictly():
    """`control_backend/model_adapter.py` calls load_state_dict with the
    default strict=True. A new buffer that is not in a shipped checkpoint
    would be a missing key, and adding it would have taken serving down."""
    m = _model()
    legacy = {k: v for k, v in m.state_dict().items()
              if k not in ("temperature_fitted", "risk_conformal_halfwidth")}
    legacy["temperature"] = torch.tensor([0.9169])
    fresh = _model()
    fresh.load_state_dict(legacy)          # strict
    assert float(fresh.temperature) == pytest.approx(0.9169)
    assert float(fresh.temperature_fitted) == 0.0


def test_a_legacy_jointly_trained_temperature_is_not_silently_applied():
    """0.9169 was fitted to nothing. Loading it must not change what the model
    currently serves."""
    m = _model()
    legacy = {k: v for k, v in m.state_dict().items() if k != "temperature_fitted"}
    legacy["temperature"] = torch.tensor([0.9169])
    m.load_state_dict(legacy)
    out = m(torch.randn(4, 5, 27))
    assert torch.allclose(out["technique_logits"], out["technique_logits_raw"])


def test_a_fitted_checkpoint_round_trips_through_the_serving_constructor():
    """Serving builds the model with defaults and loads the state dict; the
    fitted temperature has to survive that, or it never reaches an operator."""
    m = _model()
    m.fit_temperature(*_calibratable_logits(inflate=2.0))
    assert float(m.temperature_fitted) == 1.0, "precondition: the fit succeeded"
    served = _model()
    served.load_state_dict(m.state_dict())
    assert float(served.temperature_fitted) == 1.0
    assert float(served.temperature) == pytest.approx(float(m.temperature))
    x = torch.randn(4, 5, 27)
    assert torch.allclose(served(x)["technique_logits"], m(x)["technique_logits"])


@pytest.mark.parametrize("name", [
    "branch_a_lstm.pt", "branch_a_lstm.full_corpus.pt", "branch_a_lstm.ctu13_only.pt",
])
def test_a_checkpoint_on_disk_applies_a_temperature_only_if_it_earned_one(name):
    """The real artifacts, through the real serving constructor.

    `control_backend/model_adapter.py:175` builds
    `MultiTaskLSTM(input_dim=27, hidden_dim=64)` and calls `load_state_dict`
    with the default `strict=True`. Every checkpoint written before
    `temperature_fitted` and `risk_conformal_halfwidth` existed lacks those
    keys, so without `_load_from_state_dict` supplying them this load raises
    and serving is down. The synthetic version of this test cannot catch a
    checkpoint whose real key set differs, which is the case that would
    actually page someone.

    The assertion is the biconditional, not today's state: a temperature
    reaches the served logits **iff** it was fitted post-hoc. Asserting
    "T is inert" outright would start failing the moment a retrain lands a
    correctly fitted one, which is the outcome this whole change is for.

    As of 2026-09-22 all three carry a jointly-trained T (0.9169 / 0.6896 /
    0.7353 -- every one below 1.0, i.e. each learned to *sharpen* an already
    overconfident head) and none records a fit, so all three take the inert
    branch.
    """
    import os
    path = os.path.join("saved_models", "branch_a", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    served = MultiTaskLSTM(input_dim=27, hidden_dim=64).eval()
    served.load_state_dict(state, strict=True)      # strict: the serving contract

    out = served(torch.randn(2, 15, 27))
    fitted = float(served.temperature_fitted) > 0.0

    if not fitted:
        assert float(served.effective_temperature) == 1.0
        assert torch.equal(out["technique_logits"], out["technique_logits_raw"]), (
            f"{name}: a temperature that was fitted to nothing reached the "
            f"served logits")
    else:
        rep = ckpt.get("technique_calibration") or {}
        assert rep.get("fitted") is True, (
            f"{name} claims temperature_fitted=1 but carries no record of the "
            f"fit; a T nobody can audit is the defect this replaced")
        assert rep.get("at_grid_boundary") is False, (
            f"{name} recorded a boundary temperature as a fit")
        assert rep["ece_after"] <= rep["ece_before"] + 1e-9, (
            f"{name}: the fitted T made top-label ECE worse "
            f"({rep['ece_before']:.5f} -> {rep['ece_after']:.5f})")
        assert float(served.effective_temperature) == pytest.approx(
            rep["temperature"], rel=1e-5)


# --- predict_calibrated_risk -----------------------------------------------

def test_an_unfitted_interval_raises_instead_of_inventing_one():
    m = _model()
    with pytest.raises(RuntimeError, match="fitted on held-out data"):
        m.predict_calibrated_risk(torch.randn(4, 5, 27))


def test_the_error_names_the_number_it_refuses_to_default():
    m = _model()
    with pytest.raises(RuntimeError) as e:
        m.predict_calibrated_risk(torch.randn(2, 5, 27))
    assert "0.05" in str(e.value) and "95%" in str(e.value)


def test_an_explicit_half_width_is_still_allowed():
    """Refusing a default is not refusing the caller; it is refusing to make
    the choice silently."""
    m = _model()
    out = m.predict_calibrated_risk(torch.randn(4, 5, 27), conformal_error=0.05)
    assert float(out["conformal_half_width"][0]) == pytest.approx(0.05)


def test_the_arbitrary_half_gate_below_035_is_gone():
    """`raw * 0.5` below 0.35 destroyed the calibration the head is trained to
    have -- a halved probability is not the probability of anything."""
    m = _model()
    m.set_risk_conformal_halfwidth(0.3)
    # Push the head's output into the band the old code halved, so the test
    # exercises the gate rather than happening to miss it.
    with torch.no_grad():
        m.risk_head[-2].bias.fill_(-2.0)
        m.risk_head[-2].weight.zero_()
    out = m.predict_calibrated_risk(torch.randn(32, 5, 27))
    assert (out["risk_score"] < 0.35).all(), "precondition: inside the gated band"
    assert torch.equal(out["calibrated_risk"], out["risk_score"])


def test_the_fitted_conformal_interval_covers_what_it_claims():
    m = _model()
    p = torch.rand(20_000)
    y = (torch.rand(20_000) < 0.1752).float()
    rep = m.fit_risk_conformal(p, y, alpha=0.05)
    assert rep["fitted"]
    assert rep["empirical_coverage_on_calibration"] >= 0.95 - 1e-6
    assert rep["half_width"] > 0.5, "a 95% interval on a binary target is wide"


def test_the_conformal_fit_uses_the_finite_sample_correction():
    """ceil((n+1)(1-alpha))/n, not the plain empirical quantile -- that
    correction is what makes the coverage exact rather than approximate."""
    m = _model()
    n, alpha = 999, 0.05
    resid = torch.linspace(0.0, 1.0, n)
    rep = m.fit_risk_conformal(resid, torch.zeros(n), alpha=alpha)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    assert rep["half_width"] == pytest.approx(float(resid[k - 1]), abs=1e-6)


def test_the_conformal_quantile_agrees_with_the_canonical_implementation():
    """`fit_risk_conformal`'s docstring claims it uses the same corrected
    quantile as `cyberworld_v4.conformal.conformal_quantile`. Two copies of a
    finite-sample correction are two chances to get the off-by-one wrong, so
    the claim is pinned rather than trusted. `conformal_quantile` returns inf
    where there are too few points; the model refuses instead, which is the
    same decision reported differently.
    """
    conformal_quantile = pytest.importorskip(
        "cyberworld_v4.conformal").conformal_quantile
    g = torch.Generator().manual_seed(1)
    for _ in range(40):
        n = int(torch.randint(20, 4000, (1,), generator=g))
        alpha = float(torch.rand(1, generator=g)) * 0.3 + 0.01
        pred = torch.rand(n, generator=g)
        true = torch.rand(n, generator=g)
        rep = _model().fit_risk_conformal(pred, true, alpha=alpha)
        ref = conformal_quantile((pred - true).abs().numpy(), alpha)
        if not rep["fitted"]:
            assert not math.isfinite(ref), (
                f"model refused n={n} alpha={alpha:.4f} but the canonical "
                f"implementation returned a finite {ref}")
            continue
        assert rep["half_width"] == pytest.approx(ref, abs=1e-6), (
            f"n={n} alpha={alpha:.4f}: {rep['half_width']} vs {ref}")


def test_conformal_coverage_holds_on_fresh_data_not_just_the_calibration_set():
    """The coverage that matters is on data the quantile was NOT fitted to.

    `empirical_coverage_on_calibration` is >= 1-alpha by construction -- it is
    the definition of the order statistic, so asserting it proves only that
    sorting works. The guarantee split conformal actually makes is about
    exchangeable *future* points. Known noise, so the answer is checkable:
    for residuals |N(0, sigma)| the exact 95% half-width is
    sigma * 1.959964, and coverage on a fresh draw must land at ~95%.
    """
    g = torch.Generator().manual_seed(11)
    sigma = 0.3
    f_cal = torch.rand(5000, generator=g)
    y_cal = f_cal + torch.randn(5000, generator=g) * sigma
    f_te = torch.rand(20_000, generator=g)
    y_te = f_te + torch.randn(20_000, generator=g) * sigma

    m = _model()
    rep = m.fit_risk_conformal(f_cal, y_cal, alpha=0.05)
    assert rep["fitted"]
    hw = rep["half_width"]
    assert hw == pytest.approx(sigma * 1.959964, abs=0.05), (
        f"half-width {hw:.4f} is not the 95% quantile of |N(0,{sigma})|")

    coverage = float(((y_te - f_te).abs() <= hw).float().mean())
    assert 0.93 <= coverage <= 0.97, (
        f"held-out coverage {coverage:.4f} is not the 95% that was promised")

    # And the number the old code hardcoded, on the same data, for contrast.
    fake = float(((y_te - f_te).abs() <= 0.05).float().mean())
    assert fake < 0.30, (
        f"sanity: a hardcoded +/-0.05 should cover far less than 95% here, "
        f"got {fake:.4f}")


def test_too_few_calibration_points_refuses_rather_than_returning_infinity():
    m = _model()
    rep = m.fit_risk_conformal(torch.rand(10), torch.zeros(10), alpha=0.01)
    assert rep["fitted"] is False and "cannot support" in rep["reason"]
    assert float(m.risk_conformal_halfwidth) != float(m.risk_conformal_halfwidth)


def test_an_alert_flag_is_only_produced_when_a_threshold_is_given():
    """Suppression is a decision and belongs at a fitted operating point, not
    inside the model's estimate. It is available, never assumed."""
    m = _model()
    m.set_risk_conformal_halfwidth(0.2)
    assert "alert" not in m.predict_calibrated_risk(torch.randn(4, 5, 27))
    out = m.predict_calibrated_risk(torch.randn(4, 5, 27), threshold=0.25)
    assert out["alert"].dtype == torch.bool


# --- the soft-BCE objective, for the hazard target -------------------------

def test_soft_bce_tracks_a_continuous_target_where_bce_binarises_it():
    """Against `hazard = exp(-dt/tau)`, `bce` asks "is this host ever attacked
    later", which is nearly constant and carries none of the timing."""
    torch.manual_seed(1)
    x = torch.randn(64, 5, 27)
    near = torch.full((64,), 0.90)       # attack imminent
    far = torch.full((64,), 0.05)        # attack far away
    hard = {"technique": torch.zeros(64, dtype=torch.long),
            "gradation": torch.zeros(64, dtype=torch.long)}

    m_bce = _model(risk_objective="bce")
    m_soft = _model(risk_objective="soft_bce")
    m_soft.load_state_dict(m_bce.state_dict())
    preds = m_bce(x)

    b_near = float(m_bce.compute_loss(preds, dict(hard, risk=near))[1]["loss_risk"])
    b_far = float(m_bce.compute_loss(preds, dict(hard, risk=far))[1]["loss_risk"])
    assert b_near == pytest.approx(b_far, abs=1e-9), \
        "bce cannot tell 'imminent' from 'distant' -- both are just > 0"

    s_near = float(m_soft.compute_loss(preds, dict(hard, risk=near))[1]["loss_risk"])
    s_far = float(m_soft.compute_loss(preds, dict(hard, risk=far))[1]["loss_risk"])
    assert s_near != pytest.approx(s_far, abs=1e-6)


def test_soft_bce_is_minimised_where_the_prediction_equals_the_target():
    """The property that makes it a proper scoring rule for a target in [0,1]:
    cross-entropy between two Bernoullis is minimised at p == y."""
    y = 0.37
    losses = {}
    for p in (0.10, 0.25, 0.37, 0.50, 0.80):
        pt = torch.full((256,), p)
        yt = torch.full((256,), y)
        losses[p] = float(F.binary_cross_entropy(pt.clamp(1e-6, 1 - 1e-6), yt))
    assert min(losses, key=losses.get) == 0.37


def test_an_unknown_risk_objective_is_rejected_at_construction():
    with pytest.raises(ValueError, match="soft_bce"):
        _model(risk_objective="mse")
