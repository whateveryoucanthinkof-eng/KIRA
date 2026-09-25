"""v4 prediction-head semantics.

v3 had one scalar called "risk" that was variously described as risk,
confidence, hazard and calibrated risk, trained with BCE against a hand-made
severity lookup, and explained by running the model in train() mode with
dropout live. Each test here pins one of the properties that replaced that:

  * five heads with distinct shapes and distinct losses
  * predict() is deterministic regardless of the module's mode, and restores it
  * cumulative onset is derived as 1 - prod(1 - h), never max(h)
  * the hazard term is masked by at_risk, so censored rows contribute nothing
  * severity is a SmoothL1 regression target, not a probability
  * the world model emits a distribution, and stabilisation is opt-in
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from cyberworld_v4.config import DEFAULT_CONFIG  # noqa: E402
from cyberworld_v4.models import (  # noqa: E402
    CyberWorldForecaster,
    DistributionalWorldModel,
    SequenceEncoder,
    forecast_loss,
    gaussian_nll,
)

K = DEFAULT_CONFIG.temporal.forecast_steps          # 5
L = DEFAULT_CONFIG.temporal.history_steps           # 15
D = DEFAULT_CONFIG.state_dim                        # 27
N_TECH = 7
BATCH = 6


@pytest.fixture(scope="module")
def forecaster():
    torch.manual_seed(0)
    # dropout high on purpose: a non-deterministic predict() would show up loudly
    return CyberWorldForecaster(N_TECH, DEFAULT_CONFIG, hidden_dim=32, layers=2, dropout=0.5)


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, L, D)


# =========================================================================
# shapes
# =========================================================================

def test_forward_emits_all_five_heads_with_the_right_shapes(forecaster, x):
    out = forecaster(x)
    assert set(out) == {
        "current_attack_logit",
        "future_attack_logits",
        "hazard_logits",
        "technique_logits",
        "severity",
    }
    assert out["current_attack_logit"].shape == (BATCH,)
    assert out["future_attack_logits"].shape == (BATCH, K)
    assert out["hazard_logits"].shape == (BATCH, K)
    assert out["technique_logits"].shape == (BATCH, K, N_TECH)   # multilabel per step
    assert out["severity"].shape == (BATCH,)


def test_heads_emit_logits_not_probabilities(forecaster, x):
    """Sigmoid belongs at the metric/serving boundary so temperature scaling has
    logits to work on."""
    forecaster.eval()
    out = forecaster(x)
    pred = forecaster.predict(x)
    assert torch.allclose(torch.sigmoid(out["hazard_logits"]), pred["hazard"], atol=1e-6)
    assert torch.allclose(
        torch.sigmoid(out["current_attack_logit"]), pred["current_attack"], atol=1e-6
    )
    # severity passes through untouched: it is not a probability
    assert torch.allclose(out["severity"], pred["severity"], atol=1e-6)


def test_encoder_reduces_the_sequence_to_one_vector():
    enc = SequenceEncoder(D, hidden_dim=32, layers=2, dropout=0.1)
    enc.eval()
    assert enc(torch.randn(4, L, D)).shape == (4, 32)


# =========================================================================
# predict(): the v3 explainability bug
# =========================================================================

def test_predict_is_deterministic_even_in_train_mode(forecaster, x):
    """v3's explainability ran the model in train() mode with dropout active,
    so the same input produced different explanations each call."""
    forecaster.train()
    assert forecaster.training

    a = forecaster.predict(x)
    b = forecaster.predict(x)
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} changed between identical predict() calls"

    # and it matches what an explicitly-eval model produces
    forecaster.eval()
    c = forecaster.predict(x)
    for key in a:
        assert torch.allclose(a[key], c[key], atol=1e-6)


def test_predict_restores_the_prior_module_mode(forecaster, x):
    forecaster.train()
    forecaster.predict(x)
    assert forecaster.training, "predict() left the model in eval mode"

    forecaster.eval()
    forecaster.predict(x)
    assert not forecaster.training, "predict() flipped an eval model into train mode"


def test_predict_restores_train_mode_even_if_the_forward_raises(forecaster):
    forecaster.train()
    with pytest.raises(Exception):
        forecaster.predict(torch.randn(BATCH, L, D + 3))     # wrong feature width
    assert forecaster.training, "an exception inside predict() left the mode clobbered"


def test_predict_does_not_build_a_graph(forecaster, x):
    forecaster.eval()
    assert forecaster.predict(x)["current_attack"].requires_grad is False


# =========================================================================
# cumulative onset
# =========================================================================

def test_cumulative_onset_is_monotone_non_decreasing(forecaster, x):
    forecaster.eval()
    cum = forecaster.predict(x)["cumulative_onset"]
    assert cum.shape == (BATCH, K)
    assert torch.all(cum[:, 1:] - cum[:, :-1] >= -1e-6)
    assert torch.all((cum >= 0.0) & (cum <= 1.0))


def test_cumulative_onset_is_the_product_form_never_the_max(forecaster, x):
    forecaster.eval()
    p = forecaster.predict(x)
    h = p["hazard"]
    expected = 1.0 - torch.cumprod(1.0 - h, dim=1)
    assert torch.allclose(p["cumulative_onset"], expected, atol=1e-6)
    # the accumulated probability must exceed the single largest hazard
    assert torch.all(p["cumulative_onset"][:, -1] >= h.max(dim=1).values - 1e-6)
    assert not torch.allclose(p["cumulative_onset"][:, -1], h.max(dim=1).values, atol=1e-4)


# =========================================================================
# forecast_loss
# =========================================================================

def _out(hazard_logits, severity=None):
    return {
        "current_attack_logit": torch.zeros(BATCH),
        "future_attack_logits": torch.zeros(BATCH, K),
        "hazard_logits": hazard_logits,
        "technique_logits": torch.zeros(BATCH, K, N_TECH),
        "severity": torch.zeros(BATCH) if severity is None else severity,
    }


def _batch(at_risk, severity=0.5, hazard_target=None):
    return {
        "current_attack": torch.zeros(BATCH),
        "future_attack": torch.zeros(BATCH, K),
        "hazard_target": torch.zeros(BATCH, K) if hazard_target is None else hazard_target,
        "at_risk": at_risk,
        "future_techniques": torch.zeros(BATCH, K, N_TECH),
        "severity": torch.full((BATCH,), float(severity)),
    }


def test_hazard_loss_is_zero_when_nothing_is_at_risk():
    """Once a host's onset has occurred it leaves the risk set. Counting those
    rows as hazard negatives would train the model to call ongoing attacks safe.
    """
    torch.manual_seed(2)
    batch = _batch(at_risk=torch.zeros(BATCH, K))
    total, parts = forecast_loss(_out(torch.randn(BATCH, K) * 4.0), batch)

    assert parts["hazard"] == 0.0
    assert parts["at_risk_fraction"] == 0.0
    assert torch.isfinite(total)


def test_hazard_logits_cannot_move_the_loss_on_a_fully_censored_batch():
    """The mask must remove the gradient path entirely, not merely shrink it."""
    torch.manual_seed(3)
    batch = _batch(at_risk=torch.zeros(BATCH, K))

    a_total, a_parts = forecast_loss(_out(torch.randn(BATCH, K) * 6.0), batch)
    b_total, b_parts = forecast_loss(_out(torch.randn(BATCH, K) * -6.0 + 9.0), batch)

    assert a_parts["hazard"] == b_parts["hazard"] == 0.0
    assert a_parts["total"] == pytest.approx(b_parts["total"])
    assert float(a_total) == pytest.approx(float(b_total))

    # ... and no gradient reaches the hazard head
    logits = torch.randn(BATCH, K, requires_grad=True)
    forecast_loss(_out(logits), batch)[0].backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_hazard_loss_averages_over_at_risk_entries_only():
    torch.manual_seed(4)
    logits = torch.randn(BATCH, K)
    target = (torch.rand(BATCH, K) > 0.7).float()
    at_risk = torch.zeros(BATCH, K)
    at_risk[: BATCH // 2, :3] = 1.0                       # a partial risk set

    _, parts = forecast_loss(_out(logits), _batch(at_risk, hazard_target=target))

    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    expected = float((raw * at_risk).sum() / at_risk.sum())
    assert parts["hazard"] == pytest.approx(expected, rel=1e-5)
    assert parts["at_risk_fraction"] == pytest.approx(float(at_risk.mean()))

    # the censored half is genuinely excluded: perturbing it changes nothing
    perturbed = logits.clone()
    perturbed[BATCH // 2 :, :] += 7.0
    _, parts2 = forecast_loss(_out(perturbed), _batch(at_risk, hazard_target=target))
    assert parts2["hazard"] == pytest.approx(parts["hazard"], rel=1e-5)


def test_severity_is_a_smooth_l1_regression_not_a_bernoulli_likelihood():
    """Severity is an operator ranking aid. A target of 3.7 is meaningful for a
    regression loss and rejected outright by BCE — which is exactly why v3's
    BCE-against-a-severity-lookup gave that score a reading it never had.
    """
    out_of_range = 3.7
    batch = _batch(at_risk=torch.ones(BATCH, K), severity=out_of_range)
    total, parts = forecast_loss(_out(torch.zeros(BATCH, K)), batch)

    assert np.isfinite(parts["severity"])
    assert parts["severity"] == pytest.approx(
        float(F.smooth_l1_loss(torch.zeros(BATCH), torch.full((BATCH,), out_of_range)))
    )
    assert torch.isfinite(total)

    with pytest.raises(RuntimeError):
        F.binary_cross_entropy(torch.full((BATCH,), 0.5), torch.full((BATCH,), out_of_range))


def test_loss_weights_apply_and_total_is_the_weighted_sum():
    torch.manual_seed(5)
    batch = _batch(at_risk=torch.ones(BATCH, K), hazard_target=torch.ones(BATCH, K))
    out = _out(torch.randn(BATCH, K))

    total, parts = forecast_loss(out, batch)
    w = {"current": 1.0, "future": 1.0, "hazard": 1.0, "technique": 1.0, "severity": 0.1}
    expected = sum(w[k] * parts[k] for k in w)
    assert parts["total"] == pytest.approx(expected, rel=1e-5)

    zeroed, zparts = forecast_loss(out, batch, weights={"hazard": 0.0})
    assert zparts["hazard"] == pytest.approx(parts["hazard"])      # still reported
    assert float(zeroed) < float(total)                            # but not in the total


# =========================================================================
# DistributionalWorldModel
# =========================================================================

@pytest.fixture(scope="module")
def world_model():
    torch.manual_seed(6)
    wm = DistributionalWorldModel(
        d_latent=12, d_model=32, n_heads=4, n_layers=1, dropout=0.1, max_context=16
    )
    wm.eval()
    return wm


def test_rollout_returns_a_mean_and_a_logvar_of_shape_bkd(world_model):
    """The spec asks for a distribution over the next state, not a point
    estimate with a hardcoded radius bolted on."""
    torch.manual_seed(7)
    h = torch.randn(3, 8, 12)
    mean, logvar = world_model.rollout(h, K=4)

    assert mean.shape == (3, 4, 12)
    assert logvar.shape == (3, 4, 12)
    assert torch.isfinite(mean).all() and torch.isfinite(logvar).all()
    assert torch.all(logvar >= -10.0) and torch.all(logvar <= 10.0)   # clamped, so exp is safe


def test_rollout_respects_the_context_limit(world_model):
    torch.manual_seed(8)
    # a history longer than max_context must not blow up the attention mask
    mean, logvar = world_model.rollout(torch.randn(2, 20, 12), K=3)
    assert mean.shape == (2, 3, 12) and logvar.shape == (2, 3, 12)


def test_gaussian_nll_is_finite_and_falls_as_the_mean_approaches_the_target(world_model):
    torch.manual_seed(9)
    h = torch.randn(3, 8, 12)
    with torch.no_grad():
        mean, logvar = world_model.rollout(h, K=4)
    target = torch.randn(3, 4, 12)

    far = gaussian_nll(mean, logvar, target)
    half = gaussian_nll(mean + 0.5 * (target - mean), logvar, target)
    exact = gaussian_nll(target, logvar, target)

    assert torch.isfinite(far) and torch.isfinite(half) and torch.isfinite(exact)
    assert float(far) > float(half) > float(exact)

    # a confident (low-variance) model is punished harder for the same error
    assert float(gaussian_nll(mean, torch.full_like(logvar, -2.0), target)) > float(
        gaussian_nll(mean, torch.full_like(logvar, 2.0), target)
    )


def test_stabilize_horizon_is_opt_in_and_actually_changes_the_rollout(world_model):
    """In v3 the 0.95^k damping defaulted ON and no trainer overrode it, so
    predictions were shrunk toward "no change" during training while being
    validated against a persistence baseline. It is inference-only now.
    """
    torch.manual_seed(10)
    h = torch.randn(3, 8, 12)

    plain, _ = world_model.rollout(h, K=5)
    plain_again, _ = world_model.rollout(h, K=5)
    damped, _ = world_model.rollout(h, K=5, stabilize_horizon=True)

    assert torch.allclose(plain, plain_again, atol=1e-6), "eval rollout should be deterministic"
    assert not torch.allclose(plain, damped, atol=1e-5), "stabilize_horizon changed nothing"

    # damping starts at k>0, so the first step is untouched
    assert torch.allclose(plain[:, 0], damped[:, 0], atol=1e-6)
    assert (plain[:, 1:] - damped[:, 1:]).abs().max() > 1e-4

    # a decay of 1.0 is the identity, confirming decay is what does the work
    undamped, _ = world_model.rollout(h, K=5, stabilize_horizon=True, decay=1.0)
    assert torch.allclose(plain, undamped, atol=1e-5)
