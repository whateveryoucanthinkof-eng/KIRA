"""Four things that decide what the reported AUC can be, all previously unused.

  * The 27-D state is an unbounded TGNE latent glued to 15 attributes clipped
    to [0, 1]. An LSTM gate sums them, so without standardisation whichever
    block is larger dominates every gate.
  * `forecast_loss` has always accepted `pos_weight` and nothing ever passed
    one, so the positive class was weighted 1.0 against a base rate the
    shipped run measured at 0.2263.
  * `train_v4` selected the epoch with the lowest multi-task loss -- a sum of
    five terms, four of which the headline numbers do not measure -- while
    reporting ROC-AUC and PR-AUC.
  * There was no early stopping, so the epoch count was the only regulariser.
"""

import ast
import pathlib

import numpy as np
import pytest
import torch

from cyberworld_v4.config import DEFAULT_CONFIG
from cyberworld_v4.models import CyberWorldForecaster, SequenceEncoder, forecast_loss

REPO = pathlib.Path(__file__).resolve().parent.parent
L = DEFAULT_CONFIG.temporal.history_steps
D = DEFAULT_CONFIG.state_dim


def _skewed_batch(n=256, seed=0):
    """A latent block ~100x the scale of the attribute block, which is the
    shape the real 27-D state has: dims 0-11 unbounded, dims 12-26 in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, L, D, generator=g)
    x[:, :, :12] *= 100.0
    return x


# ---------------------------------------------------------------------------
# Input standardisation
# ---------------------------------------------------------------------------


def test_an_unfitted_encoder_is_exactly_the_identity():
    """Old checkpoints must keep behaving as they did."""
    enc = SequenceEncoder(D)
    assert not enc.normalizer_fitted
    assert torch.equal(enc.input_mean, torch.zeros(D))
    assert torch.equal(enc.input_std, torch.ones(D))


def test_fitting_brings_the_two_blocks_onto_one_scale():
    x = _skewed_batch()
    enc = SequenceEncoder(D).fit_input_normalizer(x)
    z = (x - enc.input_mean) / enc.input_std
    lat, att = z[:, :, :12].std(), z[:, :, 12:].std()
    assert enc.normalizer_fitted
    assert 0.5 < float(lat / att) < 2.0, (
        f"blocks still on different scales after fitting: {lat:.3f} vs {att:.3f}"
    )
    # and the raw input really was skewed, or this test proves nothing
    assert float(x[:, :, :12].std() / x[:, :, 12:].std()) > 10


def test_a_constant_feature_is_not_amplified_into_noise():
    """std 0 must not become a near-zero divisor."""
    x = torch.rand(64, L, D)
    x[:, :, 5] = 0.25                      # a perfectly constant attribute
    enc = SequenceEncoder(D).fit_input_normalizer(x)
    assert enc.input_std[5] == 1.0
    z = (x - enc.input_mean) / enc.input_std
    assert torch.isfinite(z).all()


def test_the_statistics_travel_in_the_state_dict():
    """Serving must inherit exactly what training fitted -- a normaliser
    refitted at serve time is the same class of silent mismatch as the zeroed
    node features were."""
    m = CyberWorldForecaster(n_techniques=4).fit_input_normalizer(_skewed_batch())
    sd = m.state_dict()
    assert "encoder.input_mean" in sd and "encoder.input_std" in sd
    fresh = CyberWorldForecaster(n_techniques=4)
    fresh.load_state_dict(sd)
    assert fresh.normalizer_fitted
    assert torch.allclose(fresh.encoder.input_mean, m.encoder.input_mean)


def test_normalisation_does_not_change_the_output_contract():
    m = CyberWorldForecaster(n_techniques=4).fit_input_normalizer(_skewed_batch())
    out = m(torch.rand(8, L, D))
    assert out["current_attack_logit"].shape == (8,)
    assert out["future_attack_logits"].shape == (8, DEFAULT_CONFIG.temporal.forecast_steps)
    assert torch.isfinite(out["current_attack_logit"]).all()


def test_it_is_fitted_on_train_only():
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert "fit_input_normalizer(tr[" in src, (
        "the normaliser must be fitted on the TRAIN split; fitting it on all "
        "the data leaks test statistics into training"
    )


# ---------------------------------------------------------------------------
# pos_weight
# ---------------------------------------------------------------------------


def _batch(n, pos_rate, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = DEFAULT_CONFIG.temporal.forecast_steps
    cur = (torch.rand(n, generator=g) < pos_rate).float()
    return dict(
        current_attack=cur,
        future_attack=cur.unsqueeze(1).repeat(1, K),
        hazard_target=torch.zeros(n, K), at_risk=torch.ones(n, K),
        future_techniques=torch.zeros(n, K, 4), severity=torch.zeros(n),
    )


def test_pos_weight_actually_changes_the_loss():
    m = CyberWorldForecaster(n_techniques=4)
    x, b = torch.rand(128, L, D), _batch(128, 0.1)
    out = m(x)
    plain, _ = forecast_loss(out, b)
    weighted, _ = forecast_loss(out, b, pos_weight=torch.tensor([9.0]))
    assert not torch.isclose(plain, weighted), "pos_weight had no effect"


def test_pos_weight_raises_the_cost_of_missing_the_rare_class():
    """The reason to use it: a confident 'benign for everything' must hurt more."""
    K = DEFAULT_CONFIG.temporal.forecast_steps
    n = 100
    b = _batch(n, 0.1, seed=1)
    out = {
        "current_attack_logit": torch.full((n,), -5.0),     # always "benign"
        "future_attack_logits": torch.full((n, K), -5.0),
        "hazard_logits": torch.zeros(n, K),
        "technique_logits": torch.zeros(n, K, 4),
        "severity": torch.zeros(n),
    }
    plain, _ = forecast_loss(out, b)
    weighted, _ = forecast_loss(out, b, pos_weight=torch.tensor([9.0]))
    assert float(weighted) > float(plain)


def test_the_trainer_computes_and_passes_it():
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert "pos_weight=pos_weight" in src, "forecast_loss is still called without pos_weight"
    assert "max_pos_weight" in src, "an unclamped pos_weight can destabilise training"


def test_the_automatic_weight_is_clamped():
    """A near-empty positive class must not produce a 1000x term."""
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert "min(max((1.0 - _pos)" in src or "args.max_pos_weight" in src


# ---------------------------------------------------------------------------
# Selection and early stopping
# ---------------------------------------------------------------------------


def test_selection_is_on_auc_not_on_validation_loss():
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert "roc_auc_score" in src, "the trainer does not compute a validation AUC"
    assert "if float(vloss) < best" not in src, (
        "still selecting the epoch with the lowest multi-task loss while "
        "reporting AUC"
    )


def test_the_selection_metric_covers_both_tasks():
    """Optimising nowcast alone is what produced a model that loses to
    persistence at every forecast horizon."""
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    i = src.index("def _val_score")
    body = src[i:i + 1200]
    assert "current_attack" in body and "future_attack" in body


def test_early_stopping_exists_and_is_configurable():
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    opts = {a.args[0].value for a in ast.walk(tree)
            if isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "add_argument"
            and a.args and isinstance(a.args[0], ast.Constant)}
    assert "--patience" in opts
    assert "early stop" in src


def test_a_single_class_validation_split_does_not_crash_selection():
    """It should warn and keep the last weights, not raise."""
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert 'len(np.unique(y_now)) > 1' in src
    assert "no epoch produced a finite validation AUC" in src


def test_the_choices_are_recorded_in_the_results_file():
    """A number is not reproducible if the settings that produced it are not
    written down beside it."""
    src = (REPO / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    for key in ("selection_metric", "selected_epoch", "pos_weight",
                "input_normalizer_fitted"):
        assert f'"{key}"' in src, f"{key} is not recorded in the results JSON"
