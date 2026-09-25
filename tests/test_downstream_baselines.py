"""Branch B and DeepOP must be measured against predictors that need no model.

Branch B's first full run moved train loss 1.9% across 6 epochs with its best
validation at epoch 1. That is equally consistent with "converged immediately"
and "never learned anything", and nothing it reported could separate the two.
The same blind spot produced Branch A's 0.88 technique accuracy on an
82.5%-Benign corpus, and a risk head that turned out to be 70% worse than
predicting zero.

So both trainers now report their null hypotheses:

  Branch B  -- copy the last observed embedding across the horizon
               (persistence), and predict 0.0 for risk
  DeepOP    -- repeat the last observed token, and always emit the most
               common token

These tests pin the arithmetic of those comparisons.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F


# --- Branch B: persistence skill -------------------------------------------

def _skill(mse_model, mse_persist):
    return 1.0 - mse_model / mse_persist


def test_persistence_is_the_last_observed_step_repeated():
    B, T, K, D = 4, 15, 5, 12
    h = torch.randn(B, T, D)
    last = h[:, -1:, :].expand(-1, K, -1)
    assert last.shape == (B, K, D)
    for k in range(K):
        assert torch.equal(last[:, k, :], h[:, -1, :])


def test_a_model_that_copies_the_last_step_scores_zero_skill():
    B, T, K, D = 8, 15, 5, 12
    h = torch.randn(B, T, D)
    target = torch.randn(B, K, D)
    persist = h[:, -1:, :].expand(-1, K, -1)
    mse_p = float(F.mse_loss(persist, target))
    mse_m = float(F.mse_loss(persist.clone(), target))   # model == persistence
    assert _skill(mse_m, mse_p) == pytest.approx(0.0, abs=1e-9)


def test_a_perfect_model_scores_skill_one_and_a_worse_one_goes_negative():
    B, T, K, D = 8, 15, 5, 12
    h = torch.randn(B, T, D)
    target = torch.randn(B, K, D)
    persist = h[:, -1:, :].expand(-1, K, -1)
    mse_p = float(F.mse_loss(persist, target))
    assert _skill(0.0, mse_p) == pytest.approx(1.0)
    assert _skill(mse_p * 2, mse_p) < 0, "a model worse than persistence must go negative"


def test_predicting_zero_has_mae_equal_to_the_mean():
    """Why mae_predict_zero is reported as the target mean."""
    r = torch.rand(10000)
    assert float((torch.zeros_like(r) - r).abs().mean()) == pytest.approx(float(r.mean()), abs=1e-6)


def test_constant_bce_matches_the_closed_form():
    """bce_constant = -[p log p + (1-p) log(1-p)] for the empirical rate p."""
    import math
    p = 0.3462
    target = torch.full((100000,), 0.0)
    target[: int(p * 100000)] = 1.0
    p_hat = float(target.mean())
    closed = -(p_hat * math.log(p_hat) + (1 - p_hat) * math.log(1 - p_hat))
    measured = float(F.binary_cross_entropy(torch.full_like(target, p_hat), target))
    assert closed == pytest.approx(measured, abs=1e-5)


# --- DeepOP: token baselines -----------------------------------------------

def _token_stats(pred, tgt, obs, V):
    conf = torch.bincount(tgt.reshape(-1) * V + pred.reshape(-1), minlength=V * V)
    hist = torch.bincount(tgt.reshape(-1), minlength=V)
    n = tgt.numel()
    acc = float((pred == tgt).sum()) / n
    acc_persist = float((obs.unsqueeze(1).expand_as(tgt) == tgt).sum()) / n
    acc_major = float(hist.max()) / n
    cm = conf.reshape(V, V).numpy()
    sup, pn, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(pn > 0, tp / np.maximum(pn, 1), 0.0)
        rc = np.where(sup > 0, tp / np.maximum(sup, 1), 0.0)
        dn = pr + rc
        f1 = np.where(dn > 0, 2 * pr * rc / np.maximum(dn, 1e-12), 0.0)
    present = sup > 0
    return acc, acc_persist, acc_major, float(f1[present].mean())


def test_a_constant_token_predictor_is_caught():
    V, B, K = 6, 200, 5
    g = torch.Generator().manual_seed(0)
    tgt = torch.where(torch.rand(B, K, generator=g) < 0.85, 0,
                      torch.randint(1, V, (B, K), generator=g))
    obs = torch.zeros(B, dtype=torch.long)
    pred = torch.zeros_like(tgt)                       # always the majority token
    acc, ap, am, f1 = _token_stats(pred, tgt, obs, V)
    assert acc > 0.80, "accuracy still looks fine"
    assert acc == pytest.approx(am, abs=1e-9), "but it exactly equals the majority baseline"
    assert acc - max(ap, am) == pytest.approx(0.0, abs=1e-9), "zero lift"
    assert f1 < 0.30, "macro F1 exposes it"


def test_a_perfect_token_predictor_beats_both_baselines():
    V, B, K = 6, 200, 5
    g = torch.Generator().manual_seed(1)
    tgt = torch.randint(0, V, (B, K), generator=g)
    obs = torch.randint(0, V, (B,), generator=g)
    acc, ap, am, f1 = _token_stats(tgt.clone(), tgt, obs, V)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert acc - max(ap, am) > 0.5


def test_macro_f1_matches_sklearn_on_tokens():
    sk = pytest.importorskip("sklearn.metrics")
    V, B, K = 5, 300, 4
    g = torch.Generator().manual_seed(2)
    tgt = torch.randint(0, V, (B, K), generator=g)
    pred = torch.randint(0, V, (B, K), generator=g)
    obs = torch.zeros(B, dtype=torch.long)
    _, _, _, f1 = _token_stats(pred, tgt, obs, V)
    present = sorted(set(tgt.reshape(-1).tolist()))
    expected = sk.f1_score(tgt.reshape(-1).numpy(), pred.reshape(-1).numpy(),
                           labels=present, average="macro", zero_division=0)
    assert f1 == pytest.approx(expected, abs=1e-9)


def test_persistence_baseline_uses_the_observed_token_not_the_target():
    """obs_token is the token at i-1; using a target would leak the answer."""
    V, B, K = 4, 50, 3
    tgt = torch.full((B, K), 2, dtype=torch.long)
    obs = torch.full((B,), 2, dtype=torch.long)
    _, ap, _, _ = _token_stats(torch.zeros_like(tgt), tgt, obs, V)
    assert ap == pytest.approx(1.0), "a label that never changes is perfectly persistent"
    obs_wrong = torch.full((B,), 3, dtype=torch.long)
    _, ap2, _, _ = _token_stats(torch.zeros_like(tgt), tgt, obs_wrong, V)
    assert ap2 == pytest.approx(0.0)
