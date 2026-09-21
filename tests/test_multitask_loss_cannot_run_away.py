"""Kendall & Gal uncertainty weighting is unbounded below; bound it.

For a task with loss L the optimum is log_var = log(L), worth 1 + log(L). As
L -> 0 the total dives to -inf and that task's precision exp(-log_var)
explodes, starving the others -- while the logged loss falls the whole time,
so it reads as healthy training.

Measured before the clamp, with risk_loss = 1e-4 and the others realistic:

    step   total    prec_risk   prec_tech
       1    2.100         1.0       1.000
     300   -6.133      9993.5       0.833     <- 12,000x imbalance

This matters here specifically: `risk_score` is derived from `is_attack`
(benign exactly 0.0, attack >= 0.20), so risk is the EASIEST of the three
tasks and the least informative -- exactly the one that runs away.
"""

import pytest
import torch

from branch_a_gnn_lstm.lstm_multitask import MultiTaskUncertaintyLoss


def _train(clamp, risk=1e-4, tech=1.2, grad=0.9, steps=600, lr=0.05):
    crit = MultiTaskUncertaintyLoss(clamp=clamp)
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    m = {}
    for _ in range(steps):
        total, m = crit(torch.tensor(risk), torch.tensor(tech), torch.tensor(grad))
        opt.zero_grad(); total.backward(); opt.step()
    return m


def test_unclamped_really_does_run_away():
    """Establish the premise, so the clamp is justified by measurement."""
    m = _train(clamp=False)
    assert m["weight_risk"] / m["weight_tech"] > 1000, (
        "premise failed: the unclamped objective should blow up"
    )
    assert m["loss_total"] < -3.0


def test_clamped_stays_bounded():
    m = _train(clamp=True)
    ratio = m["weight_risk"] / m["weight_tech"]
    assert ratio < 500, f"task weighting still runs away: {ratio:.0f}x"
    assert m["loss_total"] > -5.0


def test_precisions_stay_inside_the_documented_range():
    m = _train(clamp=True)
    lo = pytest.approx(torch.exp(torch.tensor(-3.0)).item(), rel=0.01)
    hi = pytest.approx(torch.exp(torch.tensor(3.0)).item(), rel=0.01)
    for k in ("weight_risk", "weight_tech", "weight_grad"):
        assert m[k] <= torch.exp(torch.tensor(3.0)).item() * 1.01, f"{k} above bound"
        assert m[k] >= torch.exp(torch.tensor(-3.0)).item() * 0.99, f"{k} below bound"


def test_clamping_is_on_by_default():
    import inspect
    sig = inspect.signature(MultiTaskUncertaintyLoss.__init__)
    assert sig.parameters["clamp"].default is True


def test_balanced_tasks_are_still_weighted_near_equally():
    """The clamp must not flatten genuine differences -- equal losses should
    still give equal weights."""
    m = _train(clamp=True, risk=1.0, tech=1.0, grad=1.0)
    assert m["weight_risk"] == pytest.approx(m["weight_tech"], rel=0.05)
    assert m["weight_tech"] == pytest.approx(m["weight_grad"], rel=0.05)


def test_a_genuinely_harder_task_can_still_be_downweighted():
    """Bounded, not disabled: the mechanism must still do its job."""
    m = _train(clamp=True, risk=0.05, tech=3.0, grad=1.0)
    assert m["weight_risk"] > m["weight_tech"], (
        "the easy task should still receive more precision than the hard one"
    )


def test_metrics_expose_the_weights():
    """A runaway must be visible in the metrics, not inferred from the loss."""
    _c = MultiTaskUncertaintyLoss()
    _t, m = _c(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    for k in ("weight_risk", "weight_tech", "weight_grad",
              "loss_risk", "loss_tech", "loss_grad", "loss_total"):
        assert k in m
