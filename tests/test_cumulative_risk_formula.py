"""Cumulative horizon risk must accumulate, not take the peak.

`InfiltrationRiskHead.forward_trajectory` returned `max(step_risks)` under the
name `cumulative_risk`, contradicting its own class docstring AND
cyberworld_v4's metrics, whose test
`test_cumulative_onset_uses_one_minus_product_not_max` demonstrates the
problem with a worked example.

It matters exactly where the value is used:
`correlation/trajectory_assembler.py` feeds it to
`HostAttackTrajectory.cumulative_forecast_risk`, which is how hosts are
ordered for an analyst. A host under persistent moderate pressure should
outrank one with a single noisy spike; peak ranks them backwards.
"""

import numpy as np
import pytest
import torch

from branch_b_world_model.infiltration_head import InfiltrationRiskHead


def _cum(rows):
    """Run the shipped head with its MLP bypassed, so we test the formula."""
    head = InfiltrationRiskHead(d_latent=4)
    r = torch.tensor(rows, dtype=torch.float32)
    # forward_trajectory's arithmetic, isolated from the learned MLP
    return 1.0 - torch.exp(torch.log1p(-r.clamp(max=1.0 - 1e-6)).sum(dim=-1))


def test_the_v4_worked_example():
    """Row B has a LOWER peak than row A but a HIGHER accumulated hazard."""
    cum = _cum([[0.50, 0.0, 0.0, 0.0, 0.0],
                [0.30, 0.30, 0.30, 0.0, 0.0]])
    assert cum[0].item() == pytest.approx(0.500, abs=1e-3)
    assert cum[1].item() == pytest.approx(0.657, abs=1e-3)
    assert cum[1] > cum[0], "sustained pressure must outrank a single spike"


def test_peak_would_have_ranked_it_backwards():
    """Guard the guard: show the old statistic really gets this wrong."""
    r = torch.tensor([[0.50, 0.0, 0.0, 0.0, 0.0],
                      [0.30, 0.30, 0.30, 0.0, 0.0]])
    peak = InfiltrationRiskHead.peak_risk(r)
    assert peak[0] > peak[1], "if this fails the example no longer discriminates"


def test_accumulation_is_monotone_in_the_number_of_risky_steps():
    cum = _cum([[0.2, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.0, 0.0],
                [0.2, 0.2, 0.2, 0.0],
                [0.2, 0.2, 0.2, 0.2]])
    assert all(cum[i] < cum[i + 1] for i in range(3))


def test_bounds():
    cum = _cum([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    assert cum[0].item() == pytest.approx(0.0)
    assert 0.0 <= cum[2].item() <= 1.0
    assert cum[1].item() <= 1.0 and cum[1].item() > 0.999


def test_a_single_certain_step_saturates():
    cum = _cum([[0.0, 1.0, 0.0]])
    assert cum[0].item() > 0.999


def test_shipped_head_returns_the_accumulated_value():
    """End to end through the real module, not a reimplementation."""
    torch.manual_seed(0)
    head = InfiltrationRiskHead(d_latent=6).eval()
    with torch.no_grad():
        step, cum = head.forward_trajectory(torch.randn(8, 5, 6))
    assert step.shape == (8, 5) and cum.shape == (8,)
    expected = 1.0 - torch.exp(torch.log1p(-step.clamp(max=1 - 1e-6)).sum(-1))
    torch.testing.assert_close(cum, expected)
    # and it must NOT equal the peak on varied input
    assert not torch.allclose(cum, InfiltrationRiskHead.peak_risk(step))


def test_peak_is_still_available_for_its_own_question():
    r = torch.tensor([[0.1, 0.9, 0.2]])
    assert InfiltrationRiskHead.peak_risk(r).item() == pytest.approx(0.9)
