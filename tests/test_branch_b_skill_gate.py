"""DeepOP must not train on a Branch B that does not beat persistence.

Branch B's first run flatlined at epoch 1 and moved loss 1.9% over six epochs.
If its rollouts are no better than repeating the last embedding, DeepOP is
conditioning on the input copied forward and can only learn the label prior --
and no amount of DeepOP tuning fixes that. The skill verdict is now stamped
into the Branch B checkpoint and the DeepOP stage refuses to start without it.
"""

import pytest

from scripts.retrain_future_models_live import (
    MIN_BRANCH_B_SKILL,
    branch_b_credibility,
    require_credible_branch_b,
)


def _hist(skill, mae=0.1, mae_zero=0.2):
    return [{"epoch": 1, "skill": skill, "mse_model": 1 - skill, "mse_persistence": 1.0,
             "risk_mae_model": mae, "risk_mae_zero": mae_zero}]


def test_near_zero_skill_is_not_credible():
    cred = branch_b_credibility(_hist(0.001), best_epoch=1)
    assert cred["checked"] and not cred["credible"]
    assert "persistence" in cred["problems"][0]


def test_real_skill_is_credible():
    cred = branch_b_credibility(_hist(MIN_BRANCH_B_SKILL + 0.1), best_epoch=1)
    assert cred["credible"], cred["problems"]


def test_risk_head_worse_than_zero_is_flagged():
    cred = branch_b_credibility(_hist(0.5, mae=0.3, mae_zero=0.2), best_epoch=1)
    assert not cred["credible"]


def test_deepop_refuses_noncredible_or_unchecked_branch_b():
    bad = {"credibility": branch_b_credibility(_hist(0.0), best_epoch=1)}
    with pytest.raises(SystemExit, match="REFUSING"):
        require_credible_branch_b(bad, allow=False)
    with pytest.raises(SystemExit, match="predates"):
        require_credible_branch_b({}, allow=False)       # legacy checkpoint
    require_credible_branch_b(bad, allow=True)           # explicit override only


def test_deepop_accepts_credible_branch_b():
    ok = {"credibility": branch_b_credibility(_hist(0.3), best_epoch=1)}
    require_credible_branch_b(ok, allow=False)
