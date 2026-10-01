"""Branch B as a distribution: N(mean_k, exp(logvar_k)) per forecast step.

The PS asks the world model for P(S_t+1 | S_t). out_head gives the mean; the
variance head makes it a distribution. It must not change how the mean trains
(the persistence gate judges the mean), old checkpoints must keep loading, and
on data with known noise it must learn that noise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer, gaussian_nll

ROOT = Path(__file__).resolve().parents[1]


def _variance_loss():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("rfml", ROOT / "scripts" / "retrain_future_models_live.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._bb_variance_loss


def test_variance_loss_does_not_touch_the_mean_path():
    torch.manual_seed(0)
    m = HostWorldDynamicsTransformer(d_latent=6, d_model=16, n_heads=2, n_layers=1, dim_feedforward=16)
    h = torch.randn(8, 15, 6)
    tgt = torch.randn(8, 3, 6)
    mean, logv = m.rollout(h, K=3, return_logvar=True)
    _variance_loss()(mean, logv, tgt).backward()
    for name, p in m.named_parameters():
        g = p.grad
        if name.startswith("logvar_head."):
            assert g is not None and g.abs().sum() > 0, name
        else:
            assert g is None or g.abs().sum() == 0, f"variance loss reached {name}"


def test_mean_is_unchanged_by_return_logvar():
    torch.manual_seed(1)
    m = HostWorldDynamicsTransformer(d_latent=6, d_model=16, n_heads=2, n_layers=1, dim_feedforward=16).eval()
    h = torch.randn(4, 15, 6)
    with torch.no_grad():
        a = m.rollout(h, K=4)
        b, lv = m.rollout(h, K=4, return_logvar=True)
    assert torch.equal(a, b) and lv.shape == (4, 4, 6)


def test_old_checkpoint_loads_and_says_its_variance_is_untrained():
    m = HostWorldDynamicsTransformer(d_latent=6, d_model=16, n_heads=2, n_layers=1, dim_feedforward=16)
    old = {k: v for k, v in m.state_dict().items() if not k.startswith("logvar_head.")}
    fresh = HostWorldDynamicsTransformer(d_latent=6, d_model=16, n_heads=2, n_layers=1, dim_feedforward=16)
    fresh.load_state_dict(old)                     # strict
    assert fresh.variance_trained is False
    fresh.load_state_dict(m.state_dict())
    assert fresh.variance_trained is True


def test_variance_head_learns_the_noise_level():
    """Targets = persistence + N(0, 0.3^2): the learned sigma should approach 0.3
    and the 90% interval should cover ~90%."""
    torch.manual_seed(2)
    m = HostWorldDynamicsTransformer(d_latent=4, d_model=16, n_heads=2, n_layers=1, dim_feedforward=16,
                                     dropout=0.0)
    opt = torch.optim.Adam(m.logvar_head.parameters(), lr=3e-2)
    vloss = _variance_loss()
    for _ in range(300):
        h = torch.randn(64, 15, 4) * 0.1
        with torch.no_grad():
            mean = m.rollout(h, K=2)
        tgt = mean + 0.3 * torch.randn_like(mean)
        _, lv = m.rollout(h, K=2, return_logvar=True)
        opt.zero_grad()
        vloss(mean, lv, tgt).backward()
        opt.step()
    h = torch.randn(2048, 15, 4) * 0.1
    with torch.no_grad():
        mean, lv = m.rollout(h, K=2, return_logvar=True)
    sigma = torch.exp(0.5 * lv).mean().item()
    tgt = mean + 0.3 * torch.randn_like(mean)
    cov = ((tgt - mean).abs() <= 1.6449 * torch.exp(0.5 * lv)).float().mean().item()
    assert abs(sigma - 0.3) < 0.05, sigma
    assert abs(cov - 0.90) < 0.03, cov
    assert gaussian_nll(mean, lv, tgt) < gaussian_nll(mean, torch.zeros_like(lv), tgt)
