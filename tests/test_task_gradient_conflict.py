"""Whether Branch A's tasks fight over the shared LSTM is measured, not assumed.

An external review asserted "unmitigated gradient interference" and called for
PCGrad/GradNorm without measuring anything. The measurement is cheap: the
cosine between per-task gradients on the shared parameters. This pins that
the diagnostic computes exactly that, leaves training untouched, and is
printed by the trainer every epoch.
"""

from pathlib import Path

import pytest
import torch

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM

REPO = Path(__file__).resolve().parents[1]


def _batch(B=32, T=15, D=27, n_tech=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=g)
    targets = {
        "risk": (torch.rand(B, generator=g) > 0.8).float() * 0.7,
        "technique": torch.randint(0, n_tech, (B,), generator=g),
        "gradation": torch.randint(0, 4, (B,), generator=g),
    }
    return x, targets


@pytest.fixture(params=["paper", "legacy"])
def model(request):
    torch.manual_seed(0)
    arch = MultiTaskLSTM.PAPER_ARCH if request.param == "paper" else MultiTaskLSTM.LEGACY_ARCH
    return MultiTaskLSTM(input_dim=27, **arch).train()


def _shared(m):
    return [p for n, p in m.named_parameters()
            if p.requires_grad and not n.startswith(MultiTaskLSTM.TASK_PARAM_PREFIXES)]


def test_cosines_match_an_independent_computation(model):
    x, t = _batch()
    torch.manual_seed(1)  # dropout draws must match between the two passes
    out = model.task_gradient_conflict(x, t)

    torch.manual_seed(1)
    with torch.backends.cudnn.flags(enabled=False):
        losses = model.task_losses(model(x), t)
    flat = []
    for L in losses:
        g = torch.autograd.grad(L, _shared(model), retain_graph=True, allow_unused=True)
        flat.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                               for gi, p in zip(g, _shared(model))]).double())
    cos = torch.nn.functional.cosine_similarity
    assert out["cosine"]["risk_vs_tech"] == pytest.approx(float(cos(flat[0], flat[1], dim=0)), abs=1e-6)
    assert out["cosine"]["risk_vs_grad"] == pytest.approx(float(cos(flat[0], flat[2], dim=0)), abs=1e-6)
    assert out["cosine"]["tech_vs_grad"] == pytest.approx(float(cos(flat[1], flat[2], dim=0)), abs=1e-6)
    assert out["min_cosine"] == min(out["cosine"].values())
    assert out["dominant_task"] in ("risk", "tech", "grad")


def test_only_the_trunk_counts_as_shared(model):
    out = model.task_gradient_conflict(*_batch())
    heads = sum(p.numel() for n, p in model.named_parameters()
                if n.startswith(MultiTaskLSTM.TASK_PARAM_PREFIXES))
    total = sum(p.numel() for p in model.parameters())
    assert out["n_shared_params"] == total - heads
    assert out["n_shared_params"] >= sum(p.numel() for p in model.lstm.parameters())


def test_it_does_not_touch_grad_or_the_optimizer_state(model):
    for p in model.parameters():
        p.grad = None
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.task_gradient_conflict(*_batch())
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_weighted_norms_use_the_objective_weights(model):
    out = model.task_gradient_conflict(*_batch())
    if model.loss_weighting == "fixed":
        w = dict(zip(("risk", "tech", "grad"), MultiTaskLSTM.PAPER_LOSS_WEIGHTS))
    else:
        w = {"risk": 1.0, "tech": 1.0, "grad": 1.0}  # log_var starts at 0
    for k in w:
        assert out["weighted_grad_norm"][k] == pytest.approx(out["grad_norm"][k] * w[k], rel=1e-6)


def test_compute_loss_is_unchanged_by_the_split(model):
    """task_losses was extracted from compute_loss; the objective must not move."""
    x, t = _batch(seed=3)
    model.eval()
    with torch.no_grad():
        preds = model(x)
        total, metrics = model.compute_loss(preds, t)
        r, te, g = model.task_losses(preds, t)
    if model.loss_weighting == "fixed":
        a, b, c = model.loss_weights
        expected = a * r + b * te + c * g
    else:
        expected, _ = model.uncertainty_loss(r, te, g)
    assert torch.allclose(total, expected)
    assert torch.allclose(metrics["loss_risk"], r) and torch.allclose(metrics["loss_tech"], te)


def test_the_trainer_reports_it_every_epoch():
    src = (REPO / "scripts" / "retrain_branch_a_live.py").read_text(encoding="utf-8")
    assert "model.task_gradient_conflict(x, targets)" in src
    assert 'metrics["task_gradients"]' in src
