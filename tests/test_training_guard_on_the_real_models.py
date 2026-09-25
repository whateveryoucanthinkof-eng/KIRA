"""The guard's step back must work on the four models it actually trains.

A snapshot/restore that works on a Linear layer can still fail on TGN (memory
buffers), on Branch B (two modules, one optimizer) or on DeepOP; and each
trainer must route its step through the guard rather than around it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "bita") not in sys.path:
    sys.path.insert(0, str(REPO / "bita"))

from cyberworld_v4.training_guard import IMPROVED, STEP_BACK, TrainingGuard  # noqa: E402


def _encoder():
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder
    n0 = 16
    return [ExtendedTGN(
        neighbor_finder=NeighborFinder([[] for _ in range(n0)], uniform=False),
        node_features=np.zeros((n0, 12), np.float32), edge_features=np.zeros((10, 12), np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=True,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5)]


def _branch_a():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    return [MultiTaskLSTM(input_dim=27, **MultiTaskLSTM.PAPER_ARCH)]


def _branch_b():
    from branch_b_world_model.infiltration_head import InfiltrationRiskHead
    from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
    return [HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3),
            InfiltrationRiskHead(d_latent=27, hidden_dim=32)]


def _deepop():
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    from deepop_decoder.joint_vocab import get_joint_vocab
    return [DeepOPForecastDecoder(d_latent=27, d_model=72, vocab_size=get_joint_vocab().vocab_size,
                                  n_heads=6, num_layers=2, window_sizes=[2, 4, 8],
                                  dim_feedforward=144)]


@pytest.mark.parametrize("build", [_encoder, _branch_a, _branch_b, _deepop],
                         ids=["encoder", "branch_a", "branch_b", "deepop"])
def test_step_back_restores_every_module_exactly(build):
    torch.manual_seed(0)
    modules = build()
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)
    g = TrainingGuard("t", modules, opt, clip_norm=1.0, log=lambda m: None)

    def step():
        loss = sum((p.float() ** 2).sum() for p in params) * 1e-3
        assert g.backward_step(loss)
        opt.zero_grad(set_to_none=True)

    step()
    assert g.end_epoch(0.5, train_loss=1.0) == IMPROVED
    best = [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
    for _ in range(3):
        step()
    assert g.end_epoch(0.4, train_loss=1.0) != STEP_BACK
    step()
    assert g.end_epoch(0.4, train_loss=1.0) == STEP_BACK
    for m, sd in zip(modules, best):
        now = m.state_dict()
        for k, v in sd.items():
            # Some buffers are legitimately NaN until fitted (e.g. Branch A's
            # conformal half-width), and NaN != NaN.
            same = (torch.allclose(now[k], v, rtol=0, atol=0, equal_nan=True)
                    if v.is_floating_point() else torch.equal(now[k], v))
            assert same, f"{type(m).__name__}.{k} not restored"
    assert g.current_lr() == pytest.approx(5e-3)


@pytest.mark.parametrize("path,name", [
    ("bita/train.py", "encoder"),
    ("scripts/retrain_branch_a_live.py", "branch_a"),
    ("scripts/retrain_future_models_live.py", "branch_b"),
    ("scripts/retrain_future_models_live.py", "deepop"),
])
def test_every_trainer_steps_through_the_guard(path, name):
    src = (REPO / path).read_text(encoding="utf-8")
    assert f'TrainingGuard(\n        "{name}"' in src or f'TrainingGuard("{name}"' in src, \
        f"{path} does not build a guard for {name}"
    assert ("guard.backward_step(" in src or "guard.backward_step_deferred" in src) and "guard.end_epoch(" in src
    assert "optimizer.step()" not in src, f"{path} still steps the optimizer outside the guard"


def test_the_plan_sets_patience_3_and_step_back_2_for_every_model():
    plan = (REPO / "scripts" / "run_training_plan.sh").read_text(encoding="utf-8")
    assert "--patience 3 --step_back_after 2" in plan          # encoder
    assert plan.count("--patience 3 --step-back-after 2") == 2  # Branch A, downstream
