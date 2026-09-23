"""Branch A learns the real time between a host's history steps.

None of Branch A's 15 temporal attributes spans windows -- they are all
per-window aggregates -- so the model saw fifteen feature vectors with no way
to tell whether they were 2 s or 2 hours apart. Measured over the full train
split, 40.9% of its targets lie more than 10 s from their history (64% on
CTU-13, a median 736 s). Branch B already encoded real elapsed time; this is
the same fix, so the gap is handled without dropping any samples.

The projection is zero-initialised and a backward-compatible load backfills
only the new keys, so every existing checkpoint keeps loading and behaves
exactly as saved until a model is retrained with times.
"""
import ast

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from branch_a_gnn_lstm.sequence_dataset import LazyHostSequenceDataset, TECHNIQUE_VOCAB
from data_unification.trajectory_store import TrajectoryStoreBuilder


def _model():
    torch.manual_seed(0)
    return MultiTaskLSTM(input_dim=27, hidden_dim=32, num_layers=2,
                         num_techniques=len(TECHNIQUE_VOCAB), num_gradations=4).eval()


def _t(spacing, n=15, b=3):
    return torch.arange(-(n - 1), 1, dtype=torch.float32).mul(spacing).expand(b, -1)


def test_zero_init_makes_times_a_no_op_until_trained():
    m, x = _model(), torch.randn(3, 15, 27)
    with torch.no_grad():
        a = m(x)["technique_logits"]
        b = m(x, t_history=_t(700.0))["technique_logits"]
    assert torch.equal(a, b), "an untrained time channel must not change the output"


def test_the_channel_is_live_once_trained():
    """Guards against a time input that is accepted and then ignored."""
    m, x = _model(), torch.randn(3, 15, 27)
    with torch.no_grad():
        m.time_proj.weight.normal_(0, 0.5)
        dense = m(x, t_history=_t(2.0))["technique_logits"]
        sparse = m(x, t_history=_t(700.0))["technique_logits"]
    assert not torch.allclose(dense, sparse), "the model cannot tell 2 s from 700 s"


def test_a_checkpoint_from_before_the_channel_still_loads_and_is_unchanged():
    old = _model()
    sd = {k: v for k, v in old.state_dict().items()
          if not k.startswith(("time_encoder.", "time_proj."))}
    new = _model()
    new.load_state_dict(sd)                       # strict=True by default
    x = torch.randn(2, 15, 27)
    with torch.no_grad():
        assert torch.equal(old(x)["risk_score"], new(x, t_history=_t(5.0, b=2))["risk_score"])


def test_the_loader_is_still_strict_for_every_other_key():
    """Backfilling the time keys must not become a blanket strict=False."""
    sd = dict(_model().state_dict())
    del sd[next(k for k in sd if k.startswith("lstm."))]
    with pytest.raises(RuntimeError, match="Missing key"):
        _model().load_state_dict(sd)


def test_the_dataset_emits_history_times_in_the_models_convention():
    b = TrajectoryStoreBuilder(spill_dir=None)
    for w in range(0, 20 * 7, 7):                 # every 7th window: 14 s apart
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=w * 2.0, window_end=w * 2.0 + 2.0,
                 embedding=np.zeros(12, np.float32), temporal_attrs=np.zeros(15, np.float32),
                 is_attack=False, coarse_category="Benign", technique_ids=[], risk_score=0.0)
    ds = LazyHostSequenceDataset(b.finalize(), seq_len=5)
    t = ds[len(ds) - 1]["t_history"].numpy()
    assert t.shape == (5,) and t[-1] == 0.0 and np.all(t <= 0)
    np.testing.assert_allclose(np.diff(t), 14.0)
    # a short history is left-padded by repeating the earliest real time
    t0 = ds[0]["t_history"].numpy()
    assert np.all(t0 == 0.0), f"one-step history should pad with its own time, got {t0}"


def _branch_a_calls(path):
    tree = ast.parse(open(path).read())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "branch_a"
            or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "model"
                and n.args)]


@pytest.mark.parametrize("path", [
    "scripts/retrain_branch_a_live.py",
    "control_backend/model_adapter.py",
    "correlation/trajectory_assembler.py",
])
def test_every_branch_a_forward_passes_real_times(path):
    calls = _branch_a_calls(path)
    assert calls, f"{path}: no Branch A forward found"
    for c in calls:
        assert "t_history" in {k.arg for k in c.keywords}, \
            f"{path}:{c.lineno} runs Branch A without the time channel"
