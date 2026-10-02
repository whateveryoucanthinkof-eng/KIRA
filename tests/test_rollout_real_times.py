"""Branch B, DeepOP and serving must all use a host's REAL step spacing.

Trajectory rows are a host's ACTIVE windows. Measured on the train split, a
host's consecutive windows are a median 14 s apart on CIC-2018 and 736 s on
CTU-13, not the 2 s grid. HostWorldDynamicsTransformer.rollout() encodes real
elapsed times, and LazyHostRolloutDataset emitted them -- but the trainer that
produces the SERVED checkpoint never passed them, so the model was told every
step was 2 s apart. A comment in the standalone trainer said as much: "Passing
these two tensors is the whole change it needs." Nobody made that change.

Wiring it into training alone would have created a new skew, so every link is
pinned here: the trainer, the DeepOP dataset and rollout calls, and both
serving paths. t_future stays None at serving on purpose -- the uniform grid
IS the question being asked ("+2 ... +10 s from now").
"""
import ast
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from data_unification.trajectory_store import TrajectoryStoreBuilder

EMB = np.zeros(12, dtype=np.float32)


def _store(windows, ip="10.0.0.1"):
    """One host active in the given window indices (2 s each)."""
    b = TrajectoryStoreBuilder(spill_dir=None)
    rng = np.random.default_rng(0)
    for w in windows:
        b.append(host_ip=ip, host_id=0, window_idx=int(w),
                 window_start=float(w) * 2.0, window_end=float(w) * 2.0 + 2.0,
                 embedding=rng.random(12).astype(np.float32),
                 temporal_attrs=np.zeros(15, np.float32), is_attack=False,
                 coarse_category="Benign", technique_ids=[], risk_score=0.0)
    return b.finalize()


def test_real_times_change_the_rollout():
    """Pins that the time signal is live inside the model -- a fix that is
    passed in and then ignored would pass every other test here."""
    torch.manual_seed(0)
    wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4, n_layers=3).eval()
    h = torch.randn(4, 15, 12)
    uniform = torch.arange(-14, 1, dtype=torch.float32).mul(2.0).expand(4, -1)
    sparse = uniform * 7.0                       # the measured median spacing
    with torch.no_grad():
        a = wdt.rollout(h, K=5, t_history=uniform)
        b = wdt.rollout(h, K=5, t_history=sparse)
    assert not torch.allclose(a, b), "rollout ignores t_history"


def test_branch_b_dataset_times_use_the_models_convention():
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
    st = _store(list(range(0, 21 * 7, 7)))         # every 7th window: 14 s apart
    ds = LazyHostRolloutDataset(st, T=15, K=5)
    item = ds[0]
    th, tf = item["t_history"].numpy(), item["t_future"].numpy()
    assert th[-1] == 0.0 and np.all(th <= 0), "history must end at 0 and be <= 0"
    assert np.all(tf > 0), "future must be > 0"
    np.testing.assert_allclose(np.diff(th), 14.0)  # real spacing, not 2 s
    np.testing.assert_allclose(tf, [14.0, 28.0, 42.0, 56.0, 70.0])


def test_deepop_dataset_emits_the_same_times_as_branch_b():
    """DeepOP conditions on Branch B rollouts, so it must feed them the same
    times Branch B was trained with."""
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
    from deepop_decoder.train_cwa_decoder import LazyCWADataset
    from deepop_decoder.joint_vocab import get_joint_vocab
    st = _store(list(range(0, 25 * 7, 7)))
    bb = LazyHostRolloutDataset(st, T=15, K=5)
    dp = LazyCWADataset(st, get_joint_vocab(network_observable_only=True), K=5, T=15)
    # align on the same target position i = 15
    i_b = [k for k in range(len(bb)) if int(bb._pos[k]) == 15][0]
    i_d = [k for k in range(len(dp)) if int(dp._pos[k]) == 15][0]
    np.testing.assert_allclose(dp[i_d]["t_history"].numpy(), bb[i_b]["t_history"].numpy())
    np.testing.assert_allclose(dp[i_d]["t_future"].numpy(), bb[i_b]["t_future"].numpy())


def test_every_rollout_in_the_live_trainer_passes_real_times():
    """The regression itself: the served checkpoint's trainer called
    wdt.rollout(h, K=...) with no times."""
    src = open("scripts/retrain_future_models_live.py").read()
    tree = ast.parse(src)
    missing = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "rollout"):
            kw = {k.arg for k in node.keywords}
            if "t_history" not in kw:
                missing.append(node.lineno)
    assert not missing, f"wdt.rollout() without t_history at lines {missing}"


@pytest.mark.parametrize("path", ["cyberworld_v4/cross_dataset.py",
                                  "deepop_decoder/train_cwa_decoder.py",
                                  "branch_b_world_model/train_branch_b.py"])
def test_every_other_rollout_caller_passes_real_times(path):
    """The held-out CIC-2017 scoring and the standalone DeepOP trainer made the
    same omission after the live trainer was fixed."""
    tree = ast.parse(open(path).read())
    missing = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "rollout" and "t_history" not in {k.arg for k in n.keywords}]
    assert not missing, f"{path}: rollout() without t_history at lines {missing}"


def test_both_serving_paths_pass_real_history_times():
    for path in ("control_backend/model_adapter.py", "correlation/trajectory_assembler.py"):
        tree = ast.parse(open(path).read())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("rollout", "rollout_with_uncertainty")]
        assert calls, f"{path}: no rollout call found"
        for c in calls:
            assert "t_history" in {k.arg for k in c.keywords}, \
                f"{path}:{c.lineno} rolls out without real history times"


def test_the_adapter_keeps_times_in_lockstep_with_states():
    """If the two lists drift, t_history describes the wrong steps."""
    src = open("control_backend/model_adapter.py").read()
    assert "h_time_history.pop(0)" in src, "times are not trimmed with states"
    assert "self.h_time_history_by_target.clear()" in src, "reset leaves stale times"
