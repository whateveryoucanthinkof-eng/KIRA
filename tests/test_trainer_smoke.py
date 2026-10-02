"""End-to-end: every trainer runs one epoch on a tiny store without crashing.

This lived in a scratch script under /tmp until a reboot deleted it -- /tmp is
tmpfs on this machine. It belongs in the suite: the failure it catches is the
expensive kind. A trainer that crashes inside its loop does so only after the
40-minute trajectory extraction has already been paid for, and several did
this week (a loss accumulated with the wrong shape, a name local to one
function used in another). A single epoch on a few hundred windows exercises
the same code in seconds.

Deliberately memmap-backed (spill_dir set) so the resident-block path in
TrajectoryStoreBuilder.finalize is exercised too, and deliberately
multi-capture so the capture namespace is.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("CYBERWORLD_ALLOW_CONTRACT_MISMATCH", "1")

from data_unification.trajectory_store import TrajectoryStoreBuilder  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    spill = tmp_path_factory.mktemp("spill")
    rng = np.random.default_rng(3)
    cats = ["Benign", "Execution", "InitialAccess", "C2"]
    b = TrajectoryStoreBuilder(spill_dir=str(spill))
    for cap in ("CSV/day_a", "CSV/day_b"):
        b.set_namespace(cap)
        for h in range(12):
            for w in range(40):
                k = w % 4
                b.append(host_ip=f"10.0.0.{h}", host_id=h, window_idx=w,
                         window_start=float(w * 2), window_end=float(w * 2 + 2),
                         embedding=rng.random(12).astype(np.float32),
                         temporal_attrs=rng.random(15).astype(np.float32),
                         is_attack=(k != 0), coarse_category=cats[k],
                         technique_ids=["T1071"] if k else [], risk_score=float(k) / 3)
    return b.finalize()


def test_store_is_scoped_and_resident(store):
    assert len(store) == 24, "12 hosts x 2 captures must be 24 trajectories"
    assert not isinstance(store.feats, np.memmap), "small block should be resident"


def test_branch_a_evaluate_and_train_steps(store):
    from torch.utils.data import DataLoader
    from branch_a_gnn_lstm.sequence_dataset import LazyHostSequenceDataset, TECHNIQUE_VOCAB
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    bra = _load("smoke_bra", "scripts/retrain_branch_a_live.py")
    ds = LazyHostSequenceDataset(store, seq_len=5)
    model = MultiTaskLSTM(input_dim=27, hidden_dim=32, num_layers=2,
                          num_techniques=len(TECHNIQUE_VOCAB), num_gradations=4)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    acc = torch.zeros((), dtype=torch.float64)
    for i, bt in enumerate(dl):
        if i == 5:
            break
        tg = {k: bt[k] for k in ("risk", "technique", "gradation")}
        opt.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(model(bt["features"]), tg)
        loss.backward(); opt.step()
        model.uncertainty_loss.project_()
        acc += loss.detach().double().sum()      # the shape bug surfaced here
    m = bra._evaluate(model, DataLoader(ds, batch_size=32), "cpu")
    for key in ("tech_macro_f1", "tech_macro_f1_baseline", "risk_auc", "risk_ece"):
        assert key in m, f"_evaluate no longer reports {key}"
    assert np.isfinite(m["loss"])


def test_branch_b_trainer_runs_an_epoch(store, tmp_path):
    rfm = _load("smoke_rfm_b", "scripts/retrain_future_models_live.py")
    wdt = rfm.train_branch_b_live(store, store, tmp_path / "wdt.pt", epochs=1,
                                  device="cpu", num_workers=0)
    assert (tmp_path / "wdt.pt").exists()
    ck = torch.load(tmp_path / "wdt.pt", map_location="cpu", weights_only=False)
    assert "baselines" in ck and "epoch_history" in ck
    assert not wdt.training, "Branch B must hand DeepOP an eval-mode world model"


def test_deepop_trainer_runs_an_epoch(store, tmp_path):
    rfm = _load("smoke_rfm_d", "scripts/retrain_future_models_live.py")
    wdt = rfm.train_branch_b_live(store, store, tmp_path / "wdt.pt", epochs=1,
                                  device="cpu", num_workers=0)
    rfm.train_deepop_live(store, store, tmp_path / "deepop.pt", epochs=1,
                          device="cpu", wdt=wdt, num_workers=0)
    ck = torch.load(tmp_path / "deepop.pt", map_location="cpu", weights_only=False)
    assert ck.get("selection_metric") in ("hm(macro_f1_free, macro_f1_transitions)",
                                         "macro_f1_free", "neg_val_ce")
    sup = ck.get("label_smoothing_support")
    assert sup is not None and sum(sup) < len(sup), \
        "smoothing support should exclude tokens that never occur"
