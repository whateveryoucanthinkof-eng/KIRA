"""Accuracy alone cannot fail visibly on this corpus, so the eval must say more.

82.5% of Branch A's validation samples are Benign (val_positive_rate 0.1752),
so a model that predicts Benign for every input scores ~0.825 aggregate
accuracy. The observed 0.880 is 5.5 points above that. Accuracy by itself
therefore cannot distinguish a working technique head from a collapsed one --
which is not a hypothetical failure: the TGNE category head scored 0.83 while
emitting a single class for every input, and was caught only once macro F1 was
computed.

These tests pin that `_evaluate` reports enough to catch it, and that the
warning fires.
"""
import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("bra_eval", "scripts/retrain_branch_a_live.py")
bra = importlib.util.module_from_spec(spec)
sys.modules["bra_eval"] = bra
spec.loader.exec_module(bra)

C = 4
BENIGN = 0


class _Loader:
    """Yields batches whose technique labels are 82.5% class 0."""

    def __init__(self, n=4096, bs=128, seed=0):
        rng = np.random.default_rng(seed)
        lab = np.where(rng.random(n) < 0.825, BENIGN, rng.integers(1, C, size=n))
        self.batches = []
        for i in range(0, n, bs):
            t = torch.from_numpy(lab[i:i + bs].astype(np.int64))
            self.batches.append({
                "features": torch.zeros(len(t), 5, 27),
                "risk": torch.zeros(len(t)),
                "technique": t,
                "gradation": torch.zeros(len(t), dtype=torch.long),
            })
        self.labels = lab

    def __iter__(self):
        return iter(self.batches)


class _ConstantModel(torch.nn.Module):
    """Predicts BENIGN for everything -- the collapse we must detect."""

    def eval(self): return self
    def forward(self, x):
        n = x.shape[0]
        logits = torch.full((n, C), -10.0)
        logits[:, BENIGN] = 10.0
        return {"risk_score": torch.zeros(n), "technique_logits": logits,
                "gradation_logits": torch.zeros(n, 4)}
    def compute_loss(self, p, t):
        return torch.zeros(1), {}


class _OracleModel(torch.nn.Module):
    """Predicts the true label -- the healthy case."""

    def __init__(self, loader):
        super().__init__()
        self.it = iter([b["technique"] for b in loader.batches])
    def eval(self): return self
    def forward(self, x):
        t = next(self.it)
        logits = torch.full((len(t), C), -10.0)
        logits[torch.arange(len(t)), t] = 10.0
        return {"risk_score": torch.zeros(len(t)), "technique_logits": logits,
                "gradation_logits": torch.zeros(len(t), 4)}
    def compute_loss(self, p, t):
        return torch.zeros(1), {}


def test_collapsed_head_is_visible_in_the_metrics():
    loader = _Loader()
    m = bra._evaluate(_ConstantModel(), loader, "cpu", num_techniques=C)

    # the trap: accuracy looks respectable
    assert m["tech_accuracy"] > 0.80
    # the tells
    assert m["tech_classes_predicted"] == 1, "should notice only one class is emitted"
    assert m["tech_classes_present"] > 1
    assert m["tech_lift_over_baseline"] == pytest.approx(0.0, abs=1e-9)
    assert m["tech_macro_f1"] < 0.30, "macro F1 must expose what accuracy hides"


def test_a_working_head_scores_well_on_the_same_data():
    loader = _Loader()
    m = bra._evaluate(_OracleModel(loader), loader, "cpu", num_techniques=C)
    assert m["tech_accuracy"] == pytest.approx(1.0)
    assert m["tech_macro_f1"] == pytest.approx(1.0)
    assert m["tech_classes_predicted"] == m["tech_classes_present"]
    assert m["tech_lift_over_baseline"] > 0.15


def test_macro_f1_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    loader = _Loader(seed=5)
    m = bra._evaluate(_ConstantModel(), loader, "cpu", num_techniques=C)
    y_true = loader.labels
    y_pred = np.full_like(y_true, BENIGN)
    present = sorted(set(y_true.tolist()))
    expected = sk.f1_score(y_true, y_pred, labels=present, average="macro",
                           zero_division=0)
    assert m["tech_macro_f1"] == pytest.approx(expected, abs=1e-9)


def test_per_class_support_sums_to_the_dataset():
    loader = _Loader()
    m = bra._evaluate(_ConstantModel(), loader, "cpu", num_techniques=C)
    assert sum(v["support"] for v in m["tech_per_class"].values()) == len(loader.labels)


def test_the_warning_names_the_single_class_case(capsys):
    loader = _Loader()
    m = bra._evaluate(_ConstantModel(), loader, "cpu", num_techniques=C)
    bra._warn_if_head_collapsed(m, "unit test")
    out = capsys.readouterr().out
    assert "SINGLE class" in out and "unit test" in out


def test_no_warning_when_the_head_works(capsys):
    loader = _Loader()
    m = bra._evaluate(_OracleModel(loader), loader, "cpu", num_techniques=C)
    bra._warn_if_head_collapsed(m, "unit test")
    assert "WARNING" not in capsys.readouterr().out
