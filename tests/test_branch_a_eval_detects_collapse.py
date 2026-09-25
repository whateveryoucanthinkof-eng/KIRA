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
    def forward(self, x, t_history=None):   # MultiTaskLSTM's time channel
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
    def forward(self, x, t_history=None):   # MultiTaskLSTM's time channel
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


# --- selecting a checkpoint on a single noisy reading -----------------------

# The real validation losses from the full-density run of 2026-09-21. Epoch 6
# won selection at 0.6601 and then scored 26.36 on the held-out test.
REAL_RUN_LOSSES = [1.8819, 2.0814, 1.3938, 1.2351, 1.3904, 0.6601, 1.4085, 1.2117]


def _history(losses):
    return [{"loss": l, "epoch": i + 1} for i, l in enumerate(losses)]


def test_outlier_selection_is_flagged_on_the_real_run(capsys):
    bra._flag_outlier_selection(_history(REAL_RUN_LOSSES),
                                {"loss": 0.6601, "epoch": 6})
    out = capsys.readouterr().out
    assert "NOTE" in out and "epoch (6)" in out
    assert "epoch_history" in out, "must say where the other epochs' metrics are"


def test_a_typical_winner_is_not_flagged(capsys):
    bra._flag_outlier_selection(_history(REAL_RUN_LOSSES),
                                {"loss": 1.2351, "epoch": 4})
    assert "NOTE" not in capsys.readouterr().out


def test_short_runs_are_not_flagged(capsys):
    """Three epochs cannot establish a trend to be an outlier against."""
    bra._flag_outlier_selection(_history([2.0, 1.5, 0.2]), {"loss": 0.2, "epoch": 3})
    assert "NOTE" not in capsys.readouterr().out


def test_flagging_never_changes_the_selection():
    """It reports; it must not pick a different checkpoint."""
    hist = _history(REAL_RUN_LOSSES)
    best = {"loss": 0.6601, "epoch": 6}
    before = dict(best)
    bra._flag_outlier_selection(hist, best)
    assert best == before
    assert len(hist) == len(REAL_RUN_LOSSES)


# --- the warning must judge macro F1, not accuracy -------------------------

def test_a_focal_loss_head_is_not_warned_about_for_losing_accuracy():
    """Branch A epoch 1 (2026-09-22): accuracy 0.791 against a 0.900 majority
    share, but macro F1 0.397 against a majority predictor's 0.316. Trading
    majority accuracy for minority recall is what focal loss is for, so the
    accuracy-based warning was punishing the head for working as designed."""
    m = {"tech_accuracy": 0.791, "tech_majority_baseline": 0.900,
         "tech_lift_over_baseline": -0.109,
         "tech_macro_f1": 0.397, "tech_macro_f1_baseline": 0.316,
         "tech_macro_f1_lift": 0.081,
         "tech_classes_present": 3, "tech_classes_predicted": 6}
    import io as _io, contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        bra._warn_if_head_collapsed(m, "unit")
    assert "WARNING" not in buf.getvalue(), buf.getvalue()


def test_a_head_below_the_macro_f1_baseline_is_warned_about(capsys):
    m = {"tech_accuracy": 0.95, "tech_majority_baseline": 0.900,
         "tech_lift_over_baseline": 0.05,
         "tech_macro_f1": 0.30, "tech_macro_f1_baseline": 0.316,
         "tech_macro_f1_lift": -0.016,
         "tech_classes_present": 3, "tech_classes_predicted": 3}
    bra._warn_if_head_collapsed(m, "unit")
    out = capsys.readouterr().out
    assert "macro F1" in out and "adding nothing over a constant" in out


def test_the_macro_f1_baseline_matches_a_hand_computed_majority_predictor():
    """A constant predictor gets recall 1.0 on the majority class and
    precision equal to its share, and F1 = 0 elsewhere."""
    import numpy as np
    C = 3
    support = np.array([900, 60, 40])          # majority share 0.90
    cm = np.zeros((C, C), dtype=np.int64)
    cm[:, 0] = support                          # predict class 0 for everything
    got = bra._metrics_from_confusion(cm)
    p_maj, r_maj = 0.90, 1.0
    expected = (2 * p_maj * r_maj / (p_maj + r_maj)) / 3
    assert got["macro_f1_baseline"] == pytest.approx(expected, abs=1e-9)
    # and a constant predictor's own macro F1 must equal that baseline
    assert got["macro_f1"] == pytest.approx(expected, abs=1e-9)
    assert got["macro_f1_lift"] == pytest.approx(0.0, abs=1e-9)
