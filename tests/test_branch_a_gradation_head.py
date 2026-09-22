"""The gradation head, and the post-hoc fits, end to end through `_evaluate`.

Until 2026-09-22 the gradation head had **no evaluation at all**; then it got
`gradation_accuracy`. On this corpus that single number cannot fail: every
benign window maps to level 0 through GRADATION_LEVELS, so ~82.5% of targets
are level 0 and a head that emits 0 for everything scores ~0.825. That is the
identical trap the technique head was already instrumented against -- the TGNE
category head scored 0.83 while predicting one class, and was only caught by
macro F1.

The second half of this file runs the real `_evaluate` -> `fit_operating_point`
-> `fit_temperature` -> conformal path on a stub model, because those pieces
are individually tested elsewhere and the thing most likely to break is the
wiring between them.
"""
import argparse
import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("bra_grad", "scripts/retrain_branch_a_live.py")
bra = importlib.util.module_from_spec(spec)
sys.modules["bra_grad"] = bra
spec.loader.exec_module(bra)

C = 6          # technique classes in these fixtures
G = 4          # gradation levels


class _Loader:
    """Batches whose gradation levels are 82.5% level 0, like the corpus."""

    def __init__(self, n=4096, bs=128, seed=0, risk_signal=True):
        rng = np.random.default_rng(seed)
        grad = np.where(rng.random(n) < 0.825, 0, rng.integers(1, G, size=n))
        tech = np.where(grad == 0, 0, rng.integers(1, C, size=n))
        # The TARGET is shaped like the real severity score: exactly 0.0 for a
        # benign window, 0.50-0.96 for an attack one. `risk > 0` has to
        # partition the split, or nothing downstream has two classes to work
        # with.
        risk = np.where(grad > 0, 0.50 + 0.46 * rng.random(n), 0.0)
        # The PREDICTION is separate and imperfect, so AUC is well above
        # chance without being 1.0.
        pred = np.where(grad > 0, rng.beta(4, 6, n), rng.beta(1, 12, n))
        if not risk_signal:
            pred = np.full(n, 0.1752)
        self.batches, self.grad, self.tech = [], grad, tech
        for i in range(0, n, bs):
            self.batches.append({
                "features": torch.zeros(len(grad[i:i + bs]), 5, 27),
                "risk": torch.from_numpy(risk[i:i + bs]).float(),
                "pred_risk": torch.from_numpy(pred[i:i + bs]).float(),
                "technique": torch.from_numpy(tech[i:i + bs].astype(np.int64)),
                "gradation": torch.from_numpy(grad[i:i + bs].astype(np.int64)),
            })

    def __iter__(self):
        return iter(self.batches)


class _Stub(torch.nn.Module):
    """A model whose heads can be told to collapse or to work."""

    def __init__(self, loader, gradation="oracle", technique="oracle",
                 risk="signal"):
        super().__init__()
        self.loader = loader
        self.gradation, self.technique, self.risk = gradation, technique, risk
        self._i = 0

    def eval(self):
        self._i = 0
        return self

    def _onehot(self, idx, k):
        out = torch.full((len(idx), k), -6.0)
        out[torch.arange(len(idx)), idx] = 6.0
        return out

    def forward(self, x):
        b = self.loader.batches[self._i]
        self._i += 1
        n = len(b["gradation"])
        g = (b["gradation"] if self.gradation == "oracle"
             else torch.zeros(n, dtype=torch.long))
        t = (b["technique"] if self.technique == "oracle"
             else torch.zeros(n, dtype=torch.long))
        r = b["pred_risk"]
        return {"risk_score": r,
                "technique_logits": self._onehot(t, C),
                "technique_logits_raw": self._onehot(t, C),
                "gradation_logits": self._onehot(g, G)}

    def compute_loss(self, p, t):
        return torch.zeros(1), {}


# --- the gradation head is now visible -------------------------------------

def test_a_collapsed_gradation_head_is_visible_in_the_metrics():
    loader = _Loader()
    m = bra._evaluate(_Stub(loader, gradation="collapsed"), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    assert m["gradation_accuracy"] > 0.80, "the trap: accuracy looks respectable"
    assert m["gradation_classes_predicted"] == 1
    assert m["gradation_classes_present"] > 1
    assert m["gradation_lift_over_baseline"] == pytest.approx(0.0, abs=1e-9)
    assert m["gradation_macro_f1"] < 0.30


def test_a_working_gradation_head_scores_well_on_the_same_data():
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    assert m["gradation_accuracy"] == pytest.approx(1.0)
    assert m["gradation_macro_f1"] == pytest.approx(1.0)
    assert m["gradation_classes_predicted"] == m["gradation_classes_present"]


def test_gradation_macro_f1_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    loader = _Loader(seed=5)
    m = bra._evaluate(_Stub(loader, gradation="collapsed"), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    y = loader.grad
    present = sorted(set(y.tolist()))
    expected = sk.f1_score(y, np.zeros_like(y), labels=present,
                           average="macro", zero_division=0)
    assert m["gradation_macro_f1"] == pytest.approx(expected, abs=1e-9)


def test_the_gradation_warning_names_the_single_level_case(capsys):
    loader = _Loader()
    m = bra._evaluate(_Stub(loader, gradation="collapsed"), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    bra._warn_if_gradation_collapsed(m, "unit test")
    out = capsys.readouterr().out
    assert "SINGLE level" in out and "unit test" in out


def test_no_gradation_warning_when_the_head_works(capsys):
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    bra._warn_if_gradation_collapsed(m, "unit test")
    assert "WARNING" not in capsys.readouterr().out


def test_gradation_per_class_support_sums_to_the_dataset():
    loader = _Loader()
    m = bra._evaluate(_Stub(loader, gradation="collapsed"), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    total = sum(v["support"] for v in m["gradation_per_class"].values())
    assert total == len(loader.grad)


def test_the_level_names_say_which_categories_each_level_covers(capsys):
    """GRADATION_LEVELS folds four coarse categories onto level 3; a bare
    "class_3" in the per-class table would be unreadable."""
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    bra._print_per_class(m["gradation_per_class"], "unit",
                         names=bra.GRADATION_NAMES, label="gradation")
    out = capsys.readouterr().out
    assert "C2/Lateral/Exfil/Impact" in out and "InitialAccess/Exec" in out


def test_the_technique_metrics_are_unchanged_by_the_refactor():
    """Both heads now share `_metrics_from_confusion`; the technique numbers
    must be exactly what they were."""
    sk = pytest.importorskip("sklearn.metrics")
    loader = _Loader(seed=9)
    m = bra._evaluate(_Stub(loader, technique="collapsed"), loader, "cpu",
                      num_techniques=C, num_gradations=G)
    y = loader.tech
    present = sorted(set(y.tolist()))
    assert m["tech_macro_f1"] == pytest.approx(
        sk.f1_score(y, np.zeros_like(y), labels=present, average="macro",
                    zero_division=0), abs=1e-9)
    assert m["tech_accuracy"] == pytest.approx(float((y == 0).mean()), abs=1e-12)


# --- the post-hoc stage, wired together ------------------------------------

def _args(**kw):
    d = dict(no_fit_temperature=False, conformal_alpha=0.05,
             operating_point_criterion="budgeted_f1", alert_budget=2.0,
             risk_objective="bce", risk_target="severity")
    d.update(kw)
    return argparse.Namespace(**d)


def test_evaluate_returns_the_histograms_the_fits_need():
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                      num_gradations=G)
    n = len(loader.grad)
    assert m["risk_pos_hist"].sum() + m["risk_neg_hist"].sum() == n
    assert m["risk_resid_hist"].sum() == n


def test_the_post_hoc_stage_fits_all_three_and_leaves_them_on_the_model():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    loader = _Loader()
    metrics = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                            num_gradations=G, collect_logits=True)
    real = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1,
                         num_techniques=C).eval()
    fits = bra.calibrate_and_fit_operating_point(real, metrics, _args())

    op = fits["operating_point"]
    assert op["fitted"] and 0.0 < op["alert_threshold"] < 1.0
    assert op["risk_objective"] == "bce" and op["risk_target"] == "severity"
    assert "legacy_0_65" in op, "the cost of the old cut has to be recorded"
    assert op["legacy_0_65"]["recall"] < op["recall"]

    conf = fits["risk_conformal"]
    assert conf["fitted"]
    assert float(real.risk_conformal_halfwidth) == pytest.approx(conf["half_width"])

    # the oracle technique head is perfectly separable, so a temperature fit
    # would run to the grid boundary and must be refused rather than stored
    temp = fits["temperature"]
    assert temp["fitted"] is False and float(real.temperature_fitted) == 0.0


def test_a_chance_risk_head_still_yields_a_threshold_but_a_useless_one():
    """A constant score cannot rank, so every threshold is degenerate."""
    loader = _Loader(risk_signal=False)
    metrics = bra._evaluate(_Stub(loader, risk="constant"), loader, "cpu",
                            num_techniques=C, num_gradations=G)
    op = bra.fit_operating_point(metrics["risk_pos_hist"],
                                 metrics["risk_neg_hist"])
    # Every threshold alerts on everything or on nothing. Returning the
    # least-bad degenerate point would put "alert on everything" into serving
    # under the name of a fitted threshold.
    assert op["fitted"] is False
    assert "constant" in op["reason"]
    assert "alert_threshold" not in op


def test_bulky_working_data_never_reaches_the_checkpoint():
    """The collected logits are 57 MB at the real validation split's size and
    the histograms are working data, not results. `slim` is what keeps a
    per-epoch history from becoming gigabytes."""
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                      num_gradations=G, collect_logits=True)
    assert m["technique_logits"] is not None
    s = bra.slim(m)
    for k in bra.BULKY_METRIC_KEYS:
        assert k not in s
    assert "tech_per_class" in s
    assert "gradation_per_class" not in bra.slim(m, drop_per_class=True)


def test_logits_are_only_collected_when_asked():
    loader = _Loader()
    m = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                      num_gradations=G)
    assert m["technique_logits"] is None and m["technique_labels"] is None


# --- the hazard target changes what "positive" means -----------------------

def test_the_positive_event_threshold_moves_the_base_rate():
    """Under the hazard target, `risk > 0` means "attacked at some later
    point" -- nearly everything. The event that matters is "within one
    forecast horizon", i.e. hazard >= exp(-1)."""
    import math
    rng = np.random.default_rng(1)
    n = 4096
    # hazard-shaped: mostly small, a tail near 1
    haz = rng.beta(0.7, 1.5, n)
    batches = []
    for i in range(0, n, 128):
        h = haz[i:i + 128]
        batches.append({
            "features": torch.zeros(len(h), 5, 27),
            "risk": torch.from_numpy(h).float(),
            "pred_risk": torch.from_numpy(h).float(),
            "technique": torch.zeros(len(h), dtype=torch.long),
            "gradation": torch.zeros(len(h), dtype=torch.long),
        })

    class _L:
        def __init__(self): self.batches = batches
        def __iter__(self): return iter(self.batches)

    loader = _L()
    m0 = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                       num_gradations=G, risk_positive_above=0.0)
    mh = bra._evaluate(_Stub(loader), loader, "cpu", num_techniques=C,
                       num_gradations=G,
                       risk_positive_above=math.exp(-1.0) - 1e-6)
    assert m0["risk_base_rate"] == pytest.approx(1.0, abs=1e-6), \
        "risk > 0 is nearly constant under a hazard target -- the defect"
    assert 0.2 < mh["risk_base_rate"] < 0.6
    assert mh["risk_positive_above"] == pytest.approx(np.exp(-1.0) - 1e-6)
