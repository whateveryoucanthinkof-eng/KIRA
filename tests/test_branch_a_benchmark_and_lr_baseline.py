"""PS 26153 benchmark: Branch A vs logistic regression vs persistence, with FPR,
overall and on the onset slice (host benign in its last input window).

The main plan never trained the logistic-regression baseline the problem
statement asks for, the operating-point report had no false positive rate, and
every headline mixed continuations of ongoing attacks (which "copy the last
window" predicts without a model) with real early warnings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _branch_a():
    spec = importlib.util.spec_from_file_location("rbal", ROOT / "scripts" / "retrain_branch_a_live.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _loader(n=4096, L=4, D=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, D, generator=g)
    y = (x[:, -1, 0] + 0.5 * x[:, -2, 1] > 0.3)
    prev = (torch.rand(n, generator=g) < 0.3).long()
    batches = []
    for a in range(0, n, 256):
        batches.append({"features": x[a:a + 256], "risk": y[a:a + 256].float(),
                        "t_history": torch.zeros(len(x[a:a + 256]), L),
                        "prev_attack": prev[a:a + 256]})
    return batches


def test_logistic_baseline_learns_a_linear_target():
    from branch_a_gnn_lstm.logistic_baseline import score_histograms, train_logistic_baseline
    rbal = _branch_a()
    tr, te = _loader(seed=0), _loader(seed=1)
    unpack = lambda b, d: b  # noqa: E731
    m = train_logistic_baseline(tr, unpack, "cpu", seq_len=4, input_dim=3, risk_positive_above=0.5,
                                epochs=5, lr=1e-2, log=lambda *_: None)
    h = score_histograms(m, te, unpack, "cpu", risk_positive_above=0.5, bins=rbal.RISK_BINS)
    assert rbal._auc_from_histograms(h["pos"], h["neg"]) > 0.95
    # onset histograms are a subset of the overall ones
    assert (h["onset_pos"] <= h["pos"]).all() and h["onset_pos"].sum() < h["pos"].sum()


def test_running_standardisation_is_exact():
    from branch_a_gnn_lstm.logistic_baseline import SequenceLogisticBaseline
    m = SequenceLogisticBaseline(2, 2, with_time=False)
    data = torch.randn(1000, 4) * 3 + 7
    for a in range(0, 1000, 64):
        m.update_stats(data[a:a + 64])
    torch.testing.assert_close(m.mean, data.mean(0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m.m2 / m.count.float(), data.var(0, unbiased=False), rtol=1e-4, atol=1e-4)


def test_benchmark_table_reports_fpr_onset_and_persistence():
    rbal = _branch_a()
    B = rbal.RISK_BINS
    pos = np.zeros(B, np.int64); neg = np.zeros(B, np.int64)
    pos[B - 1] = 80; pos[0] = 20          # 80 caught at any threshold > 0
    neg[B - 1] = 10; neg[0] = 890
    rows = rbal.benchmark_table({"m": 0.5}, {"m": (pos, neg, pos // 2, neg // 2)},
                                np.array([70, 5, 30, 895]))
    o = rows["m"]["overall"]
    assert o["tp"] == 80 and o["fp"] == 10
    assert abs(o["fpr"] - 10 / 900) < 1e-12
    assert "onset" in rows["m"]
    p = rows["persistence"]
    assert p["overall"]["recall"] == 70 / 100
    assert p["onset"]["recall"] == 0.0      # it never alerts on a benign last window
    assert "FPR" in rbal.format_benchmark(rows, "test")


def test_point_has_fpr():
    rbal = _branch_a()
    B = rbal.RISK_BINS
    pos = np.zeros(B, np.int64); neg = np.zeros(B, np.int64)
    pos[-1], neg[-1], neg[0] = 5, 5, 90
    c = rbal._pr_curve_from_histograms(pos, neg)
    assert abs(rbal._point(c, B - 1)["fpr"] - 5 / 95) < 1e-12


def test_a_silent_model_is_not_reported_as_perfectly_precise():
    rbal = _branch_a()
    B = rbal.RISK_BINS
    pos = np.zeros(B, np.int64); neg = np.zeros(B, np.int64)
    pos[0], neg[0] = 10, 90                 # every score below any threshold
    rows = rbal.benchmark_table({"m": 0.5}, {"m": (pos, neg, None, None)}, None)
    assert np.isnan(rows["m"]["overall"]["precision"]) and rows["m"]["overall"]["recall"] == 0.0
