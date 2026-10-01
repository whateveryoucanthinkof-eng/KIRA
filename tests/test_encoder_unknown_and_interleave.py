"""Encoder: UNKNOWN edges, absent classes, and the capture-interleaved training order.

* UNKNOWN (an edge with no label) is kept as a graph edge but must never be a
  category target or enter a category metric.
* A class absent from a split must read NaN, not 0.0 "recall" -- the guard's
  health check reported absent classes as dead ones.
* Interleaving captures by within-capture progress must leave every node's own
  edge sequence (hence its memory) exactly as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bita"))
from bita.train import Data, FocalLoss, interleave_by_capture  # noqa: E402
from bita.evaluation.eval_edge_prediction_with_categories import (  # noqa: E402
    eval_edge_prediction_with_categories,
)

UNK = 3


def test_alpha_ignores_unknown_and_does_not_let_it_shape_the_rest():
    labels = np.array([0] * 900 + [1] * 50 + [2] * 50 + [UNK] * 5000)
    a = FocalLoss.inverse_frequency_alpha(labels, 4, ignore_classes=[UNK])
    b = FocalLoss.inverse_frequency_alpha(labels[labels != UNK], 4)
    assert float(a[UNK]) == 0.0
    torch.testing.assert_close(a[:3], b[:3])


def test_unknown_rows_add_no_loss_and_do_not_dilute_the_mean():
    torch.manual_seed(0)
    alpha = torch.tensor([1.0, 2.0, 0.5, 0.0])
    fl = FocalLoss(alpha=alpha, gamma=2.0)
    logits = torch.randn(8, 4)
    y = torch.tensor([0, 1, 2, 0, UNK, UNK, UNK, UNK])
    labelled = FocalLoss(alpha=alpha, gamma=2.0)(logits[:4], y[:4])
    torch.testing.assert_close(fl(logits, y), labelled * 4 / 4)
    # all-unknown batch: zero, not NaN
    assert float(fl(logits[4:], y[4:])) == 0.0


def test_without_ignored_classes_the_loss_is_the_plain_mean():
    alpha = torch.tensor([1.0, 2.0, 0.5])
    fl = FocalLoss(alpha=alpha, gamma=2.0)
    assert not fl._ignore
    logits, y = torch.randn(16, 3), torch.randint(0, 3, (16,))
    ref = FocalLoss(alpha=alpha, gamma=2.0, reduction="none")(logits, y).mean()
    assert torch.equal(fl(logits, y), ref)


# ------------------------------------------------------------------- eval
class _Model:
    device = torch.device("cpu")

    def __init__(self, logits):
        self.logits = logits
        self.k = 0

    def eval(self):
        pass

    def compute_edge_probabilities_and_categories(self, s, d, n, t, e, n_neighbors=20):
        b = len(s)
        out = self.logits[self.k:self.k + b]
        self.k += b
        return torch.full((b,), 0.9), torch.full((b,), 0.1), out


class _Sampler:
    def sample(self, size, sources=None, destinations=None):
        return None, np.zeros(size, dtype=int)


def _data(labels):
    n = len(labels)
    return Data(np.arange(n), np.arange(n), np.arange(n, dtype=float), np.arange(1, n + 1),
                np.asarray(labels))


def test_eval_excludes_unknown_and_never_predicts_it():
    labels = [0, 0, 1, UNK, UNK]
    logits = torch.tensor([[5., 0, 0, 9], [5., 0, 0, 9], [0., 5, 0, 9], [0., 0, 0, 9], [0., 0, 0, 9]])
    r = eval_edge_prediction_with_categories(_Model(logits), _Sampler(), _data(labels),
                                             ignore_classes=[UNK])
    acc, per_class, f1 = r[4], r[7], r[8]
    assert acc == 1.0 and f1 == 1.0          # UNK's huge logit is masked, UNK rows left out
    assert UNK not in per_class
    assert np.isnan(per_class[2])            # absent from this split: NaN, not "dead"


def test_eval_on_an_all_unknown_split_returns_link_metrics_only():
    logits = torch.zeros(3, 4)
    r = eval_edge_prediction_with_categories(_Model(logits), _Sampler(), _data([UNK] * 3),
                                             ignore_classes=[UNK])
    assert r[1] == 1.0 and np.isnan(r[8])


# -------------------------------------------------------------- interleave
def test_interleave_keeps_every_capture_in_order_and_mixes_them():
    rng = np.random.default_rng(0)
    # three captures in disjoint absolute time ranges, sorted globally (the old order)
    caps, ts = [], []
    for c, (t0, n) in enumerate(((1e9, 300), (1.5e9, 500), (1.6e9, 200))):
        caps.append(np.full(n, c))
        ts.append(np.sort(t0 + rng.uniform(0, 3600, n)))
    cap = np.concatenate(caps)
    t = np.concatenate(ts)
    n = len(t)
    order = np.argsort(t, kind="stable")
    cap, t = cap[order], t[order]
    # nodes are per capture: node id = capture * 1000 + k
    src = cap * 1000 + rng.integers(0, 20, n)
    dst = cap * 1000 + rng.integers(20, 40, n)
    data = Data(src, dst, t, np.arange(1, n + 1), rng.integers(0, 3, n))
    out = interleave_by_capture(data, cap)        # edge e -> cap[e - 1]

    assert sorted(out.edge_idxs.tolist()) == list(range(1, n + 1))
    # every node's own edge sequence is unchanged
    for node in np.unique(np.concatenate([src, dst])):
        before = data.edge_idxs[(data.sources == node) | (data.destinations == node)]
        after = out.edge_idxs[(out.sources == node) | (out.destinations == node)]
        assert np.array_equal(before, after)
    # and the first tenth of the epoch now holds every capture, not just the oldest
    head = cap[out.edge_idxs[: n // 10] - 1]
    assert set(head.tolist()) == {0, 1, 2}
    assert set(cap[data.edge_idxs[: n // 10] - 1].tolist()) == {0}
