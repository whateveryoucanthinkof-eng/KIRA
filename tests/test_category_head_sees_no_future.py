"""An edge's category prediction must not depend on edges AFTER it in its batch.

ExtendedTGN used to embed twice per batch; with memory updated at the start
of a batch, the second (category) pass first applied the current batch's own
messages, so each edge was classified from memory holding every edge of its
batch -- including later ones. That leaked the future into training and into
the val/test category metrics, and applied every message twice.

The check: change only the LAST edge's features and require the category
logits of every earlier edge to stay put. Under the old two-pass head
(CYBERWORLD_LEGACY_CATEGORY_PASS=1) they move, which proves the test can see it.
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bita"))
sys.path.insert(0, os.path.join(HERE, ".."))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from test_fast_tgn_equivalence import _Data, _graph, _model  # noqa: E402


def _logit_shift(level, legacy, monkeypatch):
    from utils.utils import get_neighbor_finder
    from fast import enable_fast_tgn
    if legacy:
        monkeypatch.setenv("CYBERWORLD_LEGACY_CATEGORY_PASS", "1")
    else:
        monkeypatch.delenv("CYBERWORLD_LEGACY_CATEGORY_PASS", raising=False)
    src, dst, ts, eidx, ef, nf = _graph(seed=3)
    finder = get_neighbor_finder(_Data(src, dst, ts, eidx), uniform=False)
    m = _model(ef, nf, finder)
    if level:
        enable_fast_tgn(m, level=level)
    m.eval()
    bs, rng = 64, np.random.default_rng(0)
    with torch.no_grad():
        for b in range(6):                      # build up memory and pending messages
            sl = slice(b * bs, (b + 1) * bs)
            m.compute_edge_probabilities_and_categories(
                src[sl], dst[sl], rng.choice(dst, bs), ts[sl], eidx[sl], n_neighbors=10)
        sl = slice(6 * bs, 7 * bs)
        neg = rng.choice(dst, bs)
        a_model, b_model = m, copy.deepcopy(m)
        _, _, la = a_model.compute_edge_probabilities_and_categories(
            src[sl], dst[sl], neg, ts[sl], eidx[sl], n_neighbors=10)
        # Only the last edge's CONTENT changes (its features), not which nodes
        # the batch touches: TGN applies each batch node's past pending
        # messages at the start of the batch, so changing the node set would
        # legitimately change earlier edges. Future content must not.
        e2 = eidx[sl].copy()
        e2[-1] = eidx[-1]
        _, _, lb = b_model.compute_edge_probabilities_and_categories(
            src[sl], dst[sl], neg, ts[sl], e2, n_neighbors=10)
    return float((la[:-1] - lb[:-1]).abs().max())


@pytest.mark.parametrize("level", [0, 1, 2])
def test_earlier_edges_ignore_a_later_edge(level, monkeypatch):
    # Not exactly 0: two copies of the model differ by GPU-atomics noise
    # (~6e-8 measured). The old head moved these logits by ~4e-3.
    assert _logit_shift(level, legacy=False, monkeypatch=monkeypatch) < 1e-6


def test_the_old_two_pass_head_did_leak(monkeypatch):
    assert _logit_shift(0, legacy=True, monkeypatch=monkeypatch) > 1e-4
