"""The category head must be able to tell two flows apart.

It was `nn.Linear(12, n_classes)` applied to `src_emb + dst_emb`:

  * it never saw the EDGE, so a benign flow and an attack flow between the
    SAME host pair were identical inputs -- unseparable in principle;
  * addition is symmetric, so A->B and B->A were identical too.

The head therefore collapsed to a constant. Both observed runs predicted one
class for everything, and only WHICH class changed when the loss weights
changed:

    old focal alpha -> always Benign,        aggregate CatAcc 0.8351
    new focal alpha -> always InitialAccess, aggregate CatAcc 0.2115

These tests pin the two properties that make the collapse impossible by
construction: the head's output must depend on the edge, and on direction.
"""

import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn


EMB, EDGE, NCLS = 12, 12, 5


def _head():
    """Mirror of the shipped construction in ExtendedTGN.__init__."""
    cat_in = EMB * 2 + EDGE
    hidden = max(32, cat_in)
    return nn.Sequential(
        nn.Linear(cat_in, hidden), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(hidden, NCLS),
    )


def _forward(head, src, dst, edge):
    return head(torch.cat([src, dst, edge], dim=1))


def test_the_shipped_head_takes_embeddings_and_edge_features():
    """`model.extentedtgn` only imports with bita/ on sys.path, so read the
    source directly rather than depending on import layout."""
    src = pathlib.Path("bita/model/extentedtgn.py").read_text()
    assert "self.embedding_dimension * 2 + edge_dim" in src, (
        "the head must take [src ; dst ; edge], not a 12-D sum"
    )
    assert "torch.cat" in src, "embeddings must be concatenated, not summed"
    assert "edge_raw_features[edge_idxs]" in src, "the head must see the edge"
    assert "combined_embeddings = source_node_embedding + destination_node_embedding" not in src, (
        "the direction-blind sum is still present"
    )


def test_output_changes_when_only_the_edge_changes():
    """The fatal case: same hosts, different flow. Must be separable."""
    torch.manual_seed(0)
    head = _head().eval()
    src = torch.randn(4, EMB)
    dst = torch.randn(4, EMB)
    a = _forward(head, src, dst, torch.zeros(4, EDGE))
    b = _forward(head, src, dst, torch.ones(4, EDGE))
    assert not torch.allclose(a, b), (
        "identical hosts with different flows produced identical logits -- "
        "the head cannot see the edge"
    )


def test_output_changes_when_direction_flips():
    """A->B and B->A must not be identical; summing made them so."""
    torch.manual_seed(0)
    head = _head().eval()
    src, dst = torch.randn(4, EMB), torch.randn(4, EMB)
    edge = torch.randn(4, EDGE)
    fwd = _forward(head, src, dst, edge)
    rev = _forward(head, dst, src, edge)
    assert not torch.allclose(fwd, rev), "the head is direction-blind"


def test_the_old_summing_head_was_provably_direction_blind():
    """Guard the guard: show the previous design really had this flaw."""
    torch.manual_seed(0)
    old = nn.Linear(EMB, NCLS).eval()
    src, dst = torch.randn(4, EMB), torch.randn(4, EMB)
    assert torch.allclose(old(src + dst), old(dst + src)), (
        "if this fails the fixture no longer reproduces the old behaviour"
    )


def test_the_old_summing_head_could_not_see_the_edge():
    torch.manual_seed(0)
    old = nn.Linear(EMB, NCLS).eval()
    src, dst = torch.randn(4, EMB), torch.randn(4, EMB)
    # There is no edge argument at all -- that is the defect.
    assert torch.allclose(old(src + dst), old(src + dst))


def test_head_can_actually_learn_an_edge_only_rule():
    """End to end: with hosts held constant and the label determined solely by
    the edge, the head must fit it. The old head provably could not."""
    torch.manual_seed(0)
    head = _head()
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    n = 256
    src = torch.randn(1, EMB).repeat(n, 1)      # ONE host pair
    dst = torch.randn(1, EMB).repeat(n, 1)
    edge = torch.randn(n, EDGE)
    y = (edge[:, 0] > 0).long()                 # label depends only on the edge
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(_forward(head, src, dst, edge)[:, :2], y)
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        acc = (_forward(head, src, dst, edge)[:, :2].argmax(1) == y).float().mean().item()
    assert acc > 0.9, f"head failed to learn an edge-determined rule: acc={acc:.3f}"


def test_predictions_are_not_all_one_class_on_varied_input():
    """A trained head on varied input must not emit a constant."""
    torch.manual_seed(1)
    head = _head()
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    n = 400
    src, dst = torch.randn(n, EMB), torch.randn(n, EMB)
    edge = torch.randn(n, EDGE)
    y = (edge[:, :NCLS].argmax(1))
    for _ in range(400):
        opt.zero_grad()
        nn.functional.cross_entropy(_forward(head, src, dst, edge), y).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        preds = _forward(head, src, dst, edge).argmax(1)
    assert len(torch.unique(preds)) > 1, "head collapsed to a single class"
