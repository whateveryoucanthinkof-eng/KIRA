"""
tests/test_serving_causal_edge_guard.py

Pin for correlation/causal_edge_scorer.py. The module used to hold
`CausalEdgeScorer`, an MLP that gated edges at min_score_threshold=0.35 but
was NEVER loaded from any checkpoint -- every score was PyTorch's random
nn.Linear initialisation. testing-prod guarded it (score_edge refused to run
until real weights were loaded); v5.5o removed the MLP and kept only
`HeuristicCausalEdgeScorer`, a transparent, hand-set scorer whose constants
live in HEURISTIC_PARAMS. The merge keeps the removal, which is the stronger
form of the same guarantee: there is no learned edge scorer left that anyone
could wire in with untrained weights.

This file pins that, and keeps testing-prod's "no live caller" check.
"""
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import correlation.causal_edge_scorer as ces
from correlation.causal_edge_scorer import HEURISTIC_PARAMS, HeuristicCausalEdgeScorer
from correlation.trajectory_assembler import Provenance, TrajectoryEntry


def _entry(t, host="10.0.0.5", category="C2"):
    return TrajectoryEntry(
        host_ip=host,
        window_idx=0,
        timestamp=t,
        provenance=Provenance.OBSERVED.value,
        coarse_category=category,
        technique_id="T1071",
        confidence=0.9,
        risk_score=0.5,
    )


def test_no_untrained_learned_scorer_exists():
    assert not hasattr(ces, "CausalEdgeScorer"), (
        "a learned edge scorer is back; it must load real weights before scoring")
    torch = pytest.importorskip("torch")
    assert not isinstance(HeuristicCausalEdgeScorer(), torch.nn.Module), (
        "the edge scorer has parameters again; nothing in this repo trains them")


def test_the_heuristic_is_deterministic_and_bounded():
    a, b = _entry(0.0, category="Recon"), _entry(10.0, category="InitialAccess")
    s1 = HeuristicCausalEdgeScorer().score_edge(a, b)
    s2 = HeuristicCausalEdgeScorer().score_edge(a, b)
    assert s1 == s2 and 0.0 <= s1 <= 1.0
    assert "min_score_threshold" in HEURISTIC_PARAMS, "its gate must stay a named, readable constant"


#: Callers that exist deliberately. control_backend/correlation_service.py
#: feeds the console's Campaign and Incidents pages (2026-10-01); the payload
#: is labelled heuristic, and evaluating HEURISTIC_PARAMS is an open item.
#: Anything else is a new live path.
KNOWN_CALLERS = {os.path.join("control_backend", "correlation_service.py")}


def test_no_caller_anywhere_outside_the_correlation_package():
    """If this starts failing, the scorer has been wired into a NEW live path
    and its hand-set constants need a real evaluation, not just this pin.

    A Python walk rather than `grep`, so it also runs where grep is absent.
    """
    skip = {".git", "node_modules", "__pycache__", "tests", "correlation", ".spill"}
    callers = set()
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    if "HeuristicCausalEdgeScorer(" in fh.read():
                        callers.add(os.path.relpath(p, PROJECT_ROOT))
    assert callers <= KNOWN_CALLERS, (
        f"HeuristicCausalEdgeScorer has new callers: {callers - KNOWN_CALLERS}")
