"""
tests/test_serving_causal_edge_guard.py

Pin for correlation/causal_edge_scorer.py (serving-path review, "Reported,
not fixed"): CausalEdgeScorer.score_candidate_edges gates on
min_score_threshold=0.35 over an MLP that is NEVER loaded from any
checkpoint anywhere in this repository -- grep for `CausalEdgeScorer(` and
`score_candidate_edges` finds no caller outside this module and its
re-export in correlation/__init__.py, so it is unreachable from any live
serving path today.

Unreachable now is not the same as safe forever: it is a plain nn.Module
anyone could import and wire into a pipeline later, and its weights would
still be nothing but PyTorch's random nn.Linear initialisation. This test
pins the guard that makes that impossible: score_edge()/score_candidate_edges
refuse to run on an instance that has never had real weights loaded into it.
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from correlation.causal_edge_scorer import CausalEdgeScorer
from correlation.trajectory_assembler import TrajectoryEntry, Provenance


def _entry(t, host="10.0.0.5", risk=0.5):
    return TrajectoryEntry(
        host_ip=host,
        window_idx=0,
        timestamp=t,
        provenance=Provenance.OBSERVED.value,
        coarse_category="C2",
        technique_id="T1071",
        confidence=0.9,
        risk_score=risk,
    )


class TestScorerIsUnreachableFromServing:
    def test_no_caller_anywhere_outside_its_own_module(self):
        """Grep-equivalent guard: if this ever starts failing, CausalEdgeScorer
        has been wired into a live path and the reachable-path branch of the
        review (fail loudly vs. pass-through) needs to be revisited, not just
        this unreachable-path guard."""
        import subprocess
        out = subprocess.run(
            ["grep", "-rl", "CausalEdgeScorer(", PROJECT_ROOT,
             "--include=*.py"],
            capture_output=True, text=True,
        ).stdout
        callers = {
            line for line in out.splitlines()
            if os.path.relpath(line, PROJECT_ROOT) not in (
                "correlation/causal_edge_scorer.py",
            )
            and "/tests/" not in line
            and not line.endswith("__pycache__")
        }
        assert not callers, f"CausalEdgeScorer now has real callers: {callers}"


class TestScorerRefusesUntrainedWeights:
    def test_score_edge_refuses_before_a_checkpoint_is_loaded(self):
        scorer = CausalEdgeScorer()
        with pytest.raises(RuntimeError, match="(?i)untrained|checkpoint|load"):
            scorer.score_edge(_entry(0.0), _entry(10.0))

    def test_score_candidate_edges_refuses_too(self):
        scorer = CausalEdgeScorer()
        with pytest.raises(RuntimeError):
            scorer.score_candidate_edges([_entry(0.0), _entry(10.0)])

    def test_loading_real_weights_lifts_the_guard(self):
        scorer = CausalEdgeScorer()
        # Simulate a real checkpoint load the same way every other model in
        # this repo loads one, e.g.
        # `self.branch_a.load_state_dict(ckpt["model_state_dict"])`.
        scorer.load_state_dict(scorer.state_dict())
        score = scorer.score_edge(_entry(0.0), _entry(10.0))
        assert 0.0 <= score <= 1.0

    def test_fresh_instances_are_independently_untrained(self):
        """Loading weights into one instance must not lift the guard on a
        different, separately constructed instance."""
        trained = CausalEdgeScorer()
        trained.load_state_dict(trained.state_dict())

        untrained = CausalEdgeScorer()
        with pytest.raises(RuntimeError):
            untrained.score_edge(_entry(0.0), _entry(10.0))
