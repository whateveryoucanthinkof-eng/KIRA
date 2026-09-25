"""Mandatory baselines (spec 39).

A transformer beating nothing is not a result. Every claim CyberWorld makes
must be stated against these, and PS 26153 explicitly requires the logistic
regression comparison "trained on the same features" — the same 27-D vectors,
not a different or easier representation.

If the world model does not beat persistence on future-state prediction, the
world model is not earning its place in the architecture (spec 62).
"""

from .trivial import (
    PersistenceBaseline,
    LastLabelBaseline,
    MajorityClassBaseline,
    MarkovBaseline,
)
from .learned import LogisticBaseline, GradientBoostingBaseline

__all__ = [
    "PersistenceBaseline",
    "LastLabelBaseline",
    "MajorityClassBaseline",
    "MarkovBaseline",
    "LogisticBaseline",
    "GradientBoostingBaseline",
]
