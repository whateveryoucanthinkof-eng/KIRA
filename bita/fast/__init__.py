"""Opt-in fast TGN training step.

    python bita/train.py ... --fast_step          (or CYBERWORLD_FAST_TGN=1)

enable_fast_tgn(model) swaps the model's class for a subclass whose
compute_temporal_embeddings / compute_edge_probabilities[_and_categories]
compute the same function without per-message Python work or device syncs,
and swaps its Memory for one that stores pending messages as tensors. Nothing
about the model's parameters, state_dict keys, data, batch size or random
stream changes. Equivalence evidence: bita/fast/bench.py and
tests/test_fast_tgn_equivalence.py.
"""
import os


def fast_tgn_requested(flag: bool = False) -> bool:
    return bool(flag) or os.environ.get("CYBERWORLD_FAST_TGN", "") in ("1", "true", "True")


def enable_fast_tgn(tgn, level: int = 99):
    from fast.fast_tgn import enable
    return enable(tgn)
