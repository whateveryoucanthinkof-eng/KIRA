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


def enable_fast_tgn(tgn, level: int = 2):
    """level 1: eager re-expression (forward bit-identical to the reference).
    level 2: + the BiTA BiGRU as Triton kernels (fp32-rounding-level; the
             reference's cuDNN GRU computes its weight grads in TF32).
    level 3: + in training, the fixed-shape part of each batch (graph-attention
             embedding, both heads, both losses; forward and backward) replayed
             as CUDA graphs (bita/fast/graphs.py). Bit-identical to level 2:
             tests/test_fast_tgn_graphs.py, bench.py --compare."""
    from fast.fast_tgn import enable
    return enable(tgn, level=level)
