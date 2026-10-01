"""Histogram accumulation without a host-device sync.

`torch.bincount` on CUDA reads `idx.max()` back to the host to size its
output, and a boolean-mask selection (`x[mask]`) syncs on its nonzero().
Validation loops called those several times per batch. `device_hist` adds into
a fixed-size buffer with index_add_ instead: identical integer counts, and on
the CPU the same sequential float sums as bincount (a fresh buffer filled in
index order). tests/test_branch_a_eval_hist.py.
"""

import torch


def device_hist(idx: torch.Tensor, n: int, weights: torch.Tensor = None) -> torch.Tensor:
    """`torch.bincount(idx, weights, minlength=n)` for `idx` known to lie in [0, n)."""
    if weights is None:
        weights = torch.ones_like(idx, dtype=torch.long)
    return torch.zeros(n, device=idx.device, dtype=weights.dtype).index_add_(0, idx, weights)
