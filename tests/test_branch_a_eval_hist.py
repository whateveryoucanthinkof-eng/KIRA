"""`_hist` (index_add_ into a fixed buffer) must equal torch.bincount exactly.

It replaces bincount and boolean-mask selections in Branch A's validation
pass, which each forced a host-device sync per batch on CUDA.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from retrain_branch_a_live import _hist  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_counts_and_masked_counts(device):
    g = torch.Generator().manual_seed(0)
    for n in (4, 196, 2000):
        idx = torch.randint(0, n, (5000,), generator=g).to(device)
        y = (torch.rand(5000, generator=g) > 0.7).to(device)
        ref = torch.bincount(idx, minlength=n)
        assert torch.equal(_hist(idx, n), ref) and _hist(idx, n).dtype == ref.dtype
        assert torch.equal(_hist(idx, n, y.long()), torch.bincount(idx[y], minlength=n))
        assert torch.equal(_hist(idx, n, (~y).long()), torch.bincount(idx[~y], minlength=n))


def test_float64_weights_bitwise_on_cpu():
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 2000, (100_000,), generator=g)
    w = torch.rand(100_000, generator=g, dtype=torch.float64)
    acc_ref = torch.rand(2000, generator=g, dtype=torch.float64)
    acc_new = acc_ref.clone()
    acc_ref += torch.bincount(idx, weights=w, minlength=2000)
    acc_new += _hist(idx, 2000, w)
    assert torch.equal(acc_ref, acc_new)
