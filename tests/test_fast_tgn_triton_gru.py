"""The Triton BiGRU (bita/fast/triton_gru.py) against nn.GRU + pack_padded_sequence.

Not bit-identical (different summation order from cuDNN); the bound is fp32
rounding: relative error well under 1e-5 on outputs and all gradients.

The reference is run with torch.backends.cudnn.allow_tf32 = False. PyTorch's
default (True) lets cuDNN compute the GRU weight gradients in TF32: measured
against a float64 GRU, cuDNN's weight_ih grad is then off by 3.6e-4 relative
while the Triton kernel (IEEE fp32 dots) is off by 2.2e-7. So in training the
reference is the less precise side of this comparison.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bita"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(gru, x, lengths):
    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, h_n = gru(packed)
    return h_n


@pytest.mark.parametrize("E,L,seed", [(1, 1, 0), (7, 3, 1), (180, 9, 2), (256, 64, 3), (33, 64, 4)])
def test_matches_cudnn(E, L, seed):
    from fast.triton_gru import bigru_final_states
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        _check(bigru_final_states, E, L, seed)
    finally:
        torch.backends.cudnn.allow_tf32 = prev


def _check(bigru_final_states, E, L, seed):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    gru = torch.nn.GRU(48, 50, bidirectional=True, batch_first=True).to(dev)
    lengths = torch.randint(1, L + 1, (E,))
    lengths[0] = L
    x = torch.randn(E, L, 48, device=dev)
    w = torch.randn(2, E, 50, device=dev)

    x1 = x.clone().requires_grad_(True)
    h1 = _reference(gru, x1, lengths)
    (h1 * w).sum().backward()
    g1 = {n: p.grad.clone() for n, p in gru.named_parameters()}
    gx1 = x1.grad.clone()
    gru.zero_grad()

    x2 = x.clone().requires_grad_(True)
    h2 = bigru_final_states(gru, x2, lengths.to(dev, torch.int32))
    (h2 * w).sum().backward()
    g2 = {n: p.grad.clone() for n, p in gru.named_parameters()}

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))

    assert rel(h2, h1) < 1e-5
    # padded positions of x get zero gradient in both
    assert rel(x2.grad, gx1) < 1e-5
    for n in g1:
        assert rel(g2[n], g1[n]) < 1e-5, n
