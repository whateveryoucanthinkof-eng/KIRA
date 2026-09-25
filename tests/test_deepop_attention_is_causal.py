"""DeepOP's attention must not see the future.

The decoder predicts an ATT&CK technique sequence over the forecast horizon.
If its attention could attend forward, the sequence metrics would be inflated
by leakage and the model would be useless in deployment, where the future does
not exist yet.

Two paths exist and only one carries an explicit mask:

    use_cwa=True   -> CausalWindowAttention, expected to mask internally
    use_cwa=False  -> an explicit torch.triu(-inf) causal mask

So the CWA path's causality is a property of that module rather than of the
caller, which makes it exactly the kind of thing worth testing behaviourally
rather than by reading the code.

Method: perturb ONLY the inputs after position t and assert nothing at or
before t changes.
"""

import pytest
import torch

from deepop_decoder.cwa import CausalWindowAttention


def _out(mod, x):
    y = mod(x, x, x)
    return y[0] if isinstance(y, tuple) else y


@pytest.mark.parametrize("window_sizes", [[2, 4, 8], [4], [1], [16]])
def test_future_perturbation_never_changes_the_past(window_sizes):
    torch.manual_seed(0)
    B, T, D = 2, 12, 24
    m = CausalWindowAttention(d_model=D, n_heads=6,
                              window_sizes=window_sizes, dropout=0.0).eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        base = _out(m, x)

    offenders = []
    for t in range(T - 1):
        x2 = x.clone()
        x2[:, t + 1:, :] = torch.randn(B, T - t - 1, D)
        with torch.no_grad():
            got = _out(m, x2)
        if not torch.allclose(base[:, :t + 1], got[:, :t + 1], atol=1e-5):
            offenders.append(t)
    assert not offenders, (
        f"positions {offenders} changed when only the FUTURE was perturbed -- "
        f"the decoder can see ahead"
    )


def test_the_perturbation_actually_changes_something():
    """Guard the guard: if the module ignored its input entirely, the causality
    test above would pass vacuously."""
    torch.manual_seed(0)
    B, T, D = 2, 12, 24
    m = CausalWindowAttention(d_model=D, n_heads=6,
                              window_sizes=[2, 4, 8], dropout=0.0).eval()
    x = torch.randn(B, T, D)
    x2 = x.clone()
    x2[:, 6:, :] = torch.randn(B, T - 6, D)
    with torch.no_grad():
        a, b = _out(m, x), _out(m, x2)
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-5), (
        "perturbing the future changed nothing at all -- the test is vacuous"
    )


def test_the_non_cwa_path_builds_an_explicit_causal_mask():
    src = open("deepop_decoder/forecast_decoder.py").read()
    assert 'torch.triu(torch.full((seq_len, seq_len), float("-inf")' in src, (
        "the non-CWA path must supply its own causal mask"
    )
