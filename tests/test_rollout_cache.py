"""Caching the frozen Branch-B rollout must change nothing but the clock.

DeepOP conditions every batch on `wdt.rollout(h_history)`. Profiling put 25.4%
of DeepOP's wall clock in that call, and the WDT is frozen and in eval mode
there, so the output is identical on every epoch -- a 6-epoch run recomputed
the same numbers six times.

Caching is only safe because the rollout is deterministic in eval mode. It is
NOT deterministic in train mode (dropout p=0.1 is live, measured max abs diff
1.075 between successive calls on the same input), so the precompute refuses
to run against a WDT that is training. These tests pin both halves.
"""
import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer

spec = importlib.util.spec_from_file_location("rfm_cache", "scripts/retrain_future_models_live.py")
rfm = importlib.util.module_from_spec(spec)
sys.modules["rfm_cache"] = rfm
spec.loader.exec_module(rfm)

K, T, D = 5, 15, 12


class _HistoryOnly(torch.utils.data.Dataset):
    """Minimal stand-in for LazyCWADataset: only h_history is needed."""

    def __init__(self, n=257, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randn(n, T, D, generator=g)

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i):
        return {"h_history": self.h[i], "tag": i}


def _wdt(seed=0):
    torch.manual_seed(seed)
    m = HostWorldDynamicsTransformer(d_latent=D, d_model=64, n_heads=4, n_layers=3)
    m.eval()
    return m


def test_cache_matches_calling_rollout_directly():
    """Equal to float tolerance, not bit-identical -- and the distinction is
    real.

    Repeated calls at the SAME batch shape are bit-identical (verified in
    test_repeated_calls_at_one_batch_shape_are_bit_identical below). Comparing
    a cache built at batch 64 against one forward pass over all 257 samples is
    a different reduction order inside attention, so results differ by float
    non-associativity: measured max abs 3.3e-07 over 15,420 values, on
    embeddings of order 1. That is the same magnitude as changing GPU or cuDNN
    algorithm, and it is not a semantic change -- but it is not zero, so the
    honest claim is "deterministic at fixed batch shape", not "bit-identical
    always".
    """
    wdt, ds = _wdt(), _HistoryOnly()
    cache = rfm._precompute_rollouts(wdt, ds, "cpu", None, "test", K, batch=64, num_workers=0)
    assert cache.shape == (len(ds), K, D)
    with torch.no_grad():
        direct = wdt.rollout(ds.h, K=K).numpy()
    np.testing.assert_allclose(np.asarray(cache), direct, rtol=0, atol=1e-5)
    # and pin how small it actually is, so a real divergence cannot hide here
    assert np.abs(np.asarray(cache) - direct).max() < 1e-5


def test_repeated_calls_at_one_batch_shape_are_bit_identical():
    """The property the cache actually relies on."""
    wdt, ds = _wdt(), _HistoryOnly(n=128)
    a = rfm._precompute_rollouts(wdt, ds, "cpu", None, "a", K, batch=64, num_workers=0)
    b = rfm._precompute_rollouts(wdt, ds, "cpu", None, "b", K, batch=64, num_workers=0)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_cache_is_aligned_to_sample_index_not_batch_order():
    """A misalignment here would train DeepOP on another host's future and
    would not show up as an error anywhere."""
    wdt, ds = _wdt(1), _HistoryOnly(n=133, seed=3)
    cache = rfm._precompute_rollouts(wdt, ds, "cpu", None, "align", K, batch=16, num_workers=0)
    for i in (0, 1, 15, 16, 17, 64, 132):
        with torch.no_grad():
            one = wdt.rollout(ds.h[i:i + 1], K=K).numpy()[0]
        np.testing.assert_allclose(np.asarray(cache[i]), one, rtol=0, atol=1e-6)


def test_wrapper_serves_the_cached_rollout_with_the_sample():
    wdt, ds = _wdt(2), _HistoryOnly(n=40, seed=4)
    cache = rfm._precompute_rollouts(wdt, ds, "cpu", None, "wrap", K, batch=8, num_workers=0)
    wrapped = rfm._WithRollout(ds, cache)
    assert len(wrapped) == len(ds)
    item = wrapped[7]
    assert "h_rollout" in item and item["h_rollout"].shape == (K, D)
    np.testing.assert_allclose(item["h_rollout"].numpy(), np.asarray(cache[7]), rtol=0, atol=0)
    assert item["tag"] == 7, "the wrapper must not disturb the base sample"


def test_precompute_refuses_a_wdt_in_train_mode():
    """Dropout is live in train mode, so the rollout is not a function of its
    input and a cache would freeze one arbitrary draw."""
    wdt, ds = _wdt(5), _HistoryOnly(n=8)
    wdt.train()
    with pytest.raises(RuntimeError, match="train mode"):
        rfm._precompute_rollouts(wdt, ds, "cpu", None, "bad", K, batch=4, num_workers=0)


def test_train_mode_really_is_nondeterministic():
    """Justifies the guard above rather than assuming it."""
    wdt = _wdt(6)
    h = torch.randn(32, T, D)
    wdt.train()
    with torch.no_grad():
        a, b = wdt.rollout(h, K=K), wdt.rollout(h, K=K)
    assert not torch.equal(a, b)
    wdt.eval()
    with torch.no_grad():
        c, d = wdt.rollout(h, K=K), wdt.rollout(h, K=K)
    assert torch.equal(c, d)


def test_cache_file_is_unlinked_so_it_cannot_leak():
    """These are multi-GiB at full density; a crashed run must not leave them."""
    import glob, tempfile, os
    tmp = tempfile.mkdtemp(prefix="rollout_leak_")
    wdt, ds = _wdt(7), _HistoryOnly(n=24)
    rfm._precompute_rollouts(wdt, ds, "cpu", tmp, "leak", K, batch=8, num_workers=0)
    assert glob.glob(os.path.join(tmp, "*.f32")) == [], "spill file was left on disk"
