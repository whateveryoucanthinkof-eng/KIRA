"""The deduplicated rollout cache equals the original one, sample for sample.

_precompute_rollouts used to compute (and store) a rollout per DeepOP sample,
oversampled duplicates included. _precompute_rollouts_unique computes each
distinct window once, at the batch size the original used for it, and maps
every sample to its row. It must also leave torch's global RNG exactly where
the original's single DataLoader pass left it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "perf"))
import retrain_future_models_live as T  # noqa: E402
from synth_store import make_store  # noqa: E402

from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer  # noqa: E402
from deepop_decoder.joint_vocab import get_joint_vocab  # noqa: E402
from deepop_decoder.train_cwa_decoder import LazyCWADataset  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_does_not_depend_on_batch_composition(device):
    """What the dedup relies on: at a fixed batch size, a sample's rollout is
    the same whatever else is in its batch."""
    torch.manual_seed(0)
    w = HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3).to(device).eval()
    g = torch.Generator().manual_seed(1)
    N = 2048
    h = torch.randn(N, 15, 27, generator=g).to(device)
    th = (-torch.rand(N, 15, generator=g).cumsum(1).flip(1) * 30).to(device)
    th[:, -1] = 0
    tf = (torch.rand(N, 5, generator=g).cumsum(1) * 30).to(device)

    def roll(idx):
        with torch.no_grad():
            return torch.cat([w.rollout(h[idx[i:i + 1024]], K=5, t_history=th[idx[i:i + 1024]],
                                        t_future=tf[idx[i:i + 1024]]) for i in range(0, N, 1024)])
    ref = roll(torch.arange(N, device=device))
    for seed in (2, 3):
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).to(device)
        out = roll(perm)
        back = torch.empty_like(out)
        back[perm] = out
        assert torch.equal(back, ref)


@pytest.mark.parametrize("workers", [0, 2])
def test_unique_cache_equals_the_original(tmp_path, workers):
    st = make_store(40_000, 2_000, n_windows=4_000, seed=11, small_frac=0.85, attack_rate=0.3)
    vocab = get_joint_vocab(network_observable_only=True)
    ds = LazyCWADataset(st, vocab, K=5, T=15)
    torch.manual_seed(0)
    w = HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3).eval()
    assert len(ds) % 1024 != 0
    T.LEGACY_LOADER = True
    torch.manual_seed(9)
    ref = T._precompute_rollouts(w, ds, "cpu", str(tmp_path), "ref", 5, num_workers=workers)
    rng_ref = torch.get_rng_state()
    T.LEGACY_LOADER = False
    torch.manual_seed(9)
    new = T._precompute_rollouts(w, ds, "cpu", str(tmp_path), "new", 5, num_workers=workers)
    rng_new = torch.get_rng_state()
    assert isinstance(new, T._RolloutCache)
    assert len(new.rows) < len(ref), "nothing was deduplicated"
    assert torch.equal(rng_ref, rng_new)
    idx = np.arange(len(ds))
    assert np.array_equal(np.asarray(new[idx]), np.asarray(ref[idx]))
