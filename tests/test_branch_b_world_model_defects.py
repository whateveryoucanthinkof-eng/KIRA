"""Branch B: the defects found while diagnosing its immediate plateau.

Run 1 moved train loss 1.9% across 6 epochs and had its best validation at
epoch 1. Measured, rather than guessed at: the world model is fine and the
risk head is not. Running the shipped checkpoint under the rollout it was
actually trained with, over 103,110 rollout samples from the validation
capture wed_29_csv.csv:

    embeddings  mse_model 0.093225  mse_persistence 0.123277  SKILL +0.244
    risk        bce_model 0.30570   bce_constant    0.25083   WORSE
                mae_model 0.22189   mae_predict_0   0.06892   3.2x WORSE

The objective is `sum_k 0.9^k MSE_k + BCE`, and those two terms are roughly
equal in size (0.373 and ~0.39 at epoch 6). Half the loss is a term that is
already beaten by a constant and cannot move, which is most of why the total
barely moved. The defects below are the ones that are fixable in this
package; the risk target itself is not (see the note on
test_bce_of_the_best_constant_equals_the_entropy_of_the_mean).

Each test names the specific behaviour it pins. They are cheap: the model is
112k parameters and every case runs on CPU in well under a second.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from branch_b_world_model.rollout_encoder_decoder import (
    TIME_ENCODING_VERSION,
    ContinuousTimeEncoding,
    HostWorldDynamicsTransformer,
)


def _model(seed=0):
    torch.manual_seed(seed)
    return HostWorldDynamicsTransformer(
        d_latent=12, d_model=64, n_heads=4, n_layers=3).eval()


# ---------------------------------------------------------------------------
# Defect 1 -- the temporal encoding carried no positional signal
# ---------------------------------------------------------------------------

def test_time_encoding_varies_along_the_sequence():
    """Every caller passed torch.full((B, T), 2.0), so cos(Linear(t)) produced
    ONE vector broadcast over all 15 positions. Adding a constant to every
    position is adding nothing: the encoder was NoPE.
    """
    enc = ContinuousTimeEncoding(d_time=64, d_model=64)

    const = torch.full((4, 15), 2.0)
    flat = enc(const).detach()
    assert float((flat - flat[:, :1, :]).abs().max()) == 0.0, (
        "a constant input must still produce a constant encoding -- if this "
        "ever changes, the regression below stops testing what it claims to")

    # what rollout() now passes: each position's elapsed time
    elapsed = (torch.arange(-14, 1, dtype=torch.float32) * 2.0).expand(4, 15)
    varied = enc(elapsed).detach()
    assert float((varied - varied[:, :1, :]).abs().max()) > 0.5

    # every position must be distinguishable from every other
    v = varied[0]
    pair = torch.cdist(v.unsqueeze(0), v.unsqueeze(0))[0]
    off_diagonal = pair + torch.eye(15) * 1e9
    assert float(off_diagonal.min()) > 1e-3


def test_rollout_positions_carry_the_step_index():
    """The shared out_head reads the LAST context position. Its elapsed time
    must be +k*dt at rollout step k, otherwise nothing in the input tells the
    model how far ahead it is predicting -- which is the job the hardcoded
    decay_factor damping was doing by hand.
    """
    m = _model()
    dev = torch.device("cpu")
    for k in range(6):
        t = m._elapsed_times(1, 15, k, 2.0, dev)[0]
        assert t.shape == (15,)
        assert float(t[-1]) == pytest.approx(2.0 * k)
        assert torch.all(t[1:] > t[:-1]), "elapsed time must be increasing"
        # exactly min(k, 15) positions are predictions (t > 0)
        assert int((t > 0).sum()) == min(k, 15)


def test_rollout_uses_the_history_order():
    """With no positional signal a causal transformer is nearly permutation
    invariant over its history. Measured on the shipped configuration before
    the fix: reversing the first 14 steps changed the rollout by 4.35e-3 while
    the rollout itself moved 3.80e-1 off persistence -- 1.14%.

    This pins that the order now reaches the output at all. It is a capacity
    check on an untrained model, not a claim about accuracy.
    """
    m = _model()
    torch.manual_seed(3)
    h = torch.randn(32, 15, 12) * 0.5
    reversed_history = torch.cat([h[:, :14].flip(1), h[:, -1:]], dim=1)
    with torch.no_grad():
        a = m.rollout(h, K=5)
        b = m.rollout(reversed_history, K=5)
    assert float((a - b).pow(2).mean()) > 1e-4


def test_explicit_times_on_a_uniform_grid_reproduce_the_default():
    """t_history/t_future are an addition, not a change of default: supplying
    the grid the default assumes must give bit-identical output."""
    m = _model()
    torch.manual_seed(5)
    h = torch.randn(16, 15, 12) * 0.5
    t_hist = (torch.arange(-14, 1, dtype=torch.float32) * 2.0).expand(16, 15)
    t_fut = (torch.arange(1, 6, dtype=torch.float32) * 2.0).expand(16, 5)
    with torch.no_grad():
        a = m.rollout(h, K=5)
        b = m.rollout(h, K=5, t_history=t_hist, t_future=t_fut)
    assert torch.equal(a, b)


def test_irregular_step_spacing_changes_the_forecast():
    """On wed_29_csv.csv the median gap between a host's consecutive snapshots
    is 7 windows, not 1. A model told 'every step is 2 s' cannot distinguish a
    2 s horizon from a 290 s one; one told the truth must.
    """
    m = _model()
    torch.manual_seed(7)
    h = torch.randn(16, 15, 12) * 0.5
    tight = (torch.arange(-14, 1, dtype=torch.float32) * 2.0).expand(16, 15)
    loose = (torch.arange(-14, 1, dtype=torch.float32) * 14.0).expand(16, 15)
    with torch.no_grad():
        a = m.rollout(h, K=5, t_history=tight)
        b = m.rollout(h, K=5, t_history=loose)
    assert float((a - b).pow(2).mean()) > 1e-5


def test_checkpoints_record_the_encoding_they_were_trained_with():
    """A pre-fix checkpoint was fit against a rollout() with a constant
    positional input. It must still load -- serving cannot break on an import
    -- but it must say so."""
    m = _model()
    assert int(m._time_encoding_version) == TIME_ENCODING_VERSION

    legacy = {k: v for k, v in m.state_dict().items()
              if k != "_time_encoding_version"}
    other = _model(seed=1)
    other.load_state_dict(legacy)                       # strict=True, no raise
    assert int(other._time_encoding_version) == 1, (
        "loading pre-fix weights must mark the instance as pre-fix so a "
        "re-save propagates the warning rather than laundering it")


# ---------------------------------------------------------------------------
# Defect 2 -- the causal mask was rebuilt on every forward
# ---------------------------------------------------------------------------

def test_causal_mask_is_cached_and_still_correct():
    m = _model()
    dev = torch.device("cpu")
    first = m._generate_causal_mask(15, dev)
    assert m._generate_causal_mask(15, dev) is first, "not cached"
    expected = torch.triu(torch.full((15, 15), float("-inf")), diagonal=1)
    assert torch.equal(first, expected)
    assert m._generate_causal_mask(7, dev).shape == (7, 7)
    assert m._generate_causal_mask(15, dev) is first, "cache keyed wrongly"


# ---------------------------------------------------------------------------
# Defect 3 -- the "95th-percentile empirical" radii were invented
# ---------------------------------------------------------------------------

def test_uncalibrated_radii_are_nan_not_a_made_up_number():
    """The old table was [0.01890, 0.03661, 0.05327, 0.06904] with steps 5+
    from `base[-1] + 0.015*sqrt(step-3)`. Nothing produces those numbers, the
    extrapolation's growth law contradicts the table's own, and against the
    measured residual (0.0403/element at k=1 -> L2 0.70 over 12 dims) they are
    3-19x too small.
    """
    m = _model()
    torch.manual_seed(11)
    h = torch.randn(4, 15, 12) * 0.5
    with torch.no_grad():
        _, radii = m.rollout_with_uncertainty(h, K=5)
    assert radii.shape == (5,)
    assert bool(torch.isnan(radii).all())
    assert m.radii_norm_ is None
    # and specifically: not the old table
    assert not np.allclose(radii.numpy()[:4], [0.01890, 0.03661, 0.05327, 0.06904],
                           equal_nan=False)


def test_calibrate_radii_gives_finite_sample_conformal_coverage():
    """Coverage is the only thing a radius means. With alpha=0.05 the fitted
    radius must cover >= 95% of fresh residuals from the same distribution.
    """
    from cyberworld_v4.conformal import conformal_quantile

    m = _model()
    rng = np.random.default_rng(0)
    scale = np.linspace(0.2, 0.6, 5).reshape(1, 5, 1)
    cal = rng.standard_normal((4000, 5, 12)) * scale
    test = rng.standard_normal((4000, 5, 12)) * scale

    radii = m.calibrate_radii(cal, alpha=0.05, norm="l2")
    assert m.radii_norm_ == "l2"
    assert m.radii_alpha_ == 0.05
    assert m.radii_n_calibration_ == 4000
    assert torch.all(radii[1:] > radii[:-1]), "radius must grow with horizon"

    # matches the reference quantile exactly, per step
    for k in range(5):
        ref = conformal_quantile(np.linalg.norm(cal[:, k], axis=-1), 0.05)
        assert float(radii[k]) == pytest.approx(ref, rel=1e-6)

    cov = (np.linalg.norm(test, axis=-1) <= radii.numpy()[None, :]).mean(axis=0)
    assert cov.min() >= 0.93, f"per-step coverage {cov}"

    # once calibrated, rollout_with_uncertainty hands them back
    torch.manual_seed(13)
    with torch.no_grad():
        _, got = m.rollout_with_uncertainty(torch.randn(4, 15, 12), K=5)
    assert torch.allclose(got, radii)


def test_linf_radii_are_a_box_not_a_ball():
    m = _model()
    rng = np.random.default_rng(1)
    res = rng.standard_normal((2000, 3, 12)) * 0.3
    l2 = m.calibrate_radii(res, alpha=0.05, norm="l2").clone()
    linf = m.calibrate_radii(res, alpha=0.05, norm="linf")
    assert m.radii_norm_ == "linf"
    assert torch.all(linf < l2), "an Linf half-width cannot exceed the L2 radius"


def test_calibrate_radii_rejects_a_wrong_shape():
    m = _model()
    with pytest.raises(ValueError):
        m.calibrate_radii(np.zeros((10, 5)))
    with pytest.raises(ValueError):
        m.calibrate_radii(np.zeros((10, 5, 12)), norm="l3")


# ---------------------------------------------------------------------------
# Defect 4 -- rollout_multi_host runs 948 never-trained parameters
# ---------------------------------------------------------------------------

def test_multi_host_refuses_to_run_the_untrained_lateral_layer():
    """No training path in this repository touches MultiHostInteractionLayer.
    Read out of both shipped checkpoints, cross_attn.in_proj_weight has rms
    0.2021 and 0.1993 against a fresh init's 0.2051. Running it would add
    0.35 * gate(random) * attention(random) to every predicted state and it
    would look like modelled lateral movement.
    """
    m = _model()
    torch.manual_seed(17)
    h = torch.randn(4, 15, 12) * 0.5
    with pytest.raises(RuntimeError, match="random initialisation"):
        m.rollout_multi_host(h, [(0, 1), (1, 2)], K=3)

    # no edges means the layer is never entered, so that stays allowed
    with torch.no_grad():
        out = m.rollout_multi_host(h, [], K=3)
    assert out.shape == (4, 3, 12)
    with torch.no_grad():
        opted_in = m.rollout_multi_host(h, [(0, 1)], K=3, allow_untrained_lateral=True)
    assert opted_in.shape == (4, 3, 12)


# ---------------------------------------------------------------------------
# InfiltrationRiskHead -- the cumulative form is right; pin it
# ---------------------------------------------------------------------------

def test_cumulative_risk_log_space_equals_the_direct_product():
    head = InfiltrationRiskHead(d_latent=6).eval()
    torch.manual_seed(19)
    for r in (torch.rand(64, 5),
              torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0],
                            [0.3, 0.3, 0.3, 0.0, 0.0]]),
              torch.full((3, 5), 1e-8),
              torch.full((3, 5), 1.0)):
        clamped = r.clamp(max=1.0 - 1e-6)
        log_space = 1.0 - torch.exp(torch.log1p(-clamped).sum(dim=-1))
        direct = 1.0 - torch.prod(1.0 - clamped, dim=-1)
        assert torch.allclose(log_space, direct, atol=1e-6)
        assert torch.isfinite(log_space).all()

    # and the head computes that, not the peak
    torch.manual_seed(23)
    h = torch.randn(32, 5, 6)
    step, cum = head.forward_trajectory(h)
    ref = 1.0 - torch.prod(1.0 - step.clamp(max=1.0 - 1e-6), dim=-1)
    assert torch.allclose(cum, ref, atol=1e-6)
    assert not torch.allclose(cum, InfiltrationRiskHead.peak_risk(step))


def test_bce_of_the_best_constant_equals_the_entropy_of_the_mean():
    """The baseline the trainer prints for the risk term. BCE is LINEAR in the
    target, so against soft targets the optimal constant is still mean(t) and
    its loss is still H(mean(t)) -- worth pinning, because that identity is
    what makes `bce_constant` a fair comparison for a non-binary target.
    """
    rng = np.random.default_rng(3)
    # the real shape of risk_score: exactly 0.0 for benign, 0.50-0.96 by tactic
    t = np.where(rng.random(20000) < 0.88, 0.0, rng.choice([0.60, 0.75, 0.82, 0.96], 20000))
    tt = torch.from_numpy(t).float()
    rbar = float(tt.mean())
    analytic = -(rbar * math.log(rbar) + (1 - rbar) * math.log(1 - rbar))

    empirical = float(torch.nn.functional.binary_cross_entropy(
        torch.full_like(tt, rbar), tt))
    assert empirical == pytest.approx(analytic, abs=1e-5)

    # no other constant beats it
    for c in (0.01, 0.05, 0.2, 0.5, 0.9):
        worse = float(torch.nn.functional.binary_cross_entropy(
            torch.full_like(tt, c), tt))
        assert worse > analytic


# ---------------------------------------------------------------------------
# Defect 5 -- the dataset called a variable time gap a fixed 2 seconds
# ---------------------------------------------------------------------------

def test_dataset_emits_the_real_elapsed_time_of_each_step():
    """LazyHostRolloutDataset indexes by POSITION in a host's trajectory, so
    `snaps[i:i+K]` is "the next 5 snapshots", which the contract reads as
    "+10 s". Measured on wed_29_csv.csv: median gap 7 windows, mean 12.06,
    p90 29, only 15.9% of consecutive snapshots adjacent. The sample now
    carries what the spacing actually was.
    """
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
    from data_unification.trajectory_store import TrajectoryStoreBuilder

    rng = np.random.default_rng(5)
    b = TrajectoryStoreBuilder(spill_dir=None)
    # one host, deliberately gappy: windows 0, 3, 6, 9, ... (gap 3, not 1)
    windows = list(range(0, 90, 3))
    for w in windows:
        b.append(host_ip="10.0.0.1", host_id=1, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=rng.random(12).astype(np.float32),
                 temporal_attrs=rng.random(15).astype(np.float32),
                 is_attack=False, coarse_category="Benign",
                 technique_ids=[], risk_score=0.0)
    store = b.finalize()

    ds = LazyHostRolloutDataset(store, T=15, K=5)
    item = ds[0]
    assert "t_history" in item and "t_future" in item
    t_hist, t_fut = item["t_history"], item["t_future"]
    assert t_hist.shape == (15,) and t_fut.shape == (5,)

    # origin is the last observed step; history <= 0, future > 0
    assert float(t_hist[-1]) == pytest.approx(0.0)
    assert torch.all(t_hist[:-1] < 0)
    assert torch.all(t_fut > 0)

    # gap 3 windows at 2 s each = 6 s per step, NOT the 2 s a uniform grid
    # would assume
    assert float(t_hist[-1] - t_hist[-2]) == pytest.approx(6.0)
    assert float(t_fut[0]) == pytest.approx(6.0)
    assert float(t_fut[-1]) == pytest.approx(30.0)

    # and the shapes/dtypes the collate path needs
    assert t_hist.dtype == torch.float32 and t_fut.dtype == torch.float32

    # opting out restores exactly the old three keys
    lean = LazyHostRolloutDataset(store, T=15, K=5, emit_times=False)[0]
    assert set(lean) == {"h_history", "h_future", "risk_future"}


def test_dataset_times_feed_straight_into_the_rollout():
    """The keys the dataset emits must be the tensors rollout() accepts."""
    from torch.utils.data import DataLoader

    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
    from data_unification.trajectory_store import TrajectoryStoreBuilder

    rng = np.random.default_rng(9)
    b = TrajectoryStoreBuilder(spill_dir=None)
    w = 0
    for step in range(60):                       # irregular spacing
        w += int(rng.integers(1, 9))
        b.append(host_ip="10.0.0.2", host_id=2, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=rng.random(12).astype(np.float32),
                 temporal_attrs=rng.random(15).astype(np.float32),
                 is_attack=False, coarse_category="Benign",
                 technique_ids=[], risk_score=0.0)
    ds = LazyHostRolloutDataset(b.finalize(), T=15, K=5)
    batch = next(iter(DataLoader(ds, batch_size=8)))

    # Branch B models the full 27-D world state, not the bare 12-D latent.
    torch.manual_seed(0)
    m = HostWorldDynamicsTransformer(
        d_latent=batch["h_history"].shape[-1], d_model=64, n_heads=4, n_layers=3).eval()
    with torch.no_grad():
        with_times = m.rollout(batch["h_history"], K=5,
                               t_history=batch["t_history"],
                               t_future=batch["t_future"])
        assumed_uniform = m.rollout(batch["h_history"], K=5)
    assert with_times.shape == (8, 5, 27)
    assert float((with_times - assumed_uniform).pow(2).mean()) > 1e-6, (
        "the real spacing must change the forecast, or passing it is pointless")


def test_edge_padded_future_steps_do_not_advance_time():
    """When a host's trajectory ends, h_future is edge-padded -- the padded
    entries repeat the last real state, so they must repeat its timestamp too.
    A padded step is not a step further into the future.
    """
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
    from data_unification.trajectory_store import TrajectoryStoreBuilder

    rng = np.random.default_rng(11)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for w in range(18):                          # 18 snapshots, T=15, K=5
        b.append(host_ip="10.0.0.3", host_id=3, window_idx=w,
                 window_start=float(w * 2), window_end=float(w * 2 + 2),
                 embedding=rng.random(12).astype(np.float32),
                 temporal_attrs=rng.random(15).astype(np.float32),
                 is_attack=False, coarse_category="Benign",
                 technique_ids=[], risk_score=0.0)
    ds = LazyHostRolloutDataset(b.finalize(), T=15, K=5)
    last = ds[len(ds) - 1]                       # only 1 real future step left
    h_fut, t_fut = last["h_future"], last["t_future"]
    assert torch.equal(h_fut[0], h_fut[-1]), "expected edge padding"
    assert float(t_fut[0]) == pytest.approx(float(t_fut[-1]))


def test_the_fixed_sinusoid_bank_can_be_switched_off():
    """It measured neutral on the synthetic probe (0.74862 vs 0.74554 for the
    learned cosines alone) and is kept for conditioning at large elapsed
    times, not because it scored better. That makes it a choice, so it has to
    be switchable -- and switching it must not change the parameter set.
    """
    with_bank = ContinuousTimeEncoding(d_time=64, d_model=64)
    without = ContinuousTimeEncoding(d_time=64, d_model=64, fixed_sinusoids=False)
    assert ([n for n, _ in with_bank.named_parameters()]
            == [n for n, _ in without.named_parameters()])
    assert with_bank.state_dict().keys() == without.state_dict().keys()

    t = (torch.arange(-14, 1, dtype=torch.float32) * 2.0).expand(2, 15)
    without.linear.load_state_dict(with_bank.linear.state_dict())
    a, b = with_bank(t).detach(), without(t).detach()
    assert a.shape == b.shape == (2, 15, 64)
    assert not torch.allclose(a, b)
    # the learned branch alone must still vary along the sequence
    assert float((b - b[:, :1, :]).abs().max()) > 0.1
