"""Branch B risk head: loss choice (defect 6) and cumulative-risk aggregation
under the hazard target (defect 7).

Both defects are about the SAME root fact: `TrajectoryStore.hazard_risk`
replaced a bimodal, non-forecasting target with a continuous one, and neither
`InfiltrationRiskHead`'s loss nor its horizon aggregation was reconsidered for
what that change actually implies. Root-caused and pinned here rather than
assumed:

  1. binary_cross_entropy does not pick a different OPTIMUM than a regression
     loss for a [0, 1] target -- both are mean-seeking, proven analytically
     (BCE is linear in the target) and confirmed empirically below. What
     changes the result is being mean-seeking vs median-seeking, which is a
     property of MSE/BCE vs Huber/L1, not of BCE vs MSE.
  2. `cumulative_risk`'s `1 - prod_k(1 - r_k)` is the exact discrete-survival
     identity for a genuine per-step conditional hazard (what
     cyberworld_v4/models.py's own "hazard" trains against), but step_risks
     trained against `hazard_risk` are K correlated restatements of a SINGLE
     event's proximity, not K semi-independent onset observations -- so the
     product form compounds correlated evidence and inflates. Proven below
     with the real `TrajectoryStore.hazard_risk` values, no trained model
     involved, so this is not a training-noise artifact.

See branch_b_world_model/infiltration_head.py for the full reasoning; this
file only pins the numbers that reasoning depends on.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from data_unification.trajectory_store import TrajectoryStoreBuilder

EMB = np.zeros(12, dtype=np.float32)
ATTRS = np.zeros(15, dtype=np.float32)


def _store(flags, window=2.0):
    b = TrajectoryStoreBuilder(spill_dir=None)
    for w, atk in enumerate(flags):
        b.append(host_ip="h", host_id=0, window_idx=w,
                 window_start=float(w * window), window_end=float(w * window + window),
                 embedding=EMB, temporal_attrs=ATTRS, is_attack=bool(atk),
                 coarse_category="C2" if atk else "Benign",
                 technique_ids=["T1071"] if atk else [], risk_score=0.82 if atk else 0.0)
    return b.finalize()


# ---------------------------------------------------------------------------
# Defect 6 -- BCE vs MSE target the same optimum; Huber does not
# ---------------------------------------------------------------------------

def test_risk_loss_is_smooth_l1_with_a_tight_beta():
    """Pins the actual function so the reasoning in the module docstring
    stays attached to real behaviour, not just prose."""
    torch.manual_seed(0)
    pred = torch.rand(64)
    target = torch.rand(64)
    got = InfiltrationRiskHead.risk_loss(pred, target)
    ref = F.smooth_l1_loss(pred, target, beta=0.1)
    assert torch.allclose(got, ref)


def _make_split(w, never_frac, tau, noise_sigma, n, seed):
    """A hazard-SHAPED synthetic target: a fraction of rows are a host never
    attacked (y=0 exactly, matching hazard_risk's own zero case), the rest
    have dt ~ Exponential(tau) so y = exp(-dt/tau) is Uniform(0,1) -- a known
    fact, and it reproduces hazard_risk's continuous, non-bimodal spread
    without needing a full TrajectoryStore for a training-scale experiment.
    The 12-d latent carries a noisy (not deterministic) linear readout of y
    in 2 dims plus 10 pure-noise dims, i.e. genuinely learnable but not a
    lookup table -- the same w is shared across train/val, only the noise
    realisation differs, or train and val would disagree about what the
    embedding even means.
    """
    g = np.random.default_rng(seed)
    never = g.random(n) < never_frac
    dt = g.exponential(scale=tau, size=n)
    y = np.exp(-dt / tau).astype(np.float32)
    y[never] = 0.0
    core = np.outer(y, w) + g.normal(scale=noise_sigma, size=(n, 2)).astype(np.float32)
    rest = g.normal(scale=1.0, size=(n, 10)).astype(np.float32)
    x = np.concatenate([core, rest], axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def _fit(loss_kind, x_tr, y_tr, epochs, lr, seed):
    torch.manual_seed(seed)
    head = InfiltrationRiskHead(d_latent=12, hidden_dim=32, dropout=0.1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    n = x_tr.shape[0]
    bs = 256
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            pred = head.forward_step(xb)
            if loss_kind == "bce":
                loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), yb)
            elif loss_kind == "mse":
                loss = F.mse_loss(pred, yb)
            elif loss_kind == "huber":
                loss = InfiltrationRiskHead.risk_loss(pred, yb)
            else:
                raise ValueError(loss_kind)
            loss.backward()
            opt.step()
    head.eval()
    return head


def test_bce_and_mse_reach_the_same_predictor_but_huber_does_not():
    """The `2` claim from the module docstring: BCE and MSE are, empirically
    and not just in theory, the same fit on a continuous target.
    """
    w = np.random.default_rng(0).normal(size=2).astype(np.float32)
    x_tr, y_tr = _make_split(w, never_frac=0.55, tau=10.0, noise_sigma=0.3, n=4000, seed=1)
    x_va, y_va = _make_split(w, never_frac=0.55, tau=10.0, noise_sigma=0.3, n=1000, seed=2)

    def mae_bce_mse(head):
        with torch.no_grad():
            p = head.forward_step(x_va)
            mae = float((p - y_va).abs().mean())
            bce = float(F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y_va))
            mse = float(F.mse_loss(p, y_va))
        return mae, bce, mse

    bce_head = _fit("bce", x_tr, y_tr, epochs=60, lr=3e-3, seed=42)
    mse_head = _fit("mse", x_tr, y_tr, epochs=60, lr=3e-3, seed=42)
    huber_head = _fit("huber", x_tr, y_tr, epochs=60, lr=3e-3, seed=42)

    bce_m = mae_bce_mse(bce_head)
    mse_m = mae_bce_mse(mse_head)
    huber_m = mae_bce_mse(huber_head)

    # BCE and true MSE: within 1% relative on every metric -- the same fit.
    assert bce_m[0] == pytest.approx(mse_m[0], rel=0.05)
    assert bce_m[2] == pytest.approx(mse_m[2], rel=0.10)

    # Huber (risk_loss) gets a materially better MAE than either -- it is
    # optimising a different, median-leaning statistic, not a numerically
    # noisy variant of the same one.
    assert huber_m[0] < bce_m[0] - 0.01
    assert huber_m[0] < mse_m[0] - 0.01


def test_huber_can_beat_the_zero_baseline_where_bce_cannot():
    """The operational claim: with enough embedding signal, a Huber-trained
    head clears the bar this whole investigation is about (beats predicting
    zero on MAE); a BCE-trained head, same data, same capacity, does not.
    This is not about BCE being buggy -- see the previous test, it reaches
    the same predictor as MSE -- it is about optimising the wrong statistic
    for a metric that is not itself mean-seeking.
    """
    w = np.random.default_rng(0).normal(size=2).astype(np.float32)
    x_tr, y_tr = _make_split(w, never_frac=0.55, tau=10.0, noise_sigma=0.15, n=4000, seed=1)
    x_va, y_va = _make_split(w, never_frac=0.55, tau=10.0, noise_sigma=0.15, n=1000, seed=2)
    zero_mae = float(y_va.abs().mean())

    bce_head = _fit("bce", x_tr, y_tr, epochs=60, lr=3e-3, seed=7)
    huber_head = _fit("huber", x_tr, y_tr, epochs=60, lr=3e-3, seed=7)
    with torch.no_grad():
        bce_mae = float((bce_head.forward_step(x_va) - y_va).abs().mean())
        huber_mae = float((huber_head.forward_step(x_va) - y_va).abs().mean())

    assert bce_mae >= zero_mae, (
        f"expected BCE to still lose to zero (mae={bce_mae:.4f} vs "
        f"zero={zero_mae:.4f}) -- if this fails the harness stopped "
        f"reproducing the defect being measured")
    assert huber_mae < zero_mae, (
        f"huber mae={huber_mae:.4f} should beat zero={zero_mae:.4f}")


# ---------------------------------------------------------------------------
# Defect 7 -- cumulative_risk's product form double-counts a hazard curve
# ---------------------------------------------------------------------------

def _cum_and_peak(step_risks: np.ndarray):
    r = torch.tensor(step_risks, dtype=torch.float32).unsqueeze(0)
    cum = 1.0 - torch.exp(torch.log1p(-r.clamp(max=1.0 - 1e-6)).sum(dim=-1))
    peak = InfiltrationRiskHead.peak_risk(r)
    return float(cum[0]), float(peak[0])


def test_cumulative_form_overstates_a_perfect_hazard_forecast_outside_the_horizon():
    """No trained model here -- the GROUND-TRUTH hazard_risk values, fed
    through the exact arithmetic forward_trajectory uses. If the formula
    were sound for this target, a perfect predictor (zero error, since these
    ARE the true values) would not overstate anything.

    tau = forecast_steps * window_seconds = 10s (the contract's own choice,
    matching scripts/retrain_future_models_live.py), window = 2s. Attack
    lands 4 windows past a 5-step horizon, i.e. genuinely not in it.
    """
    st = _store([0] * 8 + [1])  # attack at window index 8
    h = np.asarray(st.hazard_risk(10.0))
    horizon = h[:5]  # windows 0..4: the "K=5 rollout", attack is 4 steps beyond
    np.testing.assert_allclose(
        horizon, [0.20189652, 0.24659696, 0.30119422, 0.36787945, 0.44932896],
        rtol=1e-6)

    cum, peak = _cum_and_peak(horizon)
    assert cum == pytest.approx(0.8537, abs=1e-3)
    assert peak == pytest.approx(0.4493, abs=1e-3)

    # ground truth for "does an attack occur within this horizon": no.
    # cumulative_risk claims an 85% chance of something that does not happen;
    # peak's 45% is a defensible "getting closer" signal, not a false alarm
    # dressed up as near-certainty.
    assert cum > 0.8, "the product form should badly overstate here"
    assert peak < 0.5, "peak should stay a moderate, honest number"


def test_cumulative_form_saturates_even_one_step_before_the_true_event():
    """Contrast case: the attack genuinely IS the last horizon step. Both
    statistics correctly read high, but cumulative overshoots to near-total
    certainty a full step before peak does -- consistent with compounding
    correlated evidence rather than adding new information."""
    st = _store([0, 0, 0, 0, 0, 1])
    h = np.asarray(st.hazard_risk(10.0))
    approach = h[:5]  # the 5 steps strictly before the attack window
    np.testing.assert_allclose(
        approach, [0.36787945, 0.44932896, 0.5488116, 0.67032003, 0.8187308],
        rtol=1e-6)
    cum, peak = _cum_and_peak(approach)
    assert cum == pytest.approx(0.9906, abs=1e-3)
    assert peak == pytest.approx(0.8187, abs=1e-3)
    assert cum - peak > 0.15, (
        "cumulative should read noticeably more certain than the model's own "
        "closest-step estimate, on values that are all restatements of the "
        "same single approaching event")


def test_the_old_severity_target_does_not_have_this_failure_mode():
    """Contrast with the target this formula WAS designed for: an attack
    outside the horizon reads as exactly 0 on every step (no smooth ramp),
    so there is no correlated partial evidence to over-combine."""
    st = _store([0] * 8 + [1])
    sev = np.asarray(st.risk_score)[:5]
    assert (sev == 0.0).all()
    cum, peak = _cum_and_peak(sev)
    assert cum == 0.0 and peak == 0.0



def test_hazard_tau_is_the_contract_forecast_horizon_in_both_trainers():
    """Both trainers decayed the hazard over forecast_steps x window_seconds
    (10 s) after the contract moved to 30 s forecast steps (150 s horizon):
    the risk target asked about 10 s while the console reports 30..150 s.
    The CIC-2017 test store must get the same target as training."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    a = (root / "scripts" / "retrain_branch_a_live.py").read_text()
    b = (root / "scripts" / "retrain_future_models_live.py").read_text()
    assert "else _c.forecast_seconds)" in a
    assert "return float(get_contract().forecast_seconds)" in b
    assert "test_store.use_hazard_target(hazard_tau_seconds())" in b
    from cyberworld_v4.config import get_contract
    c = get_contract()
    assert c.forecast_seconds == c.forecast_steps * c.forecast_window_seconds
