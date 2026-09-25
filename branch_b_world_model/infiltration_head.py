"""
Branch B: Infiltration Risk Prediction Head.

Estimates per-step compromise likelihood and cumulative horizon risk
over predicted future host states H_{t+1..t+K}[v].
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiltrationRiskHead(nn.Module):
    """
    MLP head estimating compromise probability from latent host states.
    r_{t+k}[v] = sigmoid(MLP(H_{t+k}[v]))
    R_{cumul}[v] = 1 - prod_{k=1}^K (1 - r_{t+k}[v])
    """

    def __init__(self, d_latent: int = 12, hidden_dim: int = 32, dropout: float = 0.1):
        super(InfiltrationRiskHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self.forward_trajectory(x)[0]
        return self.forward_step(x)

    def forward_step(self, h_state: torch.Tensor) -> torch.Tensor:
        """
        Computes single-step compromise probability.
        Args:
            h_state: [batch_size, d_latent]
        Returns:
            risk: [batch_size] in [0, 1]
        """
        return self.mlp(h_state).squeeze(-1)

    @staticmethod
    def risk_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """The loss to train this head against, for a continuous [0, 1] target.

        ## Why not binary_cross_entropy, now that the target is continuous

        Both trainers (branch_b_world_model/train_branch_b.py and
        scripts/retrain_future_models_live.py) called
        `F.binary_cross_entropy(pred_step_risks, r_fut)` unconditionally,
        including against `TrajectoryStore.hazard_risk` -- a continuous,
        deterministic decay curve, not the parameter of a Bernoulli draw. Two
        separate questions, and they get different answers:

        1. Does BCE find a different OPTIMUM than a regression loss? No.
           For a target t in [0, 1] (soft or hard), BCE(p, t) is LINEAR in t,
           so its minimiser over p is p* = E[t | x] for any x -- exactly the
           same conditional mean that MSE also minimises. Pinned in
           tests/test_branch_b_world_model_defects.py::
           test_bce_of_the_best_constant_equals_the_entropy_of_the_mean for
           the constant case, and measured here for the conditional case: a
           head trained on a synthetic hazard-shaped target with BCE and one
           trained with MSE landed within noise of each other on every
           metric -- MAE 0.2665 vs 0.2665, BCE 0.5299 vs 0.5296, MSE 0.0985
           vs 0.0984 (3 seeds each). Switching BCE -> MSE alone would have
           changed nothing.
        2. Is BCE the right NOISE MODEL for this target? No. BCE is the
           negative log-likelihood of "given x, the outcome is a Bernoulli
           coin with bias t" -- appropriate when t truly is an event
           probability. `hazard_risk = exp(-seconds_to_next_attack / tau)` is
           not that: it is a deterministic transform of a continuous quantity
           (time to the next event), so nothing about it is a coin flip.
           Framing it as cross-entropy imposes a Bernoulli variance
           structure (max at t=0.5, zero at the endpoints) that has nothing
           to do with the actual estimation error.

        What DOES move the needle is which conditional statistic the loss
        targets. BCE and MSE are both mean-seeking; the metric this project
        actually gates on ("beats predicting zero") is MAE, which is
        median-seeking. Measured on the same synthetic harness (a bounded
        hazard-shaped target, 55% exact zeros, a genuinely noisy but
        learnable 12-d latent, 3 seeds, `tests/test_branch_b_risk_loss.py`
        pins the exact numbers): with a stronger embedding signal
        (noise_sigma=0.15), BCE/MSE-trained heads scored MAE 0.244 --
        WORSE than predicting zero (0.230) -- while a smooth-L1 (Huber,
        beta=0.1) head, trained on the identical data with nothing else
        different, scored MAE 0.216, the first of the three to actually beat
        the zero baseline. Huber approximates L1 (median-seeking) near zero
        residual while staying differentiable at zero, unlike raw L1.

        This is `beta=0.1`, not the PyTorch default `beta=1.0`: at
        `beta=1.0`, `smooth_l1_loss` is nearly pure L2 over this target's
        whole [0, 1] range (its quadratic region extends a full unit past
        the domain), which reproduces the BCE/MSE mean-seeking behaviour
        this function exists to avoid. `beta=0.1` keeps the loss quadratic
        (and well-conditioned near zero gradient) only for residuals under
        0.1 -- a tenth of the target's range -- and L1 beyond that.

        Args:
            pred: [*] in [0, 1] (post-sigmoid head output).
            target: [*] in [0, 1], same shape as pred.
            beta: the L2/L1 transition point; see above for why 0.1.
        """
        return F.smooth_l1_loss(pred, target, beta=beta)

    def forward_trajectory(self, h_rollout: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-step ATT&CK Severity-Derived Risk Score S_{t+k} in [0, 1]
        and cumulative horizon risk across K forward steps.

        Args:
            h_rollout: [batch_size, K, d_latent]
        Returns:
            step_risks: [batch_size, K] per-step severity scores S_{t+k} in [0, 1]
            cumulative_risk: [batch_size] accumulated hazard over the horizon,
                1 - prod_k (1 - S_{t+k}). Use peak_risk() for the max.
        """
        B, K, D = h_rollout.shape
        h_flat = h_rollout.reshape(B * K, D)
        step_risks = self.mlp(h_flat).reshape(B, K)

        # Cumulative hazard over the horizon: 1 - prod_k (1 - r_k).
        #
        # This returned max(step_risks) -- the PEAK -- under the name
        # `cumulative_risk`, contradicting this class's own docstring and the
        # v4 metrics module, whose test
        # (tests/test_v4_metrics.py::test_cumulative_onset_uses_one_minus_product_not_max)
        # demonstrates why peak is the wrong statistic here:
        #
        #   [0.50, 0, 0, 0, 0]        peak 0.50, cumulative 0.500
        #   [0.30, 0.30, 0.30, 0, 0]  peak 0.30, cumulative 0.657  <- the real onset
        #
        # Peak ranks the single spike above the sustained threat; the product
        # form ranks them correctly. That matters precisely where this value
        # is used -- correlation/trajectory_assembler.py feeds it to
        # HostAttackTrajectory.cumulative_forecast_risk, which is how hosts
        # are ordered for an analyst. A host under persistent moderate
        # pressure should outrank one with a single noisy spike.
        #
        # CAVEAT, and it is target-dependent, not a fact about this formula in
        # general: 1 - prod_k(1 - r_k) is the exact discrete-survival identity
        # ONLY when r_k is a genuine per-step CONDITIONAL hazard -- "compromise
        # happens exactly at step k, given it has not happened in 1..k-1" (that
        # is what cyberworld_v4/models.py's own "hazard" trains against, via a
        # single onset_step target, and the worked example above is honest
        # under that reading). `step_risks` trained against
        # `TrajectoryStore.hazard_risk` is a DIFFERENT quantity: a single
        # smoothly-decaying proximity-to-the-next-event curve, so
        # step_risks[k] for k=1..K are K highly correlated restatements of the
        # same one fact ("how far away is it"), not K semi-independent
        # observations. Multiplying their complements compounds correlated
        # evidence as if it were independent, and it inflates. Measured with
        # the GROUND-TRUTH hazard values themselves (no model, no noise) via
        # tests/test_branch_b_risk_loss.py, tau = forecast_steps*window_seconds
        # = 10s, window = 2s:
        #
        #   attack lands ON the last horizon step (genuinely imminent):
        #     step_risks   [0.368, 0.449, 0.549, 0.670, 0.819]
        #     cumulative 0.991   peak 0.819          (both correctly high)
        #   attack lands 4 windows PAST a 5-step horizon (truly not "in it"):
        #     step_risks   [0.202, 0.247, 0.301, 0.368, 0.449]
        #     cumulative 0.854   peak 0.449          (ground truth: 0)
        #
        # In the second case a PERFECT hazard predictor -- zero model error --
        # still reports 85% cumulative probability of an event that is not in
        # the horizon at all, because the formula treats five smoothly-related
        # glimpses of one fact as five independent risky windows. peak_risk()
        # (== the last, largest step for a monotone-rising hazard curve) does
        # not have this failure mode, because it does not combine anything.
        #
        # This module does not know which target it was trained against (that
        # lives in the trainer, outside this file's ownership at the time of
        # writing), so the default here is left as the product form -- correct
        # for the severity/onset-style target, which is still --risk-target's
        # default. If/when a checkpoint trained with --risk-target hazard is
        # served, its caller should read peak_risk(step_risks), not this
        # value, for "risk of compromise within the horizon". That includes
        # correlation/trajectory_assembler.py's cumulative_forecast_risk,
        # which is outside this package.
        #
        # Computed in log space for numerical stability across the horizon.
        cumulative_risk = 1.0 - torch.exp(
            torch.log1p(-step_risks.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        )

        return step_risks, cumulative_risk

    @staticmethod
    def peak_risk(step_risks: torch.Tensor) -> torch.Tensor:
        """Largest single-step risk over the horizon.

        Answers "how bad does it get at worst" rather than "how likely is
        compromise at all across the horizon" -- and for a SEVERITY-style
        target (independent-ish per-window onset evidence) those are
        genuinely different questions, so this must not stand in for
        cumulative_risk there: see forward_trajectory's worked example, where
        peak ranks a single spike above sustained pressure.

        That prohibition does NOT carry over to a HAZARD-style target
        (TrajectoryStore.hazard_risk / --risk-target hazard). There,
        step_risks is a single monotone proximity curve, cumulative_risk's
        product form double-counts it (see the measured numbers in
        forward_trajectory), and peak_risk -- which for a monotone-rising
        sequence is just its last, largest element -- is the head's own
        direct, uncompounded estimate of "how close does this host get to an
        attack within the horizon". Use this, not cumulative_risk, for a
        hazard-trained checkpoint.
        """
        return torch.max(step_risks, dim=-1)[0]
