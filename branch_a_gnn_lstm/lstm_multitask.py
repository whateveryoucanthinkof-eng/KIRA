"""
Branch A: Multi-Task GNN/LSTM Attack-Sequence Prediction Model.

Ingests live per-host event sequences x_t = [H_t[v]; temporal_attrs_t[v]]
and jointly predicts:
1. Near-term compromise risk score in [0, 1]
2. MITRE ATT&CK technique probabilities in Delta^{|V_tech|}
3. Attack gradation / severity stage (0..3)
"""

import math
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from branch_a_gnn_lstm.attention import TemporalSelfAttention
from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB


class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for Extreme Class Imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy well-classified negative/benign examples (p_t -> 1)
    and concentrates gradient updates on rare/hard attack classes.

    ## alpha is NOT a registered buffer, deliberately

    `self.tech_focal_loss` is a submodule of `MultiTaskLSTM`, so anything
    registered here lands in the model's `state_dict`. Serving builds the model
    with `MultiTaskLSTM(input_dim=27, hidden_dim=64)` -- i.e. alpha=None -- and
    then calls `load_state_dict(..., strict=True)`. A checkpoint carrying
    `tech_focal_loss.alpha` would be an *unexpected key* there and would break
    serving outright. alpha is a training-time quantity; it is recorded in the
    checkpoint dict (`ckpt["focal_alpha"]`) for the audit trail instead.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.04,
    ):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    @staticmethod
    def inverse_frequency_alpha(
        labels,
        num_classes: int,
        *,
        power: float = 0.5,
        clip: Tuple[float, float] = (0.2, 5.0),
    ) -> torch.Tensor:
        """Damped, clipped, geometric-mean-centred inverse-frequency weights.

        This is the same construction as `bita.train.FocalLoss` (the TGNE
        encoder's loss), reproduced here because Branch A must not import a
        sibling branch's training module to compute a weight vector.
        `tests/test_branch_a_focal_loss.py` pins the two to agree numerically,
        so they cannot drift apart silently.

        The form is not arbitrary; it is the third attempt, and the first two
        each failed on this corpus in opposite directions:

        1. Plain 1/frequency -- a class absent from a split gets an
           astronomical raw weight, and normalising by it collapses every
           present class to ~0. Measured {Benign: 0.0, C2: 0.0, Impact: 2.0,
           ...}: the majority classes contributed no loss at all.
        2. Arithmetic-mean normalisation -- one ultra-rare class sets the
           scale and crushes everyone else onto the clip floor. Measured
           {Benign: 0.2, C2: 0.2, Impact: 0.2, InitialAccess: 0.2,
           Recon: 4.911}: four of five classes weighted IDENTICALLY, i.e. no
           balancing at all among the four carrying the data.

        These are multiplicative weights, so the geometric mean is the right
        centre: it centres them in log space, where a multiplicative
        correction belongs, so one extreme class shifts the others by a
        bounded factor instead of collapsing them.

        A class absent from `labels` keeps weight exactly 1.0 and is excluded
        from the normalisation. That exact 1.0 is a diagnostic, not a default:
        it is how a broken split was found.
        """
        counts = np.bincount(
            np.asarray(labels, dtype=np.int64), minlength=num_classes
        )
        return MultiClassFocalLoss.alpha_from_counts(
            counts, num_classes, power=power, clip=clip)

    @staticmethod
    def alpha_from_counts(
        counts,
        num_classes: int,
        *,
        power: float = 0.5,
        clip: Tuple[float, float] = (0.2, 5.0),
    ) -> torch.Tensor:
        """The weights, from a class histogram rather than from labels.

        The single implementation; `inverse_frequency_alpha` bincounts and
        delegates here. Branch A's training split is 20.7M samples and its
        class histogram is read straight off the trajectory store's columns,
        so materialising 20.7M labels purely to bincount them again would be
        the expensive way to compute a 14-element vector.

        The result depends only on the ratios between counts -- `total/count`
        normalised by its own geometric mean is invariant to a common scale
        factor -- so a histogram carries everything the weights need.
        """
        counts = np.asarray(counts, dtype=np.float64)
        if counts.size < num_classes:
            counts = np.pad(counts, (0, num_classes - counts.size))
        w = np.ones(num_classes, dtype=np.float64)
        present = counts[:num_classes] > 0
        if present.any():
            c = counts[:num_classes][present]
            inv = (c.sum() / c) ** power
            w[present] = inv / float(np.exp(np.mean(np.log(inv))))
        return torch.as_tensor(np.clip(w, clip[0], clip[1]), dtype=torch.float)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # The modulating factor must use the TRUE p_t, not exp(-ce_loss).
        #
        # With label smoothing, `ce_loss` is not -log(p_t): PyTorch returns
        #     (1 - eps) * nll  +  eps * mean_j(-log p_j)
        # and the second term diverges exactly when the model becomes
        # confident. So `pt = exp(-ce_loss)` is bounded well away from 1 no
        # matter how well the example is classified, and `(1 - pt)^gamma`
        # -- the whole mechanism focal loss exists for -- stops shrinking.
        #
        # Measured at this head's own settings (14 classes, eps=0.04,
        # gamma=2). Columns are the focal term each definition produces:
        #
        #     true p_t     exp(-ce)   focal from exp(-ce)   focal from p_t
        #       0.9         0.7541          6.05e-02          1.00e-02
        #       0.99        0.7588          5.82e-02          1.00e-04
        #       0.999       0.7027          8.84e-02          1.00e-06
        #       0.99999     0.5928          1.66e-01          1.00e-10
        #       1 - 1e-9    0.4211          3.35e-01          1.00e-18
        #
        # Two things are wrong, and the second is worse than the first.
        # At p_t = 0.999 the modulating factor is ~88,000x too large. And
        # past p_t ~ 0.99 the old factor turns around and *grows*: the more
        # confidently correct the model is, the more weight focal loss gave
        # the example. The mechanism was not merely weakened, it was
        # inverted, because the smoothing term -eps*mean_j(log p_j) grows
        # without bound exactly as the non-target probabilities shrink.
        #
        # Easy Benign windows are 82.5% of the corpus, so this is not a
        # corner case: it is most of the gradient, and it is precisely the
        # mass focal loss was added to remove.
        #
        # Keeping label smoothing in the MAGNITUDE term is correct -- it
        # regularises the target distribution, which is what it is for. Only
        # the modulating factor has to come from the unsmoothed p_t.
        log_probs = F.log_softmax(logits, dim=-1)
        logpt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        focal_term = (1.0 - pt) ** self.gamma

        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        return focal_loss.mean()


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Homoscedastic uncertainty multi-task loss weighting (Kendall & Gal):
    L = exp(-s_risk) * L_risk + s_risk +
        exp(-s_tech) * L_tech + s_tech +
        exp(-s_grad) * L_grad + s_grad
    """

    #: log-variance bounds. exp(3) ~ 20 and exp(-3) ~ 0.05, so one task can
    #: outweigh another by at most ~400x -- wide enough for genuine
    #: differences in task scale, narrow enough to prevent a runaway.
    LOG_VAR_MIN, LOG_VAR_MAX = -3.0, 3.0

    def __init__(self, clamp: bool = True):
        super(MultiTaskUncertaintyLoss, self).__init__()
        self.clamp = clamp
        self.log_var_risk = nn.Parameter(torch.zeros(1))
        self.log_var_tech = nn.Parameter(torch.zeros(1))
        self.log_var_grad = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def project_(self):
        """Clamp the log-variance parameters back into range, in place.

        `forward` calls this before using them, so the values the loss sees are
        always bounded. It is also public because an optimiser step can leave a
        parameter epsilon outside the bound until the next forward, and a
        checkpoint saved in that window records the out-of-range value -- which
        is exactly how the 2026-09-21 checkpoint came to hold -3.0090 and
        -3.0243. A training loop that calls this after `optimizer.step()` saves
        clean values.
        """
        if not self.clamp:
            return
        self.log_var_risk.clamp_(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        self.log_var_tech.clamp_(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        self.log_var_grad.clamp_(self.LOG_VAR_MIN, self.LOG_VAR_MAX)

    def forward(
        self,
        risk_loss: torch.Tensor,
        tech_loss: torch.Tensor,
        grad_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Bound the log-variances.
        #
        # Kendall & Gal's objective is unbounded below. For a task with loss L
        # the optimum is log_var = log(L), worth 1 + log(L) -- so as L -> 0 the
        # total dives to -inf and that task's precision exp(-log_var) explodes,
        # starving the others. Measured with risk_loss = 1e-4 and the other two
        # at realistic values:
        #
        #     step   total    prec_risk   prec_tech
        #        1    2.100         1.0       1.000
        #      300   -6.133      9993.5       0.833
        #
        # Risk would outweigh technique classification by ~12,000x, and the
        # logged loss falls the whole time, so it reads as healthy training.
        #
        # That matters here specifically: risk_score is derived from is_attack
        # (benign exactly 0.0, attack >= 0.20), so it is the EASIEST task and
        # the least informative -- exactly the one that would run away.
        # Project the PARAMETERS back into range, rather than clamping a copy.
        #
        # `x.clamp(lo, hi)` passes no gradient where x is outside [lo, hi]. The
        # 2026-09-21 checkpoint shows what that costs: log_var_risk settled at
        # -3.0090 and log_var_tech at -3.0243, both just past the lower bound,
        # where their gradient is identically zero -- so they were frozen for
        # the rest of training and could never come back even if the balance
        # they imply stopped being right. Two of the three task weights had
        # quietly become constants.
        #
        # Clamping the parameter in place keeps the value bounded AND the
        # gradient live at the boundary (clamp's gradient is inclusive of the
        # endpoints), which is ordinary projected gradient descent. The
        # effective weighting is unchanged; the difference is that a parameter
        # pinned at the bound can still move back off it.
        if self.clamp:
            self.project_()
        lv_risk, lv_tech, lv_grad = self.log_var_risk, self.log_var_tech, self.log_var_grad

        prec_risk = torch.exp(-lv_risk)
        prec_tech = torch.exp(-lv_tech)
        prec_grad = torch.exp(-lv_grad)

        total_loss = (
            prec_risk * risk_loss + lv_risk +
            prec_tech * tech_loss + lv_tech +
            prec_grad * grad_loss + lv_grad
        )

        # Detached 0-dim tensors, not floats.
        #
        # These seven `.item()` calls each forced a host-device sync, on every
        # batch of the training loop -- ~1.1M syncs per epoch at 161,439
        # batches. Nothing consumed the dict (the one caller that binds it,
        # train_branch_a.py:306, never reads it), so the whole cost bought
        # nothing, and it cancelled out the loop-level sync removed earlier.
        #
        # 0-dim tensors format and float() exactly like scalars, so a caller
        # that wants numbers pays for them only when it asks.
        metrics = {
            "loss_total": total_loss.detach(),
            "loss_risk": risk_loss.detach(),
            "loss_tech": tech_loss.detach(),
            "loss_grad": grad_loss.detach(),
            "weight_risk": prec_risk.detach(),
            "weight_tech": prec_tech.detach(),
            "weight_grad": prec_grad.detach(),
        }
        return total_loss, metrics


class MultiTaskLSTM(nn.Module):
    """
    Branch A: graph embedding + temporal attributes -> LSTM -> three heads.

    Vitulyova, Babenko, Kolesnikova, Kiktev and Abramkina, "A Hybrid Approach
    Using Graph Neural Networks and LSTM for Attack Vector Reconstruction",
    Computers 14(8):301, 2025, Section 3.2.3 and 3.4:

      Eq. 1, 4  H_t = LSTM([H_GNN, X_t])      graph embedding concatenated with
                                             15 temporal attributes (here the
                                             12-D TGNE latent + 15 attributes)
      Table 6   R_t = sigmoid(W_R H_t + b_R)  risk score
                P_t = softmax(W_p H_t + b_p)  technique probabilities
                G_t = sigmoid(W_G H_t + b_G)  probability gradation, a scalar
      Sec. 3.4  L = 0.5 L_risk(BCE) + 0.3 L_tech(CE) + 0.2 L_grad(MSE),
                1 LSTM layer x 256 hidden units, dropout 0.2, Adam lr 1e-3

    PAPER_ARCH is that model. LEGACY_ARCH is the variant earlier checkpoints
    were trained as (2 x 64 LSTM, attention readout, MLP heads, 4-class
    gradation, uncertainty-weighted loss); `from_checkpoint` rebuilds it for
    them.

    Two deviations are kept on purpose and apply to both:
      * the risk target follows `risk_objective` ("bce" = the paper's BCE on
        risk > 0, the default);
      * the technique loss is class-weighted (focal_gamma=0 makes the focal
        loss exactly alpha-weighted cross-entropy). The paper balances classes
        with SMOTE on flat CICIDS2017 records; SMOTE does not apply to
        overlapping host sequences, and on this ~82.5%-Benign corpus plain CE
        collapses the head onto Benign.
    """

    PAPER_ARCH = dict(hidden_dim=256, num_layers=1, dropout=0.2, readout="last",
                      head_type="linear", gradation_mode="scalar",
                      loss_weighting="fixed", focal_gamma=0.0)
    LEGACY_ARCH = dict(hidden_dim=64, num_layers=2, dropout=0.2, readout="attention",
                       head_type="mlp", gradation_mode="classes",
                       loss_weighting="uncertainty", focal_gamma=2.0)
    #: Paper Section 3.4: alpha, beta, gamma for risk, technique, gradation.
    PAPER_LOSS_WEIGHTS = (0.5, 0.3, 0.2)

    def __init__(
        self,
        input_dim: int = 27,  # 12 TGNE-TA embedding + 15 temporal attributes
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_techniques: int = len(TECHNIQUE_VOCAB),
        num_gradations: int = 4,
        dropout: float = 0.2,
        risk_objective: str = "bce",
        focal_gamma: float = 2.0,
        gradation_class_weights: Optional[torch.Tensor] = None,
        readout: str = "attention",
        head_type: str = "mlp",
        gradation_mode: str = "classes",
        loss_weighting: str = "uncertainty",
        loss_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        super(MultiTaskLSTM, self).__init__()
        for name, value, allowed in (
            ("readout", readout, ("attention", "last")),
            ("head_type", head_type, ("mlp", "linear")),
            ("gradation_mode", gradation_mode, ("classes", "scalar")),
            ("loss_weighting", loss_weighting, ("uncertainty", "fixed")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        self.readout = readout
        self.head_type = head_type
        self.gradation_mode = gradation_mode
        self.loss_weighting = loss_weighting
        self.loss_weights = tuple(float(w) for w in loss_weights)
        self._arch = dict(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_techniques=num_techniques, num_gradations=num_gradations,
            dropout=dropout, risk_objective=risk_objective, focal_gamma=focal_gamma,
            readout=readout, head_type=head_type, gradation_mode=gradation_mode,
            loss_weighting=loss_weighting, loss_weights=list(self.loss_weights),
        )
        if risk_objective not in ("bce", "soft_bce", "smooth_l1"):
            raise ValueError(
                f"risk_objective must be 'bce', 'soft_bce' or 'smooth_l1', "
                f"got {risk_objective!r}")
        #: "bce"       -- predict P(next window is an attack window). Calibrated,
        #:                and the metric that judges it (AUC/Brier) matches the
        #:                loss that trains it.
        #: "soft_bce"  -- the same proper scoring rule against a CONTINUOUS
        #:                target in [0, 1]. Cross-entropy between two Bernoullis
        #:                is minimised at p == y for any y in [0, 1], so this
        #:                stays a proper scoring rule without binarising. It
        #:                exists for the hazard target (exp(-dt/tau)), which is
        #:                already a survival probability and which "bce" would
        #:                collapse to "is this host ever attacked".
        #: "smooth_l1" -- the original scalar regression. Kept to reproduce the
        #:                2026-09-21 run; against the severity target it cannot
        #:                beat a constant on MAE.
        self.risk_objective = risk_objective
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_techniques = num_techniques
        self.num_gradations = num_gradations
        # Plain attribute, not a buffer: see MultiClassFocalLoss's docstring --
        # anything registered here enters state_dict and an unexpected key
        # breaks the serving load. Training-time only.
        self.gradation_class_weights = gradation_class_weights

        # Core recurrent backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention over sequence
        self.attention = TemporalSelfAttention(hidden_dim=hidden_dim, attn_dim=hidden_dim // 2)

        # Head 1: Risk score regression in [0, 1]
        grad_out = 1 if gradation_mode == "scalar" else num_gradations
        if head_type == "linear":
            # Paper: each head is one affine map of H_t; dropout 0.2 on H_t.
            self.head_dropout = nn.Dropout(dropout)
            self.risk_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.technique_head = nn.Linear(hidden_dim, num_techniques)
            self.gradation_head = nn.Linear(hidden_dim, grad_out)
        else:
            self.head_dropout = nn.Identity()
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        # Head 2: MITRE ATT&CK technique classification
            self.technique_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_techniques),
            )

        # Head 3: Attack gradation / severity stage (0=Benign, 1=Recon, 2=Infiltration, 3=C2/DDoS)
            self.gradation_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, grad_out),
            )

        self.uncertainty_loss = MultiTaskUncertaintyLoss()
        self.tech_focal_loss = MultiClassFocalLoss(gamma=focal_gamma)

        # Elapsed time between a host's history steps.
        #
        # A trajectory's rows are the host's ACTIVE windows, not clock ticks,
        # and none of the 15 temporal attributes spans windows -- they are all
        # per-window aggregates. So this model saw fifteen feature vectors with
        # no way to tell whether they were 2 s or 2 hours apart. Measured over
        # the full train split: 40.9% of targets lie more than 10 s from their
        # history (64% on CTU-13, a median 736 s). Branch B already solved this
        # by encoding real elapsed time; this is the same fix, so the gap is
        # handled without dropping samples.
        #
        # Time is log-compressed before encoding: gaps run from 2 s to hours,
        # and the encoder's learned term is cos(Linear(t)), which turns
        # hour-scale raw seconds into noise.
        #
        # `time_proj` is ZERO-initialised, so x + time_proj(...) == x exactly
        # until training moves it. A checkpoint written before this change has
        # no time_* keys; load_state_dict backfills them, and since the
        # projection is zero the model behaves identically -- existing
        # checkpoints keep loading and serving unchanged.
        from branch_b_world_model.rollout_encoder_decoder import ContinuousTimeEncoding
        self.time_encoder = ContinuousTimeEncoding(d_time=16, d_model=16,
                                                   min_period=1.0, max_period=32.0)
        self.time_proj = nn.Linear(16, input_dim)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

        # Temperature scaling, done the way temperature scaling is defined.
        #
        # This used to be `nn.Parameter(torch.ones(1))`, trained jointly with
        # everything else by the same Adam step, and applied ONLY inside
        # `compute_loss`. Three things were wrong with that, and they compound:
        #
        # 1. It is not temperature scaling. Guo et al. (2017) fit a single T by
        #    minimising NLL on held-out data with the model FROZEN, precisely
        #    because a network is overconfident on data it was fitted to.
        #    Training T on the training objective moves it toward whatever
        #    sharpness minimises the training loss -- the opposite correction.
        #    The 2026-09-21 checkpoint settled at T = 0.9169, i.e. it learned
        #    to SHARPEN, which is what an overconfident model does when you let
        #    it choose.
        # 2. During training it is redundant. `technique_head`'s final Linear
        #    can absorb any constant 1/T into its own weights, so the parameter
        #    adds no capacity -- only the illusion of calibration.
        # 3. It was applied in the loss and dropped at inference. `forward`
        #    returned raw logits, and the serving path
        #    (correlation/trajectory_assembler.py:163) softmaxes those and
        #    reports `max()` as the operator-facing confidence. So the model
        #    was trained under logits/0.9169 and served under logits: a
        #    train/serve mismatch on exactly the number a human reads.
        #
        # Now: a non-trainable buffer, fixed at 1.0 during training, fitted
        # post-hoc by `fit_temperature()` on held-out data with the model
        # frozen, and applied in `forward` so that every downstream softmax --
        # including serving's, which this repo cannot reach from here -- gets
        # the calibrated logits without any change at the call site.
        #
        # `temperature_fitted` is a separate buffer rather than "T != 1.0"
        # because a legacy checkpoint carries a jointly-trained T that must NOT
        # be applied: it was never fitted to anything. Loading such a
        # checkpoint leaves the flag at 0 (see `_load_from_state_dict`), so
        # serving behaviour for the weights on disk today is unchanged.
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("temperature_fitted", torch.zeros(1))

        # Half-width of the fitted split-conformal interval on the risk head.
        # NaN means "never fitted", and `predict_calibrated_risk` raises rather
        # than inventing one. See that method for why the old hardcoded 0.05
        # was not a 95% interval.
        self.register_buffer("risk_conformal_halfwidth", torch.full((1,), float("nan")))

    _TIME_KEYS = ("time_encoder.", "time_proj.")

    def load_state_dict(self, state_dict, strict: bool = True, **kw):
        """Accept checkpoints written before the time channel existed.

        Only the time_* keys are backfilled, from this module's own freshly
        initialised values. time_proj starts at zero, so a backfilled model is
        exactly the model that was saved. Every other key is still checked
        strictly -- this is not a blanket strict=False.
        """
        sd = dict(state_dict)
        for k, v in self.state_dict().items():
            if k.startswith(self._TIME_KEYS) and k not in sd:
                sd[k] = v
        return super().load_state_dict(sd, strict=strict, **kw)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] optional padding mask
        Returns:
            Dictionary with:
                risk_score: [batch_size, 1]
                technique_logits: [batch_size, num_techniques]
                gradation_logits: [batch_size, num_gradations]
                attention_weights: [batch_size, seq_len]
                context: [batch_size, hidden_dim]
        """
        if t_history is not None:
            # seconds relative to the last observed step (<= 0, last = 0)
            _lt = torch.sign(t_history) * torch.log1p(t_history.abs())
            x = x + self.time_proj(self.time_encoder(_lt.to(x.dtype)))
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        if self.readout == "last":
            # Paper: the heads read H_t, the hidden state at the last step.
            context = lstm_out[:, -1, :]
            attn_weights = torch.zeros(lstm_out.shape[:2], device=lstm_out.device,
                                       dtype=lstm_out.dtype)
            attn_weights[:, -1] = 1.0
        else:
            context, attn_weights = self.attention(lstm_out, mask=mask)

        h = self.head_dropout(context)
        risk_score = self.risk_head(h)
        tech_logits_raw = self.technique_head(h)
        gradation_score = None
        if self.gradation_mode == "scalar":
            # G_t in [0, 1]. The evaluation code scores discrete levels, so the
            # nearest of the num_gradations evenly spaced levels is also
            # exposed as logits (argmax = nearest level).
            gradation_score = torch.sigmoid(self.gradation_head(h)).squeeze(-1)
            levels = torch.linspace(0.0, 1.0, self.num_gradations, device=h.device, dtype=h.dtype)
            grad_logits = -50.0 * (gradation_score.unsqueeze(-1) - levels) ** 2
        else:
            grad_logits = self.gradation_head(h)

        # `technique_logits` is the SERVED quantity, so the fitted temperature
        # is applied here rather than at the call site. The serving path
        # softmaxes this key and reports the max as an operator-facing
        # confidence; it cannot be asked to divide by a temperature it does not
        # know about. `technique_logits_raw` is what the loss must use, because
        # T is fitted post-hoc against a frozen model -- training it through
        # the loss is the defect this replaces.
        #
        # T divides monotonically, so argmax and therefore accuracy, macro F1
        # and the confusion matrix are all unchanged. Only the probabilities
        # move, which is the entire point.
        tech_logits = tech_logits_raw / self.effective_temperature

        return {
            "risk_score": risk_score.squeeze(-1),
            "technique_logits": tech_logits,
            "technique_logits_raw": tech_logits_raw,
            "gradation_logits": grad_logits,
            "gradation_score": gradation_score,
            "attention_weights": attn_weights,
            "context": context,
        }

    # -- construction from a checkpoint ------------------------------------
    def arch_config(self) -> Dict[str, Any]:
        """Constructor kwargs; saved with every checkpoint as ckpt["arch"]."""
        return dict(self._arch)

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any], device="cpu") -> "MultiTaskLSTM":
        """Rebuild the architecture a checkpoint was trained as, then load it."""
        arch = ckpt.get("arch")
        if arch is None:
            sd = ckpt["model_state_dict"]
            arch = dict(cls.LEGACY_ARCH)
            arch["input_dim"] = int(sd["lstm.weight_ih_l0"].shape[1])
            arch["hidden_dim"] = int(sd["lstm.weight_hh_l0"].shape[1])
            tc = ckpt.get("training_contract") or {}
            arch["risk_objective"] = tc.get("risk_objective") or ckpt.get("risk_objective") or "bce"
        arch = dict(arch)
        if "loss_weights" in arch:
            arch["loss_weights"] = tuple(arch["loss_weights"])
        model = cls(**arch).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    @property
    def effective_temperature(self) -> torch.Tensor:
        """T when it has been fitted post-hoc, otherwise exactly 1.0.

        An unfitted temperature must be inert. A legacy checkpoint stores
        T = 0.9169 left over from joint training; that number was fitted to
        nothing and applying it would be a silent change to what the currently
        served model outputs.
        """
        if float(self.temperature_fitted) <= 0.0:
            return torch.ones((), device=self.temperature.device,
                              dtype=self.temperature.dtype)
        return self.temperature.clamp(min=1e-3)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Let checkpoints written before these buffers existed load strictly.

        `control_backend/model_adapter.py` calls `load_state_dict` with the
        default `strict=True`, so a buffer added here would make every
        checkpoint already on disk unloadable -- i.e. it would take serving
        down. Supplying the constructor default for any absent new buffer
        keeps old checkpoints loading and, importantly, keeps their behaviour
        identical: `temperature_fitted` defaults to 0, so the jointly-trained
        temperature they carry stays inert.
        """
        for name in ("temperature", "temperature_fitted", "risk_conformal_halfwidth"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).clone()
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes multi-task loss with calibrated focal loss and uncertainty weighting."""
        risk_loss, tech_loss, grad_loss = self.task_losses(predictions, batch)

        if self.loss_weighting == "fixed":
            a, b, c = self.loss_weights
            total_loss = a * risk_loss + b * tech_loss + c * grad_loss
            metrics = {
                "loss_total": total_loss.detach(),
                "loss_risk": risk_loss.detach(),
                "loss_tech": tech_loss.detach(),
                "loss_grad": grad_loss.detach(),
                "weight_risk": torch.tensor(a),
                "weight_tech": torch.tensor(b),
                "weight_grad": torch.tensor(c),
            }
        else:
            total_loss, metrics = self.uncertainty_loss(risk_loss, tech_loss, grad_loss)
        # Reported so a run can see whether a temperature has been fitted yet;
        # it is 1.0 (inert) for the whole of training by construction.
        metrics["temperature"] = self.effective_temperature.detach()
        return total_loss, metrics

    #: Parameters owned by one task. Everything else -- the LSTM and, in the
    #: legacy architecture, the attention readout -- is the shared trunk.
    TASK_PARAM_PREFIXES = ("risk_head.", "technique_head.", "gradation_head.", "uncertainty_loss.")

    def task_gradient_conflict(
        self,
        x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        t_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Do the three tasks pull the shared LSTM in opposing directions?

        Pass the same `t_history` the training step uses, so the gradients are
        those of the model actually being trained (time channel included).

        Negative transfer between tasks is the failure PCGrad and GradNorm
        exist for. Whether it happens here is an empirical question, and this
        answers it directly: the cosine between each pair of per-task gradients
        on the shared parameters. cos < 0 means one task's step undoes the
        other's; persistently negative values across epochs are the evidence
        that would justify gradient surgery. Near zero or positive means it
        would buy nothing.

        `weighted_grad_norm` is each task's gradient scaled by the weight the
        objective currently gives it (fixed 0.5/0.3/0.2, or exp(-log_var)), so
        a task that dominates the shared update is visible even when the
        directions agree.

        Uses torch.autograd.grad, so no `.grad` is written and an optimizer
        step is unaffected. One extra forward and three backwards: call it on
        one batch per epoch, not every batch.
        """
        shared = [p for n, p in self.named_parameters()
                  if p.requires_grad and not n.startswith(self.TASK_PARAM_PREFIXES)]
        # cuDNN refuses an RNN backward in eval mode; this is a measurement,
        # not the training step, so the slower kernel is fine.
        with torch.backends.cudnn.flags(enabled=False):
            losses = dict(zip(("risk", "tech", "grad"),
                              self.task_losses(self(x, t_history=t_history), batch)))
            flat = {}
            for name, loss in losses.items():
                g = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
                flat[name] = torch.cat([
                    (gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                    for gi, p in zip(g, shared)]).double()

        if self.loss_weighting == "fixed":
            weights = dict(zip(("risk", "tech", "grad"), map(float, self.loss_weights)))
        else:
            u = self.uncertainty_loss
            weights = {k: float(torch.exp(-lv.detach()).reshape(()))
                       for k, lv in (("risk", u.log_var_risk), ("tech", u.log_var_tech),
                                     ("grad", u.log_var_grad))}
        norms = {k: float(v.norm()) for k, v in flat.items()}
        cosine = {}
        for a, b in (("risk", "tech"), ("risk", "grad"), ("tech", "grad")):
            den = norms[a] * norms[b]
            cosine[f"{a}_vs_{b}"] = float(flat[a] @ flat[b]) / den if den > 0 else float("nan")
        weighted = {k: norms[k] * weights[k] for k in norms}
        finite = [c for c in cosine.values() if c == c]
        return {
            "cosine": cosine,
            "min_cosine": min(finite) if finite else float("nan"),
            "grad_norm": norms,
            "weighted_grad_norm": weighted,
            "dominant_task": max(weighted, key=weighted.get),
            "n_shared_params": int(sum(p.numel() for p in shared)),
        }

    def task_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three unweighted task losses: risk, technique, gradation."""
        # Risk loss.
        #
        # `risk_score` is bimodal by construction: exactly 0.0 for a benign
        # window, and base_sev + small terms (0.50..0.96 by tactic) for an
        # attack one. On validation 82.5% of targets are exactly 0.
        #
        # Regressing that with smooth L1 is the wrong formulation twice over.
        # smooth L1 is mean-seeking while MAE -- the metric we report -- is
        # median-seeking, and for a target that is 0 in 82.5% of cases the
        # median is 0. So a smooth-L1-trained head is *guaranteed* to score
        # worse on MAE than the constant 0, however well it fits. Measured:
        # the smooth-L1-optimal constant is 0.1400, whose MAE is 0.2279, and
        # the trained head scored 0.2268 -- it had learned the unconditional
        # mean and nothing else.
        #
        # "bce" reformulates the head as what it can actually be held to: a
        # calibrated probability that the next window is an attack window.
        # The magnitude it used to regress is nearly a function of the
        # category, which the category head already predicts, so nothing is
        # lost by separating them. Output stays in [0, 1], so serving is
        # unchanged.
        #
        # "soft_bce" keeps the same proper scoring rule but does NOT binarise.
        # Against the hazard target exp(-dt/tau) the binarisation is actively
        # destructive: `risk > 0` there means "this host is attacked at some
        # later point in this split", which is nearly constant and carries
        # none of the timing the hazard target was built to express. BCE with
        # a real-valued y in [0, 1] is the cross-entropy between two
        # Bernoullis; it is minimised at p == y, so it stays proper and the
        # sigmoid head stays calibrated against a continuous target.
        if self.risk_objective == "bce":
            risk_target = (batch["risk"] > 0).to(predictions["risk_score"].dtype)
            risk_loss = F.binary_cross_entropy(
                predictions["risk_score"].clamp(1e-6, 1 - 1e-6), risk_target)
        elif self.risk_objective == "soft_bce":
            risk_target = batch["risk"].to(predictions["risk_score"].dtype).clamp(0.0, 1.0)
            risk_loss = F.binary_cross_entropy(
                predictions["risk_score"].clamp(1e-6, 1 - 1e-6), risk_target)
        else:
            risk_loss = F.smooth_l1_loss(predictions["risk_score"], batch["risk"])

        # Focal loss on the RAW logits.
        #
        # Temperature scaling is a post-hoc correction fitted against a frozen
        # model (see __init__). Dividing by it inside the training objective is
        # not calibration -- it is an extra, redundant scale parameter on the
        # head, and gradient descent moves it toward the sharpness that
        # minimises the TRAINING loss, which is the direction that makes
        # miscalibration worse. `technique_logits_raw` is preferred so that
        # training is unaffected by a temperature fitted later; the fallback
        # keeps callers that build a predictions dict by hand working.
        tech_logits = predictions.get("technique_logits_raw")
        if tech_logits is None:
            tech_logits = predictions["technique_logits"]
        tech_loss = self.tech_focal_loss(tech_logits, batch["technique"])

        # Gradation cross entropy loss.
        #
        # `gradation_class_weights` is None by default. The gradation head had
        # never been evaluated at all before 2026-09-22, so there is no measured
        # evidence yet that it needs re-weighting -- and weighting a head before
        # measuring it is guessing. `--gradation-class-weights` turns it on once
        # a run's per-class table shows it collapsing onto level 0.
        _gw = self.gradation_class_weights
        if _gw is not None and _gw.device != predictions["gradation_logits"].device:
            _gw = _gw.to(predictions["gradation_logits"].device)
            self.gradation_class_weights = _gw
        if self.gradation_mode == "scalar":
            # Paper: MSE between G_t and the target g_t in [0, 1]; the target
            # here is the severity level scaled onto [0, 1].
            g_target = batch["gradation"].to(predictions["gradation_score"].dtype) / max(1, self.num_gradations - 1)
            grad_loss = F.mse_loss(predictions["gradation_score"], g_target)
        else:
            grad_loss = F.cross_entropy(
                predictions["gradation_logits"], batch["gradation"], weight=_gw)
        return risk_loss, tech_loss, grad_loss

    # ---------------------------------------------------------------- #
    # Post-hoc calibration. Fitted after training, on held-out data,    #
    # with the model frozen -- which is what makes it calibration.      #
    # ---------------------------------------------------------------- #

    #: Same grid and boundary convention as cyberworld_v4.metrics.calibration
    #: .TemperatureScaler, so the two cannot disagree about what "T hit the
    #: boundary" means. A 1-D grid is exact enough on a near-flat objective
    #: and, unlike an unconstrained optimiser, cannot diverge.
    _TEMP_GRID = (0.05, 10.0, 96, 181)

    @staticmethod
    def top_label_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        """Expected calibration error of the predicted class's confidence.

        The standard multi-class ECE (Guo et al.): bin samples by
        max_j p_j, and compare each bin's mean confidence with the fraction
        of that bin the model got right. Reported before and after a fit,
        because "we scaled the logits" is not evidence that anything improved.
        """
        conf, pred = probs.max(dim=-1)
        correct = (pred == labels).to(conf.dtype)
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device, dtype=conf.dtype)
        ece = torch.zeros((), device=conf.device, dtype=torch.float64)
        n = max(int(conf.numel()), 1)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            k = int(sel.sum())
            if k == 0:
                continue
            ece += (k / n) * (correct[sel].mean() - conf[sel].mean()).abs().double()
        return float(ece)

    @torch.no_grad()
    def fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> Dict[str, object]:
        """Fit T by minimising NLL on held-out logits, model frozen.

        `logits` must be the RAW technique logits (`technique_logits_raw`)
        collected in a no-grad pass over a split the model's *parameters* were
        not fitted on. This method changes no weight; it sets one scalar.

        Returns the evidence, not just the number: NLL and top-label ECE
        before and after, whether T landed on a grid boundary, and how many
        samples it was fitted on. A temperature is only worth the word
        "calibrated" if the after-numbers are better than the before-numbers,
        and this is what lets a caller check that instead of assuming it.
        """
        lo, hi, n_lo, n_hi = self._TEMP_GRID
        grid = torch.cat([
            torch.linspace(lo, 1.0, n_lo)[:-1],
            torch.linspace(1.0, hi, n_hi),
        ]).to(logits.device)

        logits = logits.detach().float()
        labels = labels.detach().long()
        n = int(labels.numel())
        if n == 0 or int(labels.unique().numel()) < 2:
            return {
                "fitted": False,
                "reason": (f"calibration split has {n} samples across "
                           f"{int(labels.unique().numel()) if n else 0} classes; "
                           f"a temperature fitted on one class is meaningless"),
                "temperature": 1.0,
            }

        before_nll = float(F.cross_entropy(logits, labels))
        before_ece = self.top_label_ece(logits.softmax(-1), labels, n_bins)

        best_t, best_nll = 1.0, float("inf")
        for t in grid.tolist():
            nll = float(F.cross_entropy(logits / t, labels))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)

        at_boundary = bool(best_t <= lo * 1.001 or best_t >= hi * 0.999)
        after_ece = self.top_label_ece((logits / best_t).softmax(-1), labels, n_bins)

        # A temperature on the end of the grid is not a fit, it is the
        # objective asking to go further and being stopped. Recording it as
        # fitted would put an arbitrary boundary value into serving.
        if at_boundary:
            return {
                "fitted": False,
                "reason": (f"temperature hit the search boundary ({best_t:.4g}); "
                           f"the calibration split is unrepresentative of what "
                           f"the model was trained on"),
                "temperature": best_t, "at_grid_boundary": True,
                "n_calibration": n,
                "nll_before": before_nll, "nll_after": best_nll,
                "ece_before": before_ece, "ece_after": after_ece,
            }

        self.temperature.fill_(best_t)
        self.temperature_fitted.fill_(1.0)
        return {
            "fitted": True,
            "temperature": best_t,
            "at_grid_boundary": False,
            "n_calibration": n,
            "nll_before": before_nll, "nll_after": best_nll,
            "ece_before": before_ece, "ece_after": after_ece,
            "ece_improvement": before_ece - after_ece,
            "split": "held-out, model frozen",
        }

    @torch.no_grad()
    def fit_risk_conformal(
        self,
        risk_pred: torch.Tensor,
        risk_true: torch.Tensor,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Split-conformal half-width for the risk head, from held-out residuals.

        Uses the finite-sample-corrected quantile ceil((n+1)(1-alpha))/n, the
        same one as `cyberworld_v4.conformal.conformal_quantile`; the plain
        empirical quantile makes the coverage guarantee approximate rather
        than exact.

        ## What this will honestly return, and why that is the point

        Under `--risk-objective bce` the target is Bernoulli: `risk_true > 0`
        is 1 for an attack window and 0 otherwise, and the residual is
        |p - y|. For a well-calibrated head on a ~17.5% base rate, ~17.5% of
        residuals are near 1 - p, so the 95% quantile is large -- the interval
        will be wide, possibly close to vacuous. That is not a defect in this
        function; it is what a 95% *prediction interval on a coin flip* means.
        The previous hardcoded +/-0.05 was not a narrower version of this
        answer, it was a different and false one.
        """
        s = (risk_pred.detach().float().reshape(-1)
             - risk_true.detach().float().reshape(-1)).abs()
        n = int(s.numel())
        if n == 0:
            return {"fitted": False, "reason": "no calibration residuals", "n": 0}
        k = int(math.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            return {
                "fitted": False,
                "n": n,
                "reason": (f"{n} calibration points cannot support "
                           f"{(1 - alpha) * 100:.1f}% coverage; need at least "
                           f"{int(math.ceil(1 / alpha)) - 1}"),
            }
        q = float(torch.sort(s).values[k - 1])
        covered = float(((s <= q).to(torch.float64)).mean())
        self.risk_conformal_halfwidth.fill_(q)
        return {
            "fitted": True,
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "half_width": q,
            "empirical_coverage_on_calibration": covered,
            "n": n,
        }

    @torch.no_grad()
    def set_risk_conformal_halfwidth(self, half_width: float) -> None:
        """Install a half-width computed outside the model.

        `fit_risk_conformal` needs every residual in memory. The training
        script instead accumulates them into a fixed histogram during the
        validation pass it is already running, which is exact to the bin
        width and costs one bincount per batch instead of a second pass over
        1.02M windows. Both routes end here, so there is one place that
        decides what "fitted" means.
        """
        if not (half_width == half_width) or half_width < 0:
            raise ValueError(f"conformal half-width must be finite and >= 0, "
                             f"got {half_width!r}")
        self.risk_conformal_halfwidth.fill_(float(half_width))

    def predict_calibrated_risk(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        conformal_error: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Risk with a conformal interval -- or a loud failure.

        ## What was here, and why it had to go

        The previous body was three magic numbers presented as calibration:

            calibrated_risk = where(raw < 0.35, raw * 0.5, raw)
            lower, upper    = calibrated +/- 0.05        # "95% coverage"

        1. `raw * 0.5` below 0.35. Nothing fitted this. Under the bce
           objective `raw_risk` IS a probability, and halving a probability
           does not suppress noise -- it destroys the calibration the head was
           trained to have and makes every low-risk host report a number that
           is not the probability of anything. Suppression is a *decision*,
           and a decision belongs at a fitted operating point (see
           `ckpt["operating_point"]` written by
           scripts/retrain_branch_a_live.py), not inside the model's estimate.
        2. `+/- 0.05` described as "guaranteeing 95% coverage". It guarantees
           nothing: it is a constant, fitted on no data. For the interval to
           contain a binary outcome it needs |p - y| <= 0.05, i.e. p >= 0.95
           on an attack window or p <= 0.05 on a benign one. Its real coverage
           is whatever fraction of windows happen to be that confident and
           correct -- on the 2026-09-21 head, nowhere near 95%.
        3. It claimed both to an operator, in a field named
           `risk_lower_bound`/`risk_upper_bound`.

        So: `fit_risk_conformal()` on held-out data, or this raises. A
        fabricated interval is worse than no interval, because a SOAR rule can
        act on it.
        """
        out = self.forward(x)
        raw_risk = out["risk_score"]

        if conformal_error is None:
            hw = float(self.risk_conformal_halfwidth)
            if hw != hw:        # NaN -- never fitted
                raise RuntimeError(
                    "predict_calibrated_risk needs a conformal half-width fitted "
                    "on held-out data. Call fit_risk_conformal(risk_pred, "
                    "risk_true) after training -- or pass conformal_error "
                    "explicitly and take responsibility for the number. It is "
                    "not defaulted, because the previous default (0.05) was "
                    "presented as a 95% interval and was fitted to nothing.")
        else:
            hw = float(conformal_error)

        out["calibrated_risk"] = raw_risk
        out["risk_lower_bound"] = torch.clamp(raw_risk - hw, 0.0, 1.0)
        out["risk_upper_bound"] = torch.clamp(raw_risk + hw, 0.0, 1.0)
        out["conformal_half_width"] = torch.full_like(raw_risk, hw)
        if threshold is not None:
            out["alert"] = raw_risk >= threshold
        return out
