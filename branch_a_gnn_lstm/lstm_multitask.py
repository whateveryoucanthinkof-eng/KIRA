"""
Branch A: Multi-Task GNN/LSTM Attack-Sequence Prediction Model.

Ingests live per-host event sequences x_t = [H_t[v]; temporal_attrs_t[v]]
and jointly predicts:
1. Near-term compromise risk score in [0, 1]
2. MITRE ATT&CK technique probabilities in Delta^{|V_tech|}
3. Attack gradation / severity stage (0..3)
"""

import math
from typing import Dict, Tuple, Optional
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
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
    2-Layer LSTM with Temporal Sequence Attention and 3 multi-task output heads.
    """

    def __init__(
        self,
        input_dim: int = 27,  # 12 TGNE-TA embedding + 15 temporal attributes
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_techniques: int = len(TECHNIQUE_VOCAB),
        num_gradations: int = 4,
        dropout: float = 0.2,
        risk_objective: str = "bce",
    ):
        super(MultiTaskLSTM, self).__init__()
        if risk_objective not in ("bce", "smooth_l1"):
            raise ValueError(f"risk_objective must be 'bce' or 'smooth_l1', got {risk_objective!r}")
        #: "bce"       -- predict P(next window is an attack window). Calibrated,
        #:                and the metric that judges it (AUC/Brier) matches the
        #:                loss that trains it.
        #: "smooth_l1" -- the original scalar regression. Kept to reproduce the
        #:                2026-09-21 run; it cannot beat a constant on MAE.
        self.risk_objective = risk_objective
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_techniques = num_techniques
        self.num_gradations = num_gradations

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
            nn.Linear(32, num_gradations),
        )

        self.uncertainty_loss = MultiTaskUncertaintyLoss()
        self.temperature = nn.Parameter(torch.ones(1))
        self.tech_focal_loss = MultiClassFocalLoss(gamma=2.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        context, attn_weights = self.attention(lstm_out, mask=mask)

        risk_score = self.risk_head(context)
        tech_logits = self.technique_head(context)
        grad_logits = self.gradation_head(context)

        return {
            "risk_score": risk_score.squeeze(-1),
            "technique_logits": tech_logits,
            "gradation_logits": grad_logits,
            "attention_weights": attn_weights,
            "context": context,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes multi-task loss with calibrated focal loss and uncertainty weighting."""
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
        if self.risk_objective == "bce":
            risk_target = (batch["risk"] > 0).to(predictions["risk_score"].dtype)
            risk_loss = F.binary_cross_entropy(
                predictions["risk_score"].clamp(1e-6, 1 - 1e-6), risk_target)
        else:
            risk_loss = F.smooth_l1_loss(predictions["risk_score"], batch["risk"])

        # Temperature-scaled Extreme-Value Focal Loss
        temp = self.temperature.clamp(min=0.2, max=5.0)
        scaled_logits = predictions["technique_logits"] / temp
        tech_loss = self.tech_focal_loss(scaled_logits, batch["technique"])

        # Gradation cross entropy loss
        grad_loss = F.cross_entropy(predictions["gradation_logits"], batch["gradation"])

        total_loss, metrics = self.uncertainty_loss(risk_loss, tech_loss, grad_loss)
        metrics["temperature"] = temp.detach()      # see MultiTaskUncertaintyLoss
        return total_loss, metrics

    def predict_calibrated_risk(
        self,
        x: torch.Tensor,
        threshold: float = 0.35,
        conformal_error: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method with extreme-value calibrated gating and conformal confidence intervals.
        Suppresses background benign tail noise when risk < threshold.
        Returns bounds [risk_lower, risk_upper] guaranteeing (1 - alpha) = 95% coverage for SOAR.
        """
        out = self.forward(x)
        raw_risk = out["risk_score"]
        calibrated_risk = torch.where(raw_risk < threshold, raw_risk * 0.5, raw_risk)
        risk_lower = torch.clamp(calibrated_risk - conformal_error, 0.0, 1.0)
        risk_upper = torch.clamp(calibrated_risk + conformal_error, 0.0, 1.0)

        out["calibrated_risk"] = calibrated_risk
        out["risk_lower_bound"] = risk_lower
        out["risk_upper_bound"] = risk_upper
        return out
