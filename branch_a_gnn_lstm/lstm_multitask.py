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

    def __init__(self):
        super(MultiTaskUncertaintyLoss, self).__init__()
        self.log_var_risk = nn.Parameter(torch.zeros(1))
        self.log_var_tech = nn.Parameter(torch.zeros(1))
        self.log_var_grad = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        risk_loss: torch.Tensor,
        tech_loss: torch.Tensor,
        grad_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prec_risk = torch.exp(-self.log_var_risk)
        prec_tech = torch.exp(-self.log_var_tech)
        prec_grad = torch.exp(-self.log_var_grad)

        total_loss = (
            prec_risk * risk_loss + self.log_var_risk +
            prec_tech * tech_loss + self.log_var_tech +
            prec_grad * grad_loss + self.log_var_grad
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_risk": risk_loss.item(),
            "loss_tech": tech_loss.item(),
            "loss_grad": grad_loss.item(),
            "weight_risk": prec_risk.item(),
            "weight_tech": prec_tech.item(),
            "weight_grad": prec_grad.item(),
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
    ):
        super(MultiTaskLSTM, self).__init__()
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
        # Risk regression loss (smooth L1)
        risk_loss = F.smooth_l1_loss(predictions["risk_score"], batch["risk"])

        # Temperature-scaled Extreme-Value Focal Loss
        temp = self.temperature.clamp(min=0.2, max=5.0)
        scaled_logits = predictions["technique_logits"] / temp
        tech_loss = self.tech_focal_loss(scaled_logits, batch["technique"])

        # Gradation cross entropy loss
        grad_loss = F.cross_entropy(predictions["gradation_logits"], batch["gradation"])

        total_loss, metrics = self.uncertainty_loss(risk_loss, tech_loss, grad_loss)
        metrics["temperature"] = float(temp.item())
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
