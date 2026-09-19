"""
Comprehensive Loss Functions for the World Dynamics Transformer.

Implements all primary and auxiliary objectives defined in Section 7 of the architecture plan:
1. LatentDynamicsLoss: γ-discounted MSE on predicted future latent states ẑ_{t+k}.
2. StateReconstructionLoss: Auxiliary grounding loss reconstructing interpretable features.
3. AttackStageFocalLoss: Class-imbalanced focal loss over MITRE attack stages.
4. RiskInfiltrationLoss: Per-step and cumulative infiltration hazard BCE.
5. LatentRegularizationLoss: Diversity variance preservation + temporal trajectory smoothness.
6. CombinedWorldModelLoss: Unified multi-task objective container with adaptive weighting.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsLoss(nn.Module):
    """
    Core world-model transition dynamics loss.
    Computes γ-discounted MSE across the forecast horizon K:
        L_dyn = (1/K) * Σ_{k=1}^K γ^{k-1} * ||ẑ_{t+k} - sg(z_{t+k})||^2
    
    Discount factor γ=0.95 prevents long horizons from dominating early training.
    Stop-gradient sg(·) ensures target latent states are treated as fixed ground truth.
    """

    def __init__(self, gamma: float = 0.95):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        z_pred: torch.Tensor,    # [B, K, d_z] or [B, S, d_z]
        z_target: torch.Tensor,  # [B, K, d_z] or [B, S, d_z]
        mask: Optional[torch.Tensor] = None,  # [B, K]
    ) -> torch.Tensor:
        target = z_target.detach()
        # Squared error per element: [B, K, d_z]
        sq_err = F.mse_loss(z_pred, target, reduction="none")  # [B, K, d_z]
        mse_per_step = sq_err.mean(dim=-1)  # [B, K]

        K = z_pred.size(1)
        device = z_pred.device
        # Gamma discount weights: [1, γ, γ^2, ..., γ^{K-1}]
        gammas = torch.tensor([self.gamma ** k for k in range(K)], device=device, dtype=torch.float32)  # [K]
        gammas = gammas.unsqueeze(0)  # [1, K]

        if mask is not None:
            weighted_mse = (mse_per_step * gammas * mask.float()).sum() / (mask.float().sum().clamp(min=1.0) * gammas.mean())
        else:
            weighted_mse = (mse_per_step * gammas).mean()

        return weighted_mse


class LatentRegularizationLoss(nn.Module):
    """
    Prevents latent representation collapse and pathological rollout oscillations.
    - Diversity: penalizes batch variance falling below min_variance threshold.
    - Smoothness: penalizes abnormally abrupt per-step velocity changes in latent space.
    """

    def __init__(self, min_variance: float = 0.05, smoothness_weight: float = 0.01):
        super().__init__()
        self.min_variance = min_variance
        self.smoothness_weight = smoothness_weight

    def forward(self, z_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_pred: [B, K, d_z] or [B, S, d_z]
        """
        # 1. Diversity loss: encourage variance across batch
        batch_var = torch.var(z_pred, dim=0).mean()  # average variance across dimensions
        diversity_loss = F.relu(self.min_variance - batch_var)

        # 2. Smoothness loss: consecutive delta magnitude
        if z_pred.size(1) > 1:
            diffs = z_pred[:, 1:, :] - z_pred[:, :-1, :]
            smoothness_loss = torch.mean(diffs ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=z_pred.device)

        return diversity_loss + self.smoothness_weight * smoothness_loss


class StateReconstructionLoss(nn.Module):
    """
    Grounding loss for the StateDecoder:
    Reconstructs continuous traffic statistics, protocol distributions, and attack fractions.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_dict: output from StateDecoder with keys:
                - 'continuous': [B, K, 5]
                - 'protocol': [B, K, 3] (logits or probs)
                - 'malicious_fraction': [B, K, 1]
            target_stats: [B, K, 14] ground-truth statistical features Φ_t
        """
        target = target_stats.detach()
        loss = 0.0

        if "continuous" in pred_dict:
            # First 5 stats: active_nodes, active_edges, byte_rate, packet_rate, port_entropy
            target_cont = target[:, :, :5]
            loss = loss + self.mse(pred_dict["continuous"], target_cont)

        if "protocol" in pred_dict:
            # Stats 5, 6, 7: tcp_fraction, udp_fraction, other_proto_fraction
            target_proto = target[:, :, 5:8]
            pred_proto = pred_dict["protocol"]
            if pred_proto.shape[-1] == 3:
                # Soft target cross entropy: - sum(p * log_q)
                log_probs = F.log_softmax(pred_proto, dim=-1)
                proto_loss = -torch.sum(target_proto * log_probs, dim=-1).mean()
                loss = loss + proto_loss

        if "malicious_fraction" in pred_dict:
            # Stat 11: malicious_edge_fraction
            target_mal = target[:, :, 11:12].clamp(0.0, 1.0)
            pred_mal = pred_dict["malicious_fraction"].clamp(1e-7, 1.0 - 1e-7)
            loss = loss + F.binary_cross_entropy(pred_mal, target_mal)

        return loss


class AttackStageFocalLoss(nn.Module):
    """
    Focal Loss for multi-class attack stage classification with severe class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] or [B, K, num_classes]
            targets: [B] or [B, K] long class indices (0..4)
        """
        if logits.dim() == 3:
            B, K, C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(-1)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class RiskInfiltrationLoss(nn.Module):
    """
    Loss for infiltration forecasting:
    - Per-step hazard BCE: predicts malicious activity at each future window k
    - Cumulative probability loss: aggregates horizon risk
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        r_step_pred: torch.Tensor,        # [B, K]
        r_step_target: torch.Tensor,      # [B, K]
        r_cumul_pred: Optional[torch.Tensor] = None,    # [B]
        r_cumul_target: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        r_step_pred = torch.clamp(r_step_pred, 1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(r_step_pred, r_step_target.float())

        if r_cumul_pred is not None and r_cumul_target is not None:
            r_cumul_pred = torch.clamp(r_cumul_pred, 1e-7, 1.0 - 1e-7)
            loss = loss + 0.5 * F.binary_cross_entropy(r_cumul_pred, r_cumul_target.float())

        return loss


class CombinedWorldModelLoss(nn.Module):
    """
    Master multi-task objective container for the World Dynamics Transformer.
    Balancing:
        L_total = λ_dyn * L_dyn + λ_direct * L_direct + λ_reg * L_reg
                + λ_state * L_state + λ_attack * L_attack + λ_risk * L_risk
    """

    def __init__(
        self,
        lambda_dyn: float = 1.0,
        lambda_direct: float = 0.1,
        lambda_reg: float = 0.01,
        lambda_state: float = 0.1,
        lambda_attack: float = 0.5,
        lambda_risk: float = 0.5,
        gamma_dyn: float = 0.95,
    ):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_direct = lambda_direct
        self.lambda_reg = lambda_reg
        self.lambda_state = lambda_state
        self.lambda_attack = lambda_attack
        self.lambda_risk = lambda_risk

        self.dyn_loss = LatentDynamicsLoss(gamma=gamma_dyn)
        self.direct_loss = nn.MSELoss()
        self.reg_loss = LatentRegularizationLoss()
        self.state_loss = StateReconstructionLoss()
        self.attack_loss = AttackStageFocalLoss()
        self.risk_loss = RiskInfiltrationLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss and individual itemized metrics.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["z_pred"].device)

        # 1. Primary Latent Dynamics Loss
        if "z_pred" in predictions and "z_target" in targets:
            l_dyn = self.dyn_loss(predictions["z_pred"], targets["z_target"])
            total_loss = total_loss + self.lambda_dyn * l_dyn
            losses["loss_dyn"] = l_dyn.item()

        # 2. Auxiliary Direct Prediction Loss
        if "z_direct" in predictions and "z_target_horizon" in targets:
            l_dir = self.direct_loss(predictions["z_direct"], targets["z_target_horizon"].detach())
            total_loss = total_loss + self.lambda_direct * l_dir
            losses["loss_direct"] = l_dir.item()

        # 3. Latent Regularization Loss
        if "z_pred" in predictions:
            l_reg = self.reg_loss(predictions["z_pred"])
            total_loss = total_loss + self.lambda_reg * l_reg
            losses["loss_reg"] = l_reg.item()

        # 4. State Reconstruction Loss (if state decoder active)
        if "state_recon" in predictions and "stats_target" in targets:
            l_state = self.state_loss(predictions["state_recon"], targets["stats_target"])
            total_loss = total_loss + self.lambda_state * l_state
            losses["loss_state"] = l_state.item()

        # 5. Attack Stage Loss (if attack head active)
        if "attack_logits" in predictions and "attack_stage_target" in targets:
            l_att = self.attack_loss(predictions["attack_logits"], targets["attack_stage_target"])
            total_loss = total_loss + self.lambda_attack * l_att
            losses["loss_attack"] = l_att.item()

        # 6. Risk Loss (if risk head active)
        if "risk_step_pred" in predictions and "risk_step_target" in targets:
            l_risk = self.risk_loss(
                predictions["risk_step_pred"],
                targets["risk_step_target"],
                predictions.get("risk_cumul_pred"),
                targets.get("risk_cumul_target"),
            )
            total_loss = total_loss + self.lambda_risk * l_risk
            losses["loss_risk"] = l_risk.item()

        losses["loss_total"] = total_loss.item()
        return total_loss, losses
