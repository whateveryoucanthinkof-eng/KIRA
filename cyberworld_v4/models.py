"""CyberWorld v4 prediction heads with explicit statistical semantics.

The v3 model had one scalar called "risk" that was simultaneously described as
risk, confidence, hazard and calibrated risk, trained with BCE against a
hand-designed severity lookup (TACTIC_BASE_SEVERITY). BCE against a continuous
severity score gives that score a probabilistic reading it does not have.

Here each quantity is a separate head with a loss that matches what it means
(spec 7, 8, 9, 38, 39):

  current_attack   BCE          P(A_t = 1 | X<=t)                 nowcasting
  future_attack    BCE per step P(A_{t+k} = 1 | X<=t)             forecasting
  hazard           masked BCE   P(onset = t+k | no onset before)  survival
  techniques       multilabel   P(Y_{t+k,j} = 1 | X<=t)           spec 13
  severity         SmoothL1     operator ranking aid, NOT a probability

Cumulative onset probability is derived, never predicted directly, and never
max(): P(onset <= t+K) = 1 - prod(1 - h_k).

The world model predicts a *distribution* (mean and log-variance) trained with
Gaussian NLL, so P(S_t+1 | S_t) is literal rather than a point estimate with a
confidence band bolted on afterwards.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CyberWorldConfig, DEFAULT_CONFIG


class SequenceEncoder(nn.Module):
    """LSTM over the [B, L, D] history. Deliberately plain.

    Spec 36: complexity must be earned by ablation. This is the A1 rung; a
    graph or transformer encoder must beat it on the benchmark before replacing
    it.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.norm(out[:, -1, :])


class ForecastHeads(nn.Module):
    """The four target families, as separate heads on a shared encoding."""

    def __init__(
        self,
        n_techniques: int,
        config: CyberWorldConfig = DEFAULT_CONFIG,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.K = config.temporal.forecast_steps
        self.n_techniques = n_techniques

        def mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        # All heads emit LOGITS. Sigmoid is applied at the metric/serving
        # boundary so temperature scaling has logits to work on.
        self.current_attack = mlp(1)
        self.future_attack = mlp(self.K)
        self.hazard = mlp(self.K)
        self.techniques = mlp(self.K * n_techniques)
        self.severity = mlp(1)          # regression; never fed to BCE

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        return {
            "current_attack_logit": self.current_attack(h).squeeze(-1),
            "future_attack_logits": self.future_attack(h),
            "hazard_logits": self.hazard(h),
            "technique_logits": self.techniques(h).view(B, self.K, self.n_techniques),
            "severity": self.severity(h).squeeze(-1),
        }


class CyberWorldForecaster(nn.Module):
    """Encoder + heads. The v4 replacement for Branch A."""

    def __init__(
        self,
        n_techniques: int,
        config: CyberWorldConfig = DEFAULT_CONFIG,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config.state_dim, hidden_dim, layers, dropout)
        self.heads = ForecastHeads(n_techniques, config, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(self.encoder(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Probabilities plus the derived cumulative curve. eval() enforced.

        v3's explainability ran the model in train() mode with dropout active,
        making outputs non-deterministic; predict() never does that.
        """
        was_training = self.training
        self.eval()
        try:
            o = self(x)
            hazard = torch.sigmoid(o["hazard_logits"])
            return {
                "current_attack": torch.sigmoid(o["current_attack_logit"]),
                "future_attack": torch.sigmoid(o["future_attack_logits"]),
                "hazard": hazard,
                "cumulative_onset": 1.0 - torch.cumprod(1.0 - hazard, dim=1),
                "techniques": torch.sigmoid(o["technique_logits"]),
                "severity": o["severity"],
            }
        finally:
            if was_training:
                self.train()


def forecast_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Multi-task loss where every term matches its target's semantics.

    The hazard term is masked by `at_risk`: once a host's onset occurs it leaves
    the risk set, and a host already under attack at t contributes nothing.
    Including those as negatives would train the model to call ongoing attacks
    safe.
    """
    w = {"current": 1.0, "future": 1.0, "hazard": 1.0, "technique": 1.0, "severity": 0.1}
    if weights:
        w.update(weights)

    parts: Dict[str, float] = {}

    l_cur = F.binary_cross_entropy_with_logits(
        out["current_attack_logit"], batch["current_attack"].float(), pos_weight=pos_weight
    )
    parts["current"] = float(l_cur.item())

    l_fut = F.binary_cross_entropy_with_logits(
        out["future_attack_logits"], batch["future_attack"].float(), pos_weight=pos_weight
    )
    parts["future"] = float(l_fut.item())

    at_risk = batch["at_risk"].float()
    raw = F.binary_cross_entropy_with_logits(
        out["hazard_logits"], batch["hazard_target"].float(), reduction="none"
    )
    denom = at_risk.sum().clamp(min=1.0)
    l_haz = (raw * at_risk).sum() / denom
    parts["hazard"] = float(l_haz.item())
    parts["at_risk_fraction"] = float(at_risk.mean().item())

    l_tech = F.binary_cross_entropy_with_logits(
        out["technique_logits"], batch["future_techniques"].float()
    )
    parts["technique"] = float(l_tech.item())

    # Regression, not BCE. Severity is an ordinal operator aid.
    l_sev = F.smooth_l1_loss(out["severity"], batch["severity"].float())
    parts["severity"] = float(l_sev.item())

    total = (
        w["current"] * l_cur
        + w["future"] * l_fut
        + w["hazard"] * l_haz
        + w["technique"] * l_tech
        + w["severity"] * l_sev
    )
    parts["total"] = float(total.item())
    return total, parts


class DistributionalWorldModel(nn.Module):
    """Branch B as a distribution over next states (spec 2, 23).

    Two changes from v3:

    1. Emits mean AND log-variance, trained with Gaussian NLL. The spec asks
       for "the probability distribution over the next network state"; a
       deterministic point predictor with a hardcoded radius is not that.

    2. `stabilize_horizon` damping defaults OFF. In v3 it defaulted True and no
       trainer overrode it, so delta_h was multiplied by 0.95^k *during
       training* — shrinking predictions toward "no change" while the model was
       validated against a persistence baseline. It is kept as an explicit
       inference-time option, never a silent training default.
    """

    def __init__(self, d_latent: int = 12, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, max_context: int = 32):
        super().__init__()
        self.d_latent = d_latent
        self.max_context = max_context
        self.in_proj = nn.Linear(d_latent, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_context, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.mean_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)

    def _step(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = seq[:, -self.max_context :, :]
        T = ctx.shape[1]
        x = self.in_proj(ctx) + self.pos[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=seq.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(x, mask=mask, is_causal=True)[:, -1, :]
        delta = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        return ctx[:, -1, :] + delta, logvar

    def rollout(
        self, h_seq: torch.Tensor, K: int, *, stabilize_horizon: bool = False, decay: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout. Returns (means [B,K,D], logvars [B,K,D])."""
        seq = h_seq
        means, logvars = [], []
        last = h_seq[:, -1, :]
        for k in range(K):
            mu, lv = self._step(seq)
            if stabilize_horizon and k > 0:
                mu = last + (mu - last) * (decay ** k)
            means.append(mu.unsqueeze(1))
            logvars.append(lv.unsqueeze(1))
            seq = torch.cat([seq, mu.unsqueeze(1)], dim=1)
            last = mu
        return torch.cat(means, 1), torch.cat(logvars, 1)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood, the loss that makes P(S_t+1|S_t) real."""
    inv = torch.exp(-logvar)
    return 0.5 * (logvar + (target - mean) ** 2 * inv).mean()
