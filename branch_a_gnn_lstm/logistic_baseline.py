"""The PS 26153 logistic-regression baseline, on exactly Branch A's inputs.

The problem statement asks for "benchmark results comparing model performance
(F1 score, precision, recall, false positive rate) against a logistic
regression baseline trained on the same features". The repository had one
(cyberworld_v4/baselines/learned.py) only on the separate v4 track; the plan
that produces the served Branch A never trained or reported it.

Same features means the same tensor: the [L, 27] window sequence Branch A's
LSTM reads, flattened, plus the same elapsed-time channel. A logistic
regression on a weaker hand-built representation would rig the comparison.
Same target: the binary event Branch A's risk head is scored on
(`risk > risk_positive_above`). Same decision rule: an operating point fitted
on validation with the same criterion and alert budget, then applied unchanged
to the held-out test.

Why not sklearn: the training split is ~20.7M windows x 405 features, ~33 GB
as float32. The model here IS logistic regression -- one linear layer and a
sigmoid, L2-regularised through weight decay -- fitted by streaming Adam over
the same loader, so it costs one pass of the loader and no extra copy.
Features are standardised with running statistics (accumulated from the
training stream, frozen at evaluation), which is what
StandardScaler + LogisticRegression does in one pass.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLogisticBaseline(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, with_time: bool = True):
        super().__init__()
        self.seq_len, self.input_dim, self.with_time = int(seq_len), int(input_dim), bool(with_time)
        d = self.seq_len * self.input_dim + (self.seq_len if self.with_time else 0)
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("m2", torch.zeros(d))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.linear = nn.Linear(d, 1)

    def features(self, x: torch.Tensor, t_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = x.reshape(x.shape[0], -1).float()
        if self.with_time:
            if t_history is None:
                t_history = torch.zeros(x.shape[0], self.seq_len, device=x.device)
            # the same log compression MultiTaskLSTM applies to its time channel
            f = torch.cat([f, torch.sign(t_history) * torch.log1p(t_history.abs())], dim=1)
        return f

    @torch.no_grad()
    def update_stats(self, f: torch.Tensor) -> None:
        """Chan et al. parallel update of the running mean / M2 (exact)."""
        n_b = f.shape[0]
        if n_b == 0:
            return
        mean_b = f.mean(0)
        m2_b = ((f - mean_b) ** 2).sum(0)
        n_a = self.count.to(f.dtype)
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean += delta * (n_b / n)
        self.m2 += m2_b + delta ** 2 * (n_a * n_b / n)
        self.count += n_b

    def standardise(self, f: torch.Tensor) -> torch.Tensor:
        var = self.m2 / self.count.clamp_min(2).to(f.dtype)
        return (f - self.mean) / var.clamp_min(1e-12).sqrt()

    def forward(self, x: torch.Tensor, t_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logit of P(positive)."""
        return self.linear(self.standardise(self.features(x, t_history))).squeeze(-1)


def train_logistic_baseline(loader, unpack, device, *, seq_len: int, input_dim: int,
                            risk_positive_above: float, epochs: int = 1, lr: float = 1e-3,
                            weight_decay: float = 1e-4, log=print) -> SequenceLogisticBaseline:
    """Fit on the training loader. `unpack(batch, device)` is the loader's own
    unpacker (host_major.unpack_batch for the batched loader)."""
    model = SequenceLogisticBaseline(seq_len, input_dim).to(device)
    opt = torch.optim.Adam(model.linear.parameters(), lr=lr, weight_decay=weight_decay)
    nb = (str(device) == "cuda")
    for ep in range(1, epochs + 1):
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        k = 0
        for batch in loader:
            batch = unpack(batch, device)
            x = batch["features"].to(device, non_blocking=nb)
            th = batch["t_history"].to(device, non_blocking=nb) if "t_history" in batch else None
            y = (batch["risk"].to(device, non_blocking=nb) > risk_positive_above).float().reshape(-1)
            f = model.features(x, th)
            model.update_stats(f)
            logit = model.linear(model.standardise(f)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach().double()
            k += 1
        log(f"  logistic baseline epoch {ep}: train BCE {float(loss_sum) / max(k, 1):.4f} "
            f"over {k} batches")
    return model.eval()


@torch.no_grad()
def score_histograms(model: SequenceLogisticBaseline, loader, unpack, device, *,
                     risk_positive_above: float, bins: int) -> Dict[str, torch.Tensor]:
    """Positive / negative score histograms on Branch A's bin grid, so the PR
    curve, operating point and AUC are computed by the SAME functions as for
    Branch A (retrain_branch_a_live._pr_curve_from_histograms)."""
    nb = (str(device) == "cuda")
    from cyberworld_v4.device_hist import device_hist
    h = {k: torch.zeros(bins, device=device, dtype=torch.long)
         for k in ("pos", "neg", "onset_pos", "onset_neg")}
    has_prev = False
    for batch in loader:
        batch = unpack(batch, device)
        x = batch["features"].to(device, non_blocking=nb)
        th = batch["t_history"].to(device, non_blocking=nb) if "t_history" in batch else None
        y = (batch["risk"].to(device, non_blocking=nb) > risk_positive_above).reshape(-1)
        p = torch.sigmoid(model(x, th)).clamp(0, 1)
        b = (p * (bins - 1)).long().clamp_(0, bins - 1)
        h["pos"] += device_hist(b, bins, y.long())
        h["neg"] += device_hist(b, bins, (~y).long())
        if "prev_attack" in batch:
            has_prev = True
            on = batch["prev_attack"].to(device, non_blocking=nb).reshape(-1) == 0
            h["onset_pos"] += device_hist(b, bins, (y & on).long())
            h["onset_neg"] += device_hist(b, bins, (~y & on).long())
    out = {k: v.cpu().numpy() for k, v in h.items()}
    if not has_prev:
        out["onset_pos"] = out["onset_neg"] = None
    return out
