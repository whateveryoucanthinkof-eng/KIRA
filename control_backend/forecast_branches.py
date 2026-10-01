"""
control_backend/forecast_branches.py
DeepOP's three most likely continuations, for the console's forecast tree.

The served forecast (`DeepOPForecastDecoder.forecast_sequence`) is greedy: one
path. The tree shows the alternatives the decoder was weighing at the first
forecast step. This reproduces step 0 of `forecast_sequence` exactly (BOS and
PAD masked, the same continuity bonus towards the observed token), keeps the
three most probable first tokens, and decodes each continuation greedily
through the same `forward`. Branch A is therefore the served forecast itself.

What this does NOT know, because no model predicts it: which hosts a branch
would pass through, its packet and byte volume, and a per-branch risk peak
(Branch B rolls out one future, not three). Those fields are None and the
console marks them under development.
"""

from typing import List, Optional

import torch

from control_backend.schema import BranchStep, ForecastBranch
from control_backend.tactics import BENIGN, LANES, lane_rank, split_token

_IDS = ("A", "B", "C")
_LANE_LABEL = {
    "Recon": "reconnaissance",
    "CredentialAccess": "credential access",
    "InitialAccess": "initial access",
    "Execution": "execution",
    "C2": "command and control",
    "LateralMovement": "lateral movement",
    "Exfiltration": "exfiltration",
    "Impact": "impact",
}


def _step0_logits(deepop, vocab, h_future, observed_seq, observed_token) -> torch.Tensor:
    """forecast_sequence's step-0 logits, masking and continuity bonus included."""
    from deepop_decoder.forecast_decoder import CONTINUITY_BONUS_ATTACK, CONTINUITY_BONUS_BENIGN

    bos = torch.full((h_future.shape[0], 1), vocab.bos_idx, dtype=torch.long, device=h_future.device)
    logits = deepop.forward(h_future, bos, obs_tokens=observed_seq)[:, -1, :].clone()
    logits[:, vocab.bos_idx] = -1e9
    logits[:, vocab.pad_idx] = -1e9
    bonus_scale = float(getattr(deepop, "default_continuity_bonus", 0.0) or 0.0)
    if observed_token is not None and bonus_scale != 0.0:
        obs = observed_token.reshape(-1)
        benign_idx = vocab.encode(BENIGN, None)
        usable = (obs != vocab.bos_idx) & (obs != vocab.pad_idx)
        bonus = torch.where(
            obs == benign_idx,
            torch.full_like(logits[:, 0], CONTINUITY_BONUS_BENIGN),
            torch.full_like(logits[:, 0], CONTINUITY_BONUS_ATTACK),
        ) * usable.to(logits.dtype) * bonus_scale
        logits.scatter_add_(1, obs.clamp(min=0).unsqueeze(1), bonus.unsqueeze(1))
    return logits


@torch.no_grad()
def decode_branches(
    deepop,
    vocab,
    h_future: torch.Tensor,
    observed_seq: Optional[torch.Tensor],
    observed_token: Optional[torch.Tensor],
    steps: int,
    step_seconds: float,
    current_lane: str,
    n: int = 3,
) -> List[ForecastBranch]:
    """The top-`n` continuations for one host (batch of 1)."""
    probs0 = torch.softmax(_step0_logits(deepop, vocab, h_future, observed_seq, observed_token), dim=-1)[0]
    k = min(n, int((probs0 > 0).sum().item()))
    top_p, top_tok = probs0.topk(k)

    # Decode all continuations in one batch, each forced through its own
    # first token, then greedy -- the rule the served forecast uses.
    h = h_future.expand(k, -1, -1)
    obs = observed_seq.expand(k, -1) if observed_seq is not None else None
    seq = torch.cat([
        torch.full((k, 1), vocab.bos_idx, dtype=torch.long, device=h.device),
        top_tok.unsqueeze(1),
    ], dim=1)
    step_probs = [top_p]
    for _ in range(1, steps):
        logits = deepop.forward(h, seq, obs_tokens=obs)[:, -1, :].clone()
        logits[:, vocab.bos_idx] = -1e9
        logits[:, vocab.pad_idx] = -1e9
        p = torch.softmax(logits, dim=-1)
        nxt = p.argmax(dim=-1, keepdim=True)
        step_probs.append(p.gather(1, nxt).squeeze(1))
        seq = torch.cat([seq, nxt], dim=1)
    step_probs_t = torch.stack(step_probs, dim=1)  # [k, steps]

    mass = float(top_p.sum().item()) or 1.0
    cur_rank = lane_rank(current_lane)
    branches: List[ForecastBranch] = []
    for b in range(k):
        path: List[BranchStep] = []
        for s in range(steps):
            coarse, tech = vocab.decode(int(seq[b, s + 1].item()))
            lane, technique = split_token(f"{coarse}.{tech}")
            path.append(BranchStep(
                horizon_seconds=(s + 1) * step_seconds,
                tactic_lane=lane,
                technique=technique,
                probability=round(float(step_probs_t[b, s].item()), 4),
            ))
        first_attack = next((st for st in path if st.tactic_lane != BENIGN), None)
        if first_attack is None:
            kind, stage, technique = "backoff", BENIGN, "—"
            label = "Returns to benign traffic"
            focus = path[0]
        else:
            stage, technique = first_attack.tactic_lane, first_attack.technique or "—"
            kind = "escalation" if lane_rank(stage) > cur_rank else "pivot"
            verb = "Escalates to" if kind == "escalation" else "Continues as"
            label = f"{verb} {_LANE_LABEL.get(stage, stage)}"
            focus = first_attack
        branches.append(ForecastBranch(
            id=_IDS[b],
            kind=kind,
            label=label,
            stage=stage,
            technique=technique,
            probability=round(float(top_p[b].item()) / mass, 4),
            probability_raw=round(float(top_p[b].item()), 4),
            confidence=focus.probability,
            horizon_seconds=focus.horizon_seconds,
            path=path,
        ))
    return branches


__all__ = ["decode_branches", "LANES"]
