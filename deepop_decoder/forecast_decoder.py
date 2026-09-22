"""
DeepOP-Style ATT&CK CWA Decoder conditioned on World Model Future States.

Novel conditioning formulation:
Cross-attends into predicted future host embeddings [H_hat_{t+1..t+K}] from Branch B
and autoregressively decodes upcoming MITRE ATT&CK technique tokens.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepop_decoder.joint_vocab import JointAttackVocab, get_joint_vocab, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from deepop_decoder.cwa import CausalWindowAttention


# ---------------------------------------------------------------------------
# The step-0 "continuity prior": a hand-set logit bonus, applied at inference
# only, that pushes the first forecast step towards the last OBSERVED token.
#
# It is not learned and it is not in the training objective, so anything
# measured through `forecast_sequence` with it enabled is partly this constant
# and not the model. Measured on an untrained decoder over 4,000 samples with
# an 82.5% Benign observed-token mix (the corpus rate): the bonus changes
# 71.9% of step-0 decisions, and drives agreement between the step-0 output
# and the observed token from 0.177 to 0.896. In other words, with the bonus
# on, step 0 *is* substantially the persistence baseline -- so "free-running
# accuracy beats persistence" cannot be concluded from a run that leaves it on.
#
# Serving (control_backend/model_adapter.py, correlation/trajectory_assembler.py)
# calls `forecast_sequence` without overriding it, so the default must stay 1.0
# or serving behaviour changes silently. Evaluation passes
# `continuity_bonus=0.0` to measure the model alone; `evaluate_forecast_rigor`
# reports both.
CONTINUITY_BONUS_ATTACK = 1.2   # x3.32 odds on the observed attack token
CONTINUITY_BONUS_BENIGN = 1.8   # x6.05 odds on Benign


class CWADecoderLayer(nn.Module):
    """
    Decoder layer combining:
    1. Causal Window Self-Attention (CWA)
    2. Cross-Attention into predicted future states H_hat
    3. Position-wise Feed-Forward Network
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 6,
        window_sizes: Optional[List[int]] = None,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_cwa: bool = True,
    ):
        super(CWADecoderLayer, self).__init__()
        self.use_cwa = use_cwa
        if self.use_cwa:
            self.cwa_self_attn = CausalWindowAttention(
                d_model=d_model, n_heads=n_heads, window_sizes=window_sizes, dropout=dropout
            )
        else:
            self.cwa_self_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
            )
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention (CWA or Plain Causal Attention)
        if self.use_cwa:
            tgt2 = self.cwa_self_attn(tgt, tgt, tgt, key_padding_mask=tgt_mask)
        else:
            seq_len = tgt.shape[1]
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=tgt.device), diagonal=1)
            tgt2, _ = self.cwa_self_attn(query=tgt, key=tgt, value=tgt, attn_mask=causal_mask, key_padding_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 2. Cross-Attention to predicted future state embeddings
        tgt2, _ = self.cross_attn(query=tgt, key=memory, value=memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 3. Feedforward
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class DeepOPForecastDecoder(nn.Module):
    """
    Forecasting Decoder: generates anticipated ATT&CK technique token sequences
    conditioned on predicted future host states H_hat_{t+1..t+K}.
    Supports modular ablations: use_cwa, use_direct_head, use_prototypes, use_future_gate.
    """

    def __init__(
        self,
        d_latent: int = 12,       # Branch B latent host embedding dimension
        d_model: int = 72,        # Decoder model dimension (divisible by 6 heads and 3 scales)
        vocab_size: Optional[int] = None,
        n_heads: int = 6,
        num_layers: int = 2,
        window_sizes: Optional[List[int]] = None,
        dim_feedforward: int = 144,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        use_cwa: bool = True,
        use_direct_head: bool = True,
        use_prototypes: bool = True,
        use_future_gate: bool = True,
    ):
        super(DeepOPForecastDecoder, self).__init__()
        self.vocab = get_joint_vocab()
        self.vocab_size = vocab_size or self.vocab.vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_cwa = use_cwa
        self.use_direct_head = use_direct_head
        self.use_prototypes = use_prototypes
        self.use_future_gate = use_future_gate

        # Token embedding and sinusoidal positional encoding
        self.token_embed = nn.Embedding(self.vocab_size, d_model, padding_idx=self.vocab.pad_idx)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Context projection from future state H_hat to d_model
        self.future_proj = nn.Linear(d_latent, d_model)

        # Decoder stack
        self.layers = nn.ModuleList(
            [
                CWADecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    window_sizes=window_sizes or [2, 4, 8],
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_cwa=use_cwa,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        # Residual gating parameter coupling continuous future state H_hat to discrete logits
        self.future_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Direct prediction head from future latent embedding H_hat
        self.direct_head = nn.Linear(d_latent, self.vocab_size)
        # Prototypical Metric Head: class prototype centroids in d_latent embedding space.
        #
        # This used to be `torch.zeros(...)`, which made the whole head a
        # permanent no-op for any trainer that does not seed it from class
        # centroids -- i.e. for `train_deepop_live`, which is the trainer that
        # produced every shipped checkpoint. The mechanism: all-zero
        # prototypes -> `active_proto_mask.sum() == 0` -> `forward` takes the
        # `torch.zeros(...)` branch, which is a *fresh constant*, so
        # `self.prototypes` is not in the autograd graph at all. Measured:
        # after 50 AdamW steps `prototypes.grad is None` and
        # `prototypes.abs().max() == 0.0` -- exactly zero, not merely small.
        # A zero parameter whose only gradient path is gated off by its own
        # zero-ness can never leave zero.
        #
        # trunc_normal_ std=0.02 gives ||p|| ~ 0.02*sqrt(12) = 0.069, an order
        # of magnitude above the 1e-4 activation threshold, so the head starts
        # alive and receives gradient. Loading an older all-zero checkpoint
        # restores the old (inert) behaviour exactly, so this is not a
        # silent change to anything already on disk.
        self.prototypes = nn.Parameter(torch.empty(self.vocab_size, d_latent))
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.proto_scale = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))
        # Prediction projection from autoregressive decoder
        self.fc_out = nn.Linear(d_model, self.vocab_size)

    def forward(
        self,
        h_future: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass with continuous future state residual gating.
        Args:
            h_future: [batch_size, K, d_latent] predicted future host states
            tgt_tokens: [batch_size, seq_len] input token sequence (e.g. <BOS> + tokens)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        B, S = tgt_tokens.shape
        # Embed tokens and add positional encoding
        x = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        x = x + self.pos_embed[:, :S, :]

        # Project future latent context
        memory = self.future_proj(h_future)  # [B, K, d_model]

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory=memory)

        # Residual gating: anchor token representation directly to future world state
        K = memory.shape[1]
        if self.use_future_gate:
            if S <= K:
                x = x + self.future_gate * memory[:, :S, :]
            else:
                x[:, :K, :] = x[:, :K, :] + self.future_gate * memory

        x = self.norm(x)
        dec_logits = self.fc_out(x)

        # Prototypical cosine similarity metric from continuous future state h_future
        if self.use_prototypes:
            proto_norm = self.prototypes.norm(dim=-1, keepdim=True)
            active_proto_mask = (proto_norm > 1e-4).float().squeeze(-1)
            if active_proto_mask.sum() > 0:
                h_norm = F.normalize(h_future, p=2, dim=-1)
                p_norm = F.normalize(self.prototypes + 1e-8, p=2, dim=-1)
                proto_logits = self.proto_scale * torch.einsum("bkd,vd->bkv", h_norm, p_norm)
                # An unseeded class used to be masked to logit 0. A seeded
                # class gets proto_scale * cos in [-2.5, +2.5] with mean ~0, so
                # a literal 0 sits at the MIDDLE of that range: masking to zero
                # *promotes* a class the prototype head knows nothing about
                # above every class it actively dislikes. With 6 of the 10
                # tokens absent from this corpus (measured: only Benign.None,
                # C2.T1071, Impact.T1498, InitialAccess.T1190 occur in train)
                # that is six free votes per step. Floor them at -proto_scale,
                # the least a cosine can ever award, so "no prototype" is the
                # worst evidence rather than average evidence.
                floor = -self.proto_scale.abs().expand_as(proto_logits)
                proto_logits = torch.where(
                    active_proto_mask.bool().unsqueeze(0).unsqueeze(0), proto_logits, floor
                )
            else:
                proto_logits = torch.zeros(B, K, self.vocab_size, device=h_future.device)
        else:
            proto_logits = torch.zeros(B, K, self.vocab_size, device=h_future.device)

        # Direct projection + Prototypical metric from continuous world state H_hat
        if self.use_direct_head or self.use_prototypes:
            head_contrib = self.direct_head(h_future[:, :min(S, K), :]) if self.use_direct_head else 0.0
            if S <= K:
                direct_logits = head_contrib + (proto_logits[:, :S, :] if self.use_prototypes else 0.0)
            else:
                d_part = (self.direct_head(h_future) if self.use_direct_head else 0.0) + (proto_logits if self.use_prototypes else 0.0)
                pad = d_part[:, -1:, :].repeat(1, S - K, 1)
                direct_logits = torch.cat([d_part, pad], dim=1)
            logits = dec_logits + direct_logits
        else:
            logits = dec_logits
        return logits

    def forecast_sequence(
        self,
        h_future: torch.Tensor,
        max_steps: int = 4,
        observed_token: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.0,
        temperature: float = 0.8,
        return_probs: bool = False,
        continuity_bonus: float = 1.0,
    ) -> Any:
        """
        Autoregressive sequence generation conditioned on h_future and optional observed_token.
        Args:
            h_future: [batch_size, K, d_latent]
            max_steps: maximum number of tokens to forecast
            observed_token: [batch_size] or [batch_size, 1] observed technique at t=0
            repetition_penalty: penalty discount applied to previously generated tokens (breaks mode collapse)
            temperature: INERT for the emitted tokens. Decoding is greedy, and
                argmax(x / T) == argmax(x) for every T > 0, so this cannot
                change a single output token; verified by
                `test_temperature_cannot_change_the_output`. It is kept only
                because callers pass it. Do not read it as "the model samples".
            return_probs: if True, additionally returns per-step attack and top token probabilities
            continuity_bonus: scale on the un-learned step-0 bonus towards
                `observed_token` (see CONTINUITY_BONUS_* above). 1.0 is the
                shipped serving behaviour; pass 0.0 to measure the model
                without the hard-coded persistence prior mixed in.
        Returns:
            If return_probs is False:
                pred_tokens: [batch_size, max_steps] integer token IDs
                decoded_names: list of decoded (coarse, technique) tuples per batch sample
            If return_probs is True:
                (pred_tokens, decoded_names, step_attack_probs, step_token_probs)
        """
        self.eval()
        B = h_future.shape[0]
        device = h_future.device

        # Start sequence cleanly with <BOS> token (prefix_len = 1) for 100% training/inference parity
        curr_tokens = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        prefix_len = 1

        all_attack_probs = [[] for _ in range(B)]
        all_token_probs = [[] for _ in range(B)]

        with torch.no_grad():
            for step in range(max_steps):
                logits = self.forward(h_future, curr_tokens)  # [B, curr_len, vocab_size]
                next_logits = logits[:, -1, :].clone()  # [B, vocab_size]

                # Suppress special tokens
                next_logits[:, self.vocab.bos_idx] = -1e9
                next_logits[:, self.vocab.pad_idx] = -1e9

                # If step==0 and observed_token is provided, bias toward the
                # observed state. Vectorised: the old per-sample Python loop
                # ran B iterations per call, which is 1.46M per validation
                # epoch at the live val size.
                if step == 0 and observed_token is not None and continuity_bonus != 0.0:
                    obs_t_flat = observed_token.squeeze(1) if observed_token.dim() == 2 else observed_token
                    benign_idx = self.vocab.encode("Benign", None)
                    usable = (obs_t_flat != self.vocab.bos_idx) & (obs_t_flat != self.vocab.pad_idx)
                    bonus = torch.where(
                        obs_t_flat == benign_idx,
                        torch.full_like(next_logits[:, 0], CONTINUITY_BONUS_BENIGN),
                        torch.full_like(next_logits[:, 0], CONTINUITY_BONUS_ATTACK),
                    ) * usable.to(next_logits.dtype) * continuity_bonus
                    next_logits.scatter_add_(
                        1, obs_t_flat.clamp(min=0).unsqueeze(1), bonus.unsqueeze(1)
                    )

                # Apply repetition penalty to active attack tokens generated so far
                if repetition_penalty > 1.0 and curr_tokens.shape[1] > prefix_len:
                    benign_idx = self.vocab.encode("Benign", None)
                    gen_history = curr_tokens[:, prefix_len:]
                    for b in range(B):
                        for tok in gen_history[b].unique():
                            tok_idx = int(tok.item())
                            if tok_idx != benign_idx and tok_idx != self.vocab.pad_idx and tok_idx != self.vocab.bos_idx:
                                if next_logits[b, tok_idx] > 0:
                                    next_logits[b, tok_idx] /= repetition_penalty
                                else:
                                    next_logits[b, tok_idx] *= repetition_penalty

                # Record softmax probabilities over all vocabulary classes for this step
                step_probs = torch.softmax(next_logits, dim=-1)
                benign_idx = self.vocab.encode("Benign", None)

                # Greedy argmax with temperature-scaled logits
                if temperature > 0.0:
                    scaled_logits = next_logits / temperature
                    next_token = scaled_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                # Vectorised bookkeeping. The old form did B `.item()` calls
                # per step -- 5 * B GPU syncs, which is what made
                # `train_cwa_decoder`'s whole-val-set `forecast_sequence` call
                # unrunnable at the live val size (1.46M samples).
                if return_probs:
                    p_atk = torch.clamp(1.0 - step_probs[:, benign_idx], 0.0, 1.0).cpu().tolist()
                    p_tok = step_probs.gather(1, next_token).squeeze(1).cpu().tolist()
                    for b in range(B):
                        all_attack_probs[b].append(p_atk[b])
                        all_token_probs[b].append(p_tok[b])

                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        # Exclude initial prefix (<BOS>)
        pred_tokens = curr_tokens[:, prefix_len:]
        decoded_names = []
        for b in range(B):
            sample_tokens = []
            for t_idx in pred_tokens[b].cpu().numpy():
                sample_tokens.append(self.vocab.decode(int(t_idx)))
            decoded_names.append(sample_tokens)

        if return_probs:
            return pred_tokens, decoded_names, all_attack_probs, all_token_probs

        return pred_tokens, decoded_names


def evaluate_forecast_rigor(
    decoder: DeepOPForecastDecoder,
    h_future: torch.Tensor,
    target_tokens: torch.Tensor,
    observed_tokens: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
) -> Dict[str, Any]:
    """
    Rigorously evaluates DeepOP decoder comparing:
    1. Free-running Autoregressive Token Accuracy (no ground truth during rollout)
    2. Teacher-Forced Token Accuracy
    3. Macro-F1 across all vocabulary classes
    4. Active-Technique Macro-F1 (excluding PAD and Benign)
    5. Comparison against Persistence Baseline (assuming no change from last observed token)
    6. Per-Technique Precision, Recall, and F1 metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    decoder.eval()
    B, K = target_tokens.shape
    device = h_future.device

    # 1. Free-running generation conditioned on observed token at t=0
    with torch.no_grad():
        free_pred, _ = decoder.forecast_sequence(
            h_future, max_steps=K, observed_token=observed_tokens, repetition_penalty=repetition_penalty
        )

    y_true = target_tokens.cpu().numpy().reshape(-1)
    y_pred_free = free_pred.cpu().numpy().reshape(-1)

    free_acc = float(accuracy_score(y_true, y_pred_free))
    macro_f1 = float(f1_score(y_true, y_pred_free, average="macro", zero_division=0))

    # Active technique classes: exclude PAD (0) and Benign
    pad_idx = decoder.vocab.pad_idx
    benign_idx = decoder.vocab.encode("Benign", None)
    active_mask = (y_true != pad_idx) & (y_true != benign_idx)

    if active_mask.sum() > 0:
        active_f1 = float(
            f1_score(
                y_true[active_mask],
                y_pred_free[active_mask],
                average="macro",
                zero_division=0,
            )
        )
    else:
        active_f1 = macro_f1

    # 2. Teacher-forced generation
    bos = torch.full((B, 1), decoder.vocab.bos_idx, dtype=torch.long, device=device)
    teacher_input = torch.cat([bos, target_tokens[:, :-1]], dim=1)
    with torch.no_grad():
        logits_tf = decoder.forward(h_future, teacher_input)
        preds_tf = logits_tf.argmax(dim=-1).cpu().numpy().reshape(-1)
    tf_acc = float(accuracy_score(y_true, preds_tf))

    # 3. Persistence baseline
    if observed_tokens is not None:
        persist_pred = observed_tokens.unsqueeze(1).repeat(1, K).cpu().numpy().reshape(-1)
        persist_acc = float(accuracy_score(y_true, persist_pred))
    else:
        persist_acc = 0.0

    # 4. Per-class metrics
    all_classes = sorted(list(set(y_true) | set(y_pred_free)))
    p, r, f1, supp = precision_recall_fscore_support(
        y_true, y_pred_free, labels=all_classes, zero_division=0
    )
    per_technique = {}
    for cls_idx, p_val, r_val, f1_val, s_val in zip(all_classes, p, r, f1, supp):
        coarse, tech = decoder.vocab.decode(cls_idx)
        token_name = f"{coarse}.{tech}" if tech else coarse
        per_technique[token_name] = {
            "token_id": int(cls_idx),
            "precision": float(p_val),
            "recall": float(r_val),
            "f1": float(f1_val),
            "support": int(s_val),
            "is_active": bool(cls_idx != pad_idx and cls_idx != benign_idx),
        }

    return {
        "free_running_accuracy": free_acc,
        "teacher_forced_accuracy": tf_acc,
        "free_running_macro_f1": macro_f1,
        "active_technique_macro_f1": active_f1,
        "persistence_baseline_accuracy": persist_acc,
        "per_technique": per_technique,
    }

