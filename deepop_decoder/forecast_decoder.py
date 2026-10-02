"""
DeepOP attack-sequence predictor, conditioned on the world model.

Zhang, Xue and Su, "DeepOP: A Hybrid Framework for MITRE ATT&CK Sequence
Prediction via Deep Learning and Ontology", Electronics 14(2):257, 2025,
Section 3.4. The paper's model is an encoder-decoder:

  Eq. 1-3   the OBSERVED attack sequence s = {t_1..t_n} of ATT&CK labels is
            embedded and summed with sinusoidal positional encodings;
  Eq. 4-6   an ENCODER of temporal multi-head attention (+FFN, residual,
            LayerNorm) builds its contextual representation z;
  Eq. 7-10  a DECODER generates the next techniques autoregressively with
            causal window attention (deepop_decoder/cwa.py);
  Eq. 11    cross-entropy over the technique vocabulary.

In this system the observed sequence is Branch A's technique for each of the
last `history_steps` windows of the host, and the decoder cross-attends to the
encoder output AND to Branch B's predicted future states, so DeepOP receives
both branches' outputs as learned inputs.

Before this, the decoder had no encoder at all. Branch A reached DeepOP only
through CONTINUITY_BONUS_*: a hand-set logit added at inference, not learned
and not in the loss. That path is kept (default off) solely so that
checkpoints trained without the encoder still load and behave as they did.

Deviations from the paper, each forced by the data rather than chosen:
  * token embeddings are learned (nn.Embedding), not Word2Vec: the network-
    observable vocabulary here has 10 tokens, too few for skip-gram to learn
    anything a learned embedding does not;
  * the decoder also attends to Branch B's continuous future states, which the
    paper (text-derived CTI sequences) has no equivalent of.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
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


def observed_sequence_tokens(token_ids, n_obs: int, vocab) -> List[int]:
    """The encoder input for one sample: the last n_obs observed technique
    tokens, prefixed with <BOS> and left-padded with <PAD> to n_obs + 1.

    Training builds it from the corpus labels of the host's history windows;
    serving builds it from Branch A's technique for the same windows. One
    function for both so the two cannot drift.
    """
    toks = [int(t) for t in list(token_ids)[-n_obs:]] if n_obs > 0 else []
    return [vocab.pad_idx] * (n_obs - len(toks)) + [vocab.bos_idx] + toks


class SinusoidalPositionalEncoding(nn.Module):
    """DeepOP Eq. 2: PE(i,2k) = sin(i / 10000^(2k/d)), PE(i,2k+1) = cos(...)."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        # Not a parameter and not saved: it is a fixed function of position.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]


class ObservedSequenceEncoder(nn.Module):
    """DeepOP encoder (Eq. 3-6): E_emb = H + PE, then L layers of temporal
    multi-head self-attention with feed-forward, residual and LayerNorm."""

    def __init__(self, token_embed: nn.Embedding, pos_enc: nn.Module, d_model: int,
                 n_heads: int, num_layers: int, dim_feedforward: int, dropout: float,
                 pad_idx: int):
        super().__init__()
        self.token_embed = token_embed   # shared with the decoder
        self.pos_enc = pos_enc
        self.d_model = d_model
        self.pad_idx = pad_idx
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

    def forward(self, obs_tokens: torch.Tensor):
        pad = obs_tokens == self.pad_idx
        # A fully padded row would give every query nothing to attend to.
        pad[pad.all(dim=1), -1] = False
        x = self.token_embed(obs_tokens) * math.sqrt(self.d_model) + self.pos_enc(obs_tokens.shape[1])
        return self.encoder(x, src_key_padding_mask=pad), pad


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
        window_mode: str = "partitioned",
    ):
        super(CWADecoderLayer, self).__init__()
        self.use_cwa = use_cwa
        if self.use_cwa:
            self.cwa_self_attn = CausalWindowAttention(
                d_model=d_model, n_heads=n_heads, window_sizes=window_sizes, dropout=dropout,
                window_mode=window_mode,
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
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention (CWA or Plain Causal Attention)
        if self.use_cwa:
            tgt2 = self.cwa_self_attn(tgt, tgt, tgt, key_padding_mask=tgt_mask)
        else:
            seq_len = tgt.shape[1]
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=tgt.device), diagonal=1)
            tgt2, _ = self.cwa_self_attn(query=tgt, key=tgt, value=tgt, attn_mask=causal_mask, key_padding_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 2. Cross-attention to [encoded observed sequence ; Branch B future states]
        tgt2, _ = self.cross_attn(query=tgt, key=memory, value=memory,
                                  key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 3. Feedforward
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class DeepOPForecastDecoder(nn.Module):
    """
    DeepOP encoder-decoder (see module docstring). Generates the next ATT&CK
    tokens from the observed technique sequence (Branch A) and the predicted
    future states (Branch B).

    Constructor defaults are the paper's model. The legacy_* flags reproduce
    the pre-encoder decoder exactly, so its checkpoints still load:
    use `DeepOPForecastDecoder.from_checkpoint`.
    """

    #: The paper architecture.
    PAPER_ARCH = dict(use_obs_encoder=True, pos_encoding="sinusoidal", window_mode="partitioned",
                      use_direct_head=False, use_prototypes=False, use_future_gate=False,
                      default_continuity_bonus=0.0)
    #: What every checkpoint saved before the encoder existed was trained as.
    LEGACY_ARCH = dict(use_obs_encoder=False, pos_encoding="learned", window_mode="sliding",
                       use_direct_head=True, use_prototypes=True, use_future_gate=True,
                       default_continuity_bonus=1.0)

    def __init__(
        self,
        d_latent: int = 27,       # world-state width s(t): TGNE latent + host attributes
        d_model: int = 72,        # divisible by 6 heads and 3 window scales
        vocab_size: Optional[int] = None,
        n_heads: int = 6,
        num_layers: int = 2,
        window_sizes: Optional[List[int]] = None,
        dim_feedforward: int = 144,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        use_cwa: bool = True,
        use_obs_encoder: bool = True,
        pos_encoding: str = "sinusoidal",
        window_mode: str = "partitioned",
        use_direct_head: bool = False,
        use_prototypes: bool = False,
        use_future_gate: bool = False,
        default_continuity_bonus: float = 0.0,
        max_obs_len: int = 32,
    ):
        super(DeepOPForecastDecoder, self).__init__()
        if pos_encoding not in ("sinusoidal", "learned"):
            raise ValueError(f"pos_encoding must be 'sinusoidal' or 'learned', got {pos_encoding!r}")
        self.vocab = get_joint_vocab()
        self.vocab_size = vocab_size or self.vocab.vocab_size
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_cwa = use_cwa
        self.use_obs_encoder = use_obs_encoder
        self.pos_encoding = pos_encoding
        self.window_mode = window_mode
        self.use_direct_head = use_direct_head
        self.use_prototypes = use_prototypes
        self.use_future_gate = use_future_gate
        self.default_continuity_bonus = float(default_continuity_bonus)
        self._arch = dict(
            d_latent=d_latent, d_model=d_model, vocab_size=self.vocab_size, n_heads=n_heads,
            num_layers=num_layers, window_sizes=list(window_sizes or [2, 4, 8]),
            dim_feedforward=dim_feedforward, max_seq_len=max_seq_len, dropout=dropout,
            use_cwa=use_cwa, use_obs_encoder=use_obs_encoder, pos_encoding=pos_encoding,
            window_mode=window_mode, use_direct_head=use_direct_head,
            use_prototypes=use_prototypes, use_future_gate=use_future_gate,
            default_continuity_bonus=float(default_continuity_bonus), max_obs_len=max_obs_len,
        )

        # Token embedding and positional encoding (Eq. 1-3).
        self.token_embed = nn.Embedding(self.vocab_size, d_model, padding_idx=self.vocab.pad_idx)
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self._pe = SinusoidalPositionalEncoding(d_model, max_len=max(max_seq_len, max_obs_len))

        # Encoder over the observed attack sequence (Eq. 4-6).
        if use_obs_encoder:
            self.obs_encoder = ObservedSequenceEncoder(
                self.token_embed,
                SinusoidalPositionalEncoding(d_model, max_len=max_obs_len),
                d_model, n_heads, num_layers, dim_feedforward, dropout, self.vocab.pad_idx)

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
                    window_mode=window_mode,
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

    # -- construction from a checkpoint ------------------------------------
    def arch_config(self) -> Dict[str, Any]:
        """Constructor kwargs; saved with every checkpoint as ckpt["arch"]."""
        return dict(self._arch)

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any], device="cpu") -> "DeepOPForecastDecoder":
        """Build the architecture a checkpoint was trained as, then load it.

        Checkpoints written before the encoder existed carry no "arch"; they
        are the LEGACY_ARCH decoder over the 12-D TGNE latent.
        """
        sd = ckpt["decoder_state_dict"]
        arch = ckpt.get("arch")
        if arch is None:
            arch = dict(cls.LEGACY_ARCH)
            arch["d_latent"] = int(sd["future_proj.weight"].shape[1])
            arch["d_model"] = int(sd["future_proj.weight"].shape[0])
            arch["vocab_size"] = int(sd["fc_out.weight"].shape[0])
            arch["max_seq_len"] = int(sd["pos_embed"].shape[1])
            arch["num_layers"] = len({k.split(".")[1] for k in sd if k.startswith("layers.")})
        model = cls(**arch).to(device)
        model.load_state_dict(sd)
        return model

    def _positions(self, seq_len: int) -> torch.Tensor:
        if self.pos_encoding == "learned":
            return self.pos_embed[:, :seq_len, :]
        return self._pe(seq_len)

    def forward(
        self,
        h_future: torch.Tensor,
        tgt_tokens: torch.Tensor,
        obs_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.
        Args:
            h_future:   [B, K, d_latent] Branch B's predicted future states
            tgt_tokens: [B, S] decoder input (<BOS> + previous targets)
            obs_tokens: [B, T_obs] observed technique sequence (Branch A over
                        the history window), PAD-left-padded. Ignored by a
                        legacy decoder.
        Returns:
            logits: [B, S, vocab_size]
        """
        B, S = tgt_tokens.shape
        x = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        x = x + self._positions(S)

        # Cross-attention memory: [encoded observed sequence ; future states].
        future_mem = self.future_proj(h_future)  # [B, K, d_model]
        memory, mem_pad = future_mem, None
        if self.use_obs_encoder and obs_tokens is not None:
            enc, enc_pad = self.obs_encoder(obs_tokens)
            memory = torch.cat([enc, future_mem], dim=1)
            mem_pad = torch.cat(
                [enc_pad, torch.zeros(B, future_mem.shape[1], dtype=torch.bool, device=enc_pad.device)],
                dim=1)

        for layer in self.layers:
            x = layer(x, memory=memory, memory_key_padding_mask=mem_pad)

        # Residual gating: anchor token representation directly to future world state
        memory = future_mem
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
        continuity_bonus: Optional[float] = None,
        observed_sequence: Optional[torch.Tensor] = None,
        decode_names: bool = True,
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
                `observed_token` (see CONTINUITY_BONUS_* above). None uses the
                model's default: 0.0 for the paper architecture (no hand-set
                prior), 1.0 for a legacy checkpoint, whose serving behaviour
                it reproduces.
            observed_sequence: [B, T_obs] observed technique tokens for the
                encoder (Branch A over the history window).
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
        if continuity_bonus is None:
            continuity_bonus = self.default_continuity_bonus

        # Start sequence cleanly with <BOS> token (prefix_len = 1) for 100% training/inference parity
        curr_tokens = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        prefix_len = 1

        all_attack_probs = [[] for _ in range(B)]
        all_token_probs = [[] for _ in range(B)]

        with torch.no_grad():
            for step in range(max_steps):
                logits = self.forward(h_future, curr_tokens, obs_tokens=observed_sequence)
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
        # `decode_names=False` skips the per-sample decode (one `.cpu()` sync
        # per sample) for callers that only use the tokens -- validation scores
        # 1.46M+ samples per epoch and discarded every name. The tokens are the
        # same either way; the names come back as None.
        if not decode_names:
            if return_probs:
                return pred_tokens, None, all_attack_probs, all_token_probs
            return pred_tokens, None
        for b in range(B):
            sample_tokens = []
            for t_idx in pred_tokens[b].cpu().numpy():
                sample_tokens.append(self.vocab.decode(int(t_idx)))
            decoded_names.append(sample_tokens)

        if return_probs:
            return pred_tokens, decoded_names, all_attack_probs, all_token_probs

        return pred_tokens, decoded_names


from cyberworld_v4.device_hist import device_hist  # noqa: E402

class DeepOPTokenScorer:
    """Streaming DeepOP token metrics with information-matched baselines.

    ## Why this exists

    The validation block in `scripts/retrain_future_models_live.py` scores a
    **teacher-forced** model against a **free-running** baseline:

        logits = decoder(h, batch["input_tokens"])   # position s is GIVEN y_{s-1}
        pred   = logits.argmax(-1)
        acc_persist = (obs.unsqueeze(1).expand_as(tgt) == tgt)   # obs = y_{-1}, repeated

    The model is handed the previous *true* token at every step; the baseline
    is handed one token and must hold it for the whole horizon. That is not a
    baseline, it is a handicap.

    Measured, on 200k synthetic 5-step sequences generated at the corpus's own
    statistics (val_label_churn 0.0524, val_positive_rate 0.175, so that
    free-running persistence reproduces the logged val_persistence_accuracy of
    0.8248 to within 0.0003):

        a model with ZERO parameters that simply echoes its own input token
            acc = 0.9165   macro_f1 = 0.8829
            persistence (as printed) = 0.8251   majority = 0.7366
            reported lift            = +0.0914      <-- "beats both baselines"

        the same-information baseline (repeat the token the model was given)
            acc = 0.9382

    So the printed lift is +0.0914 for a model that has no parameters and
    never looks at `h_future`, and the honest baseline is 0.9382, which that
    model does *not* reach. Per step the gap is entirely in the tail:

        step    persistence(obs)   persistence(previous target)
        0           0.9000              0.9000
        1           0.8583              0.9473
        2           0.8210              0.9479
        3           0.7881              0.9476
        4           0.7579              0.9480

    `acc_persistence_free` is the right baseline for `acc_free`; it is the
    wrong one for `acc_teacher_forced`, whose partner is
    `acc_persistence_fed`. This class reports all four so the pairing cannot
    be got wrong by accident.

    Deployment runs free-running (`forecast_sequence`, via
    control_backend/model_adapter.py), so `acc_free` is the number that
    describes production and `acc_teacher_forced` is a training diagnostic.

    Accumulates on-device; no per-sample Python and no host sync per batch.
    """

    def __init__(self, vocab_size: int, device="cpu"):
        self.V = int(vocab_size)
        self.device = device
        V = self.V
        self._conf_tf = torch.zeros(V * V, device=device, dtype=torch.long)
        self._conf_free = torch.zeros(V * V, device=device, dtype=torch.long)
        # free-running predictions at TRANSITIONS only: positions whose target
        # differs from the last observed token. Persistence scores 0 there by
        # construction, so this is the progression forecast itself.
        self._conf_trans = torch.zeros(V * V, device=device, dtype=torch.long)
        self._hit_tf = torch.zeros((), device=device, dtype=torch.long)
        self._hit_free = torch.zeros((), device=device, dtype=torch.long)
        self._hit_persist_free = torch.zeros((), device=device, dtype=torch.long)
        self._hit_persist_fed = torch.zeros((), device=device, dtype=torch.long)
        self._tgt_hist = torch.zeros(V, device=device, dtype=torch.long)
        self._n = 0
        self._have_free = False

    def update(self, target, input_tokens, obs_token, pred_tf=None, pred_free=None):
        """
        target:       [B, K] ground-truth tokens
        input_tokens: [B, K] what teacher forcing feeds -- [<BOS>, y_0 .. y_{K-2}]
        obs_token:    [B]    the last token observed BEFORE the horizon
        pred_tf:      [B, K] teacher-forced argmax, or None
        pred_free:    [B, K] free-running generation, or None
        """
        self.update_device(target, input_tokens, obs_token, pred_tf=pred_tf, pred_free=pred_free)
        self.update_host(target, has_free=pred_free is not None)

    def update_host(self, target, has_free: bool):
        """The host-side half of `update`: counts that depend on shapes only."""
        self._n += int(target.numel())
        if has_free:
            self._have_free = True

    def update_device(self, target, input_tokens, obs_token, pred_tf=None, pred_free=None):
        """The device-side half of `update`: in-place accumulator updates only,
        no host effect -- so it can be replayed as part of a CUDA graph."""
        V = self.V
        t = target.reshape(-1)
        self._tgt_hist += device_hist(t, V)      # bincount syncs on CUDA

        if pred_tf is not None:
            pf = pred_tf.reshape(-1)
            self._hit_tf += (pf == t).sum()
            self._conf_tf += device_hist(t * V + pf, V * V)
        obs = obs_token.reshape(-1)
        if pred_free is not None:
            pr = pred_free.reshape(-1)
            self._hit_free += (pr == t).sum()
            self._conf_free += device_hist(t * V + pr, V * V)
            changed = (target != obs.unsqueeze(1).expand_as(target)).reshape(-1)
            self._conf_trans += device_hist(t * V + pr, V * V, changed.long())

        # free-running persistence: one observation, held for the whole horizon
        self._hit_persist_free += (obs.unsqueeze(1).expand_as(target) == target).sum()
        # information-matched persistence: repeat whatever teacher forcing fed.
        # Slot 0 of input_tokens is <BOS>, which is not a legal output, so the
        # observed token stands in there -- that is exactly the information the
        # model has at step 0.
        fed = input_tokens.clone()
        fed[:, 0] = obs
        self._hit_persist_fed += (fed == target).sum()

    @staticmethod
    def _macro_f1(conf, V):
        cm = conf.reshape(V, V).cpu().numpy()
        sup, pred_n, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(pred_n > 0, tp / np.maximum(pred_n, 1), 0.0)
            rc = np.where(sup > 0, tp / np.maximum(sup, 1), 0.0)
            dn = pr + rc
            f1 = np.where(dn > 0, 2 * pr * rc / np.maximum(dn, 1e-12), 0.0)
        present = sup > 0
        # Macro-F1 is averaged over classes with SUPPORT only. Averaging over
        # the union of true and predicted classes -- what
        # sklearn's average="macro" does by default -- lets a single spurious
        # prediction into an absent class add a hard 0.0 to the mean. Of the
        # 10 joint tokens, only 4 occur anywhere in train (Benign.None,
        # C2.T1071, Impact.T1498, InitialAccess.T1190) and only 3 in val, so
        # that is 6-7 potential free zeros.
        return (float(f1[present].mean()) if present.any() else 0.0,
                f1, sup, pred_n, present)

    def result(self):
        n = max(self._n, 1)
        mf1_tf, f1_tf, sup, pred_tf_n, present = self._macro_f1(self._conf_tf, self.V)
        out = {
            "n_tokens": self._n,
            "acc_teacher_forced": float(self._hit_tf.item()) / n,
            "macro_f1_teacher_forced": mf1_tf,
            "acc_persistence_fed": float(self._hit_persist_fed.item()) / n,
            "acc_persistence_free": float(self._hit_persist_free.item()) / n,
            "acc_majority": float(self._tgt_hist.max().item()) / n,
            "classes_present": int(present.sum()),
            "classes_predicted_tf": int((pred_tf_n > 0).sum()),
            "target_histogram": self._tgt_hist.cpu().tolist(),
        }
        if self._have_free:
            mf1_free, _, _, pred_free_n, _ = self._macro_f1(self._conf_free, self.V)
            out["acc_free"] = float(self._hit_free.item()) / n
            out["macro_f1_free"] = mf1_free
            out["classes_predicted_free"] = int((pred_free_n > 0).sum())
            mf1_tr, _, sup_tr, _, _ = self._macro_f1(self._conf_trans, self.V)
            n_tr = int(sup_tr.sum())
            out["n_transitions"] = n_tr
            out["macro_f1_transitions"] = mf1_tr if n_tr else float("nan")
            out["acc_transitions"] = (float(np.trace(self._conf_trans.reshape(self.V, self.V)
                                                      .cpu().numpy())) / n_tr if n_tr else float("nan"))
        # The two comparisons that are actually like-for-like.
        out["lift_teacher_forced"] = out["acc_teacher_forced"] - max(
            out["acc_persistence_fed"], out["acc_majority"])
        if self._have_free:
            out["lift_free"] = out["acc_free"] - max(
                out["acc_persistence_free"], out["acc_majority"])
        return out

    @staticmethod
    def format(r):
        lines = [
            f"  tokens n={r['n_tokens']} classes_present={r['classes_present']}"
            f"/{len(r['target_histogram'])}",
            f"  TEACHER-FORCED  acc={r['acc_teacher_forced']:.4f} "
            f"macro_f1={r['macro_f1_teacher_forced']:.4f} "
            f"| matched baseline persistence_fed={r['acc_persistence_fed']:.4f} "
            f"majority={r['acc_majority']:.4f} | lift={r['lift_teacher_forced']:+.4f}"
            f"{'  <-- NO BETTER THAN A CONSTANT' if r['lift_teacher_forced'] <= 0 else ''}",
        ]
        if "acc_free" in r:
            lines.append(
                f"  FREE-RUNNING    acc={r['acc_free']:.4f} "
                f"macro_f1={r['macro_f1_free']:.4f} "
                f"| matched baseline persistence_free={r['acc_persistence_free']:.4f} "
                f"majority={r['acc_majority']:.4f} | lift={r['lift_free']:+.4f}"
                f"{'  <-- NO BETTER THAN A CONSTANT' if r['lift_free'] <= 0 else ''}")
        else:
            lines.append("  FREE-RUNNING    not measured -- this is the mode deployment uses")
        return "\n".join(lines)


def smoothed_and_plain_ce(logits, target, label_smoothing: float = 0.04, weight=None,
                          support=None, support_index=None):
    """Both losses from one forward pass, because only one of them is comparable.

    `train_deepop_live` optimises `cross_entropy(..., label_smoothing=0.04)`
    and validates with plain `cross_entropy(...)`, then prints the two side by
    side as `train_loss=` and `val_loss=`. They are different functions of the
    same predictions.

    PyTorch's smoothed loss is an exact identity, not an approximation:

        L_smooth = (1 - eps) * plain_CE + eps * U,    U = mean_c NLL(c)

    U is the mean negative log-probability over ALL classes, bounded below by
    ln(V) = ln(10) = 2.3026 and larger for any non-uniform prediction.
    Two consequences, both measured:

    1. Rigorous, assumption-free: eps*U >= 0.04 * 2.3026 = 0.0921, so the
       comparable plain CE behind the reported train 0.6644 is at most
       (0.6644 - 0.0921) / 0.96 = **0.5961**. The train-to-val degradation is
       therefore **at least 0.2087 nats**, not the 0.8048 - 0.6644 = 0.1404
       the printed pair shows.

    2. Empirical: sweeping logit distributions shaped like this problem
       (4 classes with support, 6 without, peak 1.0-4.0, confidence
       0.80-0.95) and keeping those whose SMOOTHED loss lands on 0.6644 +-0.03,
       the plain CE lies in **0.4092 .. 0.5567** and eps*U in 0.1502 .. 0.2614.
       The comparable degradation is then **0.248 .. 0.396 nats** -- 1.8x to
       2.8x the printed 0.1404.

    Note the sign: smoothing makes the TRAIN number larger, so correcting for
    it makes the gap WORSE, not better. "val is only 0.14 above train" was the
    flattering reading.

    `support_index` is `support.nonzero()` computed once by the caller (a
    non-empty LongTensor of class ids, ascending). Same columns, same order,
    same values as the boolean mask, without the two host-device syncs the
    mask costs on every call (`bool(sup.any())` and the masked select) -- which
    also makes the loss capturable in a CUDA graph. Takes precedence over
    `support`.

    Returns (loss_to_backprop, plain_ce_detached).
    """
    V = logits.shape[-1]
    flat, tgt = logits.reshape(-1, V), target.reshape(-1)
    if support_index is not None and weight is None:
        logp = F.log_softmax(flat, dim=-1)
        nll_t = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll_u = -logp.index_select(1, support_index).mean(dim=1)
        smoothed = ((1.0 - label_smoothing) * nll_t + label_smoothing * nll_u).mean()
    elif support is None or weight is not None:
        smoothed = F.cross_entropy(flat, tgt, weight=weight, label_smoothing=label_smoothing)
    else:
        # Smooth over the tokens that can actually occur, not the whole
        # vocabulary.
        #
        # PyTorch spreads eps uniformly over all V classes. Six of the ten
        # tokens here never occur as a target anywhere in the corpus --
        # <PAD>, <BOS>, <EOS>, CredentialAccess.T1110, Exfiltration.T1005,
        # Recon.T1595 -- so at eps=0.04 that is 6/10 * 0.04 = 2.4% of every
        # target's probability mass deliberately pushed onto answers that
        # are impossible. That is not regularisation, it is a loss floor.
        #
        # Same identity as PyTorch's, restricted to the support S:
        #     L = (1 - eps) * NLL(target) + eps * mean_{k in S} NLL(k)
        # With S = every class this is exactly F.cross_entropy(...,
        # label_smoothing=eps) -- pinned by a test.
        sup = torch.as_tensor(support, dtype=torch.bool, device=flat.device)
        if not bool(sup.any()):
            raise ValueError("label-smoothing support is empty")
        logp = F.log_softmax(flat, dim=-1)
        nll_t = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll_u = -logp[:, sup].mean(dim=1)
        smoothed = ((1.0 - label_smoothing) * nll_t + label_smoothing * nll_u).mean()
    with torch.no_grad():
        plain = F.cross_entropy(flat, tgt, weight=weight)
    return smoothed, plain


def evaluate_forecast_rigor(
    decoder: "DeepOPForecastDecoder",
    h_future: torch.Tensor,
    target_tokens: torch.Tensor,
    observed_tokens: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """Evaluate DeepOP with baselines matched to the decoding mode.

    Changes from the previous version, each for a stated reason:

    1. The persistence baseline is reported twice -- free-running (one
       observation held across the horizon) and information-matched (repeat
       the token teacher forcing supplied). The old code compared
       `teacher_forced_accuracy` against the free-running baseline only, which
       a zero-parameter echo model beats by +0.09 (see `DeepOPTokenScorer`).
    2. Free-running accuracy is reported with AND without the hard-coded
       step-0 continuity bonus. With it on, 71.9% of step-0 decisions are the
       bonus's rather than the model's (measured, untrained decoder, 4k
       samples), so the pair is the only honest way to present it.
    3. Macro-F1 averages over classes with support, not over the union of true
       and predicted classes; 6 of the 10 joint tokens never occur in this
       corpus and used to contribute free zeros the moment the model emitted
       one of them.
    4. Generation is batched. The old path ran `forecast_sequence` over the
       whole set at once with a per-sample Python loop inside; at the live val
       size (1.46M sequences x 5 steps) that is 7.3M Python iterations plus
       the whole set resident on the GPU.
    5. `observed_tokens` may be [B] or [B, 1]; the old `unsqueeze(1).repeat(1, K)`
       silently produced a [B, 1, K] tensor for the [B, 1] form that
       `forecast_sequence` documents as accepted.
    """
    decoder.eval()
    B, K = target_tokens.shape
    device = h_future.device
    V = decoder.vocab_size

    if observed_tokens is None:
        observed_tokens = torch.full((B,), decoder.vocab.bos_idx, dtype=torch.long, device=device)
    observed_tokens = observed_tokens.reshape(-1)

    bos = torch.full((B, 1), decoder.vocab.bos_idx, dtype=torch.long, device=device)
    teacher_input = torch.cat([bos, target_tokens[:, :-1]], dim=1)

    scorer = DeepOPTokenScorer(V, device=device)
    scorer_nobonus = DeepOPTokenScorer(V, device=device)

    with torch.no_grad():
        for lo in range(0, B, batch_size):
            hi = min(lo + batch_size, B)
            h_b = h_future[lo:hi]
            tgt_b = target_tokens[lo:hi]
            inp_b = teacher_input[lo:hi]
            obs_b = observed_tokens[lo:hi]

            pred_tf = decoder.forward(h_b, inp_b).argmax(dim=-1)
            free_b, _ = decoder.forecast_sequence(
                h_b, max_steps=K, observed_token=obs_b,
                repetition_penalty=repetition_penalty, continuity_bonus=1.0,
                decode_names=False)
            free_nb, _ = decoder.forecast_sequence(
                h_b, max_steps=K, observed_token=obs_b,
                repetition_penalty=repetition_penalty, continuity_bonus=0.0,
                decode_names=False)
            scorer.update(tgt_b, inp_b, obs_b, pred_tf=pred_tf, pred_free=free_b)
            scorer_nobonus.update(tgt_b, inp_b, obs_b, pred_tf=pred_tf, pred_free=free_nb)

    res = scorer.result()
    nb = scorer_nobonus.result()
    res["acc_free_no_continuity_bonus"] = nb["acc_free"]
    res["macro_f1_free_no_continuity_bonus"] = nb["macro_f1_free"]
    res["lift_free_no_continuity_bonus"] = nb["lift_free"]

    # Per-class breakdown over the classes that have support, plus any the
    # model emitted; each row says whether the class can occur at all.
    cm = scorer._conf_free.reshape(V, V).cpu().numpy()
    sup, pred_n, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    pad_idx = decoder.vocab.pad_idx
    benign_idx = decoder.vocab.encode("Benign", None)
    per_technique = {}
    active_f1s = []
    for c in range(V):
        if sup[c] == 0 and pred_n[c] == 0:
            continue
        pr = tp[c] / pred_n[c] if pred_n[c] else 0.0
        rc = tp[c] / sup[c] if sup[c] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        coarse, tech = decoder.vocab.decode(c)
        per_technique[f"{coarse}.{tech}" if tech else coarse] = {
            "token_id": c, "precision": float(pr), "recall": float(rc),
            "f1": float(f1), "support": int(sup[c]),
            "occurs_in_targets": bool(sup[c] > 0),
            "is_active": bool(c != pad_idx and c != benign_idx),
        }
        if sup[c] > 0 and c != pad_idx and c != benign_idx:
            active_f1s.append(f1)

    res["active_technique_macro_f1"] = float(np.mean(active_f1s)) if active_f1s else 0.0
    res["per_technique"] = per_technique
    # Backwards-compatible aliases for existing callers.
    res["free_running_accuracy"] = res["acc_free"]
    res["teacher_forced_accuracy"] = res["acc_teacher_forced"]
    res["free_running_macro_f1"] = res["macro_f1_free"]
    res["persistence_baseline_accuracy"] = res["acc_persistence_free"]
    return res
