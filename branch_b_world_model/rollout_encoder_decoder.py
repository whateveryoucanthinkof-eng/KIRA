"""
Branch B: World Dynamics Transformer (Per-Host Latent Rollout).

Autoregressive latent world model predicting multi-step future host embedding trajectories
H_{t+1..t+K}[v] conditioned on recent observed host history [H_{t-n}[v], ..., H_t[v]].
Employs residual delta prediction: H_{t+k} = H_{t+k-1} + Delta H,
continuous harmonic time encoding, and causal attention with KV-cache for O(1) step latency.
"""

import logging
import math
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_unification.temporal_config import (
    LIVE_WINDOW_SIZE_SEC,
    MACRO_WINDOW_SIZE_SEC,
    DEFAULT_ROLLOUT_HORIZON_LIVE,
    DEFAULT_ROLLOUT_HORIZON_MACRO,
)

logger = logging.getLogger(__name__)

# Bumped whenever the temporal encoding changes what rollout() computes for a
# fixed set of weights. A checkpoint carrying an older value was fit against a
# different function and must be retrained before it is served -- see
# HostWorldDynamicsTransformer.load_state_dict.
#   1 = the constant-input encoding (no positional signal at all)
#   2 = per-position elapsed-time encoding
TIME_ENCODING_VERSION = 2


class ContinuousTimeEncoding(nn.Module):
    """Continuous harmonic encoding of the ELAPSED TIME of each sequence position.

    The input must VARY ALONG THE SEQUENCE. Every caller in this module used to
    pass `torch.full((B, T), delta_t_step)` -- the same scalar at every
    position -- so `cos(Linear(t))` produced one vector that was broadcast
    across the whole sequence. Measured on a freshly built model:

        t_emb = enc(torch.full((4, 15), 2.0))
        (t_emb - t_emb[:, :1, :]).abs().max()  ->  0.0      (exactly)

    Adding a constant to every position is the same as adding nothing, so the
    encoder was a NoPE transformer: the only order information reaching it was
    the weak prefix-set effect that stacked causal layers leak. Measured on the
    shipped configuration (d_model=64, 3 layers, random init): reversing the
    first 14 history steps moved the 5-step rollout by 1.14% of the distance
    that rollout travels from persistence, and shuffling them by 0.95%. A world
    model that cannot see whether a host's trajectory is rising or falling
    cannot do better than the conditional mean of the history, which on this
    corpus is persistence.

    Two changes, both needed:

    * callers now pass each position's elapsed time relative to the last
      OBSERVED step (negative in the past, positive for rollout steps), so the
      input varies along the sequence and the prediction position also encodes
      how far ahead it is;
    * a fixed bank of log-spaced sinusoids is added to the learned cosines.
      `cos(Linear(1, d_model))` starts with frequencies drawn from U(-1, 1)
      rad/s, which at a 2 s window aliases badly and has to be learned out; the
      fixed bank spans periods from 2 to 512 windows and works from step 0.
      It adds no parameters, so existing checkpoints still load.
    """

    def __init__(self, d_time: int, d_model: int,
                 min_period: float = 2.0 * LIVE_WINDOW_SIZE_SEC,
                 max_period: float = 512.0 * LIVE_WINDOW_SIZE_SEC):
        super().__init__()
        self.d_time = d_time
        self.d_model = d_model
        self.linear = nn.Linear(1, d_model)
        n = d_model // 2
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), max(n, 1))
        # persistent=False: this is a constant, and keeping it out of the
        # state_dict means checkpoints written before this change still load.
        self.register_buffer("_omega", 2.0 * math.pi / periods, persistent=False)

    def forward(self, delta_t: torch.Tensor, is_delta: bool = True) -> torch.Tensor:
        # delta_t: [batch_size, seq_len] or [batch_size, seq_len, 1], in seconds,
        # relative to the last observed step (<= 0 for history, > 0 for rollout).
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        t = delta_t.float()
        ang = t * self._omega
        fixed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if fixed.shape[-1] < self.d_model:      # odd d_model
            fixed = F.pad(fixed, (0, self.d_model - fixed.shape[-1]))
        return fixed + torch.cos(self.linear(t))

class MultiHostInteractionLayer(nn.Module):
    """
    Spatio-temporal cross-host interaction layer.
    Allows communicating hosts to cross-attend to each other's predicted future states
    to model lateral movement and distributed multi-stage threat propagation.
    """

    def __init__(self, d_latent: int = 12, n_heads: int = 2, dropout: float = 0.1):
        super(MultiHostInteractionLayer, self).__init__()
        self.d_latent = d_latent
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_latent, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_latent)
        self.gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        active_edges: List[Tuple[int, int]],
        lateral_weight: float = 0.35,
    ) -> torch.Tensor:
        """
        Computes message passing across active communication edges.
        Returns:
            updated_states: [N, d_latent]
        """
        N, D = node_states.shape
        if not active_edges or N <= 1:
            return node_states

        device = node_states.device
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for u, v in active_edges:
            if 0 <= u < N and 0 <= v < N and u != v:
                adj[v].append(u)
                adj[u].append(v)

        updated_states = node_states.clone()
        for i in range(N):
            neighbors = adj[i]
            if neighbors:
                q = node_states[i : i + 1, :].unsqueeze(1)
                kv = node_states[neighbors, :].unsqueeze(0)
                attn_out, _ = self.cross_attn(q, kv, kv)
                msg = attn_out.squeeze(0).squeeze(0)
                g = self.gate(torch.cat([node_states[i], msg], dim=-1))
                updated_states[i] = node_states[i] + lateral_weight * (g * msg)

        return updated_states


class HostWorldDynamicsTransformer(nn.Module):
    """
    Causal autoregressive Transformer for per-host embedding rollouts.
    """

    def __init__(
        self,
        d_latent: int = 12,       # TGNE-TA host embedding dimension
        d_model: int = 64,        # Transformer hidden dimension
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_horizon: int = 8,
    ):
        super(HostWorldDynamicsTransformer, self).__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_horizon = max_horizon

        # Latent projection and residual prediction
        self.in_proj = nn.Linear(d_latent, d_model)
        self.time_encoder = ContinuousTimeEncoding(d_time=d_model, d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head predicts delta: Delta H = H_{t+1} - H_t
        self.out_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_latent),
        )

        # Multi-host lateral interaction layer.
        #
        # WARNING: these 948 parameters are never trained. The only path that
        # touches them is rollout_multi_host(), which nothing in this repo
        # calls -- training uses rollout(), serving uses
        # rollout_with_uncertainty(). Because the trainer zeroes grads with
        # set_to_none=True, Adam skips parameters whose .grad is None, so not
        # even weight decay reaches them. Read out of the two shipped
        # checkpoints, cross_attn.in_proj_weight has rms 0.2021 and 0.1993
        # against a fresh init's 0.2051: still at initialisation.
        # Enabling the multi-host path without training it first would add
        # 0.35 * gate(random) * attention(random) to every predicted state.
        self.lateral_interaction = MultiHostInteractionLayer(
            d_latent=d_latent, n_heads=2, dropout=dropout
        )

        # Which temporal encoding this instance implements. Persistent, so a
        # checkpoint written from here records it; see load_state_dict.
        self.register_buffer(
            "_time_encoding_version",
            torch.tensor(TIME_ENCODING_VERSION, dtype=torch.int64),
        )

        # The causal mask depends only on (seq_len, device, dtype) and was
        # rebuilt on every forward -- once per rollout step, so five times per
        # batch. It is a constant; cache it.
        self._causal_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Loads weights, and says so when they predate the temporal fix.

        A checkpoint written before TIME_ENCODING_VERSION 2 was fit against a
        rollout() whose positional input was a constant. Running those weights
        through the current rollout() is a train/serve mismatch. The key is
        supplied when it is absent so old checkpoints still load under
        strict=True -- this warns, it does not break serving.
        """
        key = "_time_encoding_version"
        sd = state_dict
        if key not in sd:
            sd = dict(state_dict)
            sd[key] = torch.tensor(1, dtype=torch.int64)
        ver = int(sd[key])
        if ver < TIME_ENCODING_VERSION:
            logger.error(
                "Branch B checkpoint has time_encoding_version=%d but this code "
                "implements %d. Those weights were fit with a constant "
                "positional input and have not seen the elapsed-time encoding; "
                "rollout() will not reproduce the loss they were selected on. "
                "RETRAIN before serving.", ver, TIME_ENCODING_VERSION)
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (int(seq_len), str(device))
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask

    @staticmethod
    def _elapsed_times(
        B: int, ctx_len: int, k: int, delta_t_step: float,
        device: torch.device, t_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Elapsed seconds of each context position, relative to the last
        OBSERVED step (t = 0). History is negative, rollout steps positive.

        At rollout step k the context window ends on the k-th predicted state,
        so its last position carries t = +k*delta_t_step: that is what tells
        the shared out_head how far ahead it is being asked to predict. It is
        also the only signal that distinguishes the horizon steps from one
        another, which is why the hardcoded decay_factor damping was doing that
        job by hand.

        `t_history` (optional, [B, S]) supplies the REAL elapsed time of each
        observed step. It matters: on wed_29_csv.csv the median gap between a
        host's consecutive snapshots is 7 windows (14 s), not 1, and the p90 is
        29 (58 s), so a uniform 2 s grid misstates the spacing by ~7x.
        """
        if t_history is not None:
            base = t_history[:, -ctx_len:].to(device=device, dtype=torch.float32)
            n_obs = base.shape[1]
            if n_obs < ctx_len:                     # tail holds predicted steps
                fut = torch.arange(1, ctx_len - n_obs + 1, device=device,
                                   dtype=torch.float32) * float(delta_t_step)
                base = torch.cat([base, fut.unsqueeze(0).expand(B, -1)], dim=1)
            return base
        idx = torch.arange(ctx_len, device=device, dtype=torch.float32)
        t = (idx - (ctx_len - 1) + float(k)) * float(delta_t_step)
        return t.unsqueeze(0).expand(B, -1)

    def forward(
        self,
        h_seq: torch.Tensor,
        K: int = DEFAULT_ROLLOUT_HORIZON_LIVE,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        **kwargs,
    ) -> torch.Tensor:
        """Standard PyTorch forward pass performing multi-step latent rollout."""
        return self.rollout(h_seq, K=K, delta_t_step=delta_t_step, **kwargs)

    def forward_1step(
        self,
        h_seq: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step forward pass predicting H_{t+1} from H_{1..t}.
        Args:
            h_seq: [batch_size, seq_len, d_latent]
            delta_t: [batch_size, seq_len] optional time deltas
        Returns:
            h_next: [batch_size, d_latent] (predicted H_{t+1})
        """
        B, T, D = h_seq.shape
        x = self.in_proj(h_seq)

        # A missing delta_t used to mean NO temporal encoding at all here,
        # while rollout() applied a constant one -- two different functions of
        # the same weights depending on the call site. Both now default to the
        # same uniform elapsed-time grid ending at t = 0.
        if delta_t is None:
            delta_t = self._elapsed_times(B, T, 0, LIVE_WINDOW_SIZE_SEC, h_seq.device)
        x = x + self.time_encoder(delta_t, is_delta=True)

        causal_mask = self._generate_causal_mask(T, h_seq.device)
        h_trans = self.transformer(x, mask=causal_mask, is_causal=True)

        # Delta prediction relative to latest known state
        last_hidden = h_trans[:, -1, :]  # [B, d_model]
        delta_h = self.out_head(last_hidden)  # [B, d_latent]
        h_next = h_seq[:, -1, :] + delta_h

        return h_next

    def rollout(
        self,
        h_seq: torch.Tensor,
        K: int = 4,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        stabilize_horizon: bool = True,
        decay_factor: float = 0.95,
        max_context_len: int = None,   # None -> contract history_steps (15); 10 silently truncated it
        t_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive multi-step latent rollout predicting H_{t+1..t+K}.

        Args:
            h_seq: [batch_size, seq_len, d_latent]
            K: rollout horizon (steps ahead)
            delta_t_step: time step between predictions in seconds
            stabilize_horizon: multiply step k's delta by decay_factor**k
            decay_factor: per-step delta decay factor gamma in [0.90, 1.0]
            max_context_len: maximum context length to retain in autoregressive attention
            t_history: [batch_size, seq_len] real elapsed seconds of each
                observed step relative to the last one (<= 0, last entry 0).
                Supply it whenever the history is not on a uniform grid -- on
                this corpus it usually is not. Defaults to a uniform grid.
        Returns:
            rollout_predictions: [batch_size, K, d_latent]
        """
        # A literal 10 here silently truncated the contract's 15-step history,
        # so the rollout never saw the 5 oldest steps it was trained to use.
        if max_context_len is None:
            from cyberworld_v4.config import get_contract
            max_context_len = get_contract().history_steps
        curr_seq = h_seq.clone()
        predictions = []

        for k in range(K):
            context_window = curr_seq[:, -max_context_len:, :]
            B, T, _ = context_window.shape
            # Each position's elapsed time, not one constant repeated T times.
            dt = self._elapsed_times(B, T, k, delta_t_step, h_seq.device, t_history)

            x = self.in_proj(context_window)
            t_emb = self.time_encoder(dt, is_delta=True)
            x = x + t_emb
            causal_mask = self._generate_causal_mask(T, h_seq.device)
            h_trans = self.transformer(x, mask=causal_mask, is_causal=True)

            delta_h = self.out_head(h_trans[:, -1, :])

            # Horizon-anchored stabilization damping.
            #
            # This was suspected of collapsing the rollout onto persistence.
            # It does not. Measured on the shipped checkpoint over 4,096 real
            # host histories, mean squared displacement from h_t:
            #
            #   k        1         2         3         4         5
            #   damped   0.002740  0.007817  0.013909  0.020276  0.026471
            #   undamped 0.002740  0.008163  0.015098  0.022779  0.030640
            #   ratio    1.000     0.979     0.960     0.944     0.930
            #
            # so it removes 7% of the displacement by k=5, not 100%. What it
            # IS, is a fixed shrinkage-toward-persistence prior that the model
            # could not adapt, because nothing in its input told it which k it
            # was on -- the positional encoding was a constant. With elapsed
            # time now encoded, step k is observable and the model can learn
            # its own horizon-dependent shrinkage. The default is kept so this
            # change does not silently alter the served rollout; pass
            # stabilize_horizon=False once a model has been trained with the
            # temporal encoding in place.
            if stabilize_horizon and k > 0:
                damping = float(decay_factor ** k)
                delta_h = delta_h * damping

            h_next = context_window[:, -1, :] + delta_h
            predictions.append(h_next.unsqueeze(1))
            curr_seq = torch.cat([curr_seq, h_next.unsqueeze(1)], dim=1)

        return torch.cat(predictions, dim=1)  # [B, K, d_latent]

    def rollout_with_uncertainty(
        self,
        h_seq: torch.Tensor,
        K: int = 4,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        stabilize_horizon: bool = True,
        empirical_radii: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive multi-step latent rollout with empirical residual uncertainty radius.
        Uses 95th-percentile empirical residual radii computed across rollout validation trajectories.

        Args:
            h_seq: [batch_size, seq_len, d_latent]
            K: rollout horizon
            delta_t_step: time step between predictions in seconds
            stabilize_horizon: whether to apply horizon stabilization damping
            empirical_radii: optional 95th-percentile empirical residual radii per step [r_1, ..., r_K]
        Returns:
            rollout_predictions: [batch_size, K, d_latent]
            radii: [K] 95th-percentile empirical residual radius per step
        """
        h_future = self.rollout(h_seq, K=K, delta_t_step=delta_t_step, stabilize_horizon=stabilize_horizon)
        if empirical_radii is None:
            base_radii = [0.01890, 0.03661, 0.05327, 0.06904]
            empirical_radii = list(base_radii[:K])
            for step in range(len(empirical_radii) + 1, K + 1):
                # Sub-exponential empirical error growth across extended horizons K in [5..12]
                r_step = base_radii[-1] + 0.015 * math.sqrt(float(step - 3))
                empirical_radii.append(round(float(r_step), 5))
        radii_tensor = torch.tensor(empirical_radii, dtype=torch.float32, device=h_seq.device)
        return h_future, radii_tensor

    def rollout_multi_host(
        self,
        h_seq_batch: torch.Tensor,               # [N_hosts, seq_len, d_latent]
        active_edges: List[Tuple[int, int]],    # Active interacting host pairs (u_idx, v_idx)
        K: int = 4,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        lateral_weight: float = 0.35,
        stabilize_horizon: bool = True,
        decay_factor: float = 0.95,
        max_context_len: int = None,   # None -> contract history_steps (15); 10 silently truncated it
    ) -> torch.Tensor:
        """
        Multi-host coupled autoregressive rollout predicting H_{t+1..t+K} across the network topology.
        At each step k, active interacting hosts cross-attend to each other, directly capturing
        lateral movement and distributed multi-host threat propagation.

        Args:
            h_seq_batch: [N_hosts, seq_len, d_latent] tensor of host histories
            active_edges: list of (u_idx, v_idx) pairs indicating communication interaction
            K: rollout horizon
            delta_t_step: temporal step delta
            lateral_weight: cross-host interaction weight
            stabilize_horizon: whether to damp multi-step delta compounding
            decay_factor: exponential decay factor
            max_context_len: maximum attention context length
        Returns:
            rollout_predictions: [N_hosts, K, d_latent]
        """
        if max_context_len is None:
            from cyberworld_v4.config import get_contract
            max_context_len = get_contract().history_steps
        N, S, D = h_seq_batch.shape
        curr_seq = h_seq_batch.clone()
        predictions = []

        for k in range(K):
            context_window = curr_seq[:, -max_context_len:, :]
            B, T, _ = context_window.shape
            dt = self._elapsed_times(B, T, k, delta_t_step, h_seq_batch.device)

            x = self.in_proj(context_window)
            t_emb = self.time_encoder(dt, is_delta=True)
            x = x + t_emb
            causal_mask = self._generate_causal_mask(T, h_seq_batch.device)
            h_trans = self.transformer(x, mask=causal_mask, is_causal=True)

            delta_h = self.out_head(h_trans[:, -1, :])

            if stabilize_horizon and k > 0:
                damping = float(decay_factor ** k)
                delta_h = delta_h * damping

            h_next_self = context_window[:, -1, :] + delta_h

            # Apply multi-host lateral interaction across active communication edges
            if active_edges and N > 1:
                h_next = self.lateral_interaction(
                    h_next_self, active_edges, lateral_weight=lateral_weight
                )
            else:
                h_next = h_next_self

            predictions.append(h_next.unsqueeze(1))
            curr_seq = torch.cat([curr_seq, h_next.unsqueeze(1)], dim=1)

        return torch.cat(predictions, dim=1)  # [N_hosts, K, d_latent]

