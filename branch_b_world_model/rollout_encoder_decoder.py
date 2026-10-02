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
    that rollout travels from persistence, and shuffling them by 0.95%. The
    model was, to within ~1%, blind to whether a host's trajectory was rising
    or falling.

    Two changes:

    * callers now pass each position's elapsed time relative to the last
      OBSERVED step (negative in the past, positive for rollout steps), so the
      input varies along the sequence and the prediction position also encodes
      how far ahead it is;
    * a fixed bank of log-spaced sinusoids is added to the learned cosines.

    How much that is worth, measured rather than asserted. Ablation on a
    synthetic task with the property this architecture is supposed to exploit
    -- v_{t+1} = 0.85*v_t + eps, h_{t+1} = h_t + v_t, so the next step follows
    the recent trend and the model must know which history element is most
    recent. Identical init, identical 500 Adam steps, only the encoding
    differs; held-out persistence MSE 1.31805:

        V0  as shipped, constant dt          0.79040   skill +0.400
        V1  elapsed time -> learned cos      0.74554   skill +0.434
        V2  V1 + fixed sinusoids             0.74862   skill +0.432

    Read honestly, that says three things. Feeding elapsed time is worth 5.7%
    of MSE. The fixed bank is NOT measurably better than the learned cosines
    alone on this probe (0.4% apart on one seed, i.e. noise) -- it is kept for
    conditioning, not for that number: `cos(Linear(1, d_model))` initialises
    its frequencies from U(-1, 1) rad/s, so at the elapsed times this corpus
    actually produces (p90 gap 58 s, and hundreds of seconds where a host
    trajectory crosses captures) it aliases badly, whereas the fixed bank's
    periods of 4 s to 1024 s are injective over that range by construction.
    Pass fixed_sinusoids=False to drop it. And -- the important one -- even
    the broken V0 reached skill +0.400 on a task with real trend structure,
    so this defect is NOT what pinned Branch B to persistence. It is worth
    fixing; it is not the explanation.

    Neither branch adds parameters, so existing checkpoints still load.
    """

    def __init__(self, d_time: int, d_model: int,
                 min_period: float = 2.0 * LIVE_WINDOW_SIZE_SEC,
                 max_period: float = 512.0 * LIVE_WINDOW_SIZE_SEC,
                 fixed_sinusoids: bool = True):
        super().__init__()
        self.d_time = d_time
        self.d_model = d_model
        self.fixed_sinusoids = fixed_sinusoids
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
        learned = torch.cos(self.linear(t))
        if not self.fixed_sinusoids:
            return learned
        ang = t * self._omega
        fixed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if fixed.shape[-1] < self.d_model:      # odd d_model
            fixed = F.pad(fixed, (0, self.d_model - fixed.shape[-1]))
        return fixed + learned

class MultiHostInteractionLayer(nn.Module):
    """
    Spatio-temporal cross-host interaction layer.
    Allows communicating hosts to cross-attend to each other's predicted future states
    to model lateral movement and distributed multi-stage threat propagation.
    """

    def __init__(self, d_latent: int = 12, n_heads: int = 2, dropout: float = 0.1):
        super(MultiHostInteractionLayer, self).__init__()
        self.d_latent = d_latent
        # Attention runs directly on the state width, which must divide by the
        # head count. The 27-D world state does not divide by 2; use the
        # largest head count <= n_heads that fits (27 -> 1, 12 -> 2).
        heads = max(h for h in range(1, n_heads + 1) if d_latent % h == 0)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_latent, num_heads=heads, dropout=dropout, batch_first=True
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


#: Bounds on the predicted log-variance: exp(-9) ~ 1.2e-4 to exp(4) ~ 55, wide
#: against the [0, 1]-scaled state, and they keep the NLL finite early on.
LOGVAR_MIN, LOGVAR_MAX = -9.0, 4.0


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-element negative log-likelihood of `target` under
    N(mean, exp(logvar)), constant 0.5*log(2*pi) included."""
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar)
                  + 1.8378770664093453).mean()


class HostWorldDynamicsTransformer(nn.Module):
    """
    Causal autoregressive Transformer for per-host embedding rollouts.
    """

    # Class-level so the uncalibrated-radii error is logged once per process
    # rather than once per inference.
    _radii_warned = False

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
        # enable_nested_tensor=False: norm_first=True already disables the
        # nested-tensor path, so the default True only emits a UserWarning on
        # every construction. No behavioural change.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        # Output head predicts delta: Delta H = H_{t+1} - H_t
        self.out_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_latent),
        )

        # Predictive variance: log sigma^2 of each coordinate of H_{t+k}.
        #
        # The problem statement asks the world model for P(S_t+1 | S_t), a
        # DISTRIBUTION over the next state. out_head alone gives its mean --
        # a point forecast with no statement of how sure it is. This head
        # makes each step a diagonal Gaussian N(mean_k, exp(logvar_k)).
        #
        # It reads the transformer state DETACHED and is trained by Gaussian
        # NLL against the mean's actual residual (detached as well), so the
        # mean path -- and the persistence gate it is judged by -- trains
        # exactly as before; the variance only learns how wrong the mean is
        # at each horizon. Checkpoints without it still load (backfilled,
        # `variance_trained` False) and serve the mean unchanged.
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
        self.variance_trained = True

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

        # Conformal radii, set by calibrate_radii(). Deliberately a plain
        # attribute and not a registered buffer: a buffer would either enter
        # state_dict and break strict loads of every existing checkpoint, or
        # be non-persistent and silently vanish on save, which is worse than
        # being explicit that calibration does not travel with the weights.
        self._calibrated_radii: Optional[torch.Tensor] = None
        self.radii_norm_: Optional[str] = None
        self.radii_alpha_: Optional[float] = None
        self.radii_n_calibration_: int = 0

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
        missing_var = [k for k in self.state_dict() if k.startswith("logvar_head.") and k not in sd]
        if missing_var:
            # A checkpoint from before the variance head: its mean is all it
            # has. Backfill the fresh head so strict loading still works, and
            # say that its variances mean nothing.
            sd = dict(sd)
            for k in missing_var:
                sd[k] = self.state_dict()[k]
            self.variance_trained = False
        else:
            self.variance_trained = True
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
        B: int, ctx_len: int, k: int, delta_t_step: float, device: torch.device,
        t_history: Optional[torch.Tensor] = None,
        t_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Elapsed seconds of each context position, relative to the last
        OBSERVED step (t = 0). History is <= 0, rollout steps are > 0.

        At rollout step k the context window holds min(k, ctx_len) predicted
        states at its tail, so its last position carries t = +k*delta_t_step.
        That is what tells the shared out_head how far ahead it is being asked
        to predict -- and it is the only signal that distinguishes the horizon
        steps from one another. The hardcoded decay_factor damping was doing
        that job by hand, on a model that had no way to know which k it was on.

        `t_history` ([B, S], <= 0, last entry 0) supplies the REAL elapsed time
        of each observed step and `t_future` ([B, K], > 0) of each target. They
        matter: on wed_29_csv.csv the median gap between a host's consecutive
        snapshots is 7 windows (14 s), not 1, and the p90 is 29 (58 s), so a
        uniform 2 s grid misstates the spacing by about 7x. Both default to a
        uniform delta_t_step grid, which is what the old code assumed.
        """
        n_pred = min(int(k), int(ctx_len))
        n_obs = int(ctx_len) - n_pred

        if n_obs > 0:
            if t_history is not None:
                obs = t_history[:, -n_obs:].to(device=device, dtype=torch.float32)
            else:
                idx = torch.arange(n_obs, device=device, dtype=torch.float32)
                obs = ((idx - (n_obs - 1)) * float(delta_t_step)).unsqueeze(0).expand(B, -1)
        else:
            obs = torch.empty((B, 0), device=device, dtype=torch.float32)

        if n_pred > 0:
            # the context tail holds predicted steps k-n_pred+1 .. k
            lo = int(k) - n_pred
            if t_future is not None:
                pred = t_future[:, lo:int(k)].to(device=device, dtype=torch.float32)
            else:
                idx = torch.arange(lo + 1, int(k) + 1, device=device, dtype=torch.float32)
                pred = (idx * float(delta_t_step)).unsqueeze(0).expand(B, -1)
        else:
            pred = torch.empty((B, 0), device=device, dtype=torch.float32)

        return torch.cat([obs, pred], dim=1)

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
        # was a literal 4, against the contract's forecast_steps = 5. Every
        # caller passes K explicitly, so this only removes the trap.
        K: int = DEFAULT_ROLLOUT_HORIZON_LIVE,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        stabilize_horizon: bool = True,
        decay_factor: float = 0.95,
        max_context_len: int = None,   # None -> contract history_steps (15); 10 silently truncated it
        t_history: Optional[torch.Tensor] = None,
        t_future: Optional[torch.Tensor] = None,
        return_logvar: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive multi-step latent rollout predicting H_{t+1..t+K}.

        With `return_logvar=True` returns (mean [B, K, D], logvar [B, K, D]):
        the per-step diagonal Gaussian predictive distribution. The rollout
        itself always feeds back the MEAN.

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
            t_future: [batch_size, K] real elapsed seconds of each TARGET step
                (> 0). Defaults to a uniform grid.
        Returns:
            rollout_predictions: [batch_size, K, d_latent]
        """
        # A literal 10 here silently truncated the contract's 15-step history,
        # so the rollout never saw the 5 oldest steps it was trained to use.
        if max_context_len is None:
            from cyberworld_v4.config import get_contract
            max_context_len = get_contract().history_steps
        if t_history is not None and t_history.shape[-1] < h_seq.shape[1]:
            raise ValueError(
                f"t_history has {t_history.shape[-1]} steps but h_seq has "
                f"{h_seq.shape[1]}; one elapsed time per observed step is required")
        if t_future is not None and t_future.shape[-1] < K:
            raise ValueError(
                f"t_future has {t_future.shape[-1]} steps but K={K}")
        curr_seq = h_seq.clone()
        predictions = []
        logvars = []

        for k in range(K):
            context_window = curr_seq[:, -max_context_len:, :]
            B, T, _ = context_window.shape
            # Each position's elapsed time, not one constant repeated T times.
            dt = self._elapsed_times(B, T, k, delta_t_step, h_seq.device,
                                     t_history, t_future)

            x = self.in_proj(context_window)
            t_emb = self.time_encoder(dt, is_delta=True)
            x = x + t_emb
            causal_mask = self._generate_causal_mask(T, h_seq.device)
            h_trans = self.transformer(x, mask=causal_mask, is_causal=True)

            delta_h = self.out_head(h_trans[:, -1, :])
            if return_logvar:
                logvars.append(self.logvar_head(h_trans[:, -1, :].detach())
                               .clamp(LOGVAR_MIN, LOGVAR_MAX).unsqueeze(1))

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

        if return_logvar:
            return torch.cat(predictions, dim=1), torch.cat(logvars, dim=1)
        return torch.cat(predictions, dim=1)  # [B, K, d_latent]

    def calibrate_radii(
        self,
        residuals: "np.ndarray | torch.Tensor",
        alpha: float = 0.05,
        norm: str = "l2",
    ) -> torch.Tensor:
        """Fits per-step conformal radii from held-out rollout residuals.

        Args:
            residuals: [N, K, d_latent] y_true - y_pred on a CALIBRATION split
                that was used for neither training nor model selection.
            alpha: miscoverage rate; 0.05 gives a 95% radius.
            norm: "l2" for a radius on ||residual||_2 over the latent
                (a ball around the predicted state), or "linf" for a
                per-coordinate box half-width.

        Uses cyberworld_v4.conformal.conformal_quantile, i.e. the
        ceil((n+1)(1-alpha))/n order statistic, which is what makes the
        coverage finite-sample rather than asymptotic. Stored on the instance
        and returned; `rollout_with_uncertainty` uses it from then on.

        The result is a plain attribute, so it does NOT travel inside
        `state_dict()`. Whatever saves the model must save the tensor,
        `radii_norm_` and `radii_alpha_` alongside the weights and re-apply
        them after load -- calibration belongs to a (model, split, alpha)
        triple, not to the weights, and pretending otherwise is how the old
        hardcoded table came to look official.
        """
        from cyberworld_v4.conformal import conformal_quantile
        r = residuals.detach().cpu().numpy() if torch.is_tensor(residuals) else np.asarray(residuals)
        if r.ndim != 3:
            raise ValueError(f"residuals must be [N, K, d_latent], got {r.shape}")
        if norm == "l2":
            scores = np.linalg.norm(r, axis=-1)          # [N, K]
        elif norm == "linf":
            scores = np.abs(r).max(axis=-1)              # [N, K]
        else:
            raise ValueError(f"norm must be 'l2' or 'linf', got {norm!r}")
        radii = [conformal_quantile(scores[:, k], alpha) for k in range(scores.shape[1])]
        self._calibrated_radii = torch.tensor(radii, dtype=torch.float32)
        self.radii_norm_ = norm
        self.radii_alpha_ = float(alpha)
        self.radii_n_calibration_ = int(scores.shape[0])
        return self._calibrated_radii

    def rollout_with_uncertainty(
        self,
        h_seq: torch.Tensor,
        K: int = DEFAULT_ROLLOUT_HORIZON_LIVE,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        stabilize_horizon: bool = True,
        empirical_radii: Optional[List[float]] = None,
        t_history: Optional[torch.Tensor] = None,
        t_future: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step latent rollout, plus a per-step uncertainty radius.

        The radius is NaN unless someone has actually calibrated it -- either
        by passing `empirical_radii` or by calling `calibrate_radii()` on a
        held-out split first.

        This used to return a hardcoded table presented as "95th-percentile
        empirical residual radii":

            [0.01890, 0.03661, 0.05327, 0.06904]

        extended past step 4 by `base[-1] + 0.015*sqrt(step-3)`. Three things
        are wrong with that, and they are why it is gone rather than merely
        documented:

        1. Nothing in this repository produces those numbers, and no run,
           split, alpha or norm is recorded for them. A number with no
           provenance is not a measurement.
        2. The table and its extrapolation do not even agree with each other.
           The four base values grow almost linearly (increments 0.01890,
           0.01771, 0.01666, 0.01577 -- gently decreasing, i.e. r ~ k^0.93).
           The extrapolation grows as sqrt, and its first application adds
           0.02121 at step 5 -- a jump LARGER than the first increment, into
           a sequence whose increments are shrinking -- and then adds only
           0.00477, 0.00402, 0.00354 at steps 6, 7, 8. No single fit produces
           that shape.
        3. They are the wrong order of magnitude. Measured: the shipped
           checkpoint's OWN residuals, run under the rollout it was trained
           with, over 103,110 rollout samples from the validation capture
           wed_29_csv.csv, have per-element MSE 0.066892 at k=1 rising to
           0.108372 at k=5. That is an RMS of 0.259 to 0.329 per coordinate,
           or an L2 norm of 0.896 to 1.140 over the 12-D residual vector.
           Against a table reading 0.01890 to 0.09025, that is 3.6x-13.7x too
           small read per-coordinate and 12.6x-47x too small read as an L2
           radius -- and those are RMS figures, so the 95th percentile the
           table claims to be is larger still. A "95% radius" that covers a
           few percent of the mass is worse than no radius, because it is
           believed.

        Returns:
            rollout_predictions: [batch_size, K, d_latent]
            radii: [K]; NaN when uncalibrated. `self.radii_norm_` says what
                the radius is a radius IN -- the old table never did.
        """
        h_future = self.rollout(h_seq, K=K, delta_t_step=delta_t_step,
                                stabilize_horizon=stabilize_horizon,
                                t_history=t_history, t_future=t_future)
        if empirical_radii is not None:
            radii = torch.as_tensor(list(empirical_radii), dtype=torch.float32,
                                    device=h_seq.device)
        elif self._calibrated_radii is not None and len(self._calibrated_radii) >= K:
            radii = self._calibrated_radii[:K].to(h_seq.device)
        else:
            if not HostWorldDynamicsTransformer._radii_warned:
                HostWorldDynamicsTransformer._radii_warned = True
                logger.error(
                    "rollout_with_uncertainty(): no calibrated radii. Returning "
                    "NaN. Call calibrate_radii(residuals) on a held-out split, "
                    "or pass empirical_radii=. The previous hardcoded table was "
                    "3-19x too small and had no provenance; a NaN you must "
                    "handle is safer than a number you would plot.")
            radii = torch.full((K,), float("nan"), dtype=torch.float32, device=h_seq.device)
        return h_future, radii

    def rollout_multi_host(
        self,
        h_seq_batch: torch.Tensor,               # [N_hosts, seq_len, d_latent]
        active_edges: List[Tuple[int, int]],    # Active interacting host pairs (u_idx, v_idx)
        K: int = DEFAULT_ROLLOUT_HORIZON_LIVE,
        delta_t_step: float = LIVE_WINDOW_SIZE_SEC,
        lateral_weight: float = 0.35,
        stabilize_horizon: bool = True,
        decay_factor: float = 0.95,
        max_context_len: int = None,   # None -> contract history_steps (15); 10 silently truncated it
        allow_untrained_lateral: bool = False,
    ) -> torch.Tensor:
        """
        Multi-host coupled autoregressive rollout predicting H_{t+1..t+K} across the network topology.

        DEAD CODE, AND UNSAFE TO REVIVE AS IS. Nothing in this repository calls
        it: training calls rollout() (scripts/retrain_future_models_live.py),
        serving calls rollout_with_uncertainty()
        (control_backend/model_adapter.py), and no test exercises it. So
        `self.lateral_interaction` -- 948 parameters, 0.8% of the checkpoint --
        has never received a gradient. Verified by reading both shipped
        checkpoints: cross_attn.in_proj_weight rms 0.2021 and 0.1993 against a
        fresh init's 0.2051, i.e. untouched. (The trainer calls
        zero_grad(set_to_none=True), so Adam skips them entirely; not even
        weight decay reaches them.)

        Calling this therefore adds `lateral_weight * gate(random) *
        attention(random)` -- with lateral_weight itself an unsourced 0.35 --
        to every predicted host state. That is noise injected into a forecast,
        and it would look like modelled lateral movement. It now raises rather
        than doing that quietly.

        To use it: include the lateral path in the training objective, then
        pass allow_untrained_lateral=True only while deliberately testing the
        untrained path.

        Args:
            h_seq_batch: [N_hosts, seq_len, d_latent] tensor of host histories
            active_edges: list of (u_idx, v_idx) pairs indicating communication interaction
            K: rollout horizon
            delta_t_step: temporal step delta
            lateral_weight: cross-host interaction weight
            stabilize_horizon: whether to damp multi-step delta compounding
            decay_factor: exponential decay factor
            max_context_len: maximum attention context length
            allow_untrained_lateral: opt in to running the untrained layer
        Returns:
            rollout_predictions: [N_hosts, K, d_latent]
        """
        if active_edges and h_seq_batch.shape[0] > 1 and not allow_untrained_lateral:
            raise RuntimeError(
                "rollout_multi_host() would run MultiHostInteractionLayer, whose "
                "parameters no training path in this repository ever updates -- "
                "they are still at random initialisation. Train the lateral path "
                "first, or pass allow_untrained_lateral=True if you are "
                "deliberately measuring the untrained layer.")
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

