"""
Attack Trajectory Assembler.

Stitches Branch A (observed near-term technique predictions) and
Branch B + DeepOP (forecast future technique rollouts + risk curve)
into unified per-host timelines with strict provenance tagging.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
# The assembler feeds the real Branch A / Branch B / DeepOP checkpoints, so its
# defaults must be the contract's, not the unsupported 60s MACRO granularity it
# used to default to (K=4, 60s windows, 10/5 history -- all four wrong).
from cyberworld_v4.config import get_contract

_CONTRACT = get_contract()


class Provenance(str, Enum):
    OBSERVED = "OBSERVED"
    FORECAST = "FORECAST"


@dataclass
class TrajectoryEntry:
    """A single discrete step in a host's attack trajectory."""
    host_ip: str
    window_idx: int
    timestamp: float
    provenance: str  # "OBSERVED" or "FORECAST"
    coarse_category: str
    technique_id: str
    confidence: float
    risk_score: float  # Infiltration / compromise probability in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HostAttackTrajectory:
    """Full assembled timeline for a host with observed past and forecast future."""
    host_ip: str
    observed_entries: List[TrajectoryEntry]
    forecast_entries: List[TrajectoryEntry]
    cumulative_forecast_risk: float  # R_cumul over the rollout horizon

    @property
    def all_entries(self) -> List[TrajectoryEntry]:
        return self.observed_entries + self.forecast_entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_ip": self.host_ip,
            "cumulative_forecast_risk": self.cumulative_forecast_risk,
            "observed_entries": [e.to_dict() for e in self.observed_entries],
            "forecast_entries": [e.to_dict() for e in self.forecast_entries],
        }


class AttackTrajectoryAssembler:
    """
    Coordinates Branch A, Branch B (WDT), and DeepOP CWA Decoder to assemble
    complete hybrid attack trajectories.
    """

    def __init__(
        self,
        branch_a_model,
        wdt_model,
        risk_head,
        deepop_decoder,
        device: str = "cpu",
        risk_target: str = "severity",
    ):
        self.branch_a = branch_a_model.to(device)
        self.wdt = wdt_model.to(device)
        self.risk_head = risk_head.to(device)
        self.deepop = deepop_decoder.to(device)
        self.device = device
        #: Which target Branch B's risk head was trained against. Decides how
        #: per-step risks are reduced to one number -- see the comment at the
        #: reduction site. Defaults to "severity", the current training
        #: default, so behaviour is unchanged unless a hazard-trained
        #: checkpoint says otherwise.
        if risk_target not in ("severity", "hazard"):
            raise ValueError(f"risk_target must be 'severity' or 'hazard', got {risk_target!r}")
        self.risk_target = risk_target

        self.branch_a.eval()
        self.wdt.eval()
        self.risk_head.eval()
        self.deepop.eval()

    def assemble_trajectories_batch(
        self,
        host_trajectories: Dict[str, list],
        batch_size: int = 128,
        K: int = None,
        window_size_sec: float = None,
        max_seq_len_a: int = None,
        max_seq_len_wdt: int = None,
        lateral_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, HostAttackTrajectory]:
        """
        Batched tensor vectorization for dual-branch trajectory assembly across multiple hosts.
        Replaces O(N) sequential GPU kernel launches with batched minibatches (e.g. B=128),
        achieving up to ~89x throughput acceleration.
        """
        from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB

        # Resolve against the contract. Branch A is trained with
        # seq_len=history_steps (15) and Branch B with the same history, so the
        # previous literals silently truncated both at serving time.
        K = _CONTRACT.forecast_steps if K is None else K
        window_size_sec = _CONTRACT.window_seconds if window_size_sec is None else window_size_sec
        max_seq_len_a = _CONTRACT.history_steps if max_seq_len_a is None else max_seq_len_a
        max_seq_len_wdt = _CONTRACT.history_steps if max_seq_len_wdt is None else max_seq_len_wdt

        results: Dict[str, HostAttackTrajectory] = {}
        items = list(host_trajectories.items())

        # Handle empty cases immediately
        valid_items = []
        for host_ip, snaps in items:
            if not snaps:
                results[host_ip] = HostAttackTrajectory(
                    host_ip=host_ip,
                    observed_entries=[],
                    forecast_entries=[],
                    cumulative_forecast_risk=0.0,
                )
            else:
                valid_items.append((host_ip, sorted(snaps, key=lambda s: s.window_idx)))

        # Process valid items in minibatches
        for start_idx in range(0, len(valid_items), batch_size):
            chunk = valid_items[start_idx : start_idx + batch_size]
            B = len(chunk)

            # Determine maximum lengths for this minibatch
            # Exactly history_steps, not the minibatch's longest host: both
            # models were trained on full-length, left-padded windows, and a
            # shorter sequence is a different number of LSTM / attention steps
            # than any training input. Widths come from the models (27 or 57
            # for Branch A; Branch B's state is the same full vector, 12 only
            # for a legacy latent-only checkpoint) -- they were hard-coded
            # 27 / 12, which a current 27-D Branch B cannot even accept.
            max_la, max_lw = max_seq_len_a, max_seq_len_wdt
            d_a = int(getattr(self.branch_a, "input_dim", 27))
            d_w = int(getattr(self.wdt, "d_latent", 12))

            # Pre-allocate numpy batch buffers
            x_batch = np.zeros((B, max_la, d_a), dtype=np.float32)
            h_hist_batch = np.zeros((B, max_lw, d_w), dtype=np.float32)
            t_hist_batch = np.zeros((B, max_lw), dtype=np.float32)
            t_a_batch = np.zeros((B, max_la), dtype=np.float32)   # Branch A history times

            # LEFT-pad, because that is what training does.
            #
            # branch_a_gnn_lstm/sequence_dataset.py builds every training window
            # as `padding + feature_vectors` (and LazyHostSequenceDataset as
            # `concatenate([pad, feats])`) -- zeros FIRST, the host's most
            # recent step LAST -- and passes no mask, so attention pools over
            # the pad positions too. This wrote `x_batch[i, j]` from j=0 and
            # left the tail zero, i.e. right-padding, then masked the tail out.
            # Both halves of that differ from training.
            #
            # It mattered most for Branch B. h_hist was right-padded as well,
            # so for any host with fewer snapshots than the longest host in its
            # minibatch the LAST history step -- the state the autoregressive,
            # residual-delta rollout starts from -- was the zero vector.
            # Measured with a 3-snapshot host batched against a 15-snapshot
            # host: |h[short, -1]| = 0.0 against |h[long, -1]| = 18.0. That host
            # was forecast from the origin rather than from where it actually
            # was, and nothing reported it.
            for i, (_, snaps) in enumerate(chunk):
                recent_a = snaps[-max_seq_len_a:]
                recent_w = snaps[-max_seq_len_wdt:]
                off_a = max_la - len(recent_a)
                off_w = max_lw - len(recent_w)
                for j, s in enumerate(recent_a):
                    x_batch[i, off_a + j] = np.concatenate([s.embedding, s.temporal_attrs])
                if recent_a:
                    _a0 = float(recent_a[-1].window_start)
                    _ta = [float(s.window_start) - _a0 for s in recent_a]
                    t_a_batch[i, off_a:] = _ta
                    t_a_batch[i, :off_a] = _ta[0]
                for j, s in enumerate(recent_w):
                    h_hist_batch[i, off_w + j] = (
                        np.concatenate([s.embedding, s.temporal_attrs])
                        if d_w != len(s.embedding) else s.embedding)
                if recent_w and off_w:
                    # edge padding, as Branch B is trained (first real state repeated)
                    h_hist_batch[i, :off_w] = h_hist_batch[i, off_w]
                # Real elapsed seconds relative to the latest snapshot, in the
                # convention Branch B is trained with (<= 0, last entry 0).
                # Left-padded slots repeat the earliest real time.
                if recent_w:
                    _t0 = float(recent_w[-1].window_start)
                    _ts = [float(s.window_start) - _t0 for s in recent_w]
                    t_hist_batch[i, off_w:] = _ts
                    t_hist_batch[i, :off_w] = _ts[0]

            x_tensor = torch.from_numpy(x_batch).to(self.device)
            h_hist_tensor = torch.from_numpy(h_hist_batch).to(self.device)
            t_hist_tensor = torch.from_numpy(t_hist_batch).to(self.device)

            with torch.no_grad():
                # 1. Branch A: Batched forward pass.
                #    No mask: training passes none (train_branch_a.py never
                #    builds one), so the weights were fitted with attention
                #    over the zero-pad positions. Masking them at serving time
                #    is a different function from the one that was trained.
                branch_a_out = self.branch_a(
                    x_tensor, t_history=torch.from_numpy(t_a_batch).to(self.device))
                obs_risks = branch_a_out["risk_score"].cpu().numpy()  # [B]
                obs_tech_probs = F.softmax(branch_a_out["technique_logits"], dim=-1)
                obs_tech_indices = obs_tech_probs.argmax(dim=-1).cpu().numpy()  # [B]
                obs_confs = obs_tech_probs.max(dim=-1).values.cpu().numpy()  # [B]

                # Map to technique vocabulary
                obs_techs = [
                    TECHNIQUE_VOCAB[idx] if idx < len(TECHNIQUE_VOCAB) else "Benign"
                    for idx in obs_tech_indices
                ]

                # 2. Branch B & DeepOP: Batched rollout & decoding
                # Branch A's prediction ONLY. This encoded the snapshot's
                # coarse_category -- its LABEL -- which in a replay is the
                # ground truth fed straight into the forecast.
                obs_token_ids = [self.deepop.vocab.encode("", obs_techs[i]) for i in range(B)]
                obs_t_tensor = torch.tensor(obs_token_ids, dtype=torch.long, device=self.device)
                # DeepOP's observed-sequence encoder was always trained with an
                # input; without one it ran a branch it never saw. Only the last
                # window's Branch A token is known here: [PAD.., BOS, token], a
                # shape training produces through its observed-token dropout.
                from deepop_decoder.forecast_decoder import observed_sequence_tokens
                obs_seq_tensor = torch.tensor(
                    [observed_sequence_tokens([t], _CONTRACT.history_steps, self.deepop.vocab)
                     for t in obs_token_ids], dtype=torch.long, device=self.device)

                # Real history spacing -- Branch B is trained on it. The future
                # is asked on the contract's forecast grid (forecast_window_seconds
                # per step, 30 s), as model_adapter does; this asked +2..+10 s.
                _fs = float(_CONTRACT.forecast_window_seconds)
                t_future_tensor = (torch.arange(1, K + 1, device=self.device, dtype=torch.float32)
                                   * _fs).expand(B, K)
                h_future = self.wdt.rollout(h_hist_tensor, K=K, delta_t_step=window_size_sec,
                                            t_history=t_hist_tensor,
                                            t_future=t_future_tensor)  # [B, K, d_latent]
                step_risks, cumul_risks = self.risk_head.forward_trajectory(h_future)  # [B, K], [B]
                step_risks_np = step_risks.cpu().numpy()  # [B, K]
                # Which aggregation is correct depends on what the head was
                # trained against, so it is read from the checkpoint rather
                # than assumed.
                #
                # `1 - prod(1-r)` is the discrete-survival identity, exact
                # when each step is an INDEPENDENT conditional hazard -- true
                # of the severity target. Under the hazard target it is wrong:
                # exp(-dt/tau) at k=1..K are K correlated restatements of one
                # event's proximity, not K independent draws, so the product
                # compounds the same evidence K times. Measured with exact
                # ground-truth hazards and a perfect predictor: an attack four
                # windows PAST the horizon yields cumulative 0.854 when the
                # truth is 0. peak() is the right reduction there.
                if getattr(self, "risk_target", "severity") == "hazard":
                    cumul_risks_np = step_risks_np.max(axis=1)  # [B]
                else:
                    cumul_risks_np = cumul_risks.cpu().numpy()  # [B]

                # Ask for the decoder's own per-step probabilities. Without
                # them the forecast confidence below was a hardcoded decay that
                # never looked at the model.
                pred_tokens, decoded_names, _attack_p, step_token_probs = (
                    self.deepop.forecast_sequence(
                        h_future, max_steps=K, observed_token=obs_t_tensor,
                        observed_sequence=obs_seq_tensor,
                        return_probs=True,
                    )
                )

            # Unpack batch results into HostAttackTrajectory instances
            for i, (host_ip, snaps) in enumerate(chunk):
                observed_entries = []
                for s in snaps:
                    observed_entries.append(
                        TrajectoryEntry(
                            host_ip=host_ip,
                            window_idx=s.window_idx,
                            timestamp=s.window_start,
                            provenance=Provenance.OBSERVED.value,
                            coarse_category=s.coarse_category,
                            technique_id=obs_techs[i] if s == snaps[-1] else (s.technique_ids[0] if s.technique_ids else "Benign"),
                            confidence=float(obs_confs[i]) if s == snaps[-1] else 0.95,
                            risk_score=float(obs_risks[i]) if s == snaps[-1] else s.risk_score,
                        )
                    )

                forecast_entries = []
                last_t = snaps[-1].window_start
                last_win = snaps[-1].window_idx

                host_token_probs = step_token_probs[i] if step_token_probs else []
                for k in range(K):
                    coarse, tech = decoded_names[i][k]
                    step_r = float(step_risks_np[i, k])
                    forecast_entries.append(
                        TrajectoryEntry(
                            host_ip=host_ip,
                            window_idx=last_win + k + 1,
                            timestamp=last_t + (k + 1) * _fs,
                            provenance=Provenance.FORECAST.value,
                            coarse_category=coarse,
                            # An empty technique means the decoder emitted a
                            # category with no technique. Calling that "T1071"
                            # (Application Layer Protocol, i.e. C2) labelled an
                            # unknown forecast as command-and-control, which is
                            # a specific and alarming claim to invent.
                            technique_id=tech if tech else "",
                            # The decoder's own probability for the token it
                            # emitted. This was `max(0.4, 0.9 - 0.1 * k)` -- a
                            # fixed decay from 0.9 that never consulted the
                            # model, so step 0 of every forecast for every host
                            # was reported at 90% confidence. (The untrained MLP
                            # edge scorer that read it as a feature was removed.)
                            confidence=(
                                float(host_token_probs[k])
                                if k < len(host_token_probs)
                                else 0.0
                            ),
                            risk_score=step_r,
                        )
                    )

                results[host_ip] = HostAttackTrajectory(
                    host_ip=host_ip,
                    observed_entries=observed_entries,
                    forecast_entries=forecast_entries,
                    cumulative_forecast_risk=float(cumul_risks_np[i]),
                )

        return results

    def assemble_host_trajectory(
        self,
        host_ip: str,
        observed_snapshots: list,  # List[HostWindowSnapshot]
        K: int = None,
        window_size_sec: float = None,
    ) -> HostAttackTrajectory:
        """
        Assembles trajectory for a single host (backward-compatible convenience wrapper).

        The defaults are the contract's. They were `K=4, window_size_sec=60.0`
        -- the unsupported MACRO granularity that the batch entry point above
        was already fixed to stop defaulting to. No model in this repository is
        trained at 60 s, so this wrapper produced a 4-step forecast with its
        steps timestamped 60 s apart from weights fitted for 5 steps at 2 s:
        the returned timeline claimed to reach 240 s ahead when the models can
        see 10 s.
        """
        K = _CONTRACT.forecast_steps if K is None else K
        window_size_sec = (
            _CONTRACT.window_seconds if window_size_sec is None else window_size_sec
        )
        batch_res = self.assemble_trajectories_batch(
            {host_ip: observed_snapshots},
            batch_size=1,
            K=K,
            window_size_sec=window_size_sec,
        )
        return batch_res[host_ip]
