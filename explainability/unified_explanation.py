"""
Unified Explainability Suite.

Provides multi-modal attribution for security operations:
1. Feature Attribution: Fast Gradient-Activation / Gradient * Input over temporal attributes and embeddings.
2. Multi-Scale Attention: Surfaces LSTM sequence attention weights and CWA window weights.
3. Causal DAG Extraction: Isolates local slices of the GRAIN/APTShield campaign graph.
4. Human-Readable Operator Narrative: Synthesizes findings into actionable alert summaries.
5. Asynchronous Worker Queue: Non-blocking background worker for real-time SOC pipelines.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import collections
import concurrent.futures
import threading
import numpy as np
import torch

from correlation.trajectory_assembler import HostAttackTrajectory
from correlation.campaign_merge import AttackCampaign


# Feature names for the 27-D Branch A input, DERIVED -- never duplicated.
#
# This module used to keep its own hardcoded copy of the 15 host attribute
# names, and it had already drifted from the computation: index 14 was called
# "active_conn_density" when the value is unique_peers / flow_count, i.e. peer
# fan-out, not a connection count. A duplicated name list is the same defect
# that made host_attributes.py wrong in all fifteen entries -- names and the
# code that computes them must have exactly one source.
#
# These strings are what an operator reads next to an attribution score, so a
# wrong one is a confidently-stated wrong explanation.
from data_unification.host_attributes import HOST_ATTRIBUTES

TGNE_LATENT_NAMES = [f"H_emb_{i}" for i in range(12)]
FEATURE_NAMES = TGNE_LATENT_NAMES + list(HOST_ATTRIBUTES)

assert len(FEATURE_NAMES) == 27, (
    f"Branch A input is 27-D (12 embedding + 15 attributes); got {len(FEATURE_NAMES)}"
)

# A Branch A trained with --packet-features reads 57-D: the 15 flow attributes
# plus the 30 PCAP packet-level ones (TTL, TCP window, retransmissions, IAT,
# SYN/scan signatures, ...), in the extractor's order.
from data_unification.host_attributes import EXTENDED_HOST_ATTRIBUTES  # noqa: E402

FEATURE_NAMES_PACKET = TGNE_LATENT_NAMES + list(EXTENDED_HOST_ATTRIBUTES)


def feature_names_for(width: int) -> List[str]:
    """The names of a Branch A input row of `width` columns."""
    if width == len(FEATURE_NAMES):
        return FEATURE_NAMES
    if width == len(FEATURE_NAMES_PACKET):
        return FEATURE_NAMES_PACKET
    raise ValueError(f"no feature names for a {width}-wide input "
                     f"(known: {len(FEATURE_NAMES)}, {len(FEATURE_NAMES_PACKET)})")


@dataclass
class UnifiedExplanation:
    """Unified explanation artifact presented to the security operator."""
    host_ip: str
    timestamp: float
    current_risk_score: float
    predicted_technique: str
    top_feature_attributions: List[Tuple[str, float]]  # List of (feature_name, attribution_score)
    temporal_attention_weights: List[float]           # Attention over the history window, oldest -> newest
    campaign_id: Optional[int]
    causal_chain_summary: List[Dict[str, Any]]
    operator_narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedExplainer:
    """Generates comprehensive multi-modal explanations for hosts and campaigns."""

    def __init__(self, branch_a_model, device: str = "cpu", max_cache_size: int = 2000):
        self.branch_a = branch_a_model.to(device)
        self.device = device
        self.max_cache_size = max_cache_size
        self._cache: collections.OrderedDict[Tuple[str, float, float], UnifiedExplanation] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.branch_a.eval()

    def clear_cache(self):
        """Clears the explanation LRU cache."""
        with self._lock:
            self._cache.clear()

    def _get_cache(self, key: Tuple[str, float, float]) -> Optional[UnifiedExplanation]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _set_cache(self, key: Tuple[str, float, float], value: UnifiedExplanation):
        with self._lock:
            self._cache[key] = value
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

    def explain_host_trajectory(
        self,
        trajectory: HostAttackTrajectory,
        campaign: Optional[AttackCampaign] = None,
        fast_mode: bool = False,
        use_cache: bool = True,
        feature_sequence: Optional[np.ndarray] = None,
    ) -> UnifiedExplanation:
        """
        Generates a unified explanation object for a host's current attack trajectory.

        Args:
            trajectory: Target host trajectory
            campaign: Optional associated campaign
            fast_mode: If True, uses forward-only attention-weighted activation attribution (<0.5ms)
                       If False, performs single-pass gradient-input backpropagation.
            use_cache: If True, checks and updates LRU cache.
            feature_sequence: the [T, 27] window that was actually scored. REQUIRED
                for feature attribution -- see below.

        ## Why `feature_sequence` is a parameter and not reconstructed here

        This method used to build its own input:

            for e in obs_entries[-5:]:
                emb = np.zeros(12, dtype=np.float32)
                attrs = np.zeros(15, dtype=np.float32)
                seq_features.append(np.concatenate([emb, attrs]))

        `e` is never read. Every timestep was the zero vector, so
        `attributions = |grad * input|` was identically zero, the normalisation
        was skipped by its `if total_attr > 0` guard, and the top-5 fell out of
        `sorted()` as whatever FEATURE_NAMES happens to list first. Measured on
        both paths, fast and exact:

            top_feature_attributions = [('H_emb_0', 0.0), ('H_emb_1', 0.0),
                                        ('H_emb_2', 0.0), ('H_emb_3', 0.0),
                                        ('H_emb_4', 0.0)]
            narrative: "Primary driving indicators: H_emb_0 (0.0%),
                        H_emb_1 (0.0%), H_emb_2 (0.0%)."

        An operator was told which features drove an alert, in a fixed order
        that had nothing to do with the host, the model, or the alert. That is
        worse than no explanation, which is why the absent case now says so
        instead of ranking zeros.

        `TrajectoryEntry` carries no embedding or temporal_attrs -- the
        explainer genuinely cannot recover the scored window from a trajectory,
        which is presumably how the zeros got here. The caller holds it
        (`control_backend/model_adapter.py` has it as `x_tensor`), so it is
        passed in.
        """
        obs_entries = trajectory.observed_entries
        if not obs_entries:
            return UnifiedExplanation(
                host_ip=trajectory.host_ip,
                timestamp=0.0,
                current_risk_score=0.0,
                predicted_technique="Benign",
                top_feature_attributions=[],
                temporal_attention_weights=[],
                campaign_id=None,
                causal_chain_summary=[],
                operator_narrative=f"Host {trajectory.host_ip} has no observed activity.",
            )

        last_entry = obs_entries[-1]
        host_ip = trajectory.host_ip
        cache_key = (host_ip, float(last_entry.timestamp), round(float(last_entry.risk_score), 4))

        if use_cache:
            cached = self._get_cache(cache_key)
            if cached is not None:
                return cached

        if feature_sequence is None:
            # No scored window, so no attribution. Say nothing rather than
            # rank zeros; see the docstring.
            attn_weights: List[float] = []
            feat_ranks: List[Tuple[str, float]] = []
        else:
            seq = np.asarray(feature_sequence, dtype=np.float32)
            if seq.ndim == 3 and seq.shape[0] == 1:
                seq = seq[0]
            if seq.ndim != 2 or seq.shape[-1] not in (len(FEATURE_NAMES), len(FEATURE_NAMES_PACKET)):
                raise ValueError(
                    f"feature_sequence must be [T, {len(FEATURE_NAMES)}] or "
                    f"[T, {len(FEATURE_NAMES_PACKET)}] (the window that was scored); "
                    f"got {np.asarray(feature_sequence).shape}"
                )
            names = feature_names_for(seq.shape[-1])
            # The sequence length is the model's, not a literal: this was
            # hardcoded to 5 in four places while the contract is 15, so even a
            # correctly-populated window would have been truncated to its last
            # third before being explained.
            x = torch.from_numpy(seq[None, ...]).to(self.device)

            if fast_mode:
                # Fast-path: Single forward pass with attention-proxy readout without gradient tracking
                with torch.no_grad():
                    out = self.branch_a(x)
                    attn_weights = [float(w) for w in out["attention_weights"][0].cpu().numpy()]
                    # Proxy feature attribution using input magnitude scaled by final attention weight
                    inputs = np.abs(x[0, -1, :].cpu().numpy())
                    attributions = inputs * (attn_weights[-1] if attn_weights else 1.0)
            else:
                # High-precision path: Gradient * Input attribution.
                #
                # The module stays in eval(). This called self.branch_a.train()
                # as a cuDNN-RNN-backward workaround, which also switched
                # dropout=0.2 back on across the LSTM and every head, so the
                # attributions an operator saw were a fresh random draw each
                # time the same alert was explained. Disabling cuDNN achieves
                # the same thing without perturbing the model, and is what
                # control_backend/model_adapter.py::_explain already does.
                x.requires_grad_(True)
                was_training = self.branch_a.training
                self.branch_a.eval()
                try:
                    with torch.backends.cudnn.flags(enabled=False):
                        out = self.branch_a(x)
                        risk = out["risk_score"]
                        if risk.ndim > 0:
                            risk = risk.reshape(-1)[0]
                        self.branch_a.zero_grad(set_to_none=True)
                        risk.backward()
                finally:
                    if was_training:
                        self.branch_a.train()

                # Summed over every timestep (see model_adapter._explain_full).
                grads = (
                    x.grad[0].cpu().numpy()
                    if x.grad is not None
                    else np.zeros((x.shape[1], len(names)), dtype=np.float32)
                )
                inputs = x[0].detach().cpu().numpy()
                attributions = np.abs(grads * inputs).sum(axis=0)
                attn_weights = [float(w) for w in out["attention_weights"][0].detach().cpu().numpy()]

            # Normalize and sort feature attributions
            total_attr = float(attributions.sum())
            if total_attr > 0:
                attributions = attributions / total_attr
            feat_ranks = sorted(
                zip(names, [float(v) for v in attributions]),
                key=lambda item: item[1],
                reverse=True,
            )[:5]

        # Extract local causal chain from campaign if available
        causal_chain = []
        camp_id = None
        if campaign:
            camp_id = campaign.campaign_id
            for nd in campaign.nodes:
                if nd["host_ip"] == host_ip:
                    causal_chain.append({
                        "node_id": nd["node_id"],
                        "technique": nd["technique_id"],
                        "coarse_category": nd["coarse_category"],
                        "provenance": nd["provenance"],
                        "risk_score": nd["risk_score"],
                    })

        # Generate human-readable narrative
        fc_entries = trajectory.forecast_entries
        forecast_str = ""
        if fc_entries:
            forecast_str = f" Predicted next stages: {', '.join([f'{f.technique_id} ({f.coarse_category})' for f in fc_entries[:2]])}."

        if feat_ranks:
            drivers = (
                "Primary driving indicators: "
                + ", ".join(f"{name} ({val * 100:.1f}%)" for name, val in feat_ranks[:3])
                + "."
            )
        else:
            drivers = (
                "Feature attribution unavailable: the scored feature window was not "
                "supplied to the explainer."
            )

        narrative = (
            f"Host {host_ip} exhibits an active compromise risk of {last_entry.risk_score:.2f} "
            f"classified under {last_entry.coarse_category} (Technique {last_entry.technique_id}). "
            f"{drivers}"
            f"{forecast_str}"
        )

        explanation = UnifiedExplanation(
            host_ip=host_ip,
            timestamp=last_entry.timestamp,
            current_risk_score=last_entry.risk_score,
            predicted_technique=last_entry.technique_id,
            top_feature_attributions=feat_ranks,
            temporal_attention_weights=attn_weights,
            campaign_id=camp_id,
            causal_chain_summary=causal_chain,
            operator_narrative=narrative,
        )

        if use_cache:
            self._set_cache(cache_key, explanation)

        return explanation

    def explain_trajectories_batch(
        self,
        trajectories: Dict[str, HostAttackTrajectory],
        campaigns: Optional[List[AttackCampaign]] = None,
        fast_mode: bool = False,
        feature_sequences: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, UnifiedExplanation]:
        """
        Explains a batch of host trajectories.

        fast_mode defaulted to True, i.e. the forward-only proxy: it ranks
        features by their INPUT MAGNITUDE (times one attention weight), which
        says how large a feature is, not how much it moved the prediction --
        a host with heavy benign traffic would be "explained" by its byte
        counts. The default is now the Input x Gradient attribution.
        """
        # Build host-to-campaign map
        host_to_camp: Dict[str, AttackCampaign] = {}
        if campaigns:
            for camp in campaigns:
                for hip in camp.involved_hosts:
                    host_to_camp[hip] = camp

        results: Dict[str, UnifiedExplanation] = {}
        for hip, traj in trajectories.items():
            camp = host_to_camp.get(hip)
            results[hip] = self.explain_host_trajectory(
                traj, campaign=camp, fast_mode=fast_mode,
                feature_sequence=(feature_sequences or {}).get(hip),
            )
        return results


class AsyncExplainabilityQueue:
    """
    Decoupled Asynchronous Worker Queue for Non-Blocking Alert Explanations.
    
    Prevents backpropagation / explanation overhead from stalling the ingestion pipeline.
    """

    def __init__(self, explainer: UnifiedExplainer, max_workers: int = 4):
        self.explainer = explainer
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="async-explainer"
        )
        self.futures: Dict[str, concurrent.futures.Future[UnifiedExplanation]] = {}
        self.completed_cache: Dict[str, UnifiedExplanation] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        trajectory: HostAttackTrajectory,
        campaign: Optional[AttackCampaign] = None,
        fast_mode: bool = False,
        feature_sequence: Optional[np.ndarray] = None,
    ) -> concurrent.futures.Future[UnifiedExplanation]:
        """Submits an explanation task asynchronously without blocking."""
        host_ip = trajectory.host_ip
        fut = self.executor.submit(
            self.explainer.explain_host_trajectory,
            trajectory,
            campaign,
            fast_mode,
            True,
            feature_sequence,
        )

        def _on_done(f: concurrent.futures.Future[UnifiedExplanation]):
            try:
                res = f.result()
                with self._lock:
                    self.completed_cache[host_ip] = res
            except Exception:
                pass

        fut.add_done_callback(_on_done)
        with self._lock:
            self.futures[host_ip] = fut
        return fut

    def submit_batch(
        self,
        trajectories: Dict[str, HostAttackTrajectory],
        campaigns: Optional[List[AttackCampaign]] = None,
        fast_mode: bool = False,
        feature_sequences: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, concurrent.futures.Future[UnifiedExplanation]]:
        """Submits a batch of trajectories asynchronously."""
        host_to_camp: Dict[str, AttackCampaign] = {}
        if campaigns:
            for camp in campaigns:
                for hip in camp.involved_hosts:
                    host_to_camp[hip] = camp

        future_map: Dict[str, concurrent.futures.Future[UnifiedExplanation]] = {}
        for hip, traj in trajectories.items():
            camp = host_to_camp.get(hip)
            future_map[hip] = self.submit(
                traj, campaign=camp, fast_mode=fast_mode,
                feature_sequence=(feature_sequences or {}).get(hip),
            )
        return future_map

    def get_explanation(self, host_ip: str, timeout: Optional[float] = None) -> Optional[UnifiedExplanation]:
        """Retrieves explanation result for host_ip, blocking up to timeout if running."""
        with self._lock:
            if host_ip in self.completed_cache:
                return self.completed_cache[host_ip]
            fut = self.futures.get(host_ip)

        if fut is not None:
            try:
                return fut.result(timeout=timeout)
            except (concurrent.futures.TimeoutError, Exception):
                return None
        return None

    def get_all_completed(self) -> Dict[str, UnifiedExplanation]:
        """Returns all completed explanations ready for display."""
        with self._lock:
            return dict(self.completed_cache)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for f in self.futures.values() if not f.done())

    def shutdown(self, wait: bool = False):
        """Shuts down the worker pool."""
        self.executor.shutdown(wait=wait)
