"""
control_backend/model_adapter.py
Live Antigravity dual-branch + DeepOP inference adapter.

Consumes UnifiedFlowRecord windows from Containerlab SPAN,
runs TGNE-TA → Branch A / Branch B (WDT) / DeepOP CWA, and emits PredictionEvent.
"""

from datetime import datetime
import logging
import os
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from control_backend.schema import (
    ModelMetadata,
    StateMetadata,
    ForecastPoint,
    ExplainabilityGroup,
    ExplainabilityFeature,
    ExplainabilityPayload,
    PredictionData,
    LatencyData,
    EarlyWarningData,
    PredictionEvent,
    FocusEdge,
)
from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory
from data_unification.temporal_config import (
    LIVE_WINDOW_SIZE_SEC,
    DEFAULT_ROLLOUT_HORIZON_LIVE,
    DEFAULT_HISTORY_STEPS,
)
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.behavioral_fingerprint import BehavioralFlowFingerprinter
from explainability.unified_explanation import FEATURE_NAMES

logger = logging.getLogger("antigravity.model_adapter")

TECHNIQUE_TO_MITRE = {
    "Benign": ("Benign", "None", "TA0000", "No malicious activity observed"),
    "T1046": ("Reconnaissance", "T1046 Network Service Discovery", "TA0043", "Network service discovery"),
    "T1595": ("Reconnaissance", "T1595 Active Scanning", "TA0043", "Active scanning"),
    "T1110": ("Credential Access", "T1110 Brute Force", "TA0006", "Brute force authentication"),
    "T1190": ("Initial Access", "T1190 Exploit Public-Facing Application", "TA0001", "Exploit public-facing app"),
    "T1189": ("Initial Access", "T1189 Drive-by Compromise", "TA0001", "Drive-by compromise"),
    "T1071": ("Command and Control", "T1071 Application Layer Protocol", "TA0011", "Application-layer C2"),
    "T1071.001": ("Command and Control", "T1071.001 Web Protocols", "TA0011", "Web-protocol C2"),
    "T1568.001": ("Command and Control", "T1568.001 Fast Flux DNS", "TA0011", "Fast-flux DNS"),
    "T1204": ("Execution", "T1204 User Execution", "TA0002", "User execution"),
    "T1005": ("Collection", "T1005 Data from Local System", "TA0009", "Local data collection"),
    "T1498": ("Impact", "T1498 Network Denial of Service", "TA0040", "Network DoS"),
    "T1498.001": ("Impact", "T1498.001 Direct Network Flood", "TA0040", "Direct network flood"),
    "T1020": ("Exfiltration", "T1020 Automated Exfiltration", "TA0010", "Automated exfiltration"),
}

# The SOC rule layer emits its own coarse labels. They are not ATT&CK technique
# ids, so TECHNIQUE_TO_MITRE used to miss and the dashboard rendered
# "MITRE Unknown -> Exploit". Mapping them to the real technique each rule is
# actually detecting keeps the displayed ATT&CK annotation meaningful; the
# rules_applied flag still tells the operator the label came from a rule.
RULE_TECHNIQUE_TO_ATTCK = {
    "PortScan": "T1046",     # Network Service Discovery
    "WebAttack": "T1190",    # Exploit Public-Facing Application
    "Exploit": "T1190",      # Exploit Public-Facing Application
}

FEATURE_GROUP_MAP = {
    **{f"H_emb_{i}": "TGNE Latent" for i in range(12)},
    "flow_count": "Connectivity",
    "fwd_bytes": "Volume",
    "bwd_bytes": "Volume",
    "total_bytes": "Volume",
    "fwd_packets": "Volume",
    "bwd_packets": "Volume",
    "total_packets": "Volume",
    "unique_peers": "Connectivity",
    "unique_dst_ports": "Connectivity",
    "tcp_ratio": "Connectivity",
    "udp_ratio": "Connectivity",
    "avg_flow_duration": "Timing",
    "byte_rate": "Volume",
    "packet_rate": "Volume",
    "active_conn_density": "Connectivity",
}

class AntigravityModelAdapter:
    """Dual-Branch + DeepOP live inference → dashboard PredictionEvent."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.h_state_history: List[torch.Tensor] = []
        self.feature_history: List[torch.Tensor] = []
        self.h_state_history_by_target: Dict[str, List[torch.Tensor]] = {}
        self.feature_history_by_target: Dict[str, List[torch.Tensor]] = {}
        self.alert_threshold = 0.65
        # Rules on by default so the operator console is unchanged; off for any
        # scientific run. CYBERWORLD_DISABLE_RULES=1 yields ML-only output.
        self.rules_enabled = os.environ.get("CYBERWORLD_DISABLE_RULES", "") not in ("1", "true", "True")
        # Provisional; overwritten in _load_models by the contract the
        # checkpoints actually carry. A model's temporal contract is a property
        # of the model, not a global constant — hardcoding it here is how a v3
        # checkpoint could be served under v4 settings without anything noticing.
        self.window_seconds = LIVE_WINDOW_SIZE_SEC
        self.forecast_steps = DEFAULT_ROLLOUT_HORIZON_LIVE
        self.history_steps = DEFAULT_HISTORY_STEPS
        self.checkpoint_contract: Dict[str, Any] = {}
        self.fingerprinter = BehavioralFlowFingerprinter()
        self._load_models()

    def _load_models(self):
        import sys

        # Prefer repo/bita over ambient editable installs (e.g. SIH/src/model.py)
        from control_backend.lab_config import REPO_ROOT

        repo = REPO_ROOT
        bita = os.path.join(repo, "bita")
        os.chdir(repo)
        sys.path = [
            p
            for p in sys.path
            if p
            and "SIH" not in p
            and os.path.abspath(p) != os.path.abspath(bita)
        ]
        sys.path.insert(0, bita)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        for k in list(sys.modules):
            if k == "model" or k.startswith("model."):
                del sys.modules[k]

        from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
        from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB
        from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
        from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
        from branch_b_world_model.infiltration_head import InfiltrationRiskHead
        from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
        from deepop_decoder.joint_vocab import get_joint_vocab, consolidate_network_technique

        self.technique_vocab = TECHNIQUE_VOCAB

        self.tgn = build_or_load_tgne_ta()
        self.extractor = HostTrajectoryExtractor(
            tgne_ta_model=self.tgn,
            window_size_sec=self.window_seconds,
        )

        self.branch_a = MultiTaskLSTM(input_dim=27, hidden_dim=64).to(self.device)
        ba_path = os.path.join(repo, "saved_models/branch_a/branch_a_lstm.pt")
        if not os.path.exists(ba_path):
            raise RuntimeError(f"Missing Branch A checkpoint: {ba_path}")
        ckpt = torch.load(ba_path, map_location=self.device, weights_only=False)
        self.branch_a.load_state_dict(ckpt["model_state_dict"])
        self.branch_a.eval()
        self._adopt_contract(ckpt, "branch_a")

        self.wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64).to(self.device)
        self.risk_head = InfiltrationRiskHead(d_latent=12, hidden_dim=32).to(self.device)
        bb_path = os.path.join(repo, "saved_models/branch_b/host_wdt.pt")
        if not os.path.exists(bb_path):
            raise RuntimeError(f"Missing Branch B checkpoint: {bb_path}")
        ckpt = torch.load(bb_path, map_location=self.device, weights_only=False)
        self._adopt_contract(ckpt, "branch_b")
        self.wdt.load_state_dict(ckpt["wdt_state_dict"])
        self.risk_head.load_state_dict(ckpt["risk_head_state_dict"])
        self.wdt.eval()
        self.risk_head.eval()

        self.vocab = get_joint_vocab()
        self.consolidate_network_technique = consolidate_network_technique
        self.deepop = DeepOPForecastDecoder(
            d_latent=12, d_model=72, vocab_size=self.vocab.vocab_size
        ).to(self.device)
        dp_path = os.path.join(repo, "saved_models/deepop/cwa_forecast_decoder.pt")
        if not os.path.exists(dp_path):
            raise RuntimeError(f"Missing DeepOP checkpoint: {dp_path}")
        ckpt = torch.load(dp_path, map_location=self.device, weights_only=False)
        self.deepop.load_state_dict(ckpt["decoder_state_dict"])
        self.deepop.eval()
        # DeepOP was the only checkpoint whose contract was never adopted, even
        # though its horizon is what defines the served forecast length.
        self._adopt_contract(ckpt, "deepop")

        from cyberworld_v4.config import get_contract

        # The extractor was constructed above from the PROVISIONAL window, before
        # any checkpoint had been read. If a checkpoint carries a different
        # window, _adopt_contract updated self.window_seconds but left the
        # extractor bucketing at the old one -- so features were built on a
        # different grid than the models were trained on. Rebind it now that the
        # real contract is known.
        if abs(self.extractor.window_size_sec - self.window_seconds) > 1e-9:
            logger.warning(
                "Rebinding extractor window %ss -> %ss to match the checkpoint contract",
                self.extractor.window_size_sec, self.window_seconds,
            )
            self.extractor.window_size_sec = self.window_seconds

        served_contract = {
            "window_seconds": self.window_seconds,
            "history_steps": self.history_steps,
            "forecast_steps": self.forecast_steps,
        }
        served = get_contract().matches(served_contract)

        logger.info(
            "Models loaded (device=%s) serving contract: %ss windows | %s history | %s forecast",
            self.device, self.window_seconds, self.history_steps, self.forecast_steps,
        )

        if not served:
            # This used to be a parenthetical in a log line. Serving weights that
            # disagree with the contract produces confident, wrong forecasts --
            # the horizon and the history length are not cosmetic. Refuse by
            # default; the escape hatch exists only so a demo can still run on
            # known-stale checkpoints, and it says so loudly.
            msg = (
                f"Checkpoint contract mismatch: the loaded checkpoints serve "
                f"{served_contract} but the authoritative contract is "
                f"{get_contract().to_dict()}. These weights were trained for a "
                f"different temporal granularity and their forecasts are not valid. "
                f"Retrain, or set CYBERWORLD_ALLOW_CONTRACT_MISMATCH=1 to serve anyway."
            )
            if os.environ.get("CYBERWORLD_ALLOW_CONTRACT_MISMATCH", "") in ("1", "true", "True"):
                logger.error("SERVING STALE CHECKPOINTS ANYWAY. %s", msg)
            else:
                raise RuntimeError(msg)

    def _adopt_contract(self, ckpt: Dict[str, Any], name: str) -> None:
        """Take the temporal contract from the checkpoint being loaded.

        v3 weights carry window_size_sec/history_steps (or window_seconds/
        history_steps/forecast_steps); v4 weights carry a `config.temporal`
        block. Either way the served contract comes from the artefact rather
        than from a constant that may no longer describe it.

        Checkpoints that disagree with each other are a hard error: rolling a
        5-step history against an 8-step decoder silently produces nonsense.
        """
        tc = ckpt.get("training_contract") or {}
        cfg = (ckpt.get("config") or {}).get("temporal") or {}
        src = cfg or tc or ckpt

        found = {}
        w = src.get("window_seconds", src.get("window_size_sec"))
        if w:
            found["window_seconds"] = float(w)
        if src.get("history_steps"):
            found["history_steps"] = int(src["history_steps"])
        if src.get("forecast_steps"):
            found["forecast_steps"] = int(src["forecast_steps"])
        if not found:
            return

        for k, v in found.items():
            prev = self.checkpoint_contract.get(k)
            if prev is not None and prev != v:
                raise RuntimeError(
                    f"Checkpoint contract conflict on {k}: {name} says {v}, "
                    f"an earlier checkpoint said {prev}. Checkpoints trained under "
                    f"different temporal contracts cannot be composed."
                )
            self.checkpoint_contract[k] = v

        self.window_seconds = self.checkpoint_contract.get("window_seconds", self.window_seconds)
        self.history_steps = self.checkpoint_contract.get("history_steps", self.history_steps)
        self.forecast_steps = self.checkpoint_contract.get("forecast_steps", self.forecast_steps)

    def reset_history(self):
        self.h_state_history.clear()
        self.feature_history.clear()
        self.h_state_history_by_target.clear()
        self.feature_history_by_target.clear()

    def _build_embedding(
        self, target_ip: str, flows: List[UnifiedFlowRecord]
    ) -> np.ndarray:
        h_emb = np.zeros(12, dtype=np.float32)
        if not flows:
            return h_emb
        try:
            trajectories = self.extractor.extract_trajectories(flows)
            if target_ip in trajectories and len(trajectories[target_ip]) > 0:
                return np.asarray(trajectories[target_ip][-1].embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError("TGNE embedding extraction failed") from e

        raise RuntimeError(f"TGNE produced no embedding for target host {target_ip}")

    def _explain(self, x_tensor: torch.Tensor) -> ExplainabilityPayload:
        """Input x Gradient attribution for the prediction actually made.

        Two defects this fixes:

        1. It used to rebuild a [1, 1, 27] tensor from the final feature vector
           while the scored input was [1, 5, 27]. On a temporal model that
           explains a single history-less timestep the model never scored -- not
           an explanation of the prediction.

        2. It called self.branch_a.train() as a cuDNN-RNN-backward workaround,
           leaving dropout=0.2 active on the LSTM and every head, so the
           attributions were stochastic and irreproducible run to run.
           torch.backends.cudnn.flags(enabled=False) achieves the same thing
           while the module stays in eval().

        Attribution is over the last timestep of the real sequence, with
        gradients flowing through the full history.
        """
        x = x_tensor.detach().clone().to(self.device)
        x.requires_grad_(True)

        was_training = self.branch_a.training
        self.branch_a.eval()
        try:
            # RNN backward needs cuDNN disabled in eval; this replaces the
            # train()-mode workaround without enabling dropout.
            with torch.backends.cudnn.flags(enabled=False):
                out = self.branch_a(x)
                risk = out["risk_score"]
                if risk.ndim > 0:
                    risk = risk.reshape(-1)[0]
                self.branch_a.zero_grad(set_to_none=True)
                risk.backward()

            grads = (
                x.grad[0, -1, :].detach().cpu().numpy()
                if x.grad is not None
                else np.zeros(x.shape[-1], dtype=np.float32)
            )
            inputs = x[0, -1, :].detach().cpu().numpy()
            attributions = np.abs(grads * inputs)
        finally:
            if was_training:
                self.branch_a.train()

        total = float(attributions.sum()) + 1e-12
        attributions = attributions / total

        group_scores: Dict[str, float] = {}
        top_features: List[ExplainabilityFeature] = []
        ranked = sorted(
            zip(FEATURE_NAMES, attributions.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )
        for name, score in ranked:
            group = FEATURE_GROUP_MAP.get(name, "General")
            group_scores[group] = group_scores.get(group, 0.0) + float(score)
        for name, score in ranked[:8]:
            top_features.append(
                ExplainabilityFeature(
                    feature=name,
                    score=round(float(score), 4),
                    group=FEATURE_GROUP_MAP.get(name, "General"),
                )
            )

        groups = [
            ExplainabilityGroup(name=g, percentage=round(100.0 * pct, 1))
            for g, pct in sorted(group_scores.items(), key=lambda kv: -kv[1])
        ]
        return ExplainabilityPayload(
            available=True,
            method="Input x Gradient Saliency",
            groups=groups,
            top_features=top_features,
        )

    def _alert_level(self, risk: float) -> str:
        if risk >= 0.85:
            return "CRITICAL"
        if risk >= 0.65:
            return "ELEVATED"
        if risk >= self.alert_threshold:
            return "WARNING"
        return "NOMINAL"

    def predict_window(
        self,
        target_ip: str,
        flows: List[UnifiedFlowRecord],
        window_id: int = 0,
        attack_active: bool = False,
        attack_phase: Optional[str] = None,
        is_mitigated: bool = False,
        packet_count: int = 0,
        pipeline_latency_ms: float = 0.0,
        active_flows: Optional[int] = None,
        throughput: float = 0.0,
    ) -> PredictionEvent:
        t0 = time.perf_counter()

        if is_mitigated:
            flows = []

        host_flows = [
            record
            for record in flows
            if record.src_ip == target_ip or record.dst_ip == target_ip
        ]
        if host_flows:
            window_end = max(record.end_time for record in host_flows)
            window_start = window_end - self.window_seconds
            host_flows = [
                record
                for record in host_flows
                if record.end_time >= window_start or record.start_time >= window_start
            ]

        # Host-scoped flows are correct for the temporal ATTRIBUTES: those are
        # per-host aggregates (this host's byte counts, peers, ports).
        temp_attrs = self.extractor.compute_host_temporal_attributes(
            host_ip=target_ip,
            window_records=host_flows,
            window_duration=self.window_seconds,
        )

        # ...but NOT for the TGNE embedding. TGNE is a graph encoder: it
        # aggregates over a host's neighbourhood, so the graph it is given
        # changes the embedding it returns. The offline trainers call
        # extract_trajectories() on the FULL window, while this path used to
        # pass only host_flows — a strictly smaller subgraph. Measured on a
        # replayed window with 16 flows of which 2 touched the target, the two
        # embeddings differed by 1.5e-2 per dimension, and the gap widens with
        # cross-host traffic. That is a train/serve mismatch: the model was
        # fitted on full-graph embeddings and served subgraph ones.
        #
        # The full window is passed here so live matches training. Verified by
        # scripts/verify_offline_live_parity.py.
        window_flows = flows if flows else host_flows
        h_emb = self._build_embedding(target_ip, window_flows)
        
        import model_contract
        model_contract.assert_shape(h_emb, (model_contract.TGNE_LATENT_DIM,), "TGNE Embedding")
        
        feature_vector = np.concatenate([h_emb, temp_attrs]).astype(np.float32)
        model_contract.assert_shape(feature_vector, (model_contract.BRANCH_A_INPUT_DIM,), "Branch A Input Vector")

        feature_history = self.feature_history_by_target.setdefault(target_ip, [])
        feature_history.append(
            torch.from_numpy(feature_vector).float().to(self.device)
        )
        if len(feature_history) > self.history_steps:
            feature_history.pop(0)
        self.feature_history = feature_history
        feature_history = list(feature_history)
        while len(feature_history) < self.history_steps:
            feature_history.insert(0, torch.zeros_like(feature_history[0]))
        x_tensor = torch.stack(feature_history).unsqueeze(0)

        with torch.no_grad():
            logger.debug("[TGNE] -> [BRANCH_A] -> [BRANCH_B] -> [DEEPOP]")
            branch_a_out = self.branch_a(x_tensor)
            risk_pred = branch_a_out["risk_score"]
            obs_logits = branch_a_out["technique_logits"]
            obs_probs = torch.softmax(obs_logits, dim=-1)
            pred_class_idx = int(obs_probs.argmax(dim=-1).item())
            obs_technique_confidence = float(obs_probs[0, -1, pred_class_idx].item()) if obs_probs.dim() == 3 else float(obs_probs[0, pred_class_idx].item())
            obs_technique = (
                self.technique_vocab[pred_class_idx]
                if pred_class_idx < len(self.technique_vocab)
                else "Benign"
            )
            raw_risk = float(risk_pred.item()) if risk_pred.numel() == 1 else float(risk_pred.mean().item())

        from control_backend.site_config import get_site_config
        site = get_site_config()
        ext_flows = [
            r for r in host_flows 
            if site.classify_ip(getattr(r, "src_ip", "")) == "external"
        ]
        if not ext_flows:
            ext_flows = [
                r for r in flows 
                if site.classify_ip(getattr(r, "src_ip", "")) == "external"
            ]
        ext_count = len(ext_flows)

        # --- SOC rule layer, isolated (spec 41) -------------------------------
        #
        # This block used to be inline, so the number on the dashboard was
        # `max(model_output, 0.40) + f(external_flow_count)` during every attack
        # demo -- a rule, not a prediction, and indistinguishable from one.
        #
        # It is retained because hybrid detection is legitimate in production
        # SOC tooling, but it is now: named, separable, reported alongside the
        # untouched model output, and switchable off via CYBERWORLD_DISABLE_RULES=1.
        #
        # `attack_active` is an operator/harness flag -- it is knowledge of the
        # answer. Any benchmark path must run with rules disabled (spec 42); the
        # offline harness in cyberworld_v4/benchmark.py never touches this code.
        ml_risk = float(raw_risk)
        ml_technique = obs_technique
        rule_risk: Optional[float] = None
        rules_applied = False

        if is_mitigated:
            obs_risk = max(0.02, raw_risk * 0.15)
            obs_technique = "Benign"
            rules_applied = True
        elif self.rules_enabled:
            if ext_count > 0 or attack_active:
                threat_boost = min(0.65, (max(1, ext_count) / 75.0) * 0.50 + 0.25)
                rule_risk = float(min(0.96, max(raw_risk, 0.40) + threat_boost))
                obs_risk = rule_risk
                ports_seen = {getattr(r, "dst_port", 0) for r in ext_flows}
                if len(ports_seen) >= 5:
                    obs_technique = "PortScan"
                elif any(getattr(r, "dst_port", 0) in (80, 443, 8080) for r in ext_flows):
                    obs_technique = "WebAttack"
                else:
                    obs_technique = "Exploit"
                rules_applied = True
            else:
                obs_risk = max(0.05, raw_risk * 0.4)
                rules_applied = True
        else:
            # Research mode: the displayed number IS the model output.
            obs_risk = ml_risk
            obs_technique = ml_technique

        curr_h = torch.from_numpy(h_emb).float().unsqueeze(0).to(self.device)
        h_state_history = self.h_state_history_by_target.setdefault(target_ip, [])
        h_state_history.append(curr_h)
        if len(h_state_history) > self.history_steps:
            h_state_history.pop(0)
        self.h_state_history = h_state_history
        while len(h_state_history) < self.history_steps:
            h_state_history.insert(0, torch.zeros_like(curr_h))
        h_seq = torch.stack(h_state_history, dim=1)

        with torch.no_grad():
            if hasattr(self.wdt, "rollout_with_uncertainty"):
                # NOTE: _radii holds per-step confidence bands. They are computed but
                # not yet carried in the prediction payload; wire them into the
                # PredictionEvent schema to draw confidence bands on the forecast.
                h_future, _radii = self.wdt.rollout_with_uncertainty(
                    h_seq, K=self.forecast_steps, stabilize_horizon=True
                )
            else:
                h_future = self.wdt.rollout(
                    h_seq, K=self.forecast_steps, stabilize_horizon=True
                )

            step_risks, _ = self.risk_head.forward_trajectory(h_future)
            fut_risks = [float(r) for r in step_risks.cpu().squeeze(0).numpy().tolist()]

            observed_tactic = TECHNIQUE_TO_MITRE.get(
                obs_technique,
                ("Benign" if obs_technique == "Benign" else "Recon", "", "", ""),
            )[0]
            coarse_cat, coarse_technique = self.consolidate_network_technique(
                observed_tactic,
                obs_technique,
            )
            obs_token_id = self.vocab.encode(coarse_cat, coarse_technique)
            obs_token_tensor = torch.tensor(
                [obs_token_id], dtype=torch.long, device=self.device
            )
            _, decoded_names, _, step_token_probs = self.deepop.forecast_sequence(
                h_future,
                max_steps=self.forecast_steps,
                observed_token=obs_token_tensor,
                return_probs=True,
            )
            deepop_confidences = step_token_probs[0] if step_token_probs else []

        forecast_techniques: List[str] = []
        if decoded_names and len(decoded_names) > 0:
            for item in decoded_names[0]:
                if isinstance(item, tuple):
                    c, t = item
                    forecast_techniques.append(f"{c}.{t}" if t else str(c))
                else:
                    forecast_techniques.append(str(item))
        while len(forecast_techniques) < self.forecast_steps:
            forecast_techniques.append(obs_technique if not is_mitigated else "Benign")

        if is_mitigated:
            fut_risks = [max(0.01, r * 0.1) for r in fut_risks]
            forecast_techniques = ["Benign"] * self.forecast_steps

        # Clamp risks
        obs_risk = float(np.clip(obs_risk, 0.0, 1.0))
        fut_risks = [float(np.clip(r, 0.0, 1.0)) for r in fut_risks]
        max_future = max(fut_risks) if fut_risks else obs_risk

        # Resolve rule-emitted labels to their real ATT&CK technique before lookup.
        mitre_key = RULE_TECHNIQUE_TO_ATTCK.get(obs_technique, obs_technique)
        mitre = TECHNIQUE_TO_MITRE.get(
            mitre_key,
            ("Unknown", obs_technique, "TA0000", "Model-predicted technique"),
        )
        alert = (not is_mitigated) and (
            obs_risk >= self.alert_threshold or max_future >= self.alert_threshold
        )

        explain = self._explain(x_tensor)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        now_ts = time.time()

        forecast_points = [
            ForecastPoint(
                horizon_seconds=(i + 1) * self.window_seconds,
                risk=round(fut_risks[i], 4),
                confidence=round(deepop_confidences[i], 4)
                if i < len(deepop_confidences)
                else None,
                predicted_stage=forecast_techniques[i],
            )
            for i in range(self.forecast_steps)
        ]

        stage_probs = None
        with torch.no_grad():
            probs = obs_probs[0, -1] if obs_probs.dim() == 3 else obs_probs[0]
            stage_probs = {
                self.technique_vocab[i]: round(float(probs[i].item()), 4)
                for i in range(min(len(self.technique_vocab), probs.numel()))
                if float(probs[i].item()) > 0.01
            }

        focus_ips, focus_edges = self._focus_binding(target_ip, flows)
        return PredictionEvent(
            type="prediction",
            mode="LIVE",
            timestamp=datetime.fromtimestamp(now_ts).isoformat() + "Z",
            wall_clock=time.strftime("%H:%M:%S", time.localtime(now_ts)),
            model=ModelMetadata(
                name="Antigravity-DualBranch-DeepOP",
                version="3.2-SOC",
                feature_count=27,
                history_steps=self.history_steps,
                window_seconds=self.window_seconds,
                forecast_steps=self.forecast_steps,
                checkpoint="host_wdt.pt + branch_a_lstm.pt + cwa_forecast_decoder.pt",
                threshold=self.alert_threshold,
            ),
            state=StateMetadata(
                window_id=int(window_id),
                sequence_ready=len(self.h_state_history) >= self.history_steps,
                packet_count=int(packet_count or sum(r.total_packets for r in flows)),
                active_flows=active_flows if active_flows is not None else len(flows),
                pipeline_latency_ms=float(pipeline_latency_ms),
                buffer_length=len(self.h_state_history),
            ),
            prediction=PredictionData(
                risk=round(obs_risk, 4),
                # Provenance of the number above, so a rule-driven demo cannot
                # be read as a model result (spec 21, 41).
                ml_risk=round(ml_risk, 4),
                rule_risk=(round(rule_risk, 4) if rule_risk is not None else None),
                rules_applied=rules_applied,
                max_future_risk=round(max_future, 4),
                hazard_score=round(max_future, 4),
                malicious_confidence=round(obs_risk, 4),
                precursor_confidence=round(max(0.0, max_future - obs_risk), 4),
                alert=alert,
                alert_level=self._alert_level(max(obs_risk, max_future)),
                threshold=self.alert_threshold,
                predicted_stage=obs_technique,
                mitre_tactic=mitre[0],
                mitre_technique=mitre[1],
                mitre_tactic_id=mitre[2],
                mitre_description=mitre[3],
                stage_probabilities=stage_probs,
                technique_confidence=round(obs_technique_confidence, 4),
                stage_provenance={
                    "TGNE": "h_emb(12)",
                    "BRANCH_A": "risk+technique",
                    "BRANCH_B": "H_hat trajectory",
                    "DEEPOP": "technique sequence"
                }
            ),
            forecast=forecast_points,
            explainability=explain,
            latency=LatencyData(
                telemetry_ms=round(float(pipeline_latency_ms), 2),
                inference_ms=round(inf_ms, 2),
                total_ms=round(float(pipeline_latency_ms) + inf_ms, 2),
            ),
            early_warning=EarlyWarningData(
                is_alert=alert,
                alert_timestamp=now_ts if alert else None,
                lead_time_seconds=self.forecast_steps * self.window_seconds if alert else None,
                target_milestone_desc=forecast_techniques[0] if alert else None,
            ),
            attack_active=attack_active,
            attack_phase=attack_phase,
            focus_ips=focus_ips,
            focus_edges=focus_edges,
            target_ip=target_ip or None,
            throughput=round(float(throughput), 2),
        )

    @staticmethod
    def _focus_binding(
        target_ip: str, flows: List[UnifiedFlowRecord]
    ) -> tuple[List[str], List[FocusEdge]]:
        """Bind prediction highlight to observed IPs/edges (no lab node IDs)."""
        ips: List[str] = []
        if target_ip:
            ips.append(target_ip)
        edge_keys = set()
        edges: List[FocusEdge] = []
        for r in flows or []:
            for ip in (r.src_ip, r.dst_ip):
                if ip and ip not in ips:
                    # Prefer peers of the scored host; cap list for UI
                    if not target_ip or ip == target_ip or r.src_ip == target_ip or r.dst_ip == target_ip:
                        ips.append(ip)
            if target_ip and (r.src_ip == target_ip or r.dst_ip == target_ip):
                key = (r.src_ip, r.dst_ip)
                if key not in edge_keys and r.src_ip and r.dst_ip:
                    edge_keys.add(key)
                    edges.append(FocusEdge(src=r.src_ip, dst=r.dst_ip))
        return ips[:12], edges[:24]


def select_primary_target(
    flows: List[UnifiedFlowRecord],
    fallback: Optional[str] = None,
) -> str:
    """Pick the most relevant host to score — site CIDRs + activity, not lab hardcoding."""
    from control_backend.site_config import get_site_config, select_primary_target_ip

    site = get_site_config()
    return select_primary_target_ip(
        flows,
        site=site,
        fallback=fallback if fallback is not None else (
            site.asset_ips()[0] if site.asset_ips() else ""
        ),
    )


def flows_from_span_dicts(raw_flows: List[Dict[str, Any]]) -> List[UnifiedFlowRecord]:
    """Convert SPAN flow snapshots → UnifiedFlowRecord with zero label leakage."""
    out: List[UnifiedFlowRecord] = []
    for f in raw_flows or []:
        try:
            out.append(
                UnifiedFlowRecord(
                    src_ip=str(f.get("src_ip", "")),
                    dst_ip=str(f.get("dst_ip", "")),
                    src_port=int(f.get("src_port", 0)),
                    dst_port=int(f.get("dst_port", 0)),
                    protocol=int(f.get("protocol", 6)),
                    start_time=float(f.get("start_time", time.time())),
                    end_time=float(f.get("end_time", time.time())),
                    fwd_bytes=int(f.get("fwd_bytes", 0)),
                    bwd_bytes=int(f.get("bwd_bytes", 0)),
                    fwd_packets=int(f.get("fwd_packets", 0)),
                    bwd_packets=int(f.get("bwd_packets", 0)),
                    raw_label="UNLABELED",
                    raw_label_source="SPAN_LIVE",
                    is_attack=False,
                    coarse_category=CoarseCategory.UNKNOWN.value,
                    attck_technique_ids=[],
                    metadata={"source": "span_live"},
                )
            )
        except Exception:
            continue
    return out


# Module singleton used by tests and telemetry service
model_adapter = AntigravityModelAdapter()
