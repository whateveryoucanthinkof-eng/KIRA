"""The number on the dashboard is the model's, and nothing else moves it.

Before this, control_backend/model_adapter.py ran the models and then threw the
answer away whenever the SOC rule layer was on -- and it was on by default:

  * any external flow floored risk at 0.40 + 0.2567 = 0.6567 and replaced the
    LSTM's technique with a port count (>=5 ports PortScan, 80/443 WebAttack,
    else Exploit), so ordinary browsing looked like an attack;
  * no external flow multiplied risk by 0.4, so internal lateral movement the
    model scored at 0.80 was shown as 0.32 -- under the alert line;
  * pressing ARM EXTERNAL (`attack_active`) raised risk with zero traffic;
  * a recorded mitigation emptied the host's flows, scaled risk by 0.15 and the
    forecast by 0.1, and forced every label to Benign, so a block that had not
    actually worked looked like a quiet host.

These tests stub the four networks so they run without checkpoints. The stub
Branch A returns a fixed risk and technique; everything asserted is what
predict_window does with that output.
"""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from control_backend import model_adapter as ma
from data_unification.unified_schema import UnifiedFlowRecord

TARGET = "10.0.2.10"
K = 5


class _BranchA(nn.Module):
    """Constant risk and technique, but a real module so _explain can backprop."""

    def __init__(self, risk: float, technique_idx: int, n_classes: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(27))
        self.logit = float(np.log(risk / (1 - risk)))
        self.technique_idx = technique_idx
        self.n_classes = n_classes

    def forward(self, x):
        z = (x[:, -1, :] * self.w).sum(-1) + self.logit
        logits = torch.full((x.shape[0], self.n_classes), -10.0)
        logits[:, self.technique_idx] = 10.0
        return {"risk_score": torch.sigmoid(z), "technique_logits": logits}


def _adapter(risk: float, technique: str, future=(0.1,) * K, rules=False):
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB

    a = ma.AntigravityModelAdapter.__new__(ma.AntigravityModelAdapter)
    a.device = "cpu"
    a.h_state_history, a.feature_history = [], []
    a.h_state_history_by_target, a.feature_history_by_target = {}, {}
    a.technique_history_by_target = {}
    a.world_state_dim = 27
    a.alert_threshold = 0.65
    a.rules_enabled = rules
    a.window_seconds, a.history_steps, a.forecast_steps = 2.0, 15, K
    a.forecast_step_seconds = 30.0
    a.checkpoint_contract = {}
    a.technique_vocab = TECHNIQUE_VOCAB

    emb = types.SimpleNamespace(embedding=np.zeros(12, dtype=np.float32))
    a.extractor = types.SimpleNamespace(
        compute_host_temporal_attributes=lambda **_: np.zeros(15, dtype=np.float32),
        extract_trajectories=lambda flows: {TARGET: [emb]},
        window_size_sec=2.0,
    )
    a.branch_a = _BranchA(risk, TECHNIQUE_VOCAB.index(technique), len(TECHNIQUE_VOCAB))
    a.wdt = types.SimpleNamespace(
        rollout_with_uncertainty=lambda h, K, stabilize_horizon: (torch.zeros(1, K, h.shape[-1]), None))
    a.risk_head = types.SimpleNamespace(
        forward_trajectory=lambda h: (torch.tensor([list(future)]), None))
    a.consolidate_network_technique = lambda tactic, tech: ("Benign", "")
    a.vocab = types.SimpleNamespace(encode=lambda c, t: 0, pad_idx=1, bos_idx=2)
    a.deepop = types.SimpleNamespace(
        forecast_sequence=lambda *a_, **k_: (None, [[("Recon", "T1046")] * K], None, [[0.5] * K]))
    return a


def _flow(src, dst, dport=443):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=50000, dst_port=dport, protocol=6,
        start_time=100.0, end_time=101.0, fwd_bytes=500, bwd_bytes=1500,
        fwd_packets=5, bwd_packets=5, raw_label="UNLABELED",
        raw_label_source="TEST", is_attack=False, coarse_category="Unknown",
        attck_technique_ids=[], metadata={},
    )


EXTERNAL = "203.0.113.7"          # TEST-NET-3: outside every site CIDR
INTERNAL_PEER = "10.0.3.10"


def _benign_technique():
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB
    return "Benign" if "Benign" in TECHNIQUE_VOCAB else TECHNIQUE_VOCAB[0]


def _attack_technique():
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB
    return next(t for t in TECHNIQUE_VOCAB if t != _benign_technique())


def test_external_benign_traffic_is_not_forced_to_elevated():
    """One external web flow the model calls benign stays benign."""
    a = _adapter(risk=0.02, technique=_benign_technique())
    ev = a.predict_window(TARGET, [_flow(EXTERNAL, TARGET, 443)])
    assert ev.prediction.risk == pytest.approx(0.02, abs=1e-3)
    assert ev.prediction.risk != pytest.approx(0.6567, abs=1e-3)
    assert ev.prediction.predicted_stage == _benign_technique()
    assert not ev.prediction.alert


def test_internal_lateral_movement_is_not_damped():
    """0.80 on internal-only traffic is shown as 0.80, not 0.32."""
    tech = _attack_technique()
    a = _adapter(risk=0.80, technique=tech)
    ev = a.predict_window(TARGET, [_flow(TARGET, INTERNAL_PEER, 445)])
    assert ev.prediction.risk == pytest.approx(0.80, abs=1e-3)
    assert ev.prediction.predicted_stage == tech
    assert ev.prediction.alert


def test_arm_external_button_does_not_move_the_number():
    flows = [_flow(TARGET, INTERNAL_PEER, 445)]
    off = _adapter(risk=0.10, technique=_benign_technique()).predict_window(
        TARGET, flows, attack_active=False)
    on = _adapter(risk=0.10, technique=_benign_technique()).predict_window(
        TARGET, flows, attack_active=True, attack_phase="EXTERNAL")
    assert on.prediction.risk == off.prediction.risk
    assert on.prediction.alert == off.prediction.alert
    assert on.attack_active is True        # still echoed for display


def test_zero_traffic_with_attack_armed_is_not_a_threat():
    ev = _adapter(risk=0.05, technique=_benign_technique()).predict_window(
        TARGET, [], attack_active=True)
    assert ev.prediction.risk == pytest.approx(0.05, abs=1e-3)
    assert not ev.prediction.alert


def test_rule_layer_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CYBERWORLD_ENABLE_RULES", raising=False)
    assert ma._env_flag("CYBERWORLD_ENABLE_RULES") is False
    ev = _adapter(risk=0.02, technique=_benign_technique()).predict_window(
        TARGET, [_flow(EXTERNAL, TARGET, 443)])
    assert ev.prediction.rule_risk is None
    assert ev.prediction.rule_technique is None


def test_enabled_rule_layer_is_advisory_only():
    """With rules on, their opinion is reported but the verdict is unchanged."""
    flows = [_flow(EXTERNAL, TARGET, p) for p in (22, 80, 443, 3389, 8080)]
    ev = _adapter(risk=0.02, technique=_benign_technique(), rules=True).predict_window(TARGET, flows)
    p = ev.prediction
    assert p.rule_risk is not None and p.rule_risk > 0.6
    assert p.rule_technique == "PortScan"
    assert p.risk == pytest.approx(0.02, abs=1e-3)       # verdict untouched
    assert p.predicted_stage == _benign_technique()
    assert p.ml_technique == _benign_technique()
    assert p.rules_applied is False
    assert p.risk_source == "model"
    assert not p.alert


def test_provenance_fields_agree_with_the_verdict():
    tech = _attack_technique()
    p = _adapter(risk=0.42, technique=tech).predict_window(
        TARGET, [_flow(TARGET, INTERNAL_PEER)]).prediction
    assert p.ml_risk == p.risk
    assert p.ml_technique == p.predicted_stage == tech


def test_recorded_mitigation_does_not_hide_ongoing_attack():
    """If traffic a block should stop is still there, it is scored and flagged."""
    tech = _attack_technique()
    flows = [_flow(EXTERNAL, TARGET, 443)]
    ev = _adapter(risk=0.90, technique=tech, future=(0.9,) * K).predict_window(
        TARGET, flows, mitigation_recorded=True, mitigation_bypass_flows=1)
    p = ev.prediction
    assert p.risk == pytest.approx(0.90, abs=1e-3)       # not x0.15
    assert p.predicted_stage == tech                     # not forced Benign
    assert p.alert
    assert p.mitigation_status == "traffic_persists"
    assert p.mitigation_bypass_flows == 1
    assert all(f.risk == pytest.approx(0.9, abs=1e-3) for f in ev.forecast)  # not x0.1
    assert all(f.predicted_stage != "Benign" for f in ev.forecast)


def test_quiet_after_mitigation_is_reported_as_such():
    p = _adapter(risk=0.03, technique=_benign_technique()).predict_window(
        TARGET, [], mitigation_recorded=True, mitigation_bypass_flows=0).prediction
    assert p.mitigation_status == "recorded_quiet"


def test_forecast_horizons_use_the_forecast_step_not_the_window():
    ev = _adapter(risk=0.1, technique=_benign_technique()).predict_window(TARGET, [])
    assert [f.horizon_seconds for f in ev.forecast] == [30.0 * (k + 1) for k in range(K)]


def test_lead_time_is_derived_from_the_forecast_not_a_constant():
    # Current window below threshold; forecast crosses at step 3 (90 s).
    ev = _adapter(risk=0.2, technique=_benign_technique(),
                  future=(0.1, 0.3, 0.7, 0.8, 0.9)).predict_window(TARGET, [])
    assert ev.early_warning.lead_time_seconds == pytest.approx(90.0)
    # Never crosses: no lead time, rather than K x window on every alert.
    ev = _adapter(risk=0.2, technique=_benign_technique(),
                  future=(0.1,) * K).predict_window(TARGET, [])
    assert ev.early_warning.lead_time_seconds is None


def test_importing_the_module_does_not_load_checkpoints():
    assert ma._singleton is None or isinstance(ma._singleton, ma.AntigravityModelAdapter)
    assert "model_adapter" not in vars(ma)
