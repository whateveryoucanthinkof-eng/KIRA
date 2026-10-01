"""The evidence the K.I.R.A. console shows is real model output, or absent.

Covers what control_backend now sends over /ws beyond the verdict:
  * tactic lanes, derived with DeepOP's own consolidation;
  * DeepOP's top-3 continuations -- branch A must BE the served forecast;
  * the sensor's flow list and the full 27-D state vector.
Fields no model produces (branch hops, volume, per-branch risk) must stay
None so the console can say so, never be filled with a guess.
"""

import types

import pytest
import torch

from control_backend.evidence import flow_records, state_vector
from control_backend.forecast_branches import decode_branches
from control_backend.tactics import LANES, MODEL_LANES, lane_of, split_token


# --- lanes -----------------------------------------------------------------

@pytest.mark.parametrize("label, lane", [
    ("Benign", "Benign"),
    ("Benign.None", "Benign"),
    ("T1046", "Recon"),
    ("T1595", "Recon"),
    ("T1110", "CredentialAccess"),
    ("T1190", "InitialAccess"),
    ("T1071.001", "C2"),
    ("T1498.001", "Impact"),
    ("T1020", "Exfiltration"),
    ("Recon.T1595", "Recon"),
    ("CredentialAccess.T1110", "CredentialAccess"),
    ("Impact.T1498", "Impact"),
    ("", "Benign"),
    (None, "Benign"),
])
def test_lane_of_uses_the_models_own_consolidation(label, lane):
    assert lane_of(label) == lane


def test_every_lane_the_models_can_emit_is_drawn_in_kill_chain_order():
    from deepop_decoder.joint_vocab import NETWORK_MACRO_TECHNIQUES
    emitted = {c for c, _ in NETWORK_MACRO_TECHNIQUES if c != "Benign"}
    assert emitted == set(MODEL_LANES)
    assert [l for l in LANES if l in MODEL_LANES] == MODEL_LANES


def test_split_token_keeps_the_technique():
    assert split_token("Impact.T1498") == ("Impact", "T1498")
    assert split_token("T1071.001") == ("C2", "T1071.001")
    assert split_token("Benign.None") == ("Benign", None)


# --- forecast branches -----------------------------------------------------

def _decoder(bonus=0.0, seed=0):
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    torch.manual_seed(seed)
    m = DeepOPForecastDecoder(d_latent=27, default_continuity_bonus=bonus)
    m.eval()
    return m


@pytest.mark.parametrize("bonus", [0.0, 1.0])
def test_branch_a_is_the_served_forecast(bonus):
    m = _decoder(bonus)
    vocab = m.vocab
    torch.manual_seed(1)
    h = torch.randn(1, 5, 27)
    obs_tok = torch.tensor([vocab.encode("Recon", "T1595")])
    obs_seq = torch.tensor([[vocab.pad_idx] * 14 + [int(obs_tok)]])

    tokens, names, _, step_probs = m.forecast_sequence(
        h, max_steps=5, observed_token=obs_tok, observed_sequence=obs_seq, return_probs=True)
    branches = decode_branches(m, vocab, h, obs_seq, obs_tok, steps=5,
                               step_seconds=30.0, current_lane="Recon")

    assert [b.id for b in branches] == ["A", "B", "C"]
    served = [split_token(f"{c}.{t}") for c, t in names[0]]
    assert [(st.tactic_lane, st.technique) for st in branches[0].path] == served
    assert [st.probability for st in branches[0].path] == pytest.approx(step_probs[0], abs=1e-4)
    assert [st.horizon_seconds for st in branches[0].path] == [30.0, 60.0, 90.0, 120.0, 150.0]


def test_branch_probabilities_are_shares_of_the_top3_and_ranked():
    m = _decoder()
    h = torch.randn(1, 5, 27)
    branches = decode_branches(m, m.vocab, h, None, None, steps=5,
                               step_seconds=30.0, current_lane="Benign")
    assert sum(b.probability for b in branches) == pytest.approx(1.0, abs=1e-3)
    raws = [b.probability_raw for b in branches]
    assert raws == sorted(raws, reverse=True)
    # First steps differ: these are alternatives, not three copies.
    assert len({(b.path[0].tactic_lane, b.path[0].technique) for b in branches}) == 3


def test_fields_no_model_predicts_are_left_empty():
    m = _decoder()
    for b in decode_branches(m, m.vocab, torch.randn(1, 5, 27), None, None, steps=5,
                             step_seconds=30.0, current_lane="Benign"):
        assert b.hops is None and b.packets is None and b.bytes is None and b.peak_risk is None
        assert b.kind in ("escalation", "pivot", "backoff")
        if b.kind == "backoff":
            assert b.technique == "—" and b.stage == "Benign"


# --- flows and state vector ------------------------------------------------

class _Site:
    def classify_ip(self, ip):
        return "internal" if ip.startswith("10.") else "external"


def _raw(src, dst, start, proto=6, **kw):
    return {"src_ip": src, "dst_ip": dst, "src_port": 50000, "dst_port": 443,
            "protocol": proto, "start_time": start, "end_time": start + 0.25,
            "fwd_bytes": 100, "bwd_bytes": 200, "fwd_packets": 2, "bwd_packets": 3, **kw}


def test_flow_records_direction_path_and_order():
    raw = [
        _raw("10.0.2.10", "10.0.3.10", 5.0, proto=17),
        _raw("203.0.113.7", "10.0.2.10", 1.0, flags={"syn": 3, "ack": 1}),
        _raw("10.0.3.10", "198.51.100.1", 3.0, proto=47),
    ]
    out = flow_records(raw, window_id=7, target_ip="10.0.2.10", site=_Site())
    assert [f.ts_us for f in out] == [1_000_000, 3_000_000, 5_000_000]
    assert [f.id for f in out] == ["w7-0", "w7-1", "w7-2"]
    assert [f.direction for f in out] == ["inbound", "outbound", "internal"]
    assert [f.on_path for f in out] == [True, False, True]
    assert [f.protocol for f in out] == ["TCP", "OTHER", "UDP"]
    assert out[0].flags.syn == 3 and out[1].flags.syn == 0
    assert out[0].duration_ms == pytest.approx(250.0)


def test_flow_table_exports_flag_counts_the_model_never_reads():
    from control_backend.model_adapter import flows_from_span_dicts
    from telemetry.flow.flow_table import LiveFlowTable

    table = LiveFlowTable()
    for i, flags in enumerate(({"SYN": True}, {"SYN": True, "ACK": True}, {"ACK": True})):
        table.process_packet({
            "timestamp": 100.0 + i * 0.1, "src_ip": "10.0.2.10", "dst_ip": "10.0.3.10",
            "src_port": 40000, "dst_port": 80, "protocol": 6, "packet_length": 60,
            "tcp_flags": flags,
        })
    snap = table.snapshot_flows()
    assert snap[0]["flags"]["syn"] == 2 and snap[0]["flags"]["ack"] == 2
    rec = flows_from_span_dicts(snap)[0]
    assert not hasattr(rec, "flags")


def test_state_vector_carries_all_27_dims():
    from control_backend.model_adapter import FEATURE_GROUP_MAP
    from explainability.unified_explanation import FEATURE_NAMES

    dims = state_vector(FEATURE_NAMES, FEATURE_GROUP_MAP, [0.5] * 27, [1 / 27] * 27)
    assert len(dims) == 27
    assert dims[0].feature == "H_emb_0" and dims[0].group == "TGNE Latent"
    assert dims[12].group != "General"
    assert sum(d.attribution for d in dims) == pytest.approx(1.0, abs=1e-4)


def test_predict_window_sends_lanes_and_the_state_vector():
    from test_verdict_is_the_model import TARGET, _adapter, _attack_technique

    tech = _attack_technique()
    ev = _adapter(risk=0.3, technique=tech).predict_window(TARGET, [])
    assert ev.prediction.tactic_lane == lane_of(tech)
    assert all(f.tactic_lane == "Recon" for f in ev.forecast)
    assert len(ev.state_vector) == 27
    # The stub decoder is not a real module: no alternatives rather than fake ones.
    assert ev.branches is None


# --- campaigns and incidents ----------------------------------------------

def _event(risk, technique, future=(0.1,) * 5):
    from test_verdict_is_the_model import TARGET, _adapter
    return _adapter(risk=risk, technique=technique, future=future).predict_window(TARGET, [])


def test_attack_windows_become_a_campaign_on_the_scored_host():
    from control_backend.correlation_service import LiveCorrelation
    from test_verdict_is_the_model import TARGET

    corr = LiveCorrelation()
    campaign = None
    for t, tech in ((100.0, "T1046"), (102.0, "T1046"), (110.0, "T1190"), (120.0, "T1071")):
        campaign, _ = corr.observe(_event(0.8, tech), window_end=t)
    assert campaign is not None
    assert TARGET in campaign["involved_hosts"]
    assert campaign["lanes"] == LANES
    lanes_hit = {n["coarse_category"] for n in campaign["nodes"]}
    assert {"Recon", "InitialAccess", "C2"} <= lanes_hit
    assert len(campaign["edges"]) >= 2
    for n in campaign["nodes"]:
        assert set(n) >= {"max_risk_score", "mean_confidence", "provenance", "hit_count"}
    # The stub forecast is Recon.T1595 at every step: forecast nodes exist.
    assert campaign["has_forecast_components"]


def test_the_campaign_shown_is_the_hosts_latest_activity():
    """A later episode the heuristic does not link to an earlier one is shown,
    not the stale one (Recon -> Impact is a six-stage jump: no edge)."""
    from control_backend.correlation_service import LiveCorrelation
    corr = LiveCorrelation()
    corr.observe(_event(0.9, "T1046"), window_end=100.0)
    campaign, _ = corr.observe(_event(0.8, "T1498"), window_end=150.0)
    observed = [n for n in campaign["nodes"] if n["provenance"] == "OBSERVED"]
    assert "Impact" in {n["coarse_category"] for n in observed}


def test_benign_windows_make_no_campaign_and_no_incident():
    from control_backend.correlation_service import LiveCorrelation
    corr = LiveCorrelation()
    campaign, incidents = corr.observe(_event(0.05, "Benign"), window_end=100.0)
    # The stub forecast still names Recon, so a forecast-only node may exist,
    # but nothing observed is invented and no alert means no incident.
    assert all(n["provenance"] == "FORECAST" for n in (campaign or {}).get("nodes", []))
    assert incidents == []


def test_incidents_open_count_and_reopen_after_the_evidence_horizon():
    from control_backend.correlation_service import LiveCorrelation
    from test_verdict_is_the_model import TARGET

    corr = LiveCorrelation(horizon_sec=180.0)
    _, inc = corr.observe(_event(0.9, "T1110"), window_end=100.0)
    assert [i["id"] for i in inc] == ["INC-0001"]
    assert inc[0]["host"] == TARGET and inc[0]["status"] == "new"
    _, inc = corr.observe(_event(0.95, "T1110"), window_end=110.0)
    assert inc[0]["alertCount"] == 2 and inc[0]["peakRisk"] == pytest.approx(0.95, abs=1e-3)
    _, inc = corr.observe(_event(0.9, "T1110"), window_end=110.0 + 181.0)
    assert [i["id"] for i in inc] == ["INC-0002", "INC-0001"]
    _, inc = corr.observe(_event(0.9, "T1110"), window_end=300.0, contained_hosts={TARGET})
    assert inc[0]["status"] == "contained"
