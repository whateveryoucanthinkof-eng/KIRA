"""The serving pipeline composes end to end, with real (untrained) models.

    2 s window of flows
      -> interaction graph of that window
      -> TGNE-TA with BiTA memory                      12-D latent per host
      -> [latent ; 15 host attributes]                 27-D state s(t)
      -> Branch A (LSTM over 15 x s)                   risk, technique, gradation
      -> Branch B (world model over 15 x s)            s(t+1..t+5)
      -> DeepOP (encoder: Branch A's techniques,
                 decoder: cross-attends Branch B)      next ATT&CK tokens

Uses the paper architectures with random weights, so it checks shapes, wiring
and state handling -- not accuracy.
"""

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "bita") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "bita"))

from data_unification.unified_schema import UnifiedFlowRecord


def _flows(t0, n=12, seed=0):
    rng = np.random.default_rng(seed)
    hosts = ["10.0.2.10", "10.0.2.11", "10.0.3.10", "10.0.3.20", "203.0.113.7"]
    out = []
    for i in range(n):
        a, b = rng.choice(len(hosts), 2, replace=False)
        t = t0 + float(rng.random()) * 1.9
        out.append(UnifiedFlowRecord(
            src_ip=hosts[a], dst_ip=hosts[b], src_port=40000 + i, dst_port=int(rng.choice([22, 80, 443])),
            protocol=6, start_time=t, end_time=t + 0.05, fwd_bytes=int(rng.integers(100, 5000)),
            bwd_bytes=int(rng.integers(100, 5000)), fwd_packets=int(rng.integers(1, 20)),
            bwd_packets=int(rng.integers(1, 20)), raw_label="UNLABELED", raw_label_source="TEST",
            is_attack=False, coarse_category="Unknown", attck_technique_ids=[], metadata={}))
    return out


def _adapter():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB
    from branch_b_world_model.infiltration_head import InfiltrationRiskHead
    from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
    from control_backend import model_adapter as ma
    from data_unification.multi_dataset_stream import HostTrajectoryExtractor
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    from deepop_decoder.joint_vocab import consolidate_network_technique, get_joint_vocab
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder

    torch.manual_seed(0)
    n0 = 64
    tgn = ExtendedTGN(
        neighbor_finder=NeighborFinder([[] for _ in range(n0)], uniform=False),
        node_features=np.zeros((n0, 12), np.float32), edge_features=np.zeros((10, 12), np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=True,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5).eval()

    a = ma.AntigravityModelAdapter.__new__(ma.AntigravityModelAdapter)
    a.device = "cpu"
    a.h_state_history, a.feature_history = [], []
    a.h_state_history_by_target, a.feature_history_by_target = {}, {}
    a.technique_history_by_target = {}
    a.alert_threshold, a.rules_enabled = 0.65, False
    a.window_seconds, a.history_steps, a.forecast_steps = 2.0, 15, 5
    a.forecast_step_seconds = 30.0
    a.checkpoint_contract = {}
    a.technique_vocab = TECHNIQUE_VOCAB
    a.tgn = tgn
    a.extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=2.0, persist_memory=True)
    a.branch_a = MultiTaskLSTM(input_dim=27, **MultiTaskLSTM.PAPER_ARCH).eval()
    a.world_state_dim = 27
    a.wdt = HostWorldDynamicsTransformer(d_latent=27, d_model=64).eval()
    a.risk_head = InfiltrationRiskHead(d_latent=27, hidden_dim=32).eval()
    a.vocab = get_joint_vocab()
    a.consolidate_network_technique = consolidate_network_technique
    a.deepop = DeepOPForecastDecoder(d_latent=27, vocab_size=a.vocab.vocab_size).eval()
    return a


def test_the_whole_pipeline_runs_window_after_window():
    a = _adapter()
    target = "10.0.2.10"
    events = []
    for w in range(4):
        events.append(a.predict_window(target, _flows(1000.0 + 2.0 * w, seed=w), window_id=w))

    ev = events[-1]
    assert 0.0 <= ev.prediction.risk <= 1.0
    assert ev.prediction.risk == ev.prediction.ml_risk
    assert len(ev.forecast) == 5
    assert [f.horizon_seconds for f in ev.forecast] == [30.0, 60.0, 90.0, 120.0, 150.0]
    # Branch B's input is the 27-D state, one per window seen so far.
    assert a.h_state_history_by_target[target][-1].shape[-1] == 27
    # DeepOP's encoder input is Branch A's technique per window.
    assert len(a.technique_history_by_target[target]) == 4


def test_chunked_extraction_feeds_every_flow_to_memory_exactly_once():
    """Chunks overlap (the next re-reads the previous one's tail) and each
    re-windows on its own grid; memory must still see each flow once."""
    from data_unification.chunked_extraction import extract_trajectories_chunked
    from data_unification.multi_dataset_stream import HostTrajectoryExtractor

    a = _adapter()
    ex = HostTrajectoryExtractor(tgne_ta_model=a.tgn, window_size_sec=2.0)
    stored = []
    real = a.tgn.store_interactions
    def spy(s, d, t, e):
        stored.extend(np.asarray(t).tolist())
        return real(s, d, t, e)
    a.tgn.store_interactions = spy

    records = []
    for w in range(40):
        records += _flows(5000.0 + 2.0 * w, n=6, seed=100 + w)
    records.sort(key=lambda r: r.start_time)
    extract_trajectories_chunked(ex, iter(records), chunk_size=50, history_seconds=10.0)

    assert sorted(stored) == sorted(r.start_time for r in records)


def test_encoder_memory_carries_across_windows_and_resets():
    a = _adapter()
    a.predict_window("10.0.2.10", _flows(1000.0, seed=1))
    a.predict_window("10.0.2.10", _flows(1002.0, seed=2))
    assert float(a.tgn.memory.memory.abs().sum()) > 0, "memory must have been updated"
    ids_before = dict(a.extractor._event_adapter.ip_to_id)
    a.predict_window("10.0.2.10", _flows(1004.0, seed=3))
    for ip, i in ids_before.items():
        assert a.extractor._event_adapter.ip_to_id[ip] == i, "host ids must stay stable"
    a.reset_history()
    assert float(a.tgn.memory.memory.abs().sum()) == 0.0
    assert a.extractor._event_adapter is None
