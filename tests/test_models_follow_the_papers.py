"""Each model does what the paper it cites describes.

  BiTA      Makki Nayeri & Rezvani, arXiv:2604.22781      -> bita/
  GNN-LSTM  Vitulyova et al., Computers 14(8):301, 2025   -> branch_a_gnn_lstm/
  DeepOP    Zhang, Xue & Su, Electronics 14(2):257, 2025  -> deepop_decoder/

docs/PAPER_CONFORMANCE.md maps every equation to code and lists each
deviation with its reason. These tests pin the parts that were wrong: the BiTA
aggregator never ran, DeepOP had no encoder, Branch A's heads and loss were a
different model, and Branch B never saw the enriched state.
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BITA = os.path.join(REPO, "bita")
if BITA not in sys.path:
    sys.path.insert(0, BITA)


# ---------------------------------------------------------------------------
# BiTA
# ---------------------------------------------------------------------------

def _tgn(use_memory=True, n_nodes=20, n_edges=60, seed=0):
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder

    rng = np.random.default_rng(seed)
    src = rng.integers(1, n_nodes, n_edges)
    dst = rng.integers(1, n_nodes, n_edges)
    ts = np.sort(rng.random(n_edges) * 10.0)
    adj = [[] for _ in range(n_nodes)]
    for i in range(n_edges):
        adj[src[i]].append((dst[i], i, ts[i]))
        adj[dst[i]].append((src[i], i, ts[i]))
    torch.manual_seed(seed)
    tgn = ExtendedTGN(
        neighbor_finder=NeighborFinder(adj, uniform=False),
        node_features=rng.standard_normal((n_nodes, 12)).astype(np.float32),
        edge_features=rng.standard_normal((n_edges, 12)).astype(np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=use_memory,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5)
    return tgn, src, dst, ts, np.arange(n_edges)


def test_memoryless_encoder_contains_no_bita():
    """Why memory is now on by default: without it TGN builds no aggregator."""
    tgn, *_ = _tgn(use_memory=False)
    assert not hasattr(tgn, "message_aggregator")


def test_bita_aggregator_is_built_and_trained_with_memory():
    from modules.message_aggregator import BiTAAggregator

    tgn, src, dst, ts, eidx = _tgn()
    assert isinstance(tgn.message_aggregator, BiTAAggregator)
    for b in range(2):                      # batch 2 updates memory from batch 1
        sl = slice(b * 30, (b + 1) * 30)
        pos, neg, cat = tgn.compute_edge_probabilities_and_categories(
            src[sl], dst[sl], np.roll(dst[sl], 1), ts[sl], eidx[sl], n_neighbors=5)
        (pos.mean() - neg.mean() + cat.mean()).backward()
        tgn.memory.detach_memory()
    grads = [p.grad for p in tgn.message_aggregator.parameters()]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def _agg():
    from modules.message_aggregator import BiTAAggregator
    torch.manual_seed(0)
    return BiTAAggregator(message_dim=8, d_h=4, d_trans=8, n_heads=2, dropout=0.0).eval()


def _msg(t, peer, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(8, generator=g), torch.tensor(float(t)), peer)


def test_bita_readout_is_the_mean_over_incident_edges():
    """Eq. 9: a node's aggregate is the mean of its edges' contextual vectors."""
    agg = _agg()
    from collections import defaultdict
    msgs = defaultdict(list)
    msgs[1] = [_msg(0, 7, 1), _msg(1, 7, 2), _msg(2, 9, 3)]   # two edges: (1,7), (1,9)
    with torch.no_grad():
        nodes, h, _ = agg.aggregate([1], msgs)
        # Recompute the per-edge vectors by hand and average them.
        e7 = torch.stack([msgs[1][0][0], msgs[1][1][0]]).unsqueeze(0)
        e9 = msgs[1][2][0].view(1, 1, 8)
        def z(seq, dts):
            x = seq + agg.time_encoder(torch.tensor([dts]))
            _, hn = agg.bigru(x)
            return agg.W_e(torch.cat([hn[0], hn[1]], -1))
        # Edges attend in order of their last message: (1,7) ends at t=1 and
        # (1,9) at t=2, which is already the order below.
        e = torch.cat([z(e7, [1.0, 0.0]), z(e9, [0.0])], 0)
        ctx = agg.transformer(e.unsqueeze(0)).squeeze(0)
    assert nodes == [1]
    assert torch.allclose(h[0], ctx.mean(0), atol=1e-5)


def test_bita_attends_across_edges():
    """Eq. 7: an edge's representation depends on the OTHER edges in E_t."""
    from collections import defaultdict
    agg = _agg()
    a = defaultdict(list); a[1] = [_msg(0, 7, 1)]
    b = defaultdict(list); b[1] = [_msg(0, 7, 1)]; b[2] = [_msg(0, 8, 5)]
    with torch.no_grad():
        _, ha, _ = agg.aggregate([1], a)
        _, hb, _ = agg.aggregate([1, 2], b)
    assert not torch.allclose(ha[0], hb[0], atol=1e-6)


def test_bita_uses_the_time_encoding():
    """Eq. 3: the same messages at different spacings aggregate differently."""
    from collections import defaultdict
    agg = _agg()
    m1, m2 = _msg(0, 7, 1), _msg(1, 7, 2)
    far = (m2[0], torch.tensor(500.0), 7)
    a = defaultdict(list); a[1] = [m1, m2]
    b = defaultdict(list); b[1] = [m1, far]
    with torch.no_grad():
        assert not torch.allclose(agg.aggregate([1], a)[1], agg.aggregate([1], b)[1])


def test_unimplemented_aggregators_are_refused():
    from modules.message_aggregator import get_message_aggregator
    with pytest.raises(ValueError, match="unimplemented"):
        get_message_aggregator("stacked_bitransformer")


def test_memory_is_causal_in_streaming():
    """A window's own interactions never reach its own embedding's memory."""
    tgn, src, dst, ts, eidx = _tgn()
    tgn.reset_state()
    nodes = np.unique(np.r_[src[:30], dst[:30]])
    tgn.update_memory_for(nodes)
    before = tgn.memory.memory.data.clone()
    tgn.store_interactions(src[:30], dst[:30], ts[:30], eidx[:30])
    assert torch.equal(before, tgn.memory.memory.data), "storing must not update memory"
    tgn.update_memory_for(nodes)
    assert not torch.equal(before, tgn.memory.memory.data), "the next window must"


def test_window_boundaries_follow_the_clock_after_a_gap():
    from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter
    from data_unification.unified_schema import UnifiedFlowRecord

    def rec(t):
        return UnifiedFlowRecord(
            src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80, protocol=6,
            start_time=t, end_time=t + 0.1, fwd_bytes=1, bwd_bytes=1, fwd_packets=1,
            bwd_packets=1, raw_label="x", raw_label_source="t", is_attack=False,
            coarse_category="Benign", attck_technique_ids=[], metadata={})
    ev = FlowToTemporalEventAdapter(window_size_sec=2.0).process_records(
        [rec(0.0), rec(0.5), rec(10.3), rec(10.9)])
    starts = [round(w[0], 6) for w in ev.window_boundaries]
    assert starts == [0.0, 10.0], starts


def test_neighbours_are_limited_to_the_current_window():
    from data_unification.multi_dataset_stream import _WindowedNeighborFinder
    from utils.utils import NeighborFinder

    adj = [[], [(2, 0, 1.0), (3, 1, 5.0)], [(1, 0, 1.0)], [(1, 1, 5.0)]]
    nf = _WindowedNeighborFinder(NeighborFinder(adj, uniform=False))
    nf.lower_bound = 4.0
    nbrs, _, times = nf.get_temporal_neighbor(np.array([1]), np.array([6.0]), n_neighbors=2)
    kept = set(int(n) for n in nbrs[0] if n != 0)
    assert kept == {3}, "the t=1.0 neighbour predates the window and must be masked"


# ---------------------------------------------------------------------------
# DeepOP
# ---------------------------------------------------------------------------

def _deepop(**kw):
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    torch.manual_seed(0)
    return DeepOPForecastDecoder(d_latent=27, **kw).eval()


def test_deepop_default_is_the_paper_encoder_decoder():
    d = _deepop()
    assert d.use_obs_encoder and d.pos_encoding == "sinusoidal"
    assert d.window_mode == "partitioned"
    assert not (d.use_direct_head or d.use_prototypes or d.use_future_gate)
    assert d.default_continuity_bonus == 0.0, "no hand-set prior in the paper model"


def test_deepop_prediction_depends_on_the_observed_sequence():
    d = _deepop()
    V = d.vocab_size
    h = torch.randn(2, 5, 27)
    tgt = torch.randint(3, V, (2, 5))
    obs_a = torch.randint(3, V, (2, 16))
    obs_b = torch.randint(3, V, (2, 16))
    with torch.no_grad():
        assert not torch.allclose(d(h, tgt, obs_a), d(h, tgt, obs_b))


def test_deepop_decoder_stays_causal_with_the_encoder():
    d = _deepop()
    V = d.vocab_size
    h, obs = torch.randn(1, 5, 27), torch.randint(3, V, (1, 16))
    a = torch.randint(3, V, (1, 5))
    b = a.clone(); b[0, 3:] = (b[0, 3:] + 1) % V
    with torch.no_grad():
        la, lb = d(h, a, obs), d(h, b, obs)
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)


def test_partitioned_windows_match_eq_7():
    from deepop_decoder.cwa import CausalWindowAttention
    m = CausalWindowAttention(d_model=12, n_heads=3, window_sizes=[2, 4, 8])
    mask = m._partitioned_mask(6, 6, 4, "cpu")
    visible = (mask == 0)
    # position 5 is in block [4, 8): sees 4 and 5 only
    assert visible[5].nonzero().flatten().tolist() == [4, 5]
    # position 3 is in block [0, 4): sees 0..3
    assert visible[3].nonzero().flatten().tolist() == [0, 1, 2, 3]
    # single-step decoding aligns the query to the last key
    step = m._partitioned_mask(1, 6, 4, "cpu") == 0
    assert step[0].nonzero().flatten().tolist() == [4, 5]


def test_legacy_deepop_checkpoint_still_loads():
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    p = os.path.join(REPO, "saved_models", "deepop", "cwa_forecast_decoder.pt")
    if not os.path.exists(p):
        pytest.skip("no shipped DeepOP checkpoint")
    d = DeepOPForecastDecoder.from_checkpoint(torch.load(p, map_location="cpu", weights_only=False))
    assert not d.use_obs_encoder and d.default_continuity_bonus == 1.0


def test_new_deepop_round_trips_through_arch():
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    d = _deepop()
    ck = {"decoder_state_dict": d.state_dict(), "arch": d.arch_config()}
    d2 = DeepOPForecastDecoder.from_checkpoint(ck).eval()
    h, t, o = torch.randn(1, 5, 27), torch.randint(3, d.vocab_size, (1, 5)), torch.randint(3, d.vocab_size, (1, 16))
    with torch.no_grad():
        assert torch.allclose(d(h, t, o), d2(h, t, o))


# ---------------------------------------------------------------------------
# Branch A (GNN-LSTM)
# ---------------------------------------------------------------------------

def test_branch_a_paper_architecture():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    m = MultiTaskLSTM(input_dim=27, **MultiTaskLSTM.PAPER_ARCH)
    assert m.lstm.num_layers == 1 and m.lstm.hidden_size == 256
    assert isinstance(m.technique_head, torch.nn.Linear)
    out = m(torch.randn(3, 15, 27))
    g = out["gradation_score"]
    assert g.shape == (3,) and bool(((g >= 0) & (g <= 1)).all())


def test_branch_a_paper_loss_is_the_fixed_weighted_sum():
    import torch.nn.functional as F
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    torch.manual_seed(0)
    m = MultiTaskLSTM(input_dim=27, **MultiTaskLSTM.PAPER_ARCH).eval()
    out = m(torch.randn(4, 15, 27))
    batch = {"risk": torch.tensor([0.0, 0.7, 0.0, 0.9]),
             "technique": torch.tensor([0, 1, 0, 2]),
             "gradation": torch.tensor([0, 3, 0, 2])}
    total, met = m.compute_loss(out, batch)
    risk = F.binary_cross_entropy(out["risk_score"].clamp(1e-6, 1 - 1e-6), (batch["risk"] > 0).float())
    grad = F.mse_loss(out["gradation_score"], batch["gradation"].float() / 3)
    expect = 0.5 * risk + 0.3 * met["loss_tech"] + 0.2 * grad
    assert torch.allclose(total, expect, atol=1e-6)


def test_legacy_branch_a_checkpoint_still_loads():
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    p = os.path.join(REPO, "saved_models", "branch_a", "branch_a_lstm.pt")
    if not os.path.exists(p):
        pytest.skip("no shipped Branch A checkpoint")
    m = MultiTaskLSTM.from_checkpoint(torch.load(p, map_location="cpu", weights_only=False))
    assert m.hidden_dim == 64 and m.readout == "attention"


# ---------------------------------------------------------------------------
# Branch B sees the enriched state
# ---------------------------------------------------------------------------

def test_branch_b_world_state_is_the_enriched_tensor():
    import types
    from data_unification.trajectory_store import world_state
    snap = types.SimpleNamespace(embedding=np.arange(12, dtype=np.float32),
                                 temporal_attrs=np.arange(15, dtype=np.float32) + 100)
    s = world_state(snap)
    assert s.shape == (27,)
    assert s[12] == 100, "host attributes follow the TGNE latent"
