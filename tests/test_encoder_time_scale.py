"""No absolute Unix time may reach the encoder's time encoding.

Found by the training guard on scripts/dry_run_plan.py: time_encoder.w carried
100% of the encoder's gradient norm (up to ~1e7, every step clipped). Two
paths fed it deltas of ~1e8-1e9 seconds:

  1. a node's FIRST memory message: memory starts at last_update = 0, so
     "time since last update" was the absolute timestamp;
  2. negatives drawn from every capture: a 2011 CTU-13 host scored at a 2018
     timestamp has neighbour deltas of ~2e8 s -- and is trivially separable,
     which also inflated link-prediction AP.

After both fixes the dry run's encoder gradient norm is ~60 and spread over
the model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "bita") not in sys.path:
    sys.path.insert(0, str(REPO / "bita"))

from utils.utils import NeighborFinder, RandEdgeSampler  # noqa: E402


def _tgn():
    from model.extentedtgn import ExtendedTGN
    torch.manual_seed(0)
    n0 = 8
    return ExtendedTGN(
        neighbor_finder=NeighborFinder([[] for _ in range(n0)], uniform=False),
        node_features=np.zeros((n0, 12), np.float32), edge_features=np.zeros((10, 12), np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=True,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5)


def test_a_first_contact_message_encodes_zero_elapsed_time():
    tgn = _tgn().eval()
    src, dst = np.array([1, 2]), np.array([3, 4])
    t = np.array([1.52e9, 1.52e9 + 5.0])
    emb = torch.zeros(2, 12)
    with torch.no_grad():
        _nodes, msgs = tgn.get_raw_messages(src, emb, dst, emb, t, np.array([1, 2]))
        zero_enc = tgn.time_encoder(torch.zeros(1, 1)).view(-1)
    for n in (1, 2):
        m = msgs[n][0][0]
        np.testing.assert_allclose(m[-zero_enc.numel():].numpy(), zero_enc.numpy(), atol=1e-6,
                                   err_msg="first contact must encode dt = 0, not the Unix time")


def test_after_an_update_the_real_elapsed_time_is_used():
    tgn = _tgn().eval()
    tgn.memory.last_update[1] = 1.52e9
    src, dst = np.array([1]), np.array([3])
    with torch.no_grad():
        _n, msgs = tgn.get_raw_messages(src, torch.zeros(1, 12), dst, torch.zeros(1, 12),
                                        np.array([1.52e9 + 30.0]), np.array([1]))
        want = tgn.time_encoder(torch.tensor([[30.0]])).view(-1)
    got = msgs[1][0][0][-want.numel():]
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-4)


def test_absolute_times_are_float64_end_to_end():
    """float32 resolves 128 s at 1.5e9: a 30 s gap computed as 0 (this test's
    predecessor failed exactly that way). Absolute times stay float64; only
    differences are narrowed to float32 for the time encoding."""
    from utils.utils import get_neighbor_finder

    class _D:
        sources = np.array([1, 1, 2]); destinations = np.array([2, 3, 3])
        edge_idxs = np.array([1, 2, 3]); timestamps = np.array([1.52e9, 1.52e9 + 30.0, 1.52e9 + 45.0])

    for uniform in (False, True):
        nf = get_neighbor_finder(_D(), uniform=uniform)
        _, _, et = nf.get_temporal_neighbor(np.array([1]), np.array([1.52e9 + 100.0]), n_neighbors=2)
        assert et.dtype == np.float64
        assert set(np.round(1.52e9 + 100.0 - et[0], 3)) <= {100.0, 70.0}

    tgn = _tgn()
    assert tgn.memory.last_update.dtype == torch.float64
    tgn.memory.ensure_capacity(20)
    assert tgn.memory.last_update.dtype == torch.float64 and tgn.memory.last_update.numel() == 20


def test_memory_update_keeps_sub_minute_resolution():
    tgn = _tgn().eval()
    src, dst = np.array([1]), np.array([3])
    with torch.no_grad():
        for t in (1.52e9, 1.52e9 + 7.0):
            nodes, msgs = tgn.get_raw_messages(src, torch.zeros(1, 12), dst, torch.zeros(1, 12),
                                               np.array([t]), np.array([1]))
            tgn.memory.store_raw_messages(nodes, msgs)
            tgn.update_memory(list(nodes), tgn.memory.messages)
            tgn.memory.clear_messages(list(nodes))
    assert float(tgn.memory.last_update[1]) == 1.52e9 + 7.0


def test_negatives_come_from_the_positive_edges_own_capture():
    node_group = np.array([-1, 0, 0, 0, 1, 1, 1, 2])
    s = RandEdgeSampler(src_list=[1, 4, 7], dst_list=[2, 3, 5, 6], seed=0, node_group=node_group)
    sources = np.array([1, 1, 4, 4, 1, 4] * 50)
    _, neg = s.sample(len(sources), sources=sources)
    assert set(neg[sources == 1]) <= {2, 3}
    assert set(neg[sources == 4]) <= {5, 6}
    # group 2 has no destination of its own: fall back to the full pool
    _, neg7 = s.sample(20, sources=np.full(20, 7))
    assert set(neg7) <= {2, 3, 5, 6}


def test_a_negative_is_never_the_positive_destination():
    """Dry run: the inductive pool of a capture was one server, so every
    negative equalled its positive and inductive AUC/AP were exactly 0.5000."""
    node_group = np.array([-1, 0, 0, 0, 1, 1, 1, 2])
    s = RandEdgeSampler(src_list=[1, 4], dst_list=[2, 3, 5], seed=0, node_group=node_group)
    n = 3000
    sources = np.where(np.arange(n) % 2 == 0, 1, 4)
    dests = np.where(sources == 1, np.where(np.arange(n) % 4 == 0, 2, 3), 5)
    _, neg = s.sample(n, sources=sources, destinations=dests)
    assert not np.any(neg == dests)
    # capture 0 has {2, 3}: the negative is the other one, always
    assert np.all(neg[(sources == 1) & (dests == 2)] == 3)
    # capture 1 has only {5} -- the positive -- so fall back to the full pool minus 5
    assert set(neg[sources == 4]) <= {2, 3} and len(set(neg[sources == 4])) == 2
    # no groups: uniform over the rest of the pool
    u = RandEdgeSampler([1], [2, 3, 5, 6], seed=1)
    _, neg_u = u.sample(4000, destinations=np.full(4000, 3))
    assert 3 not in set(neg_u)
    counts = np.array([np.sum(neg_u == v) for v in (2, 5, 6)])
    assert counts.min() > 1100, counts
    # a single-destination universe cannot avoid it; it must not crash
    one = RandEdgeSampler([1], [2], seed=0)
    assert set(one.sample(5, destinations=np.full(5, 2))[1]) == {2}


def test_without_groups_the_sampler_is_unchanged():
    a = RandEdgeSampler([1, 2], [3, 4, 5], seed=3)
    b = RandEdgeSampler([1, 2], [3, 4, 5], seed=3)
    np.testing.assert_array_equal(a.sample(50)[1], b.sample(50, sources=np.ones(50, int))[1])


def test_the_encoder_and_its_evaluation_pass_sources_to_the_sampler():
    train = (REPO / "bita" / "train.py").read_text(encoding="utf-8")
    ev = (REPO / "bita" / "evaluation" / "eval_edge_prediction_with_categories.py").read_text(encoding="utf-8")
    assert "train_rand_sampler.sample(size, sources=sources_batch, destinations=destinations_batch)" in train
    assert "node_group=_node_group" in train
    assert "negative_edge_sampler.sample(size, sources=sources_batch, destinations=destinations_batch)" in ev


def test_the_guard_names_the_tensor_that_carries_the_gradient():
    from cyberworld_v4.training_guard import TrainingGuard
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    logs = []
    g = TrainingGuard("t", [m], opt, clip_norm=1e-6, log=logs.append)
    x = torch.randn(4, 3) * 1e3
    g.backward_step((m(x) ** 2).mean())
    g.end_epoch(0.5, train_loss=1.0)
    rec = g.history[-1]
    assert rec["top_grads"] and 0 < rec["top_grads"][0]["share"] <= 1.0
    assert any("largest gradients" in line for line in logs)


def test_the_time_encoding_is_fixed_during_encoder_training():
    """time_encoder.w carried 91-97% of the encoder's squared gradient norm on
    the dry run once real (float64) deltas reached it: d/dw cos(w*dt) scales
    with dt in seconds. Fixed encoding (GraphMixer) unless asked otherwise."""
    src = (REPO / "bita" / "train.py").read_text(encoding="utf-8")
    freeze = src.index("_m.requires_grad_(False)")
    opt = src.index("optimizer = torch.optim.Adam([p for p in tgn.parameters() if p.requires_grad]")
    assert freeze < opt, "freeze before the optimizer is built"
    assert "if not args.learn_time_encoding:" in src and "'--learn_time_encoding'" in src

    from model.time_encoding import TimeEncode
    tgn = _tgn()
    encs = [m for m in tgn.modules() if isinstance(m, TimeEncode)]
    assert len(encs) == 2, "TGN's and BiTA's"
    for m in encs:
        m.requires_grad_(False)
    from cyberworld_v4.training_guard import TrainingGuard
    g = TrainingGuard("t", [tgn], torch.optim.Adam([p for p in tgn.parameters() if p.requires_grad]))
    guarded = {id(p) for p in g._params()}
    assert not any(id(p) in guarded for m in encs for p in m.parameters())
