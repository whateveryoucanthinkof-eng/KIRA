"""The CIC-2018 -> CIC-2017 protocol and the three-way encoder comparison.

  1. classes CIC-2017 has and CIC-2018 lacks are reported separately;
  2. CIC-2018 is read from PCAP (real host addresses), CIC-2017 with its own
     parser, by every trainer through one module;
  3. every tuning split under cross_year is CIC-2018; CIC-2017 is only ever test;
  4. the IP-feature ablation is recorded in the encoder and enforced on load.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "bita") not in sys.path:
    sys.path.insert(0, str(REPO / "bita"))


# ---------------------------------------------------------------------------
# 3. the split
# ---------------------------------------------------------------------------

def test_cross_year_tunes_only_on_2018_and_tests_only_on_2017():
    from data_unification.split_policy import load_lock, split_of
    lock = load_lock()
    for cap in lock["CIC2017"]:
        assert split_of("CIC2017", cap, "cross_year") == "test"
    for ds in ("CIC2018", "PCAP2018"):
        got = {split_of(ds, cap, "cross_year") for cap in lock[ds]}
        assert got == {"train", "val"}, (ds, got)
        for cap, locked in lock[ds].items():
            assert (split_of(ds, cap, "cross_year") == "train") == (locked == "train")
    for cap in lock["CTU13"]:
        assert split_of("CTU13", cap, "cross_year") is None


def test_frozen_scheme_is_unchanged():
    from data_unification.split_policy import load_lock, split_of
    lock = load_lock()
    for ds, m in lock.items():
        for cap, sp in m.items():
            assert split_of(ds, cap) == sp


def test_partition_drops_corpora_the_scheme_does_not_use():
    from data_unification.split_policy import partition_paths
    paths = [Path("/x/Monday-WorkingHours.pcap_ISCX.csv"), Path("/x/1/capture20110810.binetflow")]
    part = partition_paths(paths, scheme="cross_year")
    assert [p.name for p in part["test"]] == ["Monday-WorkingHours.pcap_ISCX.csv"]
    assert not part["train"] and not part["val"]


# ---------------------------------------------------------------------------
# 2. sources
# ---------------------------------------------------------------------------

def _corpus(tmp_path):
    from data_unification.split_policy import load_lock
    lock = load_lock()
    c17, c18, pc = tmp_path / "c17", tmp_path / "c18", tmp_path / "pcap"
    for d in (c17, c18, pc):
        d.mkdir()
    for n in lock["CIC2017"]:
        (c17 / n).write_text("x")
    for n in lock["CIC2018"]:
        (c18 / n).write_text("x")
    for n in lock["PCAP2018"]:
        (pc / n).mkdir()
    return c17, c18, pc


def test_cross_year_refuses_cic2018_csvs(tmp_path):
    from data_unification.training_sources import CorpusRefused, discover_captures
    c17, c18, _ = _corpus(tmp_path)
    with pytest.raises(CorpusRefused, match="fabricate host IPs"):
        discover_captures(scheme="cross_year", cic2017_dir=c17, cic2018_dir=c18)
    caps = discover_captures(scheme="cross_year", cic2017_dir=c17, cic2018_dir=c18,
                             allow_cic2018_csv=True)
    assert caps["train"] and caps["test"]


def test_cross_year_discovers_pcap_days_including_misspelled_ones(tmp_path):
    from data_unification.training_sources import discover_captures
    c17, _, pc = _corpus(tmp_path)
    caps = discover_captures(scheme="cross_year", cic2017_dir=c17, pcap2018_root=pc)
    names = {c.name for sp in caps.values() for c in sp}
    assert {"fri_23_pacap", "thu_15_pacap"} <= names
    assert all(c.dataset == "PCAP2018" for c in caps["train"] + caps["val"])
    assert all(c.dataset == "CIC2017" for c in caps["test"]) and len(caps["test"]) == 8


def test_pcap_and_csv_2018_together_are_refused(tmp_path):
    from data_unification.training_sources import CorpusRefused, discover_captures
    _, c18, pc = _corpus(tmp_path)
    with pytest.raises(CorpusRefused, match="read twice"):
        discover_captures(cic2018_dir=c18, pcap2018_root=pc)


def test_cic2017_files_use_the_cic2017_parser(monkeypatch, tmp_path):
    from data_unification import training_sources as ts
    import data_unification.cic2017_adapter as a17
    used = []

    class Fake:
        def parse_file(self, path, max_rows=None):
            used.append(path)
            return iter([])
    monkeypatch.setattr(a17, "CIC2017Adapter", Fake)
    cap = ts.Capture("CIC2017", "Monday-WorkingHours.pcap_ISCX.csv",
                     tmp_path / "Monday-WorkingHours.pcap_ISCX.csv", "test")
    ts.read_capture(cap, window_seconds=2.0)
    assert used, "CIC-2017 must be parsed by CIC2017Adapter (12-hour clock repair)"


def test_every_trainer_reads_through_the_shared_module():
    for rel in ("scripts/retrain_branch_a_live.py", "scripts/retrain_future_models_live.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "training_sources" in src, rel
    assert "pcap2018_root" in (REPO / "bita" / "train.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. unseen classes
# ---------------------------------------------------------------------------

def test_unseen_classes_are_reported_not_averaged_in():
    from cyberworld_v4.cross_dataset import unseen_class_report
    # classes: 0 Benign, 1 DoS, 2 PortScan (never in training)
    train = np.array([1000, 200, 0])
    cm = np.array([[90, 10, 0],
                   [5, 45, 0],
                   [30, 20, 0]])        # PortScan predicted as Benign 60%, DoS 40%
    r = unseen_class_report(train, cm, {0: "Benign", 1: "DoS", 2: "PortScan"})
    assert [u["class"] for u in r["unseen_classes"]] == ["PortScan"]
    u = r["unseen_classes"][0]
    assert u["test_support"] == 50 and u["recall"] == 0.0
    assert u["predicted_as"][0] == {"class": "Benign", "fraction": 0.6}
    assert r["macro_f1_seen"] > r["macro_f1_all"], "unseen misses must not drag the seen score"
    assert r["test_fraction_in_unseen_classes"] == pytest.approx(50 / 200)
    assert r["accuracy_seen"] == pytest.approx(135 / 150)


def test_branch_a_checkpoint_records_its_training_classes():
    src = (REPO / "scripts" / "retrain_branch_a_live.py").read_text(encoding="utf-8")
    assert '"train_technique_counts"' in src
    assert "unseen_class_report(" in src
    src = (REPO / "scripts" / "retrain_future_models_live.py").read_text(encoding="utf-8")
    assert '"train_token_counts"' in src


# ---------------------------------------------------------------------------
# 4. IP features, and the encoder round trip
# ---------------------------------------------------------------------------

def test_cross_network_group_keeps_only_network_invariant_flags(monkeypatch):
    from data_unification import ip_features as ipf
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "cross_network")
    ipf.reset_node_ablation_cache()
    try:
        dropped = set(ipf.ablated_node_features())
        assert dropped == {"is_private", "is_global", "octet1", "octet2", "octet3", "octet4"}
        a = ipf.build_node_feature_matrix({"172.31.64.10": 1}, 2)[1]
        b = ipf.build_node_feature_matrix({"192.168.10.5": 1}, 2)[1]
        assert np.array_equal(a, b), "two private hosts on different networks must look alike"
    finally:
        monkeypatch.delenv("CYBERWORLD_ABLATE_NODE_FEATURES")
        ipf.reset_node_ablation_cache()


def _save_memory_encoder(tmp_path, ablated_nodes):
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder
    from data_unification.tgne_features import SCHEMA_VERSION
    torch.manual_seed(0)
    n = 30
    tgn = ExtendedTGN(neighbor_finder=NeighborFinder([[] for _ in range(n)]),
                      node_features=np.zeros((n, 12), np.float32),
                      edge_features=np.zeros((5, 12), np.float32), device="cpu",
                      n_layers=1, n_heads=2, dropout=0.0, use_memory=True, message_dimension=100,
                      memory_dimension=12, message_function="identity",
                      aggregator_type="bigru_transformer", memory_updater_type="gru",
                      num_categories=5)
    path = tmp_path / "enc.pth"
    torch.save(tgn.state_dict(), path)
    cfg = {"n_layers": 1, "n_heads": 2, "dropout": 0.0, "use_memory": True,
           "message_dimension": 100, "memory_dimension": 12,
           "embedding_module_type": "graph_attention", "message_function": "identity",
           "aggregator_type": "bigru_transformer", "memory_updater_type": "gru",
           "num_categories": 5, "edge_feat_dim": 12, "node_feat_dim": 12,
           "feature_schema_version": SCHEMA_VERSION, "ablated_edge_features": [],
           "ablated_node_features": ablated_nodes}
    (tmp_path / "enc_config.json").write_text(json.dumps(cfg))
    return tgn, path


def test_memory_encoder_loads_despite_a_different_node_count(tmp_path):
    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
    trained, path = _save_memory_encoder(tmp_path, [])
    loaded = build_or_load_tgne_ta(checkpoint_path=str(path))
    assert loaded.use_memory
    k = "message_aggregator.W_e.weight"
    assert torch.equal(loaded.state_dict()[k], trained.state_dict()[k])


def test_node_ablation_mismatch_is_refused(tmp_path, monkeypatch):
    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
    from data_unification import ip_features as ipf
    _, path = _save_memory_encoder(tmp_path, ["is_global", "is_private", "octet1", "octet2",
                                             "octet3", "octet4"])
    ipf.reset_node_ablation_cache()
    with pytest.raises(ValueError, match="node features"):
        build_or_load_tgne_ta(checkpoint_path=str(path))
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "cross_network")
    ipf.reset_node_ablation_cache()
    try:
        build_or_load_tgne_ta(checkpoint_path=str(path))
    finally:
        monkeypatch.delenv("CYBERWORLD_ABLATE_NODE_FEATURES")
        ipf.reset_node_ablation_cache()


def test_init_from_copies_weights_but_not_heads_or_memory(tmp_path):
    import train as bita_train
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder
    src, path = _save_memory_encoder(tmp_path, [])
    torch.manual_seed(1)
    dst = ExtendedTGN(neighbor_finder=NeighborFinder([[] for _ in range(50)]),
                      node_features=np.zeros((50, 12), np.float32),
                      edge_features=np.zeros((5, 12), np.float32), device="cpu",
                      n_layers=1, n_heads=2, dropout=0.0, use_memory=True, message_dimension=100,
                      memory_dimension=12, message_function="identity",
                      aggregator_type="bigru_transformer", memory_updater_type="gru",
                      num_categories=7)                 # a different label set
    head_before = dst.category_predictor[-1].weight.clone()
    bita_train._init_from_checkpoint(dst, str(path), "cpu")
    k = "message_aggregator.W_e.weight"
    assert torch.equal(dst.state_dict()[k], src.state_dict()[k])
    assert torch.equal(dst.category_predictor[-1].weight, head_before)
    assert dst.memory.memory.shape[0] == 50


def test_warden_encoder_gets_ip_node_features(tmp_path):
    import pandas as pd
    import train as bita_train
    rows = [
        ("2019-03-11T00:00:0%dZ" % i, "8.8.8.%d" % i, "10.0.0.%d" % (i % 3), "tcp", "22",
         ["Recon.Scanning", "Availability.DoS", "Attempt.Login"][i % 3], i + 1)
        for i in range(9)]
    pd.DataFrame(rows, columns=["DetectTime", "SourceIP", "TargetIP", "Proto", "Port", "Category",
                                "FlowCount"]).to_csv(tmp_path / "x_March_e.csv", index=False)
    g, _e, nodes, _m = bita_train.load_and_preprocess_dataset(str(tmp_path), resample_to_median=False)
    used = np.unique(np.r_[g.u.values, g.i.values])
    assert (nodes[used].sum(axis=1) > 1.0).all(), "every attacker/victim node carries IP features"
    _g, _e, zeros, _m = bita_train.load_and_preprocess_dataset(
        str(tmp_path), resample_to_median=False, ip_node_features=False)
    assert not zeros.any()


# ---------------------------------------------------------------------------
# the comparison and evaluation scripts
# ---------------------------------------------------------------------------

def test_comparison_summary_ranks_on_seen_class_f1(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_encoder_comparison as rec
    a = types.SimpleNamespace(out=tmp_path)
    (tmp_path / "results").mkdir()
    for arm, f1, auc in (("warden", 0.31, 0.70), ("cic2018", 0.52, 0.81), ("warden_ft", 0.55, 0.79)):
        (tmp_path / "results" / f"{arm}__full.json").write_text(json.dumps({
            "validation": {"tech_macro_f1": 0.6},
            "test": {"tech_accuracy": 0.8, "tech_majority_baseline": 0.78, "risk_auc": auc,
                     "unseen_class_report": {"macro_f1_seen": f1, "macro_f1_all": f1 - 0.1,
                                             "test_fraction_in_unseen_classes": 0.06,
                                             "unseen_classes": [{"class": "T1046"}]}}}))
    s = rec.summarise(a, list(rec.ARMS), ["full"])
    assert s["ranking"][0] == "warden_ft / full"
    md = rec.to_markdown(s)
    assert "T1046" in md and "0.5500" in md


def test_branch_b_and_deepop_score_on_a_store():
    from data_unification.trajectory_store import TrajectoryStoreBuilder
    from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
    from branch_b_world_model.infiltration_head import InfiltrationRiskHead
    from cyberworld_v4.cross_dataset import branch_b_on_store, deepop_on_store
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    from deepop_decoder.joint_vocab import get_joint_vocab

    rng = np.random.default_rng(0)
    b = TrajectoryStoreBuilder(spill_dir=None)
    for w in range(30):
        atk = 10 <= w < 20
        b.append(host_ip="10.0.0.1", host_id=1, window_idx=w, window_start=2.0 * w,
                 window_end=2.0 * w + 2, embedding=rng.random(12).astype(np.float32),
                 temporal_attrs=rng.random(15).astype(np.float32), is_attack=atk,
                 coarse_category="Impact" if atk else "Benign",
                 technique_ids=["T1498"] if atk else [], risk_score=0.8 if atk else 0.0)
    store = b.finalize()
    torch.manual_seed(0)
    wdt = HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3)
    risk = InfiltrationRiskHead(d_latent=27, hidden_dim=32)
    bb = branch_b_on_store(wdt, risk, store, "cpu", T=15, K=5)
    assert bb["n"] > 0 and bb["mse_persistence"] > 0
    vocab = get_joint_vocab(network_observable_only=True)
    dec = DeepOPForecastDecoder(d_latent=27, vocab_size=vocab.vocab_size)
    counts = np.zeros(vocab.vocab_size, dtype=np.int64)
    counts[vocab.encode("Benign", None)] = 10          # training saw only Benign
    dp = deepop_on_store(dec, wdt, store, vocab, "cpu", counts, T=15, K=5)
    assert dp["n"] > 0
    assert any("Impact" in u["class"] for u in dp["unseen_classes"])
