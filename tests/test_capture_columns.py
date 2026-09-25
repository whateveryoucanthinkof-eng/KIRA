"""The columnar capture path produces the SAME trajectory store as the record path.

data_unification/capture_columns.py parses a capture into compact columns and
HostTrajectoryExtractor.extract_trajectories_columns extracts from them. That
exists for memory and speed only (a CIC-2018 PCAP day is ~12 GB as records),
so every value it produces must equal the record path's, bit for bit: the
feature block, every metadata column, the interned tables, the per-host row
grouping, and the neighbour-exposure counters.

Covered: the synthetic plan corpus (PCAP days, CIC-2017, CTU-13, all in their
real on-disk formats), a real CTU-13 scenario when present, TGN memory on and
off, all three attack roles, several captures chained into one builder the
way the trainers do it, the downstream trainer's two extra readers, the edge
ablation mask, the cache, and the helpers the equivalence rests on.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from data_unification import capture_columns as cc  # noqa: E402
from data_unification.multi_dataset_stream import (  # noqa: E402
    HostTrajectoryExtractor, _sequential_group_sums, _window_boundaries)
from data_unification.training_sources import Capture, discover_captures, read_capture  # noqa: E402
from data_unification.trajectory_store import (  # noqa: E402
    TrajectoryStoreBuilder, capture_namespace)

REAL_CTU = Path("/var/home/samito/Documents/SIH/CTU-13-Dataset/7/capture20110816-2.binetflow")
WS = 2.0


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from dry_run_plan import build_corpus
    return build_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def captures(corpus):
    caps = discover_captures(scheme="cross_year_ctu", cic2017_dir=corpus["cic2017"],
                             ctu13_dir=corpus["ctu13"], pcap2018_root=corpus["pcap"])
    return caps


def _encoder(use_memory: bool, n_neighbors: int = 10, seed: int = 0):
    from model.extentedtgn import ExtendedTGN
    from utils.utils import NeighborFinder
    torch.manual_seed(seed)
    n0 = 64
    tgn = ExtendedTGN(
        neighbor_finder=NeighborFinder([[] for _ in range(n0)], uniform=False),
        node_features=np.zeros((n0, 12), np.float32), edge_features=np.zeros((10, 12), np.float32),
        device="cpu", n_layers=1, n_heads=2, dropout=0.0, use_memory=use_memory,
        message_dimension=100, memory_dimension=12, message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru", num_categories=5).eval()
    # Random, not zero, weights everywhere they matter: an all-zero projection
    # would make every embedding equal and hide an ordering mistake.
    with torch.no_grad():
        for p in tgn.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.3)
    tgn.serving_n_neighbors = n_neighbors
    return tgn


def _extractor(use_memory, **kw):
    enc_kw = {k: kw.pop(k) for k in ("n_neighbors",) if k in kw}
    return HostTrajectoryExtractor(tgne_ta_model=_encoder(use_memory, **enc_kw),
                                   window_size_sec=WS, **kw)


def _columns(spec, tmp_path, name="c"):
    d = tmp_path / f"{name}-{abs(hash(spec)) % 10**8}"
    cc.build_columns(spec, d)
    return cc.CaptureColumns.load(d, drop_unresolved=spec.drops_unresolved)


def assert_stores_equal(a, b):
    assert a.n_snapshots == b.n_snapshots
    fa, fb = np.asarray(a.feats), np.asarray(b.feats)
    assert fa.dtype == fb.dtype and fa.shape == fb.shape
    assert fa.tobytes() == fb.tobytes(), "feature block differs"
    for col in ("host_name_id", "node_id", "window_idx", "window_start", "window_end",
                "is_attack", "risk_score", "cat_id", "tech_off", "tech_flat"):
        x, y = np.asarray(getattr(a, col)), np.asarray(getattr(b, col))
        assert x.dtype == y.dtype, col
        assert x.tobytes() == y.tobytes(), f"column {col} differs"
    assert a.categories == b.categories
    assert a.techniques == b.techniques
    assert a.host_names == b.host_names
    assert list(a._rows_by_host) == list(b._rows_by_host)
    for k in a._rows_by_host:
        assert np.array_equal(a._rows_by_host[k], b._rows_by_host[k])


def _both(caps, tmp_path, *, use_memory, reader_kw=None, spill=None, **ex_kw):
    """(record-path store, columnar store, record exposure, columnar exposure)
    for `caps` chained into one builder, as Branch A's _store_per_capture does."""
    reader_kw = dict(reader_kw or {})
    out = []
    for mode in ("records", "columns"):
        ex = _extractor(use_memory, spill_dir=spill, **dict(ex_kw))
        b = TrajectoryStoreBuilder(spill_dir=spill)
        base = 0
        covs = []
        for cap in caps:
            b.set_namespace(capture_namespace(cap))
            if mode == "records":
                cov = []
                recs = read_capture(cap, window_seconds=WS, coverage=cov, **reader_kw)
                covs.append(cov[0])
                ex.extract_trajectories(recs, builder=b, window_idx_base=base)
            else:
                spec = cc.ColumnSpec.for_read_capture(cap, window_seconds=WS, **reader_kw)
                cols = _columns(spec, tmp_path)
                covs.append(cols.coverage)
                ex.extract_trajectories_columns(cols, builder=b, window_idx_base=base)
            if b._window_idx.n:
                base = int(b._window_idx.buf[: b._window_idx.n].max()) + 1
        out.append((b.finalize(), ex.neighbor_exposure_report(reset=True), covs))
    return out


def _check(pair):
    (sa, ea, ca), (sb, eb, cb) = pair
    assert sa.n_snapshots > 0
    assert_stores_equal(sa, sb)
    assert {k: v for k, v in ea.items() if not (isinstance(v, float) and math.isnan(v))} == \
           {k: v for k, v in eb.items() if not (isinstance(v, float) and math.isnan(v))}
    assert ca == cb
    return sa


# ------------------------------------------------------- whole-path equality
@pytest.mark.parametrize("use_memory", [True, False])
def test_every_synthetic_capture_matches(captures, corpus, tmp_path, use_memory):
    caps = [c for sp in ("train", "val", "test") for c in captures[sp]]
    assert {c.dataset for c in caps} == {"PCAP2018", "CTU13", "CIC2017"}
    st = _check(_both(caps, tmp_path, use_memory=use_memory,
                      reader_kw={"pcap_label_dir": corpus["csv"]}))
    # Not vacuous: attack windows (labels, risk, techniques) are in there.
    assert st.is_attack.any() and (~st.is_attack).any() and len(st.techniques) > 0


@pytest.mark.parametrize("role", ["either", "target", "source"])
def test_attack_roles_match(captures, corpus, tmp_path, role):
    caps = [c for c in captures["train"] if c.dataset == "PCAP2018"][:3]
    st = _check(_both(caps, tmp_path, use_memory=True, attack_role=role,
                      reader_kw={"pcap_label_dir": corpus["csv"]}))
    assert st.is_attack.any()


@pytest.mark.parametrize("k", [1, 3])
def test_small_neighbour_counts_match(captures, corpus, tmp_path, k):
    """Few neighbours: most host-windows are truncated, so the exposure counts
    and the latent's view of attack flows are exercised."""
    caps = [c for c in captures["train"] if c.dataset == "PCAP2018"][:2]
    _check(_both(caps, tmp_path, use_memory=True, n_neighbors=k,
                 reader_kw={"pcap_label_dir": corpus["csv"]}))


def test_spilled_store_matches(captures, corpus, tmp_path, monkeypatch):
    """With a spill dir and a tiny block, rows cross many flush boundaries."""
    import data_unification.trajectory_store as ts
    monkeypatch.setattr(ts, "_BLOCK", 7)
    caps = [c for c in captures["train"] if c.dataset == "PCAP2018"][:2]
    spill = tmp_path / "spill"
    _check(_both(caps, tmp_path, use_memory=True, spill=str(spill),
                 reader_kw={"pcap_label_dir": corpus["csv"]}))


def test_edge_ablation_is_applied_the_same(captures, corpus, tmp_path, monkeypatch):
    import data_unification.tgne_features as tf
    monkeypatch.setenv("CYBERWORLD_ABLATE_EDGE_FEATURES", "dst_port_log_norm,is_tcp")
    tf.reset_ablation_cache()
    try:
        caps = [c for c in captures["train"] if c.dataset == "PCAP2018"][:2]
        _check(_both(caps, tmp_path, use_memory=True,
                     reader_kw={"pcap_label_dir": corpus["csv"]}))
    finally:
        monkeypatch.delenv("CYBERWORLD_ABLATE_EDGE_FEATURES")
        tf.reset_ablation_cache()


@pytest.mark.skipif(not REAL_CTU.exists(), reason="real CTU-13 scenario 7 not on this machine")
def test_real_ctu13_scenario_matches(tmp_path):
    cap = Capture("CTU13", f"{REAL_CTU.parent.name}/{REAL_CTU.name}", REAL_CTU, "train")
    st = _check(_both([cap], tmp_path, use_memory=True))
    assert st.is_attack.any() and st.n_snapshots > 10_000


# --------------------------------------------------- downstream's readers
def test_downstream_readers_match(captures, corpus, tmp_path):
    """retrain_future_models_live's PCAP reader (no label filter) and its
    read_one_capture for CTU-13 give the same records as the columns."""
    import retrain_future_models_live as rf
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter

    pcap = [c for c in captures["train"] if c.dataset == "PCAP2018"][:2]
    ctu = [c for c in captures["train"] if c.dataset == "CTU13"][:2]
    for mode in ("records", "columns"):
        ex = _extractor(True)
        b = TrajectoryStoreBuilder()
        base = 0
        for cap in pcap + ctu:
            b.set_namespace(capture_namespace(cap))
            if mode == "records":
                if cap.dataset == "PCAP2018":
                    recs = [r for _s, day, rr in rf.iter_pcap_day_records(
                                corpus["pcap"], corpus["csv"], WS, scheme="cross_year_ctu")
                            if day == cap.name for r in rr]
                else:
                    recs = rf.read_one_capture(Path(cap.path), CIC2018Adapter(), CTU13Adapter(), None, 1)
                ex.extract_trajectories(recs, builder=b, window_idx_base=base)
            else:
                spec = (cc.ColumnSpec("pcap_windows", cap.dataset, cap.name, str(cap.path), WS,
                                      str(corpus["csv"]))
                        if cap.dataset == "PCAP2018" else
                        cc.ColumnSpec("one_capture", cap.dataset, cap.name, str(cap.path), WS))
                ex.extract_trajectories_columns(_columns(spec, tmp_path), builder=b,
                                                window_idx_base=base)
            if b._window_idx.n:
                base = int(b._window_idx.buf[: b._window_idx.n].max()) + 1
        if mode == "records":
            ref = b.finalize()
        else:
            assert_stores_equal(ref, b.finalize())


def test_downstream_jobs_follow_the_record_path_order(corpus):
    import argparse
    import retrain_future_models_live as rf
    args = argparse.Namespace(split_scheme="cross_year_ctu", pcap_root=corpus["pcap"],
                              cic2018_csv_dir=corpus["csv"], ctu_dir=corpus["ctu13"],
                              pcap_max_windows_per_day=None, pcap_window_stride=1,
                              rows_per_file=None, stride=1)
    jobs = rf._downstream_column_jobs(args)
    days = [name for _sp, name, _r in rf.iter_pcap_day_records(
        corpus["pcap"], corpus["csv"], WS, scheme="cross_year_ctu")]
    assert [ns for _s, ns, sp in jobs if sp.dataset == "PCAP2018"] == [f"pcap/{d}" for d in days]
    ctu = discover_captures(scheme="cross_year_ctu", ctu13_dir=corpus["ctu13"])
    assert [ns for _s, ns, sp in jobs if sp.dataset == "CTU13"] == \
        [capture_namespace(c) for sp in ("train", "val") for c in ctu[sp]]


# ------------------------------------------------------------------ the cache
def test_cache_hits_and_invalidates(captures, corpus, tmp_path, monkeypatch):
    cap = [c for c in captures["train"] if c.dataset == "CTU13"][0]
    spec = cc.ColumnSpec.for_read_capture(cap, window_seconds=WS)
    cache = tmp_path / "cache"
    (_s, first), = list(cc.iter_capture_columns([spec], cache_dir=cache, workers=0))
    entries = [p for p in cache.iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1
    built = []
    real = cc.build_columns
    monkeypatch.setattr(cc, "build_columns", lambda *a, **k: built.append(1) or real(*a, **k))
    (_s, again), = list(cc.iter_capture_columns([spec], cache_dir=cache, workers=0))
    assert not built, "a cached capture was parsed again"
    assert np.array_equal(first.start, again.start)
    # Touching the input file is a new key: parsed again, into a new entry.
    st = os.stat(cap.path)
    os.utime(cap.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    list(cc.iter_capture_columns([spec], cache_dir=cache, workers=0))
    assert built
    assert len([p for p in cache.iterdir() if not p.name.startswith(".")]) == 2


def test_parallel_workers_give_the_same_columns(captures, corpus, tmp_path):
    caps = [c for c in captures["train"] if c.dataset == "PCAP2018"][:2] + \
           [c for c in captures["train"] if c.dataset == "CTU13"][:2]
    specs = [cc.ColumnSpec.for_read_capture(c, window_seconds=WS, pcap_label_dir=corpus["csv"])
             for c in caps]
    got_serial = [{k: np.array(v) for k, v in c.cols.items()}
                  for _s, c in cc.iter_capture_columns(specs, scratch_dir=tmp_path / "a", workers=0)]
    par = list(cc.iter_capture_columns(specs, cache_dir=tmp_path / "b", workers=2))
    assert [s for s, _c in par] == specs
    for want, (_s, c) in zip(got_serial, par):
        for k, v in want.items():
            assert np.array_equal(v, np.asarray(c.cols[k])), k


def test_columns_plan():
    ex = _extractor(False)
    assert not cc.columns_plan("off", "/x", ex).enabled
    p = cc.columns_plan(None, "/x", ex)
    assert p.enabled and p.cache_dir == Path("/x/capture_cache")
    assert cc.columns_plan("/c", None, ex).cache_dir == Path("/c")
    assert not cc.columns_plan(None, None, ex).enabled
    wide = HostTrajectoryExtractor(tgne_ta_model=_encoder(False), window_size_sec=WS,
                                   include_packet_features=True)
    assert not cc.columns_plan(None, "/x", wide).enabled


# ------------------------------------------------------------------ helpers
def test_sequential_group_sums_are_the_loop():
    rng = np.random.default_rng(3)
    counts = np.r_[rng.integers(1, 9, 500), [70, 200, 1]]
    x = rng.random(int(counts.sum())) * 10.0 ** rng.integers(-6, 6, int(counts.sum()))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    got = _sequential_group_sums(x, counts, starts)
    for g in range(counts.size):
        t = 0.0
        for v in x[starts[g]:starts[g] + counts[g]]:
            t += float(v)
        assert got[g] == t


def test_window_boundaries_are_process_records():
    from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter
    from data_unification.unified_schema import UnifiedFlowRecord
    rng = np.random.default_rng(1)
    t = np.sort(1.5e9 + np.cumsum(rng.exponential(0.7, 3000) * (rng.random(3000) < 0.9) +
                                  rng.random(3000) * (rng.random(3000) < 0.01) * 400))
    recs = [UnifiedFlowRecord(src_ip="a", dst_ip="b", src_port=1, dst_port=2, protocol=6,
                              start_time=float(x), end_time=float(x), fwd_bytes=1, bwd_bytes=1,
                              fwd_packets=1, bwd_packets=1, raw_label="", raw_label_source="",
                              is_attack=False, coarse_category="Benign") for x in t]
    want = FlowToTemporalEventAdapter(window_size_sec=WS).process_records(recs).window_boundaries
    got = _window_boundaries(t, WS)
    assert len(want) == len(got)
    for w, g in zip(want, got):
        assert w == g and [type(v) for v in w] == [type(v) for v in g]


def test_append_batch_is_append(monkeypatch, tmp_path):
    import data_unification.trajectory_store as ts
    monkeypatch.setattr(ts, "_BLOCK", 5)
    rng = np.random.default_rng(0)
    for spill in (None, str(tmp_path / "sp")):
        a, b = TrajectoryStoreBuilder(spill_dir=spill), TrajectoryStoreBuilder(spill_dir=spill)
        for bb in (a, b):
            bb.set_namespace("ns")
        for w in range(6):
            m = int(rng.integers(1, 9))
            rows = dict(
                host_ips=[f"10.0.0.{int(i)}" for i in rng.integers(0, 12, m)],
                host_ids=rng.integers(0, 50, m), window_idx=w, window_start=np.float64(w * 2.0),
                window_end=np.float64(w * 2.0 + 2), embeddings=rng.random((m, 12), dtype=np.float32),
                temporal_attrs=rng.random((m, 15), dtype=np.float32),
                is_attack=rng.random(m) < 0.3,
                coarse_categories=[["Benign", "Recon", "Impact"][int(i)] for i in rng.integers(0, 3, m)],
                technique_ids=[[["T1"], [], ["T2", "T1"]][int(i)] for i in rng.integers(0, 3, m)],
                risk_scores=rng.random(m))
            a.append_batch(**rows)
            for i in range(m):
                b.append(host_ip=rows["host_ips"][i], host_id=int(rows["host_ids"][i]),
                         window_idx=w, window_start=rows["window_start"],
                         window_end=rows["window_end"], embedding=rows["embeddings"][i],
                         temporal_attrs=rows["temporal_attrs"][i],
                         is_attack=bool(rows["is_attack"][i]),
                         coarse_category=rows["coarse_categories"][i],
                         technique_ids=rows["technique_ids"][i],
                         risk_score=float(rows["risk_scores"][i]))
        assert_stores_equal(a.finalize(), b.finalize())
