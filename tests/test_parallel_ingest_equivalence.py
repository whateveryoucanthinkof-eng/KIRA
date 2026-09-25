"""Parallel capture parsing must be bit-identical to serial.

Loading 34M records took 12.7 minutes on ONE core while fifteen sat idle.
Captures are independent files, so parsing is embarrassingly parallel -- but
naive parallelism would silently change the model.

Node ids are assigned in ENCOUNTER order. If workers finish out of order the
same host gets a different id, which changes the node-feature row it indexes,
the negative sampler's draws, and which nodes are chosen as inductive. The
graph stays isomorphic but the run stops being reproducible and can no longer
be compared against a serial baseline.

Workers therefore never assign global ids: each returns a LOCAL vocabulary and
the parent merges in fixed capture order. These tests pin that guarantee.
"""

import numpy as np
import pandas as pd
import pytest

import bita.train as bt


def _capture(path, n_rows, host_base, seed):
    """A CIC-2018-shaped CSV with real IP columns."""
    rng = np.random.default_rng(seed)
    hrs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
    df = pd.DataFrame({
        "Src IP": [f"10.{host_base}.0.{i % 50}" for i in range(n_rows)],
        "Dst IP": [f"10.{host_base}.1.{i % 30}" for i in range(n_rows)],
        "Src Port": rng.integers(1024, 65535, n_rows),
        "Dst Port": rng.choice([80, 443, 22, 53], n_rows),
        "Protocol": rng.choice([6, 17], n_rows),
        "Timestamp": [f"14/02/2018 {hrs[i % 10]:02d}:{i % 60:02d}:{i % 60:02d}"
                      for i in range(n_rows)],
        "Flow Duration": rng.integers(1, 10**6, n_rows),
        "Tot Fwd Pkts": rng.integers(1, 100, n_rows),
        "Tot Bwd Pkts": rng.integers(1, 100, n_rows),
        "TotLen Fwd Pkts": rng.integers(1, 10**5, n_rows),
        "TotLen Bwd Pkts": rng.integers(1, 10**5, n_rows),
        "Label": rng.choice(["Benign", "DDOS attack-HOIC", "Infilteration"], n_rows),
    })
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Several captures named so the frozen lock assigns them to train."""
    d = tmp_path_factory.mktemp("cic2018")
    for i, name in enumerate(["thu_22_csv.csv", "fri_23_csv.csv", "thu_15_csv.csv",
                              "wed_14_csv.csv", "fri_16_csv.csv"]):
        _capture(d / name, 1200, host_base=i, seed=i)
    return str(d)


def _load(corpus, workers):
    return bt.load_and_preprocess_unified_dataset(
        cic2018_dir=corpus, splits=("train",), parallel_workers=workers)


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_output_is_bit_identical_to_serial(corpus, workers):
    gs, es, ns, cs = _load(corpus, 0)
    gp, ep, np_, cp = _load(corpus, workers)

    for col in ("u", "i", "ts", "label", "idx", "source"):
        np.testing.assert_array_equal(
            gs[col].values, gp[col].values, err_msg=f"graph_df.{col} differs")
    np.testing.assert_array_equal(es, ep, err_msg="edge_features differ")
    np.testing.assert_array_equal(ns, np_, err_msg="node_features differ")
    assert cs == cp, "category mapping differs"


def test_node_ids_follow_encounter_order_not_completion_order(corpus):
    """The specific hazard: a worker finishing early must not claim low ids."""
    gs, _es, _ns, _cs = _load(corpus, 0)
    gp, _ep, _np, _cp = _load(corpus, 4)
    assert gs.u.max() == gp.u.max()
    assert set(gs.u.values) == set(gp.u.values)
    np.testing.assert_array_equal(gs.u.values, gp.u.values)


def test_record_count_is_preserved(corpus):
    gs, _, _, _ = _load(corpus, 0)
    gp, _, _, _ = _load(corpus, 4)
    assert len(gs) == len(gp) > 0


def test_one_capture_falls_back_to_serial(tmp_path):
    """len(captures) <= 1 has nothing to parallelise; it must still work."""
    _capture(tmp_path / "thu_22_csv.csv", 800, host_base=9, seed=9)
    g, e, n, c = _load(str(tmp_path), 4)
    assert len(g) > 0 and e.shape[0] == len(g) + 1


def test_a_worker_failure_is_reported_not_swallowed(corpus, caplog):
    """A capture that cannot be parsed must name itself in the log."""
    import logging
    from data_unification.parallel_ingest import parse_captures_parallel
    with caplog.at_level(logging.ERROR, logger="data_unification.parallel_ingest"):
        out = list(parse_captures_parallel(
            [("CIC2018", "/nonexistent/thu_22_csv.csv"),
             ("CIC2018", "/nonexistent/fri_23_csv.csv")], workers=2))
    assert out == []
    assert any("parallel parse failed" in r.message for r in caplog.records)


# ------------------------------------------------------------------ PCAP days
#
# The encoder's real corpus is CTU-13 plus CIC-2018 PCAP days. Serially that
# read ~480 GB on one core (~6.5 h). PCAP days now go through the parallel
# parser one day per worker, and must stay bit-identical to serial.

@pytest.fixture(scope="module")
def pcap_corpus(tmp_path_factory):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "dry_run_plan", Path(__file__).resolve().parents[1] / "scripts" / "dry_run_plan.py")
    drp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drp)
    return drp.build_corpus(tmp_path_factory.mktemp("synthetic"), seed=3)


def _load_pcap(paths, workers, scratch):
    return bt.load_and_preprocess_unified_dataset(
        ctu13_dir=str(paths["ctu13"]), pcap2018_root=str(paths["pcap"]),
        pcap2018_label_dir=str(paths["csv"]), scheme="cross_year_ctu",
        splits=("train",), parallel_workers=workers, window_seconds=2.0,
        ingest_scratch=scratch)


@pytest.mark.parametrize("workers", [3, 8])
def test_parallel_pcap_ingest_is_bit_identical_to_serial(pcap_corpus, tmp_path, workers):
    gs, es, ns, cs = _load_pcap(pcap_corpus, 0, None)
    gp, ep, np_, cp = _load_pcap(pcap_corpus, workers, str(tmp_path / "ingest"))

    assert (gs.source.nunique() == 2), "fixture must mix CTU-13 and PCAP records"
    for col in ("u", "i", "ts", "label", "idx", "source", "capture"):
        np.testing.assert_array_equal(
            gs[col].values, gp[col].values, err_msg=f"graph_df.{col} differs")
    np.testing.assert_array_equal(es, ep, err_msg="edge_features differ")
    np.testing.assert_array_equal(ns, np_, err_msg="node_features differ")
    assert cs == cp, "category mapping differs"
    assert not (tmp_path / "ingest").exists(), "scratch parts must be cleaned up"


def test_parallel_pcap_ingest_refuses_tmpfs_default(pcap_corpus):
    """Without an on-disk scratch the parts would land in /tmp, i.e. RAM."""
    with pytest.raises(ValueError, match="ingest_scratch"):
        _load_pcap(pcap_corpus, 4, None)


def test_multi_part_captures_merge_identically(pcap_corpus, tmp_path, monkeypatch):
    """A capture larger than one on-disk part must reassemble in order."""
    import data_unification.parallel_ingest as pi
    gs, es, _ns, _cs = _load_pcap(pcap_corpus, 0, None)
    monkeypatch.setattr(pi, "_PART_RECORDS", 7)
    gp, ep, _np, _cp = _load_pcap(pcap_corpus, 4, str(tmp_path / "ingest"))
    np.testing.assert_array_equal(gs.u.values, gp.u.values)
    np.testing.assert_array_equal(gs.ts.values, gp.ts.values)
    np.testing.assert_array_equal(es, ep)
