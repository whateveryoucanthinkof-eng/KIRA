"""After scripts/warm_ingest_cache.py, the encoder's own load must be a pure cache hit.

If the warm step computed keys even slightly differently (capture list, split
filter, parameters), every encoder would silently re-parse everything.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import bita.train as bt

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location("drp", REPO / "scripts" / "dry_run_plan.py")
    drp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drp)
    return drp.build_corpus(tmp_path_factory.mktemp("synthetic"), seed=11)


def test_warm_then_encoder_load_is_all_cached(corpus, tmp_path, caplog, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import warm_ingest_cache as w
    from cyberworld_v4.config import get_contract
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["warm", "--cache", str(cache), "--workers", "3",
                                      "--ctu-dir", str(corpus["ctu13"]), "--pcap-root", str(corpus["pcap"]),
                                      "--cic2018-csv-dir", str(corpus["csv"])])
    assert w.main() == 0
    n_entries = len([p for p in cache.iterdir() if not p.name.startswith(".")])
    assert n_entries > 0
    with caplog.at_level(logging.INFO):
        warm = bt.load_and_preprocess_unified_dataset(
            ctu13_dir=str(corpus["ctu13"]), pcap2018_root=str(corpus["pcap"]),
            pcap2018_label_dir=str(corpus["csv"]), scheme="cross_year_ctu", splits=("train",),
            parallel_workers=3, window_seconds=get_contract().window_seconds,
            ingest_cache=str(cache))
    assert any(f"{n_entries} of {n_entries} capture(s) cached" in r.getMessage() for r in caplog.records)
    serial = bt.load_and_preprocess_unified_dataset(
        ctu13_dir=str(corpus["ctu13"]), pcap2018_root=str(corpus["pcap"]),
        pcap2018_label_dir=str(corpus["csv"]), scheme="cross_year_ctu", splits=("train",),
        parallel_workers=0, window_seconds=get_contract().window_seconds)
    for col in ("u", "i", "ts", "label", "idx", "source", "capture"):
        np.testing.assert_array_equal(warm[0][col].values, serial[0][col].values)
    np.testing.assert_array_equal(warm[1], serial[1])
