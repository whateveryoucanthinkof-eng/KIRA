"""The Rust PCAP parser (rust/pcap_fast) is the default for PCAP2018 days.

It must be invisible: bit-identical to the Python reference through the whole
loader, part splitting included. Every Rust-parsed day is also re-checked on a
two-file sample against the reference, and a disagreement must stop the run.
"""

import importlib.util
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import bita.train as bt
import data_unification.parallel_ingest as pi

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rust_bin():
    """The release binary, built if cargo is around; skip otherwise."""
    b = pi.pcap_fast_binary()
    if shutil.which("cargo"):
        subprocess.run(["cargo", "build", "--release", "-q", "-j4"],
                       cwd=REPO / "rust" / "pcap_fast", check=True)
    if not b.is_file():
        pytest.skip("rust/pcap_fast binary not built and cargo not available")
    if pi.pcap_parser_choice() != "rust":
        pytest.skip("Rust parser not selected in this environment")
    return b


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "dry_run_plan", REPO / "scripts" / "dry_run_plan.py")
    drp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drp)
    return drp.build_corpus(tmp_path_factory.mktemp("synthetic"), seed=11)


def _load(paths, workers, scratch=None, cache=None):
    return bt.load_and_preprocess_unified_dataset(
        ctu13_dir=str(paths["ctu13"]), pcap2018_root=str(paths["pcap"]),
        pcap2018_label_dir=str(paths["csv"]), scheme="cross_year_ctu",
        splits=("train",), parallel_workers=workers, window_seconds=2.0,
        ingest_scratch=scratch, ingest_cache=cache)


def _same(a, b):
    ga, ea, na, ca = a
    gb, eb, nb, cb = b
    for col in ("u", "i", "ts", "label", "idx", "source", "capture"):
        np.testing.assert_array_equal(ga[col].values, gb[col].values, err_msg=col)
    assert ea.dtype == eb.dtype
    np.testing.assert_array_equal(ea.view(np.uint32), eb.view(np.uint32))
    np.testing.assert_array_equal(na, nb)
    assert ca == cb


def _days(paths):
    return sorted(p for p in Path(paths["pcap"]).iterdir() if p.is_dir())


def _args(day, out, csv, parser, part_records=500_000):
    return ("PCAP2018", str(day), str(out), 1, None, 12, str(csv), 2.0, part_records, parser)


@pytest.mark.parametrize("part_records", [500_000, 7])
def test_rust_parse_one_is_identical_to_python(corpus, rust_bin, tmp_path, part_records):
    """Same result dict, same parts (boundaries included), same coverage."""
    for day in _days(corpus):
        res = {}
        for parser in ("python", "rust"):
            out = tmp_path / parser / day.name
            out.mkdir(parents=True)
            res[parser] = pi._parse_one(_args(day, out, corpus["csv"], parser, part_records))
            assert not res[parser].get("error"), res[parser]
        py, rs = res["python"], res["rust"]
        assert py["n"] > 0
        for k in ("n", "source", "ips", "cats", "coverage", "kind"):
            assert py[k] == rs[k], (day.name, k)
        assert len(py["parts"]) == len(rs["parts"])
        for sp, sr in zip(py["parts"], rs["parts"]):
            for c in pi._COLS:
                a, b = np.load(f"{sp}.{c}.npy"), np.load(f"{sr}.{c}.npy")
                assert a.dtype == b.dtype and a.shape == b.shape, (day.name, c)
                assert np.array_equal(pi._bits(a), pi._bits(b)), (day.name, c)


@pytest.mark.parametrize("part_records", [None, 7])
def test_loader_on_rust_is_bit_identical_to_serial_python(corpus, rust_bin, tmp_path,
                                                          monkeypatch, caplog, part_records):
    serial = _load(corpus, 0)          # serial path: always the Python reference
    if part_records:
        monkeypatch.setattr(pi, "_PART_RECORDS", part_records)
    with caplog.at_level(logging.INFO, logger="data_unification.parallel_ingest"):
        par = _load(corpus, 3, scratch=str(tmp_path / "ingest"))
    assert any("PCAP parser: rust" in r.getMessage() for r in caplog.records)
    _same(serial, par)


def test_python_parser_can_be_forced(corpus, monkeypatch, caplog):
    monkeypatch.setenv(pi.PCAP_PARSER_ENV, "python")
    assert pi.pcap_parser_choice() == "python"
    monkeypatch.setenv(pi.PCAP_PARSER_ENV, "fortran")
    with pytest.raises(ValueError):
        pi.pcap_parser_choice()


def test_missing_binary_falls_back_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv(pi.PCAP_PARSER_ENV, raising=False)
    monkeypatch.setenv("PCAP_FAST_BIN", str(tmp_path / "nope"))
    with caplog.at_level(logging.WARNING, logger="data_unification.parallel_ingest"):
        assert pi.pcap_parser_choice() == "python"
    assert any("not built" in r.getMessage() for r in caplog.records)


def test_numpy_version_guard(monkeypatch):
    pi.check_numpy_for_rust_parity()           # the pinned version is what we run
    monkeypatch.setattr(pi, "_PARITY_NUMPY_VERSION", "0.0.0")
    with pytest.raises(pi.PcapParityError, match="numpy"):
        pi.check_numpy_for_rust_parity()


def test_cache_key_separates_parsers(corpus):
    day = _days(corpus)[0]
    kw = dict(stride=1, max_rows=None, edge_dim=12, pcap_label_dir=str(corpus["csv"]),
              window_seconds=2.0, code_hash="x")
    assert (pi.cache_key("PCAP2018", day, pcap_parser="rust", **kw)
            != pi.cache_key("PCAP2018", day, pcap_parser="python", **kw))


def test_code_hash_covers_rust_sources():
    _h, files = pi.parse_code_hash()
    for f in ("rust/pcap_fast/src/main.rs", "rust/pcap_fast/Cargo.toml",
              "rust/pcap_fast/pcap_fast_py.py"):
        assert f in files


def _corrupt_reference(monkeypatch):
    """The parity reference returns a ts one ulp off on its first record."""
    real = pi._python_columns

    def bad(*a, **k):
        out = real(*a, **k)
        stem = out[4][0]
        ts = np.load(f"{stem}.ts.npy")
        ts[0] = np.nextafter(ts[0], np.inf)
        np.save(f"{stem}.ts.npy", ts)
        return out
    monkeypatch.setattr(pi, "_python_columns", bad)


def test_parity_mismatch_is_fatal_in_the_worker(corpus, rust_bin, tmp_path, monkeypatch):
    _corrupt_reference(monkeypatch)
    day = _days(corpus)[0]
    res = pi._parse_one(_args(day, tmp_path, corpus["csv"], "rust"))
    assert res.get("fatal") and "disagrees with the Python reference" in res["error"]
    assert "ts: 1 row(s) differ" in res["error"]
    assert not list(tmp_path.glob(".parity-*")), "parity sample must clean up"


def test_parity_mismatch_stops_the_run(corpus, rust_bin, tmp_path, monkeypatch):
    """Not logged-and-skipped like an unreadable capture: the loader raises."""
    _corrupt_reference(monkeypatch)
    # In-process workers so the monkeypatch reaches them.
    monkeypatch.setattr(pi, "ProcessPoolExecutor",
                        lambda max_workers, mp_context=None: ThreadPoolExecutor(max_workers))
    with pytest.raises(pi.PcapParityError, match="disagrees"):
        _load(corpus, 2, scratch=str(tmp_path / "ingest"))


def test_parity_sample_passes_on_real_parse(corpus, rust_bin, tmp_path):
    day = _days(corpus)[0]
    fast = pi._pcap_fast_module()
    dw, dayname = fast.load_day_labels(day, corpus["csv"])
    out = pi._pcap_parity_sample(fast, day, dw, dayname, 2.0, 12, tmp_path, 2)
    assert len(out["files"]) == 2 and out["records"] > 0


def test_verify_py_on_synthetic_days(corpus, rust_bin, tmp_path):
    """rust/pcap_fast/verify.py, both stages, as a standing check."""
    import sys
    sys.path.insert(0, str(REPO / "rust" / "pcap_fast"))
    spec = importlib.util.spec_from_file_location("pcap_fast_verify",
                                                  REPO / "rust" / "pcap_fast" / "verify.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    for day in _days(corpus):
        s1 = v.stage1(day)
        assert s1["windows"] > 0 and s1["mismatches"] == 0, (day.name, s1)
        s2 = v.stage2(day, corpus["csv"], tmp_path / "scratch")
        assert s2["n_ref"] == s2["n_fast"] > 0
        for c in ("u", "i", "ts", "lbl", "edge", "ips", "cats", "source"):
            assert s2[c] == "EXACT", (day.name, c, s2[c])
