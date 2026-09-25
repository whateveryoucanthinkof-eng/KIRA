"""The parsed-capture cache must be invisible: bit-identical to serial, and
never serve an entry whose inputs, parameters or parsing code have changed.

Every encoder run re-parsed ~482 GB of PCAPs; the cache makes that once.
A stale hit would silently train on the wrong data, so these tests pin the
invalidation as much as the equivalence.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import bita.train as bt
import data_unification.parallel_ingest as pi


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dry_run_plan", Path(__file__).resolve().parents[1] / "scripts" / "dry_run_plan.py")
    drp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drp)
    return drp.build_corpus(tmp_path_factory.mktemp("synthetic"), seed=5)


def _load(paths, workers, cache=None, scratch=None):
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
    np.testing.assert_array_equal(ea, eb)
    np.testing.assert_array_equal(na, nb)
    assert ca == cb


def _entries(cache):
    return sorted(p.name for p in Path(cache).iterdir() if not p.name.startswith("."))


def test_cold_and_warm_cache_are_bit_identical_to_serial(corpus, tmp_path, caplog):
    import logging
    serial = _load(corpus, 0)
    cache = tmp_path / "cache"
    _same(serial, _load(corpus, 4, cache=str(cache)))
    first = _entries(cache)
    assert first, "cold run must populate the cache"
    with caplog.at_level(logging.INFO):
        warm = _load(corpus, 4, cache=str(cache))
    _same(serial, warm)
    assert _entries(cache) == first, "warm run must not add entries"
    assert any(f"{len(first)} of {len(first)} capture(s) cached" in r.getMessage()
               for r in caplog.records), "warm run must be served entirely from cache"
    assert not any(p.name.startswith(".tmp-") for p in cache.iterdir())


def test_changed_input_file_misses(corpus, tmp_path):
    cache = tmp_path / "cache"
    _load(corpus, 4, cache=str(cache))
    before = set(_entries(cache))
    day = sorted(p for p in Path(corpus["pcap"]).iterdir() if p.name == "fri_16_pcap")[0]
    f = sorted(x for x in day.rglob("*") if x.is_file())[0]
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    try:
        _load(corpus, 4, cache=str(cache))
        new = set(_entries(cache)) - before
        assert len(new) == 1 and next(iter(new)).startswith("fri_16_pcap-"), new
    finally:
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_changed_label_csv_misses(corpus, tmp_path):
    from data_unification.training_sources import _pcap_label_csv
    cache = tmp_path / "cache"
    _load(corpus, 4, cache=str(cache))
    before = set(_entries(cache))
    csv = _pcap_label_csv(Path(corpus["pcap"]) / "wed_14_pcap", Path(corpus["csv"]))
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    try:
        _load(corpus, 4, cache=str(cache))
        new = set(_entries(cache)) - before
        assert len(new) == 1 and next(iter(new)).startswith("wed_14_pcap-"), new
    finally:
        os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_changed_parse_code_misses_everything(corpus, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _load(corpus, 4, cache=str(cache))
    before = set(_entries(cache))
    real = pi.parse_code_hash
    monkeypatch.setattr(pi, "parse_code_hash", lambda: ("0" * 64, real()[1]))
    _load(corpus, 4, cache=str(cache))
    assert len(set(_entries(cache)) - before) == len(before), "a code change must re-parse every capture"


def test_code_hash_covers_lazily_imported_parse_modules():
    h, files = pi.parse_code_hash()
    for mod in ("data_unification/pcap_bridge.py", "telemetry/capture/sniffer.py",
                "telemetry/flow/flow_table.py", "telemetry/packet/pcap_engine.py",
                "data_unification/tgne_features.py"):
        assert mod in files, f"{mod} not in the code hash"
    assert not any(f.startswith("bita/") for f in files), "trainer code must not invalidate the cache"


def test_uncommitted_entry_is_never_read(corpus, tmp_path):
    """A crash can leave an entry without meta.json, or with a part missing."""
    serial = _load(corpus, 0)
    cache = tmp_path / "cache"
    _load(corpus, 4, cache=str(cache))
    victim = Path(cache) / _entries(cache)[0]
    (victim / "meta.json").unlink()
    other = Path(cache) / _entries(cache)[1]
    meta = json.loads((other / "meta.json").read_text())
    (other / f"{meta['parts'][0]}.edge.npy").unlink()
    # Both damaged entries stay on disk: the loader must re-parse them, not
    # read them, and must repair them for the next run.
    _same(serial, _load(corpus, 4, cache=str(cache)))
    assert (victim / "meta.json").exists()
    repaired = json.loads((other / "meta.json").read_text())
    assert all((other / f"{st}.{c}.npy").exists()
               for st in repaired["parts"] for c in ("u", "i", "ts", "lbl", "edge"))
    _same(serial, _load(corpus, 4, cache=str(cache)))


def test_cached_parts_survive_loading(corpus, tmp_path):
    cache = tmp_path / "cache"
    _load(corpus, 4, cache=str(cache))
    n_files = sum(1 for _ in Path(cache).rglob("*.npy"))
    _load(corpus, 4, cache=str(cache))
    assert sum(1 for _ in Path(cache).rglob("*.npy")) == n_files


def test_code_hash_ignores_modules_with_relative_file(monkeypatch):
    """torch._classes has __file__ == "_classes.py"; resolved against the cwd
    (the repo) it was taken for a repo module and the hash crashed."""
    import sys
    import types
    fake = types.ModuleType("fake_relative")
    fake.__file__ = "_classes.py"
    monkeypatch.setitem(sys.modules, "fake_relative", fake)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    repo = str(Path(__file__).resolve().parents[1])
    h, files = pi._code_hash_worker(repo)
    assert "_classes.py" not in files and h
