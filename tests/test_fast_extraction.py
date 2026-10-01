"""Extraction on the encoder's fast path (bita/fast level 1) == the reference, bit for bit.

data_unification/fast_extract.py switches the extraction encoder to
FastTGNMixin's streaming calls. Speed only: the whole trajectory store (feature
block, every column, interned tables, per-host rows) and the neighbour-exposure
counters must be identical, over several captures chained through ONE
extractor -- the encoder's memory and message pool reset between captures --
the way the trainers' extraction workers run.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_capture_columns import (  # noqa: E402
    REAL_CTU, WS, _columns, _encoder, assert_stores_equal, captures, corpus)  # noqa: F401
from data_unification import capture_columns as cc  # noqa: E402
from data_unification.fast_extract import enable_fast_extraction  # noqa: E402
from data_unification.multi_dataset_stream import HostTrajectoryExtractor  # noqa: E402
from data_unification.trajectory_store import TrajectoryStoreBuilder, capture_namespace  # noqa: E402
from data_unification.training_sources import Capture  # noqa: E402


def _extract(tgn, cols_list):
    ex = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=WS)
    b = TrajectoryStoreBuilder(spill_dir=None)
    base = 0
    for ns, cols in cols_list:
        b.set_namespace(ns)
        ex.extract_trajectories_columns(cols, builder=b, window_idx_base=base)
        base = b.next_window_base()
    return b.finalize(), ex.neighbor_exposure_report(reset=True)


@pytest.mark.parametrize("n_neighbors", [10, 3])
def test_fast_extraction_is_bit_identical(captures, corpus, tmp_path, n_neighbors):
    caps = list(captures["train"]) + list(captures["val"]) + list(captures["test"])
    if REAL_CTU.exists():
        caps.append(Capture("CTU13", f"{REAL_CTU.parent.name}/{REAL_CTU.name}", REAL_CTU, "train"))
    cols_list = []
    for i, cap in enumerate(caps):
        spec = cc.ColumnSpec.for_read_capture(cap, window_seconds=WS, pcap_label_dir=corpus["csv"])
        cols_list.append((capture_namespace(cap), _columns(spec, tmp_path, f"c{i}")))
    ref_tgn = _encoder(use_memory=True, n_neighbors=n_neighbors)
    fast_tgn = copy.deepcopy(ref_tgn)
    enable_fast_extraction(fast_tgn, log=None)
    assert type(fast_tgn).__name__.startswith("Fast"), "fast path did not engage"
    ref, ref_exp = _extract(ref_tgn, cols_list)
    fast, fast_exp = _extract(fast_tgn, cols_list)
    assert ref.n_snapshots > 1000
    assert_stores_equal(ref, fast)
    assert ref_exp == fast_exp


def test_disable_switch(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_FAST_EXTRACT", "0")
    tgn = _encoder(use_memory=True)
    enable_fast_extraction(tgn, log=None)
    assert not type(tgn).__name__.startswith("Fast")
