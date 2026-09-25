"""The out-of-core encoder setup computes exactly what the in-RAM one did.

bita/train.py keeps every per-edge array in files (utils.DiskStore) so ~133M
edges fit a 17 GiB cap and an 8 GB GPU. None of that may change a value: the
trained model has to be bit-identical. Each piece is checked here against the
original computation, kept verbatim as the oracle.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bita"))

import utils.utils as U  # noqa: E402
from model.tgn import HostEdgeFeatures  # noqa: E402


class _D:
    pass


def _graph(rng, n, nodes, t_span):
    d = _D()
    d.sources = rng.integers(0, nodes, n)
    d.destinations = rng.integers(0, nodes, n)
    d.timestamps = np.sort(rng.integers(0, t_span, n)).astype(np.float64)   # heavy ties
    d.edge_idxs = np.arange(1, n + 1)
    return d


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(U, "CHUNK", 37)


def test_streamed_csr_equals_lexsort(tmp_path, small_chunks):
    rng = np.random.default_rng(0)
    for trial in range(25):
        d = _graph(rng, int(rng.integers(1, 3000)), int(rng.integers(2, 200)), int(rng.integers(1, 100)))
        ref = U.get_neighbor_finder(d, uniform=False)
        got = U.get_neighbor_finder(d, uniform=False, store=U.DiskStore(tmp_path / f"s{trial}"))
        for a, b in zip(ref._csr, got._csr):
            assert a.dtype == b.dtype and np.array_equal(a, b), trial


def test_unsorted_times_fall_back_to_lexsort(tmp_path):
    rng = np.random.default_rng(1)
    d = _graph(rng, 500, 30, 50)
    d.timestamps = rng.permutation(d.timestamps)
    ref = U.get_neighbor_finder(d, uniform=False)
    got = U.get_neighbor_finder(d, uniform=False, store=U.DiskStore(tmp_path / "s"))
    for a, b in zip(ref._csr, got._csr):
        assert np.array_equal(a, b)


def test_first_seen_order_reproduces_set_iteration(small_chunks):
    rng = np.random.default_rng(2)
    for _ in range(30):
        a = rng.integers(0, int(rng.integers(2, 50000)), int(rng.integers(1, 20000)))
        b = rng.integers(0, 50000, int(rng.integers(1, 20000)))
        ref = set(a).union(set(b))                       # np.int64 objects, as split_data had
        got = set(U.first_seen_order(a).tolist()).union(set(U.first_seen_order(b).tolist()))
        assert [int(x) for x in ref] == list(got)       # same ITERATION order
        np.random.seed(2020)
        r1 = np.random.choice(list(ref), min(len(ref), 100), replace=False)
        np.random.seed(2020)
        r2 = np.random.choice(list(got), min(len(got), 100), replace=False)
        assert np.array_equal(r1, r2)


def test_sorted_unique_and_count_unique():
    rng = np.random.default_rng(3)
    for dt in (np.int64, np.int32, np.uint16):
        a = rng.integers(0, 1000, 5000).astype(dt)
        b = rng.integers(0, 1000, 5000).astype(dt)
        u = U.sorted_unique(a)
        assert u.dtype == np.unique(a).dtype and np.array_equal(u, np.unique(a))
        assert U.count_unique(a, b) == len(set(a) | set(b))
    assert np.array_equal(U.sorted_unique([3, 1, 3]), np.unique([3, 1, 3]))
    neg = np.array([-5, 2, 2])
    assert np.array_equal(U.sorted_unique(neg), np.unique(neg))


def _time_stats_reference(sources, destinations, timestamps):
    last_s, last_d, ds, dd = {}, {}, [], []
    for k in range(len(sources)):
        s, d, t = sources[k], destinations[k], timestamps[k]
        last_s.setdefault(s, 0)
        last_d.setdefault(d, 0)
        ds.append(t - last_s[s])
        dd.append(t - last_d[d])
        last_s[s] = t
        last_d[d] = t
    return (float(np.mean(ds)), float(np.std(ds)) if np.std(ds) > 0 else 1.0,
            float(np.mean(dd)), float(np.std(dd)) if np.std(dd) > 0 else 1.0)


def test_time_statistics_bit_identical(tmp_path, monkeypatch):
    import train as T
    monkeypatch.setattr(T, "CHUNK", 53)
    rng = np.random.default_rng(4)
    for trial in range(10):
        n = int(rng.integers(1, 4000))
        s = rng.integers(0, 300, n)
        d = rng.integers(0, 300, n)
        ts = np.sort(rng.random(n) * 1e9 + 1.3e9)
        ref = _time_stats_reference(s, d, ts)
        store = U.DiskStore(tmp_path / f"t{trial}") if trial % 2 else None
        got = T.compute_time_statistics(s, d, ts, store=store)
        assert [x.hex() for x in ref] == [float(x).hex() for x in got]


def test_host_edge_features_match_device_indexing(tmp_path):
    rng = np.random.default_rng(5)
    arr = np.memmap(tmp_path / "e.bin", dtype=np.float32, mode="w+", shape=(1000, 12))
    arr[:] = rng.random((1000, 12), dtype=np.float32)
    dev = torch.from_numpy(np.array(arr))
    host = HostEdgeFeatures(arr, "cpu")
    assert host.shape == dev.shape
    for idx in (rng.integers(0, 1000, 64), rng.integers(0, 1000, (16, 10)).astype(np.int32),
                torch.from_numpy(rng.integers(0, 1000, 7)), np.array([-1, 0, 999])):
        assert torch.equal(host[idx], dev[idx])
        assert torch.equal(host[idx, :], dev[idx, :])
    with pytest.raises(IndexError):
        host[np.array([1000])]


def test_disk_growable_loader_matches_ram(tmp_path):
    """Columns grown in files, sorted in chunks: same graph_df and edge features."""
    import train as T
    ctu = tmp_path / "ctu13" / "7"
    ctu.mkdir(parents=True)
    rng = np.random.default_rng(6)
    lines = ["StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"]
    for k in range(3000):
        t = f"2011/08/16 10:{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}.{rng.integers(0, 999999):06d}"
        lines.append(f"{t},{rng.random():.6f},tcp,147.32.84.{rng.integers(1, 40)},{rng.integers(1024, 65000)},"
                     f"->,147.32.80.{rng.integers(1, 20)},80,SA_A,0,0,{rng.integers(1, 50)},"
                     f"{rng.integers(60, 9000)},{rng.integers(60, 4000)},flow=Background-TCP-Established")
    (ctu / "capture20110816-2.binetflow").write_text("\n".join(lines) + "\n")
    kw = dict(ctu13_dir=str(tmp_path / "ctu13"), scheme="cross_year_ctu")
    g0, e0, n0, c0 = T.load_and_preprocess_unified_dataset(**kw)
    g1, e1, n1, c1 = T.load_and_preprocess_unified_dataset(**kw, store=U.DiskStore(tmp_path / "st"))
    assert isinstance(e1, np.memmap) and c0 == c1
    assert np.array_equal(e0, e1) and np.array_equal(n0, n1)
    for col in g0.columns:
        assert g0[col].dtype == g1[col].dtype and np.array_equal(g0[col].values, g1[col].values), col
