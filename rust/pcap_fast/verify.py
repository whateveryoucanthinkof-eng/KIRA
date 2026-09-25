"""Bit-exactness check: Python reference vs pcap_fast, per stage, on one day dir.

    python rust/pcap_fast/verify.py <day_dir> <label_dir> [--stage 1|2|both] [--scratch DIR]

Stage 1: iter_merged_day_windows (bucket, host, every flow-dict field, the 30
packet features), compared window by window in lockstep with floats compared
by their bit pattern (float.hex), types included.
Stage 2: _parse_one's columns on the Python parser (u, i, ts, lbl, edge) + vocabularies, compared
with np.array_equal on the raw bit patterns and exact dtype equality.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import pcap_fast_py as fast  # noqa: E402


def _canon(x):
    if isinstance(x, float):
        return ("f", x.hex())
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, int):
        return ("i", x)
    if isinstance(x, str):
        return ("s", x)
    if isinstance(x, dict):
        return ("d", tuple((k, _canon(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return ("l", tuple(_canon(v) for v in x))
    raise TypeError(type(x))


def stage1(day_dir, window_seconds=2.0):
    from data_unification.pcap_bridge import iter_merged_day_windows
    # Streamed in lockstep (a real day does not fit as Python lists), so no
    # per-side timing here; bench.py measures throughput separately.
    ref = iter_merged_day_windows(day_dir, window_seconds=window_seconds)
    got = fast.iter_merged_day_windows_fast(day_dir, window_seconds=window_seconds)
    t = time.perf_counter()
    n_win = n_hw = n_flow = 0
    mism = []
    for k, (a, b) in enumerate(itertools.zip_longest(ref, got)):
        if a is None or b is None:
            mism.append((k, "window count differs"))
            break
        n_win += 1
        n_hw += len(a[1])
        n_flow += sum(len(v[0]) for v in a[1].values())
        if _canon(a[0]) != _canon(b[0]) or list(a[1]) != list(b[1]):
            mism.append((k, "bucket/host order", a[0], b[0], list(a[1]), list(b[1])))
            continue
        for host in a[1]:
            fa, pa = a[1][host]
            fb, pb = b[1][host]
            if _canon(fa) != _canon(fb):
                bad = [(i, x, y) for i, (x, y) in enumerate(zip(fa, fb)) if _canon(x) != _canon(y)]
                mism.append((k, host, "flows", len(fa), len(fb), bad[:2]))
            if _canon(pa) != _canon(pb):
                bad = [(c, pa[c], pb.get(c)) for c in pa
                       if c not in pb or _canon(pa[c]) != _canon(pb[c])]
                mism.append((k, host, "packet_features", bad[:4]))
    return {"windows": n_win, "host_windows": n_hw, "flows": n_flow,
            "mismatches": len(mism), "first_mismatches": [str(m)[:600] for m in mism[:5]],
            "t_both_s": round(time.perf_counter() - t, 3)}


def _ref_columns(day_dir, label_dir, scratch, window_seconds=2.0):
    from data_unification.parallel_ingest import _parse_one, iter_parts
    out = Path(scratch)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    res = _parse_one(("PCAP2018", str(day_dir), str(out), 1, None, 12, str(label_dir),
                      window_seconds, 500_000, "python"))
    if res.get("error"):
        raise RuntimeError(res["error"])
    if not res.get("n"):
        return res
    parts = list(iter_parts(res))
    cols = {c: np.concatenate([p[c] for p in parts]) for c in ("u", "i", "ts", "lbl", "edge")}
    res.update(cols)
    shutil.rmtree(out, ignore_errors=True)
    return res


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view({8: np.uint64, 4: np.uint32}[a.dtype.itemsize]) if a.dtype.kind == "f" else a


def stage2(day_dir, label_dir, scratch, window_seconds=2.0):
    t = time.perf_counter()
    ref = _ref_columns(day_dir, label_dir, scratch, window_seconds)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    got = fast.parse_pcap_day_columns_fast(day_dir, label_dir, window_seconds)
    t_fast = time.perf_counter() - t
    r = {"n_ref": ref.get("n"), "n_fast": got.get("n"), "t_ref_s": round(t_ref, 3),
         "t_fast_s": round(t_fast, 3)}
    if ref.get("n"):
        for c in ("u", "i", "ts", "lbl", "edge"):
            a, b = ref[c], got[c]
            same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(_bits(a), _bits(b))
            r[c] = "EXACT" if same else f"DIFF dtype {a.dtype}/{b.dtype} shape {a.shape}/{b.shape}"
            if not same and a.shape == b.shape:
                d = np.nonzero(_bits(a) != _bits(b))
                r[c] += f" n_diff={len(d[0])} first={[tuple(int(v) for v in x) for x in zip(*d)][:3]}"
        r["ips"] = "EXACT" if ref["ips"] == got["ips"] else "DIFF"
        r["cats"] = "EXACT" if ref["cats"] == got["cats"] else f"DIFF {ref['cats']} {got['cats']}"
        r["source"] = "EXACT" if ref["source"] == got["source"] else "DIFF"
        r["cats_list"] = ref["cats"]
        r["label_counts"] = np.bincount(ref["lbl"]).tolist()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("day_dir")
    ap.add_argument("label_dir")
    ap.add_argument("--stage", default="both")
    ap.add_argument("--scratch", default="/var/home/samito/Documents/SIH/rust_parser_experiment/scratch")
    a = ap.parse_args()
    out = {"day_dir": a.day_dir}
    if a.stage in ("1", "both"):
        out["stage1"] = stage1(a.day_dir)
    if a.stage in ("2", "both"):
        out["stage2"] = stage2(a.day_dir, a.label_dir, a.scratch)
    print(json.dumps(out, indent=1))
