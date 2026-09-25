"""ctypes bridge to rust/tgn_host: host-side hot loops of the TGN step.

Built with `cargo build --release` in rust/tgn_host (no dependencies; the
library is `rust/tgn_host/target/release/libtgn_host.so`, TGN_HOST_LIB
overrides). Every function here returns None when the library is missing,
older than its sources, disabled (CYBERWORLD_HOST_RUST=0) or given arrays it
does not handle; callers then run their numpy version, which computes the same
thing bit for bit (tests/test_tgn_host_rust.py).
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import numpy as np

_CRATE = Path(__file__).resolve().parents[2] / "rust" / "tgn_host"
_state = {"lib": None, "tried": False}

_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)
_f64p = ctypes.POINTER(ctypes.c_double)


def _stale(lib_path: Path) -> bool:
    built = lib_path.stat().st_mtime
    sources = list((_CRATE / "src").glob("*.rs")) + [_CRATE / "Cargo.toml"]
    return any(p.exists() and p.stat().st_mtime > built for p in sources)


def lib():
    if os.environ.get("CYBERWORLD_HOST_RUST", "") in ("0", "false", "False"):
        return None
    if not _state["tried"]:
        _state["tried"] = True
        path = Path(os.environ.get("TGN_HOST_LIB") or _CRATE / "target" / "release" / "libtgn_host.so")
        if not path.exists():
            logging.warning("rust/tgn_host not built (%s): host loops run in numpy. "
                            "Build: (cd rust/tgn_host && cargo build --release)", path)
        elif "TGN_HOST_LIB" not in os.environ and _stale(path):
            logging.warning("rust/tgn_host library is older than its sources: host loops run in "
                            "numpy until it is rebuilt (cd rust/tgn_host && cargo build --release)")
        else:
            so = ctypes.CDLL(str(path))
            f = so.tgn_recent_neighbors
            f.restype = ctypes.c_int32
            f.argtypes = [_i32p, _i32p, _f64p, _i64p, ctypes.c_int64, _i64p, _f64p,
                          ctypes.c_int64, ctypes.c_int64, _i32p, _i32p, _f64p]
            _state["lib"] = so
    return _state["lib"]


def _ok(a: np.ndarray, dtype) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == dtype and a.flags.c_contiguous


def recent_neighbors(flat_nbr, flat_eidx, flat_ts, offsets, nodes, cut, k: int):
    """NeighborFinder.get_temporal_neighbor, most-recent-n path: (neighbors
    int32 [n, k], edge_idxs int32 [n, k], edge_times float64 [n, k]) or None."""
    so = lib()
    if so is None:
        return None
    if not (_ok(flat_nbr, np.int32) and _ok(flat_eidx, np.int32) and _ok(flat_ts, np.float64)
            and _ok(offsets, np.int64)):
        return None
    nodes = np.ascontiguousarray(nodes, dtype=np.int64)
    cut = np.ascontiguousarray(cut, dtype=np.float64)
    n = len(nodes)
    out_nbr = np.empty((n, k), np.int32)
    out_eidx = np.empty((n, k), np.int32)
    out_ts = np.empty((n, k), np.float64)
    rc = so.tgn_recent_neighbors(
        flat_nbr.ctypes.data_as(_i32p), flat_eidx.ctypes.data_as(_i32p), flat_ts.ctypes.data_as(_f64p),
        offsets.ctypes.data_as(_i64p), len(offsets), nodes.ctypes.data_as(_i64p), cut.ctypes.data_as(_f64p),
        n, k, out_nbr.ctypes.data_as(_i32p), out_eidx.ctypes.data_as(_i32p), out_ts.ctypes.data_as(_f64p))
    if rc != 0:
        return None
    return out_nbr, out_eidx, out_ts
