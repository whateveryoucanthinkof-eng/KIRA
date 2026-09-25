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
_f32p = ctypes.POINTER(ctypes.c_float)
_u8p = ctypes.POINTER(ctypes.c_uint8)


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
            vp = ctypes.c_void_p
            g = so.tgn_bita_plan
            g.restype = vp
            g.argtypes = [_i64p, _i64p, ctypes.c_int64, _i64p, _f64p, ctypes.c_int64, _i64p,
                          ctypes.c_int64, ctypes.c_int64, _i64p]
            g = so.tgn_bita_fill
            g.restype = ctypes.c_int32
            g.argtypes = [vp, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                          _i64p, _f32p, _i32p, _i64p, _f64p, _i64p, _f32p, _f64p]
            so.tgn_bita_free.restype = None
            so.tgn_bita_free.argtypes = [vp]
            g = so.tgn_nbr_plan
            g.restype = ctypes.c_int32
            g.argtypes = [_i32p, _i32p, _f64p, _i64p, ctypes.c_int64, _i64p, _f64p, ctypes.c_int64,
                          ctypes.c_int64, _f32p, ctypes.c_int64, ctypes.c_int64,
                          _i64p, _i64p, _f32p, _u8p, _u8p, _f32p]
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


def _p(a, ptype):
    return a.ctypes.data_as(ptype)


class BitaPlan:
    """A BiTA grouping computed by rust/tgn_host (plan.rs, tgn_bita_*).

    `sizes` = (n_upd, R, E, L). fill() writes the arrays into caller-owned
    buffers with the caller's padding (see tgn_bita_fill). Free with close()
    (also on garbage collection)."""

    __slots__ = ("_so", "_h", "n_upd", "R", "E", "L")

    def __init__(self, so, h, sizes):
        self._so, self._h = so, h
        self.n_upd, self.R, self.E, self.L = (int(x) for x in sizes)

    def fill(self, idx, dt, klen, owner, last_t, to_update, counts, node_ts, owner_fill=0):
        e_rows, l_cols = idx.shape
        for a, dt_ in ((idx, np.int64), (dt, np.float32), (klen, np.int32), (owner, np.int64),
                       (last_t, np.float64), (to_update, np.int64), (counts, np.float32), (node_ts, np.float64)):
            if not _ok(a, dt_):
                raise TypeError("BitaPlan.fill: buffers must be C-contiguous and correctly typed")
        if (dt.shape != idx.shape or len(klen) != e_rows or len(owner) != e_rows or len(last_t) != self.E
                or len(to_update) != self.n_upd or len(node_ts) != self.n_upd):
            raise ValueError("BitaPlan.fill: buffer shapes do not match the plan")
        rc = self._so.tgn_bita_fill(self._h, e_rows, l_cols, len(counts), int(owner_fill),
                                    _p(idx, _i64p), _p(dt, _f32p), _p(klen, _i32p), _p(owner, _i64p),
                                    _p(last_t, _f64p), _p(to_update, _i64p), _p(counts, _f32p),
                                    _p(node_ts, _f64p))
        if rc != 0:
            raise ValueError("BitaPlan.fill: padded sizes smaller than the plan")

    def close(self):
        if self._h:
            self._so.tgn_bita_free(self._h)
            self._h = None

    __del__ = close


def bita_plan(cnt, start, row_peer, row_t, nodes, max_seq_len: int):
    """BiTA grouping of the pending messages of `nodes` (see plan.rs), or
    None when the library is unavailable or refuses the input."""
    so = lib()
    if so is None:
        return None
    if not (_ok(cnt, np.int64) and _ok(start, np.int64) and _ok(row_peer, np.int64) and _ok(row_t, np.float64)):
        return None
    nodes = np.ascontiguousarray(nodes, dtype=np.int64)
    sizes = np.zeros(4, np.int64)
    h = so.tgn_bita_plan(_p(cnt, _i64p), _p(start, _i64p), len(cnt), _p(row_peer, _i64p), _p(row_t, _f64p),
                         min(len(row_peer), len(row_t)), _p(nodes, _i64p), len(nodes), int(max_seq_len),
                         _p(sizes, _i64p))
    if not h:
        return None
    return BitaPlan(so, h, sizes)


def nbr_plan(flat_nbr, flat_eidx, flat_ts, offsets, nodes, cut, k: int, host_ef, ints, deltas, mask, invalid, ef_n):
    """The neighbour block of one batch written into the given buffers (see
    plan.rs, tgn_nbr_plan): ints = [nodes ; neighbours ; edge idxs] int64,
    deltas [n, k] f32, mask [n, k] bool, invalid [n, 1] bool, ef_n [n, k, de]
    f32 or None. True on success, False when the numpy version must decide."""
    so = lib()
    if so is None or k < 1:
        return False
    if not (_ok(flat_nbr, np.int32) and _ok(flat_eidx, np.int32) and _ok(flat_ts, np.float64)
            and _ok(offsets, np.int64) and _ok(nodes, np.int64) and _ok(cut, np.float64)):
        return False
    n = len(nodes)
    if not (_ok(ints, np.int64) and ints.size == n + 2 * n * k and _ok(deltas, np.float32) and deltas.size == n * k
            and _ok(mask, np.bool_) and mask.size == n * k and _ok(invalid, np.bool_) and invalid.size == n):
        raise ValueError("nbr_plan: output buffers have the wrong type or size")
    if host_ef is not None:
        if not (_ok(host_ef, np.float32) and host_ef.ndim == 2 and ef_n is not None and _ok(ef_n, np.float32)
                and ef_n.size == n * k * host_ef.shape[1]):
            return False
        ef_ptr, n_rows, de, out_ef = _p(host_ef, _f32p), host_ef.shape[0], host_ef.shape[1], _p(ef_n, _f32p)
    else:
        ef_ptr, n_rows, de, out_ef = None, 0, 0, None
    rc = so.tgn_nbr_plan(_p(flat_nbr, _i32p), _p(flat_eidx, _i32p), _p(flat_ts, _f64p), _p(offsets, _i64p),
                         len(offsets), _p(nodes, _i64p), _p(cut, _f64p), n, k, ef_ptr, n_rows, de,
                         _p(ints, _i64p), _p(ints[n + n * k:], _i64p), _p(deltas, _f32p),
                         _p(mask.view(np.uint8), _u8p), _p(invalid.view(np.uint8), _u8p), out_ef)
    return rc == 0
