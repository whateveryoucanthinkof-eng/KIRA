"""A synthetic TrajectoryStore with the real store's columns and row layout.

For profiling and equivalence tests of the Branch A / Branch B input
pipelines without the hours-long TGNE extraction. Values are random; the
STRUCTURE is what matters for speed:

  * rows are appended window by window, so one host's consecutive rows are
    spread across the feature block, interleaved with every other host active
    in between -- the layout `TrajectoryStoreBuilder` produces;
  * host trajectory lengths are heavy-tailed (most hosts are short, a few run
    for the whole capture), like the real corpus;
  * every column has the builder's dtype, `rows_by_host` holds views into one
    stable-argsort array exactly as `TrajectoryStoreBuilder.finalize` builds it,
    and a block above RESIDENT_FEATS_MAX_BYTES stays a MADV_RANDOM memmap.

Full scale (the 2026-09-26 production train split) is
`make_store(77_468_077, 3_491_797, ...)`: 8.4 GB of features on disk.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from data_unification.trajectory_store import (
    FEAT_DIM, TrajectoryStore, _resident_or_random_advised)

#: (coarse category, first technique, risk) -- a few real label combinations
_LABELS = [
    ("Benign", None, 0.0),
    ("Recon", "T1046", 0.55),
    ("InitialAccess", "T1110", 0.7),
    ("C2", "T1071", 0.9),
    ("Impact", "T1498", 0.95),
    ("UNKNOWN", None, 0.3),
]


def make_store(n_rows: int, n_hosts: int, *, n_windows: Optional[int] = None,
               seed: int = 0, feats_path: Optional[str] = None, small_frac: float = 0.92,
               attack_rate: float = 0.15, chunk_rows: int = 1 << 22) -> TrajectoryStore:
    rng = np.random.default_rng(seed)
    n_windows = int(n_windows or max(16, n_rows // max(1, n_hosts // 50)))

    # Trajectory lengths shaped like the real train split (2026-09-26 run: 3.49M
    # hosts, 77.5M rows, 67.8M samples kept, 6.17M dropped for short history):
    # most hosts appear in a handful of windows, a few run for the whole capture.
    n_small = int(n_hosts * small_frac)
    cnt = np.r_[rng.geometric(0.6, n_small),
                np.zeros(n_hosts - n_small, dtype=np.int64)].astype(np.int64)
    rest = n_rows - int(cnt.sum())
    if n_hosts > n_small:
        w = rng.lognormal(0.0, 1.5, n_hosts - n_small)
        cnt[n_small:] = np.maximum(1, np.floor(w / w.sum() * rest))
    cnt = np.minimum(cnt, n_windows)
    short = n_rows - int(cnt.sum())
    while short != 0:                    # fix the rounding on the long trajectories
        lo = n_small if n_hosts > n_small else 0
        h = rng.integers(lo, n_hosts, abs(short))
        if short > 0:
            np.add.at(cnt, h, 1)
            cnt = np.minimum(cnt, n_windows)
        else:
            np.subtract.at(cnt, h, 1)
            cnt = np.maximum(cnt, 1)
        short = n_rows - int(cnt.sum())
    cnt = cnt[rng.permutation(n_hosts)]

    # Each host's windows: `cnt[h]` distinct windows, uniform over the capture.
    host = np.repeat(np.arange(n_hosts, dtype=np.int32), cnt)
    win = np.empty(n_rows, dtype=np.int32)
    o = 0
    starts = rng.integers(0, n_windows, n_hosts)
    for h_lo in range(0, n_hosts, 1 << 16):
        h_hi = min(n_hosts, h_lo + (1 << 16))
        for h in range(h_lo, h_hi):
            c = int(cnt[h])
            if c == 1:
                win[o] = starts[h]
            else:
                # distinct sorted windows: a random strictly increasing walk, wrapped into range
                steps = rng.integers(1, max(2, n_windows // c), c)
                win[o:o + c] = np.minimum(np.cumsum(steps) - steps[0] + starts[h] % max(1, n_windows - int(steps.sum())),
                                          n_windows - 1)
                win[o:o + c] = np.maximum.accumulate(win[o:o + c])
            o += c
    # Append order: window by window, hosts in first-appearance order within a window.
    order = np.lexsort((host, win))
    host, win = host[order], win[order]
    # A host may have collided on its last windows (the clip above); keep it, the
    # store tolerates repeated windows (window_idx is not unique per host there either).

    # Host names interned in first-appearance order, like the builder does.
    first = np.full(n_hosts, -1, dtype=np.int64)
    uniq, idx = np.unique(host, return_index=True)
    first[uniq] = idx
    rank = np.argsort(np.argsort(first, kind="stable"), kind="stable").astype(np.int32)
    host_name_id = rank[host]
    host_names = [f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}@synth/cap{i % 7}"
                  for i in range(n_hosts)]

    lab = rng.choice(len(_LABELS), n_rows, p=[1 - attack_rate] + [attack_rate * 0.25] * 4 + [0.0])
    risk = np.array([r for _, _, r in _LABELS], dtype=np.float32)[lab]
    risk = risk + (lab > 0) * rng.random(n_rows, dtype=np.float32) * 0.04
    categories = [c for c, _, _ in _LABELS]
    techniques = sorted({t for _, t, _ in _LABELS if t})
    tech_of = np.array([techniques.index(t) if t else -1 for _, t, _ in _LABELS], dtype=np.int16)
    has = tech_of[lab] >= 0
    tech_off = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(has, out=tech_off[1:])
    tech_flat = tech_of[lab][has].astype(np.int16)

    window_seconds = 2.0
    window_start = win.astype(np.float64) * window_seconds + 1.3e9
    feats = _features(n_rows, rng, feats_path, chunk_rows)

    o2 = np.argsort(host_name_id, kind="stable")
    rows_by_host: Dict[str, np.ndarray] = {}
    sid = host_name_id[o2]
    st = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    bounds = np.r_[st, sid.size]
    for k in range(st.size):
        rows_by_host[host_names[int(sid[st[k]])]] = o2[bounds[k]:bounds[k + 1]]

    return TrajectoryStore(
        feats=feats, host_name_id=host_name_id, node_id=host_name_id.copy(),
        window_idx=win, window_start=window_start,
        window_end=window_start + window_seconds,
        is_attack=(lab > 0), risk_score=risk, cat_id=lab.astype(np.int16),
        tech_off=tech_off, tech_flat=tech_flat, categories=categories,
        techniques=techniques, host_names=host_names, rows_by_host=rows_by_host)


def _features(n_rows, rng, path, chunk_rows):
    if path is None:
        return rng.standard_normal((n_rows, FEAT_DIM), dtype=np.float32)
    if not (os.path.exists(path) and os.path.getsize(path) == n_rows * FEAT_DIM * 4):
        with open(path + ".tmp", "wb") as fh:
            for lo in range(0, n_rows, chunk_rows):
                k = min(chunk_rows, n_rows - lo)
                fh.write(rng.standard_normal((k, FEAT_DIM), dtype=np.float32).tobytes())
        os.replace(path + ".tmp", path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n_rows, FEAT_DIM))
    return _resident_or_random_advised(mm, n_rows)
