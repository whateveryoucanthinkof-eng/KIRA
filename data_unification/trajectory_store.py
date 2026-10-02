"""Columnar, optionally disk-backed storage for host trajectories.

Replaces `Dict[str, List[HostWindowSnapshot]]`, which cost ~647 bytes per
snapshot to hold ~108 bytes of actual numbers. At full corpus density that is
~25M snapshots = ~16 GB of Python object and numpy-header overhead, which is
what froze the dev laptop twice (see claude_latest_analysis + memory notes).

Measured: 647 B/snapshot as Python objects vs 132 B/snapshot columnar -- 4.9x.
With `spill_dir` set, the bulk array (27 float32 per row) is written to disk and
mapped back read-only, so resident memory becomes the small metadata columns
(~23 B/row) plus OS page cache. Sequential and strided reads over a memmap of
fixed-size rows measured 516 MB/s write and effectively free reads, so the
offload costs throughput, not correctness.

The public surface is deliberately identical to the dict it replaces: callers
index by host and get a sequence of HostWindowSnapshot. Snapshots are
materialized lazily, one at a time, so only the few alive during an iteration
exist as Python objects.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Dict, Iterator, List, Optional

import numpy as np

#: Default row width: embedding(12) + temporal_attrs(15). The builder takes it
#: as a parameter because the attribute vector is 45 wide when the extractor is
#: built with `include_packet_features=True` (the 30 packet-level features from
#: telemetry/packet/pcap_engine.py). Hardcoding 27 here silently truncated the
#: wide vector into a 27-column block.
FEAT_DIM = 27          # embedding(12) + temporal_attrs(15)
EMB_DIM = 12


def world_state(snapshot) -> np.ndarray:
    """s(t) = [TGNE embedding ; host attributes] for one snapshot.

    This is the enriched tensor Branch A reads, and the state Branch B
    models and DeepOP decodes. Branch B used to see only the first 12
    columns -- the bare TGNE latent -- so the world model never saw the
    flow volumes, peer counts and port counts the attributes carry, while
    docs/ARCHITECTURE.md described it as modelling the 27-D state.
    """
    return np.concatenate([np.asarray(snapshot.embedding, dtype=np.float32),
                           np.asarray(snapshot.temporal_attrs, dtype=np.float32)])
_BLOCK = 262_144       # rows per in-RAM block before spilling


class _Col:
    """A growable numpy column, stored as fixed-size chunks.

    The builder originally accumulated these as Python lists and converted to
    numpy only in finalize(). That made the *finalized* store compact while the
    *building* peak stayed enormous: a Python list entry costs ~36 B (8 B
    pointer + 28 B int object) against 4 B for an int32 cell. At the ~100M
    snapshots a full-density corpus produces, those ten metadata lists came to
    tens of GB and dwarfed the memmapped feature block -- which is exactly how
    a run died with the spill file still at 0 bytes.

    It then grew by doubling, which keeps up to 2x the data allocated (and 3x
    while copying into the bigger buffer). At the ~80M snapshots of the
    cross_year_ctu train split that slack was several GB against a 10 GB cap.
    Chunks of `_CHUNK` cells bound the slack to one chunk per column.
    """

    __slots__ = ("_chunks", "_cur", "_k", "n", "dtype")

    _CHUNK = 1 << 20

    def __init__(self, dtype, capacity: int = 1 << 16):
        self.dtype = np.dtype(dtype)
        self._chunks: list = []
        self._cur = np.empty(min(capacity, self._CHUNK), dtype=self.dtype)
        self._k = 0          # cells used in _cur
        self.n = 0

    def _roll(self) -> None:
        if self._cur.shape[0] < self._CHUNK:
            # Small columns start small; grow the first chunk up to _CHUNK.
            bigger = np.empty(min(self._cur.shape[0] * 2, self._CHUNK), dtype=self.dtype)
            bigger[: self._k] = self._cur[: self._k]
            self._cur = bigger
            return
        self._chunks.append(self._cur)
        self._cur = np.empty(self._CHUNK, dtype=self.dtype)
        self._k = 0

    def append(self, v) -> None:
        if self._k == self._cur.shape[0]:
            self._roll()
        self._cur[self._k] = v
        self._k += 1
        self.n += 1

    def extend(self, values) -> None:
        """Append many values; the same cells `append` would write, one by one."""
        values = np.asarray(values)
        k = values.shape[0]
        i = 0
        while i < k:
            if self._k == self._cur.shape[0]:
                self._roll()
            take = min(k - i, self._cur.shape[0] - self._k)
            self._cur[self._k:self._k + take] = values[i:i + take]
            self._k += take
            i += take
        self.n += k

    @property
    def buf(self) -> np.ndarray:
        """The values so far as ONE array (a copy once there are several
        chunks). Kept for callers that read `col.buf[: col.n]`; prefer `max()`."""
        if not self._chunks:
            return self._cur[: self._k]
        return np.concatenate(self._chunks + [self._cur[: self._k]])

    def max(self):
        parts = [c.max() for c in self._chunks]
        if self._k:
            parts.append(self._cur[: self._k].max())
        return max(parts)

    def finalize(self) -> np.ndarray:
        if self._chunks:
            out = np.concatenate(self._chunks + [self._cur[: self._k]])
        else:
            out = self._cur[: self._k].copy()
        self._chunks = []                                  # release promptly
        self._cur = np.empty(0, dtype=self.dtype)
        self._k = 0
        return out


class _TrajectoryView(Sequence):
    """A single host's trajectory. Materializes snapshots on demand."""

    __slots__ = ("_store", "_rows")

    def __init__(self, store: "TrajectoryStore", rows: np.ndarray):
        self._store = store
        self._rows = rows

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._store._materialize(int(r)) for r in self._rows[i]]
        return self._store._materialize(int(self._rows[i]))

    def __iter__(self):
        for r in self._rows:
            yield self._store._materialize(int(r))

    def __repr__(self) -> str:
        return f"<trajectory of {len(self)} snapshots>"


class TrajectoryStore(Mapping):
    """Behaves like Dict[str, List[HostWindowSnapshot]] over columnar arrays."""

    def __init__(self, feats, host_name_id, node_id, window_idx, window_start, window_end,
                 is_attack, risk_score, cat_id, tech_off, tech_flat,
                 categories, techniques, host_names, rows_by_host, unknown_intervals=None):
        self.feats = feats
        self.host_name_id = host_name_id   # index into host_names
        self.node_id = node_id             # the graph node id from adapter.ip_to_id
        self.window_idx = window_idx
        self.window_start = window_start
        self.window_end = window_end
        self.is_attack = is_attack
        self.risk_score = risk_score
        self.cat_id = cat_id
        self.tech_off = tech_off
        self.tech_flat = tech_flat
        self.categories = categories
        self.techniques = techniques
        self.host_names = host_names
        self._rows_by_host = rows_by_host
        #: capture namespace -> sorted [[start, end], ...] of UNKNOWN traffic
        self.unknown_intervals = dict(unknown_intervals or {})

    # -- Mapping surface -------------------------------------------------
    def __getitem__(self, host: str) -> _TrajectoryView:
        return _TrajectoryView(self, self._rows_by_host[host])

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows_by_host)

    def __len__(self) -> int:
        return len(self._rows_by_host)

    # -- internals -------------------------------------------------------
    def _materialize(self, row: int):
        from data_unification.multi_dataset_stream import HostWindowSnapshot
        lo, hi = int(self.tech_off[row]), int(self.tech_off[row + 1])
        return HostWindowSnapshot(
            host_ip=self.host_names[int(self.host_name_id[row])],
            host_id=int(self.node_id[row]),
            window_idx=int(self.window_idx[row]),
            window_start=float(self.window_start[row]),
            window_end=float(self.window_end[row]),
            embedding=self.feats[row, :EMB_DIM],
            temporal_attrs=self.feats[row, EMB_DIM:],
            is_attack=bool(self.is_attack[row]),
            coarse_category=self.categories[int(self.cat_id[row])],
            technique_ids=[self.techniques[int(t)] for t in self.tech_flat[lo:hi]],
            risk_score=float(self.risk_score[row]),
        )

    def target_fields(self, row: int):
        """(risk, coarse_category, first_technique_or_None, window_idx) for one row.

        `_materialize` costs ~86us per call; the two memmap feature slices and
        the dataclass construction dominate it, and a *target* uses none of
        them. Reading these four columns directly measures ~1.0us. Branch A
        asks for one target per sample, so at 20.66M samples that is 29.6
        minutes per epoch of main-process time -- time the GPU spends idle.

        Returns the first technique or None; the caller owns the default,
        because "no technique" is not the store's concept to name.
        """
        lo, hi = int(self.tech_off[row]), int(self.tech_off[row + 1])
        return (
            float(self.risk_score[row]),
            self.categories[int(self.cat_id[row])],
            self.techniques[int(self.tech_flat[lo])] if hi > lo else None,
            int(self.window_idx[row]),
        )

    def time_to_next_attack(self, never: float = np.inf) -> np.ndarray:
        """Seconds from each window until that host's NEXT attack window.

        0.0 where the window is itself an attack; `never` where the host has
        no later attack window in this split. One reverse scan per host over
        columns that already exist, so this costs a pass over the store and
        no extra memory beyond the output.

        Rows within a host are in encounter order, which is time order --
        `window_idx` is assigned per window as captures are read, and the
        builder appends per host per window. A host's row slice is therefore
        already sorted by `window_start`; this asserts that rather than
        assuming it, because a hazard computed on out-of-order rows would be
        silently wrong rather than raising.
        """
        n = self.n_snapshots
        out = np.full(n, float(never), dtype=np.float64)
        atk = np.asarray(self.is_attack).astype(bool)
        start = np.asarray(self.window_start, dtype=np.float64)

        for rows in self._rows_by_host.values():
            r = np.asarray(rows)
            if r.size == 0:
                continue
            ts = start[r]
            if r.size > 1 and not np.all(np.diff(ts) >= 0):
                order = np.argsort(ts, kind="stable")
                r, ts = r[order], ts[order]
            a = atk[r]
            # Reverse running minimum of the attack timestamps: for each row,
            # the earliest attack at or after it. A Python loop here costs one
            # interpreter step per row -- 22.8M of them at full density -- so
            # it is done with an accumulate instead.
            masked = np.where(a, ts, np.inf)
            next_atk = np.minimum.accumulate(masked[::-1])[::-1]
            res = next_atk - ts
            res[a] = 0.0
            res[~np.isfinite(next_atk)] = float(never)
            out[r] = res
        return out

    def hazard_risk(self, tau_seconds: float, never: float = np.inf) -> np.ndarray:
        """A forward-looking risk target: exp(-dt / tau).

        ## Why this exists

        The `risk_score` written during extraction is
        `base_severity(tactic) + 0.04*density + 0.04*volume` for an attack
        window and exactly 0.0 otherwise. Three things follow, and all three
        were measured on the 2026-09-21 run:

        1. It is **not a forecast**. It describes whether THIS window contains
           attack traffic. A model asked to predict it is doing detection with
           a one-step delay, not forecasting -- which is the stated purpose of
           the system.
        2. It is **nearly a function of the coarse category**, which a
           different head already predicts. The two heads were being trained
           to carry the same information.
        3. It is **bimodal** -- 0.0 for 82.5% of validation windows, ~0.76 for
           the rest -- so regressing it with a mean-seeking loss produces a
           head that cannot beat predicting zero on MAE. Branch A's did not:
           0.2268 against 0.1334 for the constant 0.

        The hazard form answers the question a SOC actually asks -- *how soon
        is this host going to be in trouble* -- and is continuous, monotone in
        time, and independent of the tactic label. It is 1.0 during an attack,
        decays smoothly beforehand, and reaches 0 for a host that is never
        attacked.

        `tau_seconds` sets the decay scale; the forecast horizon
        (forecast_steps * window_seconds) is the natural choice, so a host one
        full horizon away from an attack scores exp(-1) = 0.368.
        """
        if not (tau_seconds > 0):
            raise ValueError(f"tau_seconds must be positive, got {tau_seconds}")
        dt = self.time_to_next_attack(never=never)
        with np.errstate(over="ignore"):
            out = np.exp(-dt / float(tau_seconds))
        out[~np.isfinite(dt)] = 0.0
        self._censor_unknown(out, dt, float(tau_seconds))
        return out.astype(np.float32)

    #: A target is censored only if the unknown span could move it by at least
    #: this much: beyond ~4.6 tau the hazard is < 0.01 whatever happened there.
    CENSOR_EPS = 0.01

    def _censor_unknown(self, out, dt, tau) -> int:
        """NaN where the target depends on traffic whose label is UNKNOWN.

        UNKNOWN traffic (an attack interval nobody can attribute, an unmapped
        label) is dropped before extraction, so a host's window just before
        such a span sees no "next attack" and its hazard read 0 -- a confident
        NEGATIVE for exactly the pre-attack windows early warning learns from.
        Where an unknown span starts before the next known attack and close
        enough to matter (exp(-du/tau) >= CENSOR_EPS), the true target is
        unknown: NaN, which every trainer masks. A window that is itself an
        attack (dt == 0) is known and kept."""
        start = np.asarray(self.window_start, dtype=np.float64)
        # The end of each capture is where observation stops: everything after
        # it is unobserved, exactly like a dropped UNKNOWN span. A window near
        # the end with no later attack read hazard 0 -- a negative about
        # traffic nobody recorded. Each capture (namespace) gets an open span
        # from just after its last window.
        ends: Dict[Optional[str], float] = {}
        for host, rows in self._rows_by_host.items():
            ns = host.split("@", 1)[1] if "@" in host else None
            r = np.asarray(rows)
            if r.size:
                ends[ns] = max(ends.get(ns, -np.inf), float(start[r].max()))
        spans = {ns: list(iv) for ns, iv in self.unknown_intervals.items()}
        w = float(np.median(np.asarray(self.window_end, dtype=np.float64)[:1000]
                            - start[:1000])) if len(start) else 0.0
        for ns, t_end in ends.items():
            spans.setdefault(ns, []).append([t_end + w, np.inf])
        merged = {}
        for ns, iv in spans.items():
            a = np.asarray(iv, dtype=np.float64).reshape(-1, 2)
            a = a[np.argsort(a[:, 0], kind="stable")]
            # merge overlaps so the ends are sorted too
            s_, e_ = [], []
            for lo, hi in a:
                if s_ and lo <= e_[-1]:
                    e_[-1] = max(e_[-1], hi)
                else:
                    s_.append(lo)
                    e_.append(hi)
            merged[ns] = (np.asarray(s_), np.asarray(e_))
        n = 0
        for host, rows in self._rows_by_host.items():
            ns = host.split("@", 1)[1] if "@" in host else None
            if ns not in merged:
                continue
            a, b = merged[ns]
            r = np.asarray(rows)
            t = start[r]
            k = np.searchsorted(b, t, side="left")          # first span not over before t
            inside = k < len(a)
            du = np.full(t.shape, np.inf)
            du[inside] = np.maximum(a[k[inside]] - t[inside], 0.0)
            d = dt[r]
            cens = (du <= d) & (np.exp(-du / tau) >= self.CENSOR_EPS) & (d != 0)
            out[r[cens]] = np.nan
            n += int(cens.sum())
        self.n_censored = n
        return n

    def use_hazard_target(self, tau_seconds: float) -> dict:
        """Replace `risk_score` with the hazard target, in place.

        Every consumer -- Branch A's sequence dataset, Branch B's rollout
        dataset, DeepOP -- reads `store.risk_score`, so swapping the column
        here reaches all of them without each one growing a flag. The original
        severity values stay available as `risk_score_severity` for reporting
        and for reproducing an earlier run.

        Returns a summary worth printing: a target that is 82.5% zeros is the
        thing that broke the old risk head, so the zero fraction is the number
        to look at when deciding whether the swap did what it was meant to.
        """
        if getattr(self, "_hazard_tau", None) is not None:
            raise RuntimeError(
                f"hazard target already applied with tau={self._hazard_tau}; "
                f"applying it twice would decay an already-decayed target")
        sev = np.asarray(self.risk_score)
        haz = self.hazard_risk(tau_seconds)
        self.risk_score_severity = sev
        self.risk_score = haz
        self._hazard_tau = float(tau_seconds)
        return {
            "tau_seconds": float(tau_seconds),
            "zero_fraction_before": float((sev == 0).mean()),
            "zero_fraction_after": float((haz == 0).mean()),
            "distinct_before": int(len(np.unique(np.round(sev, 4)))),
            "distinct_after": int(len(np.unique(np.round(haz, 4)))),
            "mean_before": float(sev.mean()),
            "mean_after": float(np.nanmean(haz)) if haz.size else 0.0,
            # targets that depend on dropped UNKNOWN traffic: NaN, masked
            "censored": int(np.isnan(haz).sum()),
        }

    @property
    def n_snapshots(self) -> int:
        return int(self.feats.shape[0])

    def memory_report(self) -> Dict[str, float]:
        res = sum(a.nbytes for a in (self.host_name_id, self.node_id, self.window_idx, self.window_start,
                                     self.window_end, self.is_attack, self.risk_score,
                                     self.cat_id, self.tech_off, self.tech_flat))
        mapped = self.feats.nbytes
        return {
            "snapshots": self.n_snapshots,
            "resident_mb": res / 1e6,
            "feats_mb": mapped / 1e6,
            "feats_on_disk": isinstance(self.feats, np.memmap),
            "bytes_per_snapshot": (res + mapped) / max(1, self.n_snapshots),
        }


#: A feature block at or under this size is read back into RAM instead of
#: being left as a memmap. 2.30 GiB is the full-density Branch A block, so
#: the default covers it with room to spare.
RESIDENT_FEATS_MAX_BYTES = int(os.environ.get(
    "CYBERWORLD_RESIDENT_FEATS_MAX_BYTES", 6 * 1024 ** 3))


def _resident_or_random_advised(feats, n_rows):
    """Keep a small feature block in RAM; tell the kernel not to read ahead.

    ## What this fixes

    Spilling exists so the feature block does not have to be resident. But
    training reads it in a *random* order -- every sample gathers `seq_len`
    scattered rows, 20.66M times an epoch -- and a memmap page fault triggers
    the kernel's default 128 KiB readahead for a 108-byte row. When the page
    cache cannot hold the block, essentially every access goes to disk and is
    amplified ~1000x.

    Measured on the 2026-09-22 run, with two trainers sharing a 22 GiB box:
    four DataLoader workers had each read **10.4 TiB** (41.8 TiB total) from a
    block of **2.30 GiB**, and throughput fell from 52.5 batch/s to 8.1 --
    6.5x slower, putting an 8-epoch run at ~44 hours.

    The block is 2.30 GiB. Spilling it was never worth it: holding it resident
    costs less memory than the page cache was trying to use for it anyway.
    So a block at or under RESIDENT_FEATS_MAX_BYTES is read back into RAM and
    the mapping dropped.

    A block genuinely too large to hold stays memmapped, but gets
    MADV_RANDOM, which turns off readahead: the kernel then fetches the 4 KiB
    page actually touched instead of 128 KiB around it -- a 32x reduction in
    the amplification for exactly this access pattern.
    """
    if n_rows == 0 or not isinstance(feats, np.memmap):
        return feats
    if feats.nbytes <= RESIDENT_FEATS_MAX_BYTES:
        # np.ascontiguousarray on an ALREADY C-contiguous memmap returns a
        # VIEW, not a copy, so this branch used to return something still
        # backed by the spill file and the whole fix was inert -- the block
        # got neither the resident copy NOR the MADV_RANDOM fallback below.
        # Verified: the result had OWNDATA False and a .base chain of
        # memmap -> mmap, and overwriting the backing file on disk changed
        # the array's contents.
        resident = np.empty(feats.shape, dtype=feats.dtype)
        np.copyto(resident, feats)               # one sequential read
        del feats                                # drop the mapping
        return resident
    try:
        import mmap as _mmap
        feats._mmap.madvise(_mmap.MADV_RANDOM)
    except (AttributeError, OSError, ValueError):
        pass          # advisory only; correctness does not depend on it
    return feats


def capture_namespace(path) -> str:
    """A stable, unique identity for one capture file: `<parent>/<stem>`.

    The parent directory is part of it because CTU-13 disambiguates its
    scenarios by directory (`1/`, `2/`, ...), and two captures must never share
    a namespace or their hosts would merge again.
    """
    from pathlib import Path as _P
    # Accept a training_sources.Capture too: Branch A's loader iterates those,
    # and passing one here raised TypeError on the first capture of the run
    # (found by scripts/dry_run_plan.py after the testing-prod merge).
    p = _P(getattr(path, "path", path))
    return f"{p.parent.name}/{p.stem}"


class TrajectoryStoreBuilder:
    """Accumulates snapshots columnar-side, spilling the bulk array to disk."""

    def __init__(self, spill_dir: Optional[str] = None, feat_dim: int = FEAT_DIM):
        self.spill_dir = spill_dir
        self.feat_dim = int(feat_dim)
        self._namespace: Optional[str] = None
        #: namespace -> [[start, end], ...] spans whose traffic was dropped as
        #: UNKNOWN (capture_columns.unknown_intervals); see hazard_risk.
        self._unknown: Dict[Optional[str], list] = {}
        self._spill_path: Optional[str] = None
        self._spill_fh = None
        self._block = np.zeros((_BLOCK, self.feat_dim), dtype=np.float32)
        self._block_n = 0
        self._n = 0

        self._host_name_id = _Col(np.int32)
        self._node_id = _Col(np.int32)
        self._window_idx = _Col(np.int32)
        self._window_start = _Col(np.float64)
        self._window_end = _Col(np.float64)
        self._is_attack = _Col(np.bool_)
        self._risk = _Col(np.float32)
        self._cat_id = _Col(np.int16)
        self._tech_off = _Col(np.int64); self._tech_off.append(0)
        self._tech_flat = _Col(np.int16)

        self._cat_index: Dict[str, int] = {}
        self._categories: List[str] = []
        self._tech_index: Dict[str, int] = {}
        self._techniques: List[str] = []
        self._host_index: Dict[str, int] = {}
        self._host_names: List[str] = []

        if spill_dir is not None:
            os.makedirs(spill_dir, exist_ok=True)
            fd, self._spill_path = tempfile.mkstemp(suffix=".feats", dir=spill_dir)
            self._spill_fh = os.fdopen(fd, "wb")

    def _intern(self, value, index, names) -> int:
        i = index.get(value)
        if i is None:
            i = len(names)
            index[value] = i
            names.append(value)
        return i

    def _flush_block(self) -> None:
        if self._block_n == 0 or self._spill_fh is None:
            return
        self._spill_fh.write(self._block[: self._block_n].tobytes())
        self._block_n = 0

    def set_namespace(self, namespace) -> None:
        """Scope every host key appended from now on to `namespace`.

        ## Why this exists

        Hosts were keyed by the bare IP string, and the trainers feed one
        shared builder capture after capture. So the same string in two
        captures became ONE trajectory. Measured on the train split:

          fri_16 & wed_14 (CIC-2018)     350 of 350 hosts shared
          two CTU-13 scenarios           28,474 hosts shared
          tue_20 (2018) & CTU-13 (2011)  59 hosts shared

        Nine of the ten CIC-2018 days carry no IP columns and fabricate
        `192.168.10.{i % 250 + 1}`, so every fabricated day's hosts merged with
        every other's. And rows are appended in file order, which puts the
        2018 captures before the 2011 ones: a merged host's "history" could
        come from 2018 and its "next window" from 2011 -- predicting the past
        from the future. The sequence, rollout and hazard targets all read
        `_rows_by_host`, so all three were affected.

        A host in capture A and a host in capture B are never the same
        trajectory, even when they share an address. Call this once per
        capture before extracting it; `None` restores bare-IP keys.
        """
        self._namespace = None if namespace is None else str(namespace)

    def add_unknown_intervals(self, intervals, namespace="__current__") -> None:
        """Record spans of this capture whose labels are unknown (their traffic
        was dropped as UNKNOWN). Targets that look ahead into them are censored."""
        ns = self._namespace if namespace == "__current__" else namespace
        if intervals:
            self._unknown.setdefault(ns, []).extend([float(a), float(b)] for a, b in intervals)

    def append(self, *, host_ip, host_id, window_idx, window_start, window_end,
               embedding, temporal_attrs, is_attack, coarse_category,
               technique_ids, risk_score) -> None:
        self._make_room()
        row = self._block_n
        got = EMB_DIM + len(temporal_attrs)
        if got != self.feat_dim:
            raise ValueError(
                f"snapshot is {got} wide ({EMB_DIM} embedding + "
                f"{len(temporal_attrs)} attributes) but this store was built "
                f"for {self.feat_dim}. Build the store with "
                f"feat_dim={got} -- e.g. TrajectoryStoreBuilder(feat_dim="
                f"EMB_DIM + EXTENDED_HOST_ATTR_DIM) when the extractor has "
                f"include_packet_features=True."
            )
        self._block[row, :EMB_DIM] = embedding
        self._block[row, EMB_DIM:] = temporal_attrs
        self._block_n += 1

        _key = host_ip if self._namespace is None else f"{host_ip}@{self._namespace}"
        self._host_name_id.append(self._intern(_key, self._host_index, self._host_names))
        self._node_id.append(int(host_id))
        self._window_idx.append(window_idx)
        self._window_start.append(window_start)
        self._window_end.append(window_end)
        self._is_attack.append(bool(is_attack))
        self._risk.append(float(risk_score))
        self._cat_id.append(self._intern(coarse_category, self._cat_index, self._categories))
        for t in (technique_ids or ()):
            self._tech_flat.append(self._intern(t, self._tech_index, self._techniques))
        self._tech_off.append(self._tech_flat.n)
        self._n += 1

    def next_window_base(self) -> int:
        """1 + the largest window_idx appended so far (0 when empty): the
        `window_idx_base` for the next capture."""
        return int(self._window_idx.max()) + 1 if self._window_idx.n else 0

    def _make_room(self) -> None:
        """Free the in-RAM block when it is full: spill it, or grow it.

        Growth used to trigger only at `_block_n == _BLOCK`, so without a
        spill dir the block grew once and the row after 2 * _BLOCK raised
        IndexError. Comparing against the block's actual size keeps the spill
        behaviour identical (the block is always _BLOCK rows there) and lets
        the in-RAM one keep growing.
        """
        if self._block_n == self._block.shape[0]:
            if self._spill_fh is not None:
                self._flush_block()
            else:
                self._block = np.concatenate(
                    [self._block, np.zeros((_BLOCK, self.feat_dim), dtype=np.float32)]
                )

    def append_batch(self, *, host_ips, host_ids, window_idx, window_start, window_end,
                     embeddings, temporal_attrs, is_attack, coarse_categories,
                     technique_ids, risk_scores) -> None:
        """`append` for every row of one window, in row order; identical result.

        Per-row arguments are sequences of equal length; `window_idx`,
        `window_start` and `window_end` are shared by all rows. Interning
        (hosts, categories, techniques) happens in row order, so ids come out
        as the per-row calls would assign them.
        """
        m = len(host_ips)
        if m == 0:
            return
        emb = np.asarray(embeddings)
        att = np.asarray(temporal_attrs)
        got = EMB_DIM + att.shape[1]
        if got != self.feat_dim:
            raise ValueError(
                f"snapshot is {got} wide ({EMB_DIM} embedding + "
                f"{att.shape[1]} attributes) but this store was built "
                f"for {self.feat_dim}. Build the store with "
                f"feat_dim={got} -- e.g. TrajectoryStoreBuilder(feat_dim="
                f"EMB_DIM + EXTENDED_HOST_ATTR_DIM) when the extractor has "
                f"include_packet_features=True."
            )
        i = 0
        while i < m:
            self._make_room()
            k = min(m - i, self._block.shape[0] - self._block_n)
            b = self._block_n
            self._block[b:b + k, :EMB_DIM] = emb[i:i + k]
            self._block[b:b + k, EMB_DIM:] = att[i:i + k]
            self._block_n += k
            i += k

        ns = self._namespace
        hidx, hnames = self._host_index, self._host_names
        keys = host_ips if ns is None else [f"{ip}@{ns}" for ip in host_ips]
        self._host_name_id.extend(np.fromiter(
            (self._intern(kk, hidx, hnames) for kk in keys), dtype=np.int64, count=m))
        self._node_id.extend(np.asarray(host_ids).astype(np.int64))
        self._window_idx.extend(np.full(m, int(window_idx), dtype=np.int64))
        self._window_start.extend(np.full(m, window_start, dtype=np.float64))
        self._window_end.extend(np.full(m, window_end, dtype=np.float64))
        self._is_attack.extend(np.asarray(is_attack, dtype=bool))
        self._risk.extend(np.asarray(risk_scores, dtype=np.float64))
        cidx, cnames = self._cat_index, self._categories
        self._cat_id.extend(np.fromiter(
            (self._intern(c, cidx, cnames) for c in coarse_categories), dtype=np.int64, count=m))
        tidx, tnames = self._tech_index, self._techniques
        flat: list = []
        offs = np.empty(m, dtype=np.int64)
        base = self._tech_flat.n
        for r, techs in enumerate(technique_ids):
            for t in (techs or ()):
                flat.append(self._intern(t, tidx, tnames))
            offs[r] = base + len(flat)
        if flat:
            self._tech_flat.extend(np.asarray(flat, dtype=np.int64))
        self._tech_off.extend(offs)
        self._n += m

    # -- parts: one capture extracted elsewhere, appended here -------------
    _PART_COLS = ("_host_name_id", "_node_id", "_window_idx", "_window_start", "_window_end",
                  "_is_attack", "_risk", "_cat_id", "_tech_off", "_tech_flat")

    def export_part(self, out_dir) -> None:
        """Write everything appended so far to `out_dir` for `append_part`.

        Only for a builder that started empty, spilled to `out_dir` itself and
        is not used afterwards (data_unification/parallel_extract.py). Its
        local ids are then in first-appearance order, which `append_part`
        relies on to reproduce the interning a single builder would have done.
        """
        import json
        if self._spill_fh is None or os.path.dirname(self._spill_path) != os.path.abspath(str(out_dir)):
            raise ValueError("export_part needs a builder spilling into out_dir")
        self._flush_block()
        self._spill_fh.close()
        self._spill_fh = None
        os.replace(self._spill_path, os.path.join(out_dir, "feats.bin"))
        for name in self._PART_COLS:
            np.save(os.path.join(out_dir, f"{name[1:]}.npy"), getattr(self, name).finalize())
        with open(os.path.join(out_dir, "part.json"), "w") as f:
            json.dump({"n": self._n, "feat_dim": self.feat_dim, "host_names": self._host_names,
                       "categories": self._categories, "techniques": self._techniques}, f)

    def append_part(self, part_dir, window_idx_offset: int = 0) -> None:
        """Append a part written by `export_part`, as if its rows had been
        appended here one by one with `window_idx_base` raised by
        `window_idx_offset`. Ids are re-interned in the part's local id order,
        which is its first-appearance order, so they come out identical."""
        import json
        with open(os.path.join(part_dir, "part.json")) as f:
            meta = json.load(f)
        n = int(meta["n"])
        if int(meta["feat_dim"]) != self.feat_dim:
            raise ValueError(f"part is {meta['feat_dim']} wide, store is {self.feat_dim}")
        if n == 0:
            return
        col = {name[1:]: np.load(os.path.join(part_dir, f"{name[1:]}.npy"))
               for name in self._PART_COLS}
        feats = np.memmap(os.path.join(part_dir, "feats.bin"), dtype=np.float32, mode="r",
                          shape=(n, self.feat_dim))
        i = 0
        while i < n:
            self._make_room()
            k = min(n - i, self._block.shape[0] - self._block_n)
            self._block[self._block_n:self._block_n + k] = feats[i:i + k]
            self._block_n += k
            i += k
        del feats

        def remap(names, index, table):
            return np.array([self._intern(x, index, table) for x in names], dtype=np.int64)

        hmap = remap(meta["host_names"], self._host_index, self._host_names)
        cmap = remap(meta["categories"], self._cat_index, self._categories)
        tmap = remap(meta["techniques"], self._tech_index, self._techniques)
        self._host_name_id.extend(hmap[col["host_name_id"]] if hmap.size else col["host_name_id"])
        self._node_id.extend(col["node_id"])
        self._window_idx.extend(col["window_idx"].astype(np.int64) + int(window_idx_offset))
        self._window_start.extend(col["window_start"])
        self._window_end.extend(col["window_end"])
        self._is_attack.extend(col["is_attack"])
        self._risk.extend(col["risk"])
        self._cat_id.extend(cmap[col["cat_id"]] if cmap.size else col["cat_id"])
        base = self._tech_flat.n
        if col["tech_flat"].size:
            self._tech_flat.extend(tmap[col["tech_flat"]])
        self._tech_off.extend(col["tech_off"][1:].astype(np.int64) + base)
        self._n += n

    def finalize(self) -> TrajectoryStore:
        if self._spill_fh is not None:
            self._flush_block()
            self._spill_fh.flush()
            self._spill_fh.close()
            self._spill_fh = None
            feats = (np.memmap(self._spill_path, dtype=np.float32, mode="r",
                               shape=(self._n, self.feat_dim))
                     if self._n else np.zeros((0, self.feat_dim), dtype=np.float32))
            feats = _resident_or_random_advised(feats, self._n)
            # Unlink the backing file NOW, while the mapping holds it open.
            #
            # On POSIX the inode survives until every reference is dropped, so
            # the memmap above stays fully valid, and the space is reclaimed
            # automatically when the mapping is released -- including on a
            # crash or a kill, which no explicit cleanup path can promise.
            #
            # Nothing deleted these before. At full density each of Branch A,
            # Branch B and DeepOP writes a multi-GB block, and this disk is
            # already 90% full; a handful of runs would have filled it. A
            # 216 MB orphan from an earlier run is what exposed it.
            try:
                os.unlink(self._spill_path)
            except OSError as exc:
                logging.getLogger(__name__).warning(
                    "could not unlink spill file %s: %s (it will need manual "
                    "cleanup)", self._spill_path, exc)
        else:
            feats = self._block[: self._n]

        host_name_id = self._host_name_id.finalize()

        # Group rows per host from the column itself. A stable sort keeps each
        # host's rows in the order they were appended (i.e. time order), which
        # is what downstream `sorted(snaps, key=window_idx)` expects, and costs
        # two arrays instead of one Python list per host.
        order = np.argsort(host_name_id, kind="stable")
        rows_by_host: Dict[str, np.ndarray] = {}
        if order.size:
            sorted_ids = host_name_id[order]
            starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
            bounds = np.r_[starts, sorted_ids.size]
            for k in range(starts.size):
                rows_by_host[self._host_names[int(sorted_ids[starts[k]])]] = \
                    order[bounds[k]:bounds[k + 1]]

        return TrajectoryStore(
            feats=feats,
            host_name_id=host_name_id,
            node_id=self._node_id.finalize(),
            window_idx=self._window_idx.finalize(),
            window_start=self._window_start.finalize(),
            window_end=self._window_end.finalize(),
            is_attack=self._is_attack.finalize(),
            risk_score=self._risk.finalize(),
            cat_id=self._cat_id.finalize(),
            tech_off=self._tech_off.finalize(),
            tech_flat=self._tech_flat.finalize(),
            categories=self._categories,
            techniques=self._techniques,
            host_names=self._host_names,
            rows_by_host=rows_by_host,
            unknown_intervals={k: sorted(v) for k, v in self._unknown.items()},
        )
