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

FEAT_DIM = 27          # embedding(12) + temporal_attrs(15)
EMB_DIM = 12
_BLOCK = 262_144       # rows per in-RAM block before spilling


class _Col:
    """A growable numpy column.

    The builder originally accumulated these as Python lists and converted to
    numpy only in finalize(). That made the *finalized* store compact while the
    *building* peak stayed enormous: a Python list entry costs ~36 B (8 B
    pointer + 28 B int object) against 4 B for an int32 cell. At the ~100M
    snapshots a full-density corpus produces, those ten metadata lists came to
    tens of GB and dwarfed the memmapped feature block -- which is exactly how
    a run died with the spill file still at 0 bytes.
    """

    __slots__ = ("buf", "n")

    def __init__(self, dtype, capacity: int = 1 << 16):
        self.buf = np.empty(capacity, dtype=dtype)
        self.n = 0

    def append(self, v) -> None:
        if self.n == self.buf.shape[0]:
            bigger = np.empty(self.buf.shape[0] * 2, dtype=self.buf.dtype)
            bigger[: self.n] = self.buf
            self.buf = bigger
        self.buf[self.n] = v
        self.n += 1

    def finalize(self) -> np.ndarray:
        out = self.buf[: self.n].copy()
        self.buf = np.empty(0, dtype=self.buf.dtype)   # release promptly
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
                 categories, techniques, host_names, rows_by_host):
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


class TrajectoryStoreBuilder:
    """Accumulates snapshots columnar-side, spilling the bulk array to disk."""

    def __init__(self, spill_dir: Optional[str] = None):
        self.spill_dir = spill_dir
        self._spill_path: Optional[str] = None
        self._spill_fh = None
        self._block = np.zeros((_BLOCK, FEAT_DIM), dtype=np.float32)
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

    def append(self, *, host_ip, host_id, window_idx, window_start, window_end,
               embedding, temporal_attrs, is_attack, coarse_category,
               technique_ids, risk_score) -> None:
        if self._block_n == _BLOCK:
            if self._spill_fh is not None:
                self._flush_block()
            else:
                self._block = np.concatenate(
                    [self._block, np.zeros((_BLOCK, FEAT_DIM), dtype=np.float32)]
                )
        row = self._block_n
        self._block[row, :EMB_DIM] = embedding
        self._block[row, EMB_DIM:] = temporal_attrs
        self._block_n += 1

        self._host_name_id.append(self._intern(host_ip, self._host_index, self._host_names))
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

    def finalize(self) -> TrajectoryStore:
        if self._spill_fh is not None:
            self._flush_block()
            self._spill_fh.flush()
            self._spill_fh.close()
            self._spill_fh = None
            feats = (np.memmap(self._spill_path, dtype=np.float32, mode="r",
                               shape=(self._n, FEAT_DIM))
                     if self._n else np.zeros((0, FEAT_DIM), dtype=np.float32))
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
        )
