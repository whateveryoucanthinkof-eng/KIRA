"""A host-major copy of a TrajectoryStore's feature block.

## Why

`TrajectoryStore.feats` is written window by window, so one host's
consecutive rows sit far apart, interleaved with every other host active in
between. A Branch A sample reads 15 consecutive rows of ONE host: 15 scattered
108-byte rows, i.e. ~15 separate 4 KiB pages. When the block is a memmap that
does not fit in page cache (8.4 GB at full scale) each of those is a disk read.
Measured on a full-scale synthetic store with production's page cache:
1,660 major faults per batch of 128, 24.5 batch/s -- the 29 batch/s the
2026-09-26 run showed.

`TrajectoryStoreBuilder.finalize` already computes `order`, a stable argsort of
the rows by host, and every `rows_by_host[h]` is a slice of it. So
`feats[order]` holds each host's trajectory contiguously and in the same
order, and the rows `rows_by_host[h][a:b]` are exactly
`host_major[base_h + a : base_h + b]` -- one contiguous slice, 1-2 pages.
The values are copies, so anything gathered from it is bit-identical.

The copy is written in a few sequential passes over the source (never a
random read of the memmap), unlinked as soon as it is mapped (like the
builder's spill file), and advised MADV_RANDOM.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: RAM for one destination bucket while writing the copy.
BUCKET_BYTES = int(os.environ.get("CYBERWORLD_HOST_MAJOR_BUCKET_BYTES", 1 << 30))


def flat_row_order(rows_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(flat, base): `rows_list[i] == flat[base[i] : base[i] + len(rows_list[i])]`.

    When every array is a view of one 1-D int64 base (how `finalize` builds
    `rows_by_host`), `flat` IS that base and costs nothing; otherwise the
    arrays are concatenated.
    """
    base_arr = rows_list[0].base if rows_list else None
    if (base_arr is not None and isinstance(base_arr, np.ndarray) and base_arr.ndim == 1
            and base_arr.flags.c_contiguous
            and all(r.base is base_arr and r.ndim == 1 and r.strides == base_arr.strides
                    and r.dtype == base_arr.dtype for r in rows_list)):
        item = base_arr.dtype.itemsize
        p0 = base_arr.__array_interface__["data"][0]
        base = np.fromiter(((r.__array_interface__["data"][0] - p0) // item for r in rows_list),
                           dtype=np.int64, count=len(rows_list))
        return base_arr, base
    lens = np.fromiter((len(r) for r in rows_list), dtype=np.int64, count=len(rows_list))
    base = np.zeros(len(rows_list), dtype=np.int64)
    if len(lens) > 1:
        np.cumsum(lens[:-1], out=base[1:])
    flat = (np.concatenate([np.asarray(r, dtype=np.int64) for r in rows_list])
            if rows_list else np.zeros(0, np.int64))
    return flat, base


def host_major_copy(feats: np.ndarray, flat: np.ndarray, spill_dir: Optional[str] = None,
                    bucket_bytes: int = BUCKET_BYTES) -> np.ndarray:
    """`feats[flat]`, as a read-only MADV_RANDOM memmap in `spill_dir`.

    Destination rows are produced in buckets of `bucket_bytes`; each bucket is
    filled by one sequential scan of the source, so the source is only ever
    read sequentially.
    """
    n, d = len(flat), feats.shape[1]
    if spill_dir is None:
        spill_dir = os.path.dirname(getattr(feats, "filename", "") or "") or tempfile.gettempdir()
    os.makedirs(spill_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".hostmajor", dir=spill_dir)
    t = time.time()
    row_bytes = d * feats.dtype.itemsize
    src_map = getattr(feats, "_mmap", None)
    _advise(src_map, "MADV_SEQUENTIAL")      # the source is scanned in order; MADV_RANDOM would
                                             # fault it in 4 KiB at a time
    bucket = max(1, bucket_bytes // row_bytes)
    try:
        with os.fdopen(fd, "wb") as fh:
            if n:
                # inv[src_row] = destination position, for the rows that appear in flat
                n_src = feats.shape[0]
                dest_of = np.full(n_src, -1, dtype=np.int64)
                dest_of[flat] = np.arange(n, dtype=np.int64)
                chunk = max(1, (64 << 20) // row_bytes)
                buf = np.empty((min(bucket, n), d), dtype=feats.dtype)
                for b0 in range(0, n, bucket):
                    b1 = min(n, b0 + bucket)
                    for s0 in range(0, n_src, chunk):
                        s1 = min(n_src, s0 + chunk)
                        dst = dest_of[s0:s1]
                        m = (dst >= b0) & (dst < b1)
                        if m.any():
                            buf[dst[m] - b0] = feats[s0:s1][m]
                    fh.write(buf[:b1 - b0].tobytes())
                del dest_of
        out = (np.memmap(path, dtype=feats.dtype, mode="r", shape=(n, d))
               if n else np.zeros((0, d), dtype=feats.dtype))
    finally:
        _advise(src_map, "MADV_RANDOM")
        try:
            os.unlink(path)          # the mapping keeps the inode alive; space returns on exit
        except OSError:
            pass
    _advise(getattr(out, "_mmap", None), "MADV_RANDOM")
    log.info("host-major feature copy: %d rows, %.1f GB, %.1fs", n, n * row_bytes / 1e9,
             time.time() - t)
    return out


def _advise(mm, flag: str) -> None:
    if mm is None:
        return
    try:
        import mmap as _mmap
        mm.madvise(getattr(_mmap, flag))
    except (AttributeError, OSError, ValueError):
        pass             # advisory only


def host_major_for(store, rows_list, host_major: Optional[bool] = None,
                   spill_dir: Optional[str] = None):
    """(flat, base, feats_hm) for a dataset over `rows_list` of `store`.

    `feats_hm` is None when the block is held in RAM (the default when
    `host_major` is None): random gathers from RAM are cheap, and a copy would
    double the memory. The copy is cached on the store, so Branch B and DeepOP
    -- two datasets over the same store -- share one.
    """
    flat, base = flat_row_order(rows_list)
    if host_major is None:
        host_major = isinstance(store.feats, np.memmap)
    if not host_major:
        return flat, base, None
    cached = getattr(store, "_host_major_cache", None)
    if cached is not None and cached[0] is flat:
        return flat, base, cached[1]
    hm = host_major_copy(store.feats, flat, spill_dir)
    if flat is getattr(rows_list[0], "base", None):     # the store's own order: shareable
        store._host_major_cache = (flat, hm)
    return flat, base, hm


class BatchedView:
    """A dataset whose DataLoader batches come from one `ds.gather_batch(indices)`.

    Use with `collate_fn=collate_prebatched`. The samplers see the same length,
    so they draw the same indices in the same order as over `ds` itself.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def __getitems__(self, indices):
        return self.ds.gather_batch(indices)


def collate_prebatched(batch):
    """collate_fn for BatchedView: the batch is already collated."""
    return batch


def _torch():
    import torch
    return torch


class PermutationBatchSampler:
    """`BatchSampler(RandomSampler(n), batch_size, drop_last=False)` without the list.

    RandomSampler (generator=None) draws a seed from torch's global RNG, then
    yields `torch.randperm(n, generator=g).tolist()` -- for 68M samples a
    Python list of ~2.4 GB in the trainer's main process, every epoch. This
    draws the same seed the same way and yields the same batches from the
    permutation tensor (8 B per sample). Identical indices, identical order,
    identical global-RNG consumption: tests/test_branch_a_batched_loader.py.
    """

    def __init__(self, n: int, batch_size: int):
        self.n, self.batch_size = int(n), int(batch_size)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        torch = _torch()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(self.n, generator=g).numpy()
        bs = self.batch_size
        for i in range(0, self.n, bs):
            yield perm[i:i + bs]


def batched_loader(ds, batch_size: int, shuffle: bool, **loader_kw):
    """DataLoader over `ds.gather_batch`, drawing what DataLoader(ds, ...) would."""
    from torch.utils.data import DataLoader
    if shuffle:
        return DataLoader(BatchedView(ds), batch_sampler=PermutationBatchSampler(len(ds), batch_size),
                          collate_fn=collate_prebatched, **loader_kw)
    return DataLoader(BatchedView(ds), batch_size=batch_size, shuffle=False,
                      collate_fn=collate_prebatched, **loader_kw)
