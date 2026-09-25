"""One encoder setup, built once and shared read-only by every run that needs it.

## Why

The plan trains four encoders (two IP variants, three seeds) on the same
captures. Everything the encoder's setup builds from them -- the loaded
columns, the six splits, both neighbour finders, the negative samplers, the
time statistics -- is identical across all four: no step of it depends on
the seed or on the IP ablation (the split draws with its own fixed seed).
Built per run, that was a ~20 GB out-of-core store and ~7 minutes of setup
per encoder, so four encoders side by side needed ~80 GB of disk and four
copies in page cache. Shared, it is one store, one page cache, and a resumed
run skips setup entirely.

## What is shared, and how it stays exact

`save` pickles the setup's outputs with a persistent_id hook: every array
whose memory lives in one of the store's files is written as a reference
(file, dtype, shape, strides, offset), not as data, and `load` maps those
files back READ-ONLY -- anything that tried to write into shared data would
fail loudly rather than corrupt another run. Small objects (category map,
scalars) pickle normally. numpy's global RNG state at the end of setup is
saved and restored, because training draws from it after the split.

Node features are the one input that differs between runs: the IP ablation
masks them. They are stored UNMASKED and each run applies its own mask with
the same float32 multiply the loader uses, so the result is bit-identical.

## Key

sha256 over: the ingest cache keys of every capture (which already cover the
input files, parse parameters and parse code), the loader and split
arguments, and the content of the setup code (bita/train.py, bita/utils/
utils.py, data_unification/ip_features.py). Anything that could change the
setup changes the key; a stale store is never read. The first run to need a
key builds it under a file lock; the others wait on the lock and load.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_FORMAT = 1
_SETUP_CODE = ("bita/train.py", "bita/utils/utils.py", "data_unification/ip_features.py",
               "bita/utils/setup_store.py")


def setup_key(parts: dict, repo: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"format": _FORMAT, **parts}, sort_keys=True, default=str).encode())
    for rel in _SETUP_CODE:
        with open(os.path.join(repo, rel), "rb") as f:
            h.update(rel.encode())
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:24]


@contextmanager
def locked(root: str, key: str):
    """Exclusive lock for one key: the first run builds, the rest wait."""
    Path(root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(root, f"{key}.lock"), "w") as fh:
        t = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        if time.time() - t > 1:
            logger.info("Shared setup: waited %.0fs for %s", time.time() - t, key)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _memmap_of(arr):
    base = arr
    while base is not None and not isinstance(base, np.memmap):
        base = getattr(base, "base", None)
    return base


class _Pickler(pickle.Pickler):
    def __init__(self, f, store_root):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self._root = os.path.realpath(store_root)

    def persistent_id(self, obj):
        if not isinstance(obj, np.ndarray) or obj.size == 0:
            return None
        mm = _memmap_of(obj)
        fn = getattr(mm, "filename", None)
        if not fn or os.path.dirname(os.path.realpath(fn)) != self._root:
            return None
        start = mm.__array_interface__["data"][0] - int(getattr(mm, "offset", 0) or 0)
        offset = obj.__array_interface__["data"][0] - start
        return ("store", os.path.basename(fn), obj.dtype.str, obj.shape, obj.strides, int(offset),
                isinstance(obj, np.memmap))


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, store_root):
        super().__init__(f)
        self._root = store_root
        self._maps = {}

    def persistent_load(self, pid):
        tag, name, dtype, shape, strides, offset, was_memmap = pid
        assert tag == "store", pid
        path = os.path.join(self._root, name)
        dt = np.dtype(dtype)
        if was_memmap and tuple(strides) == np.empty(shape, dtype=dt).strides:
            # Consumers detect a memmap by type (HostEdgeFeatures): return one.
            return np.memmap(path, dtype=dt, mode="r", shape=tuple(shape), offset=offset)
        mm = self._maps.get(name)
        if mm is None:
            mm = self._maps[name] = np.memmap(path, dtype=np.uint8, mode="r")
        # A read-only view straight onto the file: no copy, and no writes.
        return np.ndarray(tuple(shape), dtype=dt, buffer=mm, offset=offset, strides=tuple(strides))


def save(root: str, key: str, store_root: str, outputs: dict) -> None:
    """Pickle `outputs` next to the store; meta.json written last = committed."""
    d = Path(root) / key
    buf = io.BytesIO()
    _Pickler(buf, store_root).dump({"outputs": outputs, "np_random_state": np.random.get_state()})
    (d / "setup.pkl").write_bytes(buf.getvalue())
    (d / "meta.json").write_text(json.dumps({"format": _FORMAT, "time": time.time(),
                                             "pickle_bytes": len(buf.getvalue())}))


def load(root: str, key: str):
    """(outputs, store_dir) or None when no committed setup exists."""
    d = Path(root) / key
    if not (d / "meta.json").exists():
        return None
    with open(d / "setup.pkl", "rb") as f:
        blob = _Unpickler(f, str(d / "store")).load()
    np.random.set_state(blob["np_random_state"])
    return blob["outputs"], str(d / "store")


def prune(root: str, keep: str) -> None:
    """Remove other keys' stores (they can never be read again) unless locked."""
    for p in Path(root).iterdir() if Path(root).exists() else ():
        if p.is_dir() and p.name != keep:
            lock = Path(root) / f"{p.name}.lock"
            try:
                with open(lock, "w") as fh:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass   # in use by another run
