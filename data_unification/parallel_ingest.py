"""Parse captures in parallel, with a bit-identical result to serial parsing.

## Why this is safe

Loading 34M records took 12.7 minutes on **one** core while the other fifteen
sat idle. Captures are independent files, so parsing is embarrassingly
parallel -- but naive parallelism would silently change the model.

Node ids are assigned in *encounter order* (`ip_to_id[ip] = len(ip_to_id)+1`).
If workers finish out of order, the same host gets a different id, which
changes the node-feature row it indexes, the negative sampler's draws and the
inductive node choice. The graph would be isomorphic but the run would not be
reproducible, and it could not be compared against a serial baseline.

So workers never assign global ids. Each returns its capture's columns with a
**local** vocabulary, and the parent merges them **in a fixed capture order**
-- the same order serial parsing would have visited. Encounter order, and
therefore every id, is preserved exactly.

Results move through `.npy` files in a scratch directory rather than pickled
over a pipe: the payload is ~2.6 GB and pickling it would cost more than the
parsing it saves.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Records per on-disk part. A PCAP day is tens of millions of records; held
# as Python lists in the worker that is several GB per worker, times one
# worker per day. Flushing compact numpy parts keeps a worker's footprint to
# one part (500k records, ~300 MB as Python lists at ~600 B/record measured)
# regardless of the capture's size.
_PART_RECORDS = 500_000
_COLS = ("u", "i", "ts", "lbl", "edge")


def _parse_one(args) -> Optional[dict]:
    """Worker: parse one capture into columns with a LOCAL string vocabulary."""
    (kind, path, outdir, stride, max_rows, edge_dim, pcap_label_dir, window_seconds,
     part_records) = args
    try:
        from data_unification.cic2017_adapter import CIC2017Adapter
        from data_unification.cic2018_adapter import CIC2018Adapter
        from data_unification.ctu13_adapter import CTU13Adapter
        from data_unification.label_resolver import get_default_resolver
        from data_unification.tgne_features import extract_canonical_edge_features

        # Worker processes are reused across captures, so the resolver's
        # counters are cumulative here: report this capture's delta only.
        cov0 = get_default_resolver().unresolved_report()

        if kind == "CIC2017":
            stream = CIC2017Adapter().parse_file(path, max_rows=None)
        elif kind == "CIC2018":
            stream = CIC2018Adapter().parse_file(path, max_rows=None)
        elif kind == "PCAP2018":
            # Exactly the serial path's record stream (bita/train.py).
            from data_unification.training_sources import iter_pcap_day_windows
            stream = (r for window in iter_pcap_day_windows(
                path, pcap_label_dir, window_seconds) for r in window)
        else:
            stream = CTU13Adapter().parse_netflow_csv(path, max_rows=None)

        ip_local: Dict[str, int] = {}
        cat_local: Dict[str, int] = {}
        u, i, ts, lbl = [], [], [], []
        edges = []
        source = None
        parts: List[str] = []
        base = Path(outdir) / (Path(path).name.replace("/", "_") + f".{os.getpid()}")

        def _flush():
            stem = f"{base}.p{len(parts):04d}"
            np.save(f"{stem}.u.npy", np.asarray(u, dtype=np.int32))
            np.save(f"{stem}.i.npy", np.asarray(i, dtype=np.int32))
            np.save(f"{stem}.ts.npy", np.asarray(ts, dtype=np.float64))
            np.save(f"{stem}.lbl.npy", np.asarray(lbl, dtype=np.int32))
            np.save(f"{stem}.edge.npy", np.asarray(edges, dtype=np.float32).reshape(-1, edge_dim))
            parts.append(stem)
            for col in (u, i, ts, lbl, edges):
                col.clear()

        kept = 0
        for n, r in enumerate(stream):
            if stride > 1 and (n % stride):
                continue
            # The serial path keys the corpus by each record's own
            # raw_label_source; one capture must carry exactly one.
            if source is None:
                source = r.raw_label_source
            elif r.raw_label_source != source:
                raise ValueError(f"mixed raw_label_source in one capture: "
                                 f"{source!r} and {r.raw_label_source!r}")
            su = ip_local.get(r.src_ip)
            if su is None:
                su = ip_local[r.src_ip] = len(ip_local)
            di = ip_local.get(r.dst_ip)
            if di is None:
                di = ip_local[r.dst_ip] = len(ip_local)
            ci = cat_local.get(r.coarse_category)
            if ci is None:
                ci = cat_local[r.coarse_category] = len(cat_local)
            u.append(su); i.append(di); ts.append(r.start_time); lbl.append(ci)
            edges.append(extract_canonical_edge_features(
                fwd_bytes=r.fwd_bytes, bwd_bytes=r.bwd_bytes,
                fwd_packets=r.fwd_packets, bwd_packets=r.bwd_packets,
                duration_sec=r.duration, byte_rate=r.byte_rate,
                packet_rate=r.packet_rate, protocol=r.protocol,
                dst_port=r.dst_port))
            kept += 1
            if len(u) >= part_records:
                _flush()
            if max_rows is not None and kept >= max_rows:
                break

        cov1 = get_default_resolver().unresolved_report()
        cov = {"resolve_calls": cov1["resolve_calls"] - cov0["resolve_calls"],
               "unresolved_calls": cov1["unresolved_calls"] - cov0["unresolved_calls"],
               "labels": sorted(set(cov1["distinct_unresolved_labels"])
                                - set(cov0["distinct_unresolved_labels"]))}

        if not kept:
            return {"path": path, "kind": kind, "n": 0, "coverage": cov}
        if u:
            _flush()

        # Vocabularies in local-id order, so index == local id.
        ips = [None] * len(ip_local)
        for s, k in ip_local.items():
            ips[k] = s
        cats = [None] * len(cat_local)
        for s, k in cat_local.items():
            cats[k] = s
        return {"path": path, "kind": kind, "n": kept, "parts": parts,
                "source": source, "ips": ips, "cats": cats, "coverage": cov}
    except Exception as exc:
        # Return the error rather than only logging it: this runs in a CHILD
        # process, whose log handlers are not the parent's, so a message
        # emitted here can vanish entirely. The parent re-raises it into its
        # own log, where a failed capture is actually visible.
        return {"path": path, "kind": kind, "n": 0,
                "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ cache
#
# Every encoder run re-parsed the same ~482 GB of PCAPs (~1 h even in
# parallel): two IP variants x three seeds re-read identical bytes. A parsed
# capture is a pure function of (its input files, the parsing code, the
# parse parameters), so it is cached on disk under a key built from all three.
# Any change to a capture file (name, size, mtime), its label CSV, a
# parameter, or ANY repo module the parse imports gives a new key: a stale
# entry is never read, only left behind.

#: Entry points of the parse. Modules imported lazily inside functions are
#: listed explicitly so the code hash sees them too.
_PARSE_ENTRY_MODULES = (
    "data_unification.parallel_ingest",
    "data_unification.cic2017_adapter",
    "data_unification.cic2018_adapter",
    "data_unification.ctu13_adapter",
    "data_unification.tgne_features",
    "data_unification.label_resolver",
    "data_unification.training_sources",
    "data_unification.attack_windows",
    "data_unification.pcap_bridge",
    "data_unification.pcap_adapter",
    "telemetry.capture.sniffer",
    "telemetry.flow.flow_table",
    "telemetry.packet.pcap_engine",
)

#: Bump to invalidate every entry when the cache FORMAT changes.
_CACHE_FORMAT = 1


def _code_hash_worker(repo: str) -> Tuple[str, List[str]]:
    """Runs in a FRESH interpreter: import the parse chain, hash every repo
    module that ended up loaded (transitively), by content."""
    import hashlib
    import importlib
    import sys
    for m in _PARSE_ENTRY_MODULES:
        importlib.import_module(m)
    files = sorted({os.path.realpath(mod.__file__) for mod in list(sys.modules.values())
                    if getattr(mod, "__file__", None)
                    and os.path.realpath(mod.__file__).startswith(repo + os.sep)})
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, repo).encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest(), [os.path.relpath(f, repo) for f in files]


def parse_code_hash() -> Tuple[str, List[str]]:
    """(hash, files) of the parsing code, measured in a clean spawn process so
    whatever the caller happens to have imported does not leak in."""
    repo = os.path.realpath(str(Path(__file__).resolve().parents[1]))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(_code_hash_worker, repo).result()


def _input_fingerprint(kind: str, path: str, pcap_label_dir: Optional[str]) -> list:
    """(relative name, size, mtime_ns) of every file the capture's parse reads."""
    def stat(f: Path, rel: str):
        st = f.stat()
        return [rel, st.st_size, st.st_mtime_ns]
    p = Path(path)
    if kind == "PCAP2018":
        out = [stat(f, str(f.relative_to(p))) for f in sorted(p.rglob("*")) if f.is_file()]
        from data_unification.training_sources import _pcap_label_csv
        csv_path = _pcap_label_csv(p, Path(pcap_label_dir)) if pcap_label_dir else None
        out.append(stat(csv_path, "LABELS:" + csv_path.name) if csv_path else ["LABELS:", None, None])
        return out
    return [stat(p, p.name)]


def cache_key(kind, path, *, stride, max_rows, edge_dim, pcap_label_dir, window_seconds,
              code_hash) -> str:
    import hashlib
    blob = json.dumps({
        "format": _CACHE_FORMAT, "kind": kind, "name": Path(path).name,
        "inputs": _input_fingerprint(kind, path, pcap_label_dir),
        "stride": stride, "max_rows": max_rows, "edge_dim": edge_dim,
        "window_seconds": float(window_seconds), "code": code_hash,
    }, sort_keys=True)
    return f"{Path(path).name}-{hashlib.sha256(blob.encode()).hexdigest()[:24]}"


def _cache_read(entry: Path, path: str) -> Optional[dict]:
    """A committed entry's result dict, or None. meta.json is written last,
    so its presence is the commit marker."""
    try:
        with open(entry / "meta.json") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    for stem in meta.get("parts", ()):
        if not all((entry / f"{stem}.{c}.npy").exists() for c in _COLS):
            return None
    meta["path"] = path              # the caller's spelling, not the cached one
    meta["parts"] = [str(entry / stem) for stem in meta.get("parts", ())]
    meta["cached"] = True
    return meta


def _cache_commit(res: dict, tmpdir: Path, entry: Path) -> dict:
    """Move a finished parse into the cache atomically (rename on one fs)."""
    meta = {k: v for k, v in res.items() if k not in ("parts", "path", "cached")}
    meta["parts"] = [Path(s).name for s in res.get("parts", ())]
    with open(tmpdir / "meta.json", "w") as f:
        json.dump(meta, f)
    if entry.exists() and _cache_read(entry, res["path"]) is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)    # another run committed it first
    else:
        # Absent, or a damaged leftover (no meta.json, a part missing): replace it.
        shutil.rmtree(entry, ignore_errors=True)
        os.rename(tmpdir, entry)
    out = _cache_read(entry, res["path"])
    if out is None:
        raise RuntimeError(f"ingest cache entry {entry} failed to read back")
    return out


def parse_captures_parallel(
    captures: List[Tuple[str, str]],
    *,
    stride: int = 1,
    max_rows_per_file: Optional[int] = None,
    edge_dim: int = 12,
    workers: Optional[int] = None,
    scratch: Optional[str] = None,
    on_capture: Optional[Callable[[dict], None]] = None,
    pcap_label_dir: Optional[str] = None,
    window_seconds: float = 2.0,
    cache_dir: Optional[str] = None,
):
    """Parse `captures` = [(kind, path)] in parallel; yield results in INPUT order.

    Yields one dict per capture, in exactly the order given, so the caller's
    encounter-order id assignment is unchanged from serial. The columns are
    not loaded: `iter_parts(res)` yields them part by part, so the parent
    never holds a whole PCAP day twice.

    `scratch` should be on disk for PCAP days (their parts total GBs); the
    default is the system temp dir, which is tmpfs -- i.e. RAM -- on Fedora.

    With `cache_dir`, captures already parsed with identical inputs, code and
    parameters are served from disk, and newly parsed ones are committed there.
    """
    if workers is None:
        # Leave headroom: each worker holds one capture's columns, and the
        # largest single capture (tue_20, 7.9M rows) is ~600 MB.
        workers = max(1, min(6, (os.cpu_count() or 4) - 2))

    tmp = scratch or tempfile.mkdtemp(prefix="tgne_ingest_")
    Path(tmp).mkdir(parents=True, exist_ok=True)

    entries: List[Optional[Path]] = [None] * len(captures)
    hits: Dict[int, dict] = {}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        code_hash, code_files = parse_code_hash()
        logger.info("Ingest cache %s: parse code hash %s over %d module(s)",
                    cache_dir, code_hash[:12], len(code_files))
        for n, (k, p) in enumerate(captures):
            entries[n] = Path(cache_dir) / cache_key(
                k, p, stride=stride, max_rows=max_rows_per_file, edge_dim=edge_dim,
                pcap_label_dir=pcap_label_dir, window_seconds=window_seconds,
                code_hash=code_hash)
            got = _cache_read(entries[n], p)
            if got is not None:
                hits[n] = got
        logger.info("Ingest cache: %d of %d capture(s) cached", len(hits), len(captures))

    def _outdir(n):
        if entries[n] is None:
            return tmp
        d = Path(cache_dir) / f".tmp-{entries[n].name}-{os.getpid()}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        return str(d)

    misses = [n for n in range(len(captures)) if n not in hits]
    if misses:
        logger.info("Parsing %d captures across %d worker processes", len(misses), workers)
    try:
        # spawn, not fork: the trainer has torch (and its thread pools) live
        # by now, and forking a multi-threaded process can deadlock the child.
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(misses) or 1)),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {n: ex.submit(_parse_one, (captures[n][0], captures[n][1], _outdir(n),
                                              stride, max_rows_per_file, edge_dim,
                                              pcap_label_dir, window_seconds, _PART_RECORDS))
                    for n in misses}
            # Consumed in INPUT order, which is what keeps ids deterministic.
            for n in range(len(captures)):
                if n in hits:
                    res = hits[n]
                else:
                    res = futs.pop(n).result()
                    if entries[n] is not None:
                        tmpdir = Path(cache_dir) / f".tmp-{entries[n].name}-{os.getpid()}"
                        if res.get("error"):
                            shutil.rmtree(tmpdir, ignore_errors=True)
                        else:
                            res = _cache_commit(res, tmpdir, entries[n])
                if res.get("error"):
                    # Surfaced in the PARENT's log; a worker's own handlers
                    # are not connected to it.
                    logger.error("parallel parse failed on %s: %s",
                                 res["path"], res["error"])
                elif res.get("n", 0) == 0:
                    logger.warning("parallel parse produced 0 records for %s",
                                   res["path"])
                if on_capture:
                    on_capture(res)
                if res.get("n", 0) == 0:
                    continue
                yield res
    finally:
        if scratch is None:
            shutil.rmtree(tmp, ignore_errors=True)
        if cache_dir is not None:
            for d in Path(cache_dir).glob(f".tmp-*-{os.getpid()}"):
                shutil.rmtree(d, ignore_errors=True)


def iter_parts(res: dict):
    """Yield one capture's columns part by part. Scratch parts are deleted once
    loaded; cached parts are left in place for the next run."""
    for stem in res.get("parts", ()):
        part = {col: np.load(f"{stem}.{col}.npy") for col in _COLS}
        if not res.get("cached"):
            for col in _COLS:
                try:
                    os.unlink(f"{stem}.{col}.npy")
                except OSError:
                    pass
        yield part
