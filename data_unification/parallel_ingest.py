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


# ------------------------------------------------------------------ PCAP parser
#
# PCAP2018 days are parsed by default by rust/pcap_fast, a bit-exact port of
# the Python packet -> flow -> host-window path (rust/pcap_fast/EXPERIMENT.md),
# ~12x faster. Its columns are built with the same numpy code the reference
# uses, and every Rust-parsed day is checked against the Python reference on a
# sample (its first host file plus its largest, _PARITY_FILES in all) before it is used.

#: "rust" (default) or "python". Anything else is an error.
PCAP_PARSER_ENV = "CYBERWORLD_PCAP_PARSER"
#: Host files per day re-parsed through the Python reference as a parity
#: check; CYBERWORLD_PCAP_PARITY_FILES overrides, 0 disables it.
_PARITY_FILES = 2
#: Largest host file the parity sample will pick (bytes).
_PARITY_MAX_BYTES = int(1.5 * 2**30)

_REPO = Path(__file__).resolve().parents[1]
_PCAP_FAST_DIR = _REPO / "rust" / "pcap_fast"
#: Non-.rs files the Rust path's output depends on; hashed into the cache key.
_PCAP_FAST_CODE = ("Cargo.toml", "Cargo.lock", "pcap_fast_py.py")

#: The Rust side reproduces numpy's float reductions (pairwise summation fed
#: in 8192-element buffer blocks) to the bit, and the edge features stay in
#: numpy (its SVML log1p). Parity was verified on exactly this numpy.
_PARITY_NUMPY_VERSION = "1.26.4"
_PARITY_NUMPY_BUFSIZE = 8192


class PcapParityError(RuntimeError):
    """The Rust parse of a day disagrees with the Python reference, or cannot
    be trusted in this environment. Fatal: the run stops rather than train
    without (or on a wrong copy of) the day."""


def _pcap_fast_sources() -> List[Path]:
    return sorted((_PCAP_FAST_DIR / "src").glob("*.rs")) + [
        _PCAP_FAST_DIR / f for f in _PCAP_FAST_CODE]


def pcap_fast_binary() -> Path:
    return Path(os.environ.get("PCAP_FAST_BIN",
                               str(_PCAP_FAST_DIR / "target" / "release" / "pcap_fast")))


def check_numpy_for_rust_parity() -> None:
    if np.__version__ != _PARITY_NUMPY_VERSION or np.getbufsize() != _PARITY_NUMPY_BUFSIZE:
        raise PcapParityError(
            f"the Rust PCAP parser's float parity with the Python reference was verified on "
            f"numpy {_PARITY_NUMPY_VERSION} (bufsize {_PARITY_NUMPY_BUFSIZE}); this is numpy "
            f"{np.__version__} (bufsize {np.getbufsize()}). numpy's pairwise-summation "
            f"internals decide the last bits of the packet features, so another version can "
            f"silently change them. Install numpy=={_PARITY_NUMPY_VERSION}, or set "
            f"{PCAP_PARSER_ENV}=python, or re-verify (rust/pcap_fast/verify.py) and update "
            f"_PARITY_NUMPY_VERSION in data_unification/parallel_ingest.py.")


def pcap_parser_choice() -> str:
    """Which parser PCAP2018 days use in this run: "rust" or "python".

    Rust unless CYBERWORLD_PCAP_PARSER=python, or the binary is missing or
    older than its sources (warned: the run is then ~12x slower, not wrong).
    Raises PcapParityError if numpy is not the version parity was verified on."""
    want = os.environ.get(PCAP_PARSER_ENV, "rust").strip().lower()
    if want not in ("rust", "python"):
        raise ValueError(f"{PCAP_PARSER_ENV}={want!r}: expected 'rust' or 'python'")
    if want == "python":
        return "python"
    b = pcap_fast_binary()
    if not b.is_file() or not os.access(b, os.X_OK):
        logger.warning("Rust PCAP parser %s not built; falling back to the Python parser "
                       "(~12x slower). Build it: cd rust/pcap_fast && cargo build --release", b)
        return "python"
    if "PCAP_FAST_BIN" not in os.environ:
        newest = max(f.stat().st_mtime for f in _pcap_fast_sources()
                     if f.suffix == ".rs" or f.name in ("Cargo.toml", "Cargo.lock"))
        if b.stat().st_mtime < newest:
            logger.warning("Rust PCAP parser %s is older than its sources; falling back to the "
                           "Python parser. Rebuild: cd rust/pcap_fast && cargo build --release", b)
            return "python"
    check_numpy_for_rust_parity()
    return "rust"


def _pcap_fast_module():
    import importlib.util
    import sys
    mod = sys.modules.get("pcap_fast_py")
    if mod is None:
        spec = importlib.util.spec_from_file_location("pcap_fast_py",
                                                      _PCAP_FAST_DIR / "pcap_fast_py.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pcap_fast_py"] = mod
        spec.loader.exec_module(mod)
    return mod


class _PartWriter:
    """Writes columns as .npy parts of exactly `part_records` rows (the last
    one shorter), however the rows arrive."""

    def __init__(self, base: str, part_records: int, edge_dim: int):
        self.base, self.part_records, self.edge_dim = base, part_records, edge_dim
        self.parts: List[str] = []
        self._buf: List[tuple] = []
        self._n = 0

    def save(self, u, i, ts, lbl, edge):
        stem = f"{self.base}.p{len(self.parts):04d}"
        np.save(f"{stem}.u.npy", np.asarray(u, dtype=np.int32))
        np.save(f"{stem}.i.npy", np.asarray(i, dtype=np.int32))
        np.save(f"{stem}.ts.npy", np.asarray(ts, dtype=np.float64))
        np.save(f"{stem}.lbl.npy", np.asarray(lbl, dtype=np.int32))
        np.save(f"{stem}.edge.npy", np.asarray(edge, dtype=np.float32).reshape(-1, self.edge_dim))
        self.parts.append(stem)

    def add_arrays(self, cols: tuple):
        """Columnar rows (u, i, ts, lbl, edge); saved at part boundaries."""
        self._buf.append(cols)
        self._n += len(cols[0])
        while self._n >= self.part_records:
            cat = [np.concatenate([c[k] for c in self._buf]) for k in range(5)]
            self.save(*(c[:self.part_records] for c in cat))
            self._n -= self.part_records
            self._buf = [tuple(c[self.part_records:] for c in cat)] if self._n else []

    def close(self):
        if self._n:
            self.save(*(np.concatenate([c[k] for c in self._buf]) for k in range(5)))
            self._buf, self._n = [], 0


def _python_columns(stream, base, part_records, edge_dim, stride=1, max_rows=None):
    """The reference loop: UnifiedFlowRecords -> local-vocabulary columns in
    parts. Returns (kept, source, ips, cats, parts)."""
    from data_unification.tgne_features import extract_canonical_edge_features
    w = _PartWriter(base, part_records, edge_dim)
    ip_local: Dict[str, int] = {}
    cat_local: Dict[str, int] = {}
    u, i, ts, lbl = [], [], [], []
    edges = []
    source = None

    def _flush():
        w.save(u, i, ts, lbl, edges)
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
    if u:
        _flush()

    # Vocabularies in local-id order, so index == local id.
    ips = [None] * len(ip_local)
    for s, k in ip_local.items():
        ips[k] = s
    cats = [None] * len(cat_local)
    for s, k in cat_local.items():
        cats[k] = s
    return kept, source, ips, cats, w.parts


def _rust_columns(fast, day_dir, dw, window_seconds, base, part_records, edge_dim):
    """The same (kept, source, ips, cats, parts) from the Rust binary."""
    w = _PartWriter(base, part_records, edge_dim)
    b, chunks = fast.iter_day_column_chunks(day_dir, dw, window_seconds,
                                            binary=str(pcap_fast_binary()))
    for cols in chunks:
        w.add_arrays(cols)
    w.close()
    return b.n, (b.source if b.n else None), b.ips, b.cats, w.parts


def _load_parts(parts) -> Dict[str, np.ndarray]:
    return {c: np.concatenate([np.load(f"{s}.{c}.npy") for s in parts]) for c in _COLS}


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view({8: np.uint64, 4: np.uint32}[a.dtype.itemsize]) if a.dtype.kind == "f" else a


def _compare_parity(day: str, files: List[str], ref: tuple, got: tuple) -> None:
    """ref/got = (n, source, ips, cats, parts); PcapParityError on any difference."""
    bad = []
    for name, a, b in zip(("n", "source", "ips", "cats"), ref[:4], got[:4]):
        if a != b:
            bad.append(f"{name}: python {str(a)[:200]} != rust {str(b)[:200]}")
    if not bad and ref[0]:
        ca, cb = _load_parts(ref[4]), _load_parts(got[4])
        for c in _COLS:
            a, b = ca[c], cb[c]
            if a.dtype != b.dtype or a.shape != b.shape:
                bad.append(f"{c}: dtype/shape python {a.dtype}{a.shape} != rust {b.dtype}{b.shape}")
            elif not np.array_equal(_bits(a), _bits(b)):
                rows = (_bits(a) != _bits(b)).reshape(len(a), -1).any(axis=1)
                bad.append(f"{c}: {int(rows.sum())} row(s) differ, first at "
                           f"{int(np.argmax(rows))}")
    if bad:
        raise PcapParityError(
            f"Rust PCAP parser disagrees with the Python reference on {day} (parity sample "
            f"{files}): " + "; ".join(bad) + f". The day is NOT used. Set {PCAP_PARSER_ENV}="
            f"python to train on the reference parser; investigate with rust/pcap_fast/verify.py.")


def _pcap_parity_sample(fast, day_dir: Path, dw, day: str, window_seconds: float,
                        edge_dim: int, workdir: Path, n_files: int) -> dict:
    """Parse `n_files` of the day's host files (the first and the largest) with BOTH parsers and demand
    bit-identical columns. The reference runs unmodified on a mini-day: a
    directory of the same name holding symlinks to those files, labelled from
    the day's own (already derived) attack windows."""
    from data_unification.pcap_bridge import iter_day_records
    # The first host file plus the LARGEST ones: the first files of a day are
    # 1-17 MB (<0.03% of it) and exercise little; the largest carry the dense
    # traffic (floods, many flows per window) where a port is likeliest to
    # diverge. Kept in the day's own file order, which the merge depends on.
    # Capped at _PARITY_MAX_BYTES each: the largest files run to 18 GB, which
    # the ~20-60 MB/s reference needs up to 15 min for, longer than Rust takes
    # for the whole day.
    every = fast._captures(day_dir)[0]
    chosen = set(every[:1])
    for f in sorted((f for f in every if f.stat().st_size <= _PARITY_MAX_BYTES),
                    key=lambda f: (-f.stat().st_size, f.name)):
        if len(chosen) >= n_files:
            break
        chosen.add(f)
    files = [f for f in every if f in chosen]
    root = Path(workdir) / f".parity-{Path(day_dir).name}-{os.getpid()}"
    shutil.rmtree(root, ignore_errors=True)
    mini = root / Path(day_dir).name / "pcap"
    mini.mkdir(parents=True)
    try:
        for f in files:
            os.symlink(os.path.realpath(f), mini / f.name)
        # iter_pcap_day_windows minus its derive_windows (a per-day constant
        # of up to ~45 s, reused here rather than recomputed).
        stream = (r for _s, _e, recs in iter_day_records(
            mini.parent, dw, scenario_id=day, window_seconds=window_seconds) for r in recs)
        ref = _python_columns(stream, str(root / "ref"), _PART_RECORDS, edge_dim)
        got = _rust_columns(fast, mini.parent, dw, window_seconds, str(root / "rust"),
                            _PART_RECORDS, edge_dim)
        _compare_parity(Path(day_dir).name, [f.name for f in files], ref, got)
        return {"files": [f.name for f in files], "records": ref[0]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _parse_one(args) -> Optional[dict]:
    """Worker: parse one capture into columns with a LOCAL string vocabulary.

    args = (kind, path, outdir, stride, max_rows, edge_dim, pcap_label_dir,
            window_seconds, part_records[, pcap_parser]). pcap_parser ("rust" /
    "python") defaults to pcap_parser_choice(). The result dict has the same
    content whichever parser ran.
    """
    (kind, path, outdir, stride, max_rows, edge_dim, pcap_label_dir, window_seconds,
     part_records) = args[:9]
    pcap_parser = args[9] if len(args) > 9 else None
    try:
        from data_unification.cic2017_adapter import CIC2017Adapter
        from data_unification.cic2018_adapter import CIC2018Adapter
        from data_unification.ctu13_adapter import CTU13Adapter
        from data_unification.label_resolver import get_default_resolver

        # Worker processes are reused across captures, so the resolver's
        # counters are cumulative here: report this capture's delta only.
        cov0 = get_default_resolver().unresolved_report()
        base = str(Path(outdir) / (Path(path).name.replace("/", "_") + f".{os.getpid()}"))

        rust = False
        if kind == "PCAP2018":
            if pcap_parser is None:
                pcap_parser = pcap_parser_choice()
            # Row sampling changes how far the reference reads (and so its
            # resolver counters); those runs stay on the reference.
            rust = pcap_parser == "rust" and stride <= 1 and max_rows is None
        labels = None
        if rust:
            check_numpy_for_rust_parity()
            fast = _pcap_fast_module()
            labels = fast.load_day_labels(path, pcap_label_dir)
            if labels is None:          # the reference skips the day too
                kept, source, ips, cats, parts = 0, None, [], [], []
            else:
                kept, source, ips, cats, parts = _rust_columns(
                    fast, Path(path), labels[0], window_seconds, base, part_records, edge_dim)
        else:
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
            kept, source, ips, cats, parts = _python_columns(
                stream, base, part_records, edge_dim, stride, max_rows)

        cov1 = get_default_resolver().unresolved_report()
        cov = {"resolve_calls": cov1["resolve_calls"] - cov0["resolve_calls"],
               "unresolved_calls": cov1["unresolved_calls"] - cov0["unresolved_calls"],
               "labels": sorted(set(cov1["distinct_unresolved_labels"])
                                - set(cov0["distinct_unresolved_labels"]))}

        if labels is not None:
            # After the coverage delta: the sample's resolve() calls are not
            # this capture's.
            n_files = int(os.environ.get("CYBERWORLD_PCAP_PARITY_FILES", _PARITY_FILES))
            if n_files > 0:
                _pcap_parity_sample(fast, Path(path), labels[0], labels[1], window_seconds,
                                    edge_dim, Path(outdir), n_files)

        if not kept:
            return {"path": path, "kind": kind, "n": 0, "coverage": cov}
        return {"path": path, "kind": kind, "n": kept, "parts": parts,
                "source": source, "ips": ips, "cats": cats, "coverage": cov}
    except Exception as exc:
        # Return the error rather than only logging it: this runs in a CHILD
        # process, whose log handlers are not the parent's, so a message
        # emitted here can vanish entirely. The parent re-raises it into its
        # own log, where a failed capture is actually visible. A parity
        # failure is marked fatal: the parent raises instead of going on.
        out = {"path": path, "kind": kind, "n": 0, "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, PcapParityError):
            out["fatal"] = True
        return out


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
    # imported lazily by the day loaders, so never loaded by the imports
    # above: without it here, a change to how participants scope labels
    # would not invalidate cached parses
    "data_unification.attack_participants",
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
    # Absolute, existing files only: some extension modules carry a bare
    # relative __file__ (torch._classes has "_classes.py"), which realpath
    # would resolve against the cwd -- i.e. into the repo -- as a phantom file.
    files = sorted({os.path.realpath(f) for f in
                    (getattr(mod, "__file__", None) for mod in list(sys.modules.values()))
                    if f and os.path.isabs(f) and os.path.isfile(f)
                    and os.path.realpath(f).startswith(repo + os.sep)})
    # The Rust PCAP parser is not a Python module; its sources and the glue
    # module (loaded by path) are hashed explicitly.
    files += [os.path.realpath(f) for f in _pcap_fast_sources()]
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, repo).encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest(), [os.path.relpath(f, repo) for f in files]


def parse_code_hash() -> Tuple[str, List[str]]:
    """(hash, files) of the parsing code, measured in a FRESH interpreter.

    Not a multiprocessing spawn: a spawned child re-imports the caller's main
    script, so called from bita/train.py the hash covered train.py and all it
    imports, and never matched the key the warm stage wrote (the encoder would
    silently re-parse everything). `python -c` imports nothing but the parse
    chain, whoever calls it; tests/test_ingest_cache.py pins that.
    """
    import subprocess
    import sys
    repo = os.path.realpath(str(Path(__file__).resolve().parents[1]))
    code = ("import json, sys; sys.path.insert(0, %r); "
            "from data_unification.parallel_ingest import _code_hash_worker as w; "
            "print(json.dumps(w(%r)))" % (repo, repo))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                         capture_output=True, text=True, check=True).stdout
    h, files = json.loads(out.strip().splitlines()[-1])
    return h, files


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


def effective_parser(kind, pcap_parser, stride, max_rows) -> Optional[str]:
    """The parser that will actually produce `kind`'s columns (None: not a PCAP day)."""
    if kind != "PCAP2018":
        return None
    return "rust" if pcap_parser == "rust" and stride <= 1 and max_rows is None else "python"


def cache_key(kind, path, *, stride, max_rows, edge_dim, pcap_label_dir, window_seconds,
              code_hash, pcap_parser=None) -> str:
    """`pcap_parser` is the run's choice ("rust"/"python"); the parser that
    actually runs is part of the key, so the two never share an entry."""
    import hashlib
    extra = {}
    parser = effective_parser(kind, pcap_parser or "python", stride, max_rows)
    if parser is not None:
        extra["parser"] = parser
    if kind == "CTU13":
        from data_unification.label_resolver import ctu_background_policy
        extra["ctu_background"] = ctu_background_policy()
    if kind == "PCAP2018":
        # PCAP labels also depend on the unscoped-interval policy and on the
        # participant map (attack_windows / attack_participants): a cached day
        # labelled under one must never be read under the other.
        from data_unification.attack_participants import DEFAULT_MAP_PATH
        from data_unification.attack_windows import unscoped_label_policy
        _m = Path(DEFAULT_MAP_PATH)
        extra["labels"] = {
            "unscoped": unscoped_label_policy(),
            "participant_map": (hashlib.sha256(_m.read_bytes()).hexdigest()[:16]
                                if _m.exists() else None),
        }
    blob = json.dumps({**extra,
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

    # Decided ONCE, here, so the cache key and every worker agree on it.
    pcap_parser = (pcap_parser_choice() if any(k == "PCAP2018" for k, _p in captures)
                   else None)
    if pcap_parser is not None:
        logger.info("PCAP parser: %s%s", pcap_parser,
                    f" ({pcap_fast_binary()})" if pcap_parser == "rust" else "")

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
                code_hash=code_hash, pcap_parser=pcap_parser)
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
                                              pcap_label_dir, window_seconds, _PART_RECORDS,
                                              pcap_parser))
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
                if res.get("fatal"):
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise PcapParityError(f"{res['path']}: {res['error']}")
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
