"""One capture as compact columns on disk, for Branch A / Branch B / DeepOP extraction.

## Why

`training_sources.read_capture` returns a capture as a list of
UnifiedFlowRecord objects, ~518 B each. A CIC-2018 PCAP day is ~12M records,
so ~6 GB of Python objects per day, and parsing it takes ~30 min on one core.
The trainers read every capture once per run, and the plan runs Branch A five
times (two IP arms, two extra seeds, downstream), so each PCAP day was parsed
five times over.

This module parses a capture ONCE into fixed-width columns (~127 B/record),
already in the order `HostTrajectoryExtractor.extract_trajectories` would
process it, and caches them on disk keyed by the inputs and the parsing code.
`HostTrajectoryExtractor.extract_trajectories_columns` consumes them and
produces the same trajectory store, bit for bit (tests/test_capture_columns.py).

## What the columns hold, and why each is exact

Exactly the record fields extraction reads, nothing it does not:

* the records `read_capture` returns (same strided read, same PCAP window
  concatenation, same `drop_unresolved` filter and coverage report), in
  the order a STABLE sort on `start_time` puts them -- the sort
  `FlowToTemporalEventAdapter.process_records` and `extract_trajectories`
  both apply;
* src/dst as codes into this capture's address table;
* start/end time (float64), the four byte/packet counters (int64),
  protocol and destination port (int64), is_attack;
* coarse category and technique list as codes into small tables;
* the 12 edge features, computed per record by the SAME function
  `process_records` calls (`extract_flow_record_edge_features`), with the
  edge-feature ablation mask OFF. The mask is applied on load
  (`CaptureColumns.edge_features`) by the same float32 multiply the function
  itself would do, so one cache entry serves every ablation setting.

Packet-level features (`record.metadata["packet_features"]`) are NOT kept:
only `include_packet_features=True` reads them, and the extractor refuses the
columnar path in that mode (`columns_unsupported_reason`).

## Cache

`<cache_dir>/<name>-<sha256>` where the hash covers: every input file's
relative name, size and mtime_ns (a PCAP day's every file plus its label CSV),
every read parameter, the storage format, and a content hash of every repo
module the parse imports (measured in a clean interpreter) plus the label-map
files the resolver reads. Change any of them and the key changes: a stale
entry is never read, only left behind. `meta.json` is written last and the
entry directory is renamed into place, so a crash never leaves a readable
half-entry.

Parsing runs in separate `python -m data_unification.capture_columns build`
processes, at most `workers` at a time; results are consumed in input order,
so everything downstream (window indices, host interning) is unchanged.

    python -m data_unification.capture_columns warm --scheme cross_year_ctu \\
        --pcap-root ... --cic2018-csv-dir ... --ctu-dir ... --cic2017-dir ... \\
        --cache-dir results/training_plan/.spill/capture_cache --workers 3

pre-parses a whole plan's captures (e.g. while the encoder trains).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Bump when the on-disk layout or the meaning of a column changes.
FORMAT = 2

#: (name, dtype). Every column has one row per kept record, sorted by start.
COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("src", "int32"), ("dst", "int32"),
    ("start", "float64"), ("end", "float64"),
    ("fwd_bytes", "int64"), ("bwd_bytes", "int64"),
    ("fwd_packets", "int64"), ("bwd_packets", "int64"),
    ("protocol", "int64"), ("dst_port", "int64"),
    ("is_attack", "bool"), ("cat", "int32"), ("tech", "int32"),
    ("unresolved", "bool"),
)
EDGE_DIM = 12
_CHUNK = 1 << 16

#: Modules the parse imports lazily; the code hash imports them explicitly so
#: it sees every one of them, then hashes every repo module that got loaded.
_PARSE_MODULES = (
    "data_unification.capture_columns",
    "data_unification.training_sources",
    "data_unification.label_filter",
    "data_unification.label_resolver",
    "data_unification.unified_schema",
    "data_unification.tgne_features",
    "data_unification.cic2017_adapter",
    "data_unification.cic2018_adapter",
    "data_unification.ctu13_adapter",
    "data_unification.attack_windows",
    "data_unification.pcap_bridge",
    "data_unification.pcap_adapter",
    "telemetry.capture.sniffer",
    "telemetry.flow.flow_table",
    "telemetry.packet.pcap_engine",
)


# ----------------------------------------------------------------- the spec
@dataclass(frozen=True)
class ColumnSpec:
    """Everything that decides which records a capture read yields.

    `reader`:
      "read_capture"   -- training_sources.read_capture (Branch A, and the
                          downstream CIC-2017 test read): strided adapter read,
                          or PCAP windows, then drop_unresolved.
      "pcap_windows"   -- downstream's iter_pcap_day_records: PCAP windows
                          concatenated, NO drop_unresolved.
      "one_capture"    -- downstream's read_one_capture: CIC-2018 adapter for
                          *.csv, CTU-13 adapter otherwise, strided, NO drop.
    """
    reader: str
    dataset: str
    name: str
    path: str
    window_seconds: float
    pcap_label_dir: Optional[str] = None
    rows_per_file: Optional[int] = None
    stride: int = 1
    pcap_max_windows: Optional[int] = None
    pcap_window_stride: int = 1
    max_packets_per_host: Optional[int] = None

    @classmethod
    def for_read_capture(cls, cap, *, window_seconds, pcap_label_dir=None, rows_per_file=None,
                         stride=1, pcap_max_windows=None, pcap_window_stride=1) -> "ColumnSpec":
        return cls("read_capture", cap.dataset, cap.name, str(cap.path), float(window_seconds),
                   str(pcap_label_dir) if pcap_label_dir else None, rows_per_file, int(stride),
                   pcap_max_windows, int(pcap_window_stride))

    @property
    def drops_unresolved(self) -> bool:
        return self.reader == "read_capture"

    @property
    def source(self) -> str:
        """Which parser the reader runs: 'pcap', 'cic2017', 'cic2018' or 'ctu13'.

        The three readers differ only in this choice and in whether they drop
        unresolved labels; the drop is applied on load, so one parse (one
        cache entry) serves Branch A's read_capture and the downstream readers.
        """
        if self.reader not in ("read_capture", "pcap_windows", "one_capture"):
            raise ValueError(f"unknown reader {self.reader!r}")
        if self.dataset == "PCAP2018" and self.reader != "one_capture":
            return "pcap"
        if self.reader == "read_capture":
            return {"CIC2017": "cic2017", "CIC2018": "cic2018"}.get(self.dataset, "ctu13")
        if self.reader == "one_capture":
            return "cic2018" if Path(self.path).suffix == ".csv" else "ctu13"
        raise ValueError(f"reader {self.reader!r} cannot read dataset {self.dataset!r}")

    def source_key(self) -> dict:
        """The parameters that decide the parsed records -- and only those."""
        if self.source == "pcap":
            return {"source": "pcap", "path": self.path, "window_seconds": self.window_seconds,
                    "label_dir": self.pcap_label_dir, "max_windows": self.pcap_max_windows,
                    "window_stride": self.pcap_window_stride,
                    "max_packets_per_host": self.max_packets_per_host}
        return {"source": self.source, "path": self.path,
                "rows_per_file": self.rows_per_file, "stride": self.stride}


def iter_spec_records(spec: ColumnSpec) -> Iterator[Any]:
    """The records `spec`'s reader yields, BEFORE any drop_unresolved, in order.

    Each branch is the reader's own code path (training_sources.read_capture,
    iter_pcap_day_windows, retrain_future_models_live.read_one_capture),
    called the same way.
    """
    src = spec.source
    if src == "pcap":
        from data_unification.training_sources import iter_pcap_day_windows
        if spec.pcap_label_dir is None:
            raise ValueError("PCAP captures need the CIC-2018 CSV label directory")
        for window in iter_pcap_day_windows(Path(spec.path), Path(spec.pcap_label_dir),
                                            spec.window_seconds, spec.pcap_max_windows,
                                            spec.pcap_window_stride, spec.max_packets_per_host):
            yield from window
        return
    cap_rows = None if spec.rows_per_file is None else spec.rows_per_file * spec.stride
    path = str(spec.path)
    if src == "cic2017":
        from data_unification.cic2017_adapter import CIC2017Adapter
        gen = CIC2017Adapter().parse_file(path, max_rows=cap_rows)
    elif src == "cic2018":
        from data_unification.cic2018_adapter import CIC2018Adapter
        gen = CIC2018Adapter().parse_file(path, max_rows=cap_rows)
    else:
        from data_unification.ctu13_adapter import CTU13Adapter
        gen = CTU13Adapter().parse_netflow_csv(path, max_rows=cap_rows)
    yield from _strided_iter(gen, spec.stride, spec.rows_per_file)


def _strided_iter(gen, stride: int, want=None) -> Iterator[Any]:
    """training_sources._strided as a generator: the same records in the same
    order, without holding the whole list (4.7M records for a CTU capture)."""
    k = 0
    for i, r in enumerate(gen):
        if i % stride == 0:
            yield r
            k += 1
            if want is not None and k >= want:
                break


# ------------------------------------------------------------ the columns
class CaptureColumns:
    """A parsed capture. Columns are read-only memmaps (page cache, not anon)."""

    def __init__(self, cols: Dict[str, np.ndarray], edge_raw: np.ndarray, meta: dict,
                 path: Optional[Path] = None):
        self.cols = cols
        self.edge_raw = edge_raw
        self.meta = meta
        self.path = path
        self.ips: List[str] = meta["ips"]
        self.categories: List[str] = meta["categories"]
        self.tech_lists: List[Tuple[str, ...]] = [tuple(t) for t in meta["tech_lists"]]
        cov = meta.get("coverage")
        if cov is not None:
            cov = dict(cov)
            cov["top_unresolved_labels"] = [tuple(x) for x in cov["top_unresolved_labels"]]
        self.coverage = cov

    def __len__(self) -> int:
        return int(self.meta["n"])

    def __getattr__(self, name):
        cols = self.__dict__.get("cols")
        if cols is not None and name in cols:
            return cols[name]
        raise AttributeError(name)

    def edge_features(self) -> np.ndarray:
        """(n, 12) float32 with the process's edge ablation mask applied.

        `extract_canonical_edge_features` ends with `feat *= mask` on a float32
        vector; this is the same float32 multiply, row-wise.
        """
        from data_unification.tgne_features import ablation_mask
        out = np.array(self.edge_raw, dtype=np.float32, copy=True)
        m = ablation_mask()
        if m is not None:
            out *= m
        return out

    @classmethod
    def load(cls, d: Path, drop_unresolved: bool, mmap: bool = True) -> "CaptureColumns":
        """The capture as a reader sees it. `drop_unresolved` is read_capture's
        label_filter.drop_unresolved: those rows are removed (order kept -- a
        stable sort restricted to a subset is the stable sort of the subset)
        and the coverage report is attached; without it every record is kept
        and `coverage` is None, as the downstream readers have it."""
        d = Path(d)
        with open(d / "meta.json") as f:
            meta = dict(json.load(f))
        mode = "r" if mmap else None
        cols = {name: np.load(d / f"{name}.npy", mmap_mode=mode) for name, _ in COLUMNS}
        edge = np.load(d / "edge.npy", mmap_mode=mode)
        if drop_unresolved:
            if meta["n_unresolved"]:
                keep = ~np.asarray(cols["unresolved"])
                cols = {k: np.asarray(v)[keep] for k, v in cols.items()}
                edge = np.asarray(edge)[keep]
                meta["n"] = int(keep.sum())
        else:
            meta["coverage"] = None
        return cls(cols, edge, meta, d)


@contextmanager
def _edge_ablation_off():
    """Compute UNablated edge features; restore the process's setting after."""
    import data_unification.tgne_features as tf
    saved = (tf._ABLATION_MASK, tf._ABLATION_READ)
    tf._ABLATION_MASK, tf._ABLATION_READ = None, True
    try:
        yield
    finally:
        tf._ABLATION_MASK, tf._ABLATION_READ = saved


def _coverage_report(total: int, raw: Counter, by_source: Counter, n_bad: int, top: int = 20) -> dict:
    """label_filter.count_unresolved's dict, from counters kept while streaming."""
    from data_unification.label_resolver import UNKNOWN_CATEGORY
    return {
        "total_records": total,
        "unresolved_records": n_bad,
        "unresolved_rate": (n_bad / total) if total else 0.0,
        "resolved_rate": (1.0 - n_bad / total) if total else 0.0,
        "distinct_unresolved_labels": len(raw),
        "top_unresolved_labels": raw.most_common(top),
        "unresolved_by_source": dict(by_source),
        "unknown_category": UNKNOWN_CATEGORY,
    }


def build_columns(spec: ColumnSpec, out_dir: Path, records: Optional[Iterable[Any]] = None) -> dict:
    """Parse `spec` (or the given records) into `out_dir`. Returns the meta dict.

    Streams: records are converted in chunks of 64k and appended to raw files,
    so the parse never holds more than one chunk (plus whatever the reader
    itself holds -- one PCAP window, or `_strided`'s list for a CSV).
    """
    from data_unification.label_filter import _category
    from data_unification.label_resolver import is_unresolved
    from data_unification.tgne_features import extract_flow_record_edge_features

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ip_index: Dict[str, int] = {}
    cat_index: Dict[str, int] = {}
    tech_index: Dict[Tuple[str, ...], int] = {}
    raw_bad: Counter = Counter()
    src_bad: Counter = Counter()
    n_total = n_bad = n = 0
    fh = {name: open(out_dir / f"{name}.bin", "wb") for name, _ in COLUMNS}
    fh_edge = open(out_dir / "edge.bin", "wb")
    buf: Dict[str, list] = {name: [] for name, _ in COLUMNS}
    ebuf: List[np.ndarray] = []

    def flush():
        for name, dt in COLUMNS:
            if buf[name]:
                np.asarray(buf[name], dtype=dt).tofile(fh[name])
                buf[name].clear()
        if ebuf:
            np.stack(ebuf).astype(np.float32, copy=False).tofile(fh_edge)
            ebuf.clear()

    it = records if records is not None else iter_spec_records(spec)
    b_src, b_dst = buf["src"], buf["dst"]
    b_start, b_end = buf["start"], buf["end"]
    b_fb, b_bb, b_fp, b_bp = buf["fwd_bytes"], buf["bwd_bytes"], buf["fwd_packets"], buf["bwd_packets"]
    b_proto, b_port, b_atk = buf["protocol"], buf["dst_port"], buf["is_attack"]
    b_cat, b_tech, b_unres = buf["cat"], buf["tech"], buf["unresolved"]
    try:
        with _edge_ablation_off():
            for r in it:
                n_total += 1
                # label_filter.count_unresolved / partition_unresolved, streamed.
                bad = is_unresolved(_category(r))
                if bad:
                    n_bad += 1
                    raw_bad[str(getattr(r, "raw_label", ""))] += 1
                    src_bad[str(getattr(r, "raw_label_source", ""))] += 1
                b_unres.append(bad)
                s = r.src_ip
                i = ip_index.get(s)
                if i is None:
                    i = ip_index[s] = len(ip_index)
                b_src.append(i)
                s = r.dst_ip
                i = ip_index.get(s)
                if i is None:
                    i = ip_index[s] = len(ip_index)
                b_dst.append(i)
                b_start.append(r.start_time)
                b_end.append(r.end_time)
                b_fb.append(r.fwd_bytes)
                b_bb.append(r.bwd_bytes)
                b_fp.append(r.fwd_packets)
                b_bp.append(r.bwd_packets)
                b_proto.append(r.protocol)
                b_port.append(r.dst_port)
                b_atk.append(bool(r.is_attack))
                c = r.coarse_category
                i = cat_index.get(c)
                if i is None:
                    i = cat_index[c] = len(cat_index)
                b_cat.append(i)
                tt = tuple(r.attck_technique_ids or ())
                i = tech_index.get(tt)
                if i is None:
                    i = tech_index[tt] = len(tech_index)
                b_tech.append(i)
                ebuf.append(extract_flow_record_edge_features(r))
                n += 1
                if len(b_src) >= _CHUNK:
                    flush()
        flush()
    finally:
        for f in fh.values():
            f.close()
        fh_edge.close()
    t_parse = time.time() - t0

    # Stable sort by start time: the order process_records' and
    # extract_trajectories' `sorted(records, key=start_time)` produce.
    start = np.fromfile(out_dir / "start.bin", dtype=np.float64)
    if n and np.isnan(start).any():
        raise ValueError(
            f"{spec.name}: {int(np.isnan(start).sum())} record(s) have a NaN start time; "
            f"Python's sort and numpy's disagree on NaN, so the columnar order could "
            f"differ from the record path's. Refusing rather than guessing.")
    order = np.argsort(start, kind="stable")
    del start
    for name, dt in COLUMNS:
        a = np.fromfile(out_dir / f"{name}.bin", dtype=dt)
        np.save(out_dir / f"{name}.npy", a[order])
        del a
        os.unlink(out_dir / f"{name}.bin")
    e = np.fromfile(out_dir / "edge.bin", dtype=np.float32).reshape(-1, EDGE_DIM)
    np.save(out_dir / "edge.npy", e[order])
    del e, order
    os.unlink(out_dir / "edge.bin")

    meta = {
        "format": FORMAT, "source": spec.source_key(), "n": n, "n_unresolved": n_bad,
        "ips": list(ip_index), "categories": list(cat_index),
        "tech_lists": [list(t) for t in tech_index],
        "coverage": _coverage_report(n_total, raw_bad, src_bad, n_bad),
        "t_parse_s": round(t_parse, 1), "t_total_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "meta.json", "w") as f:       # written last: the commit marker
        json.dump(meta, f)
    return meta


# ----------------------------------------------------------------- cache key
def _stat(f: Path, rel: str) -> list:
    st = f.stat()
    return [rel, st.st_size, st.st_mtime_ns]


def input_fingerprint(spec: ColumnSpec) -> list:
    """(relative name, size, mtime_ns) of every file the read touches."""
    p = Path(spec.path)
    if spec.source == "pcap":
        out = [_stat(f, str(f.relative_to(p))) for f in sorted(p.rglob("*")) if f.is_file()]
        from data_unification.training_sources import _pcap_label_csv
        csv_path = _pcap_label_csv(p, Path(spec.pcap_label_dir)) if spec.pcap_label_dir else None
        out.append(_stat(csv_path, "LABELS:" + csv_path.name) if csv_path else ["LABELS:", None, None])
        return out
    return [_stat(p, p.name)]


def _code_hash_here() -> Tuple[str, List[str]]:
    """Hash of every repo module the parse chain loads, plus the label maps."""
    import importlib
    for m in _PARSE_MODULES:
        importlib.import_module(m)
    repo = os.path.realpath(str(REPO))
    files = sorted({os.path.realpath(mod.__file__) for mod in list(sys.modules.values())
                    if getattr(mod, "__file__", None)
                    and os.path.realpath(mod.__file__).startswith(repo + os.sep)})
    files += sorted(str(p) for p in (REPO / "data_unification" / "label_maps").glob("*") if p.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, repo).encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest(), [os.path.relpath(f, repo) for f in files]


_CODE_HASH: Optional[str] = None


def parse_code_hash() -> str:
    """Measured in a clean interpreter so the caller's imports do not leak in."""
    global _CODE_HASH
    if _CODE_HASH is None:
        r = subprocess.run([sys.executable, "-m", "data_unification.capture_columns", "codehash"],
                           cwd=str(REPO), env=_child_env(), capture_output=True, text=True, check=True)
        _CODE_HASH = r.stdout.strip().splitlines()[-1]
    return _CODE_HASH


def cache_key(spec: ColumnSpec, code_hash: str) -> str:
    blob = json.dumps({"format": FORMAT, "source": spec.source_key(),
                       "inputs": input_fingerprint(spec), "code": code_hash}, sort_keys=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in f"{spec.dataset}_{spec.name}")
    return f"{safe}-{hashlib.sha256(blob.encode()).hexdigest()[:24]}"


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    # One parse is single-threaded Python; keep BLAS/OMP from fanning out.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(v, "1")
    return env


def _is_committed(entry: Path) -> bool:
    return (entry / "meta.json").is_file() and all(
        (entry / f"{n}.npy").is_file() for n, _ in COLUMNS) and (entry / "edge.npy").is_file()


# ----------------------------------------------------------------- building
def _build_subprocess(spec: ColumnSpec, out_dir: Path, log_prefix: str = "",
                      procs: Optional[set] = None) -> None:
    """Parse in a fresh interpreter: its memory is returned to the OS at exit,
    it imports only the parse chain (no torch), and a crash is contained."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-m", "data_unification.capture_columns", "build",
                          json.dumps(asdict(spec)), str(out_dir)],
                         cwd=str(REPO), env=_child_env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    if procs is not None:
        procs.add(p)
    try:
        out, _ = p.communicate()
    finally:
        if procs is not None:
            procs.discard(p)
    for line in (out or "").splitlines():
        if line.strip():
            print(f"{log_prefix}{line}", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"parsing {spec.dataset}/{spec.name} failed (exit {p.returncode}); "
                           f"see the lines above")


@contextmanager
def _entry_lock(entry: Path):
    """Exclusive, cross-process lock for one cache entry (an flock on a sidecar file)."""
    import fcntl
    lock = entry.with_name(f".{entry.name}.lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def iter_capture_columns(
    specs: Sequence[ColumnSpec],
    *,
    cache_dir: Optional[Path] = None,
    scratch_dir: Optional[Path] = None,
    workers: int = 1,
) -> Iterator[Tuple[ColumnSpec, CaptureColumns]]:
    """(spec, columns) for each spec, in INPUT order, parsing up to `workers` at once.

    `workers <= 0` parses in this process, one at a time (tests, debugging).
    With `cache_dir`, cached entries are served and new parses are committed
    there; without it, parses go to `scratch_dir` and are deleted once the
    caller has moved past them. Both must be on disk, not /tmp (tmpfs = RAM).
    Specs that parse the same records (e.g. Branch A's and the downstream
    reader of one PCAP day) share one entry and one parse.
    """
    specs = list(specs)
    if not specs:
        return
    if cache_dir is None and scratch_dir is None:
        raise ValueError("iter_capture_columns needs cache_dir or scratch_dir (on disk)")
    base = Path(cache_dir) if cache_dir is not None else Path(scratch_dir)
    base.mkdir(parents=True, exist_ok=True)
    code = parse_code_hash() if cache_dir is not None else "uncached"
    entries = [base / cache_key(s, code) for s in specs]
    if cache_dir is None:
        entries = [e.with_name(f"{e.name}-{os.getpid()}") for e in entries]
    first_of: Dict[Path, int] = {}
    for i, e in enumerate(entries):
        first_of.setdefault(e, i)
    last_use = {e: i for i, e in enumerate(entries)}
    todo = [i for e, i in first_of.items() if not (cache_dir is not None and _is_committed(e))]
    print(f"[capture columns] {len(specs)} capture read(s), {len(first_of)} distinct: "
          f"{len(first_of) - len(todo)} cached, {len(todo)} to parse "
          f"({workers if workers > 0 else 'in-process'} at a time) | {base}", flush=True)

    procs: set = set()

    def build(i: int) -> Path:
        spec, entry = specs[i], entries[i]
        # One builder per entry across PROCESSES: the plan's lanes run Branch A
        # side by side on one cache. Without the lock a second builder
        # finishing the same capture rmtree'd the first's committed entry while
        # the first was still loading it (FileNotFoundError on a column file,
        # found by the two-lane dry run). A committed entry is never replaced.
        with _entry_lock(entry):
            if cache_dir is not None and _is_committed(entry):
                return entry
            tmp = entry.with_name(f".tmp-{entry.name}-{os.getpid()}")
            t = time.time()
            if workers > 0:
                _build_subprocess(spec, tmp, log_prefix=f"  [parse {spec.name}] ", procs=procs)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                build_columns(spec, tmp)
            shutil.rmtree(entry, ignore_errors=True)   # only ever a damaged, uncommitted leftover
            os.rename(tmp, entry)
            print(f"[capture columns] parsed {spec.dataset}/{spec.name} in {time.time() - t:.0f}s",
                  flush=True)
        return entry

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 and todo else None
    futs = {entries[i]: pool.submit(build, i) for i in todo} if pool is not None else {}
    pending = set(entries[i] for i in todo)
    finished = False
    try:
        for i, spec in enumerate(specs):
            entry = entries[i]
            if entry in pending:
                if pool is not None:
                    futs.pop(entry).result()
                else:
                    build(i)
                pending.discard(entry)
            yield spec, CaptureColumns.load(entry, drop_unresolved=spec.drops_unresolved)
            if cache_dir is None and last_use[entry] == i:
                shutil.rmtree(entry, ignore_errors=True)
        finished = True
    finally:
        if not finished:
            for f in futs.values():
                f.cancel()
            for proc in list(procs):
                try:
                    proc.kill()
                except OSError:
                    pass
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for e in set(entries):
            for t in base.glob(f".tmp-{e.name}-{os.getpid()}"):
                shutil.rmtree(t, ignore_errors=True)
            if cache_dir is None:
                shutil.rmtree(e, ignore_errors=True)


_OFF = ("0", "off", "none", "false", "no")


@dataclass(frozen=True)
class ColumnsPlan:
    """How a trainer reads its captures: columns (cached or scratch) or records."""
    enabled: bool
    cache_dir: Optional[Path]
    scratch_dir: Optional[Path]
    reason: str

    def describe(self, workers: int) -> str:
        if not self.enabled:
            return f"capture columns OFF ({self.reason}): reading UnifiedFlowRecord lists"
        where = (f"cache {self.cache_dir}" if self.cache_dir is not None
                 else f"uncached, scratch {self.scratch_dir}")
        return f"capture columns ON: {where}, {workers} parse worker(s)"


def columns_plan(capture_cache: Optional[str], spill_dir, extractor=None) -> ColumnsPlan:
    """Decide the read path from --capture-cache, --spill-dir and the extractor.

    The cache is `--capture-cache`, else $CYBERWORLD_CAPTURE_CACHE, else
    `<spill-dir>/capture_cache` -- the plan passes the same spill dir to Branch
    A and to the downstream trainer, so every run shares one cache. 'off'
    (either place) keeps the record path. Without any directory on disk to
    put columns in, the record path is kept too: /tmp is RAM on this machine.
    """
    raw = capture_cache if capture_cache is not None else os.environ.get("CYBERWORLD_CAPTURE_CACHE")
    if raw is not None and raw.strip().lower() in _OFF:
        return ColumnsPlan(False, None, None, "capture cache set to off")
    if extractor is not None:
        why = extractor.columns_unsupported_reason()
        if why:
            return ColumnsPlan(False, None, None, why)
    if raw:
        return ColumnsPlan(True, Path(raw), None, "")
    if spill_dir:
        return ColumnsPlan(True, Path(spill_dir) / "capture_cache", None, "")
    return ColumnsPlan(False, None, None, "no --spill-dir or --capture-cache to hold the columns")


# --------------------------------------------------------------------- CLI
def _cli(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m data_unification.capture_columns")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("codehash")
    b = sub.add_parser("build")
    b.add_argument("spec_json")
    b.add_argument("out_dir")
    w = sub.add_parser("warm", help="parse a scheme's captures into the cache ahead of time")
    w.add_argument("--scheme", default="cross_year_ctu")
    w.add_argument("--pcap-root", type=Path)
    w.add_argument("--cic2018-csv-dir", type=Path)
    w.add_argument("--ctu-dir", type=Path)
    w.add_argument("--cic2017-dir", type=Path)
    w.add_argument("--cache-dir", type=Path, required=True)
    w.add_argument("--workers", type=int, default=3)
    w.add_argument("--window-seconds", type=float, default=None)
    w.add_argument("--downstream", action="store_true",
                   help="also warm the downstream trainer's reads (PCAP/CTU without the "
                        "unresolved-label filter)")
    a = ap.parse_args(argv)
    if a.cmd == "codehash":
        h, files = _code_hash_here()
        print(f"{len(files)} files")
        print(h)
        return 0
    if a.cmd == "build":
        spec = ColumnSpec(**json.loads(a.spec_json))
        meta = build_columns(spec, Path(a.out_dir))
        print(f"{spec.dataset}/{spec.name}: {meta['n']:,} records ({meta['n_unresolved']:,} with "
              f"unresolved labels), parse {meta['t_parse_s']}s, total {meta['t_total_s']}s",
              flush=True)
        return 0
    from data_unification.training_sources import discover_captures
    if a.window_seconds is None:
        from cyberworld_v4.config import get_contract
        a.window_seconds = get_contract().window_seconds
    caps = discover_captures(scheme=a.scheme, cic2017_dir=a.cic2017_dir, ctu13_dir=a.ctu_dir,
                             pcap2018_root=a.pcap_root)
    specs = [ColumnSpec.for_read_capture(c, window_seconds=a.window_seconds,
                                         pcap_label_dir=a.cic2018_csv_dir)
             for sp in ("train", "val", "test") for c in caps[sp]]
    if a.downstream:
        for sp in ("train", "val", "test"):
            for c in caps[sp]:
                if c.dataset == "PCAP2018":
                    specs.append(ColumnSpec("pcap_windows", c.dataset, c.name, str(c.path),
                                            float(a.window_seconds), str(a.cic2018_csv_dir)))
                elif c.dataset == "CTU13":
                    specs.append(ColumnSpec("one_capture", c.dataset, c.name, str(c.path),
                                            float(a.window_seconds)))
    for spec, cols in iter_capture_columns(specs, cache_dir=a.cache_dir, workers=a.workers):
        print(f"  ready {spec.reader} {spec.dataset}/{spec.name}: {len(cols):,} records", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
