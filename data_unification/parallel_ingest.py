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

import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _parse_one(args) -> Optional[dict]:
    """Worker: parse one capture into columns with a LOCAL string vocabulary."""
    kind, path, outdir, stride, max_rows, edge_dim = args
    try:
        from data_unification.cic2017_adapter import CIC2017Adapter
        from data_unification.cic2018_adapter import CIC2018Adapter
        from data_unification.ctu13_adapter import CTU13Adapter
        from data_unification.tgne_features import extract_canonical_edge_features

        if kind == "CIC2017":
            stream = CIC2017Adapter().parse_file(path, max_rows=None)
        elif kind == "CIC2018":
            stream = CIC2018Adapter().parse_file(path, max_rows=None)
        else:
            stream = CTU13Adapter().parse_netflow_csv(path, max_rows=None)

        ip_local: Dict[str, int] = {}
        cat_local: Dict[str, int] = {}
        u, i, ts, lbl = [], [], [], []
        edges = []

        kept = 0
        for n, r in enumerate(stream):
            if stride > 1 and (n % stride):
                continue
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
            if max_rows is not None and kept >= max_rows:
                break

        if not kept:
            return {"path": path, "kind": kind, "n": 0}

        stem = Path(outdir) / (Path(path).name.replace("/", "_") + f".{os.getpid()}")
        np.save(f"{stem}.u.npy", np.asarray(u, dtype=np.int32))
        np.save(f"{stem}.i.npy", np.asarray(i, dtype=np.int32))
        np.save(f"{stem}.ts.npy", np.asarray(ts, dtype=np.float64))
        np.save(f"{stem}.lbl.npy", np.asarray(lbl, dtype=np.int32))
        np.save(f"{stem}.edge.npy", np.asarray(edges, dtype=np.float32))

        # Vocabularies in local-id order, so index == local id.
        ips = [None] * len(ip_local)
        for s, k in ip_local.items():
            ips[k] = s
        cats = [None] * len(cat_local)
        for s, k in cat_local.items():
            cats[k] = s
        return {"path": path, "kind": kind, "n": kept, "stem": str(stem),
                "ips": ips, "cats": cats}
    except Exception as exc:
        # Return the error rather than only logging it: this runs in a CHILD
        # process, whose log handlers are not the parent's, so a message
        # emitted here can vanish entirely. The parent re-raises it into its
        # own log, where a failed capture is actually visible.
        return {"path": path, "kind": kind, "n": 0,
                "error": f"{type(exc).__name__}: {exc}"}


def parse_captures_parallel(
    captures: List[Tuple[str, str]],
    *,
    stride: int = 1,
    max_rows_per_file: Optional[int] = None,
    edge_dim: int = 12,
    workers: Optional[int] = None,
    scratch: Optional[str] = None,
    on_capture: Optional[Callable[[dict], None]] = None,
):
    """Parse `captures` = [(kind, path)] in parallel; yield results in INPUT order.

    Yields dicts with the column arrays loaded, in exactly the order given, so
    the caller's encounter-order id assignment is unchanged from serial.
    """
    if workers is None:
        # Leave headroom: each worker holds one capture's columns, and the
        # largest single capture (tue_20, 7.9M rows) is ~600 MB.
        workers = max(1, min(6, (os.cpu_count() or 4) - 2))

    tmp = scratch or tempfile.mkdtemp(prefix="tgne_ingest_")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    args = [(k, p, tmp, stride, max_rows_per_file, edge_dim) for k, p in captures]

    logger.info("Parsing %d captures across %d worker processes", len(args), workers)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # map() preserves input order, which is what keeps ids deterministic.
            for res in ex.map(_parse_one, args):
                if res.get("error"):
                    # Surfaced in the PARENT's log; a worker's own handlers
                    # are not connected to it.
                    logger.error("parallel parse failed on %s: %s",
                                 res["path"], res["error"])
                elif res.get("n", 0) == 0:
                    logger.warning("parallel parse produced 0 records for %s",
                                   res["path"])
                if res.get("n", 0) == 0:
                    if on_capture:
                        on_capture(res)
                    continue
                stem = res["stem"]
                res["u"] = np.load(f"{stem}.u.npy")
                res["i"] = np.load(f"{stem}.i.npy")
                res["ts"] = np.load(f"{stem}.ts.npy")
                res["lbl"] = np.load(f"{stem}.lbl.npy")
                res["edge"] = np.load(f"{stem}.edge.npy")
                for suf in ("u", "i", "ts", "lbl", "edge"):
                    try:
                        os.unlink(f"{stem}.{suf}.npy")
                    except OSError:
                        pass
                if on_capture:
                    on_capture(res)
                yield res
    finally:
        if scratch is None:
            shutil.rmtree(tmp, ignore_errors=True)
