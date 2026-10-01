"""Extract several captures at once, merge them into one store in capture order.

## Why this is allowed

`HostTrajectoryExtractor.extract_trajectories_columns` treats every call as
its own capture: TGN memory and host ids are reset, the neighbour graph is the
capture's own. Nothing one capture computes reaches another. So the captures
of a split can be extracted in separate processes and appended afterwards, in
the original order, with each capture's window indices shifted exactly as the
serial loop would have shifted them (`window_idx_base = next_window_base()`).
`TrajectoryStoreBuilder.append_part` re-interns hosts, categories and
techniques in first-appearance order, so ids come out identical too.

The one thing carried between captures in the serial loop is `tgn.n_nodes`,
the size of the node tables (it only grows). Rows past a capture's own hosts
are zero padding that no lookup reaches, so the values do not depend on it;
tests/test_capture_columns.py checks parallel == serial on chained captures.

## Why it matters

Extraction is dominated by the TGN memory update (BiTA aggregation and
message construction in bita/, a per-node Python loop): ~70 us per record on
the CPU, ~15 min per CIC-2018 PCAP day, ~2.5 h for the cross_year_ctu train
split -- per Branch A run. It is single-threaded Python, so processes scale.

## Memory

Each worker holds one capture's extraction (~1.2 GB for a 12M-record PCAP
day, measured) plus torch. Parts are written to disk and appended to the
parent's builder, whose feature block spills as before.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

_WORKER: Dict[str, Any] = {}

#: `tgne="whole-model:<path>"` loads torch.save(model) instead of a checkpoint.
WHOLE_MODEL = "whole-model:"


def _worker_init(tgne: Optional[str], extractor_kw: dict, threads: int, repo: str) -> None:
    import sys
    for p in (repo, os.path.join(repo, "bita")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(max(1, int(threads)))
    from data_unification.multi_dataset_stream import HostTrajectoryExtractor
    if tgne is not None and tgne.startswith(WHOLE_MODEL):
        # A whole pickled encoder (tests: a random, never-trained model).
        tgn = torch.load(tgne[len(WHOLE_MODEL):], map_location="cpu", weights_only=False)
    else:
        from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
        tgn = build_or_load_tgne_ta(checkpoint_path=tgne)
    from data_unification.fast_extract import enable_fast_extraction
    enable_fast_extraction(tgn)      # bit-identical, ~2x (data_unification/fast_extract.py)
    _WORKER["ex"] = HostTrajectoryExtractor(tgne_ta_model=tgn, **extractor_kw)


def _extract_one(entry: str, drop_unresolved: bool, namespace: str, out_dir: str) -> dict:
    from data_unification.capture_columns import CaptureColumns
    from data_unification.trajectory_store import TrajectoryStoreBuilder
    t = time.time()
    ex = _WORKER["ex"]
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    cols = CaptureColumns.load(Path(entry), drop_unresolved=drop_unresolved)
    b = TrajectoryStoreBuilder(spill_dir=out_dir, feat_dim=12 + ex.n_temporal_attrs)
    b.set_namespace(namespace)
    ex.extract_trajectories_columns(cols, builder=b, window_idx_base=0)
    b.export_part(out_dir)
    return {"n_records": len(cols), "n_snapshots": b._n, "seconds": time.time() - t,
            "exposure": ex.neighbor_exposure_report(reset=True)}


_COUNTERS = ("host_windows", "truncated_host_windows", "flows", "flows_outside_latent",
             "attack_host_windows", "attack_host_windows_all_attack_flows_outside_latent")


def extract_parallel(
    items: Iterable[Tuple[Any, Any, str]],
    *,
    extractor,
    builder_for,
    tgne: Optional[str],
    part_dir: Path,
    workers: int,
    threads_per_worker: int = 1,
    exposure_for=None,
) -> Iterator[Tuple[Any, dict]]:
    """Extract `items` = (key, CaptureColumns, namespace) in worker processes.

    Parts are appended to `builder_for(key)` strictly in `items` order, and
    each worker's neighbour-exposure counters are added to `extractor`'s, so
    the caller sees what the serial loop would have produced. Yields
    (key, info) after each capture is appended. `extractor` supplies the
    settings the workers copy; it must support the columnar path.

    `exposure_for(key)`, if given, returns the counter dict a capture's
    neighbour-exposure counts are added to (default: the extractor's own) --
    for one call that fills several splits' builders and reports per split.
    """
    why = extractor.columns_unsupported_reason()
    if why:
        raise ValueError(f"parallel extraction needs the columnar path: {why}")
    kw = dict(window_size_sec=extractor.window_size_sec,
              n_temporal_attrs=extractor.n_temporal_attrs,
              attack_role=extractor.attack_role,
              n_neighbors=extractor.n_neighbors)
    repo = str(Path(__file__).resolve().parents[1])
    part_dir = Path(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    pending = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(tgne, kw, threads_per_worker, repo)) as pool:
        try:
            for n, (key, cols, ns) in enumerate(items):
                out = part_dir / f"part-{os.getpid()}-{n}"
                fut = pool.submit(_extract_one, str(cols.path),
                                  cols.coverage is not None, ns, str(out))
                pending.append((key, out, fut))
                # Bound the parts waiting on disk: drain finished heads first.
                while pending and (pending[0][2].done() or len(pending) > 2 * workers):
                    yield _merge(pending.pop(0), builder_for, extractor, exposure_for)
            while pending:
                yield _merge(pending.pop(0), builder_for, extractor, exposure_for)
        finally:
            for _k, out, fut in pending:
                fut.cancel()
            for _k, out, _f in pending:
                shutil.rmtree(out, ignore_errors=True)


def _merge(item, builder_for, extractor, exposure_for=None) -> Tuple[Any, dict]:
    key, out, fut = item
    info = fut.result()
    b = builder_for(key)
    t = time.time()
    b.append_part(out, window_idx_offset=b.next_window_base())
    shutil.rmtree(out, ignore_errors=True)
    target = exposure_for(key) if exposure_for is not None else extractor._exposure
    for k in _COUNTERS:
        target[k] += int(info["exposure"][k])
    info["merge_seconds"] = time.time() - t
    return key, info
