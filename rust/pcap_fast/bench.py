"""Throughput, Python reference vs pcap_fast, on one day dir (single process each).

    python rust/pcap_fast/bench.py <day_dir> <label_dir>

MB/s is over the bytes of the capture files as they sit on disk (sum of file
sizes of readable pcap files, capped at the valid prefix is NOT applied: the
same numerator is used for both sides, so the ratio is what matters).
The label CSV parse (derive_windows) is a per-day constant shared by both
paths; it is timed once and subtracted from both full-path numbers.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import pcap_fast_py as fast  # noqa: E402


def _cpu_self():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _cpu_children():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def main(day_dir, label_dir):
    from data_unification.attack_windows import derive_windows
    from data_unification.pcap_bridge import iter_merged_day_windows
    from data_unification.training_sources import _pcap_label_csv
    from data_unification.parallel_ingest import _parse_one, iter_parts

    files, hosts = fast._captures(day_dir)
    nbytes = sum(os.path.getsize(p) for p in files)
    res = {"day_dir": str(day_dir), "files": len(files), "MB": round(nbytes / 1e6, 1)}

    t = time.perf_counter()
    derive_windows(str(_pcap_label_csv(Path(day_dir), Path(label_dir))))
    t_csv = time.perf_counter() - t
    res["derive_windows_s"] = round(t_csv, 2)

    # 1. Rust binary alone, output discarded.
    c0 = _cpu_children()
    t = time.perf_counter()
    cmd = [fast.DEFAULT_BIN, "2.0", "0"]
    for h, p in zip(hosts, files):
        cmd += [h, str(p)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    w = time.perf_counter() - t
    res["rust_binary"] = {"wall_s": round(w, 2), "cpu_s": round(_cpu_children() - c0, 2),
                          "MB_s": round(nbytes / 1e6 / w, 1)}

    # 2. Fast full path: Rust + vectorised Python labels/columns.
    c0, s0 = _cpu_children(), _cpu_self()
    t = time.perf_counter()
    out = fast.parse_pcap_day_columns_fast(day_dir, label_dir, keep_columns=False)
    w = time.perf_counter() - t - t_csv
    res["fast_columns"] = {"records": out["n"], "wall_s": round(w, 2),
                           "rust_cpu_s": round(_cpu_children() - c0, 2),
                           "python_cpu_s": round(_cpu_self() - s0 - t_csv, 2),
                           "MB_s": round(nbytes / 1e6 / w, 1)}

    # 3. Python reference, stage 1 only (per-packet parse, flows, features, merge).
    s0 = _cpu_self()
    t = time.perf_counter()
    n = 0
    for _b, per_host in iter_merged_day_windows(day_dir):
        n += len(per_host)
    w = time.perf_counter() - t
    res["python_stage1"] = {"host_windows": n, "wall_s": round(w, 2),
                            "MB_s": round(nbytes / 1e6 / w, 1)}

    # 4. Python reference, full _parse_one (records, labels, columns, .npy parts).
    scratch = Path("/var/home/samito/Documents/SIH/rust_parser_experiment/scratch_bench")
    scratch.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    r = _parse_one(("PCAP2018", str(day_dir), str(scratch), 1, None, 12, str(label_dir),
                    2.0, 500_000))
    w = time.perf_counter() - t - t_csv
    for _ in iter_parts(r):
        pass
    res["python_parse_one"] = {"records": r.get("n"), "wall_s": round(w, 2),
                               "MB_s": round(nbytes / 1e6 / w, 1)}
    res["speedup_full"] = round(res["python_parse_one"]["wall_s"] / res["fast_columns"]["wall_s"], 1)
    res["speedup_cpu_full"] = round(
        res["python_parse_one"]["wall_s"]
        / (res["fast_columns"]["rust_cpu_s"] + res["fast_columns"]["python_cpu_s"]), 1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
