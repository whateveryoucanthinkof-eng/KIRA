#!/usr/bin/env python3
"""Parse every capture the encoder reads into the ingest cache, once.

The plan trains several encoders (two IP variants, three seeds) on the same
captures. Warming the cache first means none of them parses, and encoders run
side by side never race to parse the same day twice.

The capture list, parameters and split filter mirror
bita/train.py::load_and_preprocess_unified_dataset, so the keys written here
are the keys the encoder looks up. tests/test_warm_ingest_cache.py pins that:
after a warm, the encoder's own load is served entirely from cache.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def encoder_captures(*, ctu13_dir=None, pcap2018_root=None, cic2017_dir=None,
                     cic2018_dir=None, scheme="frozen", splits=("train",)):
    """[(kind, path)] exactly as the encoder's loader builds and filters it."""
    from data_unification.split_policy import split_of, split_of_path

    captures = []
    if cic2017_dir:
        captures += [("CIC2017", f) for f in sorted(glob.glob(os.path.join(cic2017_dir, "*.csv")))]
    if cic2018_dir:
        captures += [("CIC2018", f) for f in sorted(glob.glob(os.path.join(cic2018_dir, "*.csv")))]
    if ctu13_dir:
        captures += [("CTU13", f) for f in sorted(glob.glob(os.path.join(ctu13_dir, "*", "*.binetflow")))]
    if pcap2018_root:
        captures += [("PCAP2018", os.path.join(pcap2018_root, d))
                     for d in sorted(os.listdir(pcap2018_root))
                     if os.path.isdir(os.path.join(pcap2018_root, d)) and d.endswith(("_pcap", "_pacap"))]
    wanted = set(splits) if splits else None

    def in_split(kind, path):
        try:
            sp = (split_of("PCAP2018", os.path.basename(path), scheme) if kind == "PCAP2018"
                  else split_of_path(path, scheme))
        except (KeyError, ValueError):
            return False
        return sp is not None and (wanted is None or sp in wanted)

    return [(k, p) for k, p in captures if in_split(k, p)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ctu-dir")
    ap.add_argument("--pcap-root")
    ap.add_argument("--cic2018-csv-dir", help="PCAP label CSVs")
    ap.add_argument("--split-scheme", default="cross_year_ctu")
    ap.add_argument("--splits", default="train")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from cyberworld_v4.config import get_contract
    from data_unification.parallel_ingest import parse_captures_parallel
    from data_unification.tgne_features import extract_canonical_edge_features

    edge_dim = len(extract_canonical_edge_features(
        fwd_bytes=0, bwd_bytes=0, fwd_packets=0, bwd_packets=0, duration_sec=0.0,
        byte_rate=0.0, packet_rate=0.0, protocol=6, dst_port=0))
    caps = encoder_captures(ctu13_dir=a.ctu_dir, pcap2018_root=a.pcap_root,
                            scheme=a.split_scheme, splits=tuple(a.splits.split(",")))
    print(f"[warm] {len(caps)} capture(s) -> {a.cache}", flush=True)
    t0, n, errors = time.time(), 0, []
    for res in parse_captures_parallel(
            caps, workers=a.workers, cache_dir=a.cache, edge_dim=edge_dim,
            scratch=os.path.join(a.cache, ".warm-scratch"),
            pcap_label_dir=a.cic2018_csv_dir, window_seconds=get_contract().window_seconds,
            on_capture=lambda r: errors.append(r) if r.get("error") else None):
        n += res.get("n", 0)
        print(f"[warm] {os.path.basename(res['path'])}: {res['n']} records"
              f"{' (cached)' if res.get('cached') else ''} ({time.time() - t0:.0f}s)", flush=True)
    if errors:
        # The encoder would skip these captures and train on less data; stop here.
        for r in errors:
            print(f"[warm] FAILED {r['path']}: {r['error']}", flush=True)
        return 1
    print(f"[warm] done: {n} records in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
