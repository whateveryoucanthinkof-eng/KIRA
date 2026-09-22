#!/usr/bin/env python3
"""Build a real telemetry fixture for scripts/verify_offline_live_parity.py.

## Why this exists

The parity check compares the feature vector the *trainers* build against the
one the *serving adapter* builds for the same flows. A divergence there is
invisible to both test suites and shows up only as unexplained live
degradation -- it is exactly the class of defect that produced the E7
graph-parity bug, where training saw a full-window graph and serving saw a
subgraph.

The check could never actually run: `captures/states.jsonl` holds four windows
with zero packets and no flows, so every window was skipped and it reported
"no comparable windows -- cannot assert parity". A verification that always
passes vacuously is worse than none, because it reads as a green check.

This replays a real capture into the SPAN-shaped flow dicts
`flows_from_span_dicts` consumes, grouped into contract-sized windows.

## On the window cap

`--windows` bounds the FIXTURE, not any training set. Parity is a per-window
property: if offline and live construction agree on a few hundred real windows
spanning several hosts and protocols, they agree. This is not subsampling
training data and the no-dilution rule does not apply to it. Nothing produced
here is ever trained on -- the records carry no labels at all
(`raw_label="UNLABELED"`, `is_attack=False`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cyberworld_v4.config import get_contract      # noqa: E402
from data_unification.cic2018_adapter import CIC2018Adapter   # noqa: E402
from data_unification.ctu13_adapter import CTU13Adapter       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True,
                    help="One .csv (CIC) or .binetflow (CTU-13) capture file")
    ap.add_argument("--out", type=Path, default=REPO / "captures" / "parity_states.jsonl")
    ap.add_argument("--windows", type=int, default=300,
                    help="Number of populated windows to emit (fixture size, "
                         "not a training cap -- see the module docstring)")
    ap.add_argument("--max-records", type=int, default=400_000,
                    help="Stop reading the capture after this many records")
    args = ap.parse_args()

    window = get_contract().window_seconds
    cic, ctu = CIC2018Adapter(), CTU13Adapter()
    gen = (cic.parse_file(str(args.capture), max_rows=args.max_records)
           if args.capture.suffix.lower() == ".csv"
           else ctu.parse_netflow_csv(str(args.capture), max_rows=args.max_records))

    buckets: dict[int, list] = defaultdict(list)
    base = None
    n = 0
    for rec in gen:
        ts = float(getattr(rec, "start_time", 0.0) or 0.0)
        if ts <= 0:
            continue
        if base is None:
            base = ts
        idx = int((ts - base) // window)
        if idx < 0:
            continue
        buckets[idx].append(rec)
        n += 1
        if len(buckets) > args.windows * 4:
            break

    populated = sorted((i, f) for i, f in buckets.items() if len(f) >= 2)[: args.windows]
    if not populated:
        print(f"no populated windows found in {args.capture}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hosts = set()
    with args.out.open("w") as fh:
        for wid, recs in populated:
            flows = []
            for r in recs:
                flows.append({
                    "src_ip": r.src_ip, "dst_ip": r.dst_ip,
                    "src_port": int(r.src_port or 0), "dst_port": int(r.dst_port or 0),
                    "protocol": int(r.protocol or 6),
                    "fwd_bytes": int(r.fwd_bytes or 0), "bwd_bytes": int(r.bwd_bytes or 0),
                    "fwd_packets": int(r.fwd_packets or 0),
                    "bwd_packets": int(r.bwd_packets or 0),
                    "start_time": float(r.start_time), "end_time": float(r.end_time),
                })
                hosts.add(r.src_ip); hosts.add(r.dst_ip)
            fh.write(json.dumps({
                "window_id": wid,
                "window_start": base + wid * window,
                "window_end": base + (wid + 1) * window,
                "packet_count": sum(f["fwd_packets"] + f["bwd_packets"] for f in flows),
                "flows": flows,
            }) + "\n")

    print(f"wrote {len(populated)} windows, {sum(len(r) for _, r in populated)} flows, "
          f"{len(hosts)} distinct hosts -> {args.out}")
    print(f"read {n} records from {args.capture.name} at {window}s windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
