"""Build a bounded Warden-shaped flow table for canonical TGNE retraining."""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cic-dir", type=Path, required=True)
    parser.add_argument("--ctu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=1000)
    args = parser.parse_args()

    records = []
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    for path in sorted(args.cic_dir.glob("*.csv")):
        records.extend(cic.parse_file(str(path), max_rows=args.rows_per_file))
    for path in sorted(args.ctu_dir.glob("*/*.binetflow")):
        records.extend(ctu.parse_netflow_csv(str(path), max_rows=args.rows_per_file))
    if not records:
        raise RuntimeError("No SIH/CTU records were loaded")

    rows = [
        {
            "SourceIP": record.src_ip,
            "TargetIP": record.dst_ip,
            "Proto": {6: "tcp", 17: "udp", 1: "icmp"}.get(record.protocol, "tcp"),
            "Port": record.dst_port,
            "DetectTime": pd.to_datetime(record.start_time, unit="s", utc=True).isoformat(),
            "Category": record.raw_label,
            "FlowCount": record.fwd_packets + record.bwd_packets,
        }
        for record in records
        if math.isfinite(record.start_time) and record.start_time > 0.0
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "10March_e.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote={output} records={len(rows)}")


if __name__ == "__main__":
    main()