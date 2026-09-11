"""
Warden / IDEA Dataset Adapter.

Adapts CESNET Warden alert-sharing CSV records into UnifiedFlowRecord.
Handles timezone-aware DetectTime strings, IDEA dot-hierarchical categories,
string port values (e.g., 'OtherPort'), and flow counts.
"""

import os
import glob
from typing import Iterator, Optional
import pandas as pd
import numpy as np

from data_unification.unified_schema import UnifiedFlowRecord, LabelSource
from data_unification.label_resolver import get_default_resolver, LabelResolver


class WardenAdapter:
    """Adapter for Warden alert-sharing CSV files."""

    def __init__(self, resolver: Optional[LabelResolver] = None):
        self.resolver = resolver or get_default_resolver()

    def parse_file(
        self,
        filepath: str,
        max_rows: Optional[int] = None,
        chunksize: int = 50000,
    ) -> Iterator[UnifiedFlowRecord]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        chunks = pd.read_csv(filepath, chunksize=chunksize, nrows=max_rows, low_memory=False)
        rows_yielded = 0

        proto_map = {"tcp": 6, "udp": 17, "icmp": 1}

        for chunk in chunks:
            chunk.columns = [c.strip() for c in chunk.columns]
            cols = {c.lower(): c for c in chunk.columns}

            dt_col = cols.get("detecttime", "DetectTime")
            src_col = cols.get("sourceip", "SourceIP")
            dst_col = cols.get("targetip", "TargetIP")
            cat_col = cols.get("category", "Category")
            proto_col = cols.get("proto", "Proto")
            port_col = cols.get("port", "Port")
            flow_col = cols.get("flowcount", "FlowCount")

            # Parse timestamps with timezone normalization to UTC epoch
            try:
                dt_series = pd.to_datetime(chunk[dt_col], utc=True, errors="coerce")
                start_timestamps = (dt_series.astype("int64") / 1e9).to_numpy()
                start_timestamps = np.nan_to_num(start_timestamps, nan=0.0)
            except Exception:
                start_timestamps = np.zeros(len(chunk), dtype=float)

            # Warden alerts represent aggregated events over a short detection window (~10s)
            end_timestamps = start_timestamps + 10.0

            src_ips = chunk[src_col].fillna("0.0.0.0").astype(str).to_numpy()
            dst_ips = chunk[dst_col].fillna("0.0.0.0").astype(str).to_numpy()

            # Protocol normalization
            protos = [proto_map.get(str(p).lower().strip(), 6) for p in chunk[proto_col]]

            # Port normalization
            ports = []
            for p in chunk[port_col]:
                try:
                    ports.append(int(float(str(p).strip())))
                except (ValueError, TypeError):
                    ports.append(0)

            # Flow count
            flow_counts = pd.to_numeric(chunk[flow_col], errors="coerce").fillna(1).astype(int).to_numpy()
            categories = chunk[cat_col].astype(str).to_numpy()

            for i in range(len(chunk)):
                raw_cat = categories[i]
                coarse, attck, is_attack = self.resolver.resolve(raw_cat, source=LabelSource.WARDEN)
                fc = max(1, int(flow_counts[i]))

                record = UnifiedFlowRecord(
                    src_ip=src_ips[i],
                    dst_ip=dst_ips[i],
                    src_port=0,
                    dst_port=ports[i],
                    protocol=protos[i],
                    start_time=float(start_timestamps[i]),
                    end_time=float(end_timestamps[i]),
                    fwd_bytes=fc * 64,
                    bwd_bytes=0,
                    fwd_packets=fc,
                    bwd_packets=0,
                    raw_label=raw_cat,
                    raw_label_source=LabelSource.WARDEN.value,
                    is_attack=is_attack,
                    coarse_category=coarse,
                    attck_technique_ids=attck,
                    metadata={"flow_count": fc},
                )
                yield record
                rows_yielded += 1
                if max_rows is not None and rows_yielded >= max_rows:
                    return

    def parse_directory(
        self,
        dirpath: str,
        pattern: str = "*.csv",
        max_rows_per_file: Optional[int] = None,
    ) -> Iterator[UnifiedFlowRecord]:
        files = sorted(glob.glob(os.path.join(dirpath, pattern)))
        for f in files:
            yield from self.parse_file(f, max_rows=max_rows_per_file)
