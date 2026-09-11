"""
CIC-IDS2018 Dataset Adapter.

Handles chunked streaming ingestion of large-scale multiday CIC-IDS2018 CSV files,
accommodating schema variations, column headers with or without spaces, and
normalizing timestamps and flow features to UnifiedFlowRecord.
"""

import os
import glob
from typing import Iterator, Optional
import pandas as pd
import numpy as np

from data_unification.unified_schema import UnifiedFlowRecord, LabelSource
from data_unification.label_resolver import get_default_resolver, LabelResolver


class CIC2018Adapter:
    """Chunked streaming reader and normalizer for CIC-IDS2018 data."""

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

        chunks = pd.read_csv(
            filepath,
            chunksize=chunksize,
            nrows=max_rows,
            encoding="latin1",
            low_memory=False,
        )

        rows_yielded = 0
        for chunk in chunks:
            chunk.columns = [c.strip() for c in chunk.columns]
            cols = {c.lower(): c for c in chunk.columns}

            src_ip_col = cols.get("src ip", cols.get("source ip", None))
            dst_ip_col = cols.get("dst ip", cols.get("destination ip", None))
            src_port_col = cols.get("src port", cols.get("source port", None))
            dst_port_col = cols.get("dst port", cols.get("destination port", None))
            proto_col = cols.get("protocol", None)
            ts_col = cols.get("timestamp", None)
            dur_col = cols.get("flow duration", None)

            fwd_pkts_col = cols.get("tot fwd pkts", cols.get("total fwd packets", None))
            bwd_pkts_col = cols.get("tot bwd pkts", cols.get("total backward packets", None))
            fwd_bytes_col = cols.get("totlen fwd pkts", cols.get("total length of fwd packets", None))
            bwd_bytes_col = cols.get("totlen bwd pkts", cols.get("total length of bwd packets", None))
            lbl_col = cols.get("label", None)

            # Fallback if IP columns not present
            if src_ip_col:
                src_ips = chunk[src_ip_col].astype(str).to_numpy()
            else:
                src_ips = np.array([f"192.168.10.{i % 250 + 1}" for i in range(len(chunk))])

            if dst_ip_col:
                dst_ips = chunk[dst_ip_col].astype(str).to_numpy()
            else:
                dst_ips = np.array([f"172.16.0.{i % 100 + 1}" for i in range(len(chunk))])

            src_ports = (
                pd.to_numeric(chunk[src_port_col], errors="coerce").fillna(0).astype(int).to_numpy()
                if src_port_col
                else np.random.randint(1024, 65535, size=len(chunk))
            )
            dst_ports = (
                pd.to_numeric(chunk[dst_port_col], errors="coerce").fillna(80).astype(int).to_numpy()
                if dst_port_col
                else np.full(len(chunk), 80, dtype=int)
            )
            protocols = (
                pd.to_numeric(chunk[proto_col], errors="coerce").fillna(6).astype(int).to_numpy()
                if proto_col
                else np.full(len(chunk), 6, dtype=int)
            )

            # Timestamps
            if ts_col:
                ts_series = pd.to_datetime(chunk[ts_col], dayfirst=True, errors="coerce")
                start_timestamps = (ts_series.astype("int64") / 1e9).to_numpy()
            else:
                start_timestamps = np.zeros(len(chunk), dtype=float)

            # Flow Duration (microseconds)
            if dur_col:
                dur_seconds = (pd.to_numeric(chunk[dur_col], errors="coerce").fillna(0.0) / 1e6).to_numpy()
                dur_seconds = np.clip(dur_seconds, 0.0, 86400.0)
            else:
                dur_seconds = np.zeros(len(chunk), dtype=float)
            end_timestamps = start_timestamps + dur_seconds

            fwd_pkts = (
                pd.to_numeric(chunk[fwd_pkts_col], errors="coerce").fillna(1).astype(int).to_numpy()
                if fwd_pkts_col
                else np.ones(len(chunk), dtype=int)
            )
            bwd_pkts = (
                pd.to_numeric(chunk[bwd_pkts_col], errors="coerce").fillna(0).astype(int).to_numpy()
                if bwd_pkts_col
                else np.zeros(len(chunk), dtype=int)
            )
            fwd_bytes = (
                pd.to_numeric(chunk[fwd_bytes_col], errors="coerce").fillna(64).astype(int).to_numpy()
                if fwd_bytes_col
                else np.full(len(chunk), 64, dtype=int)
            )
            bwd_bytes = (
                pd.to_numeric(chunk[bwd_bytes_col], errors="coerce").fillna(0).astype(int).to_numpy()
                if bwd_bytes_col
                else np.zeros(len(chunk), dtype=int)
            )

            if lbl_col:
                labels = chunk[lbl_col].astype(str).to_numpy()
            else:
                labels = np.array(["BENIGN"] * len(chunk))

            for i in range(len(chunk)):
                raw_lbl = labels[i]
                coarse, attck, is_attack = self.resolver.resolve(raw_lbl, source=LabelSource.CIC2018)

                record = UnifiedFlowRecord(
                    src_ip=src_ips[i],
                    dst_ip=dst_ips[i],
                    src_port=int(src_ports[i]),
                    dst_port=int(dst_ports[i]),
                    protocol=int(protocols[i]),
                    start_time=float(start_timestamps[i]),
                    end_time=float(end_timestamps[i]),
                    fwd_bytes=int(fwd_bytes[i]),
                    bwd_bytes=int(bwd_bytes[i]),
                    fwd_packets=int(fwd_pkts[i]),
                    bwd_packets=int(bwd_pkts[i]),
                    raw_label=raw_lbl,
                    raw_label_source=LabelSource.CIC2018.value,
                    is_attack=is_attack,
                    coarse_category=coarse,
                    attck_technique_ids=attck,
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
