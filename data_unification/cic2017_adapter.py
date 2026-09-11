"""
CIC-IDS2017 Dataset Adapter.

Ingests raw CIC-IDS2017 flow CSVs into UnifiedFlowRecord format.
Handles stripped column names, Windows-1252/latin1 encoding, microsecond duration conversions,
and timestamp normalization to UTC epoch seconds.
"""

import os
import glob
from typing import Iterator, List, Optional, Union
import pandas as pd
import numpy as np

from data_unification.unified_schema import UnifiedFlowRecord, LabelSource
from data_unification.label_resolver import get_default_resolver, LabelResolver


class CIC2017Adapter:
    """Adapter for reading and normalizing CIC-IDS2017 CSV files."""

    def __init__(self, resolver: Optional[LabelResolver] = None):
        self.resolver = resolver or get_default_resolver()

    def parse_file(
        self,
        filepath: str,
        max_rows: Optional[int] = None,
        chunksize: Optional[int] = 50000,
    ) -> Iterator[UnifiedFlowRecord]:
        """
        Parses a single CIC-IDS2017 CSV file in chunks and yields UnifiedFlowRecord instances.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read in chunks to prevent high memory usage on large files
        chunks = pd.read_csv(
            filepath,
            chunksize=chunksize,
            nrows=max_rows,
            encoding="latin1",
            low_memory=False,
        )

        rows_yielded = 0
        for chunk in chunks:
            # Strip column whitespace
            chunk.columns = [c.strip() for c in chunk.columns]

            # Detect column names with case insensitivity
            cols = {c.lower(): c for c in chunk.columns}

            src_ip_col = cols.get("source ip", "Source IP")
            dst_ip_col = cols.get("destination ip", "Destination IP")
            src_port_col = cols.get("source port", "Source Port")
            dst_port_col = cols.get("destination port", "Destination Port")
            proto_col = cols.get("protocol", "Protocol")
            ts_col = cols.get("timestamp", "Timestamp")
            dur_col = cols.get("flow duration", "Flow Duration")
            fwd_pkts_col = cols.get("total fwd packets", "Total Fwd Packets")
            bwd_pkts_col = cols.get("total backward packets", "Total Backward Packets")
            fwd_bytes_col = cols.get("total length of fwd packets", "Total Length of Fwd Packets")
            bwd_bytes_col = cols.get("total length of bwd packets", "Total Length of Bwd Packets")
            lbl_col = cols.get("label", "Label")

            # Parse timestamps
            try:
                ts_series = pd.to_datetime(chunk[ts_col], dayfirst=True, utc=True, errors="coerce")
                start_timestamps = (ts_series.astype("int64") / 1e9).to_numpy()
                start_timestamps = np.nan_to_num(start_timestamps, nan=0.0)
            except Exception:
                start_timestamps = np.zeros(len(chunk), dtype=float)

            # Durations are in microseconds in CICFlowMeter
            dur_seconds = (pd.to_numeric(chunk[dur_col], errors="coerce").fillna(0.0) / 1e6).to_numpy()
            dur_seconds = np.clip(dur_seconds, 0.0, 86400.0)
            end_timestamps = start_timestamps + dur_seconds

            src_ips = chunk[src_ip_col].astype(str).to_numpy()
            dst_ips = chunk[dst_ip_col].astype(str).to_numpy()
            src_ports = pd.to_numeric(chunk[src_port_col], errors="coerce").fillna(0).astype(int).to_numpy()
            dst_ports = pd.to_numeric(chunk[dst_port_col], errors="coerce").fillna(0).astype(int).to_numpy()
            protocols = pd.to_numeric(chunk[proto_col], errors="coerce").fillna(6).astype(int).to_numpy()

            fwd_pkts = pd.to_numeric(chunk[fwd_pkts_col], errors="coerce").fillna(0).astype(int).to_numpy()
            bwd_pkts = pd.to_numeric(chunk[bwd_pkts_col], errors="coerce").fillna(0).astype(int).to_numpy()
            fwd_bytes = pd.to_numeric(chunk[fwd_bytes_col], errors="coerce").fillna(0).astype(int).to_numpy()
            bwd_bytes = pd.to_numeric(chunk[bwd_bytes_col], errors="coerce").fillna(0).astype(int).to_numpy()
            labels = chunk[lbl_col].astype(str).to_numpy()

            for i in range(len(chunk)):
                raw_lbl = labels[i]
                coarse, attck, is_attack = self.resolver.resolve(raw_lbl, source=LabelSource.CIC2017)

                record = UnifiedFlowRecord(
                    src_ip=src_ips[i],
                    dst_ip=dst_ips[i],
                    src_port=src_ports[i],
                    dst_port=dst_ports[i],
                    protocol=protocols[i],
                    start_time=float(start_timestamps[i]),
                    end_time=float(end_timestamps[i]),
                    fwd_bytes=int(fwd_bytes[i]),
                    bwd_bytes=int(bwd_bytes[i]),
                    fwd_packets=int(fwd_pkts[i]),
                    bwd_packets=int(bwd_pkts[i]),
                    raw_label=raw_lbl,
                    raw_label_source=LabelSource.CIC2017.value,
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
        """Iterate over all CSV files in the directory."""
        files = sorted(glob.glob(os.path.join(dirpath, pattern)))
        for f in files:
            yield from self.parse_file(f, max_rows=max_rows_per_file)
