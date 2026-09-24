"""
CIC-IDS2017 Dataset Adapter.

Ingests raw CIC-IDS2017 flow CSVs into UnifiedFlowRecord format.
Handles stripped column names, Windows-1252/latin1 encoding, microsecond duration conversions,
and timestamp normalization to UTC epoch seconds.
"""

import logging
import os
import glob
from typing import Iterator, List, Optional, Union
import pandas as pd
import numpy as np

from data_unification.row_guards import rejection_breakdown, valid_row_mask
from data_unification.time_utils import (
    detect_12h_clock_in_group,
    repair_12h_clock,
    to_epoch_seconds,
    warn_if_resolution_too_coarse,
)


# CIC-2017 splits Thursday and Friday into separate morning/afternoon files.
# An afternoon-only file holds hours 01-05 and is individually ambiguous -- a
# 12-hour dial and a genuine 01:00-05:00 capture look identical. Pooling a whole
# capture day resolves it: Friday's three files together span 01-05 and 08-12,
# which only a 12-hour dial produces. The verdict then applies to every file of
# that day, because they are one capture split across files.
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_CLOCK_REPAIR_CACHE: dict = {}


def _capture_day(filepath: str) -> Optional[str]:
    """The weekday this file belongs to, from its name (the corpus convention)."""
    name = os.path.basename(filepath).lower()
    for day in _WEEKDAYS:
        if name.startswith(day):
            return day
    return None


def _timestamp_column(filepath: str) -> Optional[str]:
    try:
        hdr = pd.read_csv(filepath, nrows=0, encoding="latin1")
    except Exception:
        return None
    for c in hdr.columns:
        if c.strip().lower() == "timestamp":
            return c
    return None


def _file_needs_clock_repair(filepath: str) -> bool:
    """12-hour-dial verdict for this file, decided over its whole capture day."""
    key = os.path.abspath(filepath)
    if key in _CLOCK_REPAIR_CACHE:
        return _CLOCK_REPAIR_CACHE[key]

    day = _capture_day(filepath)
    directory = os.path.dirname(key)
    if day:
        siblings = [
            f for f in sorted(glob.glob(os.path.join(directory, "*.csv")))
            if _capture_day(f) == day
        ]
    else:
        siblings = [key]

    pairs = [(f, _timestamp_column(f)) for f in siblings]
    pairs = [(f, c) for f, c in pairs if c]
    try:
        verdict = detect_12h_clock_in_group(pairs, utc=True) if pairs else False
    except Exception:
        verdict = False

    # One verdict for the whole day, cached against every file in it.
    for f, _c in pairs:
        _CLOCK_REPAIR_CACHE[os.path.abspath(f)] = verdict
    _CLOCK_REPAIR_CACHE[key] = verdict

    if verdict:
        logging.getLogger(__name__).warning(
            "%s capture day is on a 12-hour clock with no AM/PM (decided over %d file(s)); "
            "shifting 01:00-07:59 forward 12h.", day or os.path.basename(filepath), len(pairs),
        )
    return verdict
from data_unification.unified_schema import UnifiedFlowRecord, LabelSource
from data_unification.label_resolver import get_default_resolver, LabelResolver
from cyberworld_v4.config import get_contract


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

        # One verdict for the whole capture day; see _file_needs_clock_repair.
        needs_clock_repair = _file_needs_clock_repair(filepath)

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
                # pandas >= 2 returns datetime64[us] (or [s]/[ms]) depending on input, not
                # always [ns]. astype("int64") therefore yields MICROseconds here, and the
                # old "/ 1e9" produced epoch seconds 1000x too small -- a 12-hour capture
                # collapsed into 43 apparent seconds, so ~21,600 two-second windows became
                # ~22 and every host trajectory was meaningless. Upcast to [ns] explicitly
                # so the divisor is correct regardless of the parsed resolution.
                start_timestamps = to_epoch_seconds(ts_series)
                start_timestamps = np.nan_to_num(start_timestamps, nan=0.0)
                if needs_clock_repair:
                    start_timestamps = repair_12h_clock(start_timestamps)
            except Exception:
                start_timestamps = np.zeros(len(chunk), dtype=float)

            # 7 of 8 CIC-2017 captures are written to the MINUTE; see
            # time_utils.timestamp_resolution_seconds for the measurement and
            # what it does to the window grid and the hazard target. Kept out
            # of the try above so a bug here can never zero a timestamp column.
            warn_if_resolution_too_coarse(
                start_timestamps, get_contract().window_seconds,
                source=os.path.basename(filepath),
            )

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

            # Vectorised row guard. The previous per-row check tested only
            # `start_timestamps[i] <= 0`, which misses epoch-1970 rows (their
            # stamps are positive) and all CICFlowMeter TSO failures
            # (protocol 0 / port 0). One CIC-2017 file is 63% blank padding
            # rows appended by a spreadsheet -- 288,602 phantom records.
            keep = valid_row_mask(start_timestamps, protocols, dst_ports, src_ips)
            n_dropped = int((~keep).sum())
            if n_dropped:
                self._rejections = getattr(self, "_rejections", {})
                for k, v in rejection_breakdown(
                    start_timestamps, protocols, dst_ports, src_ips
                ).items():
                    self._rejections[k] = self._rejections.get(k, 0) + v

            for i in range(len(chunk)):
                if not keep[i]:
                    continue
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
