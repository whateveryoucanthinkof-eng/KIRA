"""
CTU-13 Dataset Adapter.

Handles both raw NetFlow / Argus bidirectional flow records and
pre-aggregated Parquet windowed states from the CTU-13 botnet dataset.
"""

import os
from typing import Iterator, Optional
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from data_unification.unified_schema import UnifiedFlowRecord, LabelSource
from data_unification.label_resolver import get_default_resolver, LabelResolver


class CTU13Adapter:
    """Adapter for CTU-13 NetFlow CSVs and preprocessed Parquet tables."""

    def __init__(self, resolver: Optional[LabelResolver] = None):
        self.resolver = resolver or get_default_resolver()

    def parse_parquet(
        self,
        filepath: str,
        max_rows: Optional[int] = None,
        batch_size: int = 10000,
    ) -> Iterator[UnifiedFlowRecord]:
        """
        Ingests pre-aggregated CTU-13 parquet time-window states into UnifiedFlowRecord.
        Synthesizes communicating host endpoints based on the scenario_id and botnet characteristics.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parquet file not found: {filepath}")

        parquet_file = pq.ParquetFile(filepath)
        rows_yielded = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            # Extract fields
            timestamps = pd.to_datetime(df["timestamp"], utc=True).astype("int64") / 1e9
            scenario_ids = df["scenario_id"].astype(str).to_numpy()
            malware_families = df["malware_family"].astype(str).to_numpy()

            fwd_pkts = df.get("total_fwd_pkts", pd.Series(np.ones(len(df)))).fillna(1).astype(int).to_numpy()
            bwd_pkts = df.get("total_bwd_pkts", pd.Series(np.zeros(len(df)))).fillna(0).astype(int).to_numpy()
            fwd_bytes = df.get("total_fwd_bytes", pd.Series(np.full(len(df), 128))).fillna(128).astype(int).to_numpy()
            bwd_bytes = df.get("total_bwd_bytes", pd.Series(np.zeros(len(df)))).fillna(0).astype(int).to_numpy()

            # Endpoints: Map scenario_id to representative infected host IP and C2/victim destination
            for i in range(len(df)):
                scen = scenario_ids[i]
                family = malware_families[i]
                coarse, attck, is_attack = self.resolver.resolve(family, source=LabelSource.CTU13)

                # Consistent host IPs per scenario
                bot_id = abs(hash(scen)) % 250 + 2
                src_ip = f"10.0.2.{bot_id}"
                dst_ip = "147.32.84.180" if is_attack else "147.32.80.1"
                src_port = 1024 + (i % 60000)
                dst_port = 6667 if "irc" in family.lower() else (80 if "http" in family.lower() else 443)

                start_t = float(timestamps.iloc[i])
                dur = 2.0  # 2-second windows in the parquet capture
                end_t = start_t + dur

                record = UnifiedFlowRecord(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=6,  # TCP
                    start_time=start_t,
                    end_time=end_t,
                    fwd_bytes=int(fwd_bytes[i]),
                    bwd_bytes=int(bwd_bytes[i]),
                    fwd_packets=int(fwd_pkts[i]),
                    bwd_packets=int(bwd_pkts[i]),
                    raw_label=family,
                    raw_label_source=LabelSource.CTU13.value,
                    is_attack=is_attack,
                    coarse_category=coarse,
                    attck_technique_ids=attck,
                    metadata={"scenario_id": scen, "active_flows": int(df.get("active_flows_count", [0])[i])},
                )
                yield record
                rows_yielded += 1
                if max_rows is not None and rows_yielded >= max_rows:
                    return

    def parse_netflow_csv(
        self,
        filepath: str,
        max_rows: Optional[int] = None,
        chunksize: int = 50000,
    ) -> Iterator[UnifiedFlowRecord]:
        """
        Parses raw Argus/NetFlow CSV from CTU-13.
        Headers: StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, TotPkts, TotBytes, SrcBytes, Label
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        chunks = pd.read_csv(filepath, chunksize=chunksize, nrows=max_rows, low_memory=False)
        rows_yielded = 0

        for chunk in chunks:
            chunk.columns = [c.strip() for c in chunk.columns]
            cols = {c.lower(): c for c in chunk.columns}

            ts_col = cols.get("starttime", "StartTime")
            dur_col = cols.get("dur", "Dur")
            proto_col = cols.get("proto", "Proto")
            src_col = cols.get("srcaddr", "SrcAddr")
            dst_col = cols.get("dstaddr", "DstAddr")
            sport_col = cols.get("sport", "Sport")
            dport_col = cols.get("dport", "Dport")
            tot_pkts_col = cols.get("totpkts", "TotPkts")
            tot_bytes_col = cols.get("totbytes", "TotBytes")
            src_bytes_col = cols.get("srcbytes", "SrcBytes")
            lbl_col = cols.get("label", "Label")

            start_timestamps = (
                pd.to_datetime(chunk[ts_col], errors="coerce")
                .astype("int64", copy=False)
                .to_numpy()
                / 1e9
            )
            durations = pd.to_numeric(chunk[dur_col], errors="coerce").fillna(0.0).to_numpy()
            end_timestamps = start_timestamps + durations

            src_ips = chunk[src_col].astype(str).to_numpy()
            dst_ips = chunk[dst_col].astype(str).to_numpy()
            src_ports = pd.to_numeric(chunk[sport_col], errors="coerce").fillna(0).astype(int).to_numpy()
            dst_ports = pd.to_numeric(chunk[dport_col], errors="coerce").fillna(0).astype(int).to_numpy()

            tot_pkts = pd.to_numeric(chunk[tot_pkts_col], errors="coerce").fillna(1).astype(int).to_numpy()
            tot_bytes = pd.to_numeric(chunk[tot_bytes_col], errors="coerce").fillna(64).astype(int).to_numpy()
            src_bytes = pd.to_numeric(chunk[src_bytes_col], errors="coerce")
            fwd_bytes = src_bytes.fillna(pd.Series(tot_bytes // 2, index=chunk.index)).astype(int).to_numpy()
            bwd_bytes = np.maximum(0, tot_bytes - fwd_bytes)
            fwd_pkts = np.maximum(1, tot_pkts // 2)
            bwd_pkts = np.maximum(0, tot_pkts - fwd_pkts)

            # Proto string to int
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            protos = [proto_map.get(str(p).lower().strip(), 6) for p in chunk[proto_col]]
            labels = chunk[lbl_col].astype(str).to_numpy()

            for i in range(len(chunk)):
                raw_lbl = labels[i]
                coarse, attck, is_attack = self.resolver.resolve(raw_lbl, source=LabelSource.CTU13)

                record = UnifiedFlowRecord(
                    src_ip=src_ips[i],
                    dst_ip=dst_ips[i],
                    src_port=int(src_ports[i]),
                    dst_port=int(dst_ports[i]),
                    protocol=int(protos[i]),
                    start_time=float(start_timestamps[i]),
                    end_time=float(end_timestamps[i]),
                    fwd_bytes=int(fwd_bytes[i]),
                    bwd_bytes=int(bwd_bytes[i]),
                    fwd_packets=int(fwd_pkts[i]),
                    bwd_packets=int(bwd_pkts[i]),
                    raw_label=raw_lbl,
                    raw_label_source=LabelSource.CTU13.value,
                    is_attack=is_attack,
                    coarse_category=coarse,
                    attck_technique_ids=attck,
                )
                yield record
                rows_yielded += 1
                if max_rows is not None and rows_yielded >= max_rows:
                    return
