"""CIC-IDS-2018 PCAP adapter — real host identity and packet-level features.

This exists because the CSV path cannot support the claims the project makes.

Two defects it fixes, both measured rather than assumed:

  * **Fabricated identity (audit D1).** Only `tue_20_csv.csv` carries Src/Dst IP.
    For the other nine days `cic2018_adapter.py:65,70` invents addresses from the
    row index — `f"192.168.10.{i % 250 + 1}"` — so a "host trajectory" is every
    250th row of a CSV. The PCAPs are stored one file per host with the address
    in the filename (`capDESKTOP-AN3U28N-172.31.64.111`), which is the real
    identity the graph needs.

  * **Missing packet features (PS 26153 §1b).** The PS names TTL variance, TCP
    window size, payload size distribution, port-scan signatures and
    retransmission counts, and says "the combination of both levels is
    required". None reach any model today. `telemetry/packet/pcap_engine.py`
    computes them and has never been wired to anything; this adapter is the wire.

Labelling is by published attack time window against packet UTC, not by joining
the CSVs. The CSV timestamps use a 12-hour clock with no AM/PM (verified across
300k sampled rows), so a join would be ambiguous; the offset `pcap_utc =
csv_local + 4:00:00` was verified to the second on two independent days.

Reads are streaming and bounded — these files are ~9 hours and hundreds of MB
each, and 5.9% of the corpus carries mid-file record corruption (see
`claude_latest_analysis/06_pcap_corruption_scan.md`), so parsing stops cleanly
at the first bad record and keeps the valid prefix rather than aborting a day.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_IP_RE = re.compile(r"((?:\d{1,3}\.){3}\d{1,3})")

# pcap magics; pcapng is detected and skipped with a clear message.
_PCAP_LE = 0xA1B2C3D4
_PCAP_LE_NS = 0xA1B23C4D
_PCAPNG = 0x0A0D0D0A


def host_ip_from_filename(path: str | Path) -> Optional[str]:
    """Extract the capture's host address.

    Handles every naming pattern present in the corpus:
      capDESKTOP-AN3U28N-172.31.64.111
      capEC2AMAZ-O4EL3NG-172.31.69.24
      UCAP172.31.69.25-part1.pcap
      capEC2AMAZ-O4EL3NG-172.31.69 - Copy.24   (fragment/copy names)
    """
    m = _IP_RE.search(Path(path).name)
    return m.group(1) if m else None


@dataclass(frozen=True)
class AttackWindow:
    """A published attack interval, in UTC."""

    name: str
    start_utc: float
    end_utc: float
    technique_ids: Tuple[str, ...] = ()
    coarse: str = "Unknown"

    def contains(self, ts: float) -> bool:
        return self.start_utc <= ts <= self.end_utc


# CSV local time -> packet UTC. Verified to the second on two independent days
# in claude_latest_analysis/05_cic2018_pcap_completeness.md.
CSV_TO_PCAP_UTC_OFFSET_SECONDS = 4 * 3600


def _utc(y: int, mo: int, d: int, h: int, mi: int) -> float:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def label_for(ts: float, windows: List[AttackWindow]) -> Tuple[bool, str, Tuple[str, ...]]:
    """(is_attack, coarse_category, technique_ids) for a packet timestamp."""
    for w in windows:
        if w.contains(ts):
            return True, w.coarse, w.technique_ids
    return False, "Benign", ()


class PcapFlowExtractor:
    """Stream a per-host capture into windowed flows plus packet features.

    Reuses the sensor's own `LiveFlowTable` and `LivePCAPEngine` rather than
    reimplementing them, so an offline-built feature vector is the same object
    the live path produces — the parity property `scripts/verify_offline_live_parity.py`
    checks.
    """

    def __init__(self, window_seconds: float = 2.0, max_packets: Optional[int] = None):
        self.window_seconds = window_seconds
        self.max_packets = max_packets

    def _iter_packets(self, path: Path) -> Iterator[Tuple[float, bytes]]:
        with open(path, "rb") as f:
            gh = f.read(24)
            if len(gh) < 24:
                return
            magic = struct.unpack("<I", gh[:4])[0]
            if magic == _PCAPNG:
                raise ValueError(f"{path.name} is pcapng; convert with editcap or use tshark")
            if magic in (_PCAP_LE, _PCAP_LE_NS):
                endian, nano = "<", magic == _PCAP_LE_NS
            else:
                be = struct.unpack(">I", gh[:4])[0]
                if be not in (_PCAP_LE, _PCAP_LE_NS):
                    raise ValueError(f"{path.name} is not a pcap file")
                endian, nano = ">", be == _PCAP_LE_NS

            n = 0
            while True:
                hdr = f.read(16)
                if len(hdr) < 16:
                    return
                sec, frac, incl, _orig = struct.unpack(endian + "IIII", hdr)
                # 5.9% of this corpus has a corrupt length field mid-file. Stop
                # cleanly and keep the valid prefix instead of aborting the day.
                if incl == 0 or incl > 262144:
                    return
                data = f.read(incl)
                if len(data) < incl:
                    return
                yield sec + (frac / 1e9 if nano else frac / 1e6), data
                n += 1
                if self.max_packets and n >= self.max_packets:
                    return

    def windows(self, path: str | Path) -> Iterator[Dict[str, Any]]:
        """Yield one dict per closed window: flows, packet features, host, time."""
        from telemetry.capture.sniffer import StreamingPacketSniffer
        from telemetry.flow.flow_table import LiveFlowTable
        from telemetry.packet.pcap_engine import LivePCAPEngine

        path = Path(path)
        host = host_ip_from_filename(path)
        table, engine = LiveFlowTable(), LivePCAPEngine()
        buf: List[Dict[str, Any]] = []
        w_start: Optional[float] = None
        wid = 0

        for ts, raw in self._iter_packets(path):
            if w_start is None:
                w_start = ts
            if ts - w_start >= self.window_seconds:
                yield {
                    "window_id": wid,
                    "host_ip": host,
                    "capture": path.name,
                    "window_start": w_start,
                    "window_end": ts,
                    "packet_count": len(buf),
                    "flows": table.snapshot_flows(max_flows=256),
                    # The 30 PS-named packet features, finally reaching a caller.
                    "packet_features": engine.extract_features(buf, self.window_seconds),
                }
                wid += 1
                buf, w_start = [], ts
                table, engine = LiveFlowTable(), LivePCAPEngine()

            pkt = StreamingPacketSniffer.parse_frame(raw, ts)
            if pkt:
                buf.append(pkt)
                table.process_packet(pkt)

        if buf and w_start is not None:
            yield {
                "window_id": wid,
                "host_ip": host,
                "capture": path.name,
                "window_start": w_start,
                "window_end": w_start + self.window_seconds,
                "packet_count": len(buf),
                "flows": table.snapshot_flows(max_flows=256),
                "packet_features": engine.extract_features(buf, self.window_seconds),
            }


def iter_day_captures(day_dir: str | Path, limit: Optional[int] = None) -> List[Path]:
    """Capture files for one day, newest-consistent order, skipping non-captures."""
    d = Path(day_dir)
    root = d / "pcap" if (d / "pcap").is_dir() else d
    files = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() not in (".lnk",) and host_ip_from_filename(p)
    ]
    return files[:limit] if limit else files
