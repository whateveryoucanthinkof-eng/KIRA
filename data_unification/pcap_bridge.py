"""Bridges data_unification/pcap_adapter.py's per-host PCAP captures into the
List[UnifiedFlowRecord] format data_unification/multi_dataset_stream.py's
HostTrajectoryExtractor consumes.

Two things pcap_adapter.PcapFlowExtractor.windows() doesn't give us directly:

  * Cross-host alignment. windows() anchors each file's windows to that
    file's own first-packet timestamp, so per-file window boundaries drift
    from a shared clock by up to one window. HostTrajectoryExtractor embeds
    one multi-host graph per window (TGNE needs every host active in that
    window, not a single host's edges -- claude_latest_analysis/
    07_v4_audit_and_migration_plan.md, finding E7), so a single PCAP file
    processed alone would starve it of neighbours. Windows here are instead
    bucketed on absolute UTC time (floor(ts / window_seconds)), so the same
    bucket index means the same wall-clock interval for every host.

  * Labels. PCAP packets carry no label of their own. attack_windows.py
    derives real attack intervals from the paired CIC-2018 CSV for the same
    day; this module attaches them per window through the same
    LabelResolver every other adapter uses (LabelSource.CIC2018), not the
    separate, unused AttackWindow/label_for in pcap_adapter.py.
"""

from __future__ import annotations

import heapq
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from data_unification.attack_windows import DerivedWindows
from data_unification.label_resolver import LabelResolver, get_default_resolver
from data_unification.pcap_adapter import PcapFlowExtractor, host_ip_from_filename, iter_day_captures
from data_unification.unified_schema import LabelSource, UnifiedFlowRecord

# _host_window_stream's 4-tuples are sorted on this element by construction
# (packets are read in file order, i.e. chronologically); heapq.merge relies on it.
_BUCKET_INDEX = 0


def _host_window_stream(
    path: Path,
    window_seconds: float,
    max_packets: Optional[int] = None,
) -> Iterator[Tuple[int, str, List[Dict[str, Any]], Dict[str, Any]]]:
    """Yields (bucket, host_ip, flows, packet_features) per absolute-UTC window for one host's capture."""
    from telemetry.capture.sniffer import StreamingPacketSniffer
    from telemetry.flow.flow_table import LiveFlowTable
    from telemetry.packet.pcap_engine import LivePCAPEngine

    host = host_ip_from_filename(path)
    if host is None:
        return

    extractor = PcapFlowExtractor(window_seconds=window_seconds, max_packets=max_packets)
    table, engine, buf = LiveFlowTable(), LivePCAPEngine(), []
    cur_bucket: Optional[int] = None

    # _iter_packets raises ValueError for a whole-file format problem (pcapng,
    # not-a-pcap) rather than tolerating it the way it tolerates mid-file
    # corruption. A handful of files in this corpus are pcapng despite the
    # .pcap name (verified: UCAP172.31.69.25-part1.pcap). One bad host must
    # not kill the k-way merge across all the other hosts in the same window.
    pkt_iter = extractor._iter_packets(path)
    while True:
        try:
            ts, raw = next(pkt_iter)
        except StopIteration:
            break
        except ValueError as e:
            import logging
            logging.getLogger(__name__).warning(f"skipping unreadable capture {path.name}: {e}")
            return
        bucket = int(ts // window_seconds)
        if cur_bucket is None:
            cur_bucket = bucket
        if bucket != cur_bucket:
            if buf:
                yield (
                    cur_bucket,
                    host,
                    table.snapshot_flows(max_flows=None),
                    engine.extract_features(buf, window_seconds),
                )
            table, engine, buf = LiveFlowTable(), LivePCAPEngine(), []
            cur_bucket = bucket
        pkt = StreamingPacketSniffer.parse_frame(raw, ts)
        if pkt:
            buf.append(pkt)
            table.process_packet(pkt)

    if buf:
        yield (
            cur_bucket,
            host,
            table.snapshot_flows(max_flows=None),
            engine.extract_features(buf, window_seconds),
        )


def iter_merged_day_windows(
    day_dir: str | Path,
    *,
    window_seconds: float = 2.0,
    limit_hosts: Optional[int] = None,
    max_packets_per_host: Optional[int] = None,
) -> Iterator[Tuple[int, Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]]]:
    """Yields (bucket, {host_ip: (flows, packet_features)}) for one capture day.

    All hosts are merged onto one absolute-UTC window grid via a lazy k-way
    merge (heapq.merge over one generator per host file) so a full day's
    ~445 hosts are never held in memory at once -- only one window's worth.
    """
    files = iter_day_captures(day_dir, limit=limit_hosts)
    streams = [_host_window_stream(p, window_seconds, max_packets_per_host) for p in files]
    merged = heapq.merge(*streams, key=lambda item: item[_BUCKET_INDEX])
    for bucket, group in groupby(merged, key=lambda item: item[_BUCKET_INDEX]):
        per_host = {host: (flows, feats) for _, host, flows, feats in group}
        yield bucket, per_host


def _flow_dict_to_record(
    f: Dict[str, Any],
    *,
    is_attack: bool,
    coarse_category: str,
    technique_ids: List[str],
    raw_label: str,
    scenario_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[UnifiedFlowRecord]:
    try:
        return UnifiedFlowRecord(
            src_ip=str(f.get("src_ip", "")),
            dst_ip=str(f.get("dst_ip", "")),
            src_port=int(f.get("src_port", 0)),
            dst_port=int(f.get("dst_port", 0)),
            protocol=int(f.get("protocol", 6)),
            start_time=float(f.get("start_time", 0.0)),
            end_time=float(f.get("end_time", 0.0)),
            fwd_bytes=int(f.get("fwd_bytes", 0)),
            bwd_bytes=int(f.get("bwd_bytes", 0)),
            fwd_packets=int(f.get("fwd_packets", 0)),
            bwd_packets=int(f.get("bwd_packets", 0)),
            raw_label=raw_label,
            raw_label_source=LabelSource.CIC2018.value,
            is_attack=is_attack,
            coarse_category=coarse_category,
            attck_technique_ids=list(technique_ids),
            # One SHARED dict per host-window, passed in by the caller. It was
            # a fresh literal per record, which the schema's own docstring
            # calls out as a 64 B/record cost, and it had nowhere to put the
            # packet features the caller had just computed and discarded.
            metadata=metadata if metadata is not None
            else {"scenario_id": scenario_id, "source": "pcap"},
        )
    except Exception:
        return None


def iter_day_records(
    day_dir: str | Path,
    attack_windows: DerivedWindows,
    *,
    scenario_id: str,
    window_seconds: float = 2.0,
    limit_hosts: Optional[int] = None,
    max_packets_per_host: Optional[int] = None,
    resolver: Optional[LabelResolver] = None,
    scope_labels_to_participants: bool = True,
    keep_packet_features: bool = True,
) -> Iterator[Tuple[float, float, List[UnifiedFlowRecord]]]:
    """Yields (window_start, window_end, records) for one capture day, all hosts active
    in that window bundled together -- ready for HostTrajectoryExtractor.extract_trajectories().

    attack_windows must come from attack_windows.derive_windows() on that same day's CIC-2018
    CSV; callers should check attack_windows.ok before relying on the labels (see that
    module's docstring -- an implausible derivation should not be trained on silently).

    `scope_labels_to_participants` restricts the attack label to the hosts the
    interval knows took part. Without it the label is a pure function of the
    clock, stamped on every one of a day's ~445 captured hosts -- so the target
    is mostly "what time is it", and a model can score well without ever
    looking at traffic. An interval with no known participants still labels
    everyone (there is nothing to narrow it with) and says so in
    `metadata["label_scope"]`, so a reader can tell the two cases apart.

    `keep_packet_features` attaches the 30 packet-level features the engine
    already computed for this host-window. They used to be unpacked into
    `_packet_features` and dropped; nothing downstream had ever seen one.
    """
    resolver = resolver or get_default_resolver()
    from data_unification.attack_windows import unscoped_label_policy
    from data_unification.label_resolver import UNKNOWN_CATEGORY
    unscoped_policy = unscoped_label_policy()

    for bucket, per_host in iter_merged_day_windows(
        day_dir,
        window_seconds=window_seconds,
        limit_hosts=limit_hosts,
        max_packets_per_host=max_packets_per_host,
    ):
        window_start = bucket * window_seconds
        window_end = window_start + window_seconds
        mid_ts = (window_start + window_end) / 2.0

        interval = attack_windows.interval_at(mid_ts)
        raw_label = interval.label if interval else "BENIGN"
        coarse, technique_ids, is_attack = resolver.resolve(raw_label, source=LabelSource.CIC2018)
        scoped = bool(interval and interval.scoped and scope_labels_to_participants)
        # An attack interval nobody can attribute: UNKNOWN by default (see
        # attack_windows.unscoped_label_policy), never "every host attacked".
        unknown = bool(is_attack and not scoped and scope_labels_to_participants
                       and unscoped_policy == "unknown")

        records: List[UnifiedFlowRecord] = []
        for host_ip, (flows, packet_features) in per_host.items():
            meta: Dict[str, Any] = {"scenario_id": scenario_id, "source": "pcap"}
            if keep_packet_features and packet_features:
                meta["packet_features"] = packet_features
            meta["label_scope"] = (
                "participants" if scoped else ("time_only" if is_attack else "benign")
            )
            for f in flows:
                # A window-level label is an interval in TIME; whether this
                # particular flow was in it is a separate question, and the
                # answer is the interval's attacking pairs (or, without pairs,
                # its participant set) when it has one.
                if unknown:
                    rec = _flow_dict_to_record(
                        f, is_attack=False, coarse_category=UNKNOWN_CATEGORY,
                        technique_ids=[], raw_label=raw_label,
                        scenario_id=scenario_id, metadata=meta)
                    if rec is not None:
                        records.append(rec)
                    continue
                if is_attack and scoped:
                    s_ip, d_ip = str(f.get("src_ip", "")), str(f.get("dst_ip", ""))
                    if interval.pairs:
                        hit = bool(interval.flow_hit(s_ip, d_ip))
                    else:
                        hit = interval.involves(s_ip, d_ip, host_ip)
                else:
                    hit = is_attack
                rec = _flow_dict_to_record(
                    f,
                    is_attack=hit,
                    coarse_category=coarse if hit else "Benign",
                    technique_ids=technique_ids if hit else [],
                    raw_label=raw_label if hit else "BENIGN",
                    scenario_id=scenario_id,
                    metadata=meta,
                )
                if rec is not None:
                    records.append(rec)

        if records:
            yield window_start, window_end, records
