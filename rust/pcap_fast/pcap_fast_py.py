"""Python side of pcap_fast. data_unification.parallel_ingest uses it by default
for PCAP2018 days (CYBERWORLD_PCAP_PARSER=python opts out).

Two entry points, each a drop-in for one stage of the reference path:

  iter_merged_day_windows_fast(day_dir, ...)
      same yields as data_unification.pcap_bridge.iter_merged_day_windows:
      (bucket, {host: (flow dicts, packet-feature dict)}). Builds Python dicts,
      so it is only for verification, not speed.

  parse_pcap_day_columns_fast(day_dir, label_dir, window_seconds)
      the columns data_unification.parallel_ingest._parse_one produces for a
      PCAP2018 capture (u, i, ts, lbl, edge + ips/cats vocabularies), computed
      vectorised from the Rust binary's columnar output. Labels use the same
      derive_windows / check_label_day / LabelResolver calls as
      iter_pcap_day_windows + iter_day_records; the 12 edge features use the
      same numpy float64 ops as tgne_features.extract_canonical_edge_features
      (numpy's log1p is SVML on AVX-512 and is NOT glibc's, so it cannot move
      into Rust without porting SVML).
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
#: Built by `cargo build --release` in this directory (scripts/run_training_plan.sh
#: preflight does it). PCAP_FAST_BIN overrides.
DEFAULT_BIN = os.environ.get("PCAP_FAST_BIN", str(HERE / "target" / "release" / "pcap_fast"))

FEATURE_COLUMNS = [
    'ttl_mean', 'ttl_std', 'ttl_unique_count', 'tcp_window_mean', 'tcp_window_std',
    'tcp_retransmission_estimate', 'tcp_retransmission_ratio', 'pkt_iat_mean', 'pkt_iat_std',
    'pkt_iat_cv', 'payload_size_mean', 'payload_size_std', 'payload_zero_ratio',
    'unique_dst_ips_pcap', 'vertical_scan_score', 'horizontal_scan_score', 'syn_only_ratio',
    'syn_ack_response_ratio', 'syn_no_response_ratio', 'rst_after_syn_ratio',
    'tcp_handshake_completion_ratio', 'http_request_count', 'http_uri_entropy_mean',
    'http_special_char_ratio', 'dns_query_count', 'dns_unique_domains',
    'dns_domain_entropy_mean', 'unique_internal_destinations_pcap',
    'unique_external_destinations_pcap', 'internal_fanout_entropy_pcap',
]

WIN_DT = np.dtype([("bucket", "<i8"), ("n_hw", "<u4")])
HW_DT = np.dtype([("host", "<u4"), ("n_flows", "<u4"), ("feats", "<f8", (30,))])
FLOW_DT = np.dtype([("src", "<u4"), ("dst", "<u4"), ("sport", "<u4"), ("dport", "<u4"),
                    ("proto", "<u4"), ("start", "<f8"), ("end", "<f8"),
                    ("fwd_bytes", "<u8"), ("bwd_bytes", "<u8"),
                    ("fwd_pkts", "<u8"), ("bwd_pkts", "<u8")])


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"pcap_fast stream truncated ({len(b)}/{n} bytes)")
    return b


def iter_chunks(paths: List[Path], hosts: List[str], window_seconds: float = 2.0,
                max_packets_per_host: Optional[int] = None, binary: str = DEFAULT_BIN,
                stats: Optional[dict] = None):
    """Run the Rust binary; yield (strings_so_far, windows, hostwins, flows) per chunk."""
    cmd = [binary, repr(float(window_seconds)), str(int(max_packets_per_host or 0))]
    for h, p in zip(hosts, paths):
        cmd += [h, str(p)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)
    strings: List[str] = []
    f = proc.stdout
    try:
        while True:
            magic = _read_exact(f, 4)
            if magic == b"PFEN":
                nbytes, npk = struct.unpack("<QQ", _read_exact(f, 16))
                if stats is not None:
                    stats.update(bytes=nbytes, packets=npk)
                break
            if magic != b"PFC1":
                raise ValueError(f"bad chunk magic {magic!r}")
            (ns,) = struct.unpack("<I", _read_exact(f, 4))
            for _ in range(ns):
                (ln,) = struct.unpack("<H", _read_exact(f, 2))
                strings.append(_read_exact(f, ln).decode("ascii"))
            (nw,) = struct.unpack("<I", _read_exact(f, 4))
            wins = np.frombuffer(_read_exact(f, nw * WIN_DT.itemsize), dtype=WIN_DT)
            (nh,) = struct.unpack("<I", _read_exact(f, 4))
            hws = np.frombuffer(_read_exact(f, nh * HW_DT.itemsize), dtype=HW_DT)
            (nf,) = struct.unpack("<I", _read_exact(f, 4))
            flows = np.frombuffer(_read_exact(f, nf * FLOW_DT.itemsize), dtype=FLOW_DT)
            yield strings, wins, hws, flows
    finally:
        f.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"pcap_fast exited {rc}")


def _captures(day_dir, limit_hosts=None):
    from data_unification.pcap_adapter import host_ip_from_filename, iter_day_captures
    files = iter_day_captures(day_dir, limit=limit_hosts)
    # _host_window_stream returns early for a None host; iter_day_captures
    # already filters those out, so this is belt and braces.
    files = [p for p in files if host_ip_from_filename(p) is not None]
    return files, [host_ip_from_filename(p) for p in files]


# ------------------------------------------------------------------ stage 1
def iter_merged_day_windows_fast(day_dir, *, window_seconds: float = 2.0,
                                 limit_hosts: Optional[int] = None,
                                 max_packets_per_host: Optional[int] = None,
                                 binary: str = DEFAULT_BIN):
    files, hosts = _captures(day_dir, limit_hosts)
    for strings, wins, hws, flows in iter_chunks(files, hosts, window_seconds,
                                                 max_packets_per_host, binary):
        h = f = 0
        fl = flows.tolist()
        for bucket, n_hw in wins.tolist():
            per_host = {}
            for host_idx, n_flows, feats in hws[h:h + n_hw].tolist():
                out = []
                for (s, d, sp, dp, pr, st, en, fb, bb, fp, bp) in fl[f:f + n_flows]:
                    out.append({"src_ip": strings[s], "dst_ip": strings[d], "src_port": sp,
                                "dst_port": dp, "protocol": pr, "start_time": st,
                                "end_time": en, "fwd_bytes": fb, "bwd_bytes": bb,
                                "fwd_packets": fp, "bwd_packets": bp})
                f += n_flows
                per_host[hosts[host_idx]] = (out, dict(zip(FEATURE_COLUMNS, feats)))
            h += n_hw
            yield bucket, per_host


# ------------------------------------------------------------------ stage 2
def _edge_features(fb, bb, fp, bp, start, end, proto, dport) -> np.ndarray:
    """extract_canonical_edge_features over UnifiedFlowRecord properties, vectorised.

    Every step is the same float64 IEEE op the scalar code performs, in the
    same order; see EXPERIMENT.md for the equivalence argument per feature.
    """
    end = np.where(start > end, start, end)            # __post_init__ end_time fix
    fbf, bbf = fb.astype(np.float64), bb.astype(np.float64)
    dur = np.maximum(0.0, end - start)                 # .duration
    tb = (fb + bb).astype(np.float64)                  # total_bytes (int) -> float
    tp = (fp + bp).astype(np.float64)
    pos = dur > 0.0
    safe = np.where(pos, dur, 1.0)
    byte_rate = np.where(pos, tb / safe, tb)
    pkt_rate = np.where(pos, tp / safe, tp)
    n = len(fb)
    feat = np.zeros((n, 12), dtype=np.float32)
    feat[:, 0] = np.log1p(np.maximum(0.0, fbf)).astype(np.float32)
    feat[:, 1] = np.log1p(np.maximum(0.0, bbf)).astype(np.float32)
    feat[:, 2] = np.log1p(np.maximum(0.0, fp.astype(np.float64))).astype(np.float32)
    feat[:, 3] = np.log1p(np.maximum(0.0, bp.astype(np.float64))).astype(np.float32)
    feat[:, 4] = (np.minimum(np.maximum(0.0, dur), 300.0) / 300.0).astype(np.float32)
    feat[:, 5] = np.minimum(1.0, np.log1p(np.maximum(0.0, byte_rate)) / 20.0).astype(np.float32)
    feat[:, 6] = np.minimum(1.0, np.log1p(np.maximum(0.0, pkt_rate)) / 10.0).astype(np.float32)
    feat[:, 7] = (proto == 6)
    feat[:, 8] = (proto == 17)
    feat[:, 9] = (proto == 1)
    dp = np.minimum(65535, np.maximum(0, dport.astype(np.int64))).astype(np.float64)
    feat[:, 10] = (np.log1p(dp) / np.log1p(65535.0)).astype(np.float32)
    tot = np.maximum(0.0, fbf) + np.maximum(0.0, bbf)
    feat[:, 11] = ((fbf - bbf) / (tot + 1e-5)).astype(np.float32)
    from data_unification.tgne_features import ablation_mask
    m = ablation_mask()
    if m is not None:
        feat *= m
    return feat


class DayColumnBuilder:
    """Turns the Rust chunk stream of one day into _parse_one's columns.

    Holds the day's LOCAL vocabularies (ips, cats) across chunks, so chunks
    can be written out as they arrive. Every choice mirrors the reference
    record loop in parallel_ingest._parse_one: ips in src-then-dst encounter
    order, cats in record encounter order, resolver.resolve called exactly
    once per merged window (so the coverage counters move identically).
    """

    def __init__(self, dw, hosts: List[str], window_seconds: float, resolver):
        from data_unification.unified_schema import LabelSource
        self.dw, self.hosts, self.w = dw, hosts, float(window_seconds)
        self.resolver, self._src = resolver, LabelSource.CIC2018
        self.remap = np.full(0, -1, dtype=np.int64)   # rust string id -> local id
        self.ips: List[str] = []
        self.cat_local: Dict[str, int] = {}
        self._member: Dict[int, np.ndarray] = {}      # id(interval) -> bool per rust string id
        self._pairs: Dict[int, tuple] = {}            # id(interval) -> (participant idx per string, pair matrix)
        from data_unification.attack_windows import unscoped_label_policy
        self._policy = unscoped_label_policy()
        from data_unification.label_resolver import UNKNOWN_CATEGORY
        self._unknown = UNKNOWN_CATEGORY
        self.source = LabelSource.CIC2018.value
        self.n = 0

    @property
    def cats(self) -> List[str]:
        return list(self.cat_local)

    def chunk(self, strings, wins, hws, flows):
        """(u, i, ts, lbl, edge) for one Rust chunk; None if it has no flows."""
        if len(self.remap) < len(strings):
            self.remap = np.concatenate(
                [self.remap, np.full(len(strings) - len(self.remap), -1, np.int64)])
        nf = len(flows)
        src = flows["src"].astype(np.int64)
        dst = flows["dst"].astype(np.int64)
        cat_local, hosts, w = self.cat_local, self.hosts, self.w

        # ---- labels, one window at a time (iter_day_records). resolve() runs
        # for EVERY window, flows or not, exactly as the reference does.
        lbl = np.empty(nf, dtype=np.int32)
        hw_flow_start = np.concatenate([[0], np.cumsum(hws["n_flows"].astype(np.int64))])
        hw_host = hws["host"].tolist()
        h = 0
        for bucket, n_hw in wins.tolist():
            a, b = int(hw_flow_start[h]), int(hw_flow_start[h + n_hw])
            window_start = bucket * w
            window_end = window_start + w
            mid_ts = (window_start + window_end) / 2.0
            interval = self.dw.interval_at(mid_ts)
            raw_label = interval.label if interval else "BENIGN"
            coarse, _tids, is_attack = self.resolver.resolve(raw_label, source=self._src)
            if b == a:                      # a window whose host-windows hold no flows
                h += n_hw
                continue
            scoped = bool(interval and interval.scoped)
            if is_attack and not scoped and self._policy == "unknown":
                # attack_windows.unscoped_label_policy: nobody can say which
                # flows were the attack, so none is called one (or benign).
                coarse = self._unknown
            if is_attack and scoped and interval.pairs:
                # Conversation-level: the flow's {src, dst} must be an attacking pair.
                key = id(interval)
                pm = self._pairs.get(key)
                if pm is None or len(pm[0]) < len(strings):
                    who = sorted({ip for pr in interval.pairs for ip in pr})
                    at = {ip: k for k, ip in enumerate(who)}
                    pidx = np.fromiter((at.get(st, -1) for st in strings),
                                       dtype=np.int64, count=len(strings))
                    mat = np.zeros((len(who) + 1, len(who) + 1), dtype=bool)
                    for pr in interval.pairs:
                        x, y = (tuple(pr) * 2)[:2]
                        mat[at[x], at[y]] = mat[at[y], at[x]] = True
                    pm = self._pairs[key] = (pidx, mat)
                pidx, mat = pm
                hit = mat[pidx[src[a:b]], pidx[dst[a:b]]]   # -1 indexes the all-False last row/col
            elif is_attack and scoped:
                key = id(interval)
                mem = self._member.get(key)
                if mem is None or len(mem) < len(strings):
                    mem = np.fromiter((s in interval.participants for s in strings),
                                      dtype=bool, count=len(strings))
                    self._member[key] = mem
                hit = mem[src[a:b]] | mem[dst[a:b]]
                for k in range(n_hw):
                    if hosts[hw_host[h + k]] in interval.participants:
                        s0 = int(hw_flow_start[h + k]) - a
                        s1 = int(hw_flow_start[h + k + 1]) - a
                        hit[s0:s1] = True
            else:
                hit = np.full(b - a, bool(is_attack))
            # cat_local encounter order: whichever of the two values comes first.
            first, other = (coarse, "Benign") if hit[0] else ("Benign", coarse)
            if first not in cat_local:
                cat_local[first] = len(cat_local)
            mixed = bool((hit != hit[0]).any())
            if mixed and other not in cat_local:
                cat_local[other] = len(cat_local)
            if mixed:
                lbl[a:b] = np.where(hit, cat_local[coarse], cat_local["Benign"])
            else:
                lbl[a:b] = cat_local[first]
            h += n_hw
        if nf == 0:
            return None

        # ---- local ip ids in encounter order: src then dst, record by record
        remap = self.remap
        seq = np.empty(2 * nf, dtype=np.int64)
        seq[0::2] = src
        seq[1::2] = dst
        new = seq[remap[seq] < 0]
        if len(new):
            uniq, first_at = np.unique(new, return_index=True)
            order = uniq[np.argsort(first_at, kind="stable")]
            remap[order] = np.arange(len(self.ips), len(self.ips) + len(order))
            self.ips.extend(strings[k] for k in order.tolist())
        u = remap[src].astype(np.int32)
        i = remap[dst].astype(np.int32)

        edge = _edge_features(flows["fwd_bytes"].astype(np.int64), flows["bwd_bytes"].astype(np.int64),
                              flows["fwd_pkts"].astype(np.int64), flows["bwd_pkts"].astype(np.int64),
                              flows["start"], flows["end"], flows["proto"].astype(np.int64),
                              flows["dport"])
        self.n += nf
        return u, i, flows["start"].astype(np.float64), lbl, edge


def load_day_labels(day_dir, label_dir):
    """(dw, day) exactly as iter_pcap_day_windows derives them, or None where
    it would skip the day (no label CSV, implausible windows)."""
    from data_unification.attack_windows import derive_windows
    from data_unification.training_sources import (_PCAP_DAY, _pcap_label_csv,
                                                   check_label_day)
    day_dir = Path(day_dir)
    csv_path = _pcap_label_csv(day_dir, label_dir)
    if csv_path is None:
        print(f"skipping {day_dir.name}: no label CSV under {label_dir}", flush=True)
        return None
    dw = derive_windows(str(csv_path))
    if not dw.ok:
        print(f"skipping {day_dir.name}: implausible attack windows ({dw.evidence})", flush=True)
        return None
    day = _PCAP_DAY.match(day_dir.name).group("day")
    check_label_day(day, dw.intervals)
    # Scope intervals the CSV could not (no IP columns) from the participant
    # map. This had no caller, so a filled-in map changed nothing.
    from data_unification.attack_participants import apply_participants
    apply_participants(dw, day)
    return dw, day


def iter_day_column_chunks(day_dir, dw, window_seconds: float = 2.0,
                           binary: str = DEFAULT_BIN, stats: Optional[dict] = None):
    """(builder, chunk-iterator) for one day; the builder's ips/cats/n are
    final once the iterator is exhausted."""
    from data_unification.label_resolver import get_default_resolver
    files, hosts = _captures(day_dir)
    b = DayColumnBuilder(dw, hosts, window_seconds, get_default_resolver())

    def gen():
        for strings, wins, hws, flows in iter_chunks(files, hosts, window_seconds, None,
                                                     binary, stats):
            cols = b.chunk(strings, wins, hws, flows)
            if cols is not None:
                yield cols
    return b, gen()


def parse_pcap_day_columns_fast(day_dir, label_dir, window_seconds: float = 2.0,
                                binary: str = DEFAULT_BIN, stats: Optional[dict] = None,
                                keep_columns: bool = True) -> Optional[dict]:
    """_parse_one("PCAP2018", day_dir, ...) columns, concatenated (not split in parts)."""
    got = load_day_labels(day_dir, label_dir)
    if got is None:
        return None
    b, it = iter_day_column_chunks(day_dir, got[0], window_seconds, binary, stats)
    cols = {"u": [], "i": [], "ts": [], "lbl": [], "edge": []}
    for chunk in it:
        if keep_columns:
            for k, v in zip(("u", "i", "ts", "lbl", "edge"), chunk):
                cols[k].append(v)
    if not b.n:
        return {"n": 0}
    out = {"n": b.n, "source": b.source, "ips": b.ips, "cats": b.cats}
    if keep_columns:
        out.update({k: np.concatenate(v) for k, v in cols.items()})
    return out
