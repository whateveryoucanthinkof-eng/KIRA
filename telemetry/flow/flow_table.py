#!/usr/bin/env python3
"""
telemetry/flow/flow_table.py
In-memory 5-tuple bidirectional flow tracking table.
Maintains active and new network flows incrementally and compiles the formal 42 continuous
flow telemetry features per 2.0-second time window.
Zero CSV or disk roundtrips.
"""

import math
from typing import Dict, List, Set, Tuple, Any
import numpy as np

FLOW_COLUMNS = [
    'new_flows_count', 'active_flows_count', 'total_fwd_pkts', 'total_bwd_pkts',
    'total_fwd_bytes', 'total_bwd_bytes', 'fwd_bwd_byte_ratio',
    'flow_byts_s_mean', 'flow_pkts_s_max', 'flow_pkts_s_mean', 'fwd_bwd_pkt_ratio',
    'pkt_len_mean', 'pkt_len_std', 'pkt_len_max', 'pkt_len_skew', 'fwd_act_data_pkts_mean',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'idle_mean', 'idle_std',
    'syn_count', 'ack_count', 'rst_count', 'psh_count', 'fin_count', 'urg_count',
    'syn_ack_ratio', 'rst_syn_ratio',
    'unique_dst_ports', 'dst_port_entropy', 'well_known_port_ratio',
    'max_pair_sequential_port_score', 'mean_pair_sequential_port_score', 'port_delta_std',
    'new_internal_edges_count', 'internal_fanout_entropy', 'lateral_privilege_ratio',
    'zero_bwd_pkts_ratio', 'small_probe_pkt_ratio',
    'tcp_ratio', 'udp_ratio'
]

ADMIN_PORTS = {445, 139, 3389, 88, 5985, 5986, 389, 464, 22}

def is_rfc1918(ip_str: str) -> bool:
    if not ip_str:
        return False
    if ip_str.startswith('10.') or ip_str.startswith('192.168.'):
        return True
    if ip_str.startswith('172.'):
        try:
            parts = ip_str.split('.')
            second = int(parts[1])
            return 16 <= second <= 31
        except Exception:
            return False
    return False

def calc_entropy(values: List[Any]) -> float:
    if not values or len(values) <= 1:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

class FlowRecord:
    def __init__(self, key: Tuple, first_ts: float):
        self.key = key  # canonical forward: (src, dst, sport, dport, proto)
        self.first_ts = first_ts
        self.last_ts = first_ts
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_act_data_pkts = 0
        self.pkt_lengths: List[int] = []
        self.packet_timestamps: List[float] = [first_ts]
        self.syn_count = 0
        self.ack_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.fin_count = 0
        self.urg_count = 0
        self.protocol = key[4]
        self.dst_port = key[3]

    def update(self, pkt: Dict[str, Any], is_forward: bool):
        ts = pkt["timestamp"]
        length = pkt["packet_length"]
        payload_len = pkt.get("payload_length", 0)
        self.last_ts = ts
        self.packet_timestamps.append(ts)
        self.pkt_lengths.append(length)

        if is_forward:
            self.fwd_pkts += 1
            self.fwd_bytes += length
            if payload_len > 0:
                self.fwd_act_data_pkts += 1
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += length

        flags = pkt.get("tcp_flags")
        if flags:
            if flags.get("SYN"): self.syn_count += 1
            if flags.get("ACK"): self.ack_count += 1
            if flags.get("RST"): self.rst_count += 1
            if flags.get("PSH"): self.psh_count += 1
            if flags.get("FIN"): self.fin_count += 1
            if flags.get("URG"): self.urg_count += 1

class LiveFlowTable:
    def snapshot_flows(self, max_flows: int = 256) -> List[Dict[str, Any]]:
        """Export active 5-tuple flows for downstream UnifiedFlowRecord ingestion."""
        out: List[Dict[str, Any]] = []
        for key, f in list(self.active_flows.items())[:max_flows]:
            src, dst, sport, dport, proto = key
            out.append({
                "src_ip": str(src),
                "dst_ip": str(dst),
                "src_port": int(sport),
                "dst_port": int(dport),
                "protocol": int(proto),
                "start_time": float(f.first_ts),
                "end_time": float(f.last_ts),
                "fwd_bytes": int(f.fwd_bytes),
                "bwd_bytes": int(f.bwd_bytes),
                "fwd_packets": int(f.fwd_pkts),
                "bwd_packets": int(f.bwd_pkts),
            })
        return out

    def __init__(self):
        # 5-tuple canonical map
        self.active_flows: Dict[Tuple, FlowRecord] = {}
        self.new_flows_this_window: Set[Tuple] = set()
        self.seen_internal_edges: Set[Tuple[str, str]] = set()

    def process_packet(self, pkt: Dict[str, Any]):
        src_ip = pkt["src_ip"]
        dst_ip = pkt["dst_ip"]
        sport = pkt["src_port"] or 0
        dport = pkt["dst_port"] or 0
        proto = pkt["protocol"]
        ts = pkt["timestamp"]

        fwd_key = (src_ip, dst_ip, sport, dport, proto)
        rev_key = (dst_ip, src_ip, dport, sport, proto)

        if fwd_key in self.active_flows:
            self.active_flows[fwd_key].update(pkt, is_forward=True)
        elif rev_key in self.active_flows:
            self.active_flows[rev_key].update(pkt, is_forward=False)
        else:
            # New flow
            flow = FlowRecord(fwd_key, ts)
            flow.update(pkt, is_forward=True)
            self.active_flows[fwd_key] = flow
            self.new_flows_this_window.add(fwd_key)

    def extract_window_features(self, window_sec: float = 2.0) -> Dict[str, float]:
        """Extracts the 42 formal continuous flow features for the elapsed 2s window."""
        new_flows_cnt = len(self.new_flows_this_window)
        active_flows_cnt = len(self.active_flows)

        if active_flows_cnt == 0:
            self.new_flows_this_window.clear()
            return {col: 0.0 for col in FLOW_COLUMNS}

        flows = list(self.active_flows.values())

        # Volume Dynamics
        tot_fwd_pkts = sum(f.fwd_pkts for f in flows)
        tot_bwd_pkts = sum(f.bwd_pkts for f in flows)
        tot_fwd_bytes = sum(f.fwd_bytes for f in flows)
        tot_bwd_bytes = sum(f.bwd_bytes for f in flows)

        fwd_bwd_byte_ratio = float(tot_fwd_bytes / (tot_bwd_bytes + 1e-5))
        fwd_bwd_pkt_ratio = float(tot_fwd_pkts / (tot_bwd_pkts + 1e-5))

        # Rates
        durations = [max(f.last_ts - f.first_ts, 0.001) for f in flows]
        flow_byts_s = [(f.fwd_bytes + f.bwd_bytes) / d for f, d in zip(flows, durations)]
        flow_pkts_s = [(f.fwd_pkts + f.bwd_pkts) / d for f, d in zip(flows, durations)]

        flow_byts_s_mean = float(np.mean(flow_byts_s))
        flow_pkts_s_max = float(np.max(flow_pkts_s))
        flow_pkts_s_mean = float(np.mean(flow_pkts_s))

        # Packet Length Moments
        all_lengths = []
        for f in flows:
            all_lengths.extend(f.pkt_lengths)

        if all_lengths:
            arr_lengths = np.array(all_lengths, dtype=np.float64)
            pkt_len_mean = float(np.mean(arr_lengths))
            pkt_len_std = float(np.std(arr_lengths))
            pkt_len_max = float(np.max(arr_lengths))
            # Skewness
            if len(arr_lengths) > 2 and pkt_len_std > 1e-5:
                pkt_len_skew = float(np.mean(((arr_lengths - pkt_len_mean) / pkt_len_std) ** 3))
            else:
                pkt_len_skew = 0.0
        else:
            pkt_len_mean, pkt_len_std, pkt_len_max, pkt_len_skew = 0.0, 0.0, 0.0, 0.0

        fwd_act_data_pkts_mean = float(np.mean([f.fwd_act_data_pkts for f in flows]))

        # Inter-Arrival Times (IAT)
        all_iats = []
        for f in flows:
            if len(f.packet_timestamps) > 1:
                all_iats.extend(np.diff(sorted(f.packet_timestamps)))

        if all_iats:
            arr_iats = np.array(all_iats, dtype=np.float64)
            flow_iat_mean = float(np.mean(arr_iats))
            flow_iat_std = float(np.std(arr_iats))
            flow_iat_max = float(np.max(arr_iats))
        else:
            flow_iat_mean, flow_iat_std, flow_iat_max = 0.0, 0.0, 0.0

        # Idle Statistics
        idle_times = [max(window_sec - (f.last_ts - f.first_ts), 0.0) for f in flows]
        idle_mean = float(np.mean(idle_times))
        idle_std = float(np.std(idle_times))

        # TCP Flag Counts & Ratios
        syn_cnt = sum(f.syn_count for f in flows)
        ack_cnt = sum(f.ack_count for f in flows)
        rst_cnt = sum(f.rst_count for f in flows)
        psh_cnt = sum(f.psh_count for f in flows)
        fin_cnt = sum(f.fin_count for f in flows)
        urg_cnt = sum(f.urg_count for f in flows)

        syn_ack_ratio = float(syn_cnt / (ack_cnt + 1e-5))
        rst_syn_ratio = float(rst_cnt / (syn_cnt + 1e-5))

        # Port & Graph Dynamics
        dst_ports = [f.dst_port for f in flows if f.dst_port > 0]
        unique_dst_ports = float(len(set(dst_ports)))
        dst_port_entropy = calc_entropy(dst_ports)
        well_known_cnt = sum(1 for p in dst_ports if p < 1024)
        well_known_port_ratio = float(well_known_cnt / len(dst_ports)) if dst_ports else 0.0

        # Sequential Port Scan Scores
        if len(dst_ports) > 1:
            sorted_ports = np.sort(dst_ports)
            deltas = np.diff(sorted_ports)
            port_delta_std = float(np.std(deltas))
            seq_scores = [1.0 if d == 1 else 0.0 for d in deltas]
            max_pair_seq = float(np.max(seq_scores)) if seq_scores else 0.0
            mean_pair_seq = float(np.mean(seq_scores)) if seq_scores else 0.0
        else:
            port_delta_std, max_pair_seq, mean_pair_seq = 0.0, 0.0, 0.0

        # RFC-1918 East-West Graph Intelligence
        new_internal_edges = 0
        internal_dsts = []
        lateral_priv_cnt = 0
        internal_flow_cnt = 0

        for f in flows:
            src_ip, dst_ip = f.key[0], f.key[1]
            if is_rfc1918(src_ip) and is_rfc1918(dst_ip):
                internal_flow_cnt += 1
                edge = (src_ip, dst_ip)
                if edge not in self.seen_internal_edges:
                    new_internal_edges += 1
                    self.seen_internal_edges.add(edge)
                internal_dsts.append(dst_ip)
                if f.dst_port in ADMIN_PORTS:
                    lateral_priv_cnt += 1

        new_internal_edges_count = float(new_internal_edges)
        internal_fanout_entropy = calc_entropy(internal_dsts)
        lateral_privilege_ratio = float(lateral_priv_cnt / internal_flow_cnt) if internal_flow_cnt > 0 else 0.0

        # Probes & Protocol Split
        zero_bwd_flows = sum(1 for f in flows if f.bwd_pkts == 0)
        zero_bwd_pkts_ratio = float(zero_bwd_flows / active_flows_cnt)

        small_probe_flows = sum(1 for f in flows if f.pkt_lengths and np.mean(f.pkt_lengths) < 100)
        small_probe_pkt_ratio = float(small_probe_flows / active_flows_cnt)

        tcp_flows = sum(1 for f in flows if f.protocol == 6)
        udp_flows = sum(1 for f in flows if f.protocol == 17)
        tcp_ratio = float(tcp_flows / active_flows_cnt)
        udp_ratio = float(udp_flows / active_flows_cnt)

        # Clear new flows tracker for next window
        self.new_flows_this_window.clear()

        # Prune old inactive flows (inactive for > 30 seconds)
        curr_ts = max(f.last_ts for f in flows)
        expired_keys = [k for k, f in self.active_flows.items() if curr_ts - f.last_ts > 30.0]
        for k in expired_keys:
            del self.active_flows[k]

        return {
            'new_flows_count': float(new_flows_cnt),
            'active_flows_count': float(active_flows_cnt),
            'total_fwd_pkts': float(tot_fwd_pkts),
            'total_bwd_pkts': float(tot_bwd_pkts),
            'total_fwd_bytes': float(tot_fwd_bytes),
            'total_bwd_bytes': float(tot_bwd_bytes),
            'fwd_bwd_byte_ratio': fwd_bwd_byte_ratio,
            'flow_byts_s_mean': flow_byts_s_mean,
            'flow_pkts_s_max': flow_pkts_s_max,
            'flow_pkts_s_mean': flow_pkts_s_mean,
            'fwd_bwd_pkt_ratio': fwd_bwd_pkt_ratio,
            'pkt_len_mean': pkt_len_mean,
            'pkt_len_std': pkt_len_std,
            'pkt_len_max': pkt_len_max,
            'pkt_len_skew': pkt_len_skew,
            'fwd_act_data_pkts_mean': fwd_act_data_pkts_mean,
            'flow_iat_mean': flow_iat_mean,
            'flow_iat_std': flow_iat_std,
            'flow_iat_max': flow_iat_max,
            'idle_mean': idle_mean,
            'idle_std': idle_std,
            'syn_count': float(syn_cnt),
            'ack_count': float(ack_cnt),
            'rst_count': float(rst_cnt),
            'psh_count': float(psh_cnt),
            'fin_count': float(fin_cnt),
            'urg_count': float(urg_cnt),
            'syn_ack_ratio': syn_ack_ratio,
            'rst_syn_ratio': rst_syn_ratio,
            'unique_dst_ports': unique_dst_ports,
            'dst_port_entropy': dst_port_entropy,
            'well_known_port_ratio': well_known_port_ratio,
            'max_pair_sequential_port_score': max_pair_seq,
            'mean_pair_sequential_port_score': mean_pair_seq,
            'port_delta_std': port_delta_std,
            'new_internal_edges_count': new_internal_edges_count,
            'internal_fanout_entropy': internal_fanout_entropy,
            'lateral_privilege_ratio': lateral_privilege_ratio,
            'zero_bwd_pkts_ratio': zero_bwd_pkts_ratio,
            'small_probe_pkt_ratio': small_probe_pkt_ratio,
            'tcp_ratio': tcp_ratio,
            'udp_ratio': udp_ratio
        }
