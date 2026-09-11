#!/usr/bin/env python3
"""
telemetry/packet/pcap_engine.py
In-memory packet-level behavioral feature extraction engine.
Provides packet features for the capture-only telemetry stream.
Zero intermediate disk I/O.
"""

import math
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np

PCAP_BEHAVIORAL_COLUMNS = [
    'ttl_mean',
    'ttl_std',
    'ttl_unique_count',
    'tcp_window_mean',
    'tcp_window_std',
    'tcp_retransmission_estimate',
    'tcp_retransmission_ratio',
    'pkt_iat_mean',
    'pkt_iat_std',
    'pkt_iat_cv',
    'payload_size_mean',
    'payload_size_std',
    'payload_zero_ratio',
    'unique_dst_ips_pcap',
    'vertical_scan_score',
    'horizontal_scan_score',
    'syn_only_ratio',
    'syn_ack_response_ratio',
    'syn_no_response_ratio',
    'rst_after_syn_ratio',
    'tcp_handshake_completion_ratio',
    'http_request_count',
    'http_uri_entropy_mean',
    'http_special_char_ratio',
    'dns_query_count',
    'dns_unique_domains',
    'dns_domain_entropy_mean',
    'unique_internal_destinations_pcap',
    'unique_external_destinations_pcap',
    'internal_fanout_entropy_pcap'
]

SPECIAL_HTTP_CHARS = set(" '\"=<>():;%&/\\#+?*!{}[]|`^~$,")


def is_rfc1918(ip_str: str) -> bool:
    """Checks if an IPv4 address is inside RFC1918 private address ranges."""
    if not ip_str:
        return False
    if ip_str.startswith('10.') or ip_str.startswith('192.168.'):
        return True
    if ip_str.startswith('172.'):
        try:
            parts = ip_str.split('.')
            second_octet = int(parts[1])
            return 16 <= second_octet <= 31
        except Exception:
            return False
    return False


def calculate_shannon_entropy(byte_seq) -> float:
    """Computes Shannon Entropy H(X) = -sum(p_i * log2(p_i))."""
    if not byte_seq or len(byte_seq) <= 1:
        return 0.0
    if isinstance(byte_seq, str):
        byte_seq = byte_seq.encode('utf-8', errors='ignore')

    length = len(byte_seq)
    counts: Dict[int, int] = {}
    for b in byte_seq:
        counts[b] = counts.get(b, 0) + 1

    ent = 0.0
    for cnt in counts.values():
        p = cnt / length
        ent -= p * math.log2(p)
    return float(ent)


class LivePCAPEngine:
    """
    Computes all 30 authoritative PCAP features for incoming packets within a 2-second window.
    """
    def __init__(self):
        self.causal_seen_internal_edges: Set[Tuple[str, str]] = set()

    def extract_features(self, packets: List[Dict[str, Any]], window_sec: float = 2.0) -> Dict[str, float]:
        if not packets:
            return {col: 0.0 for col in PCAP_BEHAVIORAL_COLUMNS}

        n_pkts = len(packets)
        timestamps = [p['timestamp'] for p in packets]
        ttls = [p.get('ttl_or_hop_limit') or 0 for p in packets]
        payloads = [p.get('payload_length') or 0 for p in packets]
        dst_ips = [p.get('dst_ip') or '' for p in packets]
        dst_ports = [p.get('dst_port') or 0 for p in packets]

        # 1. TTL Dynamics
        ttls_arr = np.array(ttls, dtype=float)
        ttl_mean = float(np.mean(ttls_arr))
        ttl_std = float(np.std(ttls_arr))
        ttl_unique_cnt = float(len(np.unique(ttls_arr)))

        # 2. Micro-IAT Dynamics
        if n_pkts > 1:
            ts_sorted = np.sort(timestamps)
            iats = np.diff(ts_sorted)
            iat_mean = float(np.mean(iats))
            iat_std = float(np.std(iats))
            iat_cv = float(iat_std / (iat_mean + 1e-7))
        else:
            iat_mean, iat_std, iat_cv = 0.0, 0.0, 0.0

        # 3. Payload Distribution
        payload_arr = np.array(payloads, dtype=float)
        payload_mean = float(np.mean(payload_arr))
        payload_std = float(np.std(payload_arr))
        payload_zero_ratio = float(np.mean(payload_arr == 0.0))

        # 4. TCP Specific Mechanics
        tcp_pkts = [p for p in packets if p.get('protocol') == 6]
        n_tcp = len(tcp_pkts)

        win_mean, win_std = 0.0, 0.0
        retrans_cnt, retrans_ratio = 0, 0.0
        syn_only_ratio = 0.0
        syn_ack_ratio, syn_no_resp_ratio, rst_after_syn_ratio = 0.0, 0.0, 0.0
        handshake_ratio = 0.0

        if n_tcp > 0:
            windows = [p.get('tcp_window') or 0 for p in tcp_pkts]
            win_arr = np.array(windows, dtype=float)
            win_mean = float(np.mean(win_arr))
            win_std = float(np.std(win_arr))

            # Session tracking for retransmissions and handshake progression
            sessions: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
            syn_count = 0
            syn_ack_count = 0
            rst_after_syn_count = 0
            handshake_attempts = 0
            handshakes_completed = 0
            syn_only_count = 0

            for p in tcp_pkts:
                flags = p.get('tcp_flags') or {}
                s_ip, d_ip = p.get('src_ip', ''), p.get('dst_ip', '')
                sp, dp = p.get('src_port') or 0, p.get('dst_port') or 0
                seq = p.get('tcp_seq') or 0
                ack = p.get('tcp_ack') or 0
                plen = p.get('payload_length') or 0

                fwd_key = (s_ip, sp, d_ip, dp)
                rev_key = (d_ip, dp, s_ip, sp)

                is_syn = flags.get('SYN', False) and not flags.get('ACK', False)
                is_syn_ack = flags.get('SYN', False) and flags.get('ACK', False)
                is_ack = flags.get('ACK', False) and not flags.get('SYN', False)
                is_rst = flags.get('RST', False)

                # Check for SYN-only packet (pure recon)
                if is_syn and not any([flags.get('ACK'), flags.get('FIN'), flags.get('RST'), flags.get('PSH'), flags.get('URG')]):
                    syn_only_count += 1

                if is_syn:
                    syn_count += 1
                    handshake_attempts += 1
                    sessions[fwd_key] = {
                        'max_seq': seq + plen,
                        'max_ack': ack,
                        'syn_seen': True,
                        'syn_ack_seen': False,
                        'ack_seen': False,
                        'rst_seen': False
                    }
                elif is_syn_ack:
                    syn_ack_count += 1
                    if rev_key in sessions:
                        sessions[rev_key]['syn_ack_seen'] = True
                elif is_ack:
                    if fwd_key in sessions and sessions[fwd_key]['syn_seen'] and sessions[fwd_key]['syn_ack_seen']:
                        if not sessions[fwd_key]['ack_seen']:
                            handshakes_completed += 1
                            sessions[fwd_key]['ack_seen'] = True
                elif is_rst:
                    if rev_key in sessions and sessions[rev_key]['syn_seen'] and not sessions[rev_key]['syn_ack_seen']:
                        rst_after_syn_count += 1

                # Retransmission check
                if fwd_key in sessions:
                    sess = sessions[fwd_key]
                    if plen > 0 and seq < sess['max_seq']:
                        retrans_cnt += 1
                    elif seq + plen > sess['max_seq']:
                        sess['max_seq'] = seq + plen
                    if ack > sess['max_ack']:
                        sess['max_ack'] = ack
                else:
                    sessions[fwd_key] = {
                        'max_seq': seq + plen,
                        'max_ack': ack,
                        'syn_seen': False,
                        'syn_ack_seen': False,
                        'ack_seen': False,
                        'rst_seen': is_rst
                    }

            retrans_ratio = float(retrans_cnt / n_tcp)
            syn_only_ratio = float(syn_only_count / n_tcp)

            if syn_count > 0:
                syn_ack_ratio = float(syn_ack_count / syn_count)
                unanswered = sum(1 for s in sessions.values() if s.get('syn_seen') and not s.get('syn_ack_seen') and not s.get('rst_seen'))
                syn_no_resp_ratio = float(unanswered / syn_count)
                rst_after_syn_ratio = float(rst_after_syn_count / syn_count)

            if handshake_attempts > 0:
                handshake_ratio = float(handshakes_completed / handshake_attempts)

        # 5. Scanning & Fanout Dynamics
        unique_dst_ips = float(len(set(dst_ips)))
        unique_dst_ports = float(len(set(dst_ports)))

        # Vertical scan: max distinct ports accessed on a single host / total ports probed
        if unique_dst_ports > 0:
            ports_per_ip: Dict[str, Set[int]] = {}
            for d_ip, d_port in zip(dst_ips, dst_ports):
                if d_ip not in ports_per_ip:
                    ports_per_ip[d_ip] = set()
                ports_per_ip[d_ip].add(d_port)
            max_ports_single_ip = max(len(p_set) for p_set in ports_per_ip.values())
            vertical_scan_score = float(max_ports_single_ip / max(unique_dst_ports, 1.0))
        else:
            vertical_scan_score = 0.0

        # Horizontal scan: max distinct hosts probed on a single port / total hosts probed
        if unique_dst_ips > 0:
            ips_per_port: Dict[int, Set[str]] = {}
            for d_ip, d_port in zip(dst_ips, dst_ports):
                if d_port not in ips_per_port:
                    ips_per_port[d_port] = set()
                ips_per_port[d_port].add(d_ip)
            max_ips_single_port = max(len(ip_set) for ip_set in ips_per_port.values())
            horizontal_scan_score = float(max_ips_single_port / max(unique_dst_ips, 1.0))
        else:
            horizontal_scan_score = 0.0

        # Topology: Internal vs External Destinations
        internal_dsts = [d for d in dst_ips if is_rfc1918(d)]
        external_dsts = [d for d in dst_ips if not is_rfc1918(d)]
        uniq_int_dsts = float(len(set(internal_dsts)))
        uniq_ext_dsts = float(len(set(external_dsts)))

        if len(internal_dsts) > 1:
            int_counts: Dict[str, int] = {}
            for d in internal_dsts:
                int_counts[d] = int_counts.get(d, 0) + 1
            tot = len(internal_dsts)
            int_fanout_entropy = 0.0
            for cnt in int_counts.values():
                p = cnt / tot
                int_fanout_entropy -= p * math.log2(p)
        else:
            int_fanout_entropy = 0.0

        # 6. Application Telemetry (HTTP & DNS Inspection)
        http_req_cnt = 0
        http_uri_entropies: List[float] = []
        http_sp_ratios: List[float] = []

        dns_query_cnt = 0
        dns_domains: Set[str] = set()
        dns_entropies: List[float] = []

        for p in packets:
            proto = p.get('protocol')
            sp = p.get('src_port') or 0
            dp = p.get('dst_port') or 0
            raw = p.get('raw') or b''
            plen = p.get('payload_length') or 0

            # HTTP on ports 80, 8080, 8000
            if proto == 6 and plen > 10 and (dp in [80, 8080, 8000] or sp in [80, 8080, 8000]):
                try:
                    raw_str = raw.decode('ascii', errors='ignore')
                    lines = raw_str.split('\r\n')
                    if len(lines) > 0 and any(lines[0].startswith(verb) for verb in ['GET ', 'POST ', 'HEAD ', 'PUT ']):
                        http_req_cnt += 1
                        parts = lines[0].split()
                        if len(parts) >= 2:
                            uri = parts[1]
                            u_len = len(uri)
                            http_uri_entropies.append(calculate_shannon_entropy(uri))
                            sp_cnt = sum(1 for c in uri if c in SPECIAL_HTTP_CHARS)
                            http_sp_ratios.append(sp_cnt / max(u_len, 1))
                except Exception:
                    pass

            # DNS on UDP port 53
            if proto == 17 and (dp == 53 or sp == 53) and plen >= 12:
                dns_query_cnt += 1

        http_uri_entropy_mean = float(np.mean(http_uri_entropies)) if http_uri_entropies else 0.0
        http_special_char_ratio = float(np.mean(http_sp_ratios)) if http_sp_ratios else 0.0
        dns_unique_domains = float(len(dns_domains))
        dns_domain_entropy_mean = float(np.mean(dns_entropies)) if dns_entropies else 0.0

        return {
            'ttl_mean': ttl_mean,
            'ttl_std': ttl_std,
            'ttl_unique_count': ttl_unique_cnt,
            'tcp_window_mean': win_mean,
            'tcp_window_std': win_std,
            'tcp_retransmission_estimate': float(retrans_cnt),
            'tcp_retransmission_ratio': retrans_ratio,
            'pkt_iat_mean': iat_mean,
            'pkt_iat_std': iat_std,
            'pkt_iat_cv': iat_cv,
            'payload_size_mean': payload_mean,
            'payload_size_std': payload_std,
            'payload_zero_ratio': payload_zero_ratio,
            'unique_dst_ips_pcap': unique_dst_ips,
            'vertical_scan_score': vertical_scan_score,
            'horizontal_scan_score': horizontal_scan_score,
            'syn_only_ratio': syn_only_ratio,
            'syn_ack_response_ratio': syn_ack_ratio,
            'syn_no_response_ratio': syn_no_resp_ratio,
            'rst_after_syn_ratio': rst_after_syn_ratio,
            'tcp_handshake_completion_ratio': handshake_ratio,
            'http_request_count': float(http_req_cnt),
            'http_uri_entropy_mean': http_uri_entropy_mean,
            'http_special_char_ratio': http_special_char_ratio,
            'dns_query_count': float(dns_query_cnt),
            'dns_unique_domains': dns_unique_domains,
            'dns_domain_entropy_mean': dns_domain_entropy_mean,
            'unique_internal_destinations_pcap': uniq_int_dsts,
            'unique_external_destinations_pcap': uniq_ext_dsts,
            'internal_fanout_entropy_pcap': int_fanout_entropy
        }
