#!/usr/bin/env python3
"""
telemetry/capture/sniffer.py
Low-overhead streaming packet sniffer.
Captures live frames from a specified network interface using Linux AF_PACKET raw sockets.
Decodes Ethernet, IPv4, IPv6, TCP, UDP layers into canonical packet dicts in memory.
Zero intermediate disk I/O.
"""

import os
import socket
import struct
import sys
import time
from typing import Callable, Optional, Dict, Any

class StreamingPacketSniffer:
    def __init__(self, interface: str = "eth1"):
        self.interface = interface
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.packet_count = 0

    def start(self):
        """Initializes the raw AF_PACKET socket in promiscuous mode."""
        try:
            # ETH_P_ALL = 0x0003 in network byte order is htons(0x0003) = 0x0300
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
            self.sock.bind((self.interface, 0))
            self.sock.settimeout(0.5)
            self.running = True
            print(f"[*] StreamingPacketSniffer active on interface '{self.interface}' (promiscuous mode).")
            sys.stdout.flush()
        except PermissionError:
            print(f"[!] Sniffer PermissionError: Raw packet capture requires CAP_NET_RAW / root on '{self.interface}'.", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[!] Sniffer init error on '{self.interface}': {e}", file=sys.stderr)
            raise

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def capture_loop(self, packet_callback: Callable[[Dict[str, Any]], None]):
        """Continuous non-blocking packet ingestion loop."""
        if not self.sock:
            self.start()

        while self.running:
            try:
                raw_data, _ = self.sock.recvfrom(65535)
                ts = time.time()
                pkt = self.parse_frame(raw_data, ts)
                if pkt:
                    self.packet_count += 1
                    packet_callback(pkt)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    time.sleep(0.01)

    @staticmethod
    def parse_frame(data: bytes, ts: float) -> Optional[Dict[str, Any]]:
        """Parses raw frame bytes into canonical packet dictionary."""
        length = len(data)
        if length < 14:
            return None

        eth_type = struct.unpack("!H", data[12:14])[0]
        eth_offset = 14

        # Handle 802.1Q VLAN Tag
        if eth_type == 0x8100:
            if length < 18:
                return None
            eth_type = struct.unpack("!H", data[16:18])[0]
            eth_offset = 18

        if eth_type == 0x0800:  # IPv4
            if length < eth_offset + 20:
                return None
            ip_data = data[eth_offset:]
            ip_ver = 4
            ihl = (ip_data[0] & 0x0F) * 4
            ttl = ip_data[8]
            proto = ip_data[9]
            flags_frag = struct.unpack("!H", ip_data[6:8])[0]
            df = (flags_frag >> 14) & 0x01
            mf = (flags_frag >> 13) & 0x01
            frag_offset = (flags_frag & 0x1FFF) * 8
            src_ip = socket.inet_ntoa(ip_data[12:16])
            dst_ip = socket.inet_ntoa(ip_data[16:20])
            l4_data = ip_data[ihl:]

        elif eth_type == 0x86DD:  # IPv6
            if length < eth_offset + 40:
                return None
            ip_data = data[eth_offset:]
            ip_ver = 6
            proto = ip_data[6]
            ttl = ip_data[7]
            df, mf, frag_offset = 0, 0, 0
            src_ip = socket.inet_ntop(socket.AF_INET6, ip_data[8:24])
            dst_ip = socket.inet_ntop(socket.AF_INET6, ip_data[24:40])
            l4_data = ip_data[40:]
        else:
            return None

        src_port = None
        dst_port = None
        tcp_flags = None
        tcp_seq = None
        tcp_ack = None
        tcp_win = None
        payload_len = 0

        if proto == 6 and len(l4_data) >= 20:  # TCP
            src_port, dst_port, seq, ack, data_offset_flags = struct.unpack("!HHIIH", l4_data[:14])
            tcp_seq = seq
            tcp_ack = ack
            data_offset = ((data_offset_flags >> 12) & 0x0F) * 4
            flags_bits = data_offset_flags & 0x01FF
            tcp_flags = {
                "FIN": bool(flags_bits & 0x0001),
                "SYN": bool(flags_bits & 0x0002),
                "RST": bool(flags_bits & 0x0004),
                "PSH": bool(flags_bits & 0x0008),
                "ACK": bool(flags_bits & 0x0010),
                "URG": bool(flags_bits & 0x0020)
            }
            tcp_win = struct.unpack("!H", l4_data[14:16])[0]
            payload_len = max(0, len(l4_data) - data_offset)

        elif proto == 17 and len(l4_data) >= 8:  # UDP
            src_port, dst_port, udp_length = struct.unpack("!HHH", l4_data[:6])
            payload_len = max(0, udp_length - 8)

        return {
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "ip_version": ip_ver,
            "protocol": proto,
            "packet_length": length,
            "src_port": src_port,
            "dst_port": dst_port,
            "ttl_or_hop_limit": ttl,
            "fragment_offset": frag_offset,
            "more_fragments": mf,
            "dont_fragment": df,
            "tcp_flags": tcp_flags,
            "tcp_seq": tcp_seq,
            "tcp_ack": tcp_ack,
            "tcp_window": tcp_win,
            "payload_length": payload_len,
            "raw": data
        }
