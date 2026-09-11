#!/usr/bin/env python3
"""
nodes/services/dns_server.py
Lightweight authoritative enterprise DNS server for corp.local domain.
Listens on UDP port 53.
"""

import socket
import struct
import sys
import time

RECORD_MAP = {
    b"corp.local": "10.0.2.10",
    b"dns.corp.local": "10.0.2.10",
    b"id.corp.local": "10.0.2.20",
    b"auth.corp.local": "10.0.2.20",
    b"file.corp.local": "10.0.2.30",
    b"storage.corp.local": "10.0.2.30",
    b"app.corp.local": "10.0.2.40",
    b"api.corp.local": "10.0.2.40",
    b"db.corp.local": "10.0.2.50",
    b"database.corp.local": "10.0.2.50",
    b"web.corp.local": "10.0.3.10",
    b"dmz.corp.local": "10.0.3.10",
}

def parse_domain(data, offset):
    domain_parts = []
    while True:
        length = data[offset]
        if length == 0:
            offset += 1
            break
        offset += 1
        domain_parts.append(data[offset:offset + length])
        offset += length
    return b".".join(domain_parts).lower(), offset

def build_response(request_data):
    if len(request_data) < 12:
        return None
    tid = request_data[:2]
    flags = b"\x81\x80"  # Standard response, no error
    qdcount = request_data[4:6]
    
    # Parse Query
    try:
        domain, offset = parse_domain(request_data, 12)
        qtype, qclass = struct.unpack("!HH", request_data[offset:offset + 4])
        query_section = request_data[12:offset + 4]
    except Exception:
        return None

    ip_str = RECORD_MAP.get(domain, "10.0.2.10")
    ancount = b"\x00\x01"  # 1 answer
    nscount = b"\x00\x00"
    arcount = b"\x00\x00"
    
    header = tid + flags + qdcount + ancount + nscount + arcount
    
    # Answer section: name pointer (0xc00c points to domain in question), type A (1), class IN (1), ttl (300s), data len (4), IP
    answer_name = b"\xc0\x0c"
    answer_meta = struct.pack("!HHIH", 1, 1, 300, 4)
    ip_bytes = socket.inet_aton(ip_str)
    answer = answer_name + answer_meta + ip_bytes
    
    return header + query_section + answer

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 53
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"[*] Enterprise DNS Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()

    while True:
        try:
            data, addr = sock.recvfrom(512)
            resp = build_response(data)
            if resp:
                sock.sendto(resp, addr)
        except Exception as e:
            print(f"[!] DNS Server Error: {e}", file=sys.stderr)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
