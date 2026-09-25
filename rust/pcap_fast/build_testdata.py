"""Build the pcap_fast test inputs under EXP (on disk, never /tmp).

  days/fri_16_pcap   real fri_16 hosts: a known-corrupt file (bad length at 6.9 MB),
                     both pcapng-named-.pcap parts of 172.31.69.25 (same host twice),
                     small real UCAP files, a 40 MB prefix copy cut mid-record,
                     two 4-minute slices across the SlowHTTPTest start, a junk
                     file, a <24-byte file and a .lnk.
  days/tue_20_pcap   real tue_20 slices across the LOIC-HTTP start; tue_20 is the
                     day whose intervals carry participants, so this exercises the
                     scoped (per-host) labels.
  days/wed_14_pcap   synthetic edge cases on the wed_14 clock (see make_edge_day).
  dryrun/            scripts/dry_run_plan.build_corpus output (tiny synthetic days).
"""

from __future__ import annotations

import os
import random
import shutil
import socket
import struct
import sys
from pathlib import Path

EXP = Path("/var/home/samito/Documents/SIH/rust_parser_experiment")
DATA = Path("/var/home/samito/Documents/SIH/DATA/pcap")
REPO = Path(__file__).resolve().parents[2]


def slice_pcap(src: Path, dst: Path, t0: float, t1: float, max_bytes: int = 400 << 20):
    """Copy the global header plus every record with t0 <= ts <= t1 (file order)."""
    with open(src, "rb") as f, open(dst, "wb") as o:
        gh = f.read(24)
        o.write(gh)
        nano = struct.unpack("<I", gh[:4])[0] == 0xA1B23C4D
        written = 0
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            sec, frac, incl, _ = struct.unpack("<IIII", h)
            if incl == 0 or incl > 262144:
                break
            ts = sec + (frac / 1e9 if nano else frac / 1e6)
            if ts > t1 + 60:          # allow a little disorder, then stop
                break
            if ts < t0:
                f.seek(incl, 1)
                continue
            data = f.read(incl)
            if len(data) < incl:
                break
            if ts <= t1:
                o.write(h + data)
                written += 16 + incl
                if written > max_bytes:
                    break


def prefix_copy(src: Path, dst: Path, nbytes: int):
    with open(src, "rb") as f, open(dst, "wb") as o:
        o.write(f.read(nbytes))


def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def make_fri16():
    d = EXP / "days" / "fri_16_pcap" / "pcap"
    d.mkdir(parents=True, exist_ok=True)
    s = DATA / "fri_16_pcap" / "pcap"
    for name in ["capDESKTOP-AN3U28N-172.31.64.38", "UCAP172.31.69.25-part1.pcap",
                 "UCAP172.31.69.25-part2.pcap", "UCAP172.31.69.27", "UCAP172.31.69.15",
                 "UCAP172.31.69.22"]:
        link(s / name, d / name)
    prefix_copy(s / "capPC1-172.31.64.31", d / "capPC1-172.31.64.31", 40_000_037)
    t = 1518790334.0   # DoS attacks-SlowHTTPTest start (UTC)
    slice_pcap(s / "capPC1-172.31.67.124", d / "capPC1-172.31.67.124", t - 120, t + 120)
    slice_pcap(s / "capPC1-172.31.67.59", d / "capPC1-172.31.67.59", t - 120, t + 120)
    (d / "capJUNK-10.9.9.9").write_bytes(b"this is not a pcap file at all" * 10)
    (d / "capTINY-10.9.9.8").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\0" * 6)
    (d / "capX-172.31.64.31 - Shortcut.lnk").write_bytes(b"lnk")


def make_tue20():
    d = EXP / "days" / "tue_20_pcap" / "pcap"
    d.mkdir(parents=True, exist_ok=True)
    s = DATA / "tue_20_pcap" / "pcap"
    t = 1519136034.0   # DDoS attacks-LOIC-HTTP start (UTC)
    names = ["UCAP172.31.69.25", "UCAP172.31.69.27", "capDESKTOP-AN3U28N-172.31.64.111",
             "capDESKTOP-AN3U28N-172.31.64.115"]
    for name in names:
        src = s / name
        if not src.exists():
            cands = sorted(s.glob(name + "*"))
            if not cands:
                print("missing", name)
                continue
            src = cands[0]
        slice_pcap(src, d / src.name, t - 90, t + 90)


# ------------------------------------------------------------------ synthetic edge cases
def _mac(b: bytes) -> bytes:
    return (b + b"\x80" * 6)[:6]


def _ip4(a: str) -> bytes:
    return socket.inet_aton(a)


def _ip6(a: str) -> bytes:
    return socket.inet_pton(socket.AF_INET6, a)


def tcp_seg(sp, dp, seq, ack, flags, payload=b"", doff=5, win=8192):
    return struct.pack("!HHIIHHHH", sp, dp, seq & 0xFFFFFFFF, ack & 0xFFFFFFFF,
                       (doff << 12) | flags, win, 0, 0) + b"\0" * (4 * doff - 20) + payload


def udp_seg(sp, dp, payload=b"", ulen=None):
    return struct.pack("!HHHH", sp, dp, 8 + len(payload) if ulen is None else ulen, 0) + payload


def ipv4(src, dst, proto, l4, ttl=64, ihl=5, flags=0x4000):
    hdr = struct.pack("!BBHHHBBH4s4s", 0x40 | ihl, 0, 20 + len(l4), 1, flags, ttl, proto, 0,
                      _ip4(src), _ip4(dst))
    return hdr + b"\0" * max(0, 4 * ihl - 20) + l4


def ipv6(src, dst, nh, l4, hop=64):
    return struct.pack("!IHBB", 0x60000000, len(l4), nh, hop) + _ip6(src) + _ip6(dst) + l4


def eth(payload, etype=0x0800, dmac=b"\x02\0\0\0\0\x01", smac=b"\x02\0\0\0\0\x02", vlan=False):
    if vlan:
        return dmac + smac + struct.pack("!HHH", 0x8100, 7, etype) + payload
    return dmac + smac + struct.pack("!H", etype) + payload


V4 = ["172.31.69.10", "172.31.69.11", "172.31.64.20", "10.0.0.5", "192.168.1.9",
      "172.15.0.1", "172.32.0.1", "18.219.211.138", "8.8.8.8", "172.16.0.1"]
V6 = ["fe80::1", "2001:db8::1", "::ffff:10.0.0.5", "::10.0.0.5", "::1", "2001:db8:0:1::"]


def random_frame(rng: random.Random) -> bytes:
    k = rng.random()
    if k < 0.03:
        return bytes(rng.randrange(256) for _ in range(rng.randrange(1, 14)))  # runt
    if k < 0.05:
        return eth(b"\0" * 28, etype=0x0806)                                     # ARP
    if k < 0.07:
        return eth(b"\x45" + b"\0" * rng.randrange(0, 19))                       # short IPv4
    if k < 0.20:                                                                  # IPv6
        s, d = rng.sample(V6, 2)
        nh = rng.choice([6, 17, 58])
        l4 = (tcp_seg(rng.randrange(1, 70000) % 65536, rng.choice([80, 443, 22]),
                      rng.randrange(1000), rng.randrange(1000), rng.randrange(64),
                      b"x" * rng.randrange(40)) if nh == 6 else
              udp_seg(rng.randrange(65536), rng.choice([53, 123]), b"q" * rng.randrange(30)))
        return eth(ipv6(s, d, nh, l4, hop=rng.choice([1, 64, 255])), etype=0x86DD,
                   vlan=rng.random() < 0.2)
    s, d = rng.sample(V4, 2)
    proto = rng.choice([6, 6, 6, 17, 17, 1, 47])
    if proto == 6:
        payload = b"a" * rng.choice([0, 0, 5, 11, 12, 100, 1400])
        if rng.random() < 0.1:
            l4 = tcp_seg(1, 2, 3, 4, 2)[: rng.randrange(20)]                       # TCP < 20 bytes
        else:
            l4 = tcp_seg(rng.choice([1025, 1026, 50000, 80]), rng.choice([80, 21, 22, 445, 8080, 1, 2, 3, 4]),
                         rng.choice([0, 100, 200, 4294967290]), rng.choice([0, 1, 101]),
                         rng.choice([0x02, 0x12, 0x10, 0x04, 0x14, 0x18, 0x11, 0x01, 0x3F, 0x00, 0x22, 0x06]),
                         payload, doff=rng.choice([5, 5, 5, 8, 15, 2]), win=rng.randrange(65536))
    elif proto == 17:
        l4 = udp_seg(rng.choice([53, 5353, 40000]), rng.choice([53, 123, 40001]),
                     b"d" * rng.choice([0, 3, 4, 20]), ulen=rng.choice([None, None, 0, 7, 3000]))
    else:
        l4 = b"\x08\0\0\0" + b"p" * rng.randrange(60)
    ihl = rng.choice([5, 5, 5, 5, 0, 3, 6, 15])
    frame = eth(ipv4(s, d, proto, l4, ttl=rng.choice([1, 64, 128, 255, 0]), ihl=ihl),
                vlan=rng.random() < 0.1)
    if rng.random() < 0.05:
        frame += b"\0" * rng.randrange(1, 30)                                     # Ethernet padding
    return frame


def http_frames(rng: random.Random):
    """Frames whose ascii-filtered raw text starts with an HTTP verb."""
    out = []
    for i in range(20):
        verb = rng.choice([b"GET ", b"POST", b"HEAD", b"PUT "])
        dmac = verb[:4] + b"/" + bytes([rng.choice(b"abc?=%")])
        smac = bytes(rng.choice(b"xyz?&=1(\x1c\x1f\t") for _ in range(6))
        payload = (b"GET /x?y=1 HTTP/1.1\r\nHost: a\r\n" if rng.random() < 0.5
                   else bytes(rng.randrange(256) for _ in range(40)) + b"\r\n")
        seg = tcp_seg(51000 + i, rng.choice([80, 8080, 8000]), 10 * i, 0, 0x18, payload)
        ipb = ipv4("172.31.69.10", "172.31.64.20", 6, seg, ttl=rng.choice([10, 64]))
        # Put the ascii part of the IP header out of the way so the "line" is long.
        out.append(eth(ipb, dmac=dmac, smac=smac))
    # Also a verb inside the TCP payload only (never counted: line 0 is the MACs).
    seg = tcp_seg(52000, 80, 1, 0, 0x18, b"GET /index.html HTTP/1.1\r\n\r\n")
    out.append(eth(ipv4("172.31.69.11", "172.31.64.20", 6, seg)))
    return out


def handshake_frames(c="172.31.69.10", s="172.31.64.20", cp=40000, sp=80):
    return [
        eth(ipv4(c, s, 6, tcp_seg(cp, sp, 100, 0, 0x02))),            # SYN
        eth(ipv4(s, c, 6, tcp_seg(sp, cp, 900, 101, 0x12))),          # SYN-ACK
        eth(ipv4(c, s, 6, tcp_seg(cp, sp, 101, 901, 0x10))),          # ACK
        eth(ipv4(c, s, 6, tcp_seg(cp, sp, 101, 901, 0x18, b"z" * 50))),
        eth(ipv4(c, s, 6, tcp_seg(cp, sp, 101, 901, 0x18, b"z" * 50))),  # retransmission
        eth(ipv4(c, s, 6, tcp_seg(cp + 1, sp, 5, 0, 0x02))),          # SYN, then RST back
        eth(ipv4(s, c, 6, tcp_seg(sp, cp + 1, 0, 6, 0x04))),
        eth(ipv4(c, s, 6, tcp_seg(cp + 2, sp, 5, 0, 0x02, b"p" * 20))),  # SYN with payload
    ]


def write_pcap(path: Path, recs, big_endian=False, nano=False, tail=b""):
    e = ">" if big_endian else "<"
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    with open(path, "wb") as f:
        f.write(struct.pack(e + "IHHiIII", magic, 2, 4, 0, 0, 65535, 1))
        for ts, raw in recs:
            sec = int(ts)
            frac = int((ts - sec) * (1e9 if nano else 1e6))
            f.write(struct.pack(e + "IIII", sec, frac, len(raw), len(raw)))
            f.write(raw)
        f.write(tail)


def make_edge_day(seed=7):
    d = EXP / "days" / "wed_14_pcap" / "pcap"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rng = random.Random(seed)
    t0 = 1518618806.0 - 40.0   # 40 s before the FTP-BruteForce start
    hosts = ["172.31.69.10", "172.31.69.11", "172.31.64.20", "172.31.69.12", "172.31.69.13"]
    for hi, host in enumerate(hosts):
        recs = []
        t = t0 + hi * 0.37
        for n in range(4000):
            t += rng.expovariate(40.0)
            ts = t
            if rng.random() < 0.01:
                ts = t - rng.choice([0.5, 2.0, 2.5, 6.0])      # out-of-order packets
            recs.append((ts, random_frame(rng)))
            if n % 500 == 0:
                for f in handshake_frames(cp=40000 + n):
                    t += 0.001
                    recs.append((t, f))
            if n == 1000:
                for f in http_frames(rng):
                    t += 0.002
                    recs.append((t, f))
        big, nano = hi == 1, hi == 2
        tail = b""
        if hi == 3:
            tail = struct.pack("<IIII", int(t) + 1, 0, 300000, 300000) + b"\0" * 64   # corrupt length
        elif hi == 4:
            tail = struct.pack("<IIII", int(t) + 1, 0, 100, 100) + b"\0" * 10          # short data
        write_pcap(d / f"UCAP{host}", recs, big_endian=big, nano=nano, tail=tail)
    # Same host twice: second file only overlaps a few windows (dict-replace semantics).
    recs = [(t0 + 10 + 0.01 * k, random_frame(rng)) for k in range(600)]
    write_pcap(d / "UCAP172.31.69.10 - Copy", recs)
    # A zero-length record right away, and an empty-after-header file.
    write_pcap(d / "UCAP172.31.69.14", [(t0 + 1, handshake_frames()[0])],
               tail=struct.pack("<IIII", int(t0) + 2, 0, 0, 0))
    write_pcap(d / "UCAP172.31.69.15", [])
    # Big-endian file whose magic is pcapng-looking only in one byte order: plain junk.
    (d / "UCAP172.31.69.16").write_bytes(b"\x0a\x0d\x0d\x0a" + b"\0" * 40)


def make_dryrun():
    sys.path.insert(0, str(REPO))
    from scripts.dry_run_plan import build_corpus
    root = EXP / "dryrun"
    if root.exists():
        shutil.rmtree(root)
    build_corpus(root, seed=0)


if __name__ == "__main__":
    make_fri16()
    make_tue20()
    make_edge_day()
    make_dryrun()
    os.system(f"du -sh {EXP}/days/* {EXP}/dryrun")
