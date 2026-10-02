//! Experimental, bit-exact Rust port of the PCAP -> host-window hot path.
//!
//! Mirrors, line for line where it matters:
//!   data_unification/pcap_adapter.py   PcapFlowExtractor._iter_packets
//!   data_unification/pcap_bridge.py    _host_window_stream, iter_merged_day_windows
//!   telemetry/capture/sniffer.py       StreamingPacketSniffer.parse_frame
//!   telemetry/flow/flow_table.py       LiveFlowTable.process_packet / snapshot_flows
//!   telemetry/packet/pcap_engine.py    LivePCAPEngine.extract_features
//!
//! Label attachment and the encoder columns stay in Python (pcap_fast_py.py):
//! numpy's float64 log1p on this CPU is the AVX-512 SVML kernel, which differs
//! from glibc's log1p in the last bit on ~25% of inputs, so the edge features
//! must be computed by numpy to be bit-identical.
//!
//! Usage: pcap_fast <window_seconds> <max_packets|0> (<host> <path>)...
//!   (pairs in iter_day_captures order; Python passes them, so ordering and
//!   host_ip_from_filename stay in exactly one implementation. Two files can
//!   share a host -- "x - Copy", "-part1/-part2" -- and the per-window dict
//!   merges them by host string, so Rust needs the strings.)
//! Output: a chunked little-endian binary stream on stdout; see write_chunk.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::{self, BufReader, BufWriter, Read, Write};

// ------------------------------------------------------------------ hashing
#[derive(Default, Clone, Copy)]
struct FxHasher {
    h: u64,
}
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.h = (self.h.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) { self.add(i as u64) }
    #[inline]
    fn write_u16(&mut self, i: u16) { self.add(i as u64) }
    #[inline]
    fn write_u32(&mut self, i: u32) { self.add(i as u64) }
    #[inline]
    fn write_u64(&mut self, i: u64) { self.add(i) }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.add(i as u64) }
    #[inline]
    fn finish(&self) -> u64 { self.h }
}
type FxBuild = BuildHasherDefault<FxHasher>;
type FxMap<K, V> = HashMap<K, V, FxBuild>;
type FxSet<K> = HashSet<K, FxBuild>;

// ------------------------------------------------------------------ addresses
/// An IP address. Equality of these is exactly equality of Python's
/// inet_ntoa/inet_ntop strings (both are injective within a family and an
/// IPv4 dotted quad never equals an IPv6 text form).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Ip {
    V4([u8; 4]),
    V6([u8; 16]),
}

impl Ip {
    /// telemetry/*: is_rfc1918(str) on the address's text form.
    fn is_rfc1918(&self) -> bool {
        match self {
            // startswith('10.') / ('192.168.') / '172.' and 16 <= second <= 31.
            Ip::V4(b) => b[0] == 10 || (b[0] == 192 && b[1] == 168) || (b[0] == 172 && (16..=31).contains(&b[1])),
            // An inet_ntop IPv6 string never starts with "10.", "192.168." or "172.".
            Ip::V6(_) => false,
        }
    }

    fn to_py_string(&self) -> String {
        match self {
            Ip::V4(b) => format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3]),
            Ip::V6(b) => ntop6(b),
        }
    }
}

/// glibc resolv/inet_ntop.c inet_ntop6, which CPython's socket.inet_ntop calls.
fn ntop6(b: &[u8; 16]) -> String {
    let mut words = [0u32; 8];
    for i in 0..8 {
        words[i] = ((b[2 * i] as u32) << 8) | b[2 * i + 1] as u32;
    }
    let (mut best_base, mut best_len): (i32, i32) = (-1, 0);
    let (mut cur_base, mut cur_len): (i32, i32) = (-1, 0);
    for i in 0..8 {
        if words[i] == 0 {
            if cur_base == -1 {
                cur_base = i as i32;
                cur_len = 1;
            } else {
                cur_len += 1;
            }
        } else if cur_base != -1 {
            if best_base == -1 || cur_len > best_len {
                best_base = cur_base;
                best_len = cur_len;
            }
            cur_base = -1;
        }
    }
    if cur_base != -1 && (best_base == -1 || cur_len > best_len) {
        best_base = cur_base;
        best_len = cur_len;
    }
    if best_base != -1 && best_len < 2 {
        best_base = -1;
    }
    let mut out = String::with_capacity(40);
    for i in 0..8i32 {
        if best_base != -1 && i >= best_base && i < best_base + best_len {
            if i == best_base {
                out.push(':');
            }
            continue;
        }
        if i != 0 {
            out.push(':');
        }
        if i == 6 && best_base == 0 && (best_len == 6 || (best_len == 5 && words[5] == 0xffff)) {
            out.push_str(&format!("{}.{}.{}.{}", b[12], b[13], b[14], b[15]));
            // glibc returns here without the trailing-"::" check below.
            return out;
        }
        out.push_str(&format!("{:x}", words[i as usize]));
    }
    if best_base != -1 && best_base + best_len == 8 {
        out.push(':');
    }
    out
}

// ------------------------------------------------------------------ numpy arithmetic
/// numpy's @TYPE@_pairwise_sum (numpy/core/src/umath/loops_utils.h.src).
fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let lim = n - (n % 8);
        while i < lim {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// np.add.reduce on a contiguous float64 array: identity 0.0, then the inner
/// loop is fed in iterator buffer-sized (8192) blocks, each pairwise-summed.
fn np_sum(a: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for c in a.chunks(8192) {
        acc += pairwise(c);
    }
    acc
}

fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// numpy _methods._var(ddof=0) then sqrt.
fn np_std(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean = np_sum(a) / n;
    let sq: Vec<f64> = a.iter().map(|&x| {
        let d = x - mean;
        d * d
    }).collect();
    (np_sum(&sq) / n).sqrt()
}

/// CPython float_floor_div (Objects/floatobject.c, _float_div_mod).
fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let m = vx % wx; // C fmod
    let mut div = (vx - m) / wx;
    if m != 0.0 && ((wx < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut fd = div.floor();
        if div - fd > 0.5 {
            fd += 1.0;
        }
        fd
    } else {
        0.0f64.copysign(vx / wx)
    }
}

// ------------------------------------------------------------------ packets
const F_FIN: u8 = 0x01;
const F_SYN: u8 = 0x02;
const F_RST: u8 = 0x04;
const F_PSH: u8 = 0x08;
const F_ACK: u8 = 0x10;
const F_URG: u8 = 0x20;

/// The fields of sniffer.parse_frame's dict that anything downstream reads.
#[derive(Clone, Copy)]
struct Pkt {
    ts: f64,
    src: Ip,
    dst: Ip,
    proto: u8,
    length: u32,
    sport: u16, // None -> 0 (every consumer does `or 0`)
    dport: u16,
    ttl: u8,
    tcp_flags: Option<u8>, // None when the dict value is None
    seq: u32,
    ack: u32,
    win: u16,
    payload: i64,
    http: Option<(f64, f64)>, // counted HTTP request: (uri entropy, special-char ratio)
    is_http_req: bool,
}

#[inline]
fn be16(d: &[u8], o: usize) -> u16 {
    ((d[o] as u16) << 8) | d[o + 1] as u16
}
#[inline]
fn be32(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn parse_frame(data: &[u8], ts: f64) -> Option<Pkt> {
    let length = data.len();
    if length < 14 {
        return None;
    }
    let mut eth_type = be16(data, 12);
    let mut eth_offset = 14;
    if eth_type == 0x8100 {
        if length < 18 {
            return None;
        }
        eth_type = be16(data, 16);
        eth_offset = 18;
    }
    let (src, dst, proto, ttl, l4): (Ip, Ip, u8, u8, &[u8]);
    if eth_type == 0x0800 {
        if length < eth_offset + 20 {
            return None;
        }
        let ip = &data[eth_offset..];
        let ihl = ((ip[0] & 0x0F) as usize) * 4;
        ttl = ip[8];
        proto = ip[9];
        src = Ip::V4([ip[12], ip[13], ip[14], ip[15]]);
        dst = Ip::V4([ip[16], ip[17], ip[18], ip[19]]);
        // Python slicing: ip_data[ihl:] is empty past the end, whole when ihl == 0.
        l4 = if ihl <= ip.len() { &ip[ihl..] } else { &ip[ip.len()..] };
    } else if eth_type == 0x86DD {
        if length < eth_offset + 40 {
            return None;
        }
        let ip = &data[eth_offset..];
        proto = ip[6];
        ttl = ip[7];
        let mut s = [0u8; 16];
        let mut d = [0u8; 16];
        s.copy_from_slice(&ip[8..24]);
        d.copy_from_slice(&ip[24..40]);
        src = Ip::V6(s);
        dst = Ip::V6(d);
        l4 = &ip[40..];
    } else {
        return None;
    }

    let mut p = Pkt {
        ts, src, dst, proto, length: length as u32, sport: 0, dport: 0, ttl,
        tcp_flags: None, seq: 0, ack: 0, win: 0, payload: 0, http: None, is_http_req: false,
    };
    if proto == 6 && l4.len() >= 20 {
        p.sport = be16(l4, 0);
        p.dport = be16(l4, 2);
        p.seq = be32(l4, 4);
        p.ack = be32(l4, 8);
        let dof = be16(l4, 12);
        let data_offset = (((dof >> 12) & 0x0F) as i64) * 4;
        p.tcp_flags = Some((dof & 0x3F) as u8); // the six flags parse_frame keeps
        p.win = be16(l4, 14);
        p.payload = (l4.len() as i64 - data_offset).max(0);
    } else if proto == 17 && l4.len() >= 8 {
        p.sport = be16(l4, 0);
        p.dport = be16(l4, 2);
        let udp_len = be16(l4, 4) as i64;
        p.payload = (udp_len - 8).max(0);
    }

    // pcap_engine section 6, which reads the whole raw frame.
    let is_web = |x: u16| x == 80 || x == 8080 || x == 8000;
    if proto == 6 && p.payload > 10 && (is_web(p.dport) || is_web(p.sport)) {
        http_inspect(data, &mut p);
    }
    Some(p)
}

/// raw.decode('ascii', errors='ignore').split('\r\n')[0], the verb check,
/// .split()[1], calculate_shannon_entropy(uri) and the special-char ratio.
fn http_inspect(raw: &[u8], p: &mut Pkt) {
    let ascii: Vec<u8> = raw.iter().copied().filter(|&b| b < 0x80).collect();
    let end = ascii.windows(2).position(|w| w == b"\r\n").unwrap_or(ascii.len());
    let line = &ascii[..end];
    let verbs: [&[u8]; 4] = [b"GET ", b"POST ", b"HEAD ", b"PUT "];
    if !verbs.iter().any(|v| line.starts_with(v)) {
        return;
    }
    p.is_http_req = true;
    // str.split() whitespace for ASCII code points: \t\n\x0b\x0c\r, \x1c-\x1f, space.
    let is_ws = |b: u8| matches!(b, 0x09..=0x0D | 0x1C..=0x1F | 0x20);
    let parts: Vec<&[u8]> = line.split(|&b| is_ws(b)).filter(|s| !s.is_empty()).collect();
    if parts.len() >= 2 {
        let uri = parts[1];
        let ent = shannon_entropy(uri);
        const SPECIAL: &[u8] = b" '\"=<>():;%&/\\#+?*!{}[]|`^~$,";
        let sp = uri.iter().filter(|c| SPECIAL.contains(c)).count();
        let ratio = sp as f64 / (uri.len().max(1)) as f64;
        p.http = Some((ent, ratio));
    }
}

/// calculate_shannon_entropy: counts in first-occurrence (dict insertion) order.
fn shannon_entropy(s: &[u8]) -> f64 {
    if s.len() <= 1 {
        return 0.0;
    }
    let mut order: Vec<u8> = Vec::new();
    let mut counts = [0usize; 256];
    for &b in s {
        if counts[b as usize] == 0 {
            order.push(b);
        }
        counts[b as usize] += 1;
    }
    let length = s.len() as f64;
    let mut ent = 0.0f64;
    for b in order {
        let pr = counts[b as usize] as f64 / length;
        ent -= pr * pr.log2();
    }
    ent
}

// ------------------------------------------------------------------ pcap reader
struct PcapReader {
    r: BufReader<File>,
    big_endian: bool,
    nano: bool,
    n: u64,
    max_packets: u64,
    buf: Vec<u8>,
    bytes: u64,
}

enum Open {
    Ok(PcapReader),
    Empty,
    FormatError(String),
}

/// Read exactly `want` bytes or as many as exist (Python f.read(n) semantics).
fn read_up_to(r: &mut impl Read, buf: &mut [u8]) -> usize {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => panic!("read error: {e}"), // Python would raise OSError too
        }
    }
    got
}

const PCAP_LE: u32 = 0xA1B2C3D4;
const PCAP_LE_NS: u32 = 0xA1B23C4D;
const PCAPNG: u32 = 0x0A0D0D0A;

impl PcapReader {
    fn open(path: &str, max_packets: u64) -> Open {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut r = BufReader::with_capacity(256 * 1024, f);
        let mut gh = [0u8; 24];
        if read_up_to(&mut r, &mut gh) < 24 {
            return Open::Empty;
        }
        let le = u32::from_le_bytes([gh[0], gh[1], gh[2], gh[3]]);
        let (big_endian, nano);
        if le == PCAPNG {
            return Open::FormatError("is pcapng".into());
        }
        if le == PCAP_LE || le == PCAP_LE_NS {
            big_endian = false;
            nano = le == PCAP_LE_NS;
        } else {
            let be = u32::from_be_bytes([gh[0], gh[1], gh[2], gh[3]]);
            if be != PCAP_LE && be != PCAP_LE_NS {
                return Open::FormatError("is not a pcap file".into());
            }
            big_endian = true;
            nano = be == PCAP_LE_NS;
        }
        Open::Ok(PcapReader { r, big_endian, nano, n: 0, max_packets, buf: Vec::with_capacity(65536), bytes: 24 })
    }

    /// _iter_packets' loop body. Returns (ts, frame) or None at the end / at
    /// the first corrupt record (valid prefix kept).
    fn next(&mut self) -> Option<f64> {
        if self.max_packets != 0 && self.n >= self.max_packets {
            return None;
        }
        let mut hdr = [0u8; 16];
        if read_up_to(&mut self.r, &mut hdr) < 16 {
            return None;
        }
        let g = |o: usize| {
            let b = [hdr[o], hdr[o + 1], hdr[o + 2], hdr[o + 3]];
            if self.big_endian { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) }
        };
        let (sec, frac, incl) = (g(0), g(4), g(8));
        if incl == 0 || incl > 262144 {
            return None;
        }
        self.buf.resize(incl as usize, 0);
        if read_up_to(&mut self.r, &mut self.buf) < incl as usize {
            return None;
        }
        self.bytes += 16 + incl as u64;
        self.n += 1;
        // sec + (frac / 1e9 if nano else frac / 1e6)
        Some(sec as f64 + if self.nano { frac as f64 / 1e9 } else { frac as f64 / 1e6 })
    }
}

// ------------------------------------------------------------------ flow table
struct Flow {
    src: Ip,
    dst: Ip,
    sport: u16,
    dport: u16,
    proto: u8,
    first_ts: f64,
    last_ts: f64,
    fwd_bytes: u64,
    bwd_bytes: u64,
    fwd_pkts: u64,
    bwd_pkts: u64,
}

type FlowKey = (Ip, Ip, u16, u16, u8);

/// LiveFlowTable restricted to what snapshot_flows exports. A table lives for
/// one window and is snapshotted once, so the exported_* deltas equal totals.
#[derive(Default)]
struct FlowTable {
    index: FxMap<FlowKey, usize>,
    flows: Vec<Flow>, // dict insertion order
}

impl FlowTable {
    fn process(&mut self, p: &Pkt) {
        let fwd = (p.src, p.dst, p.sport, p.dport, p.proto);
        let len = p.length as u64;
        if let Some(&i) = self.index.get(&fwd) {
            let f = &mut self.flows[i];
            f.last_ts = p.ts;
            f.fwd_pkts += 1;
            f.fwd_bytes += len;
            return;
        }
        let rev = (p.dst, p.src, p.dport, p.sport, p.proto);
        if let Some(&i) = self.index.get(&rev) {
            let f = &mut self.flows[i];
            f.last_ts = p.ts;
            f.bwd_pkts += 1;
            f.bwd_bytes += len;
            return;
        }
        self.index.insert(fwd, self.flows.len());
        self.flows.push(Flow {
            src: p.src, dst: p.dst, sport: p.sport, dport: p.dport, proto: p.proto,
            first_ts: p.ts, last_ts: p.ts, fwd_bytes: len, bwd_bytes: 0, fwd_pkts: 1, bwd_pkts: 0,
        });
    }
}

// ------------------------------------------------------------------ packet features
struct Sess {
    max_seq: i64,
    max_ack: i64,
    syn_seen: bool,
    syn_ack_seen: bool,
    ack_seen: bool,
    rst_seen: bool,
}

/// LivePCAPEngine.extract_features, in PCAP_BEHAVIORAL_COLUMNS order.
fn extract_features(pk: &[Pkt]) -> [f64; 30] {
    let n = pk.len();
    let ttls: Vec<f64> = pk.iter().map(|p| p.ttl as f64).collect();
    let ttl_mean = np_mean(&ttls);
    let ttl_std = np_std(&ttls);
    let mut ttl_seen = [false; 256];
    for p in pk {
        ttl_seen[p.ttl as usize] = true;
    }
    let ttl_unique = ttl_seen.iter().filter(|&&b| b).count() as f64;

    let (iat_mean, iat_std, iat_cv) = if n > 1 {
        let mut ts: Vec<f64> = pk.iter().map(|p| p.ts).collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let iats: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
        let m = np_mean(&iats);
        let s = np_std(&iats);
        (m, s, s / (m + 1e-7))
    } else {
        (0.0, 0.0, 0.0)
    };

    let payloads: Vec<f64> = pk.iter().map(|p| p.payload as f64).collect();
    let payload_mean = np_mean(&payloads);
    let payload_std = np_std(&payloads);
    let zeros = pk.iter().filter(|p| p.payload == 0).count();
    let payload_zero_ratio = zeros as f64 / n as f64;

    let tcp: Vec<&Pkt> = pk.iter().filter(|p| p.proto == 6).collect();
    let n_tcp = tcp.len();
    let (mut win_mean, mut win_std) = (0.0, 0.0);
    let (mut retrans_cnt, mut retrans_ratio) = (0u64, 0.0);
    let mut syn_only_ratio = 0.0;
    let (mut syn_ack_ratio, mut syn_no_resp_ratio, mut rst_after_syn_ratio) = (0.0, 0.0, 0.0);
    let mut handshake_ratio = 0.0;
    if n_tcp > 0 {
        let wins: Vec<f64> = tcp.iter().map(|p| p.win as f64).collect();
        win_mean = np_mean(&wins);
        win_std = np_std(&wins);
        let mut sessions: FxMap<(Ip, u16, Ip, u16), Sess> = FxMap::default();
        let (mut syn_count, mut syn_ack_count, mut rst_after_syn) = (0u64, 0u64, 0u64);
        let (mut attempts, mut completed, mut syn_only) = (0u64, 0u64, 0u64);
        for p in &tcp {
            let fl = p.tcp_flags.unwrap_or(0);
            let has = |b: u8| fl & b != 0;
            let fwd = (p.src, p.sport, p.dst, p.dport);
            let rev = (p.dst, p.dport, p.src, p.sport);
            let seq = p.seq as i64;
            let ack = p.ack as i64;
            let plen = p.payload;
            let is_syn = has(F_SYN) && !has(F_ACK);
            let is_syn_ack = has(F_SYN) && has(F_ACK);
            let is_ack = has(F_ACK) && !has(F_SYN);
            let is_rst = has(F_RST);
            if is_syn && !(has(F_ACK) || has(F_FIN) || has(F_RST) || has(F_PSH) || has(F_URG)) {
                syn_only += 1;
            }
            if is_syn {
                syn_count += 1;
                attempts += 1;
                sessions.insert(fwd, Sess { max_seq: seq + plen, max_ack: ack, syn_seen: true,
                    syn_ack_seen: false, ack_seen: false, rst_seen: false });
            } else if is_syn_ack {
                syn_ack_count += 1;
                if let Some(s) = sessions.get_mut(&rev) {
                    s.syn_ack_seen = true;
                }
            } else if is_ack {
                if let Some(s) = sessions.get_mut(&fwd) {
                    if s.syn_seen && s.syn_ack_seen && !s.ack_seen {
                        completed += 1;
                        s.ack_seen = true;
                    }
                }
            } else if is_rst {
                if let Some(s) = sessions.get(&rev) {
                    if s.syn_seen && !s.syn_ack_seen {
                        rst_after_syn += 1;
                    }
                }
            }
            if let Some(s) = sessions.get_mut(&fwd) {
                if plen > 0 && seq < s.max_seq {
                    retrans_cnt += 1;
                } else if seq + plen > s.max_seq {
                    s.max_seq = seq + plen;
                }
                if ack > s.max_ack {
                    s.max_ack = ack;
                }
            } else {
                sessions.insert(fwd, Sess { max_seq: seq + plen, max_ack: ack, syn_seen: false,
                    syn_ack_seen: false, ack_seen: false, rst_seen: is_rst });
            }
        }
        retrans_ratio = retrans_cnt as f64 / n_tcp as f64;
        syn_only_ratio = syn_only as f64 / n_tcp as f64;
        if syn_count > 0 {
            syn_ack_ratio = syn_ack_count as f64 / syn_count as f64;
            let unanswered = sessions.values().filter(|s| s.syn_seen && !s.syn_ack_seen && !s.rst_seen).count();
            syn_no_resp_ratio = unanswered as f64 / syn_count as f64;
            rst_after_syn_ratio = rst_after_syn as f64 / syn_count as f64;
        }
        if attempts > 0 {
            handshake_ratio = completed as f64 / attempts as f64;
        }
    }

    // 5. scanning & fanout
    let mut ports_per_ip: FxMap<Ip, FxSet<u16>> = FxMap::default();
    let mut ips_per_port: FxMap<u16, FxSet<Ip>> = FxMap::default();
    let mut dst_port_set: FxSet<u16> = FxSet::default();
    for p in pk {
        ports_per_ip.entry(p.dst).or_default().insert(p.dport);
        ips_per_port.entry(p.dport).or_default().insert(p.dst);
        dst_port_set.insert(p.dport);
    }
    let unique_dst_ips = ports_per_ip.len() as f64;
    let unique_dst_ports = dst_port_set.len() as f64;
    let max_ports = ports_per_ip.values().map(|s| s.len()).max().unwrap_or(0) as f64;
    let vertical = if unique_dst_ports > 0.0 { max_ports } else { 0.0 }; // count: see pcap_engine.py
    let max_ips = ips_per_port.values().map(|s| s.len()).max().unwrap_or(0) as f64;
    let horizontal = if unique_dst_ips > 0.0 { max_ips } else { 0.0 }; // count: see pcap_engine.py

    let mut int_order: Vec<Ip> = Vec::new();
    let mut int_counts: FxMap<Ip, u64> = FxMap::default();
    let mut ext_set: FxSet<Ip> = FxSet::default();
    let mut n_internal = 0u64;
    for p in pk {
        if p.dst.is_rfc1918() {
            n_internal += 1;
            let c = int_counts.entry(p.dst).or_insert(0);
            if *c == 0 {
                int_order.push(p.dst);
            }
            *c += 1;
        } else {
            ext_set.insert(p.dst);
        }
    }
    let uniq_int = int_counts.len() as f64;
    let uniq_ext = ext_set.len() as f64;
    let mut int_fanout = 0.0f64;
    if n_internal > 1 {
        let tot = n_internal as f64;
        for ip in &int_order {
            let pr = int_counts[ip] as f64 / tot;
            int_fanout -= pr * pr.log2();
        }
    }

    // 6. application telemetry
    let http_req = pk.iter().filter(|p| p.is_http_req).count() as f64;
    let ents: Vec<f64> = pk.iter().filter_map(|p| p.http.map(|h| h.0)).collect();
    let sps: Vec<f64> = pk.iter().filter_map(|p| p.http.map(|h| h.1)).collect();
    let http_ent_mean = if ents.is_empty() { 0.0 } else { np_mean(&ents) };
    let http_sp_mean = if sps.is_empty() { 0.0 } else { np_mean(&sps) };
    let dns = pk.iter().filter(|p| p.proto == 17 && (p.dport == 53 || p.sport == 53) && p.payload >= 12).count() as f64;

    [
        ttl_mean, ttl_std, ttl_unique, win_mean, win_std, retrans_cnt as f64, retrans_ratio,
        iat_mean, iat_std, iat_cv, payload_mean, payload_std, payload_zero_ratio,
        unique_dst_ips, vertical, horizontal, syn_only_ratio, syn_ack_ratio, syn_no_resp_ratio,
        rst_after_syn_ratio, handshake_ratio, http_req, http_ent_mean, http_sp_mean, dns,
        0.0, 0.0, uniq_int, uniq_ext, int_fanout,
    ]
}

// ------------------------------------------------------------------ host stream
struct HostWindow {
    bucket: i64,
    host: u32,     // file index (what the output carries)
    host_key: u32, // index of the first file with the same host string (dict key)
    flows: Vec<Flow>,
    feats: [f64; 30],
}

struct HostStream {
    host: u32,
    host_key: u32,
    reader: Option<PcapReader>,
    window: f64,
    cur_bucket: Option<i64>,
    table: FlowTable,
    buf: Vec<Pkt>,
    finished: bool,
}

impl HostStream {
    fn close_window(&mut self, bucket: i64) -> HostWindow {
        let table = std::mem::take(&mut self.table);
        let feats = extract_features(&self.buf);
        self.buf.clear();
        HostWindow { bucket, host: self.host, host_key: self.host_key, flows: table.flows, feats }
    }

    /// One step of the _host_window_stream generator.
    fn next(&mut self, stats: &mut Stats) -> Option<HostWindow> {
        if self.finished {
            return None;
        }
        loop {
            let reader = match self.reader.as_mut() {
                Some(r) => r,
                None => break,
            };
            let ts = match reader.next() {
                Some(t) => t,
                None => {
                    stats.bytes += reader.bytes;
                    stats.packets += reader.n;
                    self.reader = None;
                    break;
                }
            };
            let bucket = py_floordiv(ts, self.window) as i64;
            let mut out = None;
            match self.cur_bucket {
                None => self.cur_bucket = Some(bucket),
                Some(cb) if cb != bucket => {
                    if !self.buf.is_empty() {
                        out = Some(self.close_window(cb));
                    } else {
                        self.table = FlowTable::default();
                    }
                    self.cur_bucket = Some(bucket);
                }
                _ => {}
            }
            let reader = self.reader.as_ref().unwrap();
            if let Some(p) = parse_frame(&reader.buf, ts) {
                self.table.process(&p);
                self.buf.push(p);
            }
            if out.is_some() {
                return out;
            }
        }
        self.finished = true;
        if !self.buf.is_empty() {
            let cb = self.cur_bucket.unwrap();
            return Some(self.close_window(cb));
        }
        None
    }
}

#[derive(Default)]
struct Stats {
    bytes: u64,
    packets: u64,
}

// ------------------------------------------------------------------ output
/// Chunk layout (all little-endian):
///   b"PFC1"
///   u32 n_strings, then per string: u16 len + utf-8 bytes   (ids continue from the previous chunk)
///   u32 n_windows, then per window: i64 bucket, u32 n_hostwins
///   u32 n_hostwins, then per host-window: u32 host_index, u32 n_flows, 30 x f64 features
///   u32 n_flows,   then per flow: u32 src, u32 dst, u32 sport, u32 dport, u32 proto,
///                  f64 start, f64 end, u64 fwd_bytes, u64 bwd_bytes, u64 fwd_pkts, u64 bwd_pkts
/// The stream ends with b"PFEN" + u64 bytes_read + u64 packets_read.
#[derive(Default)]
struct Chunk {
    strings: Vec<String>,
    windows: Vec<u8>,
    n_windows: u32,
    hostwins: Vec<u8>,
    n_hostwins: u32,
    flows: Vec<u8>,
    n_flows: u32,
}

struct Interner {
    ids: FxMap<Ip, u32>,
}

impl Interner {
    fn id(&mut self, ip: Ip, chunk: &mut Chunk) -> u32 {
        let n = self.ids.len() as u32;
        *self.ids.entry(ip).or_insert_with(|| {
            chunk.strings.push(ip.to_py_string());
            n
        })
    }
}

fn write_chunk(out: &mut impl Write, c: &mut Chunk) -> io::Result<()> {
    out.write_all(b"PFC1")?;
    out.write_all(&(c.strings.len() as u32).to_le_bytes())?;
    for s in &c.strings {
        out.write_all(&(s.len() as u16).to_le_bytes())?;
        out.write_all(s.as_bytes())?;
    }
    out.write_all(&c.n_windows.to_le_bytes())?;
    out.write_all(&c.windows)?;
    out.write_all(&c.n_hostwins.to_le_bytes())?;
    out.write_all(&c.hostwins)?;
    out.write_all(&c.n_flows.to_le_bytes())?;
    out.write_all(&c.flows)?;
    *c = Chunk::default();
    Ok(())
}

fn emit_group(group: Vec<HostWindow>, bucket: i64, c: &mut Chunk, it: &mut Interner) {
    // {host: (flows, feats) for _, host, flows, feats in group}: a repeated host
    // keeps its FIRST position and takes its LAST value.
    let mut order: Vec<u32> = Vec::new();
    let mut slot: FxMap<u32, usize> = FxMap::default();
    let mut vals: Vec<Option<HostWindow>> = Vec::new();
    for hw in group {
        match slot.get(&hw.host_key) {
            Some(&k) => vals[k] = Some(hw),
            None => {
                slot.insert(hw.host_key, vals.len());
                order.push(hw.host_key);
                vals.push(Some(hw));
            }
        }
    }
    c.windows.extend_from_slice(&bucket.to_le_bytes());
    c.windows.extend_from_slice(&(vals.len() as u32).to_le_bytes());
    c.n_windows += 1;
    for v in vals.into_iter() {
        let hw = v.unwrap();
        c.hostwins.extend_from_slice(&hw.host.to_le_bytes());
        c.hostwins.extend_from_slice(&(hw.flows.len() as u32).to_le_bytes());
        for x in hw.feats {
            c.hostwins.extend_from_slice(&x.to_le_bytes());
        }
        c.n_hostwins += 1;
        for f in &hw.flows {
            let s = it.id(f.src, c);
            let d = it.id(f.dst, c);
            for x in [s, d, f.sport as u32, f.dport as u32, f.proto as u32] {
                c.flows.extend_from_slice(&x.to_le_bytes());
            }
            c.flows.extend_from_slice(&f.first_ts.to_le_bytes());
            c.flows.extend_from_slice(&f.last_ts.to_le_bytes());
            for x in [f.fwd_bytes, f.bwd_bytes, f.fwd_pkts, f.bwd_pkts] {
                c.flows.extend_from_slice(&x.to_le_bytes());
            }
            c.n_flows += 1;
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "ntop6" {
        // Test hook: print inet_ntop6 of 32-hex-digit addresses.
        for h in &args[2..] {
            let mut b = [0u8; 16];
            for i in 0..16 {
                b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
            }
            println!("{}", ntop6(&b));
        }
        return Ok(());
    }
    if args.len() < 3 {
        eprintln!("usage: pcap_fast <window_seconds> <max_packets|0> (<host> <path>)...");
        std::process::exit(2);
    }
    let window: f64 = args[1].parse().expect("window_seconds");
    let max_packets: u64 = args[2].parse().expect("max_packets");
    let pairs = &args[3..];
    assert!(pairs.len() % 2 == 0, "arguments after max_packets must be <host> <path> pairs");
    let mut key_of: HashMap<&str, u32> = HashMap::new();
    let mut paths: Vec<&String> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    for (i, pr) in pairs.chunks(2).enumerate() {
        keys.push(*key_of.entry(pr[0].as_str()).or_insert(i as u32));
        paths.push(&pr[1]);
    }

    let mut stats = Stats::default();
    let mut streams: Vec<HostStream> = Vec::with_capacity(paths.len());
    for (i, p) in paths.iter().enumerate() {
        let reader = match PcapReader::open(p, max_packets) {
            Open::Ok(r) => Some(r),
            Open::Empty => None,
            Open::FormatError(e) => {
                eprintln!("pcap_fast: skipping unreadable capture {p}: {e}");
                None
            }
        };
        streams.push(HostStream { host: i as u32, host_key: keys[i], reader, window, cur_bucket: None,
            table: FlowTable::default(), buf: Vec::new(), finished: false });
    }

    // heapq.merge(*streams, key=bucket): pops the smallest (bucket, stream order).
    let mut heads: Vec<Option<HostWindow>> = Vec::with_capacity(streams.len());
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    for (i, s) in streams.iter_mut().enumerate() {
        let h = s.next(&mut stats);
        if let Some(hw) = &h {
            heap.push(Reverse((hw.bucket, i)));
        }
        heads.push(h);
    }

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut chunk = Chunk::default();
    let mut interner = Interner { ids: FxMap::default() };
    let mut group: Vec<HostWindow> = Vec::new();
    let mut group_bucket: Option<i64> = None;

    while let Some(Reverse((bucket, i))) = heap.pop() {
        let hw = heads[i].take().unwrap();
        let nxt = streams[i].next(&mut stats);
        if let Some(n) = &nxt {
            heap.push(Reverse((n.bucket, i)));
        }
        heads[i] = nxt;
        // itertools.groupby: consecutive equal buckets form one group.
        if group_bucket != Some(bucket) {
            if let Some(gb) = group_bucket {
                emit_group(std::mem::take(&mut group), gb, &mut chunk, &mut interner);
                if chunk.n_flows >= 1_000_000 {
                    write_chunk(&mut out, &mut chunk)?;
                }
            }
            group_bucket = Some(bucket);
        }
        group.push(hw);
    }
    if let Some(gb) = group_bucket {
        emit_group(group, gb, &mut chunk, &mut interner);
    }
    if chunk.n_windows > 0 || !chunk.strings.is_empty() {
        write_chunk(&mut out, &mut chunk)?;
    }
    for s in &streams {
        if let Some(r) = &s.reader {
            stats.bytes += r.bytes;
            stats.packets += r.n;
        }
    }
    out.write_all(b"PFEN")?;
    out.write_all(&stats.bytes.to_le_bytes())?;
    out.write_all(&stats.packets.to_le_bytes())?;
    out.flush()
}
