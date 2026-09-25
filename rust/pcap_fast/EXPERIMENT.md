# pcap_fast: a bit-exact Rust port of the PCAP ingest hot path (experiment)

**Result:** on every input tested, the Rust port gives output bit-identical to
the Python reference, from the packet-level features through to the encoder
columns. The ingest cost drops by roughly 12x in wall time and 12-21x in CPU.
This is a feasibility study. It is **not** wired into training.

## What was ported

`rust/pcap_fast/src/main.rs` is a standalone binary with no dependencies. It
needs no PyO3 or maturin, and nothing was installed. It ports:

| Python reference | Rust |
|---|---|
| `pcap_adapter.PcapFlowExtractor._iter_packets`: LE/BE, µs/ns, pcapng → skip, not-a-pcap → skip, `<24 B` header → empty, and stopping at `incl==0`, `incl>262144` or a short read, which keeps the valid prefix; `max_packets` | `PcapReader` |
| `sniffer.StreamingPacketSniffer.parse_frame`: VLAN, IPv4 (including Python's slicing when `ihl` is 0 or too large), IPv6, TCP/UDP, short L4 | `parse_frame` |
| `socket.inet_ntop` IPv6 text | `ntop6`, a port of glibc `inet_ntop6` (3,011 random or edge addresses matched CPython) |
| `flow_table.LiveFlowTable.process_packet` / `snapshot_flows`: dict insertion order, fwd/rev key | `FlowTable` |
| `pcap_engine.LivePCAPEngine.extract_features`: all 30 features, with session, retransmission and handshake logic, the HTTP check on the ascii-filtered raw frame, and entropies in dict first-occurrence order | `extract_features` |
| `pcap_bridge._host_window_stream`: `int(ts // w)` via CPython's `float_floor_div` algorithm, not `floor(ts/w)` | `HostStream`, `py_floordiv` |
| `iter_merged_day_windows`: `heapq.merge` ties on stream order, `groupby` on consecutive buckets, and the `{host: ...}` dict. A repeated host string keeps its **first** position and takes its **last** value. This occurs in the corpus: `-part1`/`-part2` and `- Copy` files | heap on `(bucket, stream_idx)`, `emit_group` |

The Rust binary writes a chunked columnar stream to stdout: windows, host-windows
carrying the 30 features, and flows. `pcap_fast_py.py` reads that stream and
does the rest in Python:

* `iter_merged_day_windows_fast`: the same yields as the reference. It rebuilds
  the dicts, so it exists only to check stage 1.
* `parse_pcap_day_columns_fast`: the `_parse_one("PCAP2018", ...)` columns. It
  labels with the same `derive_windows`, `check_label_day`, `interval_at`,
  `LabelResolver.resolve` (called once per window, as the reference does) and
  participant scoping. It assigns local ids in encounter order, vectorised. The
  12 edge features are computed with numpy, vectorised.

Label attachment and the edge features stay in numpy on purpose. On this CPU,
numpy 1.26's float64 `log1p` is the AVX-512 SVML kernel. It differs from
glibc's `log1p`, which is what Rust would call, on about 25% of inputs (14,640
of 57,143 sampled). numpy's scalar and vector `log1p` agree with each other,
because both go through SVML, so the vectorised numpy path is exact. Moving the
features into Rust would mean porting SVML.

### Float arithmetic reproduced exactly

* `np.mean` and `np.std` use numpy's pairwise summation. That is 8 accumulators
  for blocks of 128 or fewer, recursive halving rounded to a multiple of 8, a
  `+=` tail below 8, and an identity of 0.0. On top of that, the reduction is
  fed in **8192-element iterator-buffer blocks**, which are summed sequentially.
  That detail made a difference: without it, 476 of 3,000 random arrays longer
  than 8,192 came out wrong. With it, all matched. `std` is `sqrt(sum((x-mean)²)/n)`,
  with the mean computed the same way.
* Integer-valued means (ttl, window, payload, bool ratios) are exact under any order.
* Entropies follow Python's dict insertion order, and `-= p*log2(p)` uses libm
  `log2`, which is what Python calls too.

## Bit-exactness results

The checks are in `verify.py`. Stage 1 compares every window in lockstep. Floats
are compared by `float.hex()` and types are included. Stage 2 runs the **real**
`parallel_ingest._parse_one` and compares with `np.array_equal` on raw bit
patterns and dtypes.

| Input | Windows | Host-windows | Flows / records | Stage 1: bucket, host, all flow fields, 30 features | Stage 2: u, i, ts, lbl, edge, ips, cats |
|---|---:|---:|---:|---|---|
| Synthetic edge day (`wed_14` clock). Covers runts, ARP, short IPv4, `ihl` 0/3/15, TCP under 20 B, odd data offsets, UDP length 0/7/3000, IPv6 including `::ffff:` and `::a.b.c.d`, VLAN, padding, crafted HTTP-verb frames, handshakes, retransmissions, out-of-order timestamps across buckets, a BE file, a ns file, a corrupt length, a short tail, a zero length, an empty file, a pcapng magic, and a duplicate host (`- Copy`) | 310 | 513 | 18,341 | exact | exact |
| Real `fri_16` mini-day. Includes a known-corrupt file (bad length at 6.9 MB), both pcapng `UCAP172.31.69.25-part1/2.pcap`, a 40 MB prefix cut mid-record, slices across the SlowHTTPTest start, and junk, tiny and `.lnk` files | 4,448 | 5,484 | 10,479 | exact | exact |
| Real `tue_20` slices across the LOIC start. The intervals here have participants, so scoped labels are exercised | 85 | 156 | 10,268 | exact | exact |
| `dry_run_plan.build_corpus`: the fri_16, thu_15, wed_28 (CSV alias) and tue_20 days | 45 each | 180 each | 690 each | exact | exact |
| Real `thu_22`, 9 full hosts, 1.58 GB, including the corrupt `capDESKTOP-AN3U28N-172.31.64.65` | 16,724 | 56,938 | 195,097 | exact | exact |
| Real `tue_20` DDoS-dense: a 370 MB victim slice plus 2 full hosts, 2.2 M packets | 7,484 | 8,907 | 620,538 | exact | exact |

**No field failed to match.** Some things were not exercised by real data:
* `http_request_count` and its two means are always 0 on real frames, because
  line 0 is the Ethernet header. They were exercised only by the crafted frames.
* `max_packets_per_host` is implemented but has no dedicated test. Training
  never sets it.
* The `CYBERWORLD_ABLATE_EDGE_FEATURES` mask is applied the same way, but was
  tested only unset.

## Throughput

The numbers come from `bench.py`. Each measurement is a single process, with
page cache warm and the memprobe training job running in parallel. MB/s is
over on-disk file size, including the unread tail of corrupt files.

| Set | Python `iter_merged_day_windows` | Python `_parse_one` (full) | Rust binary alone | Rust + Python columns (full) | Full-path speedup |
|---|---:|---:|---:|---:|---:|
| thu_22, 1.58 GB, 2.06 M packets (bulk, ~730 B/pkt) | 72 MB/s | 60 MB/s (26.4 s) | 706 MB/s (2.2 s) | 673 MB/s (2.35 s; CPU 2.17 s Rust + 0.09 s Python) | **11x wall / 12x CPU** |
| tue_20 dense, 0.63 GB, 2.2 M packets (DDoS, small pkts) | 32 MB/s | 19 MB/s (33.0 s) | 732 MB/s (0.87 s) | ~240 MB/s (CPU 0.94 s Rust + 0.64 s Python) | **~13x wall / 21x CPU** |

Per packet, the Python reference costs about 9 µs to reach stage 1 and about
15 µs end to end on dense traffic. Rust costs about 0.4 µs per packet. On dense
days the remaining Python column assembly, about 1 µs per record, is now of the
same order as the Rust parse. Both sides exclude the label-CSV parse
(`derive_windows`), which is a per-day constant: 5 s for thu_22 and 46 s for
tue_20. After this change that constant becomes a noticeable share of a day's
ingest.

## Recommendation

**Worth integrating**, provided a parity gate goes with it. At about 700 MB/s
per core, the 482 GB corpus is roughly 12 core-minutes of parsing instead of
2.5 to 7 core-hours, and the Rust side's memory is small. It holds one pending
host-window per host plus a 256 KB read buffer per file, so about 115 MB for 445 hosts.

What remains:
1. **Integration.** Add a `PCAP2018` branch in `parallel_ingest._parse_one` that
   calls `parse_pcap_day_columns_fast` and writes the same `.npy` parts. Keep it
   behind a flag, with the Python path as the default until the check in step 2
   has passed.
2. **Full-corpus parity run, once.** Run `verify.py` stage 2 on every day and
   store the hashes. It was verified here on about 2.3 GB, not all 482 GB.
3. **Build and distribution.** Handle `cargo build --release` in the run
   scripts. Consider PyO3 only if the pipe ever matters; today it does not.
4. **Resolver coverage counters.** `resolve` is called once per window, as the
   reference does, so `_parse_one`'s coverage delta is unchanged. Keep it that way.

Risks:
* **numpy version coupling.** The 30 packet features depend on numpy 1.26's
  reduction internals: pairwise blocks and the 8192 buffer size. A numpy upgrade,
  or a call to `np.setbufsize`, could change the reference's own last bits. The
  Rust side would then silently disagree. Keep `verify.py` as a CI gate pinned
  to the numpy version in use. The edge features have the same coupling to the
  numpy/SVML build, which is why they stay in numpy.
* **Two implementations of one spec.** Any later edit to `sniffer.py`,
  `flow_table.py`, `pcap_engine.py` or `pcap_bridge.py` must be mirrored in Rust.
  The parity test is the only thing that will catch a missed mirror.
* **Error behaviour.** Python lets I/O errors (OSError) propagate. Rust panics
  on them, and the wrapper raises `RuntimeError`. Both fail loudly, but with
  different exception types.
* **Reference quirks are reproduced, not fixed.** Examples: a SYN with payload
  counts as a retransmission, HTTP detection looks at the Ethernet header, and
  DNS domains are never extracted. Fixing any of them changes the training data
  and must be done in both implementations.

## Files

* `rust/pcap_fast/src/main.rs`: the port.
* `pcap_fast_py.py`: stream reader, stage-1 adapter, stage-2 column builder.
* `verify.py` / `run_verify.sh`: the parity checks, run under a 4 GB cap.
* `bench.py`: throughput.
* `build_testdata.py`: test inputs, written to `~/Documents/SIH/rust_parser_experiment/`
  (about 420 MB: symlinks, prefix and slice copies, synthetic files). Build output
  goes to `CARGO_TARGET_DIR=~/Documents/SIH/rust_parser_experiment/target`.

To reproduce:

```
CARGO_TARGET_DIR=~/Documents/SIH/rust_parser_experiment/target cargo build --release -j4
python3 rust/pcap_fast/build_testdata.py
bash rust/pcap_fast/run_verify.sh ~/Documents/SIH/DATA/CSV ~/Documents/SIH/rust_parser_experiment/days/*_pcap
```
