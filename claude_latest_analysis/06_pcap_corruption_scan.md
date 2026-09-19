# CIC-IDS-2018 PCAP — full corruption scan

**Date:** 2026-09-19
**Scope:** every file under `~/Documents/SIH/DATA/pcap/` — **complete census, not sampled**
**Tool:** `capinfos -c -a -e` (Wireshark 4.6.8) on each file, 6-way parallel, run inside the `prism-dev` toolbox
**Raw output:** [`pcap_scan_results.tsv`](pcap_scan_results.tsv) — `status / size / packets / path / error`

---

## Result

| | Files | Size | Action |
|---|---:|---:|---|
| **OK** | 4,118 | — | none |
| **SEVERE** — record-chain corruption | **263** | **56.2 GB** | **re-download** |
| **MINOR** — truncated tail | 75 | 11.8 GB | none needed |
| **Total scanned** | 4,456 | ~560 GB | |

One non-capture file was excluded: `fri_2_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.24 - Shortcut.lnk` (a Windows shortcut).

**Re-download list:** [`corrupt_files_redownload.txt`](corrupt_files_redownload.txt) — 263 paths, relative to `~/Documents/SIH/DATA/pcap/`.

---

## The two failure modes are not the same thing

This distinction matters, because conflating them triples the apparent re-download size.

### SEVERE — 263 files, re-download these

```
capinfos: An error occurred after reading 127898 packets from "...".
capinfos: The file "..." appears to be damaged or corrupt.
(pcap: File has 1312894288-byte packet, bigger than maximum of 262144)
```

A packet-length field holding garbage — 1.31 GB, 3.27 GB — where a real length belongs. The parser cannot resynchronise, so **everything after the bad record is unreachable**. Coverage is genuinely lost.

### MINOR — 75 files, leave them alone

```
capinfos: The file "..." appears to have been cut short in the middle of a packet.
(will continue anyway, checksums might be incorrect)
```

Only the final partial packet is missing. `capinfos` still returns complete results. Example — `tue_20_pcap/pcap/UCAP172.31.69.18`: 9,627 packets, 2018-02-20 18:07:54 → 2018-02-21 02:57:46, **8.83 h of full coverage**. Re-downloading these buys nothing. Listed separately in [`corrupt_files_minor_no_action.txt`](corrupt_files_minor_no_action.txt).

---

## How much of each severe file survives

Every severe file has a usable prefix — none died at the header:

| Packets read before failure | Files | Practical meaning |
|---|---:|---|
| 0 (dead at header) | **0** | — |
| < 1,000 | **0** | — |
| 1,000 – 100,000 | 107 | partial day |
| > 100,000 | 156 | most of the day usable |

`tshark` reads the valid prefix of every one of them. If you are time-constrained, ingesting prefixes is a legitimate fallback — you would lose a variable tail per host-day, not a whole day.

---

## Corruption clusters by day

| Day | Severe | Total | Rate |
|---|---:|---:|---:|
| `wed_14_pcap` | 55 | 449 | **12.2%** |
| `fri_16_pcap` | 45 | 444 | **10.1%** |
| `thu_15_pacap` | 43 | 450 | 9.6% |
| `thu_1_pcap` | 30 | 443 | 6.8% |
| `fri_23_pacap` | 28 | 446 | 6.3% |
| `wed_21_pcap` | 18 | 447 | 4.0% |
| `fri_2_pcap` | 17 | 442 | 3.8% |
| `thu_22_pcap` | 12 | 447 | 2.7% |
| `wed_28_pcap` | 10 | 437 | 2.3% |
| `tue_20_pcap` | 5 | 451 | 1.1% |

An 11× spread between best and worst day. Every day is affected; none is clean.

## But it also clusters by host — and that complicates the re-download

**19 hosts are corrupt on 5 or more of the 10 days.** The worst:

| Host capture | Corrupt on |
|---|---:|
| `capDESKTOP-AN3U28N-172.31.66.115` | 8 of 10 days |
| `capDESKTOP-AN3U28N-172.31.67.82` | 7 |
| `capDESKTOP-AN3U28N-172.31.67.81` | 7 |
| `capDESKTOP-AN3U28N-172.31.67.28` | 6 |
| `capDESKTOP-AN3U28N-172.31.67.13` | 6 |

If corruption were purely transfer damage, it would scatter across hosts at random. A host failing on 8 of 10 independent day-directories suggests **the damage may be upstream** — in the published dataset itself, or in whatever produced these files.

**Verify before pulling 56 GB:** re-fetch one file from the worst host — `wed_14_pcap/pcap/capDESKTOP-AN3U28N-172.31.66.115` — and run `capinfos` on the fresh copy. If it is still corrupt, re-downloading the other 262 will not help either, and the prefix-ingestion fallback becomes the plan.

---

## Correction to report 05

Report 05 concluded *"integrity is excellent — only 4 files (0.09%) lost real coverage."* **That is wrong. The real figure is 263 files (5.9%).**

The cause was a methodology gap, not carelessness — report 05 flagged its own limits. It used two screens:

1. a `size % 4096` test for interrupted copies, and
2. a tail-seek to read the last packet header.

Both are structurally blind to this failure. The corruption is a bad length field *mid-file*; the bytes at the tail are still valid packet records, so a tail-seek lands cleanly and reports a normal ~9-hour span. Only walking the record chain from the start reveals it — which needs a full read, which is exactly what report 05 could not afford before `capinfos` was available.

**What still stands from report 05:** all 10 capture days are present, the corpus is split per host rather than per time slice, clean captures run ~9 hours, snaplen is 65535 throughout, fragment files reassemble into full days, and the engineers' *"only one or two hours of data"* claim remains false.

**What changes:** integrity is good-but-not-excellent, and 56.2 GB needs attention before the corpus is treated as a reliable training source.

---

## Method and limits

- **Complete census** — every one of 4,456 capture files was opened and its record chain walked by `capinfos`. No sampling.
- Classification keys on `capinfos`'s own error strings; the two signatures above were the only two that appeared (verified — zero unmatched).
- **Not checked:** payload contents, checksums, and whether any capture actually contains the expected attack traffic. A file reported OK here is structurally sound, which is not the same as semantically complete.
- Salvageability is inferred from packets-read-before-failure against file size; the per-file *time* coverage retained was not computed for all 263.
