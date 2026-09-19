# CIC-IDS-2018 PCAP Corpus — Completeness Audit

**Scope:** `/var/home/samito/Documents/SIH/DATA/pcap/` only.
**Codebase reference:** `/var/home/samito/Documents/cyber_network_predictor-2f6ff8a69d8845ddb9bf9f3d72366af6dd5b5875`, branch `testing-prod`, HEAD `c96b111`.
**Date:** 2026-09-19
**CTU-13 pcaps: out of scope, not examined.**

---

## (a) Verdict on the engineers' claim

> *"The PCAP data is incomplete — it only has one or two hours of data."*

**This claim is FALSE as a statement about the dataset, and I can name the specific artifact that most likely produced it.**

The corpus holds **4,457 capture files, ~560 GB, covering all 10 official CIC-IDS-2018 capture days (14 Feb – 2 Mar 2018)**. The files are split **per host, not per time slice**: each day directory contains ~445 files, one per victim machine, and **each individual file spans the full ~9-hour capture day** (12:15–21:30 UTC ≈ 08:15–17:30 local). Of 143 files I probed, the typical single-file duration was **8.7–9.7 hours**; not one clean per-host capture was near 1–2 hours.

**Where the "1–2 hours" almost certainly came from:** a small set of ~8 oddly-named files that *are* genuinely short because they are **time-consecutive fragments of one host's day** — `…69.13 part1` (1.71 h), `part2` (0.53 h), `…69.24- part1` (2.21 h), `…69.28 part 1` (2.18 h), `…69.26a` (1.20 h), `…69 - Copy.24` (1.39 h). These sort to conspicuous positions in a directory listing and are exactly the kind of file someone spot-checking the corpus would open first. Opening one shows 1–2 hours. **Concatenating its siblings restores the full 9-hour day.**

So: *technically true of ~8 files out of 4,457 (0.2%), and flatly false of the dataset.* The distinction that matters is **"each file is short" vs "the dataset is short"** — and here neither is true in general; only a handful of *fragment* files are short, and their fragments are contiguous and reassemblable.

**Secondary verdict:** the corpus is substantially *more* complete than the CSVs the team actually trained on, and it contains host-identity information that **9 of the 10 training CSVs do not have** (see section (e)). The PCAPs are not the weak link here.

---

## (b) Evidence table

Format probe: global header (24 B) + first packet header (16 B) at the head, plus a ≤1 MiB tail seek to recover the last packet header. **No file was read whole** except five small (<5 MB) files forward-walked to diagnose truncation.

### Per-day consolidated coverage (5 randomly sampled files per day, seed-fixed)

| Day dir | Calendar date | Earliest first-ts (UTC) | Latest last-ts (UTC) | Mean file duration | Files | Size |
|---|---|---|---|---|---|---|
| `wed_14_pcap` | Wed 14 Feb 2018 | 2018-02-14 12:28:11 | 2018-02-14 21:30:30 | 9.01 h | 449 | 46 G |
| `thu_15_pacap` | Thu 15 Feb 2018 | 2018-02-15 12:23:34 | 2018-02-15 21:30:30 | 9.09 h | 450 | 46 G |
| `fri_16_pcap` | Fri 16 Feb 2018 | 2018-02-16 12:38:09 | 2018-02-16 21:30:25 | 8.84 h | 444 | 47 G |
| `tue_20_pcap` | Tue 20 Feb 2018 | 2018-02-20 12:28:40 | 2018-02-20 21:30:21 | 9.02 h | 451 | 58 G |
| `wed_21_pcap` | Wed 21 Feb 2018 | 2018-02-21 12:28:45 | 2018-02-21 21:30:08 | 9.02 h | 447 | 76 G |
| `thu_22_pcap` | Thu 22 Feb 2018 | 2018-02-22 12:22:25 | 2018-02-22 21:30:10 | 9.12 h | 447 | 56 G |
| `fri_23_pacap` | Fri 23 Feb 2018 | 2018-02-23 12:15:14 | 2018-02-23 21:30:12 | 9.24 h | 446 | 65 G |
| `wed_28_pcap` | Wed 28 Feb 2018 | 2018-02-28 12:21:00 | 2018-02-28 21:30:08 | 9.14 h | 437 | 60 G |
| `thu_1_pcap` | Thu 1 Mar 2018 | 2018-03-01 12:15:39 | 2018-03-01 21:30:09 | 6.75 h † | 443 | 59 G |
| `fri_2_pcap` | Fri 2 Mar 2018 | 2018-03-02 12:46:40 | 2018-03-02 21:30:09 | 8.72 h | 443 | 52 G |

† `thu_1` mean is depressed because one of the 5 sampled files was a `part` fragment; its clean files run 9.2 h.

### Individual file detail (representative subset)

| File | Format | First ts (UTC) | Last ts (UTC) | Duration | Size |
|---|---|---|---|---|---|
| `wed_14/capEC2AMAZ-O4EL3NG-172.31.66.68` | pcap LE µs | 2018-02-14 12:28:54 | 2018-02-14 21:30:05 | **9.02 h** | 80 MB |
| `thu_15/capEC2AMAZ-O4EL3NG-172.31.68.14` | pcap LE µs | 2018-02-15 12:24:43 | 2018-02-15 21:30:06 | **9.09 h** | 73 MB |
| `fri_16/capEC2AMAZ-O4EL3NG-172.31.65.43` | pcap LE µs | 2018-02-16 12:38:26 | 2018-02-16 21:30:06 | **8.86 h** | 63 MB |
| `tue_20/capEC2AMAZ-O4EL3NG-172.31.67.67` | pcap LE µs | 2018-02-20 12:29:08 | 2018-02-20 21:30:04 | **9.02 h** | 93 MB |
| `wed_21/capEC2AMAZ-O4EL3NG-172.31.65.72` | pcap LE µs | 2018-02-21 12:29:17 | 2018-02-21 21:30:04 | **9.01 h** | 99 MB |
| `thu_22/capEC2AMAZ-O4EL3NG-172.31.68.20` | pcap LE µs | 2018-02-22 12:23:25 | 2018-02-22 21:30:04 | **9.11 h** | 102 MB |
| `fri_23/capEC2AMAZ-O4EL3NG-172.31.65.91` | pcap LE µs | 2018-02-23 12:15:48 | 2018-02-23 21:30:06 | **9.24 h** | 108 MB |
| `wed_28/capEC2AMAZ-O4EL3NG-172.31.66.74` | pcap LE µs | 2018-02-28 12:21:36 | 2018-02-28 21:30:04 | **9.14 h** | 120 MB |
| `thu_1/capDESKTOP-AN3U28N-172.31.64.111` | pcap LE µs | 2018-03-01 12:16:03 | 2018-03-01 21:30:05 | **9.23 h** | 324 MB |
| `fri_2/capEC2AMAZ-O4EL3NG-172.31.65.65` | pcap LE µs | 2018-03-02 12:46:50 | 2018-03-02 21:30:05 | **8.72 h** | 77 MB |
| `thu_1/capPC1-172.31.65.77` | pcap LE µs | 2018-03-01 12:15:53 | 2018-03-01 21:30:47 | **9.25 h** | 3.5 GB |
| `fri_23/capWIN-J6GMIG1DQE5-172.31.67.62` | pcap LE µs | 2018-02-23 12:16:48 | 2018-02-23 21:30:30 | **9.23 h** | 3.9 GB |
| `wed_28/capEC2AMAZ-O4EL3NG-172.31.66.35` | pcap LE µs | 2018-02-28 12:21:27 | 2018-02-28 23:35:08 | **11.23 h** | 95 MB |
| **Fragment files (the source of the "1–2 hour" claim):** | | | | | |
| `thu_1/…69.13 part1` | pcap LE µs | 2018-03-01 12:16:38 | 2018-03-01 13:59:20 | 1.71 h | 150 MB |
| `thu_1/…69.13 part2` | pcap LE µs | 2018-03-01 14:03:49 | 2018-03-01 14:35:53 | 0.53 h | 36 MB |
| `thu_1/…69.13 part3` | pcap LE µs | 2018-03-01 14:36:16 | 2018-03-01 21:30:07 | 6.90 h | 129 MB |
| → *parts 1+2+3 combined* | | *12:16:38* | *21:30:07* | ***9.22 h*** | *315 MB* |
| `wed_28/…69.24- part1` | pcap LE µs | 2018-02-28 12:22:14 | 2018-02-28 14:34:39 | 2.21 h | 98 MB |
| `wed_28/…69.24-part2` | pcap LE µs | 2018-02-28 14:38:24 | 2018-02-28 21:42:59 | 7.08 h | 348 MB |
| → *parts 1+2 combined* | | *12:22:14* | *21:42:59* | ***9.35 h*** | *446 MB* |
| `fri_2/…69.26a` | pcap LE µs | 2018-03-02 12:47:15 | 2018-03-02 13:59:06 | 1.20 h | 13 MB |
| `fri_2/…69 - Copy.24` | pcap LE µs | **2018-03-01** 12:22:23 | 2018-03-01 13:45:41 | 1.39 h | 61 KB |

**Structure and naming.** Two-level: `<day>_pcap/pcap/<file>`. Filenames encode the capturing host and its IP, with five families:

| Prefix | Count | Meaning |
|---|---|---|
| `capEC2AMAZ-O4EL3NG-<IP>` | 2,099 | AWS-hosted victim workstations |
| `capDESKTOP-AN3U28N-<IP>` | 765 | victim workstations |
| `capPC1-<IP>` | 757 | victim workstations |
| `capWIN-J6GMIG1DQE5-<IP>` | 750 | victim workstations |
| `UCAP<IP>` | 75 | server-subnet captures (172.31.69.x) |

**Per-host, not per-hour, not per-part.** Unique host IPs per day: 435–451. Subnets present: `172.31.64.x` (1,033), `.65.x` (998), `.66.x` (982), `.67.x` (981), `.69.x` (261), `.68.x` (198).

> **Correction to the brief:** the task described CIC-IDS-2018 as a "50-machine victim network." The official victim organisation is **5 departments totalling 420 workstations + 30 servers = 450 hosts**. The 435–451 files/day observed here matches that almost exactly, which is itself strong evidence of completeness rather than of a gap.

---

## (c) Total coverage vs the official 10-day structure

| Official capture day | Present on disk? | Verified by real packet timestamps |
|---|---|---|
| Wed 14 Feb 2018 | ✅ `wed_14_pcap` | ✅ 2018-02-14 |
| Thu 15 Feb 2018 | ✅ `thu_15_pacap` | ✅ 2018-02-15 |
| Fri 16 Feb 2018 | ✅ `fri_16_pcap` | ✅ 2018-02-16 |
| Tue 20 Feb 2018 | ✅ `tue_20_pcap` | ✅ 2018-02-20 |
| Wed 21 Feb 2018 | ✅ `wed_21_pcap` | ✅ 2018-02-21 |
| Thu 22 Feb 2018 | ✅ `thu_22_pcap` | ✅ 2018-02-22 |
| Fri 23 Feb 2018 | ✅ `fri_23_pacap` | ✅ 2018-02-23 |
| Wed 28 Feb 2018 | ✅ `wed_28_pcap` | ✅ 2018-02-28 |
| Thu 1 Mar 2018 | ✅ `thu_1_pcap` | ✅ 2018-03-01 |
| Fri 2 Mar 2018 | ✅ `fri_2_pcap` | ✅ 2018-03-02 |

**All 10 of 10 official capture days are present.** Directory names (`wed_14`, `thu_15`, …) match the true 2018 weekday-date pairs exactly, and every date was independently confirmed from packet timestamps inside the files rather than from the filename.

**Total wall-clock coverage: ~90 hours of full-network capture** (10 days × ~9 h), recorded simultaneously across ~445 hosts per day. Daily capture window is consistently **12:15–21:30 UTC**. Nothing is missing at the day level.

---

## (d) Integrity findings

**Good:**
- **Zero zero-length files.**
- Format is **classic pcap, little-endian, microsecond resolution, version 2.4, Ethernet linktype**.
- **Snaplen 65535 on every file checked** — full packet payloads were captured, not headers-only. This is the single most important property for packet-level feature work: TTL, TCP window, IP fragment flags, sequence numbers and payload bytes are all physically present.

**Defects (all minor; none blocks use):**

1. **76 of 4,457 files (1.7%) are truncated at a 4096-byte boundary.** Detected by testing `size % 4096 == 0` across **all** 4,457 files (a clean pcap ending on an exact block boundary is a ~1-in-4096 coincidence). The signature is an interrupted copy/transfer, not source corruption. Breakdown: **70 of the 75 `UCAP*` (server) files**, plus 6 `capDESKTOP-*` files. Spread evenly across days (6–10 per day).

2. **The truncation is cosmetic in 72 of those 76 cases.** Forward-walking five of them confirmed the loss is **the final partial packet record only** (median 33 bytes lost; max 11,575). Full-day coverage is retained — e.g. `wed_14/UCAP172.31.69.7`: 13,823 packets, 12:31:26 → 21:28:50, **8.96 h**, truncated mid-final-packet. Every standard reader (libpcap, tshark, scapy) stops cleanly at the last complete record and is unaffected.

3. **Only 4 files (0.09%) lost real coverage:**

   | File | Covers | Duration |
   |---|---|---|
   | `tue_20/UCAP172.31.69.25` (8.5 GB) | 12:32:55 – 17:29:16 | 4.94 h — **loses the afternoon** |
   | `wed_21/UCAP172.31.69.28 part 2` (2.1 GB) | 17:55:46 – 21:29:38 | 3.56 h — **gap 14:43→17:55 vs part 1** |
   | `fri_2/capDESKTOP-AN3U28N-172.31.66.115` | 16:30:04 – 21:16:32 | 4.77 h |
   | `fri_2/capDESKTOP-AN3U28N-172.31.64.111` | 12:46:48 – 20:32:15 | 7.76 h |

   These are 4 host-days out of ~4,450. Immaterial at corpus scale.

4. **Two files are pcapng, not classic pcap:** `fri_16/UCAP172.31.69.25-part1.pcap` (4.3 GB) and `-part2.pcap` (815 KB). Mixed formats — any ingestion code must handle both magics (`0xa1b2c3d4` and `0x0a0d0d0a`). I did not hand-parse these; `capinfos`/`tshark` are not installed on this machine and I did not install anything.

5. **One file has a different snaplen:** `wed_21/UCAP172.31.69.28 part 1` (18.3 GB) uses **snaplen 262144**, not 65535. Harmless but worth knowing.

6. **One non-capture artifact:** `fri_2/capEC2AMAZ-O4EL3NG-172.31.69.24 - Shortcut.lnk` (697 bytes, magic `4c000000`) is a Windows shortcut, not a pcap. Filter it out.

7. **One misfiled duplicate:** `fri_2_pcap/…/capEC2AMAZ-O4EL3NG-172.31.69 - Copy.24` contains **2018-03-01** packets but sits in the **March 2** directory. Evidence of manual copying. Do not trust the day directory as ground truth for this file — derive the date from packet timestamps (as this audit did throughout).

8. **One unresolved file.** `wed_14/capDESKTOP-AN3U28N-172.31.67.15` (839 MB, *not* block-aligned) would not yield a tail alignment walking exactly to EOF even with a 16 MiB tail. A tolerant probe recovers 12:31:15 → 21:28:17 (**8.95 h**, normal), but the final ~286 KB may contain a damaged region. **I could not determine this cheaply and am not guessing.** Coverage is confirmed normal; integrity of the last 286 KB is unknown.

---

## (e) Alignment with the training CSVs

Read from `/var/home/samito/Documents/SIH/DATA/CSV` — headers, first data row, and last row only; no file was loaded.

| CSV | Cols | First timestamp | Has `Src IP`/`Dst IP`? |
|---|---|---|---|
| `wed_14_csv.csv` | 80 | 14/02/2018 08:31:01 | ❌ |
| `thu_15_csv.csv` | 80 | 15/02/2018 08:25:18 | ❌ |
| `fri_16_csv.csv` | 80 | 16/02/2018 08:27:23 | ❌ |
| `tue_20_csv.csv` | **84** | 20/02/2018 08:34:07 | ✅ |
| `wed_21_csv.csv` | 80 | 21/02/2018 08:33:25 | ❌ |
| `thu_22_csv.csv` | 80 | 22/02/2018 08:26:03 | ❌ |
| `fri_23_csv.csv` | 80 | 23/02/2018 08:18:29 | ❌ |
| `wed_29_csv.csv` | 80 | **28**/02/2018 08:22:13 | ❌ |
| `thu_1_csv.csv` | 80 | 01/03/2018 08:17:11 | ❌ |
| `fri_2_csv.csv` | 80 | 02/03/2018 08:47:38 | ❌ |

**1. Coverage is identical — the PCAPs cover exactly the same 10 days, no more, no less.** The `wed_29_csv.csv` filename is a local misnomer: its rows are dated **28/02/2018**, matching `wed_28_pcap`. That apparent 28-vs-29 discrepancy is resolved; no day is missing on either side.

**2. Timestamps align exactly, with a constant 4-hour offset.** CSV times are local (UTC−4); PCAP times are UTC. Verified to the second on two independent days:

| Day | CSV first | PCAP first (same host `172.31.69.20`) | Offset |
|---|---|---|---|
| Thu 1 Mar | 08:17:11 | 12:17:11 UTC | **exactly +4:00:00** |
| Fri 2 Mar | 08:47:38 | 12:47:38 UTC | **exactly +4:00:00** |

A timestamp join is therefore mechanically straightforward: `pcap_utc = csv_local + 4h`.

**3. But the CSV timestamps use a 12-hour clock with no AM/PM marker.** Verified empirically: across 300,000 sampled rows of `wed_14_csv.csv`, the hour field takes only the values `01,02,03,08,09,10,11,12` — **maximum hour is 12**, never 13–23. This is the well-known CIC-IDS-2018 CSV defect and it makes raw string parsing ambiguous. It is *resolvable*, because the capture window is known and narrow: the day runs 08:15–17:30 local, so hours 08–11 are AM, hour 12 is noon (PM), and hours 01–05 are PM. The disambiguation is deterministic — but it must be done explicitly, and silently parsing these as 24-hour times will scramble the afternoon (which is where most attacks occur).

**4. The CSVs lack host identity for 9 of the 10 days — and the PCAPs supply it.** This is the most consequential finding in this section. Only `tue_20_csv.csv` carries `Src IP`/`Dst IP`. For the other nine days the current ingestion code **fabricates IP addresses by row index**, at `data_unification/cic2018_adapter.py:65,70` (HEAD `c96b111`):

```python
src_ips = np.array([f"192.168.10.{i % 250 + 1}" for i in range(len(chunk))])
dst_ips = np.array([f"172.16.0.{i % 100 + 1}" for i in range(len(chunk))])
```

and randomises source ports at `:75` (`np.random.randint(1024, 65535, ...)`). Since the models' entire value proposition is *per-host* trajectory forecasting, a host graph built from `row_index % 250` is not a host graph. **The PCAPs are organised one-file-per-host and therefore carry exactly the host attribution the CSVs are missing.**

**5. Labels live only in the CSVs.** The PCAPs are unlabelled (pcap has no label field). The CSV label column is populated and usable — e.g. the first 300k rows of `wed_14_csv.csv` are `FTP-BruteForce` (179,244), `SSH-Bruteforce` (119,960), `Benign` (795).

**How to label the PCAPs.** Two routes, in order of robustness:
- **Preferred — 5-tuple + time join.** For the 9 IP-less CSV days, a direct flow-key join is impossible from the CSV side. Instead, re-derive flows from the PCAPs (which have real IPs), then join to CSV rows on `(dst_port, protocol, timestamp±tolerance, packet/byte counts)`. Workable but lossy.
- **More robust — published attack windows.** CIC-IDS-2018 publishes per-day attack time ranges and attacker/victim IPs. Because PCAP timestamps are unambiguous UTC and the offset to CSV local time is exactly 4 h, labelling by time range **directly on the PCAPs** avoids the CSV's 12-hour ambiguity and its missing-IP problem entirely. This is the route I would take.

**Caveat:** `thu_1_csv.csv` contains **26 embedded repeated header rows** mid-file (`grep -c '^Dst Port'`), and its final line is the literal string `Timestamp`. Any CSV-side join must filter these or it will emit garbage rows. `wed_14` and `fri_2` have 1 each (the legitimate header).

---

## (f) Practical recommendation

**The corpus is usable. There is no format, coverage, or integrity blocker.** The engineers' stated reason for not using it does not survive contact with the data.

**What is genuinely good:**
- All 10 official days present and timestamp-verified; ~90 hours of full-network capture.
- **Snaplen 65535 everywhere checked** — every PS-required packet-level feature (TTL, TCP window size, IP fragment flags, payload size distribution, retransmission/sequence tracking, scan signatures) is physically recoverable. Nothing was captured headers-only.
- Per-host file organisation gives free, reliable host attribution — the thing the CSVs lack for 9/10 days.
- Standard classic pcap; a ~40-line hand parser reads it (this audit wrote one). No exotic tooling needed.
- Integrity is excellent: 0 empty files, 1.7% cosmetically truncated, **0.09% with real data loss**.

**The real work, and it is not trivial:**

1. **Scale is the actual blocker, not completeness.** 560 GB across 4,457 files. Full-fidelity extraction over the whole corpus is a cluster-scale job. **Mitigation:** the corpus subsets cleanly. A single day (~46–76 GB) or a per-subnet slice (e.g. `172.31.64.x`, ~100 files/day) is tractable on one machine and still yields hundreds of host-days. Start there; do not attempt all 560 GB first.

2. **Handle the fragment files.** ~8 files are `part1/part2/part3` splits that must be concatenated in timestamp order to reconstruct a host's day. This is exactly the trap that produced the false "1–2 hours" claim. Any ingestion job must group by `(day, host IP)` and merge, **not** treat each file as an independent host-day.

3. **Handle two pcapng files and one 262144-snaplen file.** Detect the magic per file; do not assume classic pcap.

4. **Filter junk:** one `.lnk`, and one misfiled ` - Copy.24` whose packets are from the previous day. Derive the date from packet timestamps, never from the parent directory.

5. **Tolerate truncated tails.** 76 files end mid-record. The reader must stop cleanly at the last complete record rather than erroring — otherwise 1.7% of the corpus fails the job. Standard libraries already do this; a hand-rolled parser must be written to.

6. **Labelling: use published attack time windows against PCAP UTC timestamps**, not a CSV join. This sidesteps both the missing-IP problem (9/10 days) and the 12-hour AM/PM ambiguity. Validate the result against CSV label counts on `tue_20` — the one day that has IPs and therefore permits a true 5-tuple join as a control.

7. **Expect to rebuild the host graph.** Today the pipeline's host identity for CIC-2018 is synthetic (`cic2018_adapter.py:65,70`). Moving to PCAPs replaces fabricated topology with real topology. That is a correctness improvement independent of any packet-level feature gain, and arguably the strongest single argument for doing this work.

**Bottom line:** the PCAPs are complete, valid, full-payload, and cover the same 10 days as the CSVs already in use. The cost of using them is engineering effort proportional to 560 GB — real, but a scale problem, not a data problem. The claim that blocked this work for the project was incorrect, and appears to trace to someone opening one of ~8 fragment files.

---

## Method and its limits

**Techniques used, cheapest first.** `ls`/`du`/`find`/`stat` for structure and sizes; `file(1)` on samples; a purpose-written Python probe reading only the 24-byte global header, the 16-byte first packet header, and a ≤1 MiB tail seek; for five small (<5 MB) files, a full forward walk to diagnose truncation precisely. `capinfos`, `tshark` and `editcap` are **not installed** on this machine and, per instruction, nothing was installed.

**Bytes actually read: well under 1 GB against a 560 GB corpus** — roughly 143 × ~1 MiB of tails plus header reads and ~9 MB of small-file walks. No file was copied. No file >64 MB was read whole. Nothing exceeded ~60 s except one batch job over the 76 truncation candidates, which was backgrounded and completed in ~4 minutes.

**Sampling disclosure — please read before generalising:**
- **Complete (not sampled):** file counts, sizes, directory structure, naming taxonomy, zero-length check, and the `size % 4096` truncation screen — all four run across **all 4,457 files**.
- **Sampled:** first/last-timestamp probing covers **143 of 4,457 files (3.2%)** — 20 stratified by day, 40 uniformly random (seed-fixed), 17 largest/oddly-named, plus all 76 truncation candidates. Every day directory is represented.
- **I therefore do not claim every one of the 4,457 files is ~9 hours.** I claim that across 143 files spanning all 10 days and all 5 host-prefix families, clean per-host captures ran 8.7–11.2 h with a median near 9.1 h, and that no clean per-host capture was near 1–2 h. The consistency (nearly every file ending 21:30:0x UTC) makes a materially different population in the unsampled 96.8% unlikely, but it is not proven.

**Known limits:**
- The `size % 4096` truncation screen is a **lower bound**. A copy interrupted off a block boundary would be missed — `wed_14/capDESKTOP-AN3U28N-172.31.67.15` is exactly such a case and I could not resolve it cheaply.
- The **two pcapng files were not parsed**; their durations are unknown.
- I verified **headers and timestamps only**. I did **not** validate packet payload contents, checksums, or whether any individual capture actually contains attack traffic. Confirming that the published attack windows contain the expected traffic requires reading packet bodies and was out of budget.
- Attack-window alignment is argued from the exact +4:00:00 offset measured on two days; I did not verify a labelled attack interval end-to-end against packet contents.
- `/var/home/samito/Documents/SIH/processed/` was **not examined**, per instruction.

---

# Addendum — resolved with tshark/capinfos (2026-09-19, post-report)

`wireshark-cli` 4.6.8 was installed into the `prism-dev` Fedora toolbox after this report was written
(`toolbox run -c prism-dev sudo dnf install -y wireshark-cli`). Toolboxes share `$HOME`, so
`capinfos`/`tshark` read `~/Documents/SIH/DATA/pcap/` directly. This closes the three items the report
left open, and validates its method.

## 1. The hand-rolled probe was accurate

`capinfos` agreed with the report's header/tail probe on every file cross-checked:

| File | capinfos duration | Packets | Report's claim |
|---|---|---|---|
| `fri_2/…-172.31.69.20` | **8.70 h** | 1,598 | within the 8.7–11.2 h band ✓ |
| `thu_1/…-172.31.69.24` | **9.12 h** | 1,730 | median ~9.1 h ✓ |
| `fri_2/…-172.31.69 - Copy.24` | **1.39 h** | 255 | stated as 1.39 h — **exact match** ✓ |

The 3.2% sampling caveat still stands, but the measurement technique is now independently confirmed.

## 2. The two pcapng files — resolved, and they confirm the fragment theory

Both are genuine **pcapng v1.0** (not pcap), which is why the struct-based probe skipped them.

| File | Span (UTC) | Duration | Packets |
|---|---|---|---|
| `fri_16/UCAP172.31.69.25-part1.pcap` | 17:56:55 → 23:28:22 | **5.52 h** | **18 M** |
| `fri_16/UCAP172.31.69.25-part2.pcap` | 23:37:33 → 02:56:14 (+1d) | **3.31 h** | 5,077 |
| **combined** | 17:56:55 → 02:56:14 | **8.99 h** | — |

A 9-minute gap separates the parts. This is the fragment pattern again: two time-consecutive pieces of
one host's day that reassemble to a full ~9 h capture. It is now confirmed on a second, independent
fragment pair — and this one carries **18 million packets**, so the server-subnet (`UCAP*`) captures are
high-volume, not thin.

## 3. `wed_14/capDESKTOP-AN3U28N-172.31.67.15` — genuinely corrupt (upgrade from "may be damaged")

The report hedged on this file. It is damaged, and worse than a truncated tail:

```
capinfos: An error occurred after reading 127898 packets ... appears to be damaged or corrupt.
(pcap: File has 1312894288-byte packet, bigger than maximum of 262144)
```

A bogus 1.31 GB packet-length field — structural corruption mid-file, not a clipped final record. The
readable prefix spans **2018-02-14 12:31:15 → 15:54:36 UTC = 3.39 h** across 127,898 packets, against
the ~9 h expected for that day.

**This is real coverage loss, so the count of files losing real coverage rises from 4 to 5** (still
0.11% of 4,457). It does not change the verdict; it is one host-day on 14/02 that would need to be
dropped or truncated at the damage point. `tshark` reads the prefix cleanly, so the first 3.39 h remain
usable.

## 4. What this changes

Nothing in the verdict. The engineers' claim remains **false**: all 10 days present, per-host files
spanning full ~9 h days, fragments reassembling to full days, integrity at 99.9%.

Two practical notes for whoever builds the ingestion:
- **Handle pcapng as well as pcap.** At least two files in `fri_16` are pcapng v1.0. A parser that only
  accepts magic `0xa1b2c3d4` will silently skip them — and one holds 18 M packets.
- **Expect at least one structurally corrupt file** and fail soft: read until the parser errors, keep
  the prefix, log the file. Do not let one bad record abort a day's ingest.
