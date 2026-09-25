# Data integrity audit — every corpus on disk

**Date:** 2026-09-21
**Scope:** CIC-2018 CSV, CIC-2018 PCAP, CTU-13, CIC-2017 (both copies)
**Method:** full row-level scans of all CSV/binetflow corpora (not sampled); complete magic-byte
census of all 4,457 PCAP files; record-chain walks of 61 stratified-sample PCAPs plus all 338
previously-flagged PCAPs.
**Raw evidence:** regenerable with the commands in [§8](#8-acceptance-checklist-for-newly-provided-data).

---

## 0. The headline, before anything else

> **Every defect in every CSV corpus is upstream — baked into the official published
> distributions. Re-downloading the CSVs will reproduce all of them byte for byte.**

This reverses the previous working assumption that a local LibreOffice session truncated the
CIC-2018 files. It did not. Three independent lines of evidence:

1. **Row totals match the published dataset exactly.** The ten CIC-2018 CSVs hold
   **16,233,002** data rows. The canonical published figure for CSE-CIC-IDS2018 is
   16,233,002 instances, ~17% attack. Measured attack share here: **16.93%**
   (2,748,294 / 16,233,002). An independently truncated copy could not land on the
   published total.
2. **No local write ever happened.** Every CIC-2018 CSV has `mtime` in a contiguous
   8-minute window on **2018-10-11 21:32–21:40** while `birth` is 2026-08-28 15:06 — i.e. the
   files were *copied* onto this machine on 2026-08-28 with timestamps preserved, and never
   rewritten since. A LibreOffice save would have set `mtime` to 2026. The two `.~lock.*` files
   are stale locks from an open that was never saved.
3. **Same story for CIC-2017.** The damaged `Thursday-...-WebAttacks` file has
   `mtime = 2017-08-17 19:46:02`, identical to five sibling files — the original release
   timestamp, untouched.

The practical consequence runs through the whole report: **the fresh datasets will arrive with
the same CSV damage.** Plan around it rather than expecting it to disappear. The PCAPs are the
one corpus where re-downloading genuinely helps.

---

## 1. Executive summary

| Dataset | Files | Usable | Damaged | Action |
|---|---:|---|---|---|
| **CIC-2018 CSV** `~/Documents/SIH/DATA/CSV/` | 10 | 16,233,002 rows — complete vs. upstream | 7 files capped at 2^20−1 **upstream**; 9 of 10 have no IP columns; 12-hour clock; 14 epoch-1970 rows | **Do not re-download.** Regenerate flows from PCAP, or accept as-is with guards |
| **CIC-2018 PCAP** `~/Documents/SIH/DATA/pcap/` | 4,457 (600.7 GB) | 4,191 files clean | **266 files, 28.8 GB unrecoverable** — mid-file chain corruption | **Re-download these 266.** Highest-value action in this report |
| **CTU-13** `~/Documents/SIH/CTU-13-Dataset/` | 13 | 19,976,700 rows — **clean** | none found | Ship as-is. Fix the *label rule*, not the data |
| **CIC-2017 5-tuple** `~/Documents/SIH/test/extracted_flows/TrafficLabelling /` | 8 | 2,830,743 real rows | 1 file carries 288,602 blank padding rows (**upstream**); latin-1; mixed line endings | Keep. Row-guard already in place; harden the label match |
| **CIC-2017 anonymized** `~/Downloads/SIH/archive/` | 8 | row counts are a perfect reference | **79 cols — no IP, no port, no protocol, no timestamp** | **Unusable for graph/temporal work.** Keep only as a row-count oracle |

**Net:** three of five corpora are usable today. The single genuine, fixable loss is 28.8 GB of
PCAP payload. Everything else is an upstream property that must be *handled in code*.

---

## 2. CIC-2018 CSV — complete, but structurally crippled upstream

### 2.1 Per-file census

All ten files: UTF-8 clean, consistent column width within each file, final line complete
(every file ends `\n`).

| File | Data rows | Cols | 2^20−1? | Exact dup rows | Stray header rows | Timestamp span |
|---|---:|---:|:--:|---:|---:|---|
| `fri_16_csv.csv` | 1,048,575 | 80 | **YES** | **147,586** (14.1%) | 1 | 2018-02-16 |
| `fri_23_csv.csv` | 1,048,575 | 80 | **YES** | 2,614 | 0 | 2018-02-23 |
| `fri_2_csv.csv` | 1,048,575 | 80 | **YES** | 5,459 | 0 | 2018-03-02 |
| `thu_15_csv.csv` | 1,048,575 | 80 | **YES** | 2,421 | 0 | 2018-02-15 |
| `thu_1_csv.csv` | 331,125 | 80 | no | 73 | **25** | 2018-03-01 |
| `thu_22_csv.csv` | 1,048,575 | 80 | **YES** | 3,278 | 0 | 2018-02-22 **+ 9 rows in 1970** |
| `tue_20_csv.csv` | 7,948,748 | **84** | no | 2 | 0 | 2018-02-20 |
| `wed_14_csv.csv` | 1,048,575 | 80 | **YES** | **225,628** (21.5%) | 0 | 2018-02-14 **+ 5 rows in 1970** |
| `wed_21_csv.csv` | 1,048,575 | 80 | **YES** | 17,557 | 0 | 2018-02-21 |
| `wed_29_csv.csv` | 613,104 | 80 | no | 6,089 | **33** | **2018-02-28** |
| **Total** | **16,233,002** | | 7 files | 410,707 | 59 | |

### 2.2 The 2^20 cap is real, and it is upstream

Seven files hold **exactly 1,048,575** data rows. With the header that is 1,048,576 = 2^20, the
Excel/LibreOffice worksheet limit. Seven independent capture days landing on the same number by
chance is not credible — these were exported through a spreadsheet. But per §0 it happened in
2018, at dataset creation, by the dataset authors. **Correction to the earlier finding: it is 7
files, not 8.** `thu_1`, `wed_29` and `tue_20` are uncapped.

Note `tue_20` escaped both this cap *and* the column stripping — it is 4.05 GB and 84 columns.
That is consistent with the nine smaller files having gone through a processing step that
`tue_20` was too large to survive.

### 2.3 Nine of ten files have no host identity — and the adapter invents it

Only `tue_20_csv.csv` carries `Flow ID, Src IP, Src Port, Dst IP`. The other nine start at
`Dst Port` (80 columns). `data_unification/cic2018_adapter.py:65,70` fabricates addresses from
the row index:

```python
src_ips = np.array([f"192.168.10.{i % 250 + 1}" for i in range(len(chunk))])
dst_ips = np.array([f"172.16.0.{i % 100 + 1}"  for i in range(len(chunk))])
```

This manufactures a deterministic 250×100 bipartite graph with no relationship to real traffic.
Any host-graph, trajectory or GNN result computed over these nine days is an artifact of the
modulo, not of the network. **This is the most damaging single issue in the CSV corpus**, and
re-downloading cannot fix it because the columns were never published.

**The fix is to regenerate flows from the PCAPs** (the corpus is complete and holds real
5-tuples) using `data_unification/pcap_bridge.py`. Failing that, restrict all graph work to
`tue_20` and CIC-2017 and CTU-13.

### 2.4 Every timestamp is a 12-hour clock with no AM/PM — silent and corpus-wide

Hour histograms over the `Timestamp` field, all files:

```
fri_16   01:908339  08:50     09:111    10:139984 11:37     12:53
thu_22   01:110633  02:91665  03:81726  04:97839  05:49348  06:1  08:74421 09:124324 10:133254 11:161548 12:123816
fri_2    01:112091  02:117472 03:151373 04:75243  05:45799        08:36940 09:97953  10:118426 11:183414 12:109864
thu_1    01:32526   02:47897  03:41874  04:31256  05:11805        08:27100 09:36852  10:54032  11:29840  12:17918
```

**Hours 13–23 and 00 never appear.** Present hours are 01–05 and 08–12 — a working day of
08:00–17:00 recorded on a 12-hour dial. `01:00:00` means **13:00:00**.

Parsed naively, every afternoon flow is placed twelve hours *before* the morning, which is why
`ts_backward_steps` is 268k–383k per file (2,977,911 in `tue_20`). Any window grid, sequence
model or forecasting horizon built on these timestamps is ordering afternoon before morning.

**CIC-2017 has the identical defect** (§5.3). This is arguably as damaging as §2.3 and is not
mentioned in any prior report.

Deterministic repair: `if hour <= 7: hour += 12` (no capture-day hour legitimately falls in
06:00–07:59, so the rule is unambiguous on this corpus — verify per file before applying).

### 2.5 Fourteen rows dated 1970, and 94,181 protocol-0 rows

`thu_22` (9 rows, lines 246435–246441, 246717, 248316) and `wed_14` (5 rows, lines
410958–410961, 412186) contain timestamps like `10/01/1970 03:04:26`. All fourteen share a
signature:

```
DstPort=0  Protocol=0  FlowDuration=-828220000000 (negative)  Label=Benign
```

These are CICFlowMeter TCP-Segmentation-Offload failures — the same defect the DistriNet errata
documents as "causes a lot of flows to have protocol/src and dst port = 0". Unfiltered, one row
stretches a 2-second window grid across 48 years.

The population is far larger than fourteen. `Protocol == 0 AND Dst Port == 0`:

| File | rows | proto0 & port0 | % |
|---|---:|---:|---:|
| fri_16 | 1,048,574 | 162 | 0.02% |
| fri_23 | 1,048,575 | 16,067 | 1.53% |
| fri_2 | 1,048,575 | 13,382 | 1.28% |
| thu_15 | 1,048,575 | 18,565 | 1.77% |
| thu_1 | 331,100 | 7,149 | 2.16% |
| thu_22 | 1,048,575 | 16,173 | 1.54% |
| wed_14 | 1,048,575 | 11,882 | 1.13% |
| wed_21 | 1,048,575 | 61 | 0.01% |
| wed_29 | 613,071 | 10,740 | 1.75% |
| **Total (excl. tue_20)** | | **94,181** | **1.14%** |

**Recommended guard** — stronger than a year check, and catches all fourteen 1970 rows as a
side effect:

```python
drop = (protocol == 0) & (dst_port == 0) | (flow_duration < 0)
```

> **Not verified:** `tue_20_csv.csv` was excluded from this particular pass for runtime. Its
> proto-0 rate is unmeasured.

### 2.6 Smaller findings

- **`wed_29_csv.csv` is misnamed.** Its timestamps are all `28/02/2018`, and February 2018 had
  no 29th. It is Wednesday-28-02-2018, and pairs with `wed_28_pcap/`. Every other CSV/PCAP day
  name agrees. Rename on arrival or the PCAP↔CSV join silently drops a day.
- **Stray header rows mid-file:** `thu_1` has 25 extra `Dst Port,...` lines, `wed_29` has 33,
  `fri_16` has 1. `thu_1`'s *final line is a header row*. 59 rows total; `csv.reader` yields
  them as data. Skip rows whose first field equals the header's.
- **Duplicate rows:** `wed_14` 225,628 (21.5%) and `fri_16` 147,586 (14.1%) are extreme outliers
  — both are flood-attack days, so near-identical flows are expected, but these should not be
  counted as independent samples when computing base rates or splitting.
- **`wed_21` is ordered differently** from its siblings: span 8.79 h (01:55–10:43) and only
  4,056 backward steps vs. ~300k elsewhere. It is near time-sorted; the others are not. Do not
  assume a consistent row order across days.

### 2.7 Label vocabulary (exact strings, union of all ten files)

15 distinct values. `Benign` is **title-case**, not `BENIGN` as in CIC-2017.

| Label | Rows |
|---|---:|
| `Benign` | 13,484,708 |
| `DDOS attack-HOIC` | 686,012 |
| `DDoS attacks-LOIC-HTTP` | 576,191 |
| `DoS attacks-Hulk` | 461,912 |
| `Bot` | 286,191 |
| `FTP-BruteForce` | 193,360 |
| `SSH-Bruteforce` | 187,589 |
| `Infilteration` *(sic)* | 161,934 |
| `DoS attacks-SlowHTTPTest` | 139,890 |
| `DoS attacks-GoldenEye` | 41,508 |
| `DoS attacks-Slowloris` | 10,990 |
| `DDOS attack-LOIC-UDP` | 1,730 |
| `Brute Force -Web` | 611 |
| `Brute Force -XSS` | 230 |
| `SQL Injection` | 87 |

Note the inconsistent casing (`DDOS` vs `DDoS`, `SSH-Bruteforce` vs `FTP-BruteForce`), the
misspelling `Infilteration`, and the stray space in `Brute Force -Web`. Match case-insensitively
on normalized whitespace.

---

## 3. CIC-2018 PCAP — the one corpus worth re-downloading

### 3.1 Complete census (all 4,457 files, magic bytes read on every one)

**600.7 GB** total (= 559.5 GiB, consistent with the "560 GB" figure).

| Magic | Files | Size |
|---|---:|---:|
| `d4c3b2a1` — pcap, little-endian, µs | 4,454 | 596.3 GB |
| `0a0d0d0a` — **PCAPNG** | 2 | 4.3 GB |
| `4c000000` — Windows `.lnk` | 1 | 697 B |

All 4,454 true pcaps are uniform: version 2.4, snaplen 65535, linktype 1 (Ethernet),
little-endian. Packet timestamps across the sample span **2018-02-14 12:28:59 UTC** to
**2018-03-02 21:30:15 UTC** — correct for the capture campaign.

**The pcapng files are exactly two**, both in `fri_16_pcap`, and *both* carry a `.pcap`
extension — narrower than the earlier finding suggested, but note part2 as well as part1:

```
fri_16_pcap/pcap/UCAP172.31.69.25-part1.pcap   4,312,494,308 B   PCAPNG
fri_16_pcap/pcap/UCAP172.31.69.25-part2.pcap         815,212 B   PCAPNG
```

These are not damaged. Sniff the magic and dispatch to a pcapng reader; do not re-download.

### 3.2 Non-dataset files to delete

```
fri_2_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.24 - Shortcut.lnk     697 B   Windows shortcut
fri_2_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69 - Copy.24          61,469 B   mangled "- Copy" duplicate
```

Both are Windows Explorer accidents, not captures. The second is especially dangerous: its name
looks like a capture, so a glob will feed 61 KB of truncated data into the pipeline as if it
were a host's full day.

### 3.3 Eight filenames contain spaces — including an 18 GB file

```
wed_21_pcap/pcap/UCAP172.31.69.28 part 1              18,275,068,082 B   ← largest file in corpus
wed_21_pcap/pcap/UCAP172.31.69.28 part 2               2,123,980,800 B
thu_1_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.13 part1    150,323,355 B
thu_1_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.13 part2     36,372,624 B
thu_1_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.13 part3    129,061,367 B
wed_28_pcap/pcap/capEC2AMAZ-O4EL3NG-172.31.69.24- part1   98,445,934 B
```

Any unquoted shell glob or `xargs` without `-0` splits these into non-existent paths — which is
precisely what happened during this audit before the loop was quoted. Quote everything, or
rename.

### 3.4 Measured damage: 263 severe files lose 28.8 GB

I walked the record chain of all 338 previously-flagged files independently, and 61 stratified
samples (4 unflagged + 2 flagged per day across all ten days). **Agreement with the existing
`corrupt_files_redownload.txt` on the sample: 60/61.**

| Bucket | Files | On disk | Recoverable prefix | **Unrecoverable** |
|---|---:|---:|---:|---:|
| SEVERE | 263 | 60.4 GB | 31.5 GB (52.2%), 39.6 M packets | **28.8 GB (47.8%)** |
| MINOR | 74 | 10.5 GB | 10.2 GB (97.3%), 98.2 M packets | 0.3 GB (2.7%) |

Recoverable-prefix distribution within SEVERE: min 0.5%, p25 22.2%, **median 49.7%**, p75 73.6%,
max 100.0%. **133 of 263 files lose more than half their payload.** The adapter's
keep-the-valid-prefix strategy is correct but is silently discarding ~48% of the bytes in these
files.

Failure modes: `IMPLAUSIBLE_CAPLEN` 321, `SHORT_RECORD_HEADER` 11, `IMPLAUSIBLE_TIMESTAMP` 4,
`ORIGLEN_LT_CAPLEN` 1.

Per-day severe damage:

| Day | Files | Size | Lost |
|---|---:|---:|---:|
| wed_14_pcap | 55 | 10.7 GB | **6.8 GB** |
| fri_16_pcap | 45 | 7.9 GB | **5.1 GB** |
| thu_15_pacap | 43 | 5.6 GB | 3.5 GB |
| fri_23_pacap | 28 | 11.5 GB | 2.6 GB |
| wed_21_pcap | 18 | 5.9 GB | 2.5 GB |
| thu_1_pcap | 30 | 7.5 GB | 2.4 GB |
| fri_2_pcap | 17 | 3.7 GB | 2.4 GB |
| thu_22_pcap | 12 | 4.0 GB | 1.9 GB |
| tue_20_pcap | 5 | 1.4 GB | 0.9 GB |
| wed_28_pcap | 10 | 2.0 GB | 0.6 GB |
| **Total** | **263** | **60.4 GB** | **28.8 GB** |

### 3.5 Corruption is almost perfectly localised to one capture agent

This is new, and it changes the re-download strategy.

| Filename prefix | Files in corpus | In SEVERE list | Rate |
|---|---:|---:|---:|
| `capDESKTOP-AN3U28N` | 765 | **262** | **34.2%** |
| `capEC2AMAZ-O4EL3NG` | 2,107 | 0 | 0.0% |
| `capPC1` | 757 | 0 | 0.0% |
| `capWIN-J6GMIG1DQE5` | 753 | 1 | 0.1% |
| `UCAP*` | 75 | 0 | 0.0% |

**262 of 263 severe files come from a single capture host.** Random bit rot or generic transfer
failure would distribute across all four agents in proportion to file count. It does not. Either
that agent's captures were written badly at source, or the subset carrying its files took a
different transfer path.

**Implication:** if the fresh copy still shows `capDESKTOP-AN3U28N` corruption at ~34%, the
damage is upstream and further re-downloading is futile — switch to the prefix-keeping reader
permanently. Test this on **one** day before re-pulling 60 GB.

### 3.6 The MINOR bucket is wrong for 3 files

Report 06 triaged on the *capinfos error string*, so files reporting the benign-sounding
"checksums might be incorrect" went to `corrupt_files_minor_no_action.txt`. A record-chain walk
shows three of them are catastrophic:

| File | Walked | Lost |
|---|---:|---:|
| `wed_14_pcap/pcap/capDESKTOP-AN3U28N-172.31.64.122` | 24.13% | **152.1 MB** |
| `thu_15_pacap/pcap/capDESKTOP-AN3U28N-172.31.67.119` | 25.82% | **105.7 MB** |
| `fri_16_pcap/pcap/capDESKTOP-AN3U28N-172.31.65.86` | 49.19% | **27.6 MB** |

The other 71 MINOR files lose under 10 MB each and genuinely need nothing — including all 70
`UCAP*` entries. **Move these three to the re-download list** (they are all
`capDESKTOP-AN3U28N`, consistent with §3.5). Re-download total becomes **266 files**.

> **Not verified:** the 4,119 files never flagged by either scan were checked only for magic
> bytes here, plus 40 full record-chain walks in the stratified sample (all clean). A mid-file
> break in an unflagged file would have been caught by the original capinfos census, so the
> residual risk is low but non-zero.

---

## 4. CTU-13 — clean data, and the label rule that misreported it

### 4.1 Integrity: no defects found

All 13 scenarios scanned in full. **19,976,700 data rows**, 15 columns
(`StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`)
— one single column set across all 13 files.

- Exact duplicate rows: **0**
- Blank/padding rows: **0**
- Unparseable timestamps: **0** (format `%Y/%m/%d %H:%M:%S.%f` throughout)
- Every file ends with a newline; no truncated tails
- No suspicious row-count boundaries; no 2^20 or 2^16 anywhere

Timestamps span **2011-08-10 → 2011-08-19**, per-scenario 0.27 h – 66.82 h — matching the
published capture campaign. Backward steps are small (3 – 57,892 across millions of rows), i.e.
the files are near time-sorted; residual non-monotonicity is normal for NetFlow, which emits on
flow expiry.

| Sc | File | Rows | Span |
|---:|---|---:|---:|
| 1 | capture20110810.binetflow | 2,824,636 | 6.12 h |
| 2 | capture20110811.binetflow | 1,808,122 | 4.19 h |
| 3 | capture20110812.binetflow | 4,710,638 | 66.82 h |
| 4 | capture20110815.binetflow | 1,121,076 | 4.47 h |
| 5 | capture20110815-2.binetflow | 129,832 | 0.50 h |
| 6 | capture20110816.binetflow | 558,919 | 2.15 h |
| 7 | capture20110816-2.binetflow | 114,077 | 0.35 h |
| 8 | capture20110816-3.binetflow | 2,954,230 | 19.47 h |
| 9 | capture20110817.binetflow | 2,087,508 | 5.62 h |
| 10 | capture20110818.binetflow | 1,309,791 | 5.14 h |
| 11 | capture20110818-2.binetflow | 107,251 | 0.27 h |
| 12 | capture20110819.binetflow | 325,471 | 1.72 h |
| 13 | capture20110815-3.binetflow | 1,925,149 | 16.37 h |

### 4.2 The label vocabulary is 1,400 strings, and "benign" is not one of them

Exact census over all 19,976,700 rows: **1,400 distinct label strings**. The count of rows whose
label equals `benign` / `Benign` / `BENIGN`, in any case: **zero**.

That is the whole bug. An exact-match-on-`benign` rule classifies **100.000%** of CTU-13 as
attack. Confirmed by direct measurement, not inference.

The correct rule is a case-insensitive **substring** test on three families:

```python
l = label.lower()
if   "botnet"     in l: cls = "attack"            # 558 distinct strings
elif "normal"     in l: cls = "benign (verified)" #  98 distinct strings
elif "background" in l: cls = "unknown ground truth"  # 37 distinct strings
```

This partitions the corpus exhaustively — **0 rows fall through**.

| Class | Rows | Share |
|---|---:|---:|
| `flow=Background-*` / `flow=To-Background-*` — **unknown ground truth** | 19,175,568 | **95.99%** |
| `flow=From-Botnet-*` — attack | **444,699** | **2.2261%** |
| `flow=From-Normal-*` — verified benign | 356,433 | 1.7842% |

### 4.3 Three defensible attack rates — pick one and state it

This corpus admits three very different numbers, and conflating them has already caused one
misreport:

| Framing | Attack rate | When it is right |
|---|---:|---|
| Exact-match `"benign"` | **100%** | **never** — this is the bug |
| Botnet / all rows | **2.23%** | if Background is treated as benign |
| Botnet / (Botnet + Normal) | **55.51%** | **correct for supervised training** |

The third is the one to use. `Background` means *the labellers did not establish ground truth* —
it is not a benign label. Training on it as benign injects an unknown, large volume of
mislabelled attack traffic. Standard practice is to **discard the 19.18 M Background rows** and
train on the 801,132 labelled rows, which are **55.51% attack**.

The earlier "~0.7%" figure matches none of these; the measured botnet share is **2.2261%** of
all rows.

`data_unification/label_resolver.py:94–97` already implements the correct substring fallback for
CTU-13 — the adapter is fine. The defect was in ad-hoc audit scripts, which should be updated to
the rule above.

---

## 5. CIC-2017 full 5-tuple copy — usable, with one badly damaged file

`~/Documents/SIH/test/extracted_flows/TrafficLabelling /` — **the directory name ends in a
space.** Confirmed at byte level (`TrafficLabelling\ `). Quote it everywhere.

All 8 files: 85 columns, uniform width, no stray header rows, complete final line.

| File | Real rows | Blank rows | Dups | Labels |
|---|---:|---:|---:|---|
| Monday-WorkingHours | 529,918 | 0 | 34 | BENIGN |
| Tuesday-WorkingHours | 445,909 | 0 | 4 | BENIGN, FTP-Patator, SSH-Patator |
| Wednesday-workingHours | 692,703 | 0 | 17 | BENIGN, DoS Hulk/GoldenEye/slowloris/Slowhttptest, Heartbleed (11) |
| Thursday-Morning-WebAttacks | **170,366** | **288,602** | 1 | BENIGN, Web Attack × 3 |
| Thursday-Afternoon-Infilteration | 288,602 | 0 | 142 | BENIGN, Infiltration (36) |
| Friday-Morning | 191,033 | 0 | 2 | BENIGN, Bot |
| Friday-Afternoon-PortScan | 286,467 | 0 | 1 | BENIGN, PortScan |
| Friday-Afternoon-DDos | 225,745 | 0 | 2 | BENIGN, DDoS |
| **Total** | **2,830,743** | **288,602** | 203 | |

### 5.1 The blank padding, and the coincidence that explains it

`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` holds 458,968 rows. From data row
**170,367** onward, **288,602 rows are entirely blank** — literally 84 commas and `\r\n`:

```
,,,,,,,,,,,,,,,,,,,,, ... ,,,,\r\n
```

62.9% of the file is padding. Read with pandas these become `src_ip='nan'` and a NaT timestamp
that casts to `-9223372036`, producing phantom hosts on a window grid reaching back to 1677.

**The number 288,602 is not arbitrary — it is exactly the row count of
`Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`.** Someone had both files open in
a spreadsheet, pasted the Infiltration range into the WebAttacks sheet, then cleared the cell
*contents* without deleting the *rows*. In 2017, upstream — `mtime` is the untouched original.

Two independent confirmations that 170,366 is the true row count:

- the anonymized copy (§6) has exactly **170,366** rows for this file, and zero blank rows;
- every other file's row count matches the anonymized copy exactly.

**The guard is already in place** at `data_unification/cic2017_adapter.py:104`:

```python
if start_timestamps[i] <= 0 or str(src_ips[i]).lower() in ("nan", ""):
    continue
```

Correct and sufficient. Keep it — a fresh download will bring the padding back.

### 5.2 Encoding and line endings are inconsistent

- `Thursday-...-WebAttacks` is **not valid UTF-8**. The Web Attack labels contain byte `0x96`
  (cp1252 en-dash): the true strings are `Web Attack \x96 Brute Force` (1,507),
  `Web Attack \x96 XSS` (652), `Web Attack \x96 Sql Injection` (21). Read as latin-1 or cp1252;
  UTF-8 with `errors="strict"` throws, and `errors="replace"` silently mangles the label so
  exact matching fails. `label_resolver.py:44` normalizes `\x96` — good.
- The other seven files are clean UTF-8.
- **Line endings differ:** 6 files are CRLF, but `Monday-WorkingHours` and
  `Friday-Afternoon-DDos` are LF-only. `Friday-Afternoon-DDos` also has a later `mtime`
  (2018-06-07 vs 2017-08-17) — it was re-released separately.

### 5.3 Same 12-hour clock defect as CIC-2018

```
Monday      01:57600 02:48567 03:48916 04:65875 05:1841 | 08:809  09:74418 10:85079 11:88872 12:57941
Wednesday   01:48902 02:63868 03:66142 04:44878 05:3547 | 08:14107 09:54304 10:301876 11:71086 12:23993
Fri-PM-DDos 03:52500 04:171693 05:1552
```

No hours 13–23. `Friday-Afternoon-DDos` reports 03:30–05:02, which is 15:30–17:02 — it lands
*before* `Friday-Morning` (08:59–12:59) under naive parsing. Backward steps: Monday 208,847,
Tuesday 159,424, Wednesday 163,908.

**Timestamp formats also differ between files**: `Monday` uses `%d/%m/%Y %H:%M:%S` with
zero-padded hours; `Tuesday`/`Wednesday` use `%d/%m/%Y %H:%M` with *unpadded* hours (`1` not
`01`). Do not hard-code one format.

### 5.4 Data quality

Parse rate 100% on all 8 files (2,830,743 / 2,830,743). Source IPs are well-formed dotted quads
in **every** real row — the `nan` IPs come only from the blank padding. `Protocol == 0`: 1,696
rows total. Negative `Flow Duration`: 115 rows total. Both an order of magnitude cleaner than
CIC-2018.

Label strings use **`BENIGN`** (upper case) — different from CIC-2018's `Benign`. Distinct
values across the corpus: `BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS
slowloris, FTP-Patator, Heartbleed, Infiltration, PortScan, SSH-Patator, Web Attack \x96 Brute
Force, Web Attack \x96 Sql Injection, Web Attack \x96 XSS`. Note `DoS slowloris` is lower-case
`s` while `DoS GoldenEye` is not.

---

## 6. CIC-2017 anonymized copy — confirmed unusable, but keep it

`~/Downloads/SIH/archive/`, 8 files, `mtime` 2023-08-28 (a Kaggle repackage).

**Confirmed: 79 columns.** The header begins ` Destination Port, Flow Duration, ...`. Relative
to the 85-column original it is missing exactly six columns:

| Missing | Consequence |
|---|---|
| `Flow ID` | no flow identity |
| `Source IP` | **no graph edges** |
| `Source Port` | no service inference on the client side |
| `Destination IP` | **no graph edges** |
| `Protocol` | no L4 discrimination |
| `Timestamp` | **no temporal ordering — no windows, no sequences, no forecasting** |

This corpus cannot support host-graph construction, trajectory building, windowing, or any
forecasting objective. It is a flat tabular classification dataset only. **Unusable for this
project.**

**But do not delete it.** Its row counts are a clean, blank-free reference:

| File | Anonymized rows | Full-copy real rows | Match |
|---|---:|---:|:--:|
| Friday-Afternoon-DDos | 225,745 | 225,745 | ✅ |
| Friday-Afternoon-PortScan | 286,467 | 286,467 | ✅ |
| Friday-Morning | 191,033 | 191,033 | ✅ |
| Monday-WorkingHours | 529,918 | 529,918 | ✅ |
| Thursday-Afternoon-Infilteration | 288,602 | 288,602 | ✅ |
| **Thursday-Morning-WebAttacks** | **170,366** | **170,366** | ✅ |
| Tuesday-WorkingHours | 445,909 | 445,909 | ✅ |
| Wednesday-workingHours | 692,703 | 692,703 | ✅ |

Eight for eight. This is the oracle that proves the padding is padding and that no *rows* were
lost from the full copy.

It also preserves the `0x96` label bytes (`Web Attack ? Brute Force` on a UTF-8 terminal), so it
carries the same encoding caveat.

---

## 7. Prioritized re-download / repair list

### P0 — Re-download: 266 PCAP files, ~60 GB

1. The 263 in [`corrupt_files_redownload.txt`](corrupt_files_redownload.txt) — 28.8 GB
   unrecoverable, median 49.7% of each file lost.
2. **Plus these 3**, currently misfiled under "minor, no action" (§3.6):
   ```
   wed_14_pcap/pcap/capDESKTOP-AN3U28N-172.31.64.122
   thu_15_pacap/pcap/capDESKTOP-AN3U28N-172.31.67.119
   fri_16_pcap/pcap/capDESKTOP-AN3U28N-172.31.65.86
   ```

**Do `wed_14_pcap` first** (55 files, 6.8 GB lost — the worst day) and verify with
[`verify_pcap_day.sh`](verify_pcap_day.sh) **before** pulling the rest. If nothing is fixed,
the damage is upstream (§3.5) and the remaining ~54 GB of transfer is wasted.

### P1 — Verify, do not blindly re-pull: all 765 `capDESKTOP-AN3U28N` files

34.2% of that one host's captures are corrupt; every other host is at 0–0.1%. Walk all 765 on
arrival regardless of what any list says.

### P2 — Do NOT re-download any CSV

Every CSV defect is upstream (§0). A fresh pull reproduces: the 2^20 caps, the 9 files without
IP columns, the 12-hour clocks, the 288,602 blank rows, the latin-1 bytes. Instead:

1. **Regenerate CIC-2018 flows from the PCAPs** via `data_unification/pcap_bridge.py`. This is
   the only path that recovers real source/destination IPs for the nine 80-column days and lifts
   the 2^20 cap. It is the highest-value engineering task in this report.
2. Consider the **DistriNet "Improved CSE-CIC-IDS2018"** re-labelled release as a cross-check on
   labels (it documents extensive upstream mislabelling, e.g. 41.7% of Web-Brute-Force flows on
   23-02-2018). It does not fix truncation or restore IP columns.

### P3 — Housekeeping on arrival

- Delete `capEC2AMAZ-O4EL3NG-172.31.69.24 - Shortcut.lnk` and
  `capEC2AMAZ-O4EL3NG-172.31.69 - Copy.24`.
- Rename `wed_29_csv.csv` → `wed_28_csv.csv` (its data is 28/02/2018).
- Optionally rename `fri_23_pacap` / `thu_15_pacap` → `_pcap`. Already handled at
  `scripts/freeze_splits.py:86`, but only there.
- Decide on the 6 space-containing filenames: rename, or audit every glob for quoting.

---

## 8. Acceptance checklist for newly provided data

Run in order. Each step is standalone and read-only. Stop and investigate on any ❌.

### 8.1 Inventory and gross shape

```bash
# CIC-2018 CSV: expect 10 files, and the header row count 16,233,012 (16,233,002 data + 10 headers)
cd ~/Documents/SIH/DATA/CSV
ls -1 *.csv | wc -l                        # expect 10
for f in *.csv; do printf "%s\t%s\n" "$f" "$(wc -l < "$f")"; done
awk 'END{}' ; cat *.csv | wc -l            # expect 16,233,012

# ❌ FAIL if any file has exactly 1048576 lines AND you were told it is a fresh full export
#    — that is 2^20. Expect 7 of them; more than 7 means new truncation.

# CTU-13: expect 13 files, 19,976,713 lines total
find ~/Documents/SIH/CTU-13-Dataset -name '*.binetflow' | wc -l    # expect 13
cat ~/Documents/SIH/CTU-13-Dataset/*/*.binetflow | wc -l           # expect 19,976,713

# CIC-2017: expect 8 files
ls -1 "$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /"*.csv | wc -l   # expect 8
```

### 8.2 Suspicious boundaries

```bash
# flag any file landing on a power of two or a suspiciously round number
for f in ~/Documents/SIH/DATA/CSV/*.csv; do
  n=$(wc -l < "$f")
  case $n in
    1048576|1048575|65536|65535|1000000|500000)
      echo "SUSPICIOUS BOUNDARY: $f = $n" ;;
  esac
done
```

### 8.3 Column-count consistency and stray headers

```bash
cd ~/Documents/SIH/DATA/CSV
for f in *.csv; do
  echo "$f cols=$(head -1 "$f" | awk -F, '{print NF}') \
strayheaders=$(( $(grep -c '^Dst Port,\|^Flow ID,' "$f") - 1 ))"
done
# expect: tue_20 = 84 cols, all others 80
# ❌ FAIL if strayheaders > 0 on any file you expect clean (baseline: thu_1=25, wed_29=33, fri_16=1)

# per-row width must be constant within a file
for f in *.csv; do
  echo -n "$f: "; awk -F, '{print NF}' "$f" | sort -u | tr '\n' ' '; echo
done
```

### 8.4 Blank / padding rows and complete final line

```bash
# blank rows anywhere
for f in ~/Documents/SIH/DATA/CSV/*.csv \
         "$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /"*.csv; do
  n=$(grep -c '^[[:space:],]*$' "$f")
  [ "$n" -gt 0 ] && echo "BLANK ROWS: $f = $n"
done
# baseline: only Thursday-...-WebAttacks, = 288,602. Anything else is new damage.

# final line must be complete: last byte is a newline, last line is not a header
for f in ~/Documents/SIH/DATA/CSV/*.csv; do
  lb=$(tail -c1 "$f" | od -An -tx1 | tr -d ' ')
  [ "$lb" != "0a" ] && echo "TRUNCATED FINAL LINE: $f"
  tail -1 "$f" | grep -q '^Dst Port,\|^Flow ID,' && echo "FINAL LINE IS A HEADER: $f"
done
```

### 8.5 Encoding

```bash
for f in ~/Documents/SIH/DATA/CSV/*.csv \
         "$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /"*.csv; do
  iconv -f UTF-8 -t UTF-8 "$f" >/dev/null 2>&1 || echo "NOT UTF-8: $f"
done
# baseline: only Thursday-...-WebAttacks (latin-1, 0x96 en-dash in labels)
```

### 8.6 Timestamp sanity — **run this one, it catches the worst defect**

```bash
# CIC-2018: hour histogram. Field 3 for 80-col files, field 7 for tue_20.
for f in ~/Documents/SIH/DATA/CSV/*.csv; do
  echo -n "$(basename $f): "
  awk -F, 'NR>1 && $3 ~ /\// {split($3,a," "); split(a[2],t,":"); print t[1]}' "$f" \
    | sort -n | uniq -c | tr '\n' ' '; echo
done
# ❌ If hours 13-23 NEVER appear -> 12-hour clock, no AM/PM. Apply `if h<=7: h+=12`.
#    If hours 13-23 DO appear -> the new export is 24-hour. Do NOT apply the fix.
#    This single check decides whether every downstream timestamp is right or 12h wrong.

# non-2018 timestamps (the epoch-1970 garbage rows)
for f in ~/Documents/SIH/DATA/CSV/*.csv; do
  n=$(awk -F, 'NR>1 && $3 ~ /\// && $3 !~ /2018/ {c++} END{print c+0}' "$f")
  [ "$n" -gt 0 ] && echo "NON-2018 TIMESTAMPS: $(basename $f) = $n"
done
# baseline: thu_22=9, wed_14=5. All have Protocol=0, DstPort=0, negative duration.

# CTU-13 parse rate must be 100%
python3 - <<'EOF'
import glob, datetime
for p in sorted(glob.glob('/var/home/samito/Documents/SIH/CTU-13-Dataset/*/*.binetflow')):
    ok=bad=0
    with open(p) as fh:
        next(fh)
        for line in fh:
            s=line.split(',',1)[0]
            try: datetime.datetime.strptime(s,"%Y/%m/%d %H:%M:%S.%f"); ok+=1
            except ValueError: bad+=1
    print(f"{p.split('/')[-2]:>3} ok={ok:>9} bad={bad}")   # every bad must be 0
EOF
```

### 8.7 Label vocabulary — never exact-match on "benign"

```bash
# CIC-2018 (field 80 / 84 = last)
awk -F, 'NR>1 && $NF!="Label"{print $NF}' ~/Documents/SIH/DATA/CSV/*.csv \
  | sort | uniq -c | sort -rn
# expect 15 distinct, 'Benign' title-case

# CIC-2017
awk -F, 'NR>1 && NF>50 {print $NF}' \
  "$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /"*.csv \
  | sort | uniq -c | sort -rn
# expect 15 distinct, 'BENIGN' UPPER-case, three 'Web Attack \x96 ...' with a 0x96 byte

# CTU-13 — the three-way split, NOT an exact match
cat ~/Documents/SIH/CTU-13-Dataset/*/*.binetflow \
  | awk -F, 'NR>1 && $15!="Label" && NF>=15 {print $15}' | sort | uniq -c | sort -rn \
  | awk '{l=tolower($0); s=$1;
          if(l~/botnet/) b+=s; else if(l~/normal/) n+=s;
          else if(l~/background/) g+=s; else o+=s}
     END{printf "BOTNET %d  NORMAL %d  BACKGROUND %d  UNCLASSIFIED %d\nattack rate (excl. background) = %.2f%%\n",
                b,n,g,o,100*b/(b+n)}'
# expect: BOTNET 444699  NORMAL 356433  BACKGROUND 19175568  UNCLASSIFIED 0
#         attack rate (excl. background) = 55.51%
# ❌ FAIL if UNCLASSIFIED > 0 — a new label family appeared and needs mapping.
```

### 8.8 Duplicate rows

```bash
for f in ~/Documents/SIH/DATA/CSV/*.csv; do
  t=$(tail -n +2 "$f" | wc -l); u=$(tail -n +2 "$f" | sort -u | wc -l)
  echo "$(basename $f) rows=$t unique=$u dup=$((t-u)) ($(( (t-u)*100/t ))%)"
done
# baseline: wed_14 21%, fri_16 14%, others <2%. CTU-13 must be 0.
```

### 8.9 PCAP — magic bytes on every file, record-chain walk on a sample

```bash
# (a) full magic-byte census — cheap, run on all files
find ~/Documents/SIH/DATA/pcap -type f -print0 | while IFS= read -r -d '' f; do
  m=$(head -c4 "$f" | od -An -tx1 | tr -d ' \n')
  case "$m" in
    d4c3b2a1) ;;                                   # normal pcap LE usec
    0a0d0d0a) echo "PCAPNG (needs pcapng reader): $f" ;;
    *)        echo "NOT A CAPTURE FILE ($m): $f" ;;
  esac
done
# expect: 4,454 silent, 2 PCAPNG (fri_16 UCAP172.31.69.25-part{1,2}.pcap), 0 others.
# ❌ any '.lnk', '- Copy' or unknown magic = delete before ingesting.

# (b) file count and total size
find ~/Documents/SIH/DATA/pcap -type f | wc -l      # expect 4,457 (4,456 captures + 1 .lnk)
du -sb ~/Documents/SIH/DATA/pcap                    # expect ~600.7e9 bytes

# (c) record-chain walk. If Wireshark is available:
toolbox run -c prism-dev capinfos -c -a -e "<file>"
# Otherwise use the pure-Python walker (no Wireshark needed) — walks every record,
# reports the exact byte offset and bytes lost at the first bad length/timestamp field:
#   claude_latest_analysis/verify_pcap_day.sh <day>
toolbox run -c prism-dev ./claude_latest_analysis/verify_pcap_day.sh wed_14_pcap

# (d) THE decisive check — is the one bad capture agent still bad?
#     Walk all capDESKTOP-AN3U28N files and compute the corruption rate.
#     baseline 34.2% (262/765). Other hosts must stay at ~0%.
#     If it is still ~34%, the damage is UPSTREAM: stop re-downloading, keep the
#     valid-prefix reader, and document the loss.
```

### 8.10 Cross-corpus invariants

```bash
# CIC-2017: the anonymized copy is the row-count oracle. These must match exactly.
for f in ~/Downloads/SIH/archive/*.csv; do
  b=$(basename "$f")
  a=$(( $(wc -l < "$f") - 1 ))
  c=$(awk -F, 'NF>50 && $2!="" && NR>1' "$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /$b" | wc -l)
  [ "$a" -eq "$c" ] && echo "OK   $b $a" || echo "❌  $b anon=$a full=$c"
done

# every CIC-2018 CSV day must have a matching PCAP day directory
ls ~/Documents/SIH/DATA/CSV/*.csv | sed 's/.*\///; s/_csv\.csv//' | sort > /tmp/csvdays
ls -d ~/Documents/SIH/DATA/pcap/*/  | sed 's|.*/\([^/]*\)/$|\1|; s/_pa*cap//' | sort > /tmp/pcapdays
diff /tmp/csvdays /tmp/pcapdays
# baseline diff: csv has 'wed_29', pcap has 'wed_28'. They are THE SAME DAY (28/02/2018).
# ❌ any other difference is a genuinely missing day.
```

---

## 9. What I could not verify

Flagged rather than guessed:

1. **`tue_20_csv.csv` protocol-0 / negative-duration rate** — excluded from that pass for
   runtime (4.05 GB). Its row count, column count, label distribution, duplicate count and
   timestamp range *were* fully measured.
2. **Whether the upstream CSE-CIC-IDS2018 S3 objects are byte-identical to these files.** The
   inference in §0 rests on the row total matching the published 16,233,002, the untouched 2018
   mtimes, and the 16.93% attack rate matching the published ~17%. I could not retrieve an
   authoritative per-file row manifest — the CIC page, the AWS registry entry and the DistriNet
   errata do not publish one, and Kaggle's listing is JS-rendered. A direct `aws s3 ls
   --no-sign-request s3://cse-cic-ids2018/` size comparison would settle it definitively.
3. **The 4,119 never-flagged PCAP files** were checked for magic bytes (all 4,457) but only 40
   received a full record-chain walk. The original capinfos census covered them, so residual
   risk is low.
4. **Whether the 263 severe PCAP corruptions are upstream or transfer-related.** §3.5 shows they
   are localised to one capture agent, which is suggestive but not conclusive. The P0 → verify →
   P1 sequence is designed to answer this after ~7 GB of transfer rather than 60 GB.
5. **`wed_21_csv.csv`'s different row ordering** (8.79 h span, near-sorted) — observed but not
   explained.

---

## 10. Corrections to earlier findings

| # | Earlier statement | Measured |
|---|---|---|
| 1 | 8 of 10 CIC-2018 CSVs truncated at 2^20 | **7** of 10 (`thu_1`, `wed_29`, `tue_20` uncapped) |
| 1 | LibreOffice on this machine truncated them | **No.** mtimes are untouched 2018-10-11; total matches published 16,233,002. Upstream |
| 3 | WebAttacks blank padding | Confirmed exactly: 288,602 blanks from row 170,367. **Also upstream**, and equal to the Infiltration file's row count |
| 4 | 263 corrupt PCAPs | Confirmed, +3 misfiled as "minor". **266.** Quantified: 28.8 GB unrecoverable, median 49.7% recoverable |
| 5 | "a few" `.pcap` files are really pcapng | Exactly **2**, both in `fri_16_pcap`, both `UCAP172.31.69.25-part{1,2}.pcap` |
| 6 | CTU-13 clean | Confirmed — 0 dups, 0 blanks, 0 bad timestamps, 1 column set |
| 6 | CTU-13 is ~0.7% attack | **2.2261%** of all rows; **55.51%** excluding Background. The 100% figure came from exact-matching `"benign"`, which matches 0 of 19,976,700 rows |
| — | *(new)* | **Every CIC-2018 and CIC-2017 timestamp is a 12-hour clock with no AM/PM.** Afternoon sorts before morning |
| — | *(new)* | **9 of 10 CIC-2018 days have no IP columns upstream** — re-downloading cannot fix the fabricated-IP problem; only PCAP regeneration can |
