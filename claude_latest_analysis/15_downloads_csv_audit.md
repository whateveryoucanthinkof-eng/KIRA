# Audit — is `~/Downloads/DATA/` a better CSV corpus than the one we train on?

**Date:** 2026-09-21
**Target:** `/var/home/samito/Downloads/DATA/` (`CSV/`, `logs/`, empty `pcap/`; 8.3 GB)
**Baseline:** `/var/home/samito/Documents/SIH/DATA/CSV/` (what `data_unification/dataset_paths.py`
and `scripts/freeze_splits.py` actually read)
**Mode:** read-only. Nothing under `Downloads/DATA` or `~/Documents/SIH` was modified. The only
bytes written are this file; a scratch directory used for `sort -T` was created outside both trees
and deleted at the end.

---

## 0. Verdict

**The Downloads copy is not better. It is not merely equivalent — it is the *same bytes*.** All ten
CIC-2018 CSVs are byte-identical to the ten we train on (MD5 match on all ten, including a full-file
MD5 of the 4.05 GB `tue_20`). `~/Documents/SIH/DATA/CSV/` was created *from* this Downloads
directory: every Downloads file has a `btime` spread across 2026-08-28 11:16–11:34 (downloaded one
at a time), while all ten SIH files share a `btime` inside a single 0.2-second window at
2026-08-28 15:06:57 — a `cp` of the Downloads set with `mtime` preserved, and the only change was
appending `.csv` to eight filenames. **It therefore fixes neither defect: the seven files still hold
exactly 1,048,575 data rows each, and nine of ten still have 80 columns with no IP columns at all.**
The single most important number: **summing `wc -l` minus one header across the ten Downloads files
gives exactly 16,233,002 — the published CSE-CIC-IDS2018 total, to the row.** There is no missing
data to recover; the 2^20 cap was applied by CIC before publication. This **confirms** the
conclusion of `09_data_integrity_audit.md` rather than overturning it, and adds a sharper proof of
it. The one genuinely new item is `logs/`: 10 ZIP archives of Windows `.evtx` victim-host event
logs, 451 distinct host IPs — which cannot restore per-flow IPs (see §4).

---

## 1. Per-file census of `~/Downloads/DATA/CSV/`

Every number below is from a `wc -l` and a single-pass `gawk` scan of the Downloads file itself, not
inherited from the earlier audit.

| File (Downloads) | Data rows | +stray hdr | Cols | IP cols | 2^20−1? | Hours seen | Dup rows | epoch-1970 | proto=0 & port=0 | Line endings |
|---|---:|---:|---:|:--:|:--:|---|---:|---:|---:|---|
| `wed_14_csv.csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01–05, 07, 08–12 | 225,628 (21.52%) | **5** | 11,882 | LF |
| `thu_15_csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01–05, 08–12 | 2,421 (0.23%) | 0 | 18,565 | LF |
| `fri_16_csv.csv` | 1,048,574 | **1** | 80 | **no** | **YES** | 01, 08–12 | 147,586 (14.07%) | 0 | 162 | LF |
| `tue_20_csv` | 7,948,748 | 0 | **84** | **YES** | no | 01–12 | 2 (0.00%) | 0 | 144,132 | **mixed** |
| `wed_21_csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01, 02, 08–10 | 17,557 (1.67%) | 0 | 61 | LF |
| `thu_22_csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01–06, 08–12 | 3,278 (0.31%) | **9** | 16,173 | LF |
| `fri_23_csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01–05, 08–12 | 2,614 (0.25%) | 0 | 16,067 | LF |
| `wed_29_csv` | 613,071 | **33** | 80 | **no** | no | 01–05, 08–12 | 6,121 (1.00%) | 0 | 10,740 | LF |
| `thu_1_csv` | 331,100 | **25** | 80 | **no** | no | 01–05, 08–12 | 97 (0.03%) | 0 | 7,149 | LF |
| `fri_2_csv` | 1,048,575 | 0 | 80 | **no** | **YES** | 01–05, 08–12 | 5,459 (0.52%) | 0 | 13,382 | **CRLF** |
| **Total** | **16,232,943** | **59** | | 1 of 10 | **7 files** | **never 00 or 13–23** | **410,763** | **14** | **238,313** | |

Notes on the table:

- **"Data rows" excludes the stray mid-file header rows**; `wc -l − 1` per file, summed, is
  **16,233,002** = 16,232,943 + 59. That is the published total exactly.
- **"Dup rows"** = (`wc -l` − 1) − |distinct lines after dropping the first header|, computed with
  an exact disk-backed `LC_ALL=C sort -u`, not sampled. Of the 410,763, **56 are artefacts of the
  stray header rows** repeating (33→32 in `wed_29`, 25→24 in `thu_1`, 1→0 in `fri_16`).
  **410,763 − 56 = 410,707**, reproducing the earlier audit's figure to the row.
- **`proto=0 & port=0`** rows: in every file the protocol-0 count and the destination-port-0 count
  are the same rows (identical counts, and the AND equals both) except `tue_20`, where port-0 is
  144,156 and proto-0 is 144,132 — 24 rows have port 0 with a nonzero protocol. These are the
  CICFlowMeter TSO-failure flows; some carry fabricated endpoints such as
  `8.0.6.4-8.6.0.1-0-0-0`, though most (1,307 distinct source IPs in `tue_20`) carry real addresses.
- **`thu_22`** holds 9 rows dated in January 1970 (`10/01/1970`, `11/01/1970`, `12/01/1970`) and
  `wed_14` holds 5 (`05/01/1970`, `08/01/1970`, `12/01/1970`). 14 total, matching the prior audit.
- **Encoding:** zero bytes ≥ 0x80 anywhere in any file (checked over full files in the scan pass) —
  pure ASCII, so UTF-8-clean. Zero `"` characters, so comma-splitting is exact and the field-count
  pass is trustworthy: **every file had a single field-count value across every physical line**
  (80, or 84 for `tue_20`), so there are no embedded newlines and `wc -l` is a true row count.
- **Final line:** all ten files end in a complete line (last byte `0x0a`).
- **Line endings:** `fri_2_csv` is CRLF throughout (1,048,576 of 1,048,576 lines).
  `tue_20_csv` is **mixed** — 577,787 of 7,948,749 lines are CR-terminated, including the header,
  and the final line is not. The other eight are pure LF. (The earlier audit did not record this for
  CIC-2018; it is a real property of the data, present in both copies alike.)

### 1.1 The three headline defects, tested directly

1. **2^20 truncation — NOT fixed, and there is nothing to fix.** Seven files hold exactly
   1,048,575 data rows (1,048,576 lines including the header = 2^20). Identical to the current
   copy. The decisive evidence that this is upstream: the ten-file total is *exactly* the published
   16,233,002. A copy that had been truncated locally could not land on the published figure; a
   copy that was un-truncated would exceed it. Label totals corroborate — Benign 13,484,708,
   attack 2,748,235, i.e. 16.93% attack share, matching the published "~17% attack".
2. **Missing IP columns — NOT fixed.** Only `tue_20_csv` has `Flow ID, Src IP, Src Port, Dst IP`
   (84 columns). The other nine are 80 columns beginning `Dst Port,Protocol,Timestamp,…` with no IP
   field of any kind. Verified by reading the header of each file. Unchanged from the current copy.
3. **12-hour clock — NOT fixed.** A full hour histogram over the `Timestamp` column of all ten
   files: hours **01 through 12 only**. Hour `00` appears zero times and hours `13`–`23` appear
   zero times, in every file, across all 16.23M rows. There is no AM/PM field. This is a 12-hour
   dial with the meridiem discarded, exactly as before.

---

## 2. Side-by-side with `~/Documents/SIH/DATA/CSV/`

Both directories live on the same btrfs subvolume (`/dev/nvme0n1p3`, device 59). Inodes differ, so
these are two independent copies, not hardlinks.

| Downloads file | SIH file | Size (bytes) | mtime (both) | Downloads btime | SIH btime | MD5 |
|---|---|---:|---|---|---|---|
| `fri_16_csv.csv` | `fri_16_csv.csv` | 333,723,605 | 2018-10-11 21:33:10 | 2026-08-28 11:17:32 | 2026-08-28 15:06:57 | `2bb33b97…5333` **identical** |
| `fri_23_csv` | `fri_23_csv.csv` | 382,840,456 | 2018-10-11 21:33:33 | 2026-08-28 11:30:31 | 2026-08-28 15:06:57 | `56a6cec1…7344` **identical** |
| `fri_2_csv` | `fri_2_csv.csv` | 352,368,373 | 2018-10-11 21:32:49 | 2026-08-28 11:34:50 | 2026-08-28 15:06:57 | `bfde2b2b…f147` **identical** |
| `thu_15_csv` | `thu_15_csv.csv` | 375,945,899 | 2018-10-11 21:38:48 | 2026-08-28 11:16:46 | 2026-08-28 15:06:57 | `30af0113…fdb6` **identical** |
| `thu_1_csv` | `thu_1_csv.csv` | 107,842,858 | 2018-10-11 21:38:38 | 2026-08-28 11:31:48 | 2026-08-28 15:06:57 | `77ff2bd5…a396` **identical** |
| `thu_22_csv` | `thu_22_csv.csv` | 382,636,202 | 2018-10-11 21:39:20 | 2026-08-28 11:30:27 | 2026-08-28 15:06:57 | `c5747e79…fa74` **identical** |
| `tue_20_csv` | `tue_20_csv.csv` | 4,054,925,350 | 2018-10-11 21:33:59 | 2026-08-28 11:19:36 | 2026-08-28 15:06:57 | `d581b086…d5c8` **identical** |
| `wed_14_csv.csv` | `wed_14_csv.csv` | 358,223,333 | 2018-10-11 21:39:44 | 2026-08-28 11:16:10 | 2026-08-28 15:06:57 | `e69ce10d…69af` **identical** |
| `wed_21_csv` | `wed_21_csv.csv` | 328,893,673 | 2018-10-11 21:40:12 | 2026-08-28 11:20:43 | 2026-08-28 15:06:57 | `04a4865f…9827` **identical** |
| `wed_29_csv` | `wed_29_csv.csv` | 209,249,758 | 2018-10-11 21:40:33 | 2026-08-28 11:31:09 | 2026-08-28 15:06:57 | `211e5d92…6f65` **identical** |

**Ten of ten byte-identical.** The 4.05 GB `tue_20` was compared by *full* MD5 of both files, not by
a head/tail sample, so no caveat applies there. Row counts and column counts are therefore
necessarily identical too, and no separate row/column comparison is needed.

Two further observations:

- Both directories carry the same stale LibreOffice lock file `.~lock.fri_16_csv.csv#` (114 bytes,
  2026-08-28 14:49); SIH additionally has `.~lock.wed_14_csv.csv#` (14:48). These are *stale locks
  from files that were opened and never saved* — every CSV's `mtime` is still 2018-10-11, so no
  spreadsheet write ever landed. This independently re-confirms §0 of the prior audit.
- The difference in `du` (`6.5G` for both CSV dirs) with `~/Documents/SIH/DATA` reporting 568 GB is
  entirely the 560 GB PCAP corpus that lives only under SIH.

---

## 3. Filename mapping

The Downloads copy does **not** use the official CIC names
(`Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`); it uses the same short `day_dd` scheme as
the current copy. Mapping, with the **actual date found inside the file** (from the Timestamp
column) rather than the name:

| Downloads | SIH (current) | Date in the data | Official CIC file | Attack of the day |
|---|---|---|---|---|
| `wed_14_csv.csv` | `wed_14_csv.csv` | 14/02/2018 | `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | FTP-BruteForce, SSH-Bruteforce |
| `thu_15_csv` | `thu_15_csv.csv` | 15/02/2018 | `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | DoS GoldenEye, DoS Slowloris |
| `fri_16_csv.csv` | `fri_16_csv.csv` | 16/02/2018 | `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | DoS Hulk, DoS SlowHTTPTest |
| `tue_20_csv` | `tue_20_csv.csv` | 20/02/2018 | `Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | DDoS LOIC-HTTP |
| `wed_21_csv` | `wed_21_csv.csv` | 21/02/2018 | `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | DDOS HOIC, DDOS LOIC-UDP |
| `thu_22_csv` | `thu_22_csv.csv` | 22/02/2018 | `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | Brute Force Web/XSS, SQL Injection |
| `fri_23_csv` | `fri_23_csv.csv` | 23/02/2018 | `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | Brute Force Web/XSS, SQL Injection |
| `wed_29_csv` | `wed_29_csv.csv` | **28/02/2018** | `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | Infiltration |
| `thu_1_csv` | `thu_1_csv.csv` | 01/03/2018 | `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | Infiltration |
| `fri_2_csv` | `fri_2_csv.csv` | 02/03/2018 | `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | Bot |

**`wed_29_csv` is misnamed in both copies.** Every one of its 613,071 rows is dated **28/02/2018**;
there is no 29 February 2018. The `logs/` directory, tellingly, names the same day `wed_28_logs`.
Any code that parses the day-of-month out of the filename to build a timeline or a split key will
be off by one for this file. Worth grepping for.

---

## 4. `logs/` — what it is, and why it cannot rescue the IP columns

Ten files, 1.8 GB, `mtime` 2018-10-10. Despite having no extension they are **ZIP archives** (`PK`
magic). Each contains a `logs/` folder of Windows Event Log files named
`DESKTOP-AN3U28N-<host-ip>.evtx`:

| Archive | `.evtx` entries | Maps to CSV day |
|---|---:|---|
| `wed_15_log` | 442 | *(named 15, but pairs with `wed_14_csv.csv`)* |
| `thu_15_log` | 441 | `thu_15_csv` |
| `fri_16_log` | 437 | `fri_16_csv.csv` |
| `tue_20_log` | 443 | `tue_20_csv` |
| `wed_21_log` | 437 | `wed_21_csv` |
| `thu_22_log` | 439 | `thu_22_csv` |
| `fri_23_log` | 423 | `fri_23_csv` |
| `wed_28_logs` | 426 | `wed_29_csv` *(which holds 28/02 data — the log name is the correct one)* |
| `thu_1_log` | 413 | `thu_1_csv` |
| `fri_2_log` | 423 | `fri_2_csv` |

These are the CSE-CIC-IDS2018 **victim-host Windows event logs** — the host-based half of the
dataset. The union across all ten archives is **451 distinct host IPs**, all in the victim range:
104 in `172.31.64.0/24`, 100 each in `.65`, `.66`, `.67`, 20 in `.68`, 27 in `.69`.

**They cannot supply the missing IP columns.** Three reasons:

1. **Wrong side of the wire.** An `.evtx` is a host's own security/system event stream. It contains
   no flow records and no CICFlowMeter row identifiers, so there is no join key back to a row in an
   80-column CSV. The 80-column rows have only `Dst Port, Protocol, Timestamp` as potential keys,
   and the timestamp is a 12-hour clock with the meridiem discarded (§1.1) — the join is ambiguous
   before it starts.
2. **Wrong direction of information.** Even a perfect join would tell us which *victim* host logged
   an event, never the *remote* peer of an arbitrary benign flow. Our host-graph models need both
   endpoints of every edge, not a victim roster.
3. **The roster adds nothing we don't have.** All 451 log host IPs already appear in `tue_20_csv`'s
   flows (`comm -12` overlap = 451, with 0 log IPs absent from `tue_20`). `tue_20` alone carries
   33,176 distinct IPs (31,291 source, 27,076 destination). The logs are a strict subset of what the
   one good CSV day already gives us.

What the logs *are* good for: a ground-truth inventory of the 451 victim hosts, which could
constrain or sanity-check a fabricated host graph (e.g. anchoring the victim subnet structure), and
a genuinely separate host-based modality if we ever want one. They are not a fix for defect #2.

**`pcap/` is empty** (0 entries, 0 bytes). The real PCAP corpus is `~/Documents/SIH/DATA/pcap/`
(10 day-directories, 560 GB) and is untouched by this audit.

---

## 5. What we should do

**Keep the current copy. Do not switch, and do not merge — there is nothing to merge.** A merge is
arithmetically impossible here: the two directories are the same bytes, so neither holds a row or a
column the other lacks. Concretely:

1. **Delete nothing yet, but stop treating `~/Downloads/DATA/CSV/` as a candidate upgrade.** It is
   the download staging area that `~/Documents/SIH/DATA/CSV/` was copied out of. Keeping it costs
   6.5 GB on a volume that is 90% full (105 GB free). Reclaiming it is safe *from a data standpoint*
   — but that is the user's call, and this audit changed nothing on disk.
2. **Keep `~/Downloads/DATA/logs/`.** It is the only copy of the 1.8 GB host-log modality on this
   machine (`~/Documents/SIH/DATA/logs/` exists too — see §6 Unverified), and it is the one part of
   the Downloads tree that is not redundant with the CSVs.
3. **The IP problem is only solvable from PCAP.** Re-running CICFlowMeter over
   `~/Documents/SIH/DATA/pcap/` is the only path to real `Src IP`/`Dst IP` for the nine 80-column
   days, and it would fix the 12-hour clock and the 2^20 cap in the same pass. That is the
   recommendation `09_data_integrity_audit.md` already makes; this audit removes the last reason to
   hope for a cheaper route.
4. **Fix the `wed_29` misnomer in code, not on disk.** Its data is 28/02/2018. Any filename-derived
   date is wrong by one day.
5. **Guard the two duplicate-heavy days.** `wed_14` (21.52%) and `fri_16` (14.07%) carry 373,214 of
   the corpus's 410,707 genuine duplicate rows between them. Whatever dedup policy we adopt must be
   applied before the split lock, not after.
6. **Strip stray header rows on load.** 59 rows across `fri_16` (1), `wed_29` (33) and `thu_1` (25)
   are verbatim repeats of the header inside the body. If the loader coerces numerics they become
   NaN rows; if it doesn't, they become a phantom `Label == "Label"` class.
7. **Handle mixed line endings in `tue_20`.** 577,787 of its lines are CRLF and the rest are LF. A
   reader that does not strip `\r` will see the final column (`Label`) as `Benign\r` on 7.3% of rows
   — a silent two-class split of every label in the largest file we have.

---

## 6. Unverified

- **Whether the `.evtx` payloads differ from day to day.** I listed the archives' contents
  (`unzip -l`) but did not extract or parse any `.evtx`. Every entry in `fri_16_log`, `thu_15_log`
  and `wed_15_log` reports the same uncompressed size of 2,166,784 bytes, which is consistent with
  fixed-size preallocated Windows event logs and *may* mean many are empty or near-empty; other
  archives show a range of sizes up to 16,846,848 bytes. I did not compare CRCs across archives, so
  I cannot say how much unique content the 1.8 GB actually holds.
- **`~/Documents/SIH/DATA/logs/`.** The directory exists with the same 198-byte size as the
  Downloads one, so it is almost certainly the same ten archives, but I did not hash or list it —
  the task scoped the log comparison to the Downloads tree.
- **The official CIC filenames** in §3 are the published naming convention matched to the dates I
  measured inside each file. I did not fetch the CIC distribution to confirm the exact strings.
- **The published 16,233,002 figure** is taken from the dataset's documentation as quoted in
  `09_data_integrity_audit.md`; I verified that our files sum to it exactly, but did not
  independently re-source the figure from CIC.
- **Duplicate rows were counted as exact whole-line matches.** Near-duplicates (rows differing only
  in a float's last digit) were not measured.
- **Timestamp seconds/minutes distribution** was not profiled; only the hour field was histogrammed.
- **`tue_20`'s 24 rows with port 0 but nonzero protocol** were counted but not inspected
  individually.

---

## 7. Commands used (regenerable)

```sh
D=~/Downloads/DATA/CSV; S=~/Documents/SIH/DATA/CSV

# identity
stat -c '%i %d %s %n' "$D"/* "$S"/*
stat -c '%n btime=%w mtime=%y' "$D"/*
md5sum < "$D/<file>" ; md5sum < "$S/<file>"          # all ten pairs, full file

# structure, hours, 1970 rows, proto/port 0, labels, stray headers  (single pass per file)
LC_ALL=C awk -f scan.awk "$D/<file>"                  # scan.awk in this session's scratchpad
wc -l < "$D/<file>"
tail -c 2 "$D/<file>" | od -An -tx1                   # final-line completeness

# exact duplicates (disk-backed, 400 MB memory cap, temp outside both trees)
LC_ALL=C tail -n +2 "$D/<file>" | LC_ALL=C sort -S 400M -T <tmp> -u | wc -l

# logs
unzip -l ~/Downloads/DATA/logs/<archive>              # listing only, never extracted
```

Peak memory stayed under 400 MB (the `sort` cap); no pandas was used and no file was ever read
whole into memory.
