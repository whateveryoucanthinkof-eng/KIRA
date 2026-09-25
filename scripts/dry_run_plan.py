#!/usr/bin/env python3
"""Run the WHOLE training plan end to end on a tiny synthetic corpus.

    python scripts/dry_run_plan.py            # minutes on a GPU; longer on CPU
    python scripts/dry_run_plan.py --keep     # keep the corpus and outputs

Every earlier failure that cost hours was a bug that only showed after a long
extraction -- a NameError in main(), a path nothing read, a store that did not
fit. Static checks (scripts/preflight.sh) catch some of that class; nothing
ran the real trainers on real-format data before the real run.

This does. It writes a corpus in the exact on-disk formats the plan reads --
libpcap captures per host with Ethernet/IPv4/TCP frames, CIC-2018 label CSVs
on the 12-hour clock, CIC-2017 TrafficLabelling CSVs, CTU-13 .binetflow -- and
names every capture as data_unification/splits.lock.json names it. Then it runs
scripts/run_training_plan.sh (preflight, compare, seeds, downstream, summary)
with only the epoch counts shrunk, and checks that every stage produced its
outputs, that every checkpoint carries its training-guard record, and that no
log holds a traceback.

The data is synthetic and tiny, so the NUMBERS mean nothing. What it proves is
that the plan runs from the first byte of input to the last checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# lock name -> (date, label-CSV day, attack label as the CIC-2018 CSVs spell it)
PCAP_DAYS = {
    "wed_14_pcap": ((2018, 2, 14), "wed_14", "FTP-BruteForce"),
    "thu_15_pacap": ((2018, 2, 15), "thu_15", "DoS attacks-GoldenEye"),
    "fri_16_pcap": ((2018, 2, 16), "fri_16", "DoS attacks-Hulk"),
    "tue_20_pcap": ((2018, 2, 20), "tue_20", "DDoS attacks-LOIC-HTTP"),
    "wed_21_pcap": ((2018, 2, 21), "wed_21", "DDOS attack-HOIC"),
    "thu_22_pcap": ((2018, 2, 22), "thu_22", "Brute Force -Web"),
    "fri_23_pacap": ((2018, 2, 23), "fri_23", "SQL Injection"),
    "wed_28_pcap": ((2018, 2, 28), "wed_29", "Infilteration"),   # CSV named wed_29 in the corpus
    "thu_1_pcap": ((2018, 3, 1), "thu_1", "Infilteration"),
    "fri_2_pcap": ((2018, 3, 2), "fri_2", "Bot"),
}
CIC2017 = {  # file -> (date, attack label or None)
    "Monday-WorkingHours.pcap_ISCX.csv": ((2017, 7, 3), None),
    "Tuesday-WorkingHours.pcap_ISCX.csv": ((2017, 7, 4), "FTP-Patator"),
    "Wednesday-workingHours.pcap_ISCX.csv": ((2017, 7, 5), "DoS Hulk"),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": ((2017, 7, 6), "FTP-Patator"),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": ((2017, 7, 6), "Infiltration"),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": ((2017, 7, 7), "Bot"),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ((2017, 7, 7), "PortScan"),
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": ((2017, 7, 7), "DDoS"),
}
CTU13 = {  # lock capture -> date
    "1/capture20110810.binetflow": (2011, 8, 10), "2/capture20110811.binetflow": (2011, 8, 11),
    "3/capture20110812.binetflow": (2011, 8, 12), "4/capture20110815.binetflow": (2011, 8, 15),
    "5/capture20110815-2.binetflow": (2011, 8, 15), "6/capture20110816.binetflow": (2011, 8, 16),
    "7/capture20110816-2.binetflow": (2011, 8, 16), "8/capture20110816-3.binetflow": (2011, 8, 16),
    "9/capture20110817.binetflow": (2011, 8, 17), "10/capture20110818.binetflow": (2011, 8, 18),
    "11/capture20110818-2.binetflow": (2011, 8, 18), "12/capture20110819.binetflow": (2011, 8, 19),
    "13/capture20110815-3.binetflow": (2011, 8, 15),
}

DUR = 90           # seconds of traffic per capture: 45 windows, > the 15-step history + 5-step horizon
ATTACK = (30, 60)  # attack interval, seconds into the capture


# --------------------------------------------------------------- CIC-2018 PCAP
def _frame(src, dst, sport, dport, flags, payload_len, ident):
    tcp = struct.pack("!HHIIHHHH", sport, dport, ident * 1000, 0, (5 << 12) | flags, 8192, 0, 0)
    body = tcp + b"\x00" * payload_len
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(body), ident & 0xFFFF, 0x4000, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    return b"\x02\x00\x00\x00\x00\x01" + b"\x02\x00\x00\x00\x00\x02" + b"\x08\x00" + ip + body


def _write_pcap(path: Path, packets):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts, raw in sorted(packets, key=lambda p: p[0]):
            sec = int(ts)
            f.write(struct.pack("<IIII", sec, int(round((ts - sec) * 1e6)), len(raw), len(raw)))
            f.write(raw)


def make_pcap_day(root: Path, labels_dir: Path, day: str, date, csv_day: str, attack: str, rng):
    t0 = datetime(*date, 14, 0, 0, tzinfo=timezone.utc).timestamp()   # 10:00 local (+4 h)
    server, attacker = "172.31.64.20", "18.219.211.138"
    clients = [f"172.31.69.{10 + i}" for i in range(3)]
    per_host = {ip: [] for ip in clients + [server]}
    ident = 1
    for s in range(DUR):
        for ci, c in enumerate(clients):
            ts = t0 + s + 0.1 * ci + rng.random() * 0.05
            sport = 40000 + ci * 100 + (s % 50)
            req = _frame(c, server, sport, 443, 0x18, 200 + rng.randint(0, 50), ident)
            rep = _frame(server, c, 443, sport, 0x18, 800 + rng.randint(0, 200), ident + 1)
            ident += 2
            for ip in (c, server):
                per_host[ip].append((ts, req))
                per_host[ip].append((ts + 0.01, rep))
    victim = clients[0]
    for k in range((ATTACK[1] - ATTACK[0]) * 5):
        ts = t0 + ATTACK[0] + k * 0.2
        per_host[victim].append((ts, _frame(attacker, victim, 50000 + k % 1000, 21, 0x02, 0, ident)))
        ident += 1
    d = root / day / "pcap"
    d.mkdir(parents=True, exist_ok=True)
    for ip, pk in per_host.items():
        _write_pcap(d / f"UCAP{ip}", pk)

    # Label CSV: local time on the 12-hour dial the real files use, no AM/PM.
    local0 = datetime(*date, 10, 0, 0)
    with open(labels_dir / f"{csv_day}_csv.csv", "w", newline="", encoding="latin1") as fh:
        w = csv.writer(fh)
        w.writerow(["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts",
                    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Label"])
        for s in range(0, DUR, 2):
            ts = (local0 + timedelta(seconds=s)).strftime("%d/%m/%Y %H:%M:%S")
            w.writerow([443, 6, ts, 10000, 1, 1, 220, 900, "Benign"])
        for s in range(ATTACK[0], ATTACK[1]):
            ts = (local0 + timedelta(seconds=s)).strftime("%d/%m/%Y %H:%M:%S")
            w.writerow([21, 6, ts, 500, 1, 0, 0, 0, attack])


# --------------------------------------------------------------- CIC-2017
def make_cic2017(d: Path, name: str, date, attack, rng):
    hdr = ["Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
           " Protocol", " Timestamp", " Flow Duration", " Total Fwd Packets",
           " Total Backward Packets", "Total Length of Fwd Packets",
           " Total Length of Bwd Packets", " Label"]
    hosts = [f"192.168.10.{h}" for h in (5, 8, 9)]
    server = "192.168.10.50"
    t0 = datetime(*date, 9, 0, 0)
    with open(d / name, "w", newline="", encoding="latin1") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for s in range(DUR):
            ts = (t0 + timedelta(seconds=s)).strftime("%d/%m/%Y %H:%M:%S")
            for i, h in enumerate(hosts):
                sp = 40000 + i * 100 + s % 50
                lab = "BENIGN"
                if attack and i == 0 and ATTACK[0] <= s < ATTACK[1]:
                    lab = attack
                w.writerow([f"{h}-{server}-{sp}-80-6", h, sp, server, 80, 6, ts,
                            rng.randint(1000, 90000), rng.randint(1, 8), rng.randint(1, 8),
                            rng.randint(100, 3000), rng.randint(100, 9000), lab])


# --------------------------------------------------------------- CTU-13
def make_ctu(d: Path, capture: str, date, rng):
    p = d / capture
    p.parent.mkdir(parents=True, exist_ok=True)
    hosts = [f"147.32.84.{h}" for h in (100, 101, 102)]
    bot, cc = "147.32.84.165", "212.117.171.138"
    t0 = datetime(*date, 9, 46, 0)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport",
                    "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"])
        for s in range(DUR):
            for i, h in enumerate(hosts + [bot]):
                ts = (t0 + timedelta(seconds=s + 0.1 * i)).strftime("%Y/%m/%d %H:%M:%S.%f")
                if h == bot and s % 2 == 0:
                    w.writerow([ts, round(rng.random(), 6), "tcp", bot, 1025 + s % 500, "->", cc,
                                65500, "FSPA_FSPA", 0, 0, rng.randint(4, 12), rng.randint(300, 2000),
                                rng.randint(100, 500),
                                "flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data"])
                else:
                    w.writerow([ts, round(rng.random(), 6), "tcp", h, 30000 + s % 900, "->",
                                "147.32.80.9", 80, "SA_A", 0, 0, rng.randint(2, 10),
                                rng.randint(200, 4000), rng.randint(100, 900),
                                "flow=Background-TCP-Established"])


def build_corpus(root: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    paths = {k: root / k for k in ("pcap", "csv", "cic2017", "ctu13")}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for day, (date, csv_day, attack) in PCAP_DAYS.items():
        make_pcap_day(paths["pcap"], paths["csv"], day, date, csv_day, attack, rng)
    for name, (date, attack) in CIC2017.items():
        make_cic2017(paths["cic2017"], name, date, attack, rng)
    for cap, date in CTU13.items():
        make_ctu(paths["ctu13"], cap, date, rng)
    return paths


# --------------------------------------------------------------- run + verify
def _bash() -> str:
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe"):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("bash not found; the plan is a bash script")


def _expected(out: Path):
    enc = lambda seed, ip: out / f"compare_seed{seed}" / "encoders" / f"cic2018__{ip}" / \
        f"enc_cic2018__{ip}-cic2018.pth"
    exp = {}
    for ip in ("full", "cross_network"):
        exp[f"encoder seed42 {ip}"] = enc(42, ip)
        exp[f"branch A seed42 {ip}"] = out / "compare_seed42" / "branch_a" / f"cic2018__{ip}.pt"
    for seed in (123, 2024):
        exp[f"encoder seed{seed}"] = enc(seed, "cross_network")
        exp[f"branch A seed{seed}"] = out / f"compare_seed{seed}" / "branch_a" / "cic2018__cross_network.pt"
    exp["comparison seed42"] = out / "compare_seed42" / "comparison.json"
    exp["branch B"] = out / "downstream" / "branch_b" / "host_wdt.pt"
    exp["DeepOP"] = out / "downstream" / "deepop" / "cwa_forecast_decoder.pt"
    exp["cross-year scores"] = out / "downstream" / "cross_year.json"
    return exp


def verify(out: Path) -> list:
    import torch
    problems = []
    for what, p in _expected(out).items():
        if not p.exists():
            problems.append(f"missing {what}: {p}")
            continue
        if p.suffix == ".pt":
            ck = torch.load(p, map_location="cpu", weights_only=False)
            g = ck.get("training_guard")
            if not g:
                problems.append(f"{what}: checkpoint carries no training_guard record")
            else:
                acts = [h["action"] for h in g["history"]]
                print(f"  {what:24s} best epoch {g['best_epoch']} | actions {acts} | "
                      f"step-backs {g['step_backs']}")
        if p.suffix == ".pth":
            gj = Path(str(p)[:-4] + "_training_guard.json")
            if not gj.exists():
                problems.append(f"{what}: no {gj.name}")
            else:
                g = json.loads(gj.read_text())
                print(f"  {what:24s} best epoch {g['best_epoch']} | "
                      f"actions {[h['action'] for h in g['history']]}")
    for log in out.rglob("*.log"):
        text = log.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text:
            problems.append(f"traceback in {log}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="work directory (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the corpus and outputs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--stages", default="preflight,warm,train,summary")
    a = ap.parse_args()

    work = a.out or Path(tempfile.mkdtemp(prefix="cyberworld_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    t = time.time()
    paths = build_corpus(work / "corpus")
    print(f"[dry-run] synthetic corpus in {work / 'corpus'} ({time.time() - t:.1f}s)", flush=True)

    out = work / "plan"
    env = dict(os.environ)
    env.update({
        # Forward slashes: the plan passes some paths through shell-style
        # splitting, where a Windows backslash is an escape character.
        "PCAP_ROOT": paths["pcap"].as_posix(), "CIC2018_CSV_DIR": paths["csv"].as_posix(),
        "CIC2017_DIR": paths["cic2017"].as_posix(), "CTU_DIR": paths["ctu13"].as_posix(),
        "OUT": out.as_posix(), "PYTHON": Path(sys.executable).as_posix(), "NUM_WORKERS": "0",
        # The real run's own encoder flags (e.g. --ingest_workers) stay in, so
        # the dry run exercises the same ingest path; ours come last and win.
        "PLAN_ENCODER_EXTRA": (os.environ.get("PLAN_ENCODER_EXTRA", "")
                               + f" --n_epoch {a.epochs} --batch_size 200").strip(),
        "PLAN_BRANCH_A_EXTRA": f"--epochs {a.epochs}",
        # DeepOP must run even though Branch B cannot beat persistence on
        # synthetic traffic, or its code path would go untested.
        "PLAN_DOWNSTREAM_EXTRA": f"--epochs {a.epochs} --allow-noncredible-branch-b",
        "MIN_FREE_GB": "1",
        # Its own cache: synthetic captures must never land in the real one.
        "INGEST_CACHE": (out / ".ingest_cache").as_posix(),
    })
    if not shutil.which("systemd-run"):
        env["ALLOW_UNCAPPED"] = "1"
    bash = _bash()
    rc = 0
    for stage in [s for s in a.stages.split(",") if s]:
        t = time.time()
        print(f"[dry-run] stage {stage} ...", flush=True)
        r = subprocess.run([bash, "scripts/run_training_plan.sh", stage], cwd=REPO, env=env)
        print(f"[dry-run] stage {stage}: exit {r.returncode} ({(time.time() - t) / 60:.1f} min)", flush=True)
        if r.returncode != 0:
            rc = r.returncode
            break

    problems = [] if rc else verify(out)
    if rc:
        problems.append(f"a stage exited with {rc}; see {out}/**/logs")
    print()
    if problems:
        print("DRY RUN FAILED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("DRY RUN PASSED: every stage ran and wrote its outputs; every checkpoint "
              "carries its training-guard record; no tracebacks.")
    if not a.keep and not problems and a.out is None:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"work directory kept: {work}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
