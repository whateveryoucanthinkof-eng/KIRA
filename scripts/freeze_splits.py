#!/usr/bin/env python3
"""Compute the frozen train/val/test assignment once and write splits.lock.json.

Run this only to (re)create the lock or to append newly added captures. Every
trainer reads the lock; nothing recomputes a split at training time.

Stratification: captures are sorted by measured attack fraction and dealt
round-robin into train/val/test in a 4:1:1 rotation. That guarantees each split
receives quiet, mixed and busy captures instead of clustering all the benign
ones together -- which is how a test split ends up at a ~0 or ~1 base rate,
where PR-AUC is near-perfect for any ranking.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "data_unification" / "splits.lock.json"

SOURCES = {
    "CIC2018": "/var/home/samito/Documents/SIH/DATA/CSV/*.csv",
    "CIC2017": os.path.expanduser("~/Documents/SIH/test/extracted_flows/TrafficLabelling /*.csv"),
    "CTU13": os.path.expanduser("~/Documents/SIH/CTU-13-Dataset/*/*.binetflow"),
    "PCAP2018": os.path.expanduser("~/Documents/SIH/DATA/pcap/*"),
}


def _is_attack_label(raw: str) -> bool:
    """Per-corpus benign/attack classification.

    CTU-13 does not use bare "Benign": its labels look like
    `flow=Background-UDP-Established`, `flow=Normal-V42-Jist` and
    `flow=From-Botnet-V42-TCP-Attempt`. An exact match against {"benign",
    "background"} therefore classified *every* CTU-13 flow as an attack and
    reported a 100% attack fraction, when the measured rate is ~0.7%. That
    would have made the stratification meaningless and hidden a real
    base-rate problem, so matching is by substring here.
    """
    t = str(raw).strip().lower()
    if not t or t == "nan":
        return False
    if "botnet" in t:
        return True
    if "background" in t or "normal" in t or t in ("benign",):
        return False
    return True


def attack_fraction(path: str, enc=None) -> tuple[int, float]:
    head = pd.read_csv(path, nrows=0, encoding=enc, low_memory=False)
    cols = {c.strip(): c for c in head.columns}
    lab = cols.get("Label")
    if lab is None:
        return 0, 0.0
    v = pd.read_csv(path, usecols=[lab], encoding=enc, low_memory=False)[lab].astype(str).str.strip()
    c = Counter(v)
    tot = len(v)
    atk = sum(n for k, n in c.items() if _is_attack_label(k))
    return tot, atk / max(1, tot)


def main() -> int:
    inventory = []
    for f in sorted(glob.glob(SOURCES["CIC2018"])):
        n, af = attack_fraction(f)
        inventory.append(("CIC2018", os.path.basename(f), n, af))
    for f in sorted(glob.glob(SOURCES["CIC2017"])):
        n, af = attack_fraction(f, enc="latin1")
        inventory.append(("CIC2017", os.path.basename(f), n, af))
    for f in sorted(glob.glob(SOURCES["CTU13"])):
        n, af = attack_fraction(f)
        name = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
        inventory.append(("CTU13", name, n, af))
    # PCAP days inherit the attack fraction of their paired CSV day
    csv_af = {n.replace("_csv.csv", ""): af for ds, n, _c, af in inventory if ds == "CIC2018"}
    for d in sorted(glob.glob(SOURCES["PCAP2018"])):
        if not os.path.isdir(d):
            continue
        day = os.path.basename(d).replace("_pcap", "").replace("_pacap", "")
        af = csv_af.get(day)
        if af is None:  # wed_28 vs wed_29 naming mismatch
            pref = day.rsplit("_", 1)[0]
            af = next((v for k, v in csv_af.items() if k.startswith(pref)), 0.0)
        inventory.append(("PCAP2018", os.path.basename(d), 0, af))

    assignment: dict[str, dict[str, str]] = {}
    rotation = ["train"] * 4 + ["val"] + ["test"]
    for ds in sorted({d for d, *_ in inventory}):
        items = sorted([x for x in inventory if x[0] == ds], key=lambda x: x[3])
        assignment[ds] = {}
        for i, (_ds, name, _n, _af) in enumerate(items):
            assignment[ds][name] = rotation[i % len(rotation)]

    doc = {
        "policy_version": "1.0.0",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": "see data_unification/split_policy.py",
        "stratified_by": "attack_fraction, round-robin 4:1:1",
        "inventory": [
            {"dataset": d, "capture": n, "rows": c, "attack_fraction": round(a, 6)}
            for d, n, c, a in inventory
        ],
        "assignment": assignment,
    }
    LOCK.write_text(json.dumps(doc, indent=1))
    print(f"wrote {LOCK}")
    for ds, m in assignment.items():
        cnt = Counter(m.values())
        afs = {s: [a for d, n, _c, a in inventory if d == ds and m.get(n) == s] for s in ("train", "val", "test")}
        print(f"  {ds:9s} train={cnt['train']} val={cnt['val']} test={cnt['test']} | "
              f"attack-frac ranges: " +
              " ".join(f"{s}[{min(v):.3f}-{max(v):.3f}]" for s, v in afs.items() if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
