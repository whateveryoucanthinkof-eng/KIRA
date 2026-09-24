"""Which encoder should the system use? Measure it, CIC-2018 -> CIC-2017.

Three encoders, one downstream protocol:

  warden          BiTA trained on Warden alerts only (the paper's dataset)
  cic2018         BiTA trained on CIC-2018 PCAP train days only
  warden_ft       warden, then fine-tuned on the same CIC-2018 days

For each, Branch A is trained on CIC-2018 (lock train days), tuned on the
other CIC-2018 days, and scored ONCE on all of CIC-2017, with the same seed and
settings -- so the encoder is the only thing that changes. CIC-2017 is never
seen by any encoder or any tuning step.

    python scripts/run_encoder_comparison.py \\
        --warden-dir <Warden *March_e.csv dir> \\
        --pcap-root <CIC-2018 <day>_pcap dirs> --cic2018-csv-dir <CIC-2018 CSVs> \\
        --cic2017-dir <CIC-2017 TrafficLabelling CSVs> \\
        --out results/encoder_comparison [--ip-ablation] [--dry-run]

--ip-ablation adds a second run of every arm with the IP node features reduced
to the network-invariant flags (CYBERWORLD_ABLATE_NODE_FEATURES=cross_network):
the octets name CIC-2018's AWS network, which CIC-2017 does not use, and
is_private/is_global encode how the captures were staged.

Every step writes its output and a log under --out and is skipped on re-run if
its output exists (--force re-runs). The result is <out>/comparison.md and
<out>/comparison.json.

Reading the result: rank on CIC-2017 technique macro-F1 over classes training
saw, then risk ROC-AUC. The two classes 2018 lacks (PortScan, Heartbleed) are
reported per arm but cannot separate the arms: no arm's Branch A saw them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ARMS = ("warden", "cic2018", "warden_ft")
IP_VARIANTS = {"full": "", "cross_network": "cross_network"}


def encoder_path(out: Path, arm: str, ip: str) -> Path:
    return out / "encoders" / f"{arm}__{ip}" / f"enc_{arm}__{ip}-{arm}.pth"


def encoder_cmd(a, arm: str, ip: str):
    save_dir = encoder_path(a.out, arm, ip).parent
    common = [a.python, "bita/train.py", "--prefix", f"enc_{arm}__{ip}", "--data_name", arm,
              "--save_dir", str(save_dir),
              "--checkpoint_dir", str(a.out / ".spill" / f"{arm}__{ip}_epochs"),
              "--seed", str(a.seed)]
    if arm == "warden":
        return common + ["--dataset_dir", str(a.warden_dir)] + shlex.split(a.encoder_args)
    cmd = common + ["--pcap2018_root", str(a.pcap_root),
                    "--pcap2018_label_dir", str(a.cic2018_csv_dir),
                    "--split_scheme", a.split_scheme, "--train_splits", "train"]
    if a.split_scheme == "cross_year_ctu":
        cmd += ["--ctu13_dir", str(a.ctu_dir)]
    if arm == "warden_ft":
        cmd += ["--init_from", str(encoder_path(a.out, "warden", ip))]
    return cmd + shlex.split(a.encoder_args)


def branch_a_paths(out: Path, arm: str, ip: str):
    return (out / "branch_a" / f"{arm}__{ip}.pt", out / "results" / f"{arm}__{ip}.json")


def branch_a_cmd(a, arm: str, ip: str):
    ckpt, res = branch_a_paths(a.out, arm, ip)
    ctu = ["--ctu-dir", str(a.ctu_dir)] if a.split_scheme == "cross_year_ctu" else []
    return [a.python, "scripts/retrain_branch_a_live.py",
            "--pcap-root", str(a.pcap_root), "--cic2018-csv-dir", str(a.cic2018_csv_dir),
            "--cic2017-dir", str(a.cic2017_dir), "--split-scheme", a.split_scheme,
            *ctu,
            "--tgne", str(encoder_path(a.out, arm, ip)),
            "--output", str(ckpt), "--results-json", str(res),
            "--seed", str(a.seed)] + shlex.split(a.branch_a_args)


def run(step: str, cmd, output: Path, env_extra: dict, a) -> bool:
    """Run one step unless its output exists. Returns False on failure."""
    if output.exists() and not a.force:
        print(f"[skip] {step}: {output} exists", flush=True)
        return True
    env_txt = " ".join(f"{k}={v}" for k, v in env_extra.items() if v)
    print(f"[run ] {step}\n       {env_txt + ' ' if env_txt else ''}{' '.join(shlex.quote(c) for c in cmd)}",
          flush=True)
    if a.dry_run:
        return True
    log = a.out / "logs" / f"{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{REPO}{os.pathsep}{REPO / 'bita'}")
    env.pop("CYBERWORLD_ABLATE_NODE_FEATURES", None)
    env.update({k: v for k, v in env_extra.items() if v})
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not output.exists():
        print(f"[FAIL] {step}: exit {rc}; see {log}", flush=True)
        return False
    return True


def summarise(a, arms, ips) -> dict:
    rows = []
    for ip in ips:
        for arm in arms:
            _ckpt, res_path = branch_a_paths(a.out, arm, ip)
            row = {"encoder": arm, "ip_features": ip, "status": "missing"}
            if res_path.exists():
                r = json.loads(res_path.read_text())
                t, v = r.get("test", {}), r.get("validation", {})
                u = t.get("unseen_class_report") or {}
                row.update({
                    "status": "ok",
                    "val_2018_macro_f1": v.get("tech_macro_f1"),
                    "test_2017_macro_f1_seen": u.get("macro_f1_seen"),
                    "test_2017_macro_f1_all": u.get("macro_f1_all"),
                    "test_2017_accuracy": t.get("tech_accuracy"),
                    "test_2017_majority_baseline": t.get("tech_majority_baseline"),
                    "test_2017_risk_auc": t.get("risk_auc"),
                    "test_2017_risk_brier": t.get("risk_brier"),
                    "test_2017_risk_brier_baseline": t.get("risk_brier_baseline"),
                    "test_2017_unseen_share": u.get("test_fraction_in_unseen_classes"),
                    "unseen_classes": [c["class"] for c in u.get("unseen_classes", [])],
                    "credible": (r.get("credibility") or {}).get("credible"),
                })
            rows.append(row)

    def key(r):
        f = r.get("test_2017_macro_f1_seen")
        auc = r.get("test_2017_risk_auc")
        return (f if isinstance(f, (int, float)) else -1.0,
                auc if isinstance(auc, (int, float)) and auc == auc else -1.0)

    ranked = sorted([r for r in rows if r["status"] == "ok"], key=key, reverse=True)
    scheme = getattr(a, "split_scheme", "cross_year")
    trained_on = "CIC-2018 + CTU-13" if scheme == "cross_year_ctu" else "CIC-2018"
    return {"protocol": f"encoder varies; Branch A trained+tuned on {trained_on} "
                        f"({scheme}), scored once on CIC-2017",
            "split_scheme": scheme, "seed": getattr(a, "seed", None),
            "rows": rows, "ranking": [f"{r['encoder']} / {r['ip_features']}" for r in ranked]}


def to_markdown(summary: dict) -> str:
    def f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "—"
    lines = ["# Encoder comparison: CIC-2018 -> CIC-2017", "",
             summary["protocol"] + ".", "",
             "| encoder | IP features | 2018 val macro-F1 | 2017 macro-F1 (seen classes) | "
             "2017 macro-F1 (all) | 2017 accuracy / majority | 2017 risk AUC | 2017 Brier / baseline |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        if r["status"] != "ok":
            lines.append(f"| {r['encoder']} | {r['ip_features']} | not run | | | | | |")
            continue
        lines.append(
            f"| {r['encoder']} | {r['ip_features']} | {f(r['val_2018_macro_f1'])} | "
            f"{f(r['test_2017_macro_f1_seen'])} | {f(r['test_2017_macro_f1_all'])} | "
            f"{f(r['test_2017_accuracy'])} / {f(r['test_2017_majority_baseline'])} | "
            f"{f(r['test_2017_risk_auc'])} | {f(r['test_2017_risk_brier'])} / "
            f"{f(r['test_2017_risk_brier_baseline'])} |")
    ok = [r for r in summary["rows"] if r["status"] == "ok"]
    if ok:
        u = ok[0]
        lines += ["", f"Classes in CIC-2017 that no arm's training contained: "
                      f"{', '.join(u['unseen_classes']) or 'none'} "
                      f"({f(u['test_2017_unseen_share'])} of test samples). They are excluded "
                      f"from the 'seen classes' column and listed per arm in results/*.json."]
    if summary["ranking"]:
        lines += ["", "Ranking (2017 seen-class macro-F1, then risk AUC): " + " > ".join(summary["ranking"])]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warden-dir", type=Path)
    ap.add_argument("--pcap-root", type=Path)
    ap.add_argument("--cic2018-csv-dir", type=Path)
    ap.add_argument("--cic2017-dir", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "encoder_comparison")
    ap.add_argument("--arms", default=",".join(ARMS), help=f"subset of {ARMS}")
    ap.add_argument("--ip-ablation", action="store_true",
                    help="also run every arm with network-invariant IP features")
    ap.add_argument("--ip-variants", default=None,
                    help=f"comma list from {tuple(IP_VARIANTS)}; overrides --ip-ablation "
                         f"(e.g. rerun only the winning variant with more seeds)")
    ap.add_argument("--split-scheme", choices=("cross_year", "cross_year_ctu"), default="cross_year",
                    help="cross_year_ctu adds CTU-13 to training/validation only (needs --ctu-dir)")
    ap.add_argument("--ctu-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-args", default="", help="extra args for bita/train.py")
    ap.add_argument("--branch-a-args",
                    default="--risk-objective soft_bce --risk-target hazard --epochs 8 --patience 3",
                    help="extra args for scripts/retrain_branch_a_live.py (same for every arm)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; choose from {ARMS}")
    if "warden_ft" in arms and "warden" not in arms and not a.summarise_only:
        # The fine-tune starts from the warden arm's encoder.
        arms = ["warden"] + arms
    ips = ["full", "cross_network"] if a.ip_ablation else ["full"]
    failed = []
    if a.ip_variants:
        ips = [x for x in a.ip_variants.split(",") if x]
        bad = [x for x in ips if x not in IP_VARIANTS]
        if bad:
            ap.error(f"unknown IP variant(s) {bad}; choose from {tuple(IP_VARIANTS)}")

    if not a.summarise_only:
        need = {"--cic2017-dir": a.cic2017_dir, "--pcap-root": a.pcap_root,
                "--cic2018-csv-dir": a.cic2018_csv_dir}
        if a.split_scheme == "cross_year_ctu":
            need["--ctu-dir"] = a.ctu_dir
        if "warden" in arms:
            need["--warden-dir"] = a.warden_dir
        missing = [k for k, v in need.items() if not v]
        if missing:
            ap.error(f"missing {missing}")
        a.out.mkdir(parents=True, exist_ok=True)
        order = [x for x in ARMS if x in arms]       # warden before warden_ft
        for ip in ips:
            env = {"CYBERWORLD_ABLATE_NODE_FEATURES": IP_VARIANTS[ip]}
            for arm in order:
                ok = run(f"encoder__{arm}__{ip}", encoder_cmd(a, arm, ip),
                         encoder_path(a.out, arm, ip), env, a)
                if ok:
                    ok = run(f"branch_a__{arm}__{ip}", branch_a_cmd(a, arm, ip),
                             branch_a_paths(a.out, arm, ip)[1], env, a)
                if not ok:
                    failed.append(f"{arm}/{ip}")
                    print(f"arm {arm}/{ip} stopped; the others continue", flush=True)

    if a.dry_run:
        return 0
    summary = summarise(a, [x for x in ARMS if x in arms], ips)
    (a.out / "comparison.json").write_text(json.dumps(summary, indent=2))
    (a.out / "comparison.md").write_text(to_markdown(summary))
    print(to_markdown(summary))
    # A failed arm must fail the stage: the plan runs `seeds` and
    # `downstream` next, and they need these encoders.
    if failed:
        print(f"FAILED arms: {failed}; see {a.out / 'logs'}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
