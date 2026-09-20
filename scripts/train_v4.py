#!/usr/bin/env python3
"""CyberWorld v4 trainer — forecasting, with baselines, on one command.

This is the trainer the v4 contract implies. It differs from the v3 scripts in
the ways that decide whether the project's central claim is true:

  * Targets are FUTURE (cyberworld_v4.targets). v3 took the label from the last
    element of the input window, which measures nowcasting.
  * Four splits, grouped by host, with a calibration split distinct from
    validation.
  * Baselines trained on the identical features, so every number has a floor.
  * Post-hoc temperature scaling fitted on calibration only.
  * An immutable manifest embedded in the checkpoint.

It reports nowcast and forecast side by side, because the gap between them is
the actual finding.

    python scripts/train_v4.py --cic-dir ~/Documents/SIH/DATA/CSV \
                               --ctu-dir ~/Documents/SIH/CTU-13-Dataset \
                               --rows-per-file 20000 --epochs 8
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cyberworld_v4.config import DEFAULT_CONFIG, get_contract
from cyberworld_v4.identity import offline_host
from cyberworld_v4.manifest import ExperimentManifest, set_all_seeds
from cyberworld_v4.metrics.detection import detection_metrics
from cyberworld_v4.metrics.forecasting import horizon_metrics
from cyberworld_v4.metrics.calibration import TemperatureScaler
from cyberworld_v4.metrics.bootstrap import group_bootstrap_ci, format_ci
from cyberworld_v4.models import CyberWorldForecaster, forecast_loss
from cyberworld_v4.splits import chronological_split, partition, leakage_report, TRAIN, VAL, CALIB, TEST
from cyberworld_v4.targets import build_samples, describe_targets
from cyberworld_v4.baselines import LogisticBaseline, GradientBoostingBaseline, PersistenceBaseline


def load_records(cic_dir: Path, ctu_dir: Path, rows: int, stride: int = 1):
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter

    """Load records spanning each file's full time range.

    max_rows in the adapters is a PREFIX (pandas nrows). Measured on this
    corpus, the first 60k rows of a CIC-2018 day are 87-100% a single label,
    because the CSVs are ordered in time and attacks occur in contiguous blocks.
    Training on a prefix therefore yields host trajectories that never change
    label — label churn 0.0000 — so persistence scores a perfect 1.0 and the
    forecasting task has no content at all.

    Reading with a stride covers the whole day instead, so trajectories can span
    benign -> attack transitions, which is the only thing a forecaster can learn.
    """
    recs = []

    def strided(gen, stride: int, want: int):
        out = []
        for i, r in enumerate(gen):
            if i % stride == 0:
                out.append(r)
                if len(out) >= want:
                    break
        return out

    for f in sorted(glob.glob(str(cic_dir / "*.csv"))):
        got = strided(CIC2018Adapter().parse_file(f, max_rows=rows * stride), stride, rows)
        recs.extend(got)
        print(f"  {Path(f).name:<24} {len(got):>7} records (stride {stride})")
    for f in sorted(glob.glob(str(ctu_dir / "*/*.binetflow"))):
        got = strided(CTU13Adapter().parse_netflow_csv(f, max_rows=rows * stride), stride, rows)
        recs.extend(got)
        print(f"  {(Path(f).parent.name + '/' + Path(f).name)[:24]:<24} {len(got):>7} records (stride {stride})")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic-dir", type=Path, required=True)
    ap.add_argument("--ctu-dir", type=Path, required=True)
    ap.add_argument("--rows-per-file", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=20,
                    help="sample every Nth row so the set spans the file's full "
                         "time range; a prefix is single-label and gives zero churn")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU training (refused by default)")
    ap.add_argument("--output", type=Path, default=REPO / "saved_models/v4/forecaster.pt")
    ap.add_argument("--results", type=Path, default=REPO / "results/v4_benchmark.json")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    cfg, c = DEFAULT_CONFIG, get_contract()
    set_all_seeds(args.seed)
    if args.allow_cpu:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif not torch.cuda.is_available():
        # Silently training on CPU turns an 8-minute run into an overnight one
        # and is almost never what was intended. Fail rather than fall back.
        print("CUDA not available. Training on CPU is refused by default; "
              "pass --allow-cpu to override.", file=sys.stderr)
        return 2
    else:
        dev = "cuda"
        torch.backends.cudnn.benchmark = True

    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"CyberWorld v4 training — {c.describe()}")
    print(f"  device {dev} ({gpu}) | seed {args.seed}\n")

    manifest = ExperimentManifest.create(
        f"cyberworld_v4_seed{args.seed}", seed=args.seed, config=cfg, repo=REPO,
        dataset_sources=[str(args.cic_dir), str(args.ctu_dir)],
    )

    print("Loading records:")
    records = load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, args.stride)
    print(f"  total {len(records)} records\n")

    from data_unification.multi_dataset_stream import HostTrajectoryExtractor
    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB

    print("Extracting host trajectories (TGNE-TA)...")
    ex = HostTrajectoryExtractor(tgne_ta_model=build_or_load_tgne_ta(),
                                 window_size_sec=c.window_seconds)
    traj = ex.extract_trajectories(records)
    print(f"  {len(traj)} hosts\n")

    print("Building FUTURE targets (v4)...")
    samples = []
    for host_ip, snaps in traj.items():
        h = offline_host("mixed", "mixed", "mixed", host_ip)
        samples.extend(build_samples(snaps, h, TECHNIQUE_VOCAB, cfg))
    if not samples:
        print("no samples built")
        return 1
    bal = describe_targets(samples)
    print(f"  {bal['n']} samples over {bal['hosts']} hosts")
    print(f"  current attack rate      : {bal['current_attack_rate']:.3f}")
    print(f"  future attack rate by step: {[round(v,3) for v in bal['future_attack_rate_by_step']]}")
    print(f"  onset within horizon     : {bal['onset_within_horizon']:.3f}")
    print(f"  censored (already under attack): {bal['censored_already_attacking']:.3f}\n")

    # --- grouped, chronological splits -----------------------------------
    assign = chronological_split(samples, lambda s: s.t_end, lambda s: s.host.host)
    parts = partition(samples, assign, lambda s: s.host.host)
    rep = leakage_report(parts, lambda s: s.host.host, lambda s: s.t_end)
    print("Splits (grouped by host, chronological):")
    print(f"  sizes {rep['sizes']} | host overlap clean: {rep['clean']} | test after train: {rep.get('test_after_train')}\n")
    for k in (TRAIN, VAL, CALIB, TEST):
        if not parts[k]:
            print(f"split {k} is empty — increase --rows-per-file")
            return 1

    def pack(ss):
        X = torch.tensor(np.stack([s.features for s in ss]), dtype=torch.float32)
        return dict(
            X=X,
            current=torch.tensor([s.current_attack for s in ss], dtype=torch.float32),
            future=torch.tensor(np.stack([s.future_attack for s in ss]), dtype=torch.float32),
            hazard=torch.tensor(np.stack([s.hazard_target for s in ss]), dtype=torch.float32),
            at_risk=torch.tensor(np.stack([s.at_risk for s in ss]), dtype=torch.float32),
            tech=torch.tensor(np.stack([s.future_techniques for s in ss]), dtype=torch.float32),
            sev=torch.tensor([s.severity for s in ss], dtype=torch.float32),
        )

    P = {k: pack(v) for k, v in parts.items()}
    model = CyberWorldForecaster(n_techniques=len(TECHNIQUE_VOCAB), config=cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr = P[TRAIN]
    ds = TensorDataset(tr["X"], tr["current"], tr["future"], tr["hazard"], tr["at_risk"], tr["tech"], tr["sev"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("Training:")
    best, best_state = np.inf, None
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for X, cu, fu, hz, ar, te, sv in dl:
            batch = dict(current_attack=cu.to(dev), future_attack=fu.to(dev),
                         hazard_target=hz.to(dev), at_risk=ar.to(dev),
                         future_techniques=te.to(dev), severity=sv.to(dev))
            opt.zero_grad()
            loss, _ = forecast_loss(model(X.to(dev)), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        va = P[VAL]
        with torch.no_grad():
            vb = dict(current_attack=va["current"].to(dev), future_attack=va["future"].to(dev),
                      hazard_target=va["hazard"].to(dev), at_risk=va["at_risk"].to(dev),
                      future_techniques=va["tech"].to(dev), severity=va["sev"].to(dev))
            vloss, parts_l = forecast_loss(model(va["X"].to(dev)), vb)
        print(f"  epoch {ep:>2}  train {np.mean(losses):7.4f}  val {float(vloss):7.4f}  "
              f"(cur {parts_l['current']:.3f} fut {parts_l['future']:.3f} haz {parts_l['hazard']:.3f})")
        if float(vloss) < best:
            best = float(vloss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # --- calibrate on CALIB, then score TEST once -------------------------
    print("\nCalibrating on the calibration split...")
    with torch.no_grad():
        cal = model.predict(P[CALIB]["X"].to(dev))
        te_pred = model.predict(P[TEST]["X"].to(dev))

    scaler = TemperatureScaler().fit_from_probs(
        cal["current_attack"].cpu().numpy(), P[CALIB]["current"].numpy()
    )
    print(f"  temperature {scaler.temperature:.3f} | "
          f"ECE {scaler.report()['before']['ece']:.4f} -> {scaler.report()['after']['ece']:.4f}")

    y_cur = P[TEST]["current"].numpy().astype(int)
    p_cur = scaler.transform_probs(te_pred["current_attack"].cpu().numpy())
    y_fut = P[TEST]["future"].numpy().astype(int)
    p_fut = te_pred["future_attack"].cpu().numpy()

    # Threshold on VALIDATION, never on test. With the default 0.5 the F1 was
    # 0.000 while PR-AUC was 0.96 — the ranking was fine, the operating point
    # was simply never crossed.
    from cyberworld_v4.benchmark import choose_threshold
    with torch.no_grad():
        val_pred = model.predict(P[VAL]["X"].to(dev))
    p_val = scaler.transform_probs(val_pred["current_attack"].cpu().numpy())
    thr = choose_threshold(P[VAL]["current"].numpy().astype(int), p_val)
    now = detection_metrics(y_cur, p_cur, threshold=thr)
    now["threshold_from_validation"] = float(thr)
    hor = horizon_metrics(y_fut, p_fut, window_seconds=c.window_seconds)

    # --- baselines on identical features ----------------------------------
    print("\nBaselines (same features):")
    Xtr, Xte = tr["X"].numpy(), P[TEST]["X"].numpy()
    base = {}
    for B in (LogisticBaseline, GradientBoostingBaseline):
        b = B(seed=args.seed).fit(Xtr, tr["current"].numpy().astype(int))
        m = detection_metrics(y_cur, b.predict_proba(Xte))
        base[b.name] = m
        print(f"  {b.name:<22} nowcast PR-AUC {m['pr_auc']:.4f}")
    pb = PersistenceBaseline().predict_proba(y_cur, c.forecast_steps)
    base["persistence_forecast"] = horizon_metrics(y_fut, pb, window_seconds=c.window_seconds)
    print(f"  {'persistence':<22} forecast PR-AUC by step "
          f"{[round(v,3) for v in base['persistence_forecast']['pr_auc_by_step']]}")

    idx = np.arange(len(y_cur))
    groups = np.array([s.host.host for s in parts[TEST]])
    from sklearn.metrics import average_precision_score
    ci = group_bootstrap_ci(
        list(idx), lambda i: str(groups[i]),
        lambda sub: (average_precision_score(y_cur[np.asarray(sub, int)], p_cur[np.asarray(sub, int)])
                     if len(np.unique(y_cur[np.asarray(sub, int)])) > 1 else float("nan")),
        n_resamples=300, seed=args.seed,
    )

    print("\n" + "=" * 68)
    print("RESULT — nowcasting vs forecasting")
    print("=" * 68)
    print(f"  NOWCAST  P(attack now)   PR-AUC {format_ci(ci)}  F1 {now['f1']:.3f}")
    print(f"  FORECAST P(attack at t+k):")
    for m in hor["per_step"]:
        pr = base["persistence_forecast"]["pr_auc_by_step"][m["step"] - 1]
        print(f"     +{m['horizon_seconds']:>4.0f}s  PR-AUC {m['pr_auc']:.4f}   "
              f"(persistence {pr:.4f})   Brier {m['brier']:.4f}")
    print(f"  degradation across horizon: {hor['degradation_pr_auc']:+.4f}")
    persist = base["persistence_forecast"]["pr_auc_by_step"]
    if min(persist) > 0.99:
        print()
        print("  !! DEGENERATE TASK: persistence scores ~1.0 at every horizon, so the")
        print("     attack label never changes within a host trajectory. Forecasting is")
        print("     indistinguishable from nowcasting here and these numbers mean nothing.")
        print("     Increase --rows-per-file so trajectories span label transitions.")
    label_churn = float(np.mean(y_fut[:, -1] != y_cur))
    print(f"  label churn over the horizon (fraction where A_t+K != A_t): {label_churn:.4f}")
    globals()["_label_churn"] = label_churn
    print("=" * 68)

    # --- credibility gate ------------------------------------------------
    # Every one of these makes the headline numbers unreportable. They are
    # checked and stated rather than left for a reader to notice.
    label_churn = globals().get("_label_churn", 0.0)
    problems = []
    if min(base["persistence_forecast"]["pr_auc_by_step"]) > 0.99:
        problems.append("persistence is near-perfect: the label does not change over the horizon")
    if label_churn < 0.01:
        problems.append(f"label churn {label_churn:.4f}: almost nothing to forecast")
    if ci.get("n_groups", 0) < 5:
        problems.append(f"test split has {ci.get('n_groups')} host group(s): no usable confidence interval")
    if now.get("extreme_base_rate"):
        problems.append(f"test base rate {now['positive_rate']:.4f} is extreme: PR-AUC is near 1.0 for any ranking")
    if scaler.report().get("at_grid_boundary"):
        problems.append("temperature hit the search boundary: calibration split is unrepresentative")
    if len(P[VAL]["current"]) < 500:
        problems.append(f"validation split has {len(P[VAL]['current'])} samples: threshold selection is noise")

    if problems:
        print()
        print("=" * 68)
        print("BENCHMARK NOT CREDIBLE — do not report these numbers")
        print("=" * 68)
        for x in problems:
            print(f"  - {x}")
        print("=" * 68)

    out = {
        "credible": not problems,
        "credibility_problems": problems,
        "contract": c.to_dict(), "manifest": manifest.to_dict(),
        "target_balance": bal, "splits": rep,
        "nowcast": now, "nowcast_pr_auc_ci": ci,
        "forecast": hor, "baselines": base,
        "calibration": scaler.report(),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(out, indent=2, default=float))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ck = {"model_state_dict": model.state_dict(), "n_techniques": len(TECHNIQUE_VOCAB),
          "temperature": scaler.temperature}
    manifest.metrics = {"nowcast_pr_auc": now["pr_auc"], "forecast_pr_auc_by_step": hor["pr_auc_by_step"]}
    manifest.attach_to_checkpoint(ck)
    torch.save(ck, args.output)
    print(f"\ncheckpoint : {args.output}")
    print(f"results    : {args.results}")
    print(f"fingerprint: {manifest.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
