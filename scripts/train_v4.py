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
from cyberworld_v4.splits import (chronological_split, frozen_capture_split, partition,
                                 leakage_report, TRAIN, VAL, CALIB, TEST)
from cyberworld_v4.metrics.earlywarning import lead_time_from_samples
from cyberworld_v4.conformal import LabelConditionalConformal, SplitConformal
from cyberworld_v4.targets import build_samples, describe_targets
from cyberworld_v4.baselines import LogisticBaseline, GradientBoostingBaseline, PersistenceBaseline


def load_captures(cic_dir: Path, ctu_dir: Path, rows: int, stride: int = 1):
    """Load records grouped BY CAPTURE, with unmappable labels dropped.

    Two changes from the previous `load_records`, which returned one flat list.

    **Capture identity is preserved.** Flattening threw away which file each
    record came from, and main() then had to call
    `offline_host("mixed", "mixed", "mixed", ip)` -- collapsing the whole point
    of `cyberworld_v4.identity`, whose docstring exists to stop 192.168.1.10 in
    CIC-2018 and the same address in CTU-13 becoming one host. It also left the
    bare IP as the only available split group, which is how a run ended up with
    11 train hosts, 1 validation host, 1 calibration host and 1 test host
    (results/v4_benchmark.json), a test base rate of 0.9997, and a confidence
    interval that could not be computed. Captures are the unit the frozen lock
    already uses, and there are 41 of them.

    **Unresolved labels are dropped.** label_resolver returns UNKNOWN with
    is_attack=False for a label its maps do not recognise, and says callers
    building a benchmark must exclude those rows. Nothing did, so they were
    trained and scored as confident negatives.

    max_rows in the adapters is a PREFIX (pandas nrows). Measured on this
    corpus, the first 60k rows of a CIC-2018 day are 87-100% a single label,
    because the CSVs are ordered in time and attacks occur in contiguous blocks.
    Training on a prefix therefore yields host trajectories that never change
    label -- label churn 0.0000 -- so persistence scores a perfect 1.0 and the
    forecasting task has no content at all.

    Reading with a stride covers the whole day instead, so trajectories can span
    benign -> attack transitions, which is the only thing a forecaster can learn.

    Returns ([(dataset, scenario, capture, path, records), ...], coverage_report).
    """
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter
    from data_unification.label_filter import (
        drop_unresolved, format_unresolved_report, merge_unresolved_reports)
    from data_unification.split_policy import capture_name_for_path

    def strided(gen, stride: int, want):
        out = []
        for i, r in enumerate(gen):
            if i % stride == 0:
                out.append(r)
                if want is not None and len(out) >= want:
                    break
        return out

    cap = None if rows is None else rows * stride
    captures, coverage = [], []

    def _add(path: str, got):
        got, rep = drop_unresolved(got)
        coverage.append(rep)
        if not got:
            return
        dataset, capture = capture_name_for_path(path)
        # CTU-13 capture names are "<scenario>/<file>"; a CIC day is one
        # scenario, so the day stem serves as both.
        scenario = capture.split("/")[0] if "/" in capture else Path(capture).stem
        captures.append((dataset, scenario, capture, path, got))
        drop = "" if not rep["unresolved_records"] else f"  -{rep['unresolved_records']} unmapped"
        print(f"  {capture[-30:]:<30} {len(got):>7} records (stride {stride}){drop}")

    for f in sorted(glob.glob(str(cic_dir / "*.csv"))):
        _add(f, strided(CIC2018Adapter().parse_file(f, max_rows=cap), stride, rows))
    for f in sorted(glob.glob(str(ctu_dir / "*/*.binetflow"))):
        _add(f, strided(CTU13Adapter().parse_netflow_csv(f, max_rows=cap), stride, rows))

    merged = merge_unresolved_reports(coverage)
    print(format_unresolved_report(merged, where="corpus"))
    return captures, merged


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
    ap.add_argument("--split", choices=("frozen", "chronological"), default="frozen",
                    help="frozen: apply data_unification/splits.lock.json, the same "
                         "capture assignment every v3 trainer uses, with CALIBRATION "
                         "carved off the end of its train captures. chronological: "
                         "recompute a time-ordered split over the captures present.")
    ap.add_argument("--calibration-fraction", type=float, default=0.15,
                    help="share of the lock's TRAIN captures held out for calibration "
                         "(--split frozen only). Never taken from validation or test.")
    ap.add_argument("--attack-role", choices=("either", "target", "source"), default="either",
                    help="which endpoint of an attack flow is labelled attacked")
    ap.add_argument("--alert-persistence", type=int, default=1,
                    help="consecutive windows above threshold before an alert counts, "
                         "for the lead-time report")
    ap.add_argument("--pos-weight", default="auto",
                    help="positive-class weight for the BCE terms. 'auto' uses "
                         "(#neg/#pos) from the train split, clamped by "
                         "--max-pos-weight; a number sets it explicitly; 1 disables it.")
    ap.add_argument("--max-pos-weight", type=float, default=20.0,
                    help="ceiling on the automatic pos_weight, so a near-empty "
                         "positive class cannot produce a term that destabilises training")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs without improving validation "
                         "AUC; 0 disables early stopping")
    ap.add_argument("--conformal-alpha", type=float, default=0.05,
                    help="miscoverage rate for the split-conformal interval fitted on "
                         "the calibration split")
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

    print("Loading captures:")
    captures, coverage = load_captures(args.cic_dir, args.ctu_dir,
                                       args.rows_per_file, args.stride)
    n_records = sum(len(r) for *_x, r in captures)
    print(f"  {len(captures)} captures, {n_records} records")
    print()
    if not captures:
        print("no captures loaded")
        return 1

    from data_unification.multi_dataset_stream import HostTrajectoryExtractor
    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
    from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB

    print("Extracting host trajectories (TGNE-TA), one capture at a time...")
    # Per capture, not over the pooled corpus. A CIC-2018 day and a CTU-13
    # botnet scenario are unrelated networks; one merged TGNE neighbour graph
    # would make their hosts each other's temporal neighbours. Branch A already
    # extracts this way (scripts/retrain_branch_a_live.py::_store_per_capture);
    # this trainer did not.
    ex = HostTrajectoryExtractor(tgne_ta_model=build_or_load_tgne_ta(),
                                 window_size_sec=c.window_seconds,
                                 attack_role=args.attack_role)

    print("Building FUTURE targets (v4)...")
    samples = []
    for dataset, scenario, capture, _path, recs in captures:
        traj = ex.extract_trajectories(recs)
        for host_ip, snaps in traj.items():
            # Full namespaced identity. This was offline_host("mixed", "mixed",
            # "mixed", ip), which made the same private address in two corpora
            # one host and left no group above the IP to split on.
            h = offline_host(dataset, scenario, capture, host_ip)
            samples.extend(build_samples(snaps, h, TECHNIQUE_VOCAB, cfg))
        print(f"  {capture[-30:]:<30} {len(traj):>5} hosts, {len(samples):>7} samples cumulative")
    if not samples:
        print("no samples built")
        return 1
    bal = describe_targets(samples)
    print()
    print(f"  {bal['n']} samples over {bal['hosts']} hosts")
    print(f"  current attack rate      : {bal['current_attack_rate']:.3f}")
    print(f"  future attack rate by step: {[round(v,3) for v in bal['future_attack_rate_by_step']]}")
    print(f"  onset within horizon     : {bal['onset_within_horizon']:.3f}")
    print(f"  censored (already under attack): {bal['censored_already_attacking']:.3f}")
    print()

    # --- grouped splits ---------------------------------------------------
    # The group is the CAPTURE, not the host. Splitting on the host put 1 host
    # in test and made the bootstrap interval undefined; the capture is also
    # the unit the frozen lock already assigns, so the two strategies below
    # speak the same language.
    def group_of(sample):
        return f"{sample.host.dataset}|{sample.host.capture}"

    if args.split == "frozen":
        from data_unification.split_policy import load_lock
        assign = frozen_capture_split(samples, group_of, load_lock(),
                                      calibration_fraction=args.calibration_fraction)
    else:
        assign = chronological_split(samples, lambda s: s.t_end, group_of)
    parts = partition(samples, assign, group_of)
    rep = leakage_report(parts, group_of, lambda s: s.t_end)
    rep["strategy"] = assign.strategy
    rep["notes"] = assign.notes
    rep["groups"] = assign.groups
    rep["unique_hosts_by_split"] = {
        k: len({s.host.key for s in v}) for k, v in parts.items()}
    rep["label_coverage"] = coverage
    print(f"Splits (grouped by capture, {assign.strategy}):")
    print(f"  sizes {rep['sizes']}")
    print(f"  captures {assign.counts()} | hosts {rep['unique_hosts_by_split']}")
    print(f"  capture overlap clean: {rep['clean']} | test after train: {rep.get('test_after_train')}")
    print()
    for k in (TRAIN, VAL, CALIB, TEST):
        if not parts[k]:
            print(f"split {k} is empty -- increase --rows-per-file, or add captures")
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

    tr = P[TRAIN]

    # Input standardisation, fitted on TRAIN only. The 27-D state is an
    # unbounded TGNE latent glued to 15 attributes clipped to [0, 1]; an LSTM
    # gate sums them, so without this whichever block is larger dominates.
    model.fit_input_normalizer(tr["X"].to(dev))
    _m = model.encoder.input_mean.detach().cpu().numpy()
    _s = model.encoder.input_std.detach().cpu().numpy()
    print(f"Input normaliser fitted on {len(tr['X'])} train windows:")
    print(f"  latent dims 0-11  mean |{np.abs(_m[:12]).mean():.3f}|  std {_s[:12].mean():.3f}")
    print(f"  attrs  dims 12-26 mean |{np.abs(_m[12:]).mean():.3f}|  std {_s[12:].mean():.3f}")
    _ratio = _s[:12].mean() / max(_s[12:].mean(), 1e-9)
    print(f"  latent/attr scale ratio {_ratio:.1f}x"
          + ("  <- the two blocks were NOT comparable" if _ratio > 3 or _ratio < 1 / 3 else ""))
    print()

    # Class imbalance. `forecast_loss` has always accepted pos_weight and
    # nothing ever passed one, so the positive class was weighted 1.0 against
    # a base rate the shipped run measured at 0.2263. pos_weight is the
    # standard BCE correction, (#neg / #pos), and is clamped so a near-empty
    # positive class cannot produce a 1000x term that destabilises training.
    _pos = float(tr["current"].mean())
    if args.pos_weight == "auto":
        _pw = min(max((1.0 - _pos) / max(_pos, 1e-6), 1.0), args.max_pos_weight)
    else:
        _pw = float(args.pos_weight)
    pos_weight = torch.tensor([_pw], device=dev) if _pw > 1.0 else None
    print(f"Class balance: train positive rate {_pos:.4f} -> pos_weight "
          f"{_pw:.2f}" + ("" if pos_weight is not None else " (disabled)"))
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(tr["X"], tr["current"], tr["future"], tr["hazard"], tr["at_risk"], tr["tech"], tr["sev"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("Training:")
    # Select on the METRIC, not on validation loss.
    #
    # Loss and ranking quality are not the same thing, and this trainer reports
    # ROC-AUC and PR-AUC while selecting the epoch with the lowest multi-task
    # loss -- a sum of five terms, four of which the headline numbers do not
    # measure. The same defect was already fixed for Branch A (composite
    # selection, claude_latest_analysis/27) and not here.
    #
    # The score is the mean of nowcast ROC-AUC and forecast ROC-AUC, both on
    # VALIDATION. ROC rather than PR because the base rate moves between
    # splits and ROC-AUC does not move with it; mean of the two because
    # optimising nowcast alone is what produced a model that loses to
    # persistence at every horizon.
    from sklearn.metrics import roc_auc_score

    def _val_score(m):
        m.eval()
        with torch.no_grad():
            pr = m.predict(P[VAL]["X"].to(dev))
        y_now = P[VAL]["current"].numpy().astype(int)
        p_now = pr["current_attack"].cpu().numpy()
        y_fut = P[VAL]["future"].numpy().astype(int).ravel()
        p_fut_ = pr["future_attack"].cpu().numpy().ravel()
        a = roc_auc_score(y_now, p_now) if len(np.unique(y_now)) > 1 else float("nan")
        b = roc_auc_score(y_fut, p_fut_) if len(np.unique(y_fut)) > 1 else float("nan")
        both = [v for v in (a, b) if np.isfinite(v)]
        return (float(np.mean(both)) if both else float("nan")), a, b

    best, best_state, best_ep, stale = -np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for X, cu, fu, hz, ar, te, sv in dl:
            batch = dict(current_attack=cu.to(dev), future_attack=fu.to(dev),
                         hazard_target=hz.to(dev), at_risk=ar.to(dev),
                         future_techniques=te.to(dev), severity=sv.to(dev))
            opt.zero_grad()
            loss, _ = forecast_loss(model(X.to(dev)), batch, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        va = P[VAL]
        with torch.no_grad():
            vb = dict(current_attack=va["current"].to(dev), future_attack=va["future"].to(dev),
                      hazard_target=va["hazard"].to(dev), at_risk=va["at_risk"].to(dev),
                      future_techniques=va["tech"].to(dev), severity=va["sev"].to(dev))
            vloss, parts_l = forecast_loss(model(va["X"].to(dev)), vb, pos_weight=pos_weight)
        score, auc_now, auc_fut = _val_score(model)
        flag = ""
        if np.isfinite(score) and score > best:
            best, best_ep, stale = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  *best"
        else:
            stale += 1
        print(f"  epoch {ep:>2}  train {np.mean(losses):7.4f}  val {float(vloss):7.4f}  "
              f"| val AUC now {auc_now:.4f} fut {auc_fut:.4f} -> {score:.4f}{flag}")
        if args.patience and stale >= args.patience:
            print(f"  early stop: {stale} epochs without improving validation AUC")
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"  selected epoch {best_ep} (validation AUC {best:.4f})")
    else:
        print("  WARNING: no epoch produced a finite validation AUC; keeping the last "
              "weights. The validation split is probably single-class.")

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
    # Bootstrap resamples CAPTURES. Resampling on the bare host made n_groups=1
    # and the interval undefined; the capture is also the unit the split is
    # made on, which is what a grouped bootstrap is supposed to respect.
    groups = np.array([group_of(s) for s in parts[TEST]])
    from sklearn.metrics import average_precision_score
    ci = group_bootstrap_ci(
        list(idx), lambda i: str(groups[i]),
        lambda sub: (average_precision_score(y_cur[np.asarray(sub, int)], p_cur[np.asarray(sub, int)])
                     if len(np.unique(y_cur[np.asarray(sub, int)])) > 1 else float("nan")),
        n_resamples=300, seed=args.seed,
    )

    # --- early warning: lead time, the metric the product claim is about ---
    # cyberworld_v4/metrics/earlywarning.py has existed and been tested since
    # v4 landed, and had no caller. "Forecast horizon x window seconds" is not
    # lead time; lead time is measured against the event, over episodes the
    # model missed as well as the ones it caught.
    # The forecast head needs its OWN operating point. `thr` was chosen for
    # P(attack now); applying it to P(attack at t+k) compares a threshold to a
    # distribution it was never fitted on. Both are still chosen on validation,
    # never on test.
    p_val_fut = val_pred["future_attack"].cpu().numpy()
    y_val_fut_any = (P[VAL]["future"].numpy().astype(int).max(axis=1))
    thr_fut = float(choose_threshold(y_val_fut_any, p_val_fut.max(axis=1)))
    lead = lead_time_from_samples(parts[TEST], p_fut, threshold=thr_fut,
                                  persistence=args.alert_persistence,
                                  window_seconds=c.window_seconds)
    lead["threshold_from_validation"] = thr_fut

    # --- conformal intervals on the hazard curve, fitted on CALIB only ------
    with torch.no_grad():
        cal_fut = model.predict(P[CALIB]["X"].to(dev))["future_attack"].cpu().numpy()
    conf = SplitConformal(alpha=args.conformal_alpha)
    conf.fit(P[CALIB]["future"].numpy().ravel(), cal_fut.ravel())
    # groups=label: the interval's marginal coverage is set by the benign
    # majority, so attack-window coverage is reported on its own.
    conformal = conf.evaluate(y_fut.ravel(), p_fut.ravel(),
                              groups=y_fut.ravel().astype(int))
    conformal["fitted_on"] = "calibration"
    # Per-class guarantee: one quantile per label, so attack windows get their
    # own 1-alpha coverage instead of borrowing it from the benign ones.
    lcc = LabelConditionalConformal(alpha=args.conformal_alpha)
    lcc.fit(P[CALIB]["future"].numpy().ravel().astype(int), cal_fut.ravel())
    conformal["label_conditional"] = lcc.evaluate(y_fut.ravel().astype(int), p_fut.ravel())

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
    print()
    print(f"  EARLY WARNING over {lead['episodes']} onset episodes:")
    if lead.get("detected"):
        print(f"     detection rate {lead['detection_rate']:.3f}  "
              f"median lead {lead['median_lead_seconds']:.1f}s  "
              f"p10 {lead['p10_lead_seconds']:.1f}s  p90 {lead['p90_lead_seconds']:.1f}s")
        print(f"     recall at lead: {lead.get('recall_at_lead')}")
    else:
        print(f"     {lead.get('note', 'no episodes')}")
    print(f"  CONFORMAL  target {conformal['target_coverage']:.2f}  "
          f"empirical {conformal['empirical_coverage']:.4f}  "
          f"median width {conformal['median_width']:.4f}")
    for g, rec in sorted(conformal.get("coverage_by_group", {}).items()):
        print(f"     interval coverage on {'attack' if g == '1' else 'benign'} windows: "
              f"{rec['empirical_coverage']:.4f}  (n={rec['n']})")
    _lc = conformal["label_conditional"]
    print(f"  LABEL-CONDITIONAL SETS  worst class {_lc['worst_class']} coverage "
          f"{_lc['worst_class_coverage']:.4f}  mean set size {_lc['mean_set_size']:.3f}  "
          f"ambiguous {_lc['ambiguous_rate']:.3f}")
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
        problems.append(f"test split has {ci.get('n_groups')} capture group(s): no usable confidence interval")
    if coverage.get("unresolved_rate", 0.0) > 0.05:
        problems.append(
            f"{coverage['unresolved_rate']:.1%} of the corpus could not be mapped to a "
            f"label ontology and was dropped: the metric describes the mapped subset only")
    if lead.get("episodes", 0) and not lead.get("detected"):
        problems.append("no attack onset was warned about before it happened: "
                        "lead time is undefined, so there is no early warning to report")
    _cov_gap = abs(conformal["empirical_coverage"] - conformal["target_coverage"])
    if _cov_gap > 0.05:
        problems.append(
            f"conformal coverage {conformal['empirical_coverage']:.3f} misses its "
            f"{conformal['target_coverage']:.2f} target by {_cov_gap:.3f}: the "
            f"calibration split does not represent test")
    # Marginal coverage can pass while the attack windows are badly under-
    # covered; the benign majority sets the average. Gate on the worst label.
    _worst = conformal.get("worst_group_coverage", float("nan"))
    if np.isfinite(_worst) and conformal["target_coverage"] - _worst > 0.05:
        problems.append(
            f"conformal interval covers only {_worst:.3f} of "
            f"{'attack' if conformal['worst_group'] == '1' else 'benign'} windows against a "
            f"{conformal['target_coverage']:.2f} target: the marginal figure hides it")
    _lcw = conformal["label_conditional"]["worst_class_coverage"]
    if np.isfinite(_lcw) and conformal["target_coverage"] - _lcw > 0.05:
        problems.append(
            f"label-conditional coverage for class {conformal['label_conditional']['worst_class']} "
            f"is {_lcw:.3f} against {conformal['target_coverage']:.2f}: that class's "
            f"calibration windows do not represent its test windows")
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
        "early_warning": lead,
        "conformal": conformal,
        "label_coverage": coverage,
        "split_strategy": assign.strategy,
        "attack_role": args.attack_role,
        "training": {
            "pos_weight": float(_pw),
            "selection_metric": "mean(val nowcast ROC-AUC, val forecast ROC-AUC)",
            "selected_epoch": int(best_ep),
            "best_validation_auc": float(best) if np.isfinite(best) else None,
            "patience": int(args.patience),
            "input_normalizer_fitted": bool(model.normalizer_fitted),
        },
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
