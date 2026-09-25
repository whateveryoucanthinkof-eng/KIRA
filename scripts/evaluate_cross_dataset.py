"""Score trained models on CIC-IDS-2017 after training on CIC-IDS-2018.

    python scripts/evaluate_cross_dataset.py \\
        --cic2017-dir  <CIC-2017 TrafficLabelling CSVs> \\
        --tgne         <encoder .pth> \\
        --branch-a     <branch_a_lstm.pt> \\
        [--branch-b    <host_wdt.pt> --deepop <cwa_forecast_decoder.pt>] \\
        --out          results/cross_year/<name>

Writes <out>.json and <out>.md.

This is the "score once" step of the cross_year protocol
(data_unification/split_policy.py). Nothing here trains, tunes or selects:
thresholds, temperatures and conformal widths are the ones the checkpoint
carries, fitted on CIC-2018 validation days. Re-fitting any of them on 2017
would turn the test set into a tuning set.

Reported for Branch A:
  * risk: ROC-AUC, Brier (and the base-rate Brier), and the checkpoint's own
    alert threshold scored on 2017 without re-fitting;
  * technique: macro-F1 over classes training saw, macro-F1 over all classes,
    and each class training never saw (PortScan, Heartbleed) with what the
    model predicted instead -- cyberworld_v4/cross_dataset.py;
  * the majority-class baseline, so an accuracy can be read against it.
For Branch B: MSE against persistence (skill). For DeepOP: token metrics the
same seen/unseen way, against the persistence baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "bita")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_test_store(extractor, cic2017_dir: Path, window_seconds: float, spill_dir=None):
    """Every CIC-2017 capture, extracted with `extractor`'s encoder."""
    import gc
    from data_unification.trajectory_store import TrajectoryStoreBuilder
    from data_unification.training_sources import discover_captures, read_capture

    caps = discover_captures(scheme="cross_year", cic2017_dir=cic2017_dir)["test"]
    if not caps:
        raise RuntimeError(f"no CIC-2017 captures found under {cic2017_dir}")
    b = TrajectoryStoreBuilder(spill_dir=str(spill_dir) if spill_dir else None)
    wbase = 0
    for cap in caps:
        recs = read_capture(cap, window_seconds=window_seconds)
        extractor.extract_trajectories(recs, builder=b, window_idx_base=wbase)
        if b._window_idx.n:
            wbase = int(b._window_idx.buf[: b._window_idx.n].max()) + 1
        print(f"  [CIC-2017] {cap.name}: {len(recs)} records, store {b._n} snapshots", flush=True)
        del recs
        gc.collect()
    return b.finalize()


def evaluate_branch_a(ckpt_path: Path, store, device, batch_size=512, num_workers=0) -> dict:
    import importlib.util
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    from branch_a_gnn_lstm.sequence_dataset import TECH_TO_IDX, TECHNIQUE_VOCAB, LazyHostSequenceDataset
    from cyberworld_v4.config import get_contract
    from cyberworld_v4.cross_dataset import unseen_class_report

    spec = importlib.util.spec_from_file_location("_bra", REPO / "scripts" / "retrain_branch_a_live.py")
    bra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bra)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiTaskLSTM.from_checkpoint(ck, device=device)
    tc = ck.get("training_contract") or {}

    # The test store must carry the same risk target the model was trained on.
    risk_positive_above = 0.0
    if tc.get("risk_target") == "hazard":
        tau = tc.get("hazard_tau_seconds") or get_contract().forecast_steps * get_contract().window_seconds
        store.use_hazard_target(float(tau))
        risk_positive_above = math.exp(-1.0) - 1e-6

    ds = LazyHostSequenceDataset(store, seq_len=get_contract().history_steps, min_trajectory_len=1)
    m = bra._evaluate(model, DataLoader(ds, batch_size=batch_size, num_workers=num_workers),
                      device, num_techniques=len(TECHNIQUE_VOCAB),
                      risk_positive_above=risk_positive_above)

    train_counts = ck.get("train_technique_counts")
    if train_counts is None:
        print("WARNING: checkpoint predates train_technique_counts; every class is treated as "
              "seen, so unseen classes cannot be separated out", flush=True)
        train_counts = np.ones(len(TECHNIQUE_VOCAB), dtype=np.int64)
    unseen = unseen_class_report(train_counts, m["tech_confusion"],
                                 {v: k for k, v in TECH_TO_IDX.items()})

    out = {k: v for k, v in bra.slim(m, drop_per_class=True).items()
           if isinstance(v, (int, float, str))}
    out["n_samples"] = len(ds)
    out["unseen_class_report"] = unseen
    op = ck.get("operating_point") or {}
    if op.get("fitted"):
        curve = bra._pr_curve_from_histograms(m["risk_pos_hist"], m["risk_neg_hist"])
        b = int(round(op["alert_threshold"] * (bra.RISK_BINS - 1)))
        out["operating_point_on_2017"] = {"threshold_fitted_on_2018": op["alert_threshold"],
                                          **bra._point(curve, b)}
    out["split_scheme_trained_under"] = ck.get("split_scheme") or tc.get("split_scheme")
    return out


def evaluate_future(bb_path: Path, dp_path, store, device) -> dict:
    from branch_b_world_model.infiltration_head import InfiltrationRiskHead
    from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
    from cyberworld_v4.config import get_contract
    from cyberworld_v4.cross_dataset import branch_b_on_store, deepop_on_store
    from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
    from deepop_decoder.joint_vocab import get_joint_vocab

    c = get_contract()
    ck = torch.load(bb_path, map_location=device, weights_only=False)
    d = int(ck.get("d_state") or ck["wdt_state_dict"]["in_proj.weight"].shape[1])
    wdt = HostWorldDynamicsTransformer(d_latent=d, d_model=64, n_heads=4, n_layers=3).to(device)
    wdt.load_state_dict(ck["wdt_state_dict"])
    risk = InfiltrationRiskHead(d_latent=d, hidden_dim=32).to(device)
    risk.load_state_dict(ck["risk_head_state_dict"])
    out = {"branch_b": branch_b_on_store(wdt, risk, store, device, c.history_steps, c.forecast_steps)}
    if dp_path:
        dck = torch.load(dp_path, map_location=device, weights_only=False)
        vocab = get_joint_vocab(network_observable_only=True)
        counts = dck.get("train_token_counts")
        if counts is None:
            print("WARNING: DeepOP checkpoint predates train_token_counts; every token treated "
                  "as seen", flush=True)
            counts = np.ones(vocab.vocab_size, dtype=np.int64)
        dec = DeepOPForecastDecoder.from_checkpoint(dck, device=device)
        out["deepop"] = deepop_on_store(dec, wdt, store, vocab, device, counts,
                                        c.history_steps, c.forecast_steps)
    return out


def to_markdown(name: str, res: dict) -> str:
    a = res.get("branch_a", {})
    u = a.get("unseen_class_report", {})
    lines = [f"# Cross-dataset evaluation: {name}", "",
             "Trained and tuned on CIC-IDS-2018; scored once on CIC-IDS-2017.", "",
             f"Encoder: `{res.get('encoder')}`", ""]
    if a:
        lines += ["## Branch A", "", "| metric | value |", "|---|---|"]
        for k in ("n_samples", "risk_auc", "risk_brier", "risk_brier_baseline", "risk_base_rate",
                  "tech_accuracy", "tech_majority_baseline", "tech_macro_f1"):
            if k in a:
                v = a[k]
                lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        if u:
            lines += [f"| macro-F1, classes seen in training | {u['macro_f1_seen']:.4f} |",
                      f"| test share in unseen classes | {u['test_fraction_in_unseen_classes']:.4f} |"]
        op = a.get("operating_point_on_2017")
        if op:
            lines += [f"| alert threshold (fitted on 2018) | {op['threshold_fitted_on_2018']:.4f} |",
                      f"| precision / recall at that threshold | {op['precision']:.4f} / {op['recall']:.4f} |"]
        if u and u.get("unseen_classes"):
            lines += ["", "Classes CIC-2017 has and training never saw:", ""]
            for c in u["unseen_classes"]:
                inst = ", ".join(f"{p['class']} {100 * p['fraction']:.1f}%" for p in c["predicted_as"])
                lines.append(f"- **{c['class']}**: {c['test_support']} samples, predicted as {inst}")
    if "branch_b" in res:
        b = res["branch_b"]
        if b.get("n"):
            lines += ["", "## Branch B", "",
                      f"MSE {b['mse_model']:.6f} vs persistence {b['mse_persistence']:.6f} "
                      f"(skill {b['skill']:+.4f})"]
    if "deepop" in res and res["deepop"].get("n"):
        d = res["deepop"]
        lines += ["", "## DeepOP", "",
                  f"token macro-F1 seen {d['macro_f1_seen']:.4f}, all {d['macro_f1_all']:.4f}; "
                  f"accuracy {d['accuracy_all']:.4f} vs persistence {d['acc_persistence']:.4f}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cic2017-dir", type=Path, required=True)
    ap.add_argument("--tgne", type=Path, required=True)
    ap.add_argument("--branch-a", type=Path, required=True)
    ap.add_argument("--branch-b", type=Path, default=None)
    ap.add_argument("--deepop", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="output path prefix (.json and .md)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--spill-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()

    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
    from cyberworld_v4.config import get_contract
    from data_unification.multi_dataset_stream import HostTrajectoryExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgn = build_or_load_tgne_ta(checkpoint_path=str(a.tgne))
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds,
                                        spill_dir=str(a.spill_dir) if a.spill_dir else None)
    store = build_test_store(extractor, a.cic2017_dir, get_contract().window_seconds, a.spill_dir)

    res = {"encoder": str(a.tgne), "test": "CIC-IDS-2017 (all captures)",
           "branch_a": evaluate_branch_a(a.branch_a, store, device, a.batch_size, a.num_workers)}
    if a.branch_b:
        res.update(evaluate_future(a.branch_b, a.deepop, store, device))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(res, indent=2, default=str))
    Path(str(a.out) + ".md").write_text(to_markdown(a.name or a.out.name, res))
    print(to_markdown(a.name or a.out.name, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
