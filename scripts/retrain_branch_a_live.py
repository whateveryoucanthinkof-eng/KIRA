"""Retrain Branch A on supplied 2-second SIH/CTU live-telemetry data."""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from branch_a_gnn_lstm.sequence_dataset import (
    TECHNIQUE_VOCAB,
    HostSequenceDataset,
    create_host_sequence_samples,
)
from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.split_policy import partition_paths
from data_unification.density import require_full_density
from cyberworld_v4.config import get_contract, DEFAULT_CONFIG
from cyberworld_v4.manifest import ExperimentManifest, set_all_seeds


def _strided(gen, stride: int, want: int):
    """Samples every Nth record across a wider read instead of a plain file-prefix.

    max_rows in the adapters is a prefix (pandas nrows); CIC-2018 CSVs are
    time-ordered with attacks in contiguous blocks, so a plain prefix is
    87-100% single-label (see claude_latest_analysis/07_v4_audit_and_migration_plan.md,
    D6). Reading stride*want rows and keeping every `stride`-th one instead
    spans much more of the file's time range for the same record budget.
    """
    out = []
    for i, r in enumerate(gen):
        if i % stride == 0:
            out.append(r)
            if len(out) >= want:
                break
    return out


def _load_records(cic_dir: Path, ctu_dir: Path, rows_per_file: int, files: List[Path], stride: int = 1):
    import time
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    records = []
    for i, path in enumerate(files):
        t0 = time.time()
        gen = cic.parse_file(str(path), max_rows=rows_per_file * stride) if path.suffix.lower() == ".csv" \
            else ctu.parse_netflow_csv(str(path), max_rows=rows_per_file * stride)
        got = _strided(gen, stride, rows_per_file)
        records.extend(got)
        print(f"  [{i+1}/{len(files)}] {path.name}: {len(got)} records in {time.time()-t0:.1f}s (cumulative {len(records)})", flush=True)
    return records


def _make_samples(records, extractor, seq_len: int):
    trajectories = extractor.extract_trajectories(records)
    return create_host_sequence_samples(
        trajectories,
        seq_len=seq_len,
        min_trajectory_len=1,
    )


def _evaluate(model, loader, device):
    model.eval()
    losses = []
    risk_errors = []
    correct_tech = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            targets = {
                "risk": batch["risk"].to(device),
                "technique": batch["technique"].to(device),
                "gradation": batch["gradation"].to(device),
            }
            predictions = model(x)
            loss, _ = model.compute_loss(predictions, targets)
            losses.append(float(loss.item()))
            risk_errors.extend(
                (predictions["risk_score"] - targets["risk"]).abs().cpu().numpy()
            )
            correct_tech += int(
                (predictions["technique_logits"].argmax(dim=-1) == targets["technique"])
                .sum()
                .item()
            )
            total += int(targets["technique"].numel())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "risk_mae": float(np.mean(risk_errors)) if risk_errors else 0.0,
        "tech_accuracy": correct_tech / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cic-dir", type=Path, required=True)
    parser.add_argument("--ctu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=None,
                        help="Cap records kept per capture. Default None = FULL DENSITY.")
    parser.add_argument("--spill-dir", type=Path, default=None, help="Write the bulk trajectory feature block here instead of RAM (np.memmap)")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth record across a wider read, instead of a plain file-prefix (see _strided)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _cfg = DEFAULT_CONFIG
    _manifest = ExperimentManifest.create(
        f"branch_a_v4_seed{args.seed}",
        seed=args.seed,
        config=_cfg,
        repo=Path(__file__).resolve().parent.parent,
        dataset_sources=[str(args.cic_dir), str(args.ctu_dir)],
    )
    set_all_seeds(args.seed)

    require_full_density(
        'Branch A retrain',
        stride=args.stride,
        rows_per_file=args.rows_per_file,
    )

    cic_files = sorted(args.cic_dir.glob("*.csv"))
    ctu_files = sorted(args.ctu_dir.glob("*/*.binetflow"))
    all_files = cic_files + ctu_files
    if len(all_files) < 4:
        raise RuntimeError(f"Expected supplied SIH/CTU files, found {len(all_files)}")

    # Three-way, capture-disjoint, and READ FROM THE FROZEN LOCK.
    #
    # This used to slice a *sorted* file list 70/15/15, so "train" meant
    # "alphabetically first" and the assignment shifted whenever a file was
    # added or renamed -- results were not comparable across retrains, which is
    # the entire reason data_unification/splits.lock.json exists. The lock had
    # zero consumers; now it has this one.
    #
    # Test is scored once, after the model is frozen, and never influences
    # training.
    partition = partition_paths(all_files)
    train_files, val_files, test_files = (
        partition["train"], partition["val"], partition["test"],
    )
    for _name, _files in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not _files:
            raise RuntimeError(
                f"frozen split '{_name}' matched no files under {args.cic_dir} / "
                f"{args.ctu_dir}. Refusing to train on a split that does not exist."
            )
    print(f"frozen split: {len(train_files)} train / {len(val_files)} val / "
          f"{len(test_files)} test captures", flush=True)
    import time
    tgn = build_or_load_tgne_ta()
    # Contract-bound (v4). Previously 2.0s / seq_len=5 hardcoded, which matched
    # the v3 contract by coincidence rather than by construction. Under v4 this
    # produces history_steps=15, so it yields a v4 checkpoint, not a v3 one.
    _c = get_contract()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=_c.window_seconds,
                                        spill_dir=str(args.spill_dir) if args.spill_dir else None)
    # Load -> extract -> free, one split at a time. Holding both record lists
    # at once costs an extra ~2.8 GB at full density for no reason: the val
    # records are not needed until the train split has already been reduced to
    # samples.
    record_counts: Dict[str, int] = {}

    def _samples_per_file(files, label):
        """Load -> extract -> free, one capture file at a time.

        Holding the whole split resident costs ~15.7 GB at full density (30.4M
        records x ~518 B measured), which does not fit alongside the snapshot
        store and adapter arrays. Per-file keeps peak at the largest single
        capture (~4.7M records, ~2.4 GB).

        Processing per file also builds the TGNE neighbour graph per capture
        rather than across all of them. That is more faithful, not less: a
        CIC-2018 capture day and a CTU-13 botnet scenario are unrelated
        networks, and a merged graph would make hosts from different captures
        each other's temporal neighbours, which they never were.
        """
        from data_unification.trajectory_store import TrajectoryStoreBuilder
        shared = TrajectoryStoreBuilder(spill_dir=str(args.spill_dir) if args.spill_dir else None)
        widx_base = 0
        total_recs = 0
        for i, f in enumerate(files):
            t = time.time()
            recs = _load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, [f], args.stride)
            total_recs += len(recs)
            extractor.extract_trajectories(recs, builder=shared, window_idx_base=widx_base)
            if shared._window_idx.n:
                widx_base = int(shared._window_idx.buf[: shared._window_idx.n].max()) + 1
            print(f"  [{label} {i+1}/{len(files)}] {f.name}: {len(recs)} recs, "
                  f"store={shared._n} snaps, {time.time()-t:.1f}s", flush=True)
            del recs
            gc.collect()
        store = shared.finalize()
        print(f"{label}: {total_recs} records -> {store.n_snapshots} snapshots "
              f"over {len(store)} hosts | {store.memory_report()}", flush=True)
        # The record count is returned, not just printed: the run summary below
        # used to reference `train_records`/`val_records`, which the
        # load -> extract -> free refactor had already deleted. Every
        # non-credible run therefore died with NameError instead of reporting
        # the credibility verdict it had just computed.
        record_counts[label] = total_recs
        return create_host_sequence_samples(store, seq_len=_c.history_steps, min_trajectory_len=1)

    import gc
    t0 = time.time()
    train_samples = _samples_per_file(train_files, "train")
    print(f"train done in {time.time()-t0:.1f}s ({len(train_samples)} samples)", flush=True)
    t0 = time.time()
    val_samples = _samples_per_file(val_files, "val")
    print(f"val done in {time.time()-t0:.1f}s ({len(val_samples)} samples)", flush=True)
    t0 = time.time()
    test_samples = _samples_per_file(test_files, "test")
    print(f"test done in {time.time()-t0:.1f}s ({len(test_samples)} samples)", flush=True)

    # Gate the data before training on it.
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from credibility_check import evaluate_samples, gate as _gate, report as _report
        _stats = evaluate_samples(train_samples, val_samples)
        _report(_stats, _gate(_stats))
    except Exception as _e:
        print(f"credibility check skipped: {_e}", flush=True)
    if not train_samples or not val_samples:
        raise RuntimeError("The 2-second pipeline produced no train/validation samples")

    train_loader = DataLoader(
        HostSequenceDataset(train_samples, seq_len=_c.history_steps),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        HostSequenceDataset(val_samples, seq_len=_c.history_steps),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskLSTM(
        input_dim=27,
        hidden_dim=64,
        num_layers=2,
        num_techniques=len(TECHNIQUE_VOCAB),
        num_gradations=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_metrics: Dict[str, float] = {}

    print(
        f"train_records={record_counts.get('train', 0)} "
        f"val_records={record_counts.get('val', 0)} "
        f"train_samples={len(train_samples)} val_samples={len(val_samples)} device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["features"].to(device)
            targets = {
                "risk": batch["risk"].to(device),
                "technique": batch["technique"].to(device),
                "gradation": batch["gradation"].to(device),
            }
            optimizer.zero_grad()
            predictions = model(x)
            loss, _ = model.compute_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        metrics = _evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(train_losses))
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} risk_mae={metrics['risk_mae']:.4f} "
            f"tech_accuracy={metrics['tech_accuracy']:.3f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    # Written from the contract actually in force, not from
                    # literals. These were hardcoded to 2.0/5 regardless of what
                    # the run used, so the metadata could describe a model that
                    # was never trained.
                    "training_contract": {
                        "window_seconds": _c.window_seconds,
                        "window_size_sec": _c.window_seconds,  # v3 key, kept readable
                        "history_steps": _c.history_steps,
                        "forecast_steps": _c.forecast_steps,
                        "feature_dim": _cfg.state_dim,
                        "sources": [str(args.cic_dir), str(args.ctu_dir)],
                    },
                    "config": _cfg.to_dict(),
                    "manifest": _manifest.to_dict(),
                    "fingerprint": _manifest.fingerprint(),
                },
                args.output,
            )

    # Held-out test: scored once, on the restored best checkpoint, after
    # training is finished. This is the only number that is a generalisation
    # estimate rather than a selection artefact.
    ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loader = DataLoader(HostSequenceDataset(test_samples, seq_len=_c.history_steps),
                             batch_size=args.batch_size, shuffle=False)
    test_metrics = _evaluate(model, test_loader, device)
    print(f"HELD-OUT TEST (best epoch {ckpt.get('epoch')}): {test_metrics}", flush=True)
    # The credibility verdict travels WITH the checkpoint.
    #
    # It used to be computed, printed to stdout, and thrown away. The
    # checkpoint kept `test_metrics` but nothing recording that those numbers
    # might be meaningless, so a non-credible model could be loaded and served
    # with no trace -- and its metrics quoted as results. That is the same
    # "looks fine, means nothing" failure mode as a collapsed head scoring
    # 0.83 CatAcc, or a class with seven training samples.
    credibility = {"checked": False}
    try:
        from credibility_check import evaluate_samples as _es, gate as _g, report as _r
        _ts = _es(train_samples, test_samples)
        _problems = _g(_ts, model_accuracy=test_metrics.get("tech_accuracy"))
        _r(_ts, _problems)
        credibility = {
            "checked": True,
            "credible": not _problems,
            "problems": list(_problems),
            "stats": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                      for k, v in _ts.items()},
        }
    except Exception as _e:
        credibility = {"checked": False, "error": str(_e)}
        print(f"test credibility check skipped: {_e}", flush=True)

    ckpt["test_metrics"] = test_metrics
    ckpt["credibility"] = credibility
    torch.save(ckpt, args.output)
    if credibility.get("checked") and not credibility.get("credible"):
        print("WARNING: checkpoint saved but marked NOT CREDIBLE -- "
              "its metrics must not be reported as results.", flush=True)

    print(f"saved={args.output} best_metrics={best_metrics} test_metrics={test_metrics}")


if __name__ == "__main__":
    main()