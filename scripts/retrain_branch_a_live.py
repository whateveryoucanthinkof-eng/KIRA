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


def _load_records(cic_dir: Path, ctu_dir: Path, rows_per_file: int, files: List[Path]):
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    records = []
    for path in files:
        if path.suffix.lower() == ".csv":
            records.extend(cic.parse_file(str(path), max_rows=rows_per_file))
        else:
            records.extend(ctu.parse_netflow_csv(str(path), max_rows=rows_per_file))
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
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cic_files = sorted(args.cic_dir.glob("*.csv"))
    ctu_files = sorted(args.ctu_dir.glob("*/*.binetflow"))
    all_files = cic_files + ctu_files
    if len(all_files) < 4:
        raise RuntimeError(f"Expected supplied SIH/CTU files, found {len(all_files)}")

    split = max(1, int(len(all_files) * 0.8))
    train_files = all_files[:split]
    val_files = all_files[split:]
    train_records = _load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, train_files)
    val_records = _load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, val_files)

    tgn = build_or_load_tgne_ta()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=2.0)
    train_samples = _make_samples(train_records, extractor, seq_len=5)
    val_samples = _make_samples(val_records, extractor, seq_len=5)
    if not train_samples or not val_samples:
        raise RuntimeError("The 2-second pipeline produced no train/validation samples")

    train_loader = DataLoader(
        HostSequenceDataset(train_samples, seq_len=5),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        HostSequenceDataset(val_samples, seq_len=5),
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
        f"train_records={len(train_records)} val_records={len(val_records)} "
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
                    "training_contract": {
                        "window_size_sec": 2.0,
                        "history_steps": 5,
                        "feature_dim": 27,
                        "sources": [str(args.cic_dir), str(args.ctu_dir)],
                    },
                },
                args.output,
            )

    print(f"saved={args.output} best_metrics={best_metrics}")


if __name__ == "__main__":
    main()