"""
Main Training Pipeline for the World Dynamics Transformer (WDT).

Executes Stage 1 & Stage 2 world-model training:
1. Loads cached CTU-13 latent state sequence z_t in R^{128}.
2. Performs chronological temporal split (60% train / 15% val / 25% test).
3. Computes the persistence baseline (ẑ_{t+1} = z_t) as the rigorous benchmark.
4. Trains WDT with AdamW, Cosine Annealing, AMP, and gradient clipping.
5. Verifies success criterion: 1-step prediction MSE < persistence baseline MSE.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import csv
import logging
import math
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from world_model.data.feature_schema import D_Z
from world_model.data.latent_dataset import LatentSequenceDataset
from world_model.data.splits import chronological_split
from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer
from world_model.training.checkpointing import load_checkpoint, save_checkpoint
from world_model.training.losses import CombinedWorldModelLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_persistence_baseline(dataset: LatentSequenceDataset) -> float:
    """
    Computes the persistence baseline MSE: predicting ẑ_{t+1} = z_t.
    Vectorized over the dataset's latent_states tensor.
    """
    states = dataset.latent_states  # [T, d_z]
    if len(states) < 2:
        return 0.0
    # True step transition diff: z_{t+1} - z_t
    diff = states[1:] - states[:-1]
    persistence_mse = (diff ** 2).mean().item()
    return persistence_mse


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedWorldModelLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    direct_k: Optional[int] = 4,
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    loss_accum = {}
    num_batches = 0
    use_cuda = (device.type == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for batch in loader:
        input_z = batch["tf_input_z"].to(device)    # [B, H+K-1, d_z]
        input_ts = batch["tf_input_ts"].to(device)  # [B, H+K-1]
        target_z = batch["tf_target_z"].to(device)  # [B, H+K-1, d_z]
        future_z = batch["z_future"].to(device)     # [B, K, d_z]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            output = model(input_z, input_ts, direct_k=direct_k)
            predictions = {
                "z_pred": output["z_pred"],
                "z_direct": output.get("z_direct"),
            }
            targets = {
                "z_target": target_z,
                "z_target_horizon": future_z,
            }
            loss, loss_dict = loss_fn(predictions, targets)

        if use_bf16:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v
        num_batches += 1

    metrics = {k: v / max(1, num_batches) for k, v in loss_accum.items()}
    metrics["loss"] = total_loss / max(1, num_batches)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedWorldModelLoss,
    device: torch.device,
    direct_k: Optional[int] = 4,
) -> Tuple[float, Dict[str, float]]:
    """Evaluates the model on validation or test split."""
    model.eval()
    total_loss = 0.0
    loss_accum = {}
    total_sq_err = 0.0
    total_elements = 0
    num_batches = 0

    use_cuda = (device.type == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for batch in loader:
        input_z = batch["tf_input_z"].to(device)
        input_ts = batch["tf_input_ts"].to(device)
        target_z = batch["tf_target_z"].to(device)
        future_z = batch["z_future"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            output = model(input_z, input_ts, direct_k=direct_k)
            predictions = {
                "z_pred": output["z_pred"],
                "z_direct": output.get("z_direct"),
            }
            targets = {
                "z_target": target_z,
                "z_target_horizon": future_z,
            }
            loss, loss_dict = loss_fn(predictions, targets)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        # Compute pure 1-step next state MSE
        diff = output["z_pred"] - target_z
        total_sq_err += (diff ** 2).sum().item()
        total_elements += diff.numel()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    metrics = {k: v / max(1, num_batches) for k, v in loss_accum.items()}
    metrics["mse_1step"] = total_sq_err / max(1, total_elements)
    return avg_loss, metrics


def train_world_model(config_path: str = "world_model/config/default_config.yaml", max_epochs_override: Optional[int] = None):
    """Full training pipeline execution."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load cached latent states
    cache_path = Path(cfg["data"]["latent_cache_dir"]) / "ctu13_latent_dataset.pt"
    if not cache_path.exists():
        logger.info(f"Latent cache {cache_path} not found. Generating now...")
        from world_model.scripts.generate_latent_dataset import generate_and_cache_ctu13_latents
        generate_and_cache_ctu13_latents(cfg["data"]["parquet_path"], cfg["data"]["latent_cache_dir"])

    logger.info(f"Loading latent dataset from {cache_path}...")
    data_dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    latent_states = data_dict["latent_states"]      # [N, 128]
    timestamps = data_dict["timestamps"]            # [N]
    stats = data_dict["statistics"]                 # [N, 14]
    labels = data_dict["labels"]                    # [N]
    mal_fractions = data_dict.get("malicious_fractions", torch.zeros_like(timestamps))

    N_total = len(latent_states)
    logger.info(f"Total time windows available: {N_total:,}")

    # 2. Chronological temporal split
    train_mask, val_mask, test_mask = chronological_split(
        timestamps,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    logger.info(f"Split sizes: Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")

    H = cfg["data"]["history_len"]
    K = cfg["data"]["horizon_k"]

    # Datasets
    train_ds = LatentSequenceDataset(
        latent_states[train_mask],
        timestamps[train_mask],
        stats[train_mask],
        labels[train_mask],
        mal_fractions[train_mask],
        history_len=H,
        horizon_k=K,
    )
    val_ds = LatentSequenceDataset(
        latent_states[val_mask],
        timestamps[val_mask],
        stats[val_mask],
        labels[val_mask],
        mal_fractions[val_mask],
        history_len=H,
        horizon_k=K,
    )
    test_ds = LatentSequenceDataset(
        latent_states[test_mask],
        timestamps[test_mask],
        stats[test_mask],
        labels[test_mask],
        mal_fractions[test_mask],
        history_len=H,
        horizon_k=K,
    )

    logger.info(f"Sequence samples: Train={len(train_ds):,} | Val={len(val_ds):,} | Test={len(test_ds):,}")

    # 3. Compute Persistence Baseline on Validation Set
    val_persistence_mse = compute_persistence_baseline(val_ds)
    logger.info(f"==> Validation Persistence Baseline MSE (ẑ_{{t+1}} = z_t): {val_persistence_mse:.6f}")

    # DataLoaders
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. Initialize World Dynamics Transformer
    m_cfg = cfg["model"]
    model = WorldDynamicsTransformer(
        d_z=m_cfg["d_z"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_layers=m_cfg["n_layers"],
        d_ff=m_cfg["d_ff"],
        d_time=m_cfg["d_time"],
        max_len=m_cfg["max_len"],
        dropout=m_cfg["dropout"],
        default_horizon=m_cfg["default_horizon"],
    ).to(device)

    param_count = model.count_parameters()
    logger.info(f"Initialized WDT with {param_count:,} trainable parameters.")

    # 5. Loss & Optimizer
    l_cfg = cfg["loss_weights"]
    loss_fn = CombinedWorldModelLoss(
        lambda_dyn=l_cfg["lambda_dyn"],
        lambda_direct=l_cfg["lambda_direct"],
        lambda_reg=l_cfg["lambda_reg"],
        gamma_dyn=l_cfg["gamma_dyn"],
    ).to(device)

    t_cfg = cfg["training"]
    epochs = max_epochs_override or t_cfg["epochs"]
    lr = float(t_cfg["lr"])
    min_lr = float(t_cfg.get("min_lr", 1e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Checkpoint dir
    save_dir = Path(t_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = save_dir / "training_metrics.csv"

    # CSV Logger
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_loss_dyn", "val_loss", "val_loss_dyn",
            "val_mse_1step", "persistence_mse", "mse_improvement_pct", "lr", "epoch_time_s"
        ])

    best_val_loss = float("inf")
    patience = t_cfg.get("patience", 8)
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            grad_clip=t_cfg.get("grad_clip", 1.0),
            direct_k=K,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            direct_k=K,
        )
        scheduler.step()
        epoch_time = time.time() - t0

        val_mse = val_metrics["mse_1step"]
        pct_improvement = ((val_persistence_mse - val_mse) / val_persistence_mse) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val 1-step MSE: {val_mse:.6f} vs Base: {val_persistence_mse:.6f} "
            f"({pct_improvement:+.2f}%) | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log to CSV
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.5f}",
                f"{train_metrics.get('loss_dyn', 0.0):.5f}",
                f"{val_loss:.5f}",
                f"{val_metrics.get('loss_dyn', 0.0):.5f}",
                f"{val_mse:.6f}",
                f"{val_persistence_mse:.6f}",
                f"{pct_improvement:.2f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.1f}",
            ])

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "val_mse_1step": val_mse,
                "persistence_mse": val_persistence_mse,
                "config": cfg,
            },
            checkpoint_dir=save_dir,
            is_best=is_best,
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    # 6. Final Test Evaluation
    logger.info("Running final evaluation on Holdout Test Split with best model...")
    load_checkpoint(save_dir / "wdt_best.pt", model=model, device=device)
    test_loss, test_metrics = evaluate(model=model, loader=test_loader, loss_fn=loss_fn, device=device, direct_k=K)
    test_persistence_mse = compute_persistence_baseline(test_ds)
    test_pct = ((test_persistence_mse - test_metrics["mse_1step"]) / test_persistence_mse) * 100.0

    logger.info("==========================================================")
    logger.info("FINAL HOLDOUT TEST EVALUATION RESULTS:")
    logger.info(f"Test Loss:            {test_loss:.4f}")
    logger.info(f"Test 1-step MSE:      {test_metrics['mse_1step']:.6f}")
    logger.info(f"Persistence Baseline: {test_persistence_mse:.6f}")
    logger.info(f"Improvement over Base:{test_pct:+.2f}%")
    logger.info("==========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Dynamics Transformer")
    parser.add_argument("--config", type=str, default="world_model/config/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_world_model(args.config, max_epochs_override=args.epochs)
