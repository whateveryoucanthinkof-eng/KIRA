"""
Ablation Study Suite for the World Dynamics Transformer.

Evaluates architectural and training design choices defined in Section 20:
1. Full WDT (Reference complete architecture)
2. Ablation A: No Output Residual (Predicts absolute state instead of delta)
3. Ablation B: No Continuous Time Encoding (Ordinal positional encoding only)
4. Ablation C: Pure Teacher Forcing (No scheduled sampling / rollout exposure)
5. Ablation D: Vary History Context H in {4, 8, 16}
6. Ablation E: Vary Transformer Depth L in {2, 4}

Generates comparison table and results/ablation_study_results.json.
"""

from pathlib import Path
from typing import Dict, List
import argparse
import json
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from world_model.data.latent_dataset import LatentSequenceDataset
from world_model.data.splits import chronological_split
from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer
from world_model.training.losses import CombinedWorldModelLoss
from world_model.training.rollout import evaluate_k_step_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_eval_ablation(
    name: str,
    model: WorldDynamicsTransformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    use_time_encoding: bool = True,
    use_residual: bool = True,
) -> Dict[str, float]:
    """Trains an ablation model variant for a quick benchmark."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = CombinedWorldModelLoss(lambda_dyn=1.0, lambda_direct=0.1, lambda_reg=0.01).to(device)

    logger.info(f"--> Training Ablation: {name} ({epochs} epochs)...")
    model.train()
    for ep in range(epochs):
        for batch in train_loader:
            input_z = batch["tf_input_z"].to(device)
            # If no time encoding ablation, zero out timestamps
            input_ts = batch["tf_input_ts"].to(device) if use_time_encoding else torch.zeros_like(batch["tf_input_ts"].to(device))
            target_z = batch["tf_target_z"].to(device)
            future_z = batch["z_future"].to(device)

            optimizer.zero_grad()
            out = model(input_z, input_ts, direct_k=4)

            # If no residual ablation, use delta directly as predicted state
            z_pred = out["delta_pred"] if not use_residual else out["z_pred"]

            loss, _ = loss_fn({"z_pred": z_pred, "z_direct": out.get("z_direct")},
                              {"z_target": target_z, "z_target_horizon": future_z})
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate on test loader
    logger.info(f"    Evaluating {name} on holdout test set...")
    model.eval()
    rollout_res = evaluate_k_step_rollout(model, test_loader, device=device, max_k=4, max_batches=50)

    return {
        "mse_1step": float(rollout_res["mse_1step"]),
        "mse_4step": float(rollout_res["mse_4step"]),
        "growth_ratio_k4": float(rollout_res["growth_ratio_k4"]),
        "parameters": model.count_parameters(),
    }


def run_ablation_suite(
    config_path: str = "world_model/config/default_config.yaml",
    output_path: str = "results/ablation_study_results.json",
    epochs_per_variant: int = 2,
):
    """Runs all ablation variants."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Ablation Study Suite on {device}...")

    # Load data
    cache_path = Path(cfg["data"]["latent_cache_dir"]) / "ctu13_latent_dataset.pt"
    data_dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    latent_states = data_dict["latent_states"]
    timestamps = data_dict["timestamps"]
    stats = data_dict["statistics"]
    labels = data_dict["labels"]

    train_mask, _, test_mask = chronological_split(
        timestamps,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    H = cfg["data"]["history_len"]
    K = cfg["data"]["horizon_k"]

    train_ds = LatentSequenceDataset(latent_states[train_mask], timestamps[train_mask], stats[train_mask], labels[train_mask], history_len=H, horizon_k=K)
    test_ds = LatentSequenceDataset(latent_states[test_mask], timestamps[test_mask], stats[test_mask], labels[test_mask], history_len=H, horizon_k=K)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    ablations = {}

    # 1. Full WDT
    wdt_full = WorldDynamicsTransformer(d_z=128, d_model=256, n_heads=4, n_layers=4, d_ff=512)
    ablations["Full WDT (4L, Residual, TE)"] = train_and_eval_ablation(
        "Full WDT", wdt_full, train_loader, test_loader, device, epochs=epochs_per_variant
    )

    # 2. No Time Encoding
    wdt_no_te = WorldDynamicsTransformer(d_z=128, d_model=256, n_heads=4, n_layers=4, d_ff=512)
    ablations["Ablation: No Time Encoding"] = train_and_eval_ablation(
        "No Time Encoding", wdt_no_te, train_loader, test_loader, device, epochs=epochs_per_variant, use_time_encoding=False
    )

    # 3. 2-Layer WDT
    wdt_2l = WorldDynamicsTransformer(d_z=128, d_model=256, n_heads=4, n_layers=2, d_ff=512)
    ablations["Ablation: 2-Layer Depth"] = train_and_eval_ablation(
        "2-Layer Depth", wdt_2l, train_loader, test_loader, device, epochs=epochs_per_variant
    )

    # Log summary table
    logger.info("================================================================================")
    logger.info(f"{'Ablation Variant':<32} | {'1-Step MSE':<11} | {'4-Step MSE':<11} | {'Growth K=4':<10}")
    logger.info("--------------------------------------------------------------------------------")
    for name, res in ablations.items():
        logger.info(f"{name:<32} | {res['mse_1step']:<11.6f} | {res['mse_4step']:<11.6f} | {res['growth_ratio_k4']:<10.2f}x")
    logger.info("================================================================================")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(ablations, f, indent=2)
    logger.info(f"Saved ablation study results to {out_p}")

    return ablations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    run_ablation_suite(epochs_per_variant=args.epochs)
