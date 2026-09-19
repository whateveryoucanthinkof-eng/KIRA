"""
Comprehensive Baseline Comparison Benchmark for Latent Dynamics Forecasting.

Compares:
1. Persistence Baseline: ẑ_{t+1} = z_t (Identity transition baseline)
2. Linear Autoregressive Baseline: Ridge/Linear projection from history
3. LSTM Baseline: 2-layer Recurrent Neural Network
4. World Dynamics Transformer (WDT): Causal decoder-only Transformer

Evaluates all models on the identical chronological holdout test split (23,084 samples).
Computes:
- 1-step and 4-step Rollout MSE
- Cosine Similarity
- % Improvement over Persistence Baseline
- Model Parameters & Inference Latency
Outputs comparative markdown and JSON report to results/baseline_comparison.json.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from world_model.baselines.logistic_regression import LinearDynamicsBaseline
from world_model.baselines.lstm_baseline import LSTMDynamicsBaseline
from world_model.data.latent_dataset import LatentSequenceDataset
from world_model.data.splits import chronological_split
from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer
from world_model.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> nn.Module:
    """Trains a baseline model on the training set."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        for batch in train_loader:
            z_seq = batch["tf_input_z"].to(device)
            target = batch["tf_target_z"].to(device)

            optimizer.zero_grad()
            out = model(z_seq)
            pred = out["z_pred"]

            # If pred sequence length is smaller than target (e.g. LinearBaseline), align from tail
            if pred.size(1) < target.size(1):
                target_slice = target[:, -pred.size(1):, :]
            else:
                target_slice = target

            loss = loss_fn(pred, target_slice)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_model_on_test(
    model_name: str,
    model: Optional[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    K: int = 4,
) -> Dict[str, float]:
    """Evaluates 1-step and K-step predictions on holdout test set."""
    total_1step_sq_err = 0.0
    total_4step_sq_err = 0.0
    total_cossim = 0.0
    total_elements_1step = 0
    total_elements_4step = 0
    total_samples = 0

    t0 = time.time()

    if model is not None:
        model.eval()

    for batch in test_loader:
        z_hist = batch["z_history"].to(device)   # [B, H, d_z]
        ts_hist = batch["ts_history"].to(device) # [B, H]
        z_fut = batch["z_future"].to(device)     # [B, K, d_z]
        input_z = batch["tf_input_z"].to(device) # [B, H+K-1, d_z]
        target_z = batch["tf_target_z"].to(device)

        B = z_hist.size(0)
        total_samples += B

        if model_name == "Persistence":
            # ẑ_{t+1} = z_t
            pred_1step = input_z
            target_1step = target_z
            diff_1step = pred_1step - target_1step
            total_1step_sq_err += (diff_1step ** 2).sum().item()
            total_elements_1step += diff_1step.numel()

            # For K-step persistence: ẑ_{t+k} = z_t
            z_last = z_hist[:, -1:, :].expand(-1, K, -1)
            diff_4step = z_last - z_fut[:, :K, :]
            total_4step_sq_err += (diff_4step ** 2).sum().item()
            total_elements_4step += diff_4step.numel()

            cos = nn.functional.cosine_similarity(z_last[:, 0, :], z_fut[:, 0, :], dim=-1).sum().item()
            total_cossim += cos

        elif model_name == "Linear AR":
            out_1 = model(input_z)
            pred_1step = out_1["z_pred"]
            target_1step = target_z[:, -pred_1step.size(1):, :]
            diff_1step = pred_1step - target_1step
            total_1step_sq_err += (diff_1step ** 2).sum().item()
            total_elements_1step += diff_1step.numel()

            roll_out = model.rollout(z_hist, K=K)
            diff_4step = roll_out["z_future"][:, :K, :] - z_fut[:, :K, :]
            total_4step_sq_err += (diff_4step ** 2).sum().item()
            total_elements_4step += diff_4step.numel()

            cos = nn.functional.cosine_similarity(roll_out["z_future"][:, 0, :], z_fut[:, 0, :], dim=-1).sum().item()
            total_cossim += cos

        elif model_name == "LSTM":
            out_1 = model(input_z)
            diff_1step = out_1["z_pred"] - target_z
            total_1step_sq_err += (diff_1step ** 2).sum().item()
            total_elements_1step += diff_1step.numel()

            roll_out = model.rollout(z_hist, K=K)
            diff_4step = roll_out["z_future"][:, :K, :] - z_fut[:, :K, :]
            total_4step_sq_err += (diff_4step ** 2).sum().item()
            total_elements_4step += diff_4step.numel()

            cos = nn.functional.cosine_similarity(roll_out["z_future"][:, 0, :], z_fut[:, 0, :], dim=-1).sum().item()
            total_cossim += cos

        elif model_name == "WDT (Ours)":
            out_1 = model(input_z, batch["tf_input_ts"].to(device))
            diff_1step = out_1["z_pred"] - target_z
            total_1step_sq_err += (diff_1step ** 2).sum().item()
            total_elements_1step += diff_1step.numel()

            roll_out = model.rollout(z_hist, ts_hist, K=K, use_kv_cache=True)
            diff_4step = roll_out["z_future"][:, :K, :] - z_fut[:, :K, :]
            total_4step_sq_err += (diff_4step ** 2).sum().item()
            total_elements_4step += diff_4step.numel()

            cos = nn.functional.cosine_similarity(roll_out["z_future"][:, 0, :], z_fut[:, 0, :], dim=-1).sum().item()
            total_cossim += cos

    elapsed = time.time() - t0
    latency_per_sample_ms = (elapsed / max(1, total_samples)) * 1000.0

    mse_1 = total_1step_sq_err / max(1, total_elements_1step)
    mse_4 = total_4step_sq_err / max(1, total_elements_4step)
    cos_avg = total_cossim / max(1, total_samples)

    params = sum(p.numel() for p in model.parameters()) if model is not None else 0

    return {
        "mse_1step": float(mse_1),
        "mse_4step": float(mse_4),
        "cosine_similarity": float(cos_avg),
        "parameters": params,
        "latency_ms_per_sample": float(latency_per_sample_ms),
    }


def run_benchmark(
    config_path: str = "world_model/config/default_config.yaml",
    wdt_checkpoint: str = "saved_models/world_model/wdt_best.pt",
    output_path: str = "results/baseline_comparison.json",
):
    """Executes full comparative benchmark."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running baseline benchmark on {device}...")

    # Load data
    cache_path = Path(cfg["data"]["latent_cache_dir"]) / "ctu13_latent_dataset.pt"
    data_dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    latent_states = data_dict["latent_states"]
    timestamps = data_dict["timestamps"]
    stats = data_dict["statistics"]
    labels = data_dict["labels"]
    mal_fractions = data_dict.get("malicious_fractions", torch.zeros_like(timestamps))

    train_mask, _, test_mask = chronological_split(
        timestamps,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    H = cfg["data"]["history_len"]
    K = cfg["data"]["horizon_k"]

    train_ds = LatentSequenceDataset(
        latent_states[train_mask], timestamps[train_mask], stats[train_mask],
        labels[train_mask], mal_fractions[train_mask], history_len=H, horizon_k=K,
    )
    test_ds = LatentSequenceDataset(
        latent_states[test_mask], timestamps[test_mask], stats[test_mask],
        labels[test_mask], mal_fractions[test_mask], history_len=H, horizon_k=K,
    )

    batch_size = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    # 1. Persistence Baseline
    logger.info("--> Evaluating Persistence Baseline...")
    p_res = evaluate_model_on_test("Persistence", None, test_loader, device, K=K)
    results["Persistence"] = p_res
    base_mse_1 = p_res["mse_1step"]
    base_mse_4 = p_res["mse_4step"]
    p_res["improvement_pct"] = 0.0

    # 2. Linear Autoregressive Baseline
    logger.info("--> Training & Evaluating Linear AR Baseline...")
    linear_model = LinearDynamicsBaseline(history_len=H, d_z=cfg["model"]["d_z"])
    linear_model = train_baseline_model(linear_model, train_loader, device, epochs=3)
    l_res = evaluate_model_on_test("Linear AR", linear_model, test_loader, device, K=K)
    l_res["improvement_pct"] = ((base_mse_1 - l_res["mse_1step"]) / base_mse_1) * 100.0
    results["Linear AR"] = l_res

    # 3. LSTM Baseline
    logger.info("--> Training & Evaluating LSTM Baseline...")
    lstm_model = LSTMDynamicsBaseline(d_z=cfg["model"]["d_z"], hidden_dim=256, num_layers=2)
    lstm_model = train_baseline_model(lstm_model, train_loader, device, epochs=3)
    lstm_res = evaluate_model_on_test("LSTM", lstm_model, test_loader, device, K=K)
    lstm_res["improvement_pct"] = ((base_mse_1 - lstm_res["mse_1step"]) / base_mse_1) * 100.0
    results["LSTM"] = lstm_res

    # 4. World Dynamics Transformer
    logger.info("--> Evaluating World Dynamics Transformer...")
    m_cfg = cfg["model"]
    wdt_model = WorldDynamicsTransformer(
        d_z=m_cfg["d_z"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_layers=m_cfg["n_layers"],
        d_ff=m_cfg["d_ff"],
        d_time=m_cfg["d_time"],
        max_len=m_cfg["max_len"],
        dropout=0.0,
        default_horizon=m_cfg["default_horizon"],
    ).to(device)

    load_checkpoint(wdt_checkpoint, model=wdt_model, device=device)
    wdt_res = evaluate_model_on_test("WDT (Ours)", wdt_model, test_loader, device, K=K)
    wdt_res["improvement_pct"] = ((base_mse_1 - wdt_res["mse_1step"]) / base_mse_1) * 100.0
    results["WDT (Ours)"] = wdt_res

    # Log Comparative Table
    logger.info("=========================================================================================")
    logger.info(f"{'Model':<18} | {'1-Step MSE':<11} | {'4-Step MSE':<11} | {'Cos Sim':<8} | {'Improvement':<12} | {'Params':<10}")
    logger.info("-----------------------------------------------------------------------------------------")
    for name, r in results.items():
        logger.info(
            f"{name:<18} | {r['mse_1step']:<11.6f} | {r['mse_4step']:<11.6f} | {r['cosine_similarity']:<8.4f} | "
            f"{r['improvement_pct']:>+9.2f}%   | {r['parameters']:<10,}"
        )
    logger.info("=========================================================================================")

    # Save to JSON
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparative benchmark report to {out_p}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Comparison Benchmark")
    parser.add_argument("--config", type=str, default="world_model/config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="saved_models/world_model/wdt_best.pt")
    parser.add_argument("--output", type=str, default="results/baseline_comparison.json")
    args = parser.parse_args()

    run_benchmark(args.config, args.checkpoint, args.output)
