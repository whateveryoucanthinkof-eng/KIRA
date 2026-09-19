"""
Explainability and Trajectory Attribution Demo Script.

Extracts interpretability insights from the trained World Dynamics Transformer:
1. Multi-head causal attention maps and temporal importance weights.
2. Axiomatic Integrated Gradients feature attribution.
3. High-resolution rollout trajectory plots comparing predicted dynamics vs ground truth.
Outputs artifacts to results/ directory.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from world_model.data.latent_dataset import LatentSequenceDataset
from world_model.data.splits import chronological_split
from world_model.explainability.attention_extractor import extract_causal_attention
from world_model.explainability.integrated_gradients import compute_integrated_gradients
from world_model.explainability.trajectory_visualizer import plot_attention_heatmap, plot_trajectory_rollout
from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer
from world_model.training.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_explainability_demo(
    config_path: str = "world_model/config/default_config.yaml",
    checkpoint_path: str = "saved_models/world_model/wdt_best.pt",
    results_dir: str = "results",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    cache_path = Path(cfg["data"]["latent_cache_dir"]) / "ctu13_latent_dataset.pt"
    data_dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    latent_states = data_dict["latent_states"]
    timestamps = data_dict["timestamps"]
    stats = data_dict["statistics"]
    labels = data_dict["labels"]

    _, _, test_mask = chronological_split(
        timestamps,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    test_ds = LatentSequenceDataset(
        latent_states[test_mask],
        timestamps[test_mask],
        stats[test_mask],
        labels[test_mask],
        history_len=cfg["data"]["history_len"],
        horizon_k=cfg["data"]["horizon_k"],
    )

    # Find an interesting attack sequence in test set (where future contains attack)
    sample_idx = 0
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        if item["future_risk_cumulative"].item() > 0.5:
            sample_idx = idx
            break

    logger.info(f"Selected attack trajectory test sample index: {sample_idx}")
    sample = test_ds[sample_idx]
    z_hist = sample["z_history"].unsqueeze(0).to(device)    # [1, H, d_z]
    ts_hist = sample["ts_history"].unsqueeze(0).to(device)  # [1, H]
    z_fut_true = sample["z_future"].numpy()                 # [K, d_z]

    # 2. Load Model
    m_cfg = cfg["model"]
    model = WorldDynamicsTransformer(
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

    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()

    # 3. Autoregressive Rollout
    logger.info("--> Running K-step autoregressive rollout...")
    rollout = model.rollout(z_hist, ts_hist, K=4, use_kv_cache=True)
    z_fut_pred = rollout["z_future"].squeeze(0).cpu().numpy()

    # Plot rollout trajectory
    traj_plot_path = out_dir / "wdt_rollout_trajectory.png"
    plot_trajectory_rollout(
        z_history=sample["z_history"].numpy(),
        z_future_pred=z_fut_pred,
        z_future_true=z_fut_true,
        output_path=str(traj_plot_path),
        dimensions_to_plot=4,
    )
    logger.info(f"Saved trajectory rollout plot to {traj_plot_path}")

    # 4. Extract Causal Attention Map
    logger.info("--> Extracting causal attention weights...")
    attn_data = extract_causal_attention(model, z_hist, ts_hist)
    attn_plot_path = out_dir / "wdt_attention_heatmap.png"
    plot_attention_heatmap(attn_data["mean_attention"], output_path=str(attn_plot_path))
    logger.info(f"Saved causal attention heatmap to {attn_plot_path}")

    # 5. Integrated Gradients Feature Attribution
    logger.info("--> Computing Integrated Gradients temporal attribution...")
    ig_data = compute_integrated_gradients(model, z_hist, ts_hist, steps=20)
    window_importance = ig_data["window_importance"].tolist()

    summary = {
        "sample_index": sample_idx,
        "history_windows": cfg["data"]["history_len"],
        "forecast_windows": cfg["data"]["horizon_k"],
        "temporal_importance_weights": attn_data["temporal_importance"].tolist(),
        "integrated_gradients_window_importance": window_importance,
        "most_influential_past_window": int(np.argmax(window_importance)),
    }

    summary_file = out_dir / "explainability_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved explainability summary to {summary_file}")
    logger.info("Explainability demo completed successfully!")

    return summary


if __name__ == "__main__":
    run_explainability_demo()
