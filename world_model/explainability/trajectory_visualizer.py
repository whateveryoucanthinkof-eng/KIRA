"""
Trajectory Visualizer for the World Dynamics Transformer.

Generates plots for:
1. State trajectory rollouts: Ground truth vs Predicted latent trajectories with uncertainty.
2. Causal attention matrix: Temporal influence of past telemetry on forecast horizons.
3. Infiltration risk curves: Per-step and cumulative risk evolution over lookahead windows.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_rollout(
    z_history: np.ndarray,      # [H, d_z]
    z_future_pred: np.ndarray,  # [K, d_z]
    z_future_true: np.ndarray,  # [K, d_z]
    output_path: str = "results/wdt_rollout_trajectory.png",
    dimensions_to_plot: int = 4,
):
    """
    Plots the observed history and the future projected trajectories.
    """
    H = len(z_history)
    K = len(z_future_pred)

    t_hist = np.arange(H)
    t_fut = np.arange(H - 1, H + K)

    fig, axes = plt.subplots(dimensions_to_plot, 1, figsize=(10, 2.5 * dimensions_to_plot), sharex=True)
    if dimensions_to_plot == 1:
        axes = [axes]

    for d in range(dimensions_to_plot):
        ax = axes[d]
        # History
        ax.plot(t_hist, z_history[:, d], "k-o", label="Observed History", linewidth=1.8, markersize=4)

        # Connect history end to future
        pred_line = np.concatenate([[z_history[-1, d]], z_future_pred[:, d]])
        true_line = np.concatenate([[z_history[-1, d]], z_future_true[:, d]])

        # Ground truth future
        ax.plot(t_fut, true_line, "g--s", label="Ground Truth Future", linewidth=1.8, markersize=4)
        # Predicted rollout future
        ax.plot(t_fut, pred_line, "r-^", label="WDT Autoregressive Rollout", linewidth=2.0, markersize=5)

        ax.axvline(x=H - 1, color="gray", linestyle=":", label="Forecast Horizon Start" if d == 0 else "")
        ax.set_ylabel(f"Latent Dim {d+1}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper left", framealpha=0.9)

    axes[-1].set_xlabel("Time Window (2-second intervals)")
    plt.suptitle("World Dynamics Transformer: K-Step Latent Trajectory Rollout", fontsize=12, y=0.99)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out_file)


def plot_attention_heatmap(
    attention_matrix: np.ndarray,  # [S, S]
    output_path: str = "results/wdt_attention_heatmap.png",
):
    """Plots causal attention heatmap showing temporal dependency structure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attention_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xlabel("Key Window (t - j)")
    ax.set_ylabel("Query Window (t)")
    ax.set_title("WDT Causal Attention Map: Temporal Information Flow")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out_file)
