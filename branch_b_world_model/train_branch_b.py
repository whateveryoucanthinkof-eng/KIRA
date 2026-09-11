"""
Training pipeline for Branch B (HostWorldDynamicsTransformer & InfiltrationRiskHead).
Trains multi-step latent rollout on unified host trajectories and benchmarks K-step horizons.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.split_manager import ScientificSplitManager
from data_unification.multi_dataset_stream import HostTrajectoryExtractor, HostWindowSnapshot
from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from branch_b_world_model.infiltration_head import InfiltrationRiskHead


class HostRolloutDataset(Dataset):
    """
    Dataset yielding (h_history, h_future_targets, risk_targets).
    h_history: [T, d_latent]
    h_future_targets: [K, d_latent]
    risk_targets: [K]
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_history": torch.from_numpy(s["h_history"]).float(),
            "h_future": torch.from_numpy(s["h_future"]).float(),
            "risk_future": torch.from_numpy(s["risk_future"]).float(),
        }


def create_rollout_samples(trajectories, T: int = 4, K: int = 4):
    samples = []
    for host_ip, snaps in trajectories.items():
        if len(snaps) < T + 1:
            continue
        snaps = sorted(snaps, key=lambda s: s.window_idx)
        n = len(snaps)

        for i in range(T, n):
            h_hist = np.array([s.embedding for s in snaps[i - T : i]], dtype=np.float32)
            # Future slice up to K
            future_snaps = snaps[i : min(n, i + K)]
            k_avail = len(future_snaps)

            # Pad future if less than K
            h_fut = np.array([s.embedding for s in future_snaps], dtype=np.float32)
            r_fut = np.array([s.risk_score for s in future_snaps], dtype=np.float32)

            if k_avail < K:
                pad_k = K - k_avail
                h_fut = np.pad(h_fut, ((0, pad_k), (0, 0)), mode="edge")
                r_fut = np.pad(r_fut, (0, pad_k), mode="edge")

            samples.append({
                "h_history": h_hist,
                "h_future": h_fut,
                "risk_future": r_fut,
            })
    return samples


def train_branch_b(
    epochs: int = 4,
    T: int = 4,
    K: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "saved_models/branch_b/host_wdt.pt",
):
    print("Loading disjoint train/val multi-dataset partitions via ScientificSplitManager for Branch B...")
    sm = ScientificSplitManager()
    train_records = sm.get_train_records(max_per_source=600)
    val_records = sm.get_val_records(max_per_source=200)

    tgn = build_or_load_tgne_ta()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=60.0)
    train_trajectories = extractor.extract_trajectories(train_records)
    val_trajectories = extractor.extract_trajectories(val_records)

    train_samples = create_rollout_samples(train_trajectories, T=T, K=K)
    val_samples = create_rollout_samples(val_trajectories, T=T, K=K)
    print(f"Disjoint rollout samples: Train={len(train_samples)}, Val={len(val_samples)}")

    if len(train_samples) < 10:
        train_samples = train_samples * 5
    if len(val_samples) < 5:
        val_samples = val_samples * 5

    train_set = HostRolloutDataset(train_samples)
    val_set = HostRolloutDataset(val_samples)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4, n_layers=3).to(device)
    risk_head = InfiltrationRiskHead(d_latent=12, hidden_dim=32).to(device)

    params = list(wdt.parameters()) + list(risk_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float("inf")

    print(f"Starting Branch B WDT Rollout training on {device}...")
    gamma = 0.9  # discount factor for horizon loss

    for epoch in range(1, epochs + 1):
        wdt.train()
        risk_head.train()
        train_losses = []

        for batch in train_loader:
            h_hist = batch["h_history"].to(device)
            h_fut = batch["h_future"].to(device)
            r_fut = batch["risk_future"].to(device)

            optimizer.zero_grad()

            # Autoregressive rollout across K steps
            h_pred = wdt.rollout(h_hist, K=K)  # [B, K, d_latent]

            # Discounted multi-horizon MSE loss
            mse_loss = 0.0
            for k in range(K):
                mse_k = F.mse_loss(h_pred[:, k, :], h_fut[:, k, :])
                mse_loss += (gamma ** k) * mse_k

            # Infiltration risk BCE loss
            pred_step_risks, _ = risk_head.forward_trajectory(h_pred)
            risk_loss = F.binary_cross_entropy(pred_step_risks, r_fut)

            total_loss = mse_loss + risk_loss
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

        # Evaluation
        wdt.eval()
        risk_head.eval()
        val_losses = []
        k1_mses, k4_mses, persist_mses = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                h_hist = batch["h_history"].to(device)
                h_fut = batch["h_future"].to(device)
                r_fut = batch["risk_future"].to(device)

                h_pred = wdt.rollout(h_hist, K=K)
                mse_loss = F.mse_loss(h_pred, h_fut)
                pred_step_risks, _ = risk_head.forward_trajectory(h_pred)
                risk_loss = F.binary_cross_entropy(pred_step_risks, r_fut)
                val_losses.append((mse_loss + risk_loss).item())

                # Compare Horizon 1 MSE vs Persistence Baseline
                k1_mse = F.mse_loss(h_pred[:, 0, :], h_fut[:, 0, :]).item()
                persist_mse = F.mse_loss(h_hist[:, -1, :], h_fut[:, 0, :]).item()
                k4_mse = F.mse_loss(h_pred[:, -1, :], h_fut[:, -1, :]).item()

                k1_mses.append(k1_mse)
                persist_mses.append(persist_mse)
                k4_mses.append(k4_mse)

        mean_val = float(np.mean(val_losses))
        mean_k1 = float(np.mean(k1_mses))
        mean_persist = float(np.mean(persist_mses))
        mean_k4 = float(np.mean(k4_mses))
        improvement = ((mean_persist - mean_k1) / max(1e-6, mean_persist)) * 100.0

        print(
            f"Epoch {epoch:02d} | Val Loss: {mean_val:.4f} | "
            f"1-Step MSE: {mean_k1:.4f} vs Persist: {mean_persist:.4f} (+{improvement:.1f}%) | "
            f"K=4 MSE: {mean_k4:.4f}"
        )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "wdt_state_dict": wdt.state_dict(),
                    "risk_head_state_dict": risk_head.state_dict(),
                    "epoch": epoch,
                    "k1_mse": mean_k1,
                    "k4_mse": mean_k4,
                },
                save_path,
            )

    print(f"Branch B models saved to {save_path}")
    return {
        "best_val_loss": best_val_loss,
        "k1_mse": mean_k1,
        "k4_mse": mean_k4,
        "save_path": save_path,
    }


if __name__ == "__main__":
    train_branch_b(epochs=3)
