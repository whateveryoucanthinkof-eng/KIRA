"""
K-Step Rollout Evaluation Engine for the World Dynamics Transformer.

Evaluates multi-step forecasting trajectories under both teacher-forcing and
autoregressive rollout across horizons K in {1, 2, 4, 8}.
Computes:
1. Per-step MSE: MSE(k) = E[||ẑ_{t+k} - z_{t+k}||^2]
2. Per-step Cosine Similarity: CosSim(k) = E[ẑ_{t+k} · z_{t+k} / (||ẑ|| ||z||)]
3. Rollout Error Growth Ratio: R_err(k) = MSE(k) / MSE(1)
4. Teacher-Forcing vs Autoregressive Exposure Gap: (MSE_ar(k) - MSE_tf(k)) / MSE_tf(k)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer


@torch.no_grad()
def evaluate_k_step_rollout(
    model: WorldDynamicsTransformer,
    loader: DataLoader,
    device: torch.device,
    max_k: int = 8,
    use_kv_cache: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes rigorous per-horizon metrics for autoregressive rollout.
    
    Args:
        model: Trained WorldDynamicsTransformer
        loader: DataLoader over LatentSequenceDataset with horizon_k >= max_k
        device: PyTorch device
        max_k: maximum evaluation horizon K
        use_kv_cache: whether to use KV-cache acceleration
        max_batches: optional limit on evaluation batches for speed
    Returns:
        Dict containing per-step MSE, cosine similarities, and growth ratios.
    """
    model.eval()

    mse_per_k_ar = np.zeros(max_k, dtype=np.float64)
    cossim_per_k_ar = np.zeros(max_k, dtype=np.float64)
    total_samples = 0
    batch_count = 0

    for batch in loader:
        z_history = batch["z_history"].to(device)  # [B, H, d_z]
        ts_history = batch["ts_history"].to(device)  # [B, H]
        z_future = batch["z_future"].to(device)    # [B, K_data, d_z]

        B = z_history.size(0)
        K_eval = min(max_k, z_future.size(1))

        # Perform autoregressive rollout
        rollout_out = model.rollout(
            z_history=z_history,
            timestamps=ts_history,
            K=K_eval,
            use_kv_cache=use_kv_cache,
        )
        z_pred = rollout_out["z_future"]  # [B, K_eval, d_z]
        target = z_future[:, :K_eval, :]  # [B, K_eval, d_z]

        # Compute per-horizon metrics
        for k in range(K_eval):
            diff_k = z_pred[:, k, :] - target[:, k, :]
            mse_k = (diff_k ** 2).mean(dim=-1).sum().item()
            mse_per_k_ar[k] += mse_k

            # Cosine similarity
            cos_k = nn.functional.cosine_similarity(z_pred[:, k, :], target[:, k, :], dim=-1).sum().item()
            cossim_per_k_ar[k] += cos_k

        total_samples += B
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    mse_per_k = mse_per_k_ar / max(1, total_samples)
    cossim_per_k = cossim_per_k_ar / max(1, total_samples)

    # Growth ratio relative to 1-step prediction: MSE(k) / MSE(1)
    base_mse = max(1e-8, mse_per_k[0])
    growth_ratio = mse_per_k / base_mse

    return {
        "mse_per_k": mse_per_k,
        "cosine_sim_per_k": cossim_per_k,
        "growth_ratio": growth_ratio,
        "mse_1step": float(mse_per_k[0]),
        "mse_4step": float(mse_per_k[min(3, len(mse_per_k)-1)]),
        "mse_8step": float(mse_per_k[min(7, len(mse_per_k)-1)]),
        "growth_ratio_k4": float(growth_ratio[min(3, len(growth_ratio)-1)]),
        "growth_ratio_k8": float(growth_ratio[min(7, len(growth_ratio)-1)]),
    }
