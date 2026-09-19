"""
Dynamics Evaluation Metrics for Latent Network States.

Computes transition quality metrics:
1. Per-step and horizon-averaged MSE: E[||ẑ_{t+k} - z_{t+k}||^2]
2. Per-step Mean Absolute Error (MAE): E[|ẑ_{t+k} - z_{t+k}|]
3. Directional Cosine Similarity: E[cos(ẑ_{t+k}, z_{t+k})]
4. Relative Frobenius Norm Error: ||ẑ - z||_F / ||z||_F
5. Error Accumulation Rate: MSE(k) / MSE(1)
"""

from typing import Dict, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_dynamics_metrics(
    z_pred: Union[np.ndarray, torch.Tensor],
    z_target: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Args:
        z_pred: [B, K, d_z] or [B, d_z] predicted latent state
        z_target: [B, K, d_z] or [B, d_z] true latent state
    Returns:
        Dict with overall and per-horizon metrics.
    """
    if isinstance(z_pred, np.ndarray):
        z_pred = torch.from_numpy(z_pred).float()
    if isinstance(z_target, np.ndarray):
        z_target = torch.from_numpy(z_target).float()

    if z_pred.dim() == 2:
        z_pred = z_pred.unsqueeze(1)
    if z_target.dim() == 2:
        z_target = z_target.unsqueeze(1)

    B, K, D = z_pred.shape

    # 1. Per-step MSE
    sq_diff = (z_pred - z_target) ** 2  # [B, K, D]
    mse_per_step = sq_diff.mean(dim=(0, 2)).cpu().numpy()  # [K]
    overall_mse = float(sq_diff.mean().item())

    # 2. Per-step MAE
    abs_diff = torch.abs(z_pred - z_target)
    mae_per_step = abs_diff.mean(dim=(0, 2)).cpu().numpy()
    overall_mae = float(abs_diff.mean().item())

    # 3. Directional Cosine Similarity
    cossim = F.cosine_similarity(z_pred, z_target, dim=-1)  # [B, K]
    cossim_per_step = cossim.mean(dim=0).cpu().numpy()  # [K]
    overall_cossim = float(cossim.mean().item())

    # 4. Relative Frobenius Norm Error
    norm_diff = torch.norm(z_pred - z_target, p="fro")
    norm_target = torch.norm(z_target, p="fro").clamp(min=1e-7)
    rel_frobenius = float((norm_diff / norm_target).item())

    # 5. Error growth ratio
    base_mse = max(1e-8, float(mse_per_step[0]))
    growth_ratio = mse_per_step / base_mse

    return {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "overall_cosine_similarity": overall_cossim,
        "relative_frobenius_error": rel_frobenius,
        "mse_per_step": mse_per_step,
        "mae_per_step": mae_per_step,
        "cosine_similarity_per_step": cossim_per_step,
        "growth_ratio": growth_ratio,
        "sub_exponential_growth": bool(growth_ratio[-1] < (len(growth_ratio) * 2.5)),
    }
