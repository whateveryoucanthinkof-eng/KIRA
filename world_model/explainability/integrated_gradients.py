"""
Integrated Gradients for the World Dynamics Transformer.

Computes axiomatic feature attribution (Sundararajan et al., 2017) to explain
which specific latent dimensions and input windows caused a change in predicted
dynamics or triggered downstream risk/stage alerts.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer


def compute_integrated_gradients(
    model: WorldDynamicsTransformer,
    z_input: torch.Tensor,               # [1, H, d_z]
    timestamps: torch.Tensor,            # [1, H]
    baseline_z: Optional[torch.Tensor] = None,  # [1, H, d_z] (defaults to zeros or mean)
    steps: int = 25,
    target_dim: Optional[int] = None,    # specific latent dimension to explain, or None for norm
) -> Dict[str, np.ndarray]:
    """
    Computes path-integrated gradients from baseline to input.
    
    Args:
        model: Trained WDT model
        z_input: input history tensor [1, H, d_z]
        timestamps: timestamps tensor [1, H]
        baseline_z: reference baseline (default all zeros)
        steps: Riemann sum approximation steps
        target_dim: latent index to attribute (default None = L2 norm of delta)
    Returns:
        Dict with 'attributions' [H, d_z] and 'window_importance' [H].
    """
    model.eval()
    device = next(model.parameters()).device
    z_input = z_input.to(device)
    timestamps = timestamps.to(device)

    if baseline_z is None:
        baseline_z = torch.zeros_like(z_input)
    else:
        baseline_z = baseline_z.to(device)

    # Linear interpolation path: α from 0 to 1
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    delta_z = z_input - baseline_z

    grads_accum = torch.zeros_like(z_input)

    for alpha in alphas:
        # Interpolated input
        z_step = (baseline_z + alpha * delta_z).requires_grad_(True)

        output = model(z_step, timestamps)
        # Take predicted next-step delta from the last history window
        last_delta = output["delta_pred"][:, -1, :]  # [1, d_z]

        if target_dim is not None:
            target_scalar = last_delta[0, target_dim]
        else:
            target_scalar = torch.norm(last_delta, p=2)

        model.zero_grad()
        if z_step.grad is not None:
            z_step.grad.zero_()

        target_scalar.backward(retain_graph=True)
        grads_accum += z_step.grad.detach()

    # Riemann approximation: (input - baseline) * avg_grads
    avg_grads = grads_accum / steps
    attributions = (delta_z * avg_grads).squeeze(0).cpu().numpy()  # [H, d_z]

    # Overall window importance: L1 norm across dimensions for each time window
    window_importance = np.abs(attributions).sum(axis=-1)  # [H]
    window_importance = window_importance / max(1e-8, window_importance.sum())

    return {
        "attributions": attributions,
        "window_importance": window_importance,
    }
