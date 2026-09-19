"""
Attention Extraction and Temporal Attribution for the World Dynamics Transformer.

Extracts multi-head causal attention maps from the Transformer stack to explain:
1. Temporal Influence: Which past windows t-j had the highest impact on ẑ_{t+k}?
2. Attention Rollout: Aggregates attention across layers to produce full causal attribution DAGs.
3. Head Specialization: Identifies heads focusing on recent bursts vs long-term trends.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer


def extract_causal_attention(
    model: WorldDynamicsTransformer,
    z_history: torch.Tensor,
    timestamps: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Extracts attention weights across all layers and heads for a given sequence.
    
    Args:
        model: Trained WDT instance
        z_history: [1, S, d_z] or [B, S, d_z] input sequence
        timestamps: [1, S] or [B, S] timestamps
    Returns:
        Dict containing:
            "layer_attentions": list of [B, n_heads, S, S] arrays per layer
            "mean_attention": [S, S] mean attention map averaged over layers and heads
            "temporal_importance": [S] relative weight allocated to each past window
    """
    model.eval()
    if z_history.dim() == 2:
        z_history = z_history.unsqueeze(0)
    if timestamps.dim() == 1:
        timestamps = timestamps.unsqueeze(0)

    device = next(model.parameters()).device
    z_history = z_history.to(device)
    timestamps = timestamps.to(device)

    with torch.no_grad():
        output = model(
            z_history,
            timestamps,
            return_attn_weights=True,
        )

    attn_list = output["attn_weights"]  # List of L tensors [B, n_heads, S, S]
    layer_arrays = [a.cpu().numpy() for a in attn_list]

    # Stack layers: [L, B, n_heads, S, S]
    stacked = np.stack(layer_arrays, axis=0)

    # Average over layers, batch, heads: [S, S]
    mean_attn = stacked.mean(axis=(0, 1, 2))

    # The last row [S-1, :] represents what the final predicted state attended to in the history
    temporal_importance = mean_attn[-1, :]
    temporal_importance = temporal_importance / max(1e-8, temporal_importance.sum())

    return {
        "layer_attentions": layer_arrays,
        "mean_attention": mean_attn,
        "temporal_importance": temporal_importance,
    }
