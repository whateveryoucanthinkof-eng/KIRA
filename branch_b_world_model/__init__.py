"""
Branch B: World Dynamics Transformer (Per-Host Latent Rollout & Infiltration Risk).
"""

from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from branch_b_world_model.infiltration_head import InfiltrationRiskHead

__all__ = [
    "HostWorldDynamicsTransformer",
    "InfiltrationRiskHead",
]
