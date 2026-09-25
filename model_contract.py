"""
model_contract.py
The single authoritative contract for all model dimensions derived from checkpoints.
"""

TGNE_LATENT_DIM = 12
TEMPORAL_ATTRS = 15
BRANCH_A_INPUT_DIM = 27  # TGNE_LATENT_DIM + TEMPORAL_ATTRS
BRANCH_B_ENCODER_LAYERS = 3
DEEPOP_VOCAB_SIZE = 10
DEEPOP_NHEAD = 6
DEEPOP_WINDOW_SIZES = [2, 4, 8]  # matches cwa.py, both trainers and deepop.manifest.json

def assert_shape(tensor, expected_shape, name="Tensor"):
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch. Expected {expected_shape}, got {tensor.shape}")

def assert_last_dim(tensor, expected_dim, name="Tensor"):
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension mismatch. Expected {expected_dim}, got {tensor.shape[-1]}")
