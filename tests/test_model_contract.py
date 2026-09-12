"""
test_model_contract.py
Pytest cases locking in the dimensions from the authoritative contract.
"""
import sys
from pathlib import Path
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import model_contract

def test_dimensions():
    assert model_contract.TGNE_LATENT_DIM == 12
    assert model_contract.TEMPORAL_ATTRS == 15
    assert model_contract.BRANCH_A_INPUT_DIM == 27
    assert model_contract.BRANCH_B_ENCODER_LAYERS == 3
    assert model_contract.DEEPOP_VOCAB_SIZE == 10
    assert model_contract.DEEPOP_NHEAD == 6
    assert model_contract.DEEPOP_WINDOW_SIZES == [2, 4, 8]

def test_assertions():
    tensor = torch.zeros(1, 10, 12)
    model_contract.assert_shape(tensor, (1, 10, 12))
    with pytest.raises(ValueError):
        model_contract.assert_shape(tensor, (1, 10, 15))

    model_contract.assert_last_dim(tensor, 12)
    with pytest.raises(ValueError):
        model_contract.assert_last_dim(tensor, 15)
