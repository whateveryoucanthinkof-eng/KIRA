"""The checkpoints the live adapter serves must match the authoritative contract.

This test is expected to FAIL until Branch A / Branch B / DeepOP are retrained.
It is the tracked reminder that `saved_models/` still holds v3 weights, and it
is marked xfail(strict=True) so that it also fails if it starts *passing*
without this marker being removed -- i.e. the retrain is not quietly forgotten.
"""

import os

import pytest
import torch

from cyberworld_v4.config import get_contract

SERVED = {
    "branch_a": "saved_models/branch_a/branch_a_lstm.pt",
    "branch_b": "saved_models/branch_b/host_wdt.pt",
    "deepop": "saved_models/deepop/cwa_forecast_decoder.pt",
}


def _carried_contract(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tc = ckpt.get("training_contract") or {}
    cfg = (ckpt.get("config") or {}).get("temporal") or {}
    return cfg or tc or ckpt


@pytest.mark.parametrize("name,path", sorted(SERVED.items()))
def test_served_checkpoint_records_a_contract_at_all(name, path):
    """A checkpoint with no contract can never be validated -- that alone is a defect."""
    if not os.path.exists(path):
        pytest.skip(f"{name} checkpoint not present")
    src = _carried_contract(path)
    has = [k for k in ("window_seconds", "window_size_sec", "history_steps", "forecast_steps")
           if src.get(k) is not None]
    assert has, f"{name} records no temporal contract whatsoever"


@pytest.mark.xfail(
    strict=True,
    reason="saved_models/ still holds v3 weights (history 5 / forecast 8). "
           "Remove this marker once the v4 retrain lands.",
)
@pytest.mark.parametrize("name,path", sorted(SERVED.items()))
def test_served_checkpoints_match_the_contract(name, path):
    if not os.path.exists(path):
        pytest.skip(f"{name} checkpoint not present")
    contract = get_contract()
    src = _carried_contract(path)
    assert contract.matches(src), (
        f"{name} was trained under a different contract: "
        f"{ {k: src.get(k) for k in ('window_seconds', 'window_size_sec', 'history_steps', 'forecast_steps')} } "
        f"vs {contract.to_dict()}"
    )
