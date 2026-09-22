"""The checkpoints the live adapter serves must match the authoritative contract.

Each branch carries its own marker, because they are retrained one at a time.
The markers are xfail(strict=True) so a branch that starts *passing* fails the
suite until its marker is removed -- that is how a finished retrain announces
itself instead of being quietly forgotten.

Status: all three retrained under the v4 contract (2s windows, history 15 /
forecast 5) on 2026-09-21/22 at full density. Every marker is gone and all
three are live assertions. If one starts failing, a checkpoint was replaced by
something trained under different temporal settings.
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
