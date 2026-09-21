"""The checkpoints the live adapter serves must match the authoritative contract.

Each branch carries its own marker, because they are retrained one at a time.
The markers are xfail(strict=True) so a branch that starts *passing* fails the
suite until its marker is removed -- that is how a finished retrain announces
itself instead of being quietly forgotten.

Status:
  branch_a -- retrained 2026-09-21 under the v4 contract (history 15 /
              forecast 5, 2s windows), full density, 20,664,255 samples.
              Marker removed; this is now a live assertion.
  branch_b -- still v3 weights (history 5 / forecast 8). Retrain is queued
              behind Branch A in scripts/retrain_future_models_live.py.
  deepop   -- same as branch_b; it trains from Branch B's rollouts.
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


_AWAITING_RETRAIN = pytest.mark.xfail(
    strict=True,
    reason="still v3 weights (history 5 / forecast 8); retrain queued behind "
           "Branch A. Remove this marker once that retrain lands.",
)


@pytest.mark.parametrize("name,path", [
    # Retrained under the v4 contract -- asserted for real, no marker.
    pytest.param("branch_a", SERVED["branch_a"]),
    pytest.param("branch_b", SERVED["branch_b"], marks=_AWAITING_RETRAIN),
    pytest.param("deepop", SERVED["deepop"], marks=_AWAITING_RETRAIN),
])
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
