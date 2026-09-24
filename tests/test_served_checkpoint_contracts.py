"""The checkpoints the live adapter serves must match the authoritative contract.

Each branch carries its own marker, because they are retrained one at a time.
The markers are xfail(strict=True) so a branch that starts *passing* fails the
suite until its marker is removed -- that is how a finished retrain announces
itself instead of being quietly forgotten.

Status: all three are STALE and marked xfail, pending the retrain that two
deliberate changes require.

  1. Feature schema 1.0.0 -> 2.0.0. dst_port is log-scaled instead of divided
     by 65535, and unique_peers / unique_dst_ports no longer saturate at 147.
     Every stored feature value changed.
  2. Forecast horizon 10s -> 150s. `forecast_window_seconds` is 30.0, so a
     forecast step covers fifteen input windows instead of one. At 10s the
     label essentially never changed and persistence scored Brier 0.00067.

Both are value changes with unchanged tensor shapes, which is the failure mode
that produces no error anywhere -- so the contract check is the only thing that
can catch it, and it is doing so here on purpose.

Remove a marker when that branch is retrained. The markers are strict, so a
retrained checkpoint makes the suite fail until its marker goes, which is how a
finished retrain announces itself instead of being quietly forgotten.
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

#: Branches awaiting a retrain under schema 2.0.0 and the 150s horizon.
#: Delete an entry when its checkpoint is regenerated.
STALE_PENDING_RETRAIN = {
    "branch_a": "trained at a 10s horizon under feature schema 1.0.0",
    "branch_b": "trained at a 10s horizon under feature schema 1.0.0",
    "deepop": "trained at a 10s horizon under feature schema 1.0.0",
}


def _served_params():
    out = []
    for name, path in sorted(SERVED.items()):
        marks = []
        if name in STALE_PENDING_RETRAIN:
            marks.append(pytest.mark.xfail(
                strict=True,
                reason=f"{name}: {STALE_PENDING_RETRAIN[name]}; retrain required",
            ))
        out.append(pytest.param(name, path, marks=marks, id=f"{name}-{path}"))
    return out


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


@pytest.mark.parametrize("name,path", _served_params())
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


def test_a_stale_checkpoint_is_actually_detected():
    """The guard must not pass a 10s-horizon checkpoint as a 150s one.

    This is the assertion that keeps its teeth while the three above are
    xfailed: a value-only contract change leaves tensor shapes identical, so
    nothing else in the stack would notice.
    """
    contract = get_contract()
    legacy = {"window_seconds": 2.0, "history_steps": 15, "forecast_steps": 5}
    assert not contract.matches(legacy), (
        "a checkpoint predating forecast_window_seconds was accepted; it was "
        "trained on a 10-second horizon"
    )
    assert contract.matches({**legacy, "forecast_window_seconds":
                             contract.forecast_window_seconds})
