"""
cyberworld_v4.contract — enforcement of the temporal contract.

Spec 44: the contract must be enforced at training, validation, calibration,
test, replay, checkpoint load and live inference. Enforcement means *refusing
to run*, not logging a warning. A model trained on 2 s transitions with a
15-step history is a different model from one trained at 60 s with 5 steps;
silently serving one as the other is the defect this module exists to prevent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import TemporalContract, get_contract


class ContractViolation(RuntimeError):
    """Raised when runtime and model temporal contracts disagree."""


def validate(
    observed: Dict[str, Any],
    expected: Optional[TemporalContract] = None,
    *,
    where: str = "runtime",
) -> None:
    """Raise ContractViolation unless `observed` matches the contract exactly.

    `observed` is any mapping carrying window_seconds / history_steps /
    forecast_steps — a checkpoint dict, a manifest, a config payload.
    """
    expected = expected or get_contract()
    if expected.matches(observed):
        return

    got = {
        k: observed.get(k, "<missing>")
        for k in ("window_seconds", "history_steps", "forecast_steps")
    }
    raise ContractViolation(
        f"Temporal contract violation at {where}.\n"
        f"  expected: {expected.describe()}\n"
        f"  observed: window={got['window_seconds']} "
        f"history={got['history_steps']} forecast={got['forecast_steps']}\n"
        f"A model trained under a different contract cannot be served under this one. "
        f"Retrain under the v4 contract or load the matching checkpoint."
    )


def validate_checkpoint(ckpt: Dict[str, Any], *, name: str = "checkpoint") -> None:
    """Validate a loaded checkpoint's embedded contract.

    Accepts either a v4 `config` block or the v3 `training_contract` /
    top-level keys, so v3 artefacts fail with a precise message rather than a
    KeyError.
    """
    payload: Dict[str, Any] = {}

    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        payload = dict(cfg.get("temporal") or {})

    if not payload:
        tc = ckpt.get("training_contract")
        if isinstance(tc, dict):
            payload = {
                "window_seconds": tc.get("window_seconds", tc.get("window_size_sec")),
                "history_steps": tc.get("history_steps"),
                "forecast_steps": tc.get("forecast_steps"),
            }

    if not payload:
        payload = {
            "window_seconds": ckpt.get("window_seconds", ckpt.get("window_size_sec")),
            "history_steps": ckpt.get("history_steps"),
            "forecast_steps": ckpt.get("forecast_steps"),
        }

    validate(payload, where=name)


def stamp(obj: Dict[str, Any], contract: Optional[TemporalContract] = None) -> Dict[str, Any]:
    """Embed the contract into a checkpoint/manifest dict, in place."""
    contract = contract or get_contract()
    obj["temporal"] = contract.to_dict()
    return obj
