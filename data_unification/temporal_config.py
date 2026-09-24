"""
Temporal granularity constants, DERIVED from the one authoritative contract.

This module used to declare a *second, conflicting* contract. It hardcoded
`DEFAULT_ROLLOUT_HORIZON_LIVE = 8` and `DEFAULT_HISTORY_STEPS = 5` while
`cyberworld_v4/config.py` -- the contract every trainer and checkpoint
validator uses -- says `forecast_steps = 5` and `history_steps = 15`.

That mattered because `control_backend/model_adapter.py` imports its serving
defaults from *here*, not from the contract. So the live path requested an
8-step rollout with 5 steps of history from models trained for a 5-step rollout
with 15 steps of history. Nothing raised: the shapes are permissive enough to
run, they are just wrong.

Everything below is now computed from `cyberworld_v4.config.get_contract()`.
There is exactly one contract. To change the temporal granularity, change it
there; this module follows.

`cyberworld_v4.config` imports only the standard library, so importing it here
introduces no cycle.
"""

from enum import Enum
from typing import Union

import torch

from cyberworld_v4.config import get_contract

_CONTRACT = get_contract()


class TimescaleMode(str, Enum):
    """Operating timescales for the predictive cybersecurity platform.

    Only LIVE_MICRO is supported. MACRO_BATCH is retained so that existing
    imports keep resolving, but **no model in this repository is trained at
    60-second granularity**, so selecting it is an error rather than a mode.
    """

    LIVE_MICRO = "live_micro"    # the contract: 2.0s windows, 15 history, 5 forecast
    MACRO_BATCH = "macro_batch"  # unsupported; see _unsupported_macro()


# ---------------------------------------------------------------------------
# Authoritative temporal constants -- all derived, none declared.
# ---------------------------------------------------------------------------

LIVE_WINDOW_SIZE_SEC: float = _CONTRACT.window_seconds          # 2.0
DEFAULT_ROLLOUT_HORIZON_LIVE: int = _CONTRACT.forecast_steps    # 5  (was hardcoded 8)
DEFAULT_HISTORY_STEPS: int = _CONTRACT.history_steps            # 15 (was hardcoded 5)

# Kept only so that `from ... import MACRO_WINDOW_SIZE_SEC` does not break.
# Any code that *uses* these is requesting a granularity no trained model has.
MACRO_WINDOW_SIZE_SEC: float = 60.0
DEFAULT_ROLLOUT_HORIZON_MACRO: int = 4


def _unsupported_macro(what: str):
    raise ValueError(
        f"MACRO_BATCH {what} is not supported: no model in this repository is trained "
        f"at {MACRO_WINDOW_SIZE_SEC}s granularity. The contract is "
        f"{_CONTRACT.describe()}. Use TimescaleMode.LIVE_MICRO."
    )


def get_default_window_size(mode: Union[TimescaleMode, str] = TimescaleMode.LIVE_MICRO) -> float:
    """The contract's window size. MACRO_BATCH raises rather than silently diverging."""
    if isinstance(mode, str):
        mode = TimescaleMode(mode)
    if mode == TimescaleMode.MACRO_BATCH:
        _unsupported_macro("window size")
    return LIVE_WINDOW_SIZE_SEC


def get_default_horizon_steps(mode: Union[TimescaleMode, str] = TimescaleMode.LIVE_MICRO) -> int:
    """The contract's forecast horizon. MACRO_BATCH raises rather than silently diverging."""
    if isinstance(mode, str):
        mode = TimescaleMode(mode)
    if mode == TimescaleMode.MACRO_BATCH:
        _unsupported_macro("rollout horizon")
    return DEFAULT_ROLLOUT_HORIZON_LIVE


def get_default_history_steps() -> int:
    """The contract's history length (15). Present so callers stop hardcoding 5."""
    return DEFAULT_HISTORY_STEPS


def normalize_time_delta(
    delta_t: Union[float, torch.Tensor],
    reference_dt: float = LIVE_WINDOW_SIZE_SEC,
) -> Union[float, torch.Tensor]:
    """
    Normalizes time delta relative to the reference timescale so that
    ContinuousTimeEncoding frequency responses stay consistent across contexts.
    """
    if isinstance(delta_t, torch.Tensor):
        return torch.clamp(delta_t / max(0.1, reference_dt), 0.0, 300.0)
    return max(0.0, min(300.0, float(delta_t) / max(0.1, reference_dt)))
