"""
Centralized Temporal Granularity Configuration.
Unifies timescale contracts across data unification, GNN dynamic graph snapshots,
Branch B World Dynamics Transformer rollouts, and SOC dashboard operations.
"""

from enum import Enum
from typing import Union
import torch


class TimescaleMode(str, Enum):
    """
    Operating timescales for the predictive cybersecurity platform.
    """
    LIVE_MICRO = "live_micro"    # Real-time streaming: 2.0s windows, K=8 steps (16s forward horizon)
    MACRO_BATCH = "macro_batch"  # Offline PCAP batching: 60.0s windows, K=4 steps (240s forward horizon)


# Authoritative Temporal Constants
LIVE_WINDOW_SIZE_SEC: float = 2.0
MACRO_WINDOW_SIZE_SEC: float = 60.0

DEFAULT_ROLLOUT_HORIZON_LIVE: int = 8    # 8 * 2.0s = 16.0s
DEFAULT_ROLLOUT_HORIZON_MACRO: int = 4   # 4 * 60.0s = 240.0s
DEFAULT_HISTORY_STEPS: int = 5           # History steps L = 5 across both live & macro


def get_default_window_size(mode: Union[TimescaleMode, str] = TimescaleMode.LIVE_MICRO) -> float:
    """Returns the default temporal step size in seconds for the given mode."""
    if isinstance(mode, str):
        mode = TimescaleMode(mode)
    if mode == TimescaleMode.LIVE_MICRO:
        return LIVE_WINDOW_SIZE_SEC
    elif mode == TimescaleMode.MACRO_BATCH:
        return MACRO_WINDOW_SIZE_SEC
    return LIVE_WINDOW_SIZE_SEC


def get_default_horizon_steps(mode: Union[TimescaleMode, str] = TimescaleMode.LIVE_MICRO) -> int:
    """Returns the default forecast horizon step count K for the given mode."""
    if isinstance(mode, str):
        mode = TimescaleMode(mode)
    if mode == TimescaleMode.LIVE_MICRO:
        return DEFAULT_ROLLOUT_HORIZON_LIVE
    elif mode == TimescaleMode.MACRO_BATCH:
        return DEFAULT_ROLLOUT_HORIZON_MACRO
    return DEFAULT_ROLLOUT_HORIZON_LIVE


def validate_temporal_contract(
    runtime_window_size: float,
    model_window_size: float,
    requested_horizon: int,
    model_horizon: int,
    requested_history: int = DEFAULT_HISTORY_STEPS,
    model_history: int = DEFAULT_HISTORY_STEPS,
) -> None:
    """
    Validates that runtime execution respects the temporal contract of the model.
    Fails loudly with ValueError if there is an irreconcilable temporal mismatch.
    """
    if abs(runtime_window_size - model_window_size) > 1e-3:
        raise ValueError(
            f"Temporal mismatch: runtime window ({runtime_window_size}s) != model window ({model_window_size}s). "
            f"A model trained on {model_window_size}s transitions cannot be deployed with {runtime_window_size}s windows."
        )
    if requested_horizon > model_horizon:
        raise ValueError(
            f"Horizon mismatch: requested horizon ({requested_horizon}) exceeds model capability ({model_horizon})."
        )
    if requested_history != model_history:
        raise ValueError(
            f"History mismatch: requested history ({requested_history}) != model history ({model_history})."
        )



def normalize_time_delta(
    delta_t: Union[float, torch.Tensor],
    reference_dt: float = LIVE_WINDOW_SIZE_SEC,
) -> Union[float, torch.Tensor]:
    """
    Normalizes time delta relative to reference timescale to maintain consistent
    ContinuousTimeEncoding frequency responses across differing temporal contracts.
    """
    if isinstance(delta_t, torch.Tensor):
        return torch.clamp(delta_t / max(0.1, reference_dt), 0.0, 300.0)
    return max(0.0, min(300.0, float(delta_t) / max(0.1, reference_dt)))
