"""
cyberworld_v4.config — the single source of truth for the v4 temporal contract.

Authoritative per the CyberWorld v4 specification, section 64:

    window_seconds  = 2
    history_steps   = 15   (30 s of causal history)
    forecast_steps  = 5    (10 s forecast horizon)

No other module may define these constants. Import them from here or load a
`CyberWorldConfig` from YAML. Anything that hardcodes a window size, a history
length or a horizon is a contract violation and should fail review.

This contract deliberately differs from the v3 live contract (history 5,
horizon 8). v3 checkpoints are therefore not loadable under v4 without
retraining; `contract.validate_checkpoint` refuses them by design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

CONFIG_VERSION = "4.0.0"

# --- Temporal contract (spec 64) ------------------------------------------
WINDOW_SECONDS: float = 2.0
HISTORY_STEPS: int = 15
FORECAST_STEPS: int = 5

#: Seconds per FORECAST step. Detection and forecasting do not want the same
#: resolution and were sharing one by accident.
#:
#: At the original 2.0s the horizon was 5 x 2 = 10 seconds. Attack stage
#: progression -- recon to initial access to lateral movement to exfiltration --
#: takes minutes to hours, so over ten seconds "nothing changes" is almost
#: always the right answer. That is not a modelling failure, it is the task
#: being trivial: results/v4_benchmark.json shows a persistence baseline at
#: Brier 0.00067 and PR-AUC 0.9997 at every step, and the credibility gate in
#: scripts/train_v4.py fires DEGENERATE TASK on exactly this.
#:
#: Raising it instead of raising FORECAST_STEPS is deliberate. K=150 at a 2s
#: step would give the same 5-minute horizon, but Branch B rolls out
#: autoregressively, so 150 compounding steps is a far worse estimator than 10
#: coarse ones -- and the technique/hazard heads would each need 150 outputs.
#:
#: History stays at 2s. Detection still sees fine detail; only the TARGETS are
#: coarsened, by OR-aggregating the future windows in each bucket (an attack
#: anywhere in the bucket makes the bucket an attack).
FORECAST_WINDOW_SECONDS: float = 30.0

HISTORY_SECONDS: float = WINDOW_SECONDS * HISTORY_STEPS             # 30.0
FORECAST_SECONDS: float = FORECAST_WINDOW_SECONDS * FORECAST_STEPS  # 150.0

#: How many input windows make one forecast bucket. Must be a whole number, or
#: a bucket would straddle a window boundary.
FORECAST_STRIDE: int = int(round(FORECAST_WINDOW_SECONDS / WINDOW_SECONDS))
assert abs(FORECAST_STRIDE * WINDOW_SECONDS - FORECAST_WINDOW_SECONDS) < 1e-9, (
    "FORECAST_WINDOW_SECONDS must be a whole multiple of WINDOW_SECONDS"
)

# --- Representation -------------------------------------------------------
TGNE_LATENT_DIM: int = 12
HOST_ATTR_DIM: int = 15
STATE_DIM: int = TGNE_LATENT_DIM + HOST_ATTR_DIM  # 27

# --- Evaluation (spec 64) -------------------------------------------------
SEEDS: tuple = (42, 123, 2024, 3407, 9001)

# Gaps larger than this break a trajectory rather than being treated as one
# transition. Spec 20: a 102-minute gap is not a 2-second transition.
MAX_GAP_SECONDS: float = WINDOW_SECONDS * 3.0


@dataclass(frozen=True)
class TemporalContract:
    window_seconds: float = WINDOW_SECONDS
    history_steps: int = HISTORY_STEPS
    forecast_steps: int = FORECAST_STEPS
    #: Seconds per forecast step. Defaults to FORECAST_WINDOW_SECONDS; set it
    #: equal to window_seconds to recover the original single-scale contract.
    forecast_window_seconds: float = FORECAST_WINDOW_SECONDS

    @property
    def history_seconds(self) -> float:
        return self.window_seconds * self.history_steps

    @property
    def forecast_seconds(self) -> float:
        return self.forecast_window_seconds * self.forecast_steps

    @property
    def forecast_stride(self) -> int:
        """Input windows per forecast bucket."""
        r = self.forecast_window_seconds / self.window_seconds
        n = int(round(r))
        if abs(n * self.window_seconds - self.forecast_window_seconds) > 1e-9 or n < 1:
            raise ValueError(
                f"forecast_window_seconds ({self.forecast_window_seconds}) must be a "
                f"whole multiple of window_seconds ({self.window_seconds})"
            )
        return n

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history_seconds"] = self.history_seconds
        d["forecast_seconds"] = self.forecast_seconds
        d["forecast_stride"] = self.forecast_stride
        return d

    def matches(self, other: Dict[str, Any], tol: float = 1e-6) -> bool:
        """True when `other` describes the same contract.

        `forecast_window_seconds` is compared too, and a checkpoint that
        predates the field is assumed single-scale (it equals window_seconds),
        which is what those checkpoints actually were.
        """
        try:
            fw = float(other.get("forecast_window_seconds",
                                 other["window_seconds"]))
            return (
                abs(float(other["window_seconds"]) - self.window_seconds) < tol
                and int(other["history_steps"]) == self.history_steps
                and int(other["forecast_steps"]) == self.forecast_steps
                and abs(fw - self.forecast_window_seconds) < tol
            )
        except (KeyError, TypeError, ValueError):
            return False

    def describe(self) -> str:
        scale = ("" if self.forecast_window_seconds == self.window_seconds
                 else f" @ {self.forecast_window_seconds:g}s/step")
        return (
            f"{self.window_seconds:g}s windows | "
            f"{self.history_steps} history ({self.history_seconds:g}s) | "
            f"{self.forecast_steps} forecast ({self.forecast_seconds:g}s{scale})"
        )


@dataclass
class CyberWorldConfig:
    """Full experiment configuration. Embedded verbatim in every checkpoint."""

    name: str = "CyberWorld"
    version: str = "v4"
    config_version: str = CONFIG_VERSION

    temporal: TemporalContract = field(default_factory=TemporalContract)

    tgne_latent_dim: int = TGNE_LATENT_DIM
    host_attr_dim: int = HOST_ATTR_DIM

    # Targets (spec 64). Severity is retained but demoted: it is an operator
    # ranking aid, never a probability and never a loss target for BCE.
    predict_current_attack: bool = True
    predict_future_hazard: bool = True
    predict_future_probability: bool = True
    predict_future_techniques: bool = True
    predict_future_host_state: bool = True
    predict_future_edges: bool = False

    # Labels (spec 13, 15)
    label_representation: str = "multilabel"
    unknown_policy: str = "explicit_unknown"

    # Identity (spec 18)
    identity_namespace: List[str] = field(
        default_factory=lambda: ["dataset", "scenario", "capture", "host"]
    )

    # Splits (spec 36)
    split_strategy: str = "chronological_and_scenario_held_out"
    seeds: List[int] = field(default_factory=lambda: list(SEEDS))

    max_gap_seconds: float = MAX_GAP_SECONDS

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["temporal"] = self.temporal.to_dict()
        d["state_dim"] = self.state_dim
        return d

    @property
    def state_dim(self) -> int:
        return self.tgne_latent_dim + self.host_attr_dim

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CyberWorldConfig":
        d = dict(d)
        d.pop("state_dim", None)
        t = d.pop("temporal", {}) or {}
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if t:
            object.__setattr__(
                cfg,
                "temporal",
                TemporalContract(
                    window_seconds=float(t.get("window_seconds", WINDOW_SECONDS)),
                    history_steps=int(t.get("history_steps", HISTORY_STEPS)),
                    forecast_steps=int(t.get("forecast_steps", FORECAST_STEPS)),
                ),
            )
        return cfg

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CyberWorldConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


DEFAULT_CONFIG = CyberWorldConfig()
DEFAULT_CONTRACT = DEFAULT_CONFIG.temporal


def get_contract() -> TemporalContract:
    """The authoritative contract. Prefer this over importing constants."""
    return DEFAULT_CONTRACT
