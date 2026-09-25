"""
cyberworld_v4.identity — namespaced, stable host identity.

Two defects this replaces (spec 18, 19):

1. Trajectories were grouped by raw IP, so 192.168.1.10 in CIC-2018 and the
   same address in CTU-13 became one host. Private ranges recur across every
   capture; they are not the same machine.

2. Python's built-in hash() was used to derive ids. It is salted per process
   (PYTHONHASHSEED), so ids are not reproducible across runs — unusable as a
   scientific identifier.

Identity here is the tuple (dataset, scenario, capture, host), rendered as a
readable string and, where an integer is required, as a stable SHA-256 digest.

Collapsing to a bare address is legitimate only inside a single live network
namespace, which `live_host` makes explicit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

LIVE_DATASET = "live"
LIVE_SCENARIO = "live"
LIVE_CAPTURE = "live"


def _clean(part: Optional[str]) -> str:
    if part is None:
        return "-"
    s = str(part).strip().replace("|", "_")
    return s or "-"


@dataclass(frozen=True, order=True)
class HostId:
    """A host, unambiguous across datasets and captures."""

    dataset: str
    scenario: str
    capture: str
    host: str

    def __post_init__(self) -> None:
        for f in ("dataset", "scenario", "capture", "host"):
            object.__setattr__(self, f, _clean(getattr(self, f)))

    @property
    def key(self) -> str:
        """Canonical string form. Stable across processes and runs."""
        return f"{self.dataset}|{self.scenario}|{self.capture}|{self.host}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()

    def node_index(self, modulo: Optional[int] = None) -> int:
        """Deterministic integer id. Never Python hash()."""
        v = int.from_bytes(hashlib.sha256(self.key.encode("utf-8")).digest()[:8], "big")
        return v % modulo if modulo else v

    @property
    def is_live(self) -> bool:
        return self.dataset == LIVE_DATASET

    def __str__(self) -> str:
        return self.key

    @classmethod
    def parse(cls, key: str) -> "HostId":
        parts = key.split("|")
        if len(parts) != 4:
            raise ValueError(f"not a HostId key: {key!r}")
        return cls(*parts)


def offline_host(dataset: str, scenario: str, capture: str, host: str) -> HostId:
    """Identity for a host observed in a stored capture."""
    return HostId(dataset=dataset, scenario=scenario, capture=capture, host=host)


def live_host(address: str) -> HostId:
    """Identity for a host on the live sensor.

    The only context where an address alone identifies a machine, because there
    is exactly one network namespace.
    """
    return HostId(
        dataset=LIVE_DATASET, scenario=LIVE_SCENARIO, capture=LIVE_CAPTURE, host=address
    )


def stable_id(*parts: str, modulo: Optional[int] = None) -> int:
    """Reproducible integer from arbitrary parts. Drop-in for hash()."""
    raw = "|".join(_clean(p) for p in parts)
    v = int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big")
    return v % modulo if modulo else v
