"""
Unified Flow Record Schema.

Defines the standard cross-dataset flow contract for the predictive attack-trajectory pipeline.
All adapters (CIC-IDS2017, CIC-IDS2018, CTU-13, Warden) project their heterogeneous raw records
into this unified format.
"""

import sys as _sys
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional, ClassVar
import pandas as pd
import numpy as np


class LabelSource(str, Enum):
    CIC2017 = "CIC2017"
    CIC2018 = "CIC2018"
    CTU13 = "CTU13"
    WARDEN = "WARDEN"
    SYNTHETIC = "SYNTHETIC"


class CoarseCategory(str, Enum):
    BENIGN = "Benign"
    RECON = "Recon"
    INITIAL_ACCESS = "InitialAccess"
    EXECUTION = "Execution"
    C2 = "C2"
    LATERAL_MOVEMENT = "LateralMovement"
    EXFILTRATION = "Exfiltration"
    IMPACT = "Impact"
    UNKNOWN = "Unknown"


@dataclass(slots=True)
class UnifiedFlowRecord:
    """One normalized flow.

    slots=True and the string interning in __post_init__ are load-bearing at
    corpus scale, not style. Measured on real CTU-13 data: 643 B/record, of
    which ~420 B was duplicated strings and the per-instance __dict__ --
    src_ip/dst_ip repeat across ~22k unique values per 120k references, and
    raw_label_source/coarse_category have 1-2 unique values across the whole
    file. At 37M records (full corpus density) that overhead alone is ~15 GB
    on a 22 GiB machine, which is what made full-density training impossible.
    Interning makes equal strings share one object; slots drops the __dict__.
    """
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # e.g., 6 for TCP, 17 for UDP, 1 for ICMP
    start_time: float  # Normalized to UTC epoch seconds
    end_time: float  # Normalized to UTC epoch seconds
    fwd_bytes: int
    bwd_bytes: int
    fwd_packets: int
    bwd_packets: int
    raw_label: str  # Original label string verbatim
    raw_label_source: str  # e.g. CIC2017, CIC2018, CTU13, WARDEN
    is_attack: bool  # Derived from mapping
    coarse_category: str  # Standard coarse taxonomy
    attck_technique_ids: List[str] = field(default_factory=list)  # MITRE ATT&CK technique IDs (e.g. ['T1046'])
    # None rather than {} : every record allocating an empty dict costs 64 B,
    # and 100% of CSV/netflow-path records never populate it (2.4 GB at 37M).
    # Read through the `meta` property, which always returns a mapping.
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Share one object per distinct string. These fields are drawn from
        # small vocabularies (hosts, labels, sources, categories) but are
        # created fresh per row by the pandas-based adapters.
        self.src_ip = _sys.intern(self.src_ip) if type(self.src_ip) is str else self.src_ip
        self.dst_ip = _sys.intern(self.dst_ip) if type(self.dst_ip) is str else self.dst_ip
        self.raw_label = _sys.intern(self.raw_label) if type(self.raw_label) is str else self.raw_label
        self.raw_label_source = _sys.intern(self.raw_label_source) if type(self.raw_label_source) is str else self.raw_label_source
        self.coarse_category = _sys.intern(self.coarse_category) if type(self.coarse_category) is str else self.coarse_category

        # Validate critical numerical invariants
        if self.start_time > self.end_time:
            # If duration is 0 or end_time < start_time due to precision, align end_time
            self.end_time = self.start_time
        self.fwd_bytes = max(0, int(self.fwd_bytes))
        self.bwd_bytes = max(0, int(self.bwd_bytes))
        self.fwd_packets = max(0, int(self.fwd_packets))
        self.bwd_packets = max(0, int(self.bwd_packets))
        self.src_port = int(self.src_port)
        self.dst_port = int(self.dst_port)
        self.protocol = int(self.protocol)
        if not isinstance(self.attck_technique_ids, list):
            if isinstance(self.attck_technique_ids, str):
                self.attck_technique_ids = [t.strip() for t in self.attck_technique_ids.split(";") if t.strip()]
            else:
                self.attck_technique_ids = []

    _EMPTY_META: ClassVar[Dict[str, Any]] = {}

    @property
    def meta(self) -> Dict[str, Any]:
        """metadata, never None -- read through this instead of .metadata."""
        return self.metadata if self.metadata is not None else self._EMPTY_META

    @property
    def src_host_key(self) -> str:
        """Returns scoped global host identifier preventing IP collision across datasets."""
        scenario = self.meta.get("scenario_id", "default")
        return f"{self.raw_label_source}::{scenario}::{self.src_ip}"

    @property
    def dst_host_key(self) -> str:
        """Returns scoped global host identifier preventing IP collision across datasets."""
        scenario = self.meta.get("scenario_id", "default")
        return f"{self.raw_label_source}::{scenario}::{self.dst_ip}"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes

    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets

    @property
    def byte_rate(self) -> float:
        dur = self.duration
        return (self.total_bytes / dur) if dur > 0.0 else float(self.total_bytes)

    @property
    def packet_rate(self) -> float:
        dur = self.duration
        return (self.total_packets / dur) if dur > 0.0 else float(self.total_packets)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration
        d["total_bytes"] = self.total_bytes
        d["total_packets"] = self.total_packets
        d["attck_technique_ids_str"] = ";".join(self.attck_technique_ids)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedFlowRecord":
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


def records_to_dataframe(records: List[UnifiedFlowRecord]) -> pd.DataFrame:
    """Efficiently convert list of UnifiedFlowRecord into a structured pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    return df
