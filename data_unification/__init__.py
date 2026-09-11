"""
Data Unification Package.

Provides unified schema, label resolution, and dataset adapters for
CIC-IDS2017, CIC-IDS2018, CTU-13, and Warden.
"""

from data_unification.unified_schema import (
    UnifiedFlowRecord,
    LabelSource,
    CoarseCategory,
    records_to_dataframe,
)
from data_unification.label_resolver import LabelResolver, get_default_resolver
from data_unification.cic2017_adapter import CIC2017Adapter
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.warden_adapter import WardenAdapter

from data_unification.auth_log_adapter import (
    AuthEventRecord,
    AuthLogAdapter,
    AuthEventType,
    AuthLogSource,
)
from data_unification.multi_dataset_stream import BoundedHostSlidingBuffer
from data_unification.behavioral_fingerprint import (
    BehavioralFlowFingerprinter,
    BehavioralProfile,
    FlowBehavioralResult,
)

__all__ = [
    "UnifiedFlowRecord",
    "LabelSource",
    "CoarseCategory",
    "records_to_dataframe",
    "LabelResolver",
    "get_default_resolver",
    "CIC2017Adapter",
    "CIC2018Adapter",
    "CTU13Adapter",
    "WardenAdapter",
    "AuthEventRecord",
    "AuthLogAdapter",
    "AuthEventType",
    "AuthLogSource",
    "BoundedHostSlidingBuffer",
    "BehavioralFlowFingerprinter",
    "BehavioralProfile",
    "FlowBehavioralResult",
]
