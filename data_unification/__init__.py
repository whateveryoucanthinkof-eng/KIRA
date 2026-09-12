"""
Data Unification Package.

Provides unified schema, label resolution, and dataset adapters for
CIC-IDS2017, CIC-IDS2018, CTU-13, and Warden.
"""

_import_mapping = {
    "UnifiedFlowRecord": "data_unification.unified_schema",
    "LabelSource": "data_unification.unified_schema",
    "CoarseCategory": "data_unification.unified_schema",
    "records_to_dataframe": "data_unification.unified_schema",
    "LabelResolver": "data_unification.label_resolver",
    "get_default_resolver": "data_unification.label_resolver",
    "CIC2017Adapter": "data_unification.cic2017_adapter",
    "CIC2018Adapter": "data_unification.cic2018_adapter",
    "CTU13Adapter": "data_unification.ctu13_adapter",
    "WardenAdapter": "data_unification.warden_adapter",
    "AuthEventRecord": "data_unification.auth_log_adapter",
    "AuthLogAdapter": "data_unification.auth_log_adapter",
    "AuthEventType": "data_unification.auth_log_adapter",
    "AuthLogSource": "data_unification.auth_log_adapter",
    "BoundedHostSlidingBuffer": "data_unification.multi_dataset_stream",
    "BehavioralFlowFingerprinter": "data_unification.behavioral_fingerprint",
    "BehavioralProfile": "data_unification.behavioral_fingerprint",
    "FlowBehavioralResult": "data_unification.behavioral_fingerprint",
}

def __getattr__(name):
    if name in _import_mapping:
        import importlib
        module = importlib.import_module(_import_mapping[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
