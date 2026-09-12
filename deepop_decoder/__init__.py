"""
DeepOP-Style ATT&CK CWA Forecasting Decoder Package.
"""

_import_mapping = {
    "JointAttackVocab": "deepop_decoder.joint_vocab",
    "get_joint_vocab": "deepop_decoder.joint_vocab",
    "PAD_TOKEN": "deepop_decoder.joint_vocab",
    "BOS_TOKEN": "deepop_decoder.joint_vocab",
    "EOS_TOKEN": "deepop_decoder.joint_vocab",
    "CausalWindowAttention": "deepop_decoder.cwa",
    "CWADecoderLayer": "deepop_decoder.forecast_decoder",
    "DeepOPForecastDecoder": "deepop_decoder.forecast_decoder",
}

def __getattr__(name):
    if name in _import_mapping:
        import importlib
        module = importlib.import_module(_import_mapping[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "JointAttackVocab",
    "get_joint_vocab",
    "PAD_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "CausalWindowAttention",
    "CWADecoderLayer",
    "DeepOPForecastDecoder",
]
