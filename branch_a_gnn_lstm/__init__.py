"""
Branch A: GNN/LSTM Multi-Task Attack-Sequence Prediction Package.
"""

_import_mapping = {
    "HostSequenceDataset": "branch_a_gnn_lstm.sequence_dataset",
    "create_host_sequence_samples": "branch_a_gnn_lstm.sequence_dataset",
    "TECHNIQUE_VOCAB": "branch_a_gnn_lstm.sequence_dataset",
    "TECH_TO_IDX": "branch_a_gnn_lstm.sequence_dataset",
    "GRADATION_LEVELS": "branch_a_gnn_lstm.sequence_dataset",
    "TemporalSelfAttention": "branch_a_gnn_lstm.attention",
    "MultiTaskLSTM": "branch_a_gnn_lstm.lstm_multitask",
    "MultiTaskUncertaintyLoss": "branch_a_gnn_lstm.lstm_multitask",
}

def __getattr__(name):
    if name in _import_mapping:
        import importlib
        module = importlib.import_module(_import_mapping[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HostSequenceDataset",
    "create_host_sequence_samples",
    "TECHNIQUE_VOCAB",
    "TECH_TO_IDX",
    "GRADATION_LEVELS",
    "TemporalSelfAttention",
    "MultiTaskLSTM",
    "MultiTaskUncertaintyLoss",
]
