"""
Branch A: GNN/LSTM Multi-Task Attack-Sequence Prediction Package.
"""

from branch_a_gnn_lstm.sequence_dataset import (
    HostSequenceDataset,
    create_host_sequence_samples,
    TECHNIQUE_VOCAB,
    TECH_TO_IDX,
    GRADATION_LEVELS,
)
from branch_a_gnn_lstm.attention import TemporalSelfAttention
from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM, MultiTaskUncertaintyLoss

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
