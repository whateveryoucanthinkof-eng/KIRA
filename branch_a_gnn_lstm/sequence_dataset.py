"""
Sequence Dataset for Branch A (GNN/LSTM Attack-Sequence Prediction).

Constructs per-host sequence samples x_t = [H_t[v]; temporal_attrs_t[v]]
paired with multi-task targets: risk score, technique class, and attack gradation.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_unification.multi_dataset_stream import HostWindowSnapshot


# Unified Technique Vocabulary for Classification Head
TECHNIQUE_VOCAB = [
    "Benign",
    "T1046",       # Network Service Discovery
    "T1595",       # Active Scanning
    "T1110",       # Brute Force
    "T1190",       # Exploit Public-Facing Application
    "T1189",       # Drive-by Compromise
    "T1071",       # Application Layer Protocol (C2)
    "T1071.001",   # Web / IRC Protocols
    "T1568.001",   # Fast Flux DNS
    "T1204",       # User Execution
    "T1005",       # Data from Local System / Heartbleed
    "T1498",       # Network Denial of Service
    "T1498.001",   # Direct Network Flood
    "T1020",       # Automated Exfiltration
]
TECH_TO_IDX = {t: i for i, t in enumerate(TECHNIQUE_VOCAB)}

GRADATION_LEVELS = {
    "Benign": 0,
    "Recon": 1,
    "InitialAccess": 2,
    "Execution": 2,
    "C2": 3,
    "LateralMovement": 3,
    "Exfiltration": 3,
    "Impact": 3,
    "Unknown": 1,
}


class HostSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding (sequence_features, mask, risk_target, tech_target, grad_target).
    Sequence features shape: [seq_len, feature_dim] where feature_dim = d_emb + 15.
    """

    def __init__(
        self,
        samples: List[Dict[str, np.ndarray]],
        seq_len: int = 5,
    ):
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "features": torch.from_numpy(s["features"]).float(),        # [seq_len, dim]
            "risk": torch.tensor(s["risk"], dtype=torch.float),        # scalar in [0, 1]
            "technique": torch.tensor(s["technique"], dtype=torch.long), # class index
            "gradation": torch.tensor(s["gradation"], dtype=torch.long), # severity 0..3
            "host_ip": s.get("host_ip", ""),
            "window_idx": s.get("window_idx", 0),
        }


def create_host_sequence_samples(
    trajectories_by_host: Dict[str, List[HostWindowSnapshot]],
    seq_len: int = 5,
    min_trajectory_len: int = 2,
) -> List[Dict[str, np.ndarray]]:
    """
    Extracts sliding window sequence samples of length seq_len from per-host trajectories.
    Pads shorter sequences with initial zero/replicated states.
    """
    samples = []

    for host_ip, snapshots in trajectories_by_host.items():
        if len(snapshots) < min_trajectory_len:
            continue

        # Sort snapshots by window index
        snapshots = sorted(snapshots, key=lambda s: s.window_idx)
        n_snaps = len(snapshots)

        for end_idx in range(1, n_snaps + 1):
            start_idx = max(0, end_idx - seq_len)
            window_slice = snapshots[start_idx:end_idx]

            feature_vectors = []
            for snap in window_slice:
                # x_t = concat(H_t[v], attrs_t)
                x_t = np.concatenate([snap.embedding, snap.temporal_attrs])
                feature_vectors.append(x_t)

            feat_dim = feature_vectors[0].shape[0]

            # Left-pad if sequence is shorter than seq_len
            if len(feature_vectors) < seq_len:
                pad_len = seq_len - len(feature_vectors)
                padding = [np.zeros(feat_dim, dtype=np.float32) for _ in range(pad_len)]
                feature_seq = np.array(padding + feature_vectors, dtype=np.float32)
            else:
                feature_seq = np.array(feature_vectors[-seq_len:], dtype=np.float32)

            target_snap = window_slice[-1]

            # Technique index
            tech_id = "Benign"
            if target_snap.technique_ids:
                tech_id = target_snap.technique_ids[0]
            tech_idx = TECH_TO_IDX.get(tech_id, TECH_TO_IDX.get("Benign", 0))

            # Gradation index
            grad_idx = GRADATION_LEVELS.get(target_snap.coarse_category, 0)

            sample = {
                "features": feature_seq,
                "risk": target_snap.risk_score,
                "technique": tech_idx,
                "gradation": grad_idx,
                "host_ip": host_ip,
                "window_idx": target_snap.window_idx,
            }
            samples.append(sample)

    return samples
