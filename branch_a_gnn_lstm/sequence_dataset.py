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
    # label_resolver emits the upper-case "UNKNOWN" (label_resolver.py:14); without
    # this key an unresolved label fell through .get(..., 0) and was silently
    # graded Benign, which is the opposite of unknown.
    "UNKNOWN": 1,
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

        # end_idx stops one short of n_snaps so snapshots[end_idx] is always a
        # real future snapshot, never the window's own last input step (nowcast bug).
        for end_idx in range(1, n_snaps):
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

            target_snap = snapshots[end_idx]

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


class LazyHostSequenceDataset(Dataset):
    """Builds each [seq_len, 27] window on demand instead of materialising all.

    ## Why this exists

    `create_host_sequence_samples` materialises a `[seq_len, 27]` float32 array
    per sample. Measured: **2,053 bytes per sample**, of which 1,620 is the
    window itself. At full corpus density Branch A produces roughly 42M
    samples (measured snapshot ratios: 1.99 per record for CIC-2018, 0.72 for
    CTU-13):

        8.5M samples ->  16.3 GiB
         42M samples ->  80.3 GiB

    That does not fit, and thinning the data is not an option. But nothing
    needs those arrays to exist: a window is just `seq_len` consecutive rows
    of a host's trajectory, and `TrajectoryStore` already holds the features
    in one memmapped block. Storing two int32 columns instead of the arrays
    costs ~8 bytes per sample -- **336 MB instead of 80 GB** -- and the window
    is gathered on `__getitem__`, which is what a DataLoader worker is for.

    Semantics are identical to `create_host_sequence_samples`: the window is
    `snapshots[max(0, end-seq_len) : end]`, left-padded with zeros when short,
    and the target is `snapshots[end]` -- strictly a future snapshot, never
    the last input step.
    """

    def __init__(self, store, seq_len: int = 5, min_trajectory_len: int = 2):
        self.store = store
        self.seq_len = seq_len

        hosts, host_idx, pos = [], [], []
        for h in store:
            rows = store._rows_by_host[h]
            n = len(rows)
            if n < min_trajectory_len:
                continue
            hi = len(hosts)
            hosts.append(h)
            # end_idx from 1..n-1, matching create_host_sequence_samples
            host_idx.append(np.full(n - 1, hi, dtype=np.int32))
            pos.append(np.arange(1, n, dtype=np.int32))

        self.hosts = hosts
        self._host_idx = np.concatenate(host_idx) if host_idx else np.zeros(0, np.int32)
        self._pos = np.concatenate(pos) if pos else np.zeros(0, np.int32)

    def __len__(self) -> int:
        return int(len(self._pos))

    def __getitem__(self, idx: int):
        host = self.hosts[int(self._host_idx[idx])]
        end = int(self._pos[idx])
        rows = self.store._rows_by_host[host]
        start = max(0, end - self.seq_len)

        feats = self.store.feats[rows[start:end]]          # (<=seq_len, 27)
        if len(feats) < self.seq_len:
            pad = np.zeros((self.seq_len - len(feats), feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([pad, feats], axis=0)

        target = self.store._materialize(int(rows[end]))
        tech = target.technique_ids[0] if target.technique_ids else "Benign"
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "risk": torch.tensor(target.risk_score, dtype=torch.float),
            "technique": torch.tensor(
                TECH_TO_IDX.get(tech, TECH_TO_IDX.get("Benign", 0)), dtype=torch.long),
            "gradation": torch.tensor(
                GRADATION_LEVELS.get(target.coarse_category, 0), dtype=torch.long),
            "host_ip": host,
            "window_idx": int(target.window_idx),
        }
