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

    def __init__(self, store, seq_len: int = 5, min_trajectory_len: int = 2,
                 max_gap_seconds: "float | None" = None):
        self.store = store
        self.seq_len = seq_len
        #: If set, a host's trajectory is cut wherever two consecutive active
        #: windows are more than this many seconds apart, and no sample's
        #: history or target crosses a cut.
        #:
        #: Rows are a host's ACTIVE windows, not its clock ticks, so "the next
        #: window" is whenever the host next appears -- and that can be far
        #: outside the contract's horizon. Measured on the train split with the
        #: real adapters: consecutive active windows of a CTU-13 host are a
        #: median 736 s apart and over an hour apart in 35% of pairs, against a
        #: 2 s window and a 10 s forecast horizon. Uncapped, "15 steps of
        #: history" can span hours and "the next step" can be an hour ahead.
        #:
        #: None (the default) keeps every sample, as the no-dilution rule asks.
        #: Enabling it removes samples whose target lies beyond a gap, which
        #: can be a large share -- see claude_latest_analysis/30_time_gaps.md.
        self.max_gap_seconds = max_gap_seconds
        self.n_dropped_by_gap = 0
        ws_all = np.asarray(store.window_start) if max_gap_seconds is not None else None

        hosts, host_idx, pos, lo = [], [], [], []
        for h in store:
            rows = store._rows_by_host[h]
            n = len(rows)
            if n < min_trajectory_len:
                continue
            if max_gap_seconds is None:
                # end_idx from 1..n-1, matching create_host_sequence_samples
                ends = np.arange(1, n, dtype=np.int32)
                seg_lo = np.zeros(n - 1, dtype=np.int32)
            else:
                ts = ws_all[np.asarray(rows)]
                cut = np.diff(ts) > float(max_gap_seconds)   # cut[i]: gap between rows i and i+1
                # segment start of every position = the latest cut at or before it
                starts = np.zeros(n, dtype=np.int64)
                starts[1:] = np.where(cut, np.arange(1, n), 0)
                starts = np.maximum.accumulate(starts)
                keep = starts[1:] < np.arange(1, n)          # a target may not open a segment
                ends = np.arange(1, n, dtype=np.int32)[keep]
                seg_lo = starts[1:][keep].astype(np.int32)
                self.n_dropped_by_gap += int((~keep).sum())
            if len(ends) == 0:
                continue
            hi = len(hosts)
            hosts.append(h)
            host_idx.append(np.full(len(ends), hi, dtype=np.int32))
            pos.append(ends)
            lo.append(seg_lo)

        self.hosts = hosts
        # Row arrays already exist in the store; holding references to them
        # here costs nothing and saves a dict hash on every __getitem__.
        self._rows = [store._rows_by_host[h] for h in hosts]
        self._host_idx = np.concatenate(host_idx) if host_idx else np.zeros(0, np.int32)
        self._pos = np.concatenate(pos) if pos else np.zeros(0, np.int32)
        #: earliest row a sample's history may reach back to (its segment start)
        self._lo = np.concatenate(lo) if lo else np.zeros(0, np.int32)

    def __len__(self) -> int:
        return int(len(self._pos))

    def __getitem__(self, idx: int):
        h = int(self._host_idx[idx])
        host = self.hosts[h]
        end = int(self._pos[idx])
        rows = self._rows[h]
        start = max(int(self._lo[idx]), end - self.seq_len)

        feats = self.store.feats[rows[start:end]]          # (<=seq_len, 27)
        if len(feats) < self.seq_len:
            pad = np.zeros((self.seq_len - len(feats), feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([pad, feats], axis=0)

        # Four scalars, read straight off the columns. `_materialize` would
        # build a whole HostWindowSnapshot -- including both feature slices --
        # to have all but four of its fields discarded here, at 87x the cost.
        risk, category, first_tech, window_idx = self.store.target_fields(int(rows[end]))
        tech = first_tech if first_tech is not None else "Benign"
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "risk": torch.tensor(risk, dtype=torch.float),
            "technique": torch.tensor(
                TECH_TO_IDX.get(tech, TECH_TO_IDX.get("Benign", 0)), dtype=torch.long),
            "gradation": torch.tensor(
                GRADATION_LEVELS.get(category, 0), dtype=torch.long),
            "host_ip": host,
            "window_idx": window_idx,
        }
