"""
Training pipeline for Branch B (HostWorldDynamicsTransformer & InfiltrationRiskHead).
Trains multi-step latent rollout on unified host trajectories and benchmarks K-step horizons.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.split_manager import ScientificSplitManager
from data_unification.multi_dataset_stream import HostTrajectoryExtractor, HostWindowSnapshot
from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from cyberworld_v4.config import STATE_DIM, get_contract
from data_unification.trajectory_store import world_state


class HostRolloutDataset(Dataset):
    """
    Dataset yielding (h_history, h_future_targets, risk_targets).
    h_history: [T, d_latent]
    h_future_targets: [K, d_latent]
    risk_targets: [K]
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        out = {
            "h_history": torch.from_numpy(s["h_history"]).float(),
            "h_future": torch.from_numpy(s["h_future"]).float(),
            "risk_future": torch.from_numpy(s["risk_future"]).float(),
        }
        for k in ("t_history", "t_future"):
            if k in s:
                out[k] = torch.from_numpy(s[k]).float()
        return out


def create_rollout_samples(trajectories, T: int = None, K: int = None):
    # T/K were literal 4s, contradicting the v4 contract (history 15, forecast 5)
    # that every other stage and every checkpoint validator uses.
    _c = get_contract()
    T = _c.history_steps if T is None else T
    K = _c.forecast_steps if K is None else K
    samples = []
    for host_ip, snaps in trajectories.items():
        if len(snaps) < T + 1:
            continue
        snaps = sorted(snaps, key=lambda s: s.window_idx)
        n = len(snaps)

        for i in range(T, n):
            h_hist = np.array([world_state(s) for s in snaps[i - T : i]], dtype=np.float32)
            # Future slice up to K
            future_snaps = snaps[i : min(n, i + K)]
            k_avail = len(future_snaps)

            # Pad future if less than K
            h_fut = np.array([world_state(s) for s in future_snaps], dtype=np.float32)
            r_fut = np.array([s.risk_score for s in future_snaps], dtype=np.float32)

            # Real elapsed seconds, origin at the last observed step. See the
            # note in LazyHostRolloutDataset: the gap between a host's
            # consecutive snapshots has median 7 windows on wed_29_csv.csv,
            # not 1, so step index is not time.
            w0 = float(snaps[i - 1].window_idx)
            t_hist = np.array([(s.window_idx - w0) * _c.window_seconds
                               for s in snaps[i - T:i]], dtype=np.float32)
            t_fut = np.array([(s.window_idx - w0) * _c.window_seconds
                              for s in future_snaps], dtype=np.float32)

            if k_avail < K:
                pad_k = K - k_avail
                h_fut = np.pad(h_fut, ((0, pad_k), (0, 0)), mode="edge")
                r_fut = np.pad(r_fut, (0, pad_k), mode="edge")
                t_fut = np.pad(t_fut, (0, pad_k), mode="edge")

            samples.append({
                "h_history": h_hist,
                "h_future": h_fut,
                "risk_future": r_fut,
                "t_history": t_hist,
                "t_future": t_fut,
            })
    return samples


def train_branch_b(
    epochs: int = 4,
    T: int = None,
    K: int = None,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "saved_models/branch_b/host_wdt.pt",
):
    _c = get_contract()
    T = _c.history_steps if T is None else T
    K = _c.forecast_steps if K is None else K
    print("Loading disjoint train/val multi-dataset partitions via ScientificSplitManager for Branch B...")
    sm = ScientificSplitManager()
    # Full density. These were 600/200 records per capture.
    train_records = sm.get_train_records()
    val_records = sm.get_val_records()

    tgn = build_or_load_tgne_ta()
    # Contract-bound. This was hardcoded to 60.0 while the shipped checkpoints
    # and live inference ran at 2.0s, so this trainer could not reproduce them.
    # The window now comes from the single source of truth.
    extractor = HostTrajectoryExtractor(
        tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds
    )
    train_trajectories = extractor.extract_trajectories(train_records)
    val_trajectories = extractor.extract_trajectories(val_records)

    train_samples = create_rollout_samples(train_trajectories, T=T, K=K)
    val_samples = create_rollout_samples(val_trajectories, T=T, K=K)
    print(f"Disjoint rollout samples: Train={len(train_samples)}, Val={len(val_samples)}")

    if len(train_samples) < 10:
        train_samples = train_samples * 5
    if len(val_samples) < 5:
        val_samples = val_samples * 5

    train_set = HostRolloutDataset(train_samples)
    val_set = HostRolloutDataset(val_samples)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_state = int(train_samples[0]["h_history"].shape[-1]) if len(train_samples) else STATE_DIM
    wdt = HostWorldDynamicsTransformer(d_latent=d_state, d_model=64, n_heads=4, n_layers=3).to(device)
    risk_head = InfiltrationRiskHead(d_latent=d_state, hidden_dim=32).to(device)

    params = list(wdt.parameters()) + list(risk_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float("inf")

    print(f"Starting Branch B WDT Rollout training on {device}...")
    gamma = 0.9  # discount factor for horizon loss

    for epoch in range(1, epochs + 1):
        wdt.train()
        risk_head.train()
        train_losses = []

        for batch in train_loader:
            h_hist = batch["h_history"].to(device)
            h_fut = batch["h_future"].to(device)
            r_fut = batch["risk_future"].to(device)
            # Real elapsed times. scripts/retrain_future_models_live.py -- the
            # trainer that produces the served checkpoint -- used to call
            # wdt.rollout(h, K=...) without these, telling the model every
            # step was 2 s apart when the measured median is 14 s. It now
            # passes both, as do DeepOP's rollout paths and both serving paths
            # (pinned by tests/test_rollout_real_times.py).
            t_hist = batch["t_history"].to(device) if "t_history" in batch else None
            t_fut = batch["t_future"].to(device) if "t_future" in batch else None

            optimizer.zero_grad()

            # Autoregressive rollout across K steps
            h_pred = wdt.rollout(h_hist, K=K, t_history=t_hist, t_future=t_fut)

            # Discounted multi-horizon MSE loss
            mse_loss = 0.0
            for k in range(K):
                mse_k = F.mse_loss(h_pred[:, k, :], h_fut[:, k, :])
                mse_loss += (gamma ** k) * mse_k

            # Infiltration risk loss. Was F.binary_cross_entropy: appropriate
            # only if risk_future is a Bernoulli parameter, which it is not
            # for either target this repo uses -- base_severity(tactic) is a
            # tactic-derived scalar, and hazard_risk = exp(-dt/tau) is a
            # deterministic decay of a continuous quantity. Both are
            # regression targets in [0, 1]. BCE and MSE share the same
            # optimum on such a target (BCE is linear in the target, so its
            # minimiser is also the conditional mean -- see
            # InfiltrationRiskHead.risk_loss's docstring for the derivation
            # and the measured numbers), so switching to plain MSE would not
            # have changed anything; what matters is trading the mean-seeking
            # objective for one closer to the MAE this project actually
            # reports and gates on. Measured in
            # tests/test_branch_b_risk_loss.py on a hazard-shaped synthetic
            # target: BCE/MSE-trained heads scored MAE 0.244, worse than
            # predicting zero (0.230); a Huber-trained head, identical data
            # and capacity, scored 0.216 -- the only one of the three to
            # clear that bar.
            pred_step_risks, _ = risk_head.forward_trajectory(h_pred)
            risk_loss = InfiltrationRiskHead.risk_loss(pred_step_risks, r_fut)

            total_loss = mse_loss + risk_loss
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

        # Evaluation
        wdt.eval()
        risk_head.eval()
        val_losses = []
        k1_mses, k_last_mses, persist_mses = [], [], []
        # Baseline-aware risk reporting -- the gap this whole investigation
        # is about. A risk MAE with nothing next to it let the head sit worse
        # than predicting zero for five epochs before anyone noticed (see
        # logs/branch_b_20260922-095949.out). scripts/retrain_future_models_live.py
        # (the trainer that actually produced that run) already reports this;
        # this legacy entry point did not, and is fixed here for the same
        # reason it needed the loss fix above -- it is still runnable
        # (`python -m branch_b_world_model.train_branch_b`) and silently
        # incomplete reporting is how the original defect went unnoticed.
        risk_maes, risk_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                h_hist = batch["h_history"].to(device)
                h_fut = batch["h_future"].to(device)
                r_fut = batch["risk_future"].to(device)
                t_hist = batch["t_history"].to(device) if "t_history" in batch else None
                t_fut = batch["t_future"].to(device) if "t_future" in batch else None

                h_pred = wdt.rollout(h_hist, K=K, t_history=t_hist, t_future=t_fut)
                mse_loss = F.mse_loss(h_pred, h_fut)
                pred_step_risks, _ = risk_head.forward_trajectory(h_pred)
                risk_loss = InfiltrationRiskHead.risk_loss(pred_step_risks, r_fut)
                val_losses.append((mse_loss + risk_loss).item())

                # Compare Horizon 1 MSE vs Persistence Baseline
                k1_mse = F.mse_loss(h_pred[:, 0, :], h_fut[:, 0, :]).item()
                persist_mse = F.mse_loss(h_hist[:, -1, :], h_fut[:, 0, :]).item()
                k_last_mse = F.mse_loss(h_pred[:, -1, :], h_fut[:, -1, :]).item()

                k1_mses.append(k1_mse)
                persist_mses.append(persist_mse)
                k_last_mses.append(k_last_mse)

                risk_maes.append((pred_step_risks - r_fut).abs().mean().item())
                risk_targets.append(r_fut.mean().item())

        mean_val = float(np.mean(val_losses))
        mean_k1 = float(np.mean(k1_mses))
        mean_persist = float(np.mean(persist_mses))
        mean_k_last = float(np.mean(k_last_mses))
        improvement = ((mean_persist - mean_k1) / max(1e-6, mean_persist)) * 100.0

        # mae_predict_zero: the MAE of predicting the constant 0 for every
        # row, i.e. what you get for free with no model at all. Comparable
        # across severity and hazard targets since it is computed from
        # whatever risk_future actually holds, not assumed.
        mean_risk_mae = float(np.mean(risk_maes))
        mean_risk_target = float(np.mean(risk_targets))

        print(
            f"Epoch {epoch:02d} | Val Loss: {mean_val:.4f} | "
            f"1-Step MSE: {mean_k1:.4f} vs Persist: {mean_persist:.4f} (+{improvement:.1f}%) | "
            f"K={K} MSE: {mean_k_last:.4f}"
        )
        print(
            f"  risk: mae_model={mean_risk_mae:.4f} mae_predict_zero={mean_risk_target:.4f}"
            f"{'  <-- WORSE THAN PREDICTING ZERO' if mean_risk_mae >= mean_risk_target else ''}"
        )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "wdt_state_dict": wdt.state_dict(),
                    "risk_head_state_dict": risk_head.state_dict(),
                    "epoch": epoch,
                    "k1_mse": mean_k1,
                    "k_last_mse": mean_k_last,
                    # Without these, validate_checkpoint() can never accept the
                    # file -- it has nothing to compare the contract against.
                    "window_seconds": _c.window_seconds,
                    "window_size_sec": _c.window_seconds,  # v3 key, kept readable
                    "history_steps": T,
                    "forecast_steps": K,
                    "baselines": {
                        "mse_persistence": mean_persist,
                        "risk_mae_zero": mean_risk_target,
                    },
                },
                save_path,
            )

    print(f"Branch B models saved to {save_path}")
    return {
        "best_val_loss": best_val_loss,
        "k1_mse": mean_k1,
        "k_last_mse": mean_k_last,
        "risk_mae": mean_risk_mae,
        "risk_mae_zero": mean_risk_target,
        "save_path": save_path,
    }


if __name__ == "__main__":
    train_branch_b(epochs=3)


class LazyHostRolloutDataset(Dataset):
    """Branch B rollout samples built on demand from a TrajectoryStore.

    `create_rollout_samples` materialises three arrays per sample --
    h_history [T,12], h_future [K,12], risk_future [K] -- for **1,164 bytes
    each**. Branch B draws on the same trajectories as Branch A, roughly 42M
    snapshots at full corpus density:

        10M samples -> 10.8 GiB
        35M samples -> 37.9 GiB

    That does not fit, and thinning the data is not an option. Same fix as
    Branch A: keep two int32 columns (~8 bytes per sample) and gather the
    windows from the store's memmapped feature block on __getitem__.

    Semantics match create_rollout_samples exactly: history is
    `snaps[i-T:i]`, future is `snaps[i:i+K]` edge-padded when the trajectory
    ends early, and only hosts with at least T+1 snapshots contribute.
    """

    def __init__(self, store, T: int = None, K: int = None,
                 emit_times: bool = True, window_seconds: float = None,
                 min_history_steps: int = None):
        _c = get_contract()
        self.T = _c.history_steps if T is None else T
        self.K = _c.forecast_steps if K is None else K
        self.store = store
        # How many REAL history steps a sample needs (default: all T). Serving
        # scores a host from its first window, so a model trained only on
        # full histories meets short ones there first. With fewer than T the
        # history is edge-padded (the first real state repeated, its time
        # repeated too) -- the padding DeepOP's rollout conditioning and
        # serving use.
        self.min_history_steps = self.T if min_history_steps is None else max(1, int(min_history_steps))
        # Emit the REAL elapsed time of each step, not just its ordinal.
        #
        # This dataset indexes a host's snapshots by POSITION in its own
        # trajectory, so `snaps[i:i+K]` is "the next 5 snapshots of this host",
        # which the model and the contract both read as "the next 5 windows =
        # +10 s". Those are not the same thing. Measured on wed_29_csv.csv
        # (the validation capture; 295,483 records, 488,284 snapshots over 350
        # hosts), the window_idx gap between a host's consecutive snapshots is:
        #
        #     median 7   mean 12.06   p90 29   fraction equal to 1: 0.159
        #
        # so the median "10 second" forecast is really 5*7*2 = 70 s, the p90 is
        # 290 s, and the spacing varies by more than an order of magnitude
        # between samples. A host is only present in a window it had traffic
        # in, and most hosts are not busy every 2 seconds.
        #
        # There are two ways to handle that: drop samples whose steps are not
        # contiguous, which throws away data, or tell the model how far apart
        # the steps actually are, which does not. This does the second: 20
        # extra float32 per sample (80 bytes on ~1.2 kB) and the model can
        # condition on real elapsed seconds through ContinuousTimeEncoding.
        self.emit_times = emit_times
        self.window_seconds = (_c.window_seconds if window_seconds is None
                               else float(window_seconds))

        hosts, host_idx, pos = [], [], []
        for h in store:
            rows = store._rows_by_host[h]
            n = len(rows)
            m = self.min_history_steps
            if n < m + 1:
                continue
            hi = len(hosts)
            hosts.append(h)
            host_idx.append(np.full(n - m, hi, dtype=np.int32))
            pos.append(np.arange(m, n, dtype=np.int32))

        self.hosts = hosts
        self._host_idx = np.concatenate(host_idx) if host_idx else np.zeros(0, np.int32)
        self._pos = np.concatenate(pos) if pos else np.zeros(0, np.int32)

    def __len__(self):
        return int(len(self._pos))

    def __getitem__(self, idx):
        host = self.hosts[int(self._host_idx[idx])]
        i = int(self._pos[idx])
        rows = self.store._rows_by_host[host]

        # The full feature block: TGNE embedding AND host attributes.
        # rows[max(0, i-T):i], edge-padded on the left when i < T.
        hist_rows = np.asarray(rows)[np.maximum(np.arange(i - self.T, i), 0)]
        h_hist = self.store.feats[hist_rows]

        fut_rows = rows[i:i + self.K]
        h_fut = self.store.feats[fut_rows]
        # risk_score is its own column -- one fancy-index, not K snapshot
        # builds. `_materialize` would construct K full HostWindowSnapshots
        # per sample and read one float off each.
        r_fut = np.asarray(self.store.risk_score[fut_rows], dtype=np.float32)
        n_fut = len(fut_rows)
        if n_fut < self.K:
            pad = self.K - n_fut
            h_fut = np.pad(h_fut, ((0, pad), (0, 0)), mode="edge")
            r_fut = np.pad(r_fut, (0, pad), mode="edge")

        valid = np.zeros(self.K, dtype=np.float32)
        valid[:n_fut] = 1.0
        out = {
            "h_history": torch.from_numpy(np.ascontiguousarray(h_hist, dtype=np.float32)),
            "h_future": torch.from_numpy(np.ascontiguousarray(h_fut, dtype=np.float32)),
            "risk_future": torch.from_numpy(np.ascontiguousarray(r_fut, dtype=np.float32)),
            # 1 for a real future step, 0 for an edge-padded copy of the last
            # one (trajectory ended). Padded steps are not observations: the
            # losses and metrics mask them, or "the state stops changing"
            # would be trained -- and credited to persistence -- as a fact.
            "future_valid": torch.from_numpy(valid),
        }
        if self.emit_times:
            # Seconds relative to the last OBSERVED step (rows[i-1]), which is
            # the origin HostWorldDynamicsTransformer._elapsed_times uses:
            # history <= 0 with its last entry exactly 0, future > 0.
            w = self.store.window_idx
            w0 = float(w[rows[i - 1]])
            t_hist = (np.asarray(w[hist_rows], dtype=np.float64) - w0)
            t_fut = (np.asarray(w[fut_rows], dtype=np.float64) - w0)
            if n_fut < self.K:
                # Edge-padded future states repeat the last real one, so they
                # repeat its timestamp too -- a padded step is not a step
                # further into the future.
                t_fut = np.pad(t_fut, (0, self.K - n_fut), mode="edge")
            out["t_history"] = torch.from_numpy(
                np.ascontiguousarray(t_hist * self.window_seconds, dtype=np.float32))
            out["t_future"] = torch.from_numpy(
                np.ascontiguousarray(t_fut * self.window_seconds, dtype=np.float32))
        return out

    # -- batched access ------------------------------------------------------
    #
    # A sample reads T history + K future rows of one host. In the store those
    # rows are scattered across the feature block (one disk read per row when
    # the block is a memmap bigger than page cache); in the host-major copy
    # (data_unification/host_major.py) they are one contiguous slice.
    # `gather_batch` returns exactly what default_collate([self[i] ...]) did:
    # tests/test_branch_b_deepop_batched_loader.py.

    def enable_batched(self, host_major=None, spill_dir=None):
        """Precompute what `gather_batch` needs. Call before DataLoader workers fork."""
        from data_unification.host_major import host_major_for
        rows = [self.store._rows_by_host[h] for h in self.hosts]
        self._flat, self._base, self._feats_hm = host_major_for(
            self.store, rows, host_major=host_major, spill_dir=spill_dir)
        self._n_rows = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        self._batched_ready = True
        return self

    def gather_batch(self, indices):
        if not hasattr(self, "_flat"):
            self.enable_batched(host_major=False)
        st, T, K = self.store, self.T, self.K
        idx = np.asarray(indices, dtype=np.int64)
        h = self._host_idx[idx].astype(np.int64)
        i = self._pos[idx].astype(np.int64)
        b = self._base[h]
        last = (b + self._n_rows[h] - 1)[:, None]
        ph = np.maximum((b + i)[:, None] - T + np.arange(T, dtype=np.int64), b[:, None])  # history, edge-padded
        pf = np.minimum((b + i)[:, None] + np.arange(K, dtype=np.int64), last)  # future, edge-padded
        p = np.concatenate([ph, pf], axis=1)
        if self._feats_hm is not None:
            x = np.asarray(self._feats_hm[p])
        else:
            x = np.asarray(st.feats[self._flat[p]])
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows_f = self._flat[pf]
        real = (b + i)[:, None] + np.arange(K, dtype=np.int64) <= last
        out = {
            "h_history": torch.from_numpy(np.ascontiguousarray(x[:, :T])),
            "h_future": torch.from_numpy(np.ascontiguousarray(x[:, T:])),
            "risk_future": torch.from_numpy(
                np.ascontiguousarray(np.asarray(st.risk_score)[rows_f], dtype=np.float32)),
            "future_valid": torch.from_numpy(real.astype(np.float32)),
        }
        if self.emit_times:
            w = np.asarray(st.window_idx)
            w0 = w[self._flat[(b + i - 1)]].astype(np.float64)[:, None]
            t_h = w[self._flat[ph]].astype(np.float64) - w0
            t_f = w[rows_f].astype(np.float64) - w0
            out["t_history"] = torch.from_numpy(
                np.ascontiguousarray(t_h * self.window_seconds, dtype=np.float32))
            out["t_future"] = torch.from_numpy(
                np.ascontiguousarray(t_f * self.window_seconds, dtype=np.float32))
        return out
