"""
Training pipeline for DeepOP-Style ATT&CK CWA Forecasting Decoder.
Conditioned on predicted future latent host states H_hat_{t+1..t+K}.
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

from branch_a_gnn_lstm.train_branch_a import load_sample_multi_dataset_records, build_or_load_tgne_ta
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.forecast_decoder import DeepOPForecastDecoder


class CWASequenceDataset(Dataset):
    """
    Dataset yielding:
    h_future: [K, d_latent]
    input_tokens: [K] (<BOS> + y_{1..K-1})
    target_tokens: [K] (y_{1..K})
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_future": torch.from_numpy(s["h_future"]).float(),
            "input_tokens": torch.from_numpy(s["input_tokens"]).long(),
            "target_tokens": torch.from_numpy(s["target_tokens"]).long(),
            "obs_token": torch.tensor(s["obs_token"], dtype=torch.long),
        }


def create_cwa_training_samples(trajectories, vocab, K: int = 4):
    samples = []
    for host_ip, snaps in trajectories.items():
        if len(snaps) < K:
            continue
        snaps = sorted(snaps, key=lambda s: s.window_idx)
        n = len(snaps)

        for i in range(0, n - K + 1):
            future_snaps = snaps[i : i + K]
            h_fut = np.array([s.embedding for s in future_snaps], dtype=np.float32)

            token_ids = []
            for s in future_snaps:
                t_id = s.technique_ids[0] if s.technique_ids else "None"
                token_ids.append(vocab.encode(s.coarse_category, t_id))

            # Preceding observed token
            obs_tok = vocab.bos_idx
            if i > 0:
                prev_s = snaps[i - 1]
                p_id = prev_s.technique_ids[0] if prev_s.technique_ids else "None"
                obs_tok = vocab.encode(prev_s.coarse_category, p_id)

            # Input: <BOS> + first K-1 tokens
            input_seq = [vocab.bos_idx] + token_ids[:-1]
            target_seq = token_ids

            sample_item = {
                "h_future": h_fut,
                "input_tokens": np.array(input_seq, dtype=int),
                "target_tokens": np.array(target_seq, dtype=int),
                "obs_token": int(obs_tok),
            }
            samples.append(sample_item)

            # Duplicate active attack sequences (2x) to prevent quiescent mode collapse
            benign_idx = vocab.encode("Benign", None)
            if any(t != benign_idx and t != vocab.pad_idx for t in target_seq):
                samples.append(sample_item)

    return samples


def train_cwa_decoder(
    epochs: int = 6,
    K: int = 4,
    batch_size: int = 32,
    lr: float = 5e-4,
    save_path: str = "saved_models/deepop/cwa_forecast_decoder.pt",
    max_per_source: int = 1000,
):
    from data_unification.split_manager import get_split_manager
    print("Loading multi-dataset records for DeepOP CWA Decoder from ScientificSplitManager...")
    sm = get_split_manager()
    train_records = sm.get_train_records(max_per_source=max_per_source)
    val_records = sm.get_val_records(max_per_source=max(50, max_per_source // 2))

    tgn = build_or_load_tgne_ta()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=60.0)
    train_trajectories = extractor.extract_trajectories(train_records)
    val_trajectories = extractor.extract_trajectories(val_records)

    vocab = get_joint_vocab(network_observable_only=True)
    train_samples = create_cwa_training_samples(train_trajectories, vocab=vocab, K=K)
    val_samples = create_cwa_training_samples(val_trajectories, vocab=vocab, K=K)
    print(f"Disjoint CWA sequence samples - Train: {len(train_samples)}, Val: {len(val_samples)} (Vocab size: {vocab.vocab_size})")

    if len(train_samples) < 20:
        train_samples = train_samples * 5
    if len(val_samples) < 10:
        val_samples = val_samples * 5

    # Compute smooth inverse-frequency class weights
    all_targets = np.concatenate([s["target_tokens"] for s in train_samples])
    counts = np.bincount(all_targets, minlength=vocab.vocab_size)
    valid_counts = counts[counts > 0]
    median_c = float(np.median(valid_counts)) if len(valid_counts) > 0 else 1.0
    weights = np.zeros(vocab.vocab_size, dtype=np.float32)
    for i in range(vocab.vocab_size):
        if counts[i] > 0:
            weights[i] = float(median_c / (counts[i] ** 0.30 + 1.0))
    weights[vocab.pad_idx] = 0.0
    active_weights = weights[weights > 0]
    if len(active_weights) > 0:
        weights = weights / np.mean(active_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    train_set = CWASequenceDataset(train_samples)
    val_set = CWASequenceDataset(val_samples)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    decoder = DeepOPForecastDecoder(
        d_latent=12,
        d_model=72,
        vocab_size=vocab.vocab_size,
        n_heads=6,
        num_layers=2,
        window_sizes=[2, 4, 8],
        dim_feedforward=144,
    ).to(device)

    # Initialize prototypes from training class centroids
    centroids = torch.zeros(vocab.vocab_size, 12)
    counts_c = torch.zeros(vocab.vocab_size)
    for s in train_samples:
        h_arr = torch.from_numpy(s["h_future"])
        t_arr = torch.from_numpy(s["target_tokens"])
        for k in range(len(t_arr)):
            tok = int(t_arr[k].item())
            centroids[tok] += h_arr[k]
            counts_c[tok] += 1

    for c in range(vocab.vocab_size):
        if counts_c[c] > 0:
            centroids[c] = centroids[c] / counts_c[c]

    decoder.prototypes.data.copy_(centroids.to(device))

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_f1 = 0.0

    print(f"Starting DeepOP CWA Decoder training on {device} with Smooth-Weighted Cross-Entropy...")
    for epoch in range(1, epochs + 1):
        decoder.train()
        train_losses = []

        for batch in train_loader:
            h_fut = batch["h_future"].to(device)
            inp_tok = batch["input_tokens"].to(device)
            tgt_tok = batch["target_tokens"].to(device)

            # Rollout noise augmentation to prevent train-test covariate shift on noisy WDT outputs (Issue 7)
            # WDT rollout error compounds with horizon step k: sigma_k grows from 0.015 at k=1 to 0.055 at k=K
            B_curr, K_curr, _ = h_fut.shape
            step_sigma = torch.linspace(0.015, 0.055, steps=K_curr, device=device).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
            wdt_noise = torch.randn_like(h_fut) * step_sigma
            h_fut_augmented = h_fut + wdt_noise

            optimizer.zero_grad()
            logits = decoder(h_fut_augmented, inp_tok)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab.vocab_size),
                tgt_tok.reshape(-1),
                weight=weights_t,
                label_smoothing=0.04,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation evaluation
        decoder.eval()
        val_losses = []
        val_h_list = []
        val_tgt_list = []
        val_inp_list = []
        val_obs_list = []

        with torch.no_grad():
            for batch in val_loader:
                h_fut = batch["h_future"].to(device)
                inp_tok = batch["input_tokens"].to(device)
                tgt_tok = batch["target_tokens"].to(device)
                obs_tok = batch["obs_token"].to(device)

                logits = decoder(h_fut, inp_tok)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab.vocab_size),
                    tgt_tok.reshape(-1),
                    weight=weights_t,
                    label_smoothing=0.04,
                )
                val_losses.append(loss.item())
                val_h_list.append(h_fut)
                val_tgt_list.append(tgt_tok)
                val_inp_list.append(inp_tok)
                val_obs_list.append(obs_tok)

        val_h_cat = torch.cat(val_h_list, dim=0)
        val_tgt_cat = torch.cat(val_tgt_list, dim=0)
        val_inp_cat = torch.cat(val_inp_list, dim=0)
        val_obs_cat = torch.cat(val_obs_list, dim=0)

        # Measure free-running performance on validation set
        with torch.no_grad():
            val_free_pred, _ = decoder.forecast_sequence(
                val_h_cat, max_steps=K, observed_token=val_obs_cat, repetition_penalty=1.0
            )

        y_true = val_tgt_cat.cpu().numpy().reshape(-1)
        y_pred = val_free_pred.cpu().numpy().reshape(-1)

        from sklearn.metrics import accuracy_score, f1_score
        val_acc = float(accuracy_score(y_true, y_pred))
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Active Macro-F1 (excluding PAD and Benign)
        pad_idx = vocab.pad_idx
        benign_idx = vocab.encode("Benign", None)
        active_mask = (y_true != pad_idx) & (y_true != benign_idx)
        if active_mask.sum() > 0:
            val_active_f1 = float(
                f1_score(y_true[active_mask], y_pred[active_mask], average="macro", zero_division=0)
            )
        else:
            val_active_f1 = val_macro_f1

        mean_val_loss = float(np.mean(val_losses))

        # Test sample sequence
        sample_h = val_set[0]["h_future"].unsqueeze(0).to(device)
        sample_obs = val_set[0]["obs_token"].unsqueeze(0).to(device)
        _, decoded_seq = decoder.forecast_sequence(
            sample_h, max_steps=K, observed_token=sample_obs, repetition_penalty=1.0
        )

        print(
            f"Epoch {epoch:02d} | Train: {np.mean(train_losses):.4f} | "
            f"Val Loss: {mean_val_loss:.4f} | Free Acc: {val_acc*100:.1f}% | "
            f"Macro-F1: {val_macro_f1:.4f} | Active-F1: {val_active_f1:.4f} | "
            f"Sample: {decoded_seq[0]}"
        )

        score = val_active_f1 * 0.7 + val_acc * 0.3
        if score > best_val_f1 or (val_acc > 0.75 and val_active_f1 > 0.5) or epoch == 1:
            best_val_f1 = max(score, best_val_f1)
            torch.save(
                {
                    "decoder_state_dict": decoder.state_dict(),
                    "epoch": epoch,
                    "accuracy": val_acc,
                    "macro_f1": val_macro_f1,
                    "active_macro_f1": val_active_f1,
                    "vocab_size": vocab.vocab_size,
                    "network_observable_only": True,
                },
                save_path,
            )

    print(f"DeepOP CWA Decoder saved to {save_path}")
    return {
        "best_score": best_val_f1,
        "save_path": save_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DeepOP CWA Decoder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--K", type=int, default=4, help="Forecast horizon steps")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test with small sample count")
    args = parser.parse_args()

    if args.smoke_test:
        print("[*] Running DeepOP CWA Decoder Smoke Test (1 epoch)...")
        train_cwa_decoder(epochs=1, K=args.K, batch_size=16, max_per_source=100)
        print("[+] DeepOP CWA Decoder Smoke Test Succeeded!")
    else:
        train_cwa_decoder(epochs=args.epochs, K=args.K)

