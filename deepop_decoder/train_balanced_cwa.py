import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.split_manager import ScientificSplitManager
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
from deepop_decoder.train_cwa_decoder import create_cwa_training_samples, CWASequenceDataset


def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = get_joint_vocab()

    print("Loading disjoint train/val telemetry records for CWA Decoder via ScientificSplitManager...")
    sm = ScientificSplitManager()
    train_records = sm.get_train_records(max_per_source=600)
    val_records = sm.get_val_records(max_per_source=200)

    tgn = build_or_load_tgne_ta()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=60.0)
    train_trajectories = extractor.extract_trajectories(train_records)
    val_trajectories = extractor.extract_trajectories(val_records)

    train_samples = create_cwa_training_samples(train_trajectories, vocab=vocab, K=4)
    val_samples = create_cwa_training_samples(val_trajectories, vocab=vocab, K=4)
    print(f"Disjoint CWA sequence samples: Train={len(train_samples)}, Val={len(val_samples)}")

    # Calculate class frequencies for inverse weighting
    all_targets = np.array([s["target_tokens"] for s in train_samples]).reshape(-1)
    counts = np.bincount(all_targets, minlength=vocab.vocab_size)
    weights = np.zeros(vocab.vocab_size, dtype=np.float32)
    for i in range(vocab.vocab_size):
        if counts[i] > 0:
            weights[i] = 1.0 / (counts[i] ** 0.5)

    weights[vocab.pad_idx] = 0.0
    benign_idx = vocab.encode("Benign", None)
    weights[benign_idx] = weights[benign_idx] * 0.2
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    train_set = CWASequenceDataset(train_samples)
    val_set = CWASequenceDataset(val_samples)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    decoder = DeepOPForecastDecoder(d_latent=12, d_model=72, vocab_size=vocab.vocab_size, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=2e-3, weight_decay=1e-4)

    print("Training DeepOP with Class-Weighted Loss & Autoregressive Scheduled Sampling...")
    for epoch in range(1, 10):
        decoder.train()
        losses = []
        for batch in train_loader:
            h_fut = batch["h_future"].to(device)
            tgt_tok = batch["target_tokens"].to(device)
            B, K = tgt_tok.shape

            # Scheduled sampling: with 50% probability, rollout free-running
            if np.random.rand() > 0.4:
                curr = torch.full((B, 1), vocab.bos_idx, dtype=torch.long, device=device)
                logits_list = []
                for t in range(K):
                    out = decoder(h_fut, curr)
                    next_logit = out[:, -1:, :]
                    logits_list.append(next_logit)
                    next_tok = next_logit.argmax(dim=-1)
                    curr = torch.cat([curr, next_tok], dim=1)
                logits = torch.cat(logits_list, dim=1)
            else:
                inp_tok = batch["input_tokens"].to(device)
                logits = decoder(h_fut, inp_tok)

            loss = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), tgt_tok.reshape(-1), weight=weights_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:02d} | Weighted Loss: {np.mean(losses):.4f}")

    # Evaluate free-running on validation set
    decoder.eval()
    val_h = torch.tensor(np.array([val_set[i]["h_future"].numpy() for i in range(len(val_set))]), dtype=torch.float32).to(device)
    val_tgt = torch.tensor(np.array([val_set[i]["target_tokens"].numpy() for i in range(len(val_set))]), dtype=torch.long).to(device)
    val_inp = torch.tensor(np.array([val_set[i]["input_tokens"].numpy() for i in range(len(val_set))]), dtype=torch.long).to(device)

    with torch.no_grad():
        val_pred, _ = decoder.forecast_sequence(val_h, max_steps=4)
        tf_logits = decoder(val_h, val_inp)
        tf_pred = tf_logits.argmax(dim=-1)

    y_true = val_tgt.cpu().numpy().reshape(-1)
    y_pred = val_pred.cpu().numpy().reshape(-1)
    y_tf = tf_pred.cpu().numpy().reshape(-1)

    # Baselines
    vals, c_counts = np.unique(y_true, return_counts=True)
    maj_token = vals[np.argmax(c_counts)]
    y_maj = np.full_like(y_true, maj_token)
    y_pers = val_inp[:, 0].unsqueeze(1).repeat(1, 4).cpu().numpy().reshape(-1)

    free_acc = accuracy_score(y_true, y_pred)
    tf_acc = accuracy_score(y_true, y_tf)
    maj_acc = accuracy_score(y_true, y_maj)
    pers_acc = accuracy_score(y_true, y_pers)

    all_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    act_mask = (y_true != vocab.pad_idx) & (y_true != benign_idx)
    act_macro = f1_score(y_true[act_mask], y_pred[act_mask], average="macro", zero_division=0) if act_mask.sum() > 0 else 0.0

    print("\n" + "=" * 60)
    print("DEEPOOP FORECASTING EVALUATION RESULTS")
    print("=" * 60)
    print(f"DeepOP Teacher-Forced Accuracy:   {tf_acc * 100:.2f}%")
    print(f"DeepOP Free-Running Accuracy:     {free_acc * 100:.2f}%")
    print(f"Majority-Token Baseline Accuracy: {maj_acc * 100:.2f}% [Predicting '{vocab.decode(maj_token)}']")
    print(f"Persistence Baseline Accuracy:    {pers_acc * 100:.2f}%")
    print(f"Free-Running Macro-F1 (All):      {all_macro:.4f}")
    print(f"Active-Technique Macro-F1:        {act_macro:.4f}")

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [f"{vocab.decode(l)[0]}_{vocab.decode(l)[1]}" for l in unique_labels]

    print("\n" + "=" * 60)
    print("PER-TECHNIQUE CLASSIFICATION REPORT (FREE-RUNNING)")
    print("=" * 60)
    report_dict = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))

    # Save improved checkpoint
    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "accuracy": tf_acc,
            "free_running_accuracy": free_acc,
            "free_running_macro_f1": all_macro,
            "active_technique_macro_f1": act_macro,
            "vocab_size": vocab.vocab_size,
        },
        "saved_models/deepop/cwa_forecast_decoder.pt",
    )
    print("Updated checkpoint saved to saved_models/deepop/cwa_forecast_decoder.pt")


if __name__ == "__main__":
    run_experiment()
