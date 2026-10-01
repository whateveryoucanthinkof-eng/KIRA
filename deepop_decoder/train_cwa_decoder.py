"""
Training pipeline for DeepOP-Style ATT&CK CWA Forecasting Decoder.
Conditioned on predicted future latent host states H_hat_{t+1..t+K}.
"""

import os
import sys
import numpy as np
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from branch_a_gnn_lstm.train_branch_a import load_sample_multi_dataset_records, build_or_load_tgne_ta
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.trajectory_store import world_state
from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.forecast_decoder import (
    DeepOPForecastDecoder, DeepOPTokenScorer, observed_sequence_tokens, smoothed_and_plain_ce,
)
from cyberworld_v4.config import get_contract


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
            **({"h_history": torch.from_numpy(s["h_history"]).float()} if "h_history" in s else {}),
            "input_tokens": torch.from_numpy(s["input_tokens"]).long(),
            "target_tokens": torch.from_numpy(s["target_tokens"]).long(),
            "obs_token": torch.tensor(s["obs_token"], dtype=torch.long),
            **({"obs_tokens": torch.from_numpy(s["obs_tokens"]).long()} if "obs_tokens" in s else {}),
        }


def create_cwa_training_samples(trajectories, vocab, K: int = None, T: int = None,
                                oversample: bool = True):
    """Build DeepOP samples.

    When T > 0 each sample also carries `h_history`, the T observed states
    preceding the horizon. That is what lets the trainer condition DeepOP on
    Branch B's *predicted* future states rather than the oracle ones stored in
    `h_future` -- audit finding E1: the shipped decoder was trained on ground
    truth it will never see at serve time, where its input is a WDT rollout.
    """
    # K/T were literal 4/0, contradicting the v4 contract (history 15, forecast 5).
    _c = get_contract()
    K = _c.forecast_steps if K is None else K
    T = _c.history_steps if T is None else T
    samples = []
    # A window at i == 0 has NO history at all. The old code filled it with
    # `[future_snaps[0].embedding] * T` -- i.e. the first TARGET's own latent,
    # repeated T times. Whatever consumes h_history (the live trainer rolls
    # Branch B forward from it) is then started from the answer. That is a
    # leak, and it leaks in the direction that flatters the model. When
    # T == 0 nothing reads h_history, so the window is legitimate and kept.
    first_i = 1 if T > 0 else 0
    for host_ip, snaps in trajectories.items():
        if len(snaps) < K + first_i:
            continue
        snaps = sorted(snaps, key=lambda s: s.window_idx)
        n = len(snaps)

        for i in range(first_i, n - K + 1):
            future_snaps = snaps[i : i + K]
            h_fut = np.array([world_state(s) for s in future_snaps], dtype=np.float32)
            h_hist = None
            if T > 0:
                lo = max(0, i - T)
                hist = [world_state(s) for s in snaps[lo:i]]
                # i >= 1 whenever T > 0, so `hist` is never empty here and the
                # left-pad repeats a real PAST state, never a target.
                assert hist, "T>0 window with no history -- first_i guard failed"
                if len(hist) < T:
                    hist = [hist[0]] * (T - len(hist)) + hist
                h_hist = np.array(hist, dtype=np.float32)

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

            # DeepOP encoder input: the observed attack sequence over the
            # history window (paper Eq. 1-3).
            n_obs = _c.history_steps
            obs_seq = observed_sequence_tokens(
                [vocab.encode(ps.coarse_category,
                              ps.technique_ids[0] if ps.technique_ids else "None")
                 for ps in snaps[max(0, i - n_obs):i]],
                n_obs, vocab)

            # Input: <BOS> + first K-1 tokens
            input_seq = [vocab.bos_idx] + token_ids[:-1]
            target_seq = token_ids

            sample_item = {
                "h_future": h_fut,
                "input_tokens": np.array(input_seq, dtype=int),
                "target_tokens": np.array(target_seq, dtype=int),
                "obs_token": int(obs_tok),
                "obs_tokens": np.array(obs_seq, dtype=int),
            }
            if h_hist is not None:
                sample_item["h_history"] = h_hist
            samples.append(sample_item)

            # Duplicate active attack sequences (2x) to prevent quiescent mode
            # collapse. `dict(sample_item)` rather than the same object twice:
            # the old code aliased one dict into two slots, so any later
            # in-place edit of a sample silently edited its twin.
            benign_idx = vocab.encode("Benign", None)
            if oversample and any(t != benign_idx and t != vocab.pad_idx for t in target_seq):
                samples.append(dict(sample_item))

    return samples


def train_cwa_decoder(
    epochs: int = 6,
    K: int = None,
    batch_size: int = 32,
    lr: float = 5e-4,
    save_path: str = "saved_models/deepop/cwa_forecast_decoder.pt",
    max_per_source: Optional[int] = None,
    wdt=None,
    label_smoothing: float = 0.04,
):
    """Standalone DeepOP trainer.

    `wdt`: a trained Branch B world model. When given, DeepOP is conditioned
    on `wdt.rollout(h_history)` -- what it receives at serve time. When None,
    it is conditioned on the ORACLE future latents plus a Gaussian step-sigma
    schedule, which is audit finding E1 and is still a train/serve mismatch;
    the run says so out loud rather than leaving it in a docstring.

    Previously this function requested `h_history` for every sample (T
    defaults to the contract's 15 steps, so 15x12 float32 = 720 bytes each)
    and then never read it: `create_cwa_training_samples` built it, the
    Dataset forwarded it, and the training loop looked only at `h_future`. The
    memory was paid and the E1 fix was not applied. T is now tied to whether
    a `wdt` was actually supplied.
    """
    K = get_contract().forecast_steps if K is None else K
    from data_unification.split_manager import get_split_manager
    print("Loading multi-dataset records for DeepOP CWA Decoder from ScientificSplitManager...")
    sm = get_split_manager()
    train_records = sm.get_train_records(max_per_source=max_per_source)
    val_records = sm.get_val_records(
        max_per_source=None if max_per_source is None else max(50, max_per_source // 2))

    tgn = build_or_load_tgne_ta()
    # Contract-bound. This was hardcoded to 60.0 while the shipped checkpoints
    # and live inference ran at 2.0s, so this trainer could not reproduce them.
    # The window now comes from the single source of truth.
    extractor = HostTrajectoryExtractor(
        tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds
    )
    train_trajectories = extractor.extract_trajectories(train_records)
    val_trajectories = extractor.extract_trajectories(val_records)

    vocab = get_joint_vocab(network_observable_only=True)
    _T = get_contract().history_steps if wdt is not None else 0
    if wdt is None:
        print("[!] DeepOP conditioning: ORACLE future latents + noise (audit E1 "
              "NOT fixed). At serve time the decoder is fed a Branch B rollout. "
              "Pass wdt= to train on what it will actually see.")
    else:
        print("DeepOP conditioning: Branch-B rollouts (E1 fixed)")
        wdt.eval()
    train_samples = create_cwa_training_samples(train_trajectories, vocab=vocab, K=K, T=_T)
    # Validation is NOT oversampled: the 2x duplication of attack windows is a
    # training device, and applying it to the eval split silently moves the
    # class prior every metric is computed at (measured -0.0879 on the Benign
    # share of a synthetic store). Baselines and accuracy must be read at the
    # deployment prior.
    val_samples = create_cwa_training_samples(val_trajectories, vocab=vocab, K=K, T=_T,
                                              oversample=False)
    print(f"Disjoint CWA sequence samples - Train: {len(train_samples)}, Val: {len(val_samples)} (Vocab size: {vocab.vocab_size})")

    # Say which of the 10 joint tokens can actually occur. Cross-entropy is
    # normalised over all 10 and label smoothing spends eps/V on each, so a
    # vocabulary that is mostly unreachable is a fact about the loss, not
    # trivia. On the shipped corpus only 4 of 10 occur in train and 3 in val.
    _vh = np.bincount(np.concatenate([s["target_tokens"] for s in val_samples]),
                      minlength=vocab.vocab_size) if val_samples else np.zeros(vocab.vocab_size, int)
    _th = np.bincount(np.concatenate([s["target_tokens"] for s in train_samples]),
                      minlength=vocab.vocab_size) if train_samples else np.zeros(vocab.vocab_size, int)
    print("  token support  " + "  ".join(
        f"{vocab.tokens[i]}:{int(_th[i])}/{int(_vh[i])}" for i in range(vocab.vocab_size)))
    print(f"  reachable: train {int((_th>0).sum())}/{vocab.vocab_size}, "
          f"val {int((_vh>0).sum())}/{vocab.vocab_size} "
          f"-- label smoothing eps={label_smoothing} puts "
          f"{(vocab.vocab_size-int((_th>0).sum()))*label_smoothing/vocab.vocab_size:.4f} "
          f"of every target's mass on tokens that cannot occur")

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

    d_state = int(train_samples[0]["h_future"].shape[-1]) if train_samples else 27
    decoder = DeepOPForecastDecoder(
        d_latent=d_state,
        d_model=72,
        vocab_size=vocab.vocab_size,
        n_heads=6,
        num_layers=2,
        window_sizes=[2, 4, 8],
        dim_feedforward=144,
    ).to(device)

    # Initialize prototypes from training class centroids (only when the
    # optional prototype head is enabled; the paper architecture has none).
    centroids = torch.zeros(vocab.vocab_size, d_state)
    counts_c = torch.zeros(vocab.vocab_size)
    for s in (train_samples if decoder.use_prototypes else []):
        h_arr = torch.from_numpy(s["h_future"])
        t_arr = torch.from_numpy(s["target_tokens"])
        for k in range(len(t_arr)):
            tok = int(t_arr[k].item())
            centroids[tok] += h_arr[k]
            counts_c[tok] += 1

    # Seed ONLY the classes that have support. Copying the whole tensor wrote
    # exact zeros into the 6 tokens that never occur in this corpus, and a
    # zero-norm prototype is gated out of the forward pass and therefore
    # receives no gradient -- it could never recover. The random init from
    # DeepOPForecastDecoder.__init__ is left in place for those.
    seeded = 0
    for c in (range(vocab.vocab_size) if decoder.use_prototypes else []):
        if counts_c[c] > 0:
            decoder.prototypes.data[c] = (centroids[c] / counts_c[c]).to(device)
            seeded += 1
    print(f"  prototypes seeded from centroids for {seeded}/{vocab.vocab_size} "
          f"tokens; the rest keep their random init so they stay differentiable")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_f1 = 0.0

    print(f"Starting DeepOP CWA Decoder training on {device} with Smooth-Weighted Cross-Entropy...")
    for epoch in range(1, epochs + 1):
        decoder.train()
        train_losses = []

        train_plain = []
        for batch in train_loader:
            h_fut = batch["h_future"].to(device)
            inp_tok = batch["input_tokens"].to(device)
            tgt_tok = batch["target_tokens"].to(device)

            # Exactly one of these two, never both and never neither:
            #   wdt given  -> condition on the real rollout (E1 fixed)
            #   wdt None   -> oracle future + a step-sigma noise schedule that
            #                 approximates rollout error (0.015 at k=1 growing
            #                 to 0.055 at k=K). This is an approximation of the
            #                 covariate shift, not a fix for it.
            if wdt is not None and "h_history" in batch:
                with torch.no_grad():
                    h_in = wdt.rollout(batch["h_history"].to(device),
                                       K=h_fut.shape[1]).detach()
            else:
                K_curr = h_fut.shape[1]
                step_sigma = torch.linspace(0.015, 0.055, steps=K_curr, device=device).unsqueeze(0).unsqueeze(-1)
                h_in = h_fut + torch.randn_like(h_fut) * step_sigma

            optimizer.zero_grad()
            obs_seq = batch["obs_tokens"].to(device) if "obs_tokens" in batch else None
            logits = decoder(h_in, inp_tok, obs_tokens=obs_seq)

            loss, plain = smoothed_and_plain_ce(
                logits, tgt_tok, label_smoothing=label_smoothing, weight=weights_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_plain.append(float(plain))

        # Validation evaluation.
        #
        # The old form appended every val batch's h_future/tokens to a Python
        # list, torch.cat'd the lot onto the GPU, and called
        # forecast_sequence() on the whole thing -- which then ran a per-sample
        # Python loop for each of K steps. At the live validation size
        # (1,459,428 sequences x 5 steps) that is 7.3M Python iterations plus
        # the entire split resident on the device. It cannot complete.
        # DeepOPTokenScorer accumulates per batch and holds only a VxV
        # confusion matrix, so peak memory is one batch.
        decoder.eval()
        val_losses = []
        val_plain = []
        scorer = DeepOPTokenScorer(vocab.vocab_size, device=device)
        scorer_nb = DeepOPTokenScorer(vocab.vocab_size, device=device)

        with torch.no_grad():
            for batch in val_loader:
                h_fut = batch["h_future"].to(device)
                inp_tok = batch["input_tokens"].to(device)
                tgt_tok = batch["target_tokens"].to(device)
                obs_tok = batch["obs_token"].to(device)
                if wdt is not None and "h_history" in batch:
                    h_fut = wdt.rollout(batch["h_history"].to(device),
                                        K=h_fut.shape[1]).detach()

                obs_seq = batch["obs_tokens"].to(device) if "obs_tokens" in batch else None
                logits = decoder(h_fut, inp_tok, obs_tokens=obs_seq)
                # Both numbers, because the objective and the reported metric
                # were different functions: training optimised the smoothed CE
                # and validation printed the plain one, so "val above train"
                # was reading a +0.19..+0.26 nat offset as generalisation gap.
                sm, pl = smoothed_and_plain_ce(
                    logits, tgt_tok, label_smoothing=label_smoothing, weight=weights_t)
                val_losses.append(float(sm))
                val_plain.append(float(pl))

                pred_tf = logits.argmax(dim=-1)
                free, _ = decoder.forecast_sequence(
                    h_fut, max_steps=K, observed_token=obs_tok, observed_sequence=obs_seq,
                    repetition_penalty=1.0, continuity_bonus=1.0)
                free_nb, _ = decoder.forecast_sequence(
                    h_fut, max_steps=K, observed_token=obs_tok, observed_sequence=obs_seq,
                    repetition_penalty=1.0, continuity_bonus=0.0)
                scorer.update(tgt_tok, inp_tok, obs_tok, pred_tf=pred_tf, pred_free=free)
                scorer_nb.update(tgt_tok, inp_tok, obs_tok, pred_free=free_nb)

        m = scorer.result()
        m_nb = scorer_nb.result()
        val_acc = m["acc_free"]
        val_macro_f1 = m["macro_f1_free"]
        val_active_f1 = val_macro_f1

        mean_val_loss = float(np.mean(val_plain))

        print(
            f"Epoch {epoch:02d} | train_ce(smoothed) {np.mean(train_losses):.4f} "
            f"train_ce(plain) {np.mean(train_plain):.4f} | "
            f"val_ce(smoothed) {np.mean(val_losses):.4f} "
            f"val_ce(plain) {mean_val_loss:.4f}"
        )
        print(DeepOPTokenScorer.format(m))
        print(f"  FREE-RUNNING without the step-0 continuity bonus: "
              f"acc={m_nb['acc_free']:.4f} macro_f1={m_nb['macro_f1_free']:.4f} "
              f"lift={m_nb['lift_free']:+.4f}   <-- the model's own number")

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
    parser.add_argument("--K", type=int, default=get_contract().forecast_steps,
                        help="Forecast horizon steps (default: the v4 contract)")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test with small sample count")
    args = parser.parse_args()

    if args.smoke_test:
        print("[*] Running DeepOP CWA Decoder Smoke Test (1 epoch)...")
        train_cwa_decoder(epochs=1, K=args.K, batch_size=16, max_per_source=100)
        print("[+] DeepOP CWA Decoder Smoke Test Succeeded!")
    else:
        train_cwa_decoder(epochs=args.epochs, K=args.K)



class LazyCWADataset(Dataset):
    """DeepOP samples built on demand from a TrajectoryStore.

    `create_cwa_training_samples` materialises h_future [K,12], h_history
    [T,12] and two token arrays per sample -- roughly **1,250 bytes each**,
    and one sample per snapshot plus a duplicate for every attack sequence.
    At full corpus density (~42M snapshots) that is **~52 GiB**, which does
    not fit and cannot be solved by using less data.

    Same fix as Branch A and Branch B: keep int32 index columns and gather
    from the store's memmapped block on __getitem__.

    Semantics match the eager path, including the 2x oversampling of
    sequences that contain any non-Benign token -- represented here by
    listing those positions twice in the index rather than by duplicating
    the arrays.

    ## `oversample` must be False for validation

    The duplication exists to stop the decoder collapsing to the quiescent
    sequence during TRAINING. Applying it to the validation set changes the
    class prior the metrics are computed at, and every metric moves with it:
    accuracy, macro-F1, the persistence baseline and the majority baseline.
    Measured on the synthetic store in tests/test_lazy_cwa_dataset.py, the
    Benign share of validation targets goes 0.8571 -> 0.7692 (-0.0879) purely
    from the duplication. `scripts/retrain_future_models_live.py` builds both
    splits with the same call, so every DeepOP validation number logged so far
    was computed at an inflated attack rate rather than the deployment one.

    Note also that duplicating is not the same as weighting: the two copies of
    a window land in different minibatches of the same epoch, so they are
    correlated gradient contributions rather than independent ones, and the
    resulting prior shift is never undone at serve time (no logit adjustment
    anywhere in the serving path). If the intent is a prior correction, a
    `weight=` on the loss is the thing that composes with evaluation; the
    duplication is kept here because it is what the shipped runs did.
    """

    def __init__(self, store, vocab, K: int = None, T: int = None,
                 oversample: bool = True):
        _c = get_contract()
        self.K = _c.forecast_steps if K is None else K
        self.T = _c.history_steps if T is None else T
        # Length of the observed sequence the encoder reads. Independent of T,
        # which only controls whether Branch B's input history is emitted.
        self.n_obs = _c.history_steps
        self.store = store
        self.vocab = vocab
        self._window_seconds = float(_c.window_seconds)
        self.oversample = oversample

        benign_cat = None
        for i, name in enumerate(store.categories):
            if name == "Benign":
                benign_cat = i
                break

        # A window at i == 0 has no history at all, and the old code filled
        # h_history by repeating `h_fut[0]` -- the FIRST TARGET's own latent.
        # The live trainer then rolls Branch B forward from that, so for those
        # samples DeepOP is conditioned on the oracle latent of the step it is
        # being asked to predict. It is a small leak (bounded at roughly
        # 0.5-4.5% of the live validation set from the run's own aggregates:
        # val_n 1,147,616 snapshots, 1,459,428 DeepOP samples after
        # duplication) but it is a leak in the flattering direction. When
        # T == 0 nothing consumes h_history and the window is legitimate.
        first_i = 1 if self.T > 0 else 0
        self.n_windows_dropped_no_history = 0

        hosts, host_idx, pos = [], [], []
        for h in store:
            rows = store._rows_by_host[h]
            n = len(rows)
            if n < self.K:
                continue
            if n < self.K + first_i:
                self.n_windows_dropped_no_history += 1
                continue
            hi = len(hosts)
            hosts.append(h)
            self.n_windows_dropped_no_history += first_i
            starts = np.arange(first_i, n - self.K + 1, dtype=np.int32)
            # 2x oversample any window containing a non-Benign step, matching
            # create_cwa_training_samples' anti-collapse duplication.
            cats = store.cat_id[rows]
            # any(cats[i:i+K] != Benign) for every start, from one cumulative
            # sum -- the per-start Python loop this replaces cost minutes at
            # ~80M windows. Integer counts, so exactly the same booleans.
            if benign_cat is not None:
                _c = np.zeros(n + 1, dtype=np.int64)
                np.cumsum(cats != benign_cat, out=_c[1:])
                active = (_c[starts + self.K] - _c[starts]) > 0
            else:
                active = np.zeros(len(starts), bool)
            sel = np.concatenate([starts, starts[active]]) if self.oversample else starts
            host_idx.append(np.full(len(sel), hi, dtype=np.int32))
            pos.append(sel)

        self.hosts = hosts
        self._host_idx = np.concatenate(host_idx) if host_idx else np.zeros(0, np.int32)
        self._pos = np.concatenate(pos) if pos else np.zeros(0, np.int32)

    def target_token_histogram(self) -> np.ndarray:
        """Counts per vocab id over every target token this dataset will yield.

        Cheap: reads `store.cat_id` only, no feature gather. It exists because
        the DeepOP loss and macro-F1 are computed over a 10-token vocabulary of
        which most tokens cannot occur. Measured on the shipped corpus, from
        the categories the retrain logged:

            train categories  Execution, Benign, Impact, InitialAccess, C2
              -> reachable tokens {Benign.None, C2.T1071, Impact.T1498,
                                   InitialAccess.T1190}   4 of 10
            val   categories  Benign, Execution, InitialAccess, C2
              -> reachable tokens {Benign.None, C2.T1071,
                                   InitialAccess.T1190}   3 of 10

        <PAD>, <BOS>, <EOS>, CredentialAccess.T1110, Exfiltration.T1005 and
        Recon.T1595 never occur as a target anywhere in this corpus, and
        Impact.T1498 occurs in train but not in val. Consequences worth
        printing next to the loss: label smoothing at eps=0.04 assigns
        6 * eps/V = 2.4% of every target's mass to tokens that cannot occur,
        and macro-F1 must be averaged over classes with support or those six
        contribute free zeros.
        """
        row_tok = self._row_tokens()
        hist = np.zeros(self.vocab.vocab_size, dtype=np.int64)
        if len(self._pos) == 0:
            return hist
        # Every sample's K target rows, counted in chunks of samples. The
        # per-sample Python loop this replaces ran ~80M iterations at full
        # scale, three times per run. Integer counts: identical.
        flat, base = self._flat_base()
        ar = np.arange(self.K, dtype=np.int64)
        for c0 in range(0, len(self._pos), 1 << 21):
            hi = self._host_idx[c0:c0 + (1 << 21)].astype(np.int64)
            pos = base[hi] + self._pos[c0:c0 + (1 << 21)].astype(np.int64)
            hist += np.bincount(row_tok[flat[pos[:, None] + ar]].ravel(),
                                minlength=self.vocab.vocab_size)
        return hist

    def _flat_base(self):
        if not hasattr(self, "_flat"):
            from data_unification.host_major import flat_row_order
            self._flat, self._base = flat_row_order(
                [self.store._rows_by_host[h] for h in self.hosts])
        return self._flat, self._base

    def _row_tokens(self) -> np.ndarray:
        """The vocab token of every store row, `self._token(row)` for all rows at once.

        `vocab.encode` is a function of (category, first technique) only, so it
        is evaluated once per distinct pair and broadcast.
        """
        if getattr(self, "_row_tok", None) is not None:
            return self._row_tok
        st = self.store
        n_rows = int(len(st.cat_id))
        nt = len(st.techniques) + 1
        # One encode per (category, technique-or-None) pair -- a few hundred --
        # then a table lookup per row, in chunks so the peak stays small (an
        # np.unique over 77M int64 keys peaked at several GB).
        lut = np.empty(max(1, len(st.categories)) * nt, dtype=np.int64)
        for c in range(len(st.categories)):
            for t in range(-1, nt - 1):
                lut[c * nt + t + 1] = self.vocab.encode(
                    st.categories[c], st.techniques[t] if t >= 0 else "None")
        dt = np.int16 if (lut.size == 0 or (lut.min() >= -2 ** 15 and lut.max() < 2 ** 15)) else np.int64
        lut = lut.astype(dt)
        out = np.empty(n_rows, dtype=dt)
        tech_off = np.asarray(st.tech_off)
        tech_flat = np.asarray(st.tech_flat)
        for r0 in range(0, n_rows, 1 << 22):
            r1 = min(n_rows, r0 + (1 << 22))
            lo, hi = tech_off[r0:r1], tech_off[r0 + 1:r1 + 1]
            first = np.full(r1 - r0, -1, dtype=np.int64)
            has = hi > lo
            if has.any():
                first[has] = tech_flat[lo[has]]
            out[r0:r1] = lut[np.asarray(st.cat_id[r0:r1], dtype=np.int64) * nt + first + 1]
        self._row_tok = out
        return self._row_tok

    def __len__(self):
        return int(len(self._pos))

    def _token(self, row: int) -> int:
        lo, hi = int(self.store.tech_off[row]), int(self.store.tech_off[row + 1])
        tech = self.store.techniques[int(self.store.tech_flat[lo])] if hi > lo else "None"
        return self.vocab.encode(self.store.categories[int(self.store.cat_id[row])], tech)

    def __getitem__(self, idx):
        host = self.hosts[int(self._host_idx[idx])]
        i = int(self._pos[idx])
        rows = self.store._rows_by_host[host]
        fut = rows[i:i + self.K]

        h_fut = np.ascontiguousarray(self.store.feats[fut], dtype=np.float32)
        token_ids = [self._token(int(r)) for r in fut]

        out = {
            "h_future": torch.from_numpy(h_fut),
            "input_tokens": torch.tensor([self.vocab.bos_idx] + token_ids[:-1], dtype=torch.long),
            "target_tokens": torch.tensor(token_ids, dtype=torch.long),
            "obs_token": torch.tensor(
                self._token(int(rows[i - 1])) if i > 0 else self.vocab.bos_idx,
                dtype=torch.long),
            # DeepOP encoder input: observed techniques over the history window.
            "obs_tokens": torch.tensor(observed_sequence_tokens(
                [self._token(int(r)) for r in rows[max(0, i - self.n_obs):i]],
                self.n_obs, self.vocab), dtype=torch.long),
        }
        if self.T > 0:
            lo = max(0, i - self.T)
            hist_rows = rows[lo:i]
            if len(hist_rows):
                hist = self.store.feats[hist_rows]
                if len(hist) < self.T:            # left-pad by repeating the first
                    hist = np.concatenate(
                        [np.repeat(hist[:1], self.T - len(hist), axis=0), hist], axis=0)
            else:
                # Unreachable: `first_i` guarantees i >= 1 whenever T > 0.
                # It used to repeat h_fut[0], the first TARGET's latent.
                raise AssertionError(
                    f"T={self.T}>0 window at i=0 has no history; the first_i "
                    f"guard in __init__ was bypassed")
            out["h_history"] = torch.from_numpy(
                np.ascontiguousarray(hist, dtype=np.float32))

            # Real elapsed times for the Branch-B rollout DeepOP conditions on,
            # in exactly LazyHostRolloutDataset's convention: seconds relative
            # to the last OBSERVED step rows[i-1], history <= 0 ending at 0,
            # future > 0.
            #
            # Branch B is now trained with these, so the rollouts DeepOP is
            # conditioned on must be produced the same way -- otherwise DeepOP
            # trains on rollouts from a uniform 2 s grid while Branch B's own
            # training and serving use real spacing, a skew between the two
            # models rather than inside one.
            w = self.store.window_idx
            w0 = float(w[rows[i - 1]])
            t_h = np.asarray(w[hist_rows], dtype=np.float64) - w0
            if len(t_h) < self.T:        # padded steps repeat the first real one's time
                t_h = np.concatenate([np.repeat(t_h[:1], self.T - len(t_h)), t_h])
            t_f = np.asarray(w[fut], dtype=np.float64) - w0
            out["t_history"] = torch.from_numpy(
                np.ascontiguousarray(t_h * self._window_seconds, dtype=np.float32))
            out["t_future"] = torch.from_numpy(
                np.ascontiguousarray(t_f * self._window_seconds, dtype=np.float32))
        return out

    # -- batched access ------------------------------------------------------
    #
    # The default-collated batch of `[self[i] for i in indices]`, built with a
    # few vectorised gathers: features from the host-major copy of the block
    # (data_unification/host_major.py), tokens from a per-row token table
    # instead of ~21 `vocab.encode` calls per sample.
    # tests/test_branch_b_deepop_batched_loader.py pins bit-identity.

    def enable_batched(self, host_major=None, spill_dir=None):
        from data_unification.host_major import host_major_for
        rows = [self.store._rows_by_host[h] for h in self.hosts]
        self._flat, self._base, self._feats_hm = host_major_for(
            self.store, rows, host_major=host_major, spill_dir=spill_dir)
        self._row_tokens()
        self._batched_ready = True
        return self

    def _feats_at(self, p):
        if getattr(self, "_feats_hm", None) is not None:
            return np.asarray(self._feats_hm[p])
        return np.asarray(self.store.feats[self._flat[p]])

    def gather_batch(self, indices):
        if not hasattr(self, "_feats_hm"):
            self.enable_batched(host_major=False)
        st, K, T, n_obs, V = self.store, self.K, self.T, self.n_obs, self.vocab
        flat, tok = self._flat, self._row_tok
        idx = np.asarray(indices, dtype=np.int64)
        b = self._base[self._host_idx[idx].astype(np.int64)]
        i = self._pos[idx].astype(np.int64)
        bi = (b + i)[:, None]
        pf = bi + np.arange(K, dtype=np.int64)                 # the K target rows (always real)
        tok_f = tok[flat[pf]].astype(np.int64)
        bos = np.full((len(idx), 1), V.bos_idx, dtype=np.int64)
        # observed sequence: [PAD]*(n_obs-m) + [BOS] + tokens of the m = min(i, n_obs)
        # rows before i (observed_sequence_tokens)
        m = np.minimum(i, n_obs)[:, None]
        col = np.arange(n_obs + 1, dtype=np.int64)[None, :]
        po = bi - n_obs - 1 + col
        lead = n_obs - m
        val = tok[flat[np.maximum(po, b[:, None])]].astype(np.int64)
        obs_seq = np.where(col < lead, V.pad_idx, np.where(col == lead, V.bos_idx, val))
        prev = tok[flat[np.maximum(b + i - 1, b)]].astype(np.int64)
        out = {
            "h_future": torch.from_numpy(np.ascontiguousarray(self._feats_at(pf), dtype=np.float32)),
            "input_tokens": torch.from_numpy(np.ascontiguousarray(
                np.concatenate([bos, tok_f[:, :-1]], axis=1))),
            "target_tokens": torch.from_numpy(np.ascontiguousarray(tok_f)),
            "obs_token": torch.from_numpy(np.where(i > 0, prev, V.bos_idx).astype(np.int64)),
            "obs_tokens": torch.from_numpy(np.ascontiguousarray(obs_seq, dtype=np.int64)),
        }
        if T > 0:
            if (i < 1).any():
                raise AssertionError(f"T={T}>0 window at i=0 has no history; the first_i "
                                     f"guard in __init__ was bypassed")
            # rows[max(0, i-T):i], left-padded by repeating the first
            ph = np.maximum(bi - T + np.arange(T, dtype=np.int64), b[:, None])
            out["h_history"] = torch.from_numpy(
                np.ascontiguousarray(self._feats_at(ph), dtype=np.float32))
            w = np.asarray(st.window_idx)
            w0 = w[flat[b + i - 1]].astype(np.float64)[:, None]
            t_h = w[flat[ph]].astype(np.float64) - w0
            t_f = w[flat[pf]].astype(np.float64) - w0
            out["t_history"] = torch.from_numpy(
                np.ascontiguousarray(t_h * self._window_seconds, dtype=np.float32))
            out["t_future"] = torch.from_numpy(
                np.ascontiguousarray(t_f * self._window_seconds, dtype=np.float32))
        return out
