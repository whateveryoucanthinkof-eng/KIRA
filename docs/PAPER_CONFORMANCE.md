# Paper conformance

Which published method each model implements, where each equation lives in the code, and every place
the code deliberately differs from the paper, with the reason. Tests that pin this:
`tests/test_models_follow_the_papers.py` and `tests/test_pipeline_composes.py`.

| Component | Paper | Code |
|---|---|---|
| TGNE-TA encoder | Makki Nayeri & Rezvani, *BiTA: Bidirectional GRU-Transformer Aggregator in a Temporal Graph Network Framework for Alert Prediction in Computer Networks*, arXiv:2604.22781 (Expert Systems with Applications) | `bita/` |
| Branch A | Vitulyova, Babenko, Kolesnikova, Kiktev & Abramkina, *A Hybrid Approach Using Graph Neural Networks and LSTM for Attack Vector Reconstruction*, Computers 14(8):301, 2025 | `branch_a_gnn_lstm/` |
| DeepOP | Zhang, Xue & Su, *DeepOP: A Hybrid Framework for MITRE ATT&CK Sequence Prediction via Deep Learning and Ontology*, Electronics 14(2):257, 2025 | `deepop_decoder/` |
| Branch B | none of the three; a transformer world model over the host state | `branch_b_world_model/` |
| (not implemented) | Graph Transformer Autoencoder for NIDS, *GTAE-IDS*, IEEE TIFS 2025, doc. 10948513 | see [GTAE-IDS](#gtae-ids) |

---

## The pipeline as implemented

```
2 s window of flows (SPAN live, or CIC/CTU/PCAP offline, same code path)
  │
  ▼  one interaction graph per window: nodes = IP addresses, edges = flows
TGNE-TA (BiTA)                          bita/model/extentedtgn.py
  │  graph attention over this window's edges        ← short range
  │  + TGN memory updated by the BiTA aggregator     ← long range, across windows
  ▼
z(t) ∈ R^12 per host
  │  ⊕ 15 host attributes a(t) (flow counts, bytes, peers, ports, rates)
  ▼
s(t) = [z(t) ; a(t)] ∈ R^27             data_unification/trajectory_store.world_state
  │
  ├──► Branch A   LSTM over s(t-14..t) → risk R_t, technique P_t, gradation G_t
  │                                                        │ technique per window
  └──► Branch B   world model over s(t-14..t) → ŝ(t+1..t+5)│
                                         │                 │
                                         ▼                 ▼
                     DeepOP  decoder (CWA) ◄── cross-attention ── encoder(observed techniques)
                                         │
                                         ▼
                           next ATT&CK techniques for t+1..t+5
```

Three things are worth knowing because they are easy to get wrong when describing this system:

* **The per-window graph is not bipartite.** BiTA's graph is bipartite because alerts are attacker→victim.
  Flows are not: an internal host is both source and destination. The bipartite construction is used
  only by the Warden alert loader in `bita/train.py`, which is how the BiTA paper's experiment is
  reproduced.
* **Both branches receive the same 27-D state.** Branch B used to receive only the 12-D TGNE latent.
* **Both branches feed DeepOP as learned inputs.** Branch B's forecast states are the decoder's
  cross-attention memory. Branch A's technique for each window of the history is the "observed attack
  sequence" DeepOP's encoder reads. Before this change Branch A reached DeepOP only as a hand-set
  step-0 logit bonus.

---

## BiTA → `bita/`

| Paper | Code |
|---|---|
| Eq. 1 `m = MSG(u, v, e_uv, t)` | TGN message function, applied first for BiTA: `TGN._aggregate`, `bita/model/tgn.py` |
| Eq. 2 messages grouped per edge, time-ordered | `BiTAAggregator.aggregate`, step 2, `bita/modules/message_aggregator.py` |
| Eq. 3 `x = m + TimeEnc(t)` | `BiTAAggregator.time_encoder` (TGAT/Bochner encoding) |
| Eq. 4–5 `z_temp = BiGRU(x)[-1]` | `BiTAAggregator.bigru`, final hidden state of both directions |
| Eq. 6 `e = W_e z_temp + b_e` | `BiTAAggregator.W_e` |
| Eq. 7–8 Transformer across the edges of the batch | `BiTAAggregator.transformer` (2 heads, FFN, residual, LayerNorm) |
| Eq. 9 mean over edges incident to v | `index_add_` readout in `BiTAAggregator.aggregate` |
| Eq. 11 `Mem_v = GRU(Mem_v, h̄_v)` | `GRUMemoryUpdater`, `bita/modules/memory_updater.py` |
| Causality (raw message store, update at batch start) | `ExtendedTGN.update_memory_for` / `store_interactions`; training uses TGN's own loop |
| Eq. 13–14 category head, focal loss | `ExtendedTGN.category_predictor`, `FocalLoss` in `bita/train.py` |
| Eq. 15–16 link prediction, BCE | `TGN.compute_edge_probabilities`, `edge_criterion` |
| Eq. 17 `L = L_link + λ L_cat` | `--cat_loss_weight` |
| Adam, lr 1e-4, batch 128, 2 heads, message dim 100, ≤50 epochs, patience 5 | `bita/train.py` defaults |
| Warden: median-size class resampling before the split | `median_resample`, on by default for the Warden loader |
| 70th/85th percentile temporal split; 10% of nodes held out as new (inductive) | `split_data` |

**What was wrong before.** TGN builds a message aggregator only when memory is on, and every shipped
encoder ran with `use_memory=False`. The configured `aggregator_type: "bigru_transformer"` was
therefore never constructed or executed, so the encoder contained no BiTA component at all. The
aggregator code itself also did not follow Algorithm 1: it had no time encoding and no Transformer
block, did not attend across edges, and read out the last token instead of mean-pooling. Memory is
now on by default and the aggregator follows Algorithm 1.

**Deviations**

| Paper | Here | Why |
|---|---|---|
| memory dimension 9 | 12 | The embedding module adds memory to the node features (`embedding_module.py`), so the two widths must match. Node features are the 12-D contract. 9 would crash. |
| λ = 1 | `--cat_loss_weight 15` | On this corpus focal loss puts the category term ~14× below the link term, and at λ = 1 the category head was measured to stop learning (`tests/test_category_loss_is_not_swamped.py`). `--cat_loss_weight 1` gives the paper's value. |
| Transformer over all edges in the batch | groups of ≤128 edges | One 2 s window can hold thousands of flows. 128 is the paper's own batch size. |
| messages per edge: all | last 64 | Memory bound for flooding windows. |
| `TimeEnc(t)` on absolute time | on the message's age within its edge sequence | Encoding raw epoch seconds puts every message at an arbitrary phase of every frequency. |
| unbounded temporal neighbourhood | neighbours limited to the current 2 s window when extracting host states | Serving only ever has the current window. Long-range history reaches the embedding through memory. Training extraction used to see older neighbours that serving never has. |
| aggregator ablations (BiTransformer, relative, stacked, TCN) | not implemented; the trainer refuses them | They used to be accepted and silently trained as BiGRU-Transformer. |

---

## GNN-LSTM → `branch_a_gnn_lstm/`

`MultiTaskLSTM.PAPER_ARCH` is the default (`scripts/retrain_branch_a_live.py --architecture paper`).

| Paper | Code |
|---|---|
| Eq. 1, 4 `H_t = LSTM([H_GNN, X_t])` | `MultiTaskLSTM.lstm` over `[z(t) ; a(t)]`: the graph embedding is TGNE's 12-D latent, and the 15 temporal attributes are the paper's 15 |
| 1 LSTM layer × 256 units, dropout 0.2 | `PAPER_ARCH` |
| Table 6 `R_t = σ(W_R H_t + b_R)` | `risk_head` (Linear + Sigmoid) on the last hidden state |
| Table 6 `P_t = softmax(W_p H_t + b_p)` | `technique_head` (Linear) |
| Table 6 `G_t = σ(W_G H_t + b_G)`, scalar | `gradation_head` (Linear → 1) + sigmoid, `gradation_mode="scalar"` |
| §3.4 `L = 0.5 L_risk(BCE) + 0.3 L_tech(CE) + 0.2 L_grad(MSE)` | `compute_loss`, `loss_weighting="fixed"` |
| Adam, η = 1e-3 | `retrain_branch_a_live.py` |

**Deviations**

| Paper | Here | Why |
|---|---|---|
| GCN over the MITRE ATT&CK technique graph (200 nodes) | TGN over the observed network (hosts, flows) | The live input is traffic, not an ATT&CK graph. The GNN's role (structural embedding joined to temporal attributes before the LSTM) is the same. |
| SMOTE for class balance, then plain CE | class-weighted CE (`focal_gamma=0`) | SMOTE interpolates flat records and has no meaning for overlapping host sequences. Plain CE on a ~82.5%-Benign corpus collapses the technique head. |
| gradation target g_t "target confidence" | severity level / 3 ∈ {0, ⅓, ⅔, 1} | The corpus has severity levels, not confidence labels. |
| T = 50 steps, batch 32, 100 epochs | T = 15 (the 30 s contract), larger batches | Sequence length is the system's temporal contract. Batch 32 over ~23M sequences is impractical. |

---

## DeepOP → `deepop_decoder/`

| Paper | Code |
|---|---|
| Eq. 1–3 label embedding + sinusoidal PE | `token_embed`, `SinusoidalPositionalEncoding` |
| Eq. 4–6 encoder: temporal multi-head attention + FFN + Add & Norm | `ObservedSequenceEncoder` |
| Eq. 7 `Split(x, cw_i)` into non-overlapping windows | `CausalWindowAttention(window_mode="partitioned")` |
| Eq. 8 concatenate window groups; h = 6 heads, n_cw = 3 | `CausalWindowAttention`, `window_sizes=[2, 4, 8]` |
| Eq. 9–10 causal mask inside each window | `_partitioned_mask` |
| Algorithm 1: autoregressive decoding from `<BOS>` | `forecast_sequence` |
| Eq. 11 cross-entropy | `smoothed_and_plain_ce` (plain CE is reported alongside) |
| detection-failure robustness (§4.4, Fig. 7) | observed techniques randomly dropped during training (`OBS_TOKEN_DROPOUT = 0.1`) |

**What was wrong before.** The model had no encoder. It decoded only from Branch B's predicted states,
and the observed technique entered through `CONTINUITY_BONUS_*`: a hand-set logit added at inference,
never learned. It used learned rather than sinusoidal positions and sliding rather than partitioned
windows, and it carried three heads the paper does not have. Those are kept only for the legacy
checkpoint (`LEGACY_ARCH`) and are off in the paper model.

**Deviations**

| Paper | Here | Why |
|---|---|---|
| Word2Vec label embeddings | learned `nn.Embedding` | 10 network-observable tokens is too small a vocabulary for skip-gram to learn anything a learned embedding does not. |
| mask 1 for t′ < t (Eq. 9, strict) | t′ ≤ t | The strict form leaves the first position of every window with no admissible key, so the softmax is undefined. |
| sequences extracted from CTI reports by ontology reasoning | sequences of per-window technique labels from flow datasets | This system observes traffic. The ontology/CTI module is out of scope. |
| decoder attends to the encoder only | decoder attends to [encoder ; Branch B forecast] | The forecast states are what makes this a world-model forecaster rather than a pure sequence model. |

---

## GTAE-IDS

Only the abstract is publicly accessible (IEEE Xplore paywall). From it: an **unsupervised**
graph-transformer **autoencoder**, trained on benign packet-level graphs built from the first packets of
each flow, which scores anomalies by reconstruction error. **Nothing in this repository implements it.**
Implementing it from the abstract alone would mean inventing its equations, so it is not claimed
anywhere. If it is to be cited as a basis, the natural place is an unsupervised novelty signal beside
Branch A, built from `telemetry/packet/pcap_engine.py`, and only once the full text is available.

---

## Reproducing the papers' experiments

**The numbers in the three papers are not targets for this system.** They were measured on
different data and different tasks:

| Paper | Their data | Their headline |
|---|---|---|
| BiTA | Warden alerts, 11–17 March 2019; NF-UNSW-NB15-v2 | Warden transductive AUC 0.9511, AP 0.9509, F1 0.9809; inductive AUC 0.7689 |
| GNN-LSTM | CICIDS2017 plus synthetic sequences (Markov chain and GAN, §3.3) | AUC 0.99, technique F1 0.85, risk MSE 0.05 |
| DeepOP | ATT&CK sequences extracted from APT reports | F1 0.894, recall 0.978, precision 0.822 |

This system predicts per-host, per-window outcomes on CIC-IDS-2017/2018 and CTU-13 flows. Its
results must be reported as its own, against its own baselines (persistence, majority class). They
are not directly comparable to the tables above.

**BiTA can be reproduced directly.** Its Warden protocol is implemented in the encoder trainer, with
the defaults above:

```bash
python bita/train.py --dataset_dir <Warden March CSVs> --data_name warden_alerts
```

Compare the logged per-epoch transductive and inductive AUC/AP with the paper's Table 6
(BiGRU-Transformer row). Pass `--no_memory` to measure the memoryless ablation.

**Retraining.** `./train.sh` trains everything in dependency order (encoder → Branch A and
Branch B → DeepOP, which refuses to start if Branch B does not beat persistence), and
`./train.sh promote` installs the result into `saved_models/` and writes the manifests. See the
README, "Train the models".
