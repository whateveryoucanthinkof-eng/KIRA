# Training Data Lineage Audit — Are PCAPs Used?

**Date:** 2026-09-19
**Scope:** Do the live models (Branch A, Branch B, DeepOP, TGNE-TA/BiTA) train on raw PCAP, or only flow records?
**Status:** Core claims independently re-verified. One inference corrected (see §6).

---

## Verdict

**NO. Not one byte of any PCAP file is read by any training path, for any model.**

Every live model trains on CSV / Argus-netflow-text / Parquet flow records only. This holds for the working tree and for every commit on every ref in the repo's history.

| Model | PCAP in training? | Actual training input |
|---|---|---|
| TGNE-TA / BiTA (12-D latent encoder) | **NO** | Warden/CESNET IDEA alert CSV (`*March_e.csv`, 7 cols) |
| Branch A (MultiTaskLSTM, 27-D) | **NO** | CIC-IDS-2018 CSV + CTU-13 `.binetflow` (Argus text) |
| Branch B (HostWorldDynamicsTransformer) | **NO** | Same records → 12-D TGNE latents only |
| DeepOP CWA forecast decoder | **NO** | Same records → 12-D latents + ATT&CK token vocab |
| `world_model/` research track | **NO** (and non-runnable) | Cached `.pt` of latents from pre-aggregated flow Parquet |

---

## 1. Hard proof

- **Zero** occurrences of `scapy`, `dpkt`, `pyshark`, `rdpcap`, `PcapReader`, `tshark`, `libpcap` in any `.py` file — verified against the working tree *and* by iterating `git rev-list --all` over every commit on all refs. Empty result set. **(Re-verified independently.)**
- **Exactly two** binary-mode file operations in the whole repo:
  - `telemetry/run_telemetry.py:168` — `open(args.replay, "rb")` (live-demo replay)
  - `telemetry/run_telemetry.py:83` — `AsyncPcapRecorder` *writing* a pcap header
- Every `.pcap` string inside a `.py` is part of a **CICFlowMeter CSV filename** (`*.pcap_ISCX.csv`), not a capture file: `branch_a_gnn_lstm/train_branch_a.py:40-41`, `data_unification/split_manager.py:61-63,99,126-128`.

### The decisive artifact

`saved_models/branch_a/branch_a_lstm.pt` carries its own provenance:

```
training_contract = {'window_size_sec': 2.0, 'history_steps': 5, 'feature_dim': 27,
  'sources': ['/var/home/samito/Documents/SIH/DATA/CSV',
              '/var/home/samito/Documents/SIH/CTU-13-Dataset']}
```

Consumed by `scripts/retrain_branch_a_live.py:99-100`:

```python
cic_files = sorted(args.cic_dir.glob("*.csv"))
ctu_files = sorted(args.ctu_dir.glob("*/*.binetflow"))
```

`CTU-13-Dataset/*/` holds **13 `.pcap` AND 13 `.binetflow`** files — the glob matches only the latter.

### Verified disk volumes

| Path | Size | Used in training |
|---|---|---|
| `~/Documents/SIH/DATA/pcap/` | **560 GB** | no |
| `~/Documents/SIH/CTU-13-Dataset/` | **75 GB** (13 pcap + 13 binetflow) | binetflow only |
| `~/Documents/SIH/DATA/CSV/` | 6.5 GB | yes |

**~635 GB of packet captures unused** vs ~9 GB of flow records consumed.

---

## 2. Exact data lineage

**TGNE-TA** (produces the 12-D latent all three downstream models consume)
```
Dataset/*March_e.csv (Warden IDEA alerts: DetectTime, SourceIP, TargetIP, Proto, Port, Category, FlowCount)
  → bita/train.py:96   glob.glob(os.path.join(dataset_dir, "*March_e.csv"))
  → bita/train.py:101  pd.read_csv(f)
  → bita/train.py:156-175 extract_canonical_edge_features(...)
  → 12-D edge feature → ExtendedTGN(bigru_transformer)
  → bita/saved_models/bita_bigru_transformer-warden_alerts.pth
```
Note `bita/train.py:163-166`: `bwd_bytes`, `bwd_packets`, `duration` are **hardcoded to 0.0** for this source, so 3 of 12 canonical edge dims are constant-zero in TGNE pretraining.

**Branch A** (the shipped path)
```
DATA/CSV/*.csv               → data_unification/cic2018_adapter.py:34  pd.read_csv(...)
CTU-13-Dataset/*/*.binetflow → data_unification/ctu13_adapter.py:107   pd.read_csv(...)
  → UnifiedFlowRecord (unified_schema.py:36-54 — 16 fields, all flow-level)
  → FlowToTemporalEventAdapter.extract_edge_feature (flow_to_temporal_event.py:79-84)
  → tgne_features.extract_canonical_edge_features (tgne_features.py:68-99) = 12-D
  → TGNE-TA get_host_embeddings (multi_dataset_stream.py:271) = H_t ∈ R^12
  ⊕ compute_host_temporal_attributes (multi_dataset_stream.py:137-206) = 15-D
  → 27-D → MultiTaskLSTM(input_dim=27)  (scripts/retrain_branch_a_live.py:130-136)
```

**Branch B** — `train_branch_b.py:56,62` builds `h_history`/`h_future` from `s.embedding` only. Input space is **the 12-D latent alone** — it never sees the 15 host temporal attributes.

**DeepOP** — `train_cwa_decoder.py:57` (`h_fut` from `s.embedding`), conditioned on the same 12-D latents.

**All offline adapters are `pd.read_csv` / `pq.ParquetFile`:**
- `cic2017_adapter.py:38`, `cic2018_adapter.py:34`, `ctu13_adapter.py:37` (parquet), `ctu13_adapter.py:107` (binetflow), `warden_adapter.py:34`
- `flow_to_temporal_event.py` and `multi_dataset_stream.py` open **no files** — pure in-memory transforms.

---

## 3. PS-required packet features absent from the input space

The models' entire input is 27 numbers: a 5-tuple plus byte/packet counters plus duration.

| PS-required feature | Status | Evidence |
|---|---|---|
| **TTL variance / stats** | **ABSENT** | Parsed at `sniffer.py:91,107`, emitted as `ttl_or_hop_limit` (`:153`), then dropped — `FlowRecord.update` (`flow_table.py:72-96`) reads only timestamp, packet_length, payload_length, tcp_flags |
| **TCP window size** | **ABSENT** | Parsed at `sniffer.py:160`, dropped by `FlowRecord.update` |
| **IP fragment flags** | **ABSENT** | Parsed at `sniffer.py:94-96`, emitted `:154-156`, dropped |
| **Payload size distribution** | **EFFECTIVELY ABSENT** | Survives only as a boolean: `if payload_len > 0: self.fwd_act_data_pkts += 1` (`flow_table.py:83-84`) |
| **Port scan signatures** | **ABSENT** (weak proxy only) | Only `unique_dst_ports_count` and `unique_peers_count`. `flow_table.py:318-322` computes `dst_port_entropy`, `max_pair_sequential_port_score` — discarded |
| **Retransmission counts** | **ABSENT ENTIRELY** | No sequence-number state. `sniffer.py:124` unpacks `seq, ack` and stores neither |
| *(bonus)* TCP flag counts | **ABSENT from model input** | Counted in `FlowRecord` (`flow_table.py:91-96`) but in neither the 12-D edge schema nor the 15-D host attrs — **the models cannot see a SYN flood or a failed handshake** |

**Net effect:** in input-space terms these are NetFlow-v5-class models.

---

## 4. The abandoned PCAP path

**(a) In this repo.** `telemetry/packet/pcap_engine.py` (343 lines, 30 features) was imported by exactly one file across every commit on every ref: `telemetry/state/state_builder.py`. No training script ever imported it. *(Restored in commit `0a01138`, still unwired.)*

**(b) In the predecessor repo — the big opportunity.** `~/Documents/SIH/src/` has a complete working pcap stack (`pcap_features.py` with a struct-based pcap/pcapng parser, `pcap_state_builder.py`, `pcap_flow_join.py`). It produced real artifacts, **verified on disk**:

- `~/Documents/SIH/processed/pcap_states_2s.parquet` — **32,265 rows × 33 cols, 31 `pcap_*` columns**
- `~/Documents/SIH/processed/network_states_2s_pcap.parquet` — **139,677 rows × 82 cols**, flow states already *joined* with packet features

At **2-second windows — the same window size the live pipeline uses.** Columns include `pcap_handshake_completion_ratio`, `pcap_unanswered_syn_ratio`, `pcap_syn_ratio`, `pcap_rst_ratio`, `pcap_pkt_iat_mean/std`, `pcap_pkt_len_mean/std`, `pcap_tcp_window_mean/std`, `pcap_tcp_retransmission_count/ratio`, `pcap_fragment_ratio`.

**Closing the biggest PS gap does not require re-parsing 635 GB.** The features are computed and joined, sitting in a 26 MB parquet.

**(c) The blocker.** `ctu13_adapter.py:44-51,87` (`parse_parquet`) reads exactly 8 aggregate columns and ignores everything else — all 31 `pcap_*` columns would be dropped at the adapter boundary even if pointed there.

---

## 5. Additional findings

**(i) Offline training scripts train on zero records and do not error.** `train_branch_a.py`, `train_branch_b.py`, `train_balanced_cwa.py` route through `ScientificSplitManager`, whose roots (`cic2017csv/`, `bita/Dataset/`, `data/ctu13/`, and `CIC2018_DIR = r"C:\SIH_DATA\dump\..."` — a Windows path, `split_manager.py:29`) are **all missing**. Every load is wrapped in `if os.path.exists(...)`, so `get_train_records()` silently returns `[]` and the epoch loop no-ops. Only `scripts/retrain_*_live.py` produced the shipped checkpoints.

**(ii) Label leakage in the CTU-13 parquet adapter.** `data_unification/ctu13_adapter.py:62` — **verified verbatim**:
```python
dst_ip = "147.32.84.180" if is_attack else "147.32.80.1"
```
The destination node ID — which determines the graph edge TGNE attends over — is chosen **from the ground-truth label**. Ports similarly synthesized from the malware family string (`:64`).

*Mitigating fact (verified):* this sits in `parse_parquet`. The shipped checkpoint went through `parse_netflow_csv` (`retrain_branch_a_live.py:38`), so live weights are **not** contaminated. But the code is live and `train_branch_a.py:48-50` still references that path.

**(iii) Branch A metrics.** `branch_a_lstm.pt` records `{'loss': -1.9988, 'risk_mae': 0.3195, 'tech_accuracy': 0.99941, 'epoch': 5}`.

**(iv) `world_model/` cannot execute.** `train_world_model.py:26` imports `world_model.data.latent_dataset`; neither `world_model/data/` nor `world_model/models/` exists. Config points at `c:/Users/vyomk/OneDrive/...`.

---

## 6. Correction to the original agent report

The agent flagged Branch A's **negative loss (-1.9988)** as evidence of a degenerate, benign-dominated training set. **That inference is wrong.**

`MultiTaskUncertaintyLoss` (`branch_a_gnn_lstm/lstm_multitask.py:56-81`) is Kendall & Gal homoscedastic weighting:

```
L = exp(-s)·L_task + s
```

The bare `+ s` (log-variance) terms go negative as learned variances drop, so a **negative total is mathematically expected here**, not a red flag.

The other half of the finding stands: **99.94% technique accuracy at epoch 5 alongside 0.32 risk MAE** is the signature of severe class imbalance, not skill. Worth a separate look.
