# Cross-year protocol: train on CIC-IDS-2018, test on CIC-IDS-2017

The headline evaluation. Every model is trained and tuned on CIC-IDS-2018 and scored **once** on
CIC-IDS-2017. The two datasets differ in year, network (AWS `172.31.0.0/16` vs. a lab
`192.168.10.0/24`), attack tooling and CICFlowMeter build, so this measures generalisation rather
than fit to one capture. Expect lower numbers than a same-dataset split. That is what makes them
believable.

## Splits (`--split-scheme cross_year`)

Derived from `data_unification/splits.lock.json`, not a second lock (`split_policy.py`):

| Corpus | train | validation (all tuning) | test (scored once) |
|---|---|---|---|
| CIC-2018 (PCAP) | the lock's 8 train days | the lock's val + test days (2) | — |
| CIC-2017 | — | — | all 8 captures |
| CTU-13 | not used | not used | not used |

Early stopping, the alert threshold, the technique temperature and the conformal width are all
fitted on the two CIC-2018 validation days. CIC-2017 is loaded only after the model is frozen,
and no encoder ever reads it.

## The four rules and where they are enforced

1. **Classes CIC-2017 has and CIC-2018 lacks are reported separately.** CIC-2018 has no PortScan
   (reconnaissance, T1046) and no Heartbleed. The 2017 test reports technique macro-F1 over *classes
   training saw*, macro-F1 over *all* classes, and each unseen class with what the model predicted
   instead. Code: `cyberworld_v4/cross_dataset.py`. Checkpoints record their training class counts
   (`train_technique_counts`, `train_token_counts`) so any later evaluation can do the same.
2. **CIC-2018 is read from PCAP.** Nine of its ten CSV days fabricate host IPs from the row number,
   so a host graph built from them is synthetic. Under `cross_year`, all three trainers (encoder,
   Branch A, Branch B/DeepOP) read CIC-2018 from `--pcap-root`, with the CSVs used only as labels.
   CSV input is refused unless `--allow-cic2018-csv` is passed. CIC-2017 is parsed with its own
   adapter, which repairs its 12-hour clock. Code: `data_unification/training_sources.py`.
3. **Tuning uses CIC-2018 only.** This follows from the split table above.
4. **IP-address features are tested for leakage.** The octets identify CIC-2018's network, which
   CIC-2017 does not use, and `is_private`/`is_global` reflect how the captures were staged
   (attackers on public addresses). The `cross_network` ablation keeps only flags that mean the same
   on any network (`CYBERWORLD_ABLATE_NODE_FEATURES=cross_network`). The ablation is recorded in
   the encoder's config, and loading an encoder under a different setting is refused.

## Which encoder? Measure it

`scripts/run_encoder_comparison.py` trains three encoders and gives each the identical Branch A
protocol, so the encoder is the only thing that varies:

| Arm | Encoder training data |
|---|---|
| `warden` | Warden alerts only (BiTA's dataset) |
| `cic2018` | CIC-2018 PCAP train days |
| `warden_ft` | `warden`, then fine-tuned on the same CIC-2018 days (`bita/train.py --init_from`) |

```bash
python scripts/run_encoder_comparison.py \
    --warden-dir   <Warden *March_e.csv> \
    --pcap-root    <CIC-2018 <day>_pcap dirs> --cic2018-csv-dir <CIC-2018 CSVs> \
    --cic2017-dir  <CIC-2017 TrafficLabelling CSVs> \
    --out results/encoder_comparison --ip-ablation
```

The output is `results/encoder_comparison/comparison.md`. Arms are ranked on CIC-2017 technique
macro-F1 over seen classes, then risk ROC-AUC. Add `--dry-run` to print every command first. The
script is resumable: completed steps are skipped.

**Why Warden alone is not expected to win.** Warden records alerts, not traffic, and contains no
benign activity. Its loader has only a flow count, a port and a protocol: the flow count is copied
into bytes, packets and both rates, and return bytes, return packets and duration are always 0
(`bita/train.py::load_and_preprocess_dataset`). Those are the features that separate attacks from
normal flows in CIC. Its graph is also bipartite (attacker → victim), and CIC hosts are all
unseen nodes to it. `warden_ft` keeps what Warden pre-training learned and adapts it to flows.
The comparison says which of these matters in practice.

## Training the full system under the protocol

```bash
# encoder (use the arm the comparison ranked first; cic2018 shown)
python bita/train.py --pcap2018_root <pcap> --pcap2018_label_dir <csv> \
    --split_scheme cross_year --train_splits train

# Branch A
python scripts/retrain_branch_a_live.py --pcap-root <pcap> --cic2018-csv-dir <csv> \
    --cic2017-dir <cic2017> --split-scheme cross_year --tgne <encoder.pth> \
    --risk-objective soft_bce --risk-target hazard --output saved_models/branch_a/branch_a_lstm.pt \
    --results-json results/cross_year/branch_a.json

# Branch B, then DeepOP (DeepOP starts only if Branch B beats persistence)
python scripts/retrain_future_models_live.py --pcap-root <pcap> --cic2018-csv-dir <csv> \
    --cic2017-dir <cic2017> --split-scheme cross_year --tgne <encoder.pth> \
    --out-dir saved_models --risk-target hazard --results-json results/cross_year/future.json

# re-score any saved set of checkpoints on CIC-2017 without retraining
python scripts/evaluate_cross_dataset.py --cic2017-dir <cic2017> --tgne <encoder.pth> \
    --branch-a <branch_a.pt> --branch-b <host_wdt.pt> --deepop <cwa_forecast_decoder.pt> \
    --out results/cross_year/final
```

## Caveats to state with any result

- Reconnaissance cannot be learned from CIC-2018 (it has no PortScan class), so the model has never
  seen the first stage of the kill chain. The unseen-class report shows what it predicts instead.
- All 158,930 CIC-2017 PortScan records come from a single host (`172.16.0.1`).
- The corpora are mostly benign. Read accuracy against the majority-class baseline printed beside
  it, and use macro-F1, ROC-AUC and Brier as the headline.
