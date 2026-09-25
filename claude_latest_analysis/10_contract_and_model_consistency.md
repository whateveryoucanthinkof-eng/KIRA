# 10 — Contract and model consistency audit

**Question:** if you retrain everything on fresh data tomorrow, will the models, trainers
and serving path agree on one temporal contract and one feature schema?

**Answer: no, not yet.** The v4 contract is correctly plumbed into the *window size* almost
everywhere, and into *history/forecast* only on the two `scripts/retrain_*_live.py` paths.
The three in-tree trainers, all three shipped checkpoints, the serving adapter's advertised
metadata and the whole split layer still describe a different model.

Authoritative contract — `cyberworld_v4/config.py:29-46`:
`window_seconds=2.0`, `history_steps=15`, `forecast_steps=5`, `TGNE_LATENT_DIM=12`,
`HOST_ATTR_DIM=15`, `STATE_DIM=27`, `MAX_GAP_SECONDS=6.0`, `SEEDS=(42,123,2024,3407,9001)`.

### Headline numbers

| Measure | Result |
|---|---|
| Checkpoints on disk that `contract.validate_checkpoint` accepts | **3 of 6** |
| Checkpoints the serving path actually loads that it accepts | **0 of 3** |
| Trainers that use v4 `history_steps`/`forecast_steps` | **2 of 5** (both `scripts/retrain_*_live.py`) |
| Trainers that read `splits.lock.json` | **0 of 5** |
| Trainers that use v4 splits / targets / metrics | **0 of 3** three-branch trainers (config only, as expected) |
| Competing definitions of "the contract" in-tree | **3** (`cyberworld_v4/config.py`, `data_unification/temporal_config.py`, `model_contract.py`) |
| Feature dimension agreement (12 + 15 = 27) | **unanimous** — no dimension violation found |
| Host-attribute *names* matching computed *values* | **0 of 15** |

---

## 1. Compliance matrix

Verdicts: **AGREES** = matches v4 and is bound to it. **COINCIDES** = matches v4 today but
via an independent literal, so it will drift. **VIOLATES** = disagrees now.
**NOT-ENFORCED** = nothing would fail if it disagreed.

### 1.1 Temporal constants

| Component | Expected | Actual | Verdict | Evidence |
|---|---|---|---|---|
| v4 source of truth | 2.0 / 15 / 5 | 2.0 / 15 / 5 | AGREES | `cyberworld_v4/config.py:29-31` |
| `data_unification/temporal_config.py` | no second definition | `LIVE_WINDOW_SIZE_SEC=2.0`, `DEFAULT_ROLLOUT_HORIZON_LIVE=8`, `DEFAULT_HISTORY_STEPS=5`, `MACRO_WINDOW_SIZE_SEC=60.0`, `DEFAULT_ROLLOUT_HORIZON_MACRO=4` | **VIOLATES** | `data_unification/temporal_config.py:21-26` |
| `temporal_config.validate_temporal_contract` | enforce 15 | defaults `requested_history=model_history=DEFAULT_HISTORY_STEPS` (5) | **VIOLATES** | `data_unification/temporal_config.py:51-75` |
| `model_contract.py` | no temporal keys | dims only, no temporal keys — correct scope | AGREES | `model_contract.py:6-12` |
| Branch A trainer — window | 2.0 from contract | `get_contract().window_seconds` | AGREES | `branch_a_gnn_lstm/train_branch_a.py:154-156` |
| Branch A trainer — history | 15 from contract | `seq_len=get_contract().history_steps` | AGREES | `train_branch_a.py:161,172,177,178` |
| Branch A trainer — forecast | 5 | **no horizon at all**; target is `snapshots[end_idx]`, i.e. t+1 | **VIOLATES** | `branch_a_gnn_lstm/sequence_dataset.py:98,118` |
| Branch A trainer — checkpoint stamp | contract embedded | saves `model_state_dict`/`epoch`/`metrics` only | **VIOLATES** | `train_branch_a.py:269-276` |
| `HostSequenceDataset` / `create_host_sequence_samples` | — | `seq_len: int = 5` defaults | NOT-ENFORCED | `sequence_dataset.py:57,79` |
| Branch B trainer — T, K | 15, 5 | `T: int = 4, K: int = 4` (both in the sample builder and the entry point) | **VIOLATES** | `branch_b_world_model/train_branch_b.py:48,81-82` |
| Branch B trainer — window | 2.0 from contract | `get_contract().window_seconds` | AGREES | `train_branch_b.py:96-98` |
| Branch B trainer — checkpoint stamp | contract embedded | `wdt_state_dict`,`risk_head_state_dict`,`epoch`,`k1_mse`,`k4_mse` | **VIOLATES** | `train_branch_b.py:201-210` |
| `HostWorldDynamicsTransformer.max_horizon` | ≥5 | `8` | COINCIDES (headroom) | `branch_b_world_model/rollout_encoder_decoder.py:107` |
| `rollout(K=…)` default | 5 | `K: int = 4` | **VIOLATES** | `rollout_encoder_decoder.py:193`, `rollout_multi_host`: `:276` |
| `max_context_len` | ≥15 (history is 15) | `10` — **truncates a 15-step history to its last 10** | **VIOLATES** | `rollout_encoder_decoder.py:195,215` and `:280,304` |
| DeepOP trainer — K | 5 | `K: int = 4` in builder, entry point and argparse | **VIOLATES** | `deepop_decoder/train_cwa_decoder.py:49,113,332` |
| DeepOP trainer — window | 2.0 from contract | `get_contract().window_seconds` | AGREES | `train_cwa_decoder.py:130-132` |
| DeepOP trainer — checkpoint stamp | contract embedded | no temporal keys (`vocab_size`, metrics only) | **VIOLATES** | `train_cwa_decoder.py:308-319` |
| `DeepOPForecastDecoder.pos_embed` | ≥5 rows | `max_seq_len=16`, sliced `[:, :S, :]` | COINCIDES (headroom) | `deepop_decoder/forecast_decoder.py:104,123,171` |
| `scripts/retrain_branch_a_live.py` | 2.0/15/5, stamped | contract-bound throughout; writes `training_contract` + full `config` | **AGREES** | `:156,195,220,225,284-294` |
| `scripts/retrain_future_models_live.py` — Branch B | 2.0/15/5, stamped | `T=_c.history_steps, K=_c.forecast_steps`; saves all three keys | **AGREES** | `:167-169,182,184,196` |
| `scripts/retrain_future_models_live.py` — DeepOP | 2.0/15/5, stamped | correct K and T, but saves **only** `forecast_steps` + `window_seconds` | **VIOLATES** | `:210-212` vs `:241` |
| `control_backend/model_adapter.py` init | from contract | `LIVE_WINDOW_SIZE_SEC` / `DEFAULT_ROLLOUT_HORIZON_LIVE` (8) / `DEFAULT_HISTORY_STEPS` (5) | **VIOLATES** | `control_backend/model_adapter.py:107-109` |
| `model_adapter` contract check | refuse to run | computes `get_contract().matches(...)` and **logs a string** | **NOT-ENFORCED** | `model_adapter.py:188-202` |
| `model_adapter` DeepOP contract adoption | `_adopt_contract` called | **not called** for DeepOP (is for branch_a `:160`, branch_b `:168`) | **VIOLATES** | `model_adapter.py:182-184` |
| `control_backend/main.py` `ModelMetadata` | from adapter | hardcoded `history_steps=5, window_seconds=2.0, forecast_steps=8` | **VIOLATES** | `control_backend/main.py:150-152` |
| Web dashboard | derive from backend | derives `steps × window` from `ModelMetadata`; no hardcoded steps | AGREES | `web_dashboard/src/api/adapter.ts:220-228`, `pages/Predictions.tsx:99-100` |
| `telemetry/state/state_builder.py` | 2.0 from contract | `window_sec: float = 2.0` literal | COINCIDES | `telemetry/state/state_builder.py:19` |
| `HostTrajectoryExtractor` default window | contract | `window_size_sec: float = 60.0` (every caller overrides) | NOT-ENFORCED | `data_unification/multi_dataset_stream.py:134` |
| `FlowToTemporalEventAdapter` default window | contract | `window_size_sec: float = 60.0` (overridden at `multi_dataset_stream.py:237`) | NOT-ENFORCED | `data_unification/flow_to_temporal_event.py:63` |
| `MAX_GAP_SECONDS=6.0` | enforced on all trajectories | consumed **only** by `cyberworld_v4/targets.py:96,166` → `scripts/train_v4.py` | **NOT-ENFORCED** on all three branches | `multi_dataset_stream.py` has no gap logic |
| `SEEDS` (5 seeds) | multi-seed runs | tuple is defined and copied into config; **never iterated anywhere** | **NOT-ENFORCED** | `cyberworld_v4/config.py:42,121`; `retrain_future_models_live.py:286` hardcodes 42 |

### 1.2 Feature dimensions

| Component | Expected | Actual | Verdict | Evidence |
|---|---|---|---|---|
| TGNE latent | 12 | 12 | AGREES | `data_unification/tgne_features.py:46`, `model_contract.py:6` |
| Host attrs | 15 | 15 | AGREES | `tgne_features.py:47`, `multi_dataset_stream.py:135,175` |
| Branch A input | 27 | 27 | AGREES | `branch_a_gnn_lstm/lstm_multitask.py:105`, `train_branch_a.py:185` |
| Branch B `d_latent` | 12 | 12 | AGREES | `train_branch_b.py:118-119` |
| DeepOP `d_latent` | 12 | 12 | AGREES | `train_cwa_decoder.py:169` |
| Runtime shape assertion | 12 / 27 | asserted at inference | AGREES | `model_adapter.py:408-412` |
| TGNE edge features | 12 named | 12, names identical in both configs | AGREES | `tgne_features.py:27`, `saved_models/…_config.json` |
| **Host attr *names* vs *values*** | names describe what is computed | **15 names, 0 of which describe the computed value** | **VIOLATES** | `data_unification/host_attributes.py:6-22` vs `multi_dataset_stream.py:151-220` |
| TGNE `num_categories` | one value | `81` (`bita/saved_models/bita_config.json`), `6` (`saved_models/…unified_cic_ctu13_config.json`), `4` (code default) | **VIOLATES** | `train_branch_a.py:86` |

### 1.3 Checkpoints — measured, not inferred

Every artefact was loaded and passed through `cyberworld_v4.contract.validate_checkpoint`:

| Checkpoint | Embedded contract | `validate_checkpoint` |
|---|---|---|
| `saved_models/branch_a/branch_a_lstm.pt` *(served)* | `training_contract{window_size_sec:2.0, history_steps:5}`, no forecast | **REFUSED** — `history=5 forecast=None` |
| `saved_models/branch_a/branch_a_lstm.full_corpus.pt` | full v4 `config.temporal` 2.0/15/5 | **PASS** |
| `saved_models/branch_a/branch_a_lstm.ctu13_only.pt` | full v4 `config.temporal` 2.0/15/5 | **PASS** |
| `saved_models/branch_b/host_wdt.pt` *(served)* | `window_seconds:2.0, history_steps:5, forecast_steps:8` | **REFUSED** — `history=5 forecast=8` |
| `saved_models/deepop/cwa_forecast_decoder.pt` *(served)* | `window_seconds:2.0, forecast_steps:8` | **REFUSED** — `history=None forecast=8` |
| `saved_models/v4/forecaster.pt` | full v4 `config.temporal` 2.0/15/5 | **PASS** |

The three checkpoints the live adapter loads are exactly the three that fail. The two that
pass on the Branch A side are the `full_corpus` / `ctu13_only` variants, which nothing serves.

JSON manifests disagree with both the contract and their own `.pt` files:

| Manifest | Declares | Verdict |
|---|---|---|
| `saved_models/branch_a/branch_a.manifest.json` | `window_size_sec:2.0, history_steps:5`, no forecast; `split_policy:"file_disjoint_80_20"` | **VIOLATES** |
| `saved_models/branch_b/branch_b.manifest.json` | `delta_t_sec:2.0, history_steps:5, forecast_horizon_steps:8` | **VIOLATES** |
| `saved_models/deepop/deepop.manifest.json` | `delta_t_sec:2.0, forecast_horizon_steps:8`, no history | **VIOLATES** |

Note the key name: the manifests say `delta_t_sec`, which
`contract.validate_checkpoint` does not look for (it accepts `window_seconds` or
`window_size_sec`, `contract.py:68,75`). Even a manifest with correct *values* under
`delta_t_sec` would be read as `<missing>`.

`scripts/offline_contract_check.py` reads these manifests but asserts only `vocab_size`,
`nhead` and `num_encoder_layers` against `model_contract.py` — it never looks at a temporal
key, which is why all three have stayed wrong while the check reports
`[+] All offline contract checks passed.`

### 1.4 Splits, labels, vocabularies

| Component | Expected | Actual | Verdict | Evidence |
|---|---|---|---|---|
| `splits.lock.json` consumers | trainers read it | **zero callers** of `load_lock` / `split_of` / `captures_for` outside the module itself | **NOT-ENFORCED** | `data_unification/split_policy.py:56-82` |
| Branch A trainer split | frozen lock | `get_split_manager()` — hardcoded filename lists | VIOLATES | `train_branch_a.py:144`; `split_manager.py:52-155` |
| Branch B trainer split | frozen lock | `ScientificSplitManager()` | VIOLATES | `train_branch_b.py:88` |
| DeepOP trainer split | frozen lock | `get_split_manager()` | VIOLATES | `train_cwa_decoder.py:122` |
| `retrain_branch_a_live.py` split | frozen lock | file-order 70/15/15 | VIOLATES | `scripts/retrain_branch_a_live.py:144-150` |
| `retrain_future_models_live.py` split | frozen lock | file-order 80/20 (`:51-52`) and day-order 80/20 (`:79-80`), **no test split** | VIOLATES | as cited |
| `scripts/train_v4.py` split | frozen lock | `cyberworld_v4.splits.chronological_split` | VIOLATES (different axis) | `scripts/train_v4.py:161` |
| `split_manager.py` portability | — | `CIC2018_DIR = r"C:\SIH_DATA\..."` — a Windows path on a Linux box, so CIC-2018 is silently skipped | **VIOLATES** | `data_unification/split_manager.py:29,43` |
| Branch A technique vocab | one vocabulary | 14 classes | — | `sequence_dataset.py:17-32` |
| DeepOP joint vocab | one vocabulary | 10 tokens (3 special + 7 pairs) | AGREES with `model_contract.DEEPOP_VOCAB_SIZE=10` | `joint_vocab.py:20-28` |
| Branch A vs DeepOP label space | reconcilable | 14-way fine-grained vs 7-way macro; no shared mapping module | **VIOLATES** | see §2.7 |
| `GRADATION_LEVELS` coverage | covers every coarse category | missing `"UNKNOWN"` (resolver emits uppercase) and `"CredentialAccess"` | **VIOLATES** | `sequence_dataset.py:35-45,127` vs `label_resolver.py:14,128` |
| Brute-force coarse category | one answer | resolver → `InitialAccess`; joint vocab → `CredentialAccess` | **VIOLATES** | `label_resolver.py:109` vs `joint_vocab.py:58-59` |

---

## 2. Violations and their consequences

### 2.1 `data_unification/temporal_config.py` is a second, contradictory source of truth
`config.py:10-12` says "No other module may define these constants." `temporal_config.py`
defines all of them, with different values: `DEFAULT_HISTORY_STEPS=5` and
`DEFAULT_ROLLOUT_HORIZON_LIVE=8` (`:24-26`). Worse, its own
`validate_temporal_contract()` (`:51-75`) defaults both `requested_history` and
`model_history` to 5, so calling it with no history arguments *validates the v3 contract and
passes*. `model_adapter.py:107-109` seeds the live serving contract from this file.

**If unfixed:** the serving process starts life believing history is 5 and horizon 8, and
only the checkpoint's own metadata can talk it out of that. A checkpoint that omits a key
(as the DeepOP one does) leaves the v3 default standing. You will serve a 15-step model with
a 5-step buffer and get no error.

### 2.2 The three in-tree trainers still default to T=4 / K=4
`train_branch_b.py:48,81-82` and `train_cwa_decoder.py:49,113,332`. They import
`get_contract()` — but only for `window_seconds`. Horizon and history remain literals.

**If unfixed:** a "full retrain" run through `train_branch_b.py` / `train_cwa_decoder.py`
produces a 4-step model, while a run through `scripts/retrain_future_models_live.py`
produces a 5-step model. Both write to `saved_models/branch_b/` and
`saved_models/deepop/`. Nothing in the filename or the checkpoint distinguishes them, and
§2.5 means nothing refuses the wrong one at load.

### 2.3 `max_context_len=10` silently truncates the 15-step history
`rollout_encoder_decoder.py:195,215` (and `:280,304` for the multi-host path) slice
`curr_seq[:, -max_context_len:, :]`. With `history_steps=15` the transformer sees the last
**10** steps and the first 5 are discarded — no warning, no shape error.

**If unfixed:** you pay the full cost of building 15-step histories and the model is
trained and served on 10. The advertised "30 s of causal history" is really 20 s. Every
Branch B number is for a contract that is written down nowhere.

### 2.4 Three of three served checkpoints fail `validate_checkpoint`
Measured above. Branch A's served `.pt` records `history_steps:5`; Branch B records
`forecast_steps:8`; DeepOP records no history at all.

**If unfixed:** the moment anything calls `validate_checkpoint` on the serving path, the
system stops booting. Today nothing does, so the failure is invisible — which is the worse
state of the two.

### 2.5 The serving adapter checks the contract and then ignores the result
`model_adapter.py:188-202` computes `get_contract().matches(...)` and passes the boolean into
a **log format string** (`"v4" if served else "v3 (…)"`). `contract.py:4-8` states that
enforcement means refusing to run. It does not refuse.

Two further defects in the same function:
- **DeepOP's contract is never adopted.** `_adopt_contract` is called at `:160` (branch_a)
  and `:168` (branch_b) but not at `:182-184` (deepop). The cross-checkpoint conflict
  detector at `:231-238` therefore cannot see DeepOP, so a DeepOP decoder trained at K=4
  composed with a WDT rolled out at K=5 raises nothing.
- **Ordering.** `self.extractor` is constructed at `:148-151` with the *provisional*
  `self.window_seconds`, and `_adopt_contract` only updates the attribute at `:240-242`.
  The extractor keeps the pre-adoption window. Harmless today (both are 2.0); wrong the
  first time a checkpoint carries a different window.

**If unfixed:** every guard in this file is advisory. The one genuinely load-bearing
behaviour is the conflict check at `:231-238`, and it has a DeepOP-shaped hole in it.

### 2.6 `control_backend/main.py:150-152` hardcodes the contract the UI displays
`ModelMetadata(history_steps=5, window_seconds=2.0, forecast_steps=8)` — literals, next to
`feature_count=model_contract.BRANCH_A_INPUT_DIM` which *is* sourced properly.

The dashboard is innocent here and does the right thing: `adapter.ts:220-228` computes
`steps × window` from whatever the backend sends. That is precisely why this matters — the UI
faithfully renders `8 × 2.0 = 16 s`, which after a v4 retrain will be a **10 s** horizon.
The adapter already knows the true numbers (`model_adapter.py:606-612`); `main.py` does not
ask it.

**If unfixed:** the SOC console reports a 16-second forecast horizon for a model that
forecasts 10 seconds. An operator acts on a lead time that is 60 % longer than the real one.

### 2.7 `host_attributes.py` names describe features that do not exist
`HOST_ATTRIBUTES` (`host_attributes.py:6-22`) lists `bytes_in_rate`, `pkts_in_rate`,
`conn_in_rate`, `tcp_flags_syn/ack/fin/rst`, `avg_payload_in/out`, `conn_duration_avg`.
`compute_host_temporal_attributes` (`multi_dataset_stream.py:151-220`) computes, in order:
`flow_count`, `log1p(fwd_bytes)`, `log1p(bwd_bytes)`, `log1p(total_bytes)`,
`log1p(fwd_packets)`, `log1p(bwd_packets)`, `log1p(total_packets)`, `unique_peers`,
`unique_dst_ports`, `tcp_ratio`, `udp_ratio`, `avg_duration`, `byte_rate`, `packet_rate`,
`connection_density`.

The count matches (15), so nothing breaks. **Not one position matches semantically.** In
particular **no TCP flag is computed anywhere** — positions 8-11 are advertised as
SYN/ACK/FIN/RST and actually hold `unique_peers`, `unique_dst_ports`, `tcp_ratio`,
`udp_ratio`. `tgne_features.py:44` re-exports the wrong list as
`HOST_TEMPORAL_ATTR_NAMES`, which is the name any explainability or feature-importance
consumer would reach for.

**If unfixed:** every feature attribution produced by this system is mislabelled. A report
saying "this host was flagged on `tcp_flags_syn`" is really reporting `unique_dst_ports`.
That is a conclusion-level error, not a cosmetic one — and it survives a retrain untouched,
because the dimension is right.

### 2.8 Label vocabularies disagree across branches
- **`UNKNOWN` collapses to Benign.** `label_resolver.py:14` defines `UNKNOWN_CATEGORY =
  "UNKNOWN"` (uppercase) and returns it at `:128` for every label the maps miss.
  `GRADATION_LEVELS` (`sequence_dataset.py:35-45`) has a `"Unknown"` key (mixed case) but no
  `"UNKNOWN"`, and `:127` does `GRADATION_LEVELS.get(target_snap.coarse_category, 0)` → **0
  = Benign**. The explicit-unknown policy (`config.py:112 unknown_policy="explicit_unknown"`)
  is defeated by a case mismatch, and unresolved labels become confident benign training
  targets — the exact failure mode the v3 fix note at `label_resolver.py:113-124` says was
  removed.
- **Brute force has two coarse categories.** `label_resolver.py:109` maps
  brute/patator → `("InitialAccess", ["T1110"])`. `joint_vocab.py:58-59` maps T1110 →
  `("CredentialAccess", "T1110")`. Branch A trains gradation 2 (InitialAccess) on the same
  event DeepOP tokenises as CredentialAccess. `"CredentialAccess"` is also absent from
  `GRADATION_LEVELS`, so if the resolver is ever corrected to emit it, those samples silently
  become gradation 0.
- **Two granularities, no bridge.** Branch A predicts 14 techniques; DeepOP predicts 7
  macro classes. `consolidate_network_technique` maps fine→macro but nothing maps back, and
  no test asserts the two stay reconcilable.

**If unfixed:** Branch A and DeepOP can report contradictory categories for the same host in
the same window, and unresolved labels quietly inflate the benign class.

### 2.9 `MAX_GAP_SECONDS` and `SEEDS` are declared but unused
`MAX_GAP_SECONDS=6.0` is consumed only by `cyberworld_v4/targets.py:96,166`, reachable only
from `scripts/train_v4.py`. `multi_dataset_stream.extract_trajectories` has no gap logic at
all — `create_host_sequence_samples` sorts by `window_idx` and slides, so a 102-minute gap
becomes one 2-second transition, which is the defect `config.py:44-45` names explicitly.
`SEEDS` is copied into config and never iterated; `retrain_future_models_live.py:286`
hardcodes 42.

**If unfixed:** three-branch trajectories contain fabricated transitions across capture
gaps, and every reported number is single-seed with no variance estimate.

### 2.10 `splits.lock.json` is built but wired to nothing
`freeze_splits.py` wrote the lock (inventory + `assignment`, stratified 4:1:1 by attack
fraction). `split_policy.py:56-82` exposes `load_lock`/`split_of`/`captures_for`. **No module
imports any of them.** Meanwhile five different splits are computed independently
(§1.4 table), one of which (`split_manager.py:29`) points at `C:\SIH_DATA\...` and so drops
CIC-2018 entirely on this machine without saying so.

**If unfixed:** the freeze buys nothing. Branch A, Branch B and DeepOP numbers remain
mutually incomparable, and "test" still means a different set of captures per branch.

### 2.11 Branch A has no forecast horizon at all
`sequence_dataset.py:98` iterates `end_idx in range(1, n_snaps)` and takes the target from
`snapshots[end_idx]` — always exactly one step ahead. `forecast_steps=5` is never consulted
on this path. The docstring at `:96-97` shows this was deliberately moved off the *current*
window to fix a nowcast bug, but it stopped at t+1.

**If unfixed:** Branch A is a 1-step (2-second) predictor presented alongside 5-step
(10-second) models. Comparing their forecast metrics is meaningless, and a fresh retrain
will not change this — it is structural, not a constant.

### 2.12 A fresh DeepOP retrain will *still* produce a refused checkpoint
This is the one forward-looking defect worth isolating, because it survives the retrain you
are planning. `retrain_future_models_live.py:241` saves
`{"decoder_state_dict", "epoch", "forecast_steps", "window_seconds", "vocab_size"}` — no
`history_steps`. `contract.validate_checkpoint` falls through to the top-level branch
(`contract.py:73-78`), gets `history_steps=None`, and `TemporalContract.matches` raises
`TypeError` on `int(None)` → returns `False` → `ContractViolation`.

Branch B's save one function up (`:196`) includes all three keys and will pass. So after a
clean full retrain you get: Branch A **PASS**, Branch B **PASS**, DeepOP **REFUSED**. One
added dictionary key is the difference.

### 2.13 Two TGNE checkpoints with incompatible category heads
`bita/saved_models/bita_config.json` declares `num_categories: 81`;
`saved_models/bita_bigru_transformer-unified_cic_ctu13_config.json` declares `6`;
`train_branch_a.py:86` defaults to `4`. The guard at `:97-100` validates
`feature_schema_version`, `edge_feat_dim` and `node_feat_dim` — all three agree at 12 — but
**not** `num_categories`, `message_dimension` (12 vs 100) or `memory_dimension` (12 vs 9).
`load_state_dict(..., strict=True)` at `:128` will catch a real shape conflict, so this fails
loudly rather than silently; the risk is that which TGNE you get depends on an env var
(`TGNE_CHECKPOINT_PATH`, `:70`) and is not recorded in any downstream checkpoint.

**Flagged, not verified:** I did not confirm which of the two TGNE checkpoints produced the
currently-shipped Branch A/B/DeepOP weights. Nothing in those checkpoints records it.

---

## 3. Remediation checklist, in dependency order

Each step assumes the ones above it are done. Steps 1-4 are prerequisites for a retrain;
5-8 must land before the retrained models are served; 9-12 are correctness work that a
retrain will not fix by itself.

**Phase 1 — before you retrain (these change what the model learns)**

1. **Delete the competing contract.** Reduce `data_unification/temporal_config.py` to
   re-exports of `cyberworld_v4.config`, or remove it. Specifically kill
   `DEFAULT_HISTORY_STEPS=5` (`:26`) and `DEFAULT_ROLLOUT_HORIZON_LIVE=8` (`:24`), and either
   delete `validate_temporal_contract` (`:51-75`) or strip its defaulted history arguments.
   Everything below depends on there being one answer.
2. **Fix `max_context_len`.** `rollout_encoder_decoder.py:195` and `:280` → default to
   `get_contract().history_steps`, or drop the parameter. *Do this before retraining* —
   at 10 you would train a 15-step model on 10 steps of context and never know.
3. **Bind T and K in the three in-tree trainers.** `train_branch_b.py:48,81-82`,
   `train_cwa_decoder.py:49,113,332`, `rollout_encoder_decoder.py:193,276` → take
   `get_contract().history_steps` / `.forecast_steps`. These three files are the only
   remaining places a retrain can silently pick 4.
4. **Wire the frozen split.** Have `split_manager.py` resolve capture → split through
   `split_policy.split_of` instead of its hardcoded lists (`:52-155`), fix or remove the
   `C:\SIH_DATA` path (`:29`), and replace the ad-hoc splits in
   `retrain_branch_a_live.py:144-150` and `retrain_future_models_live.py:51-52,79-80`.
   Must precede the retrain: changing the split afterwards invalidates the run.

**Phase 2 — before you serve the result**

5. **Add `history_steps` to the DeepOP save.** `retrain_future_models_live.py:241`. One key.
   Without it the retrain's own output is refused (§2.12).
6. **Stamp the three in-tree trainers' checkpoints.** `train_branch_a.py:269-276`,
   `train_branch_b.py:201-210`, `train_cwa_decoder.py:308-319` → embed
   `DEFAULT_CONFIG.to_dict()` (or call `contract.stamp`), matching what
   `retrain_branch_a_live.py:284-294` already does.
7. **Make the serving path refuse.** In `model_adapter._load_models`: call
   `validate_checkpoint` on each of the three checkpoints; add the missing
   `self._adopt_contract(ckpt, "deepop")` at `:184`; move the `self.extractor` construction
   (`:148-151`) to after contract adoption; replace the log-only check at `:188-202` with
   `contract.validate(...)`. Requires 5 and 6 first, or the process will not boot.
8. **Source `ModelMetadata` from the adapter.** `control_backend/main.py:150-152` → read
   `model_adapter.history_steps / .window_seconds / .forecast_steps`. Also regenerate the
   three `*.manifest.json` files and rename `delta_t_sec` → `window_seconds` so
   `validate_checkpoint` can actually read them. Extend
   `scripts/offline_contract_check.py` to assert the temporal block, not just `nhead`.

**Phase 3 — correctness the retrain will not fix**

9. **Fix the host-attribute names.** Rewrite `host_attributes.py:6-22` to the 15 values
   `multi_dataset_stream.py:175-219` actually computes, or implement the advertised
   features. Until then no feature attribution from this system can be quoted. Add a test
   asserting `len(HOST_ATTRIBUTES) == HOST_ATTR_DIM` *and* that the names match the
   docstring order at `multi_dataset_stream.py:158-173`.
10. **Unify the label vocabularies.** Make `GRADATION_LEVELS` (`sequence_dataset.py:35-45`)
    cover `"UNKNOWN"` and `"CredentialAccess"`, and make `:127` raise on an unknown key
    rather than defaulting to 0. Reconcile brute force: `label_resolver.py:109` and
    `joint_vocab.py:58-59` must agree on one coarse category. Add a test that every coarse
    category emitted by `LabelResolver` and by `consolidate_network_technique` is a key in
    `GRADATION_LEVELS`.
11. **Enforce `MAX_GAP_SECONDS` on the three-branch path.** Apply the segmentation logic
    from `cyberworld_v4/targets.py:81-96` inside `extract_trajectories` /
    `create_host_sequence_samples`, so gaps break trajectories instead of becoming
    transitions.
12. **Decide Branch A's horizon.** Either extend `sequence_dataset.py:98,118` to produce
    targets at t+`forecast_steps` (making it comparable with B and DeepOP), or document it
    as a nowcaster and stop reporting its numbers alongside forecast metrics. Then wire
    `SEEDS` into a multi-seed loop so the comparison carries an interval.

---

## 4. What I could not verify

- **Which TGNE checkpoint produced the shipped Branch A/B/DeepOP weights.** Two configs
  exist with `num_categories` 81 and 6; the selection is an env var
  (`train_branch_a.py:70`) and is not recorded in any downstream checkpoint.
- **Whether the `splits.lock.json` inventory is complete.** I read the assignment structure
  and the CIC2017/CIC2018 entries, not every capture; I did not re-run `freeze_splits.py`
  or confirm the on-disk corpus still matches the frozen inventory.
- **Runtime behaviour of any of this.** Every verdict above is from source and from loading
  the six `.pt` files. No trainer or server was executed.
- **`bita/` internals.** `bita/train.py` seeds independently (`:341` `np.random.seed(2020)`,
  `:394-395` `args.seed`) and has its own splitter; I treated it as the upstream TGNE trainer
  and did not audit it against the v4 contract, which does not appear to govern it.
