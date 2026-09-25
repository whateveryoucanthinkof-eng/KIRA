# 28 — Accuracy changes applied to this tree

**Date:** 2026-09-22
**Tree:** `Cyber_World-improved/`, a copy of `Cyber_World-main/`. The original is untouched.
**Tests:** 778 passed / 3 xfailed / 8 failed (8 = the branch's own baseline), against a baseline of 630 passed / 7 failed on the same machine.
The same 7 failures fail before and after; all are environmental (see §6).

Everything here is a **data, label, feature or measurement** change. No model architecture was
altered, because none of the architectural levers can be evaluated until the target means something.

---

## 0. Why this order

The committed benchmark could neither fail nor improve. `results/v4_benchmark.json` reports a test
split of **one host** at a **0.9997** base rate, recall 0.000, the model's nowcast ROC-AUC at 0.690
against a gradient-boosting baseline at 0.780, and per-step Brier ~0.99 against persistence at
0.00067. Tuning a model against that number would have optimised noise.

The reason the base rate is 0.9997 on corpora whose published attack fractions run 0.03%–57% is that
the pipeline was writing most of its own labels. So labels came first, features second, measurement
third.

---

## 1. Labels — `data_unification/multi_dataset_stream.py`

### 1.1 The pipeline stopped labelling its own training data

Two blocks overwrote `is_attack` from heuristics rather than from the corpus:

```python
if auth_metrics.get("auth_failed_count", 0.0) >= 5:            # -> CredentialAccess / T1110
if behavioral_metrics.get("port_mismatch_count", 0.0) >= 2:    # -> Execution / T1059
```

The second is the damaging one. `BehavioralFlowFingerprinter` calls a flow C2 beaconing at
`tot_bytes <= 1200`, 1–3 packets each way, mean packet size < 180, and calls that a port mismatch
whenever the destination port is not in `{80, 443, 8000, 8080, 8443}`. **Verified directly against
the fingerprinter:** two mDNS, NetBIOS, NTP or retrying-DNS exchanges lasting more than half a
second in one window satisfy both, so an entirely benign host was written down as under attack.
That traffic is on every LAN continuously.

It is fatal twice over. The labels stop being ground truth, so no accuracy number measures
detection; and the heuristic reads the same flow statistics the model is handed as input, so the
model was graded on re-deriving a function of its own features.

Now behind `heuristic_label_augmentation=False`. The capability is kept, the default is off, and
turning it on reproduces the old behaviour exactly (pinned by test).

### 1.2 Technique labels are the union, not whichever record sorted first

`coarse = atk_recs[0].coarse_category; techs = atk_recs[0].attck_technique_ids` discarded every
other attack flow in the window. That silently nullified the multilabel technique target
`cyberworld_v4/targets.py` builds downstream — it can only be as multilabel as the snapshot it
reads. `_window_attack_label()` now takes the **most severe** category present and the **union** of
techniques, ordered by first appearance so the result is deterministic.

### 1.3 `attack_role` is now a decision instead of an accident

`host_recs` matched either endpoint, so an attacker, its victim, and any benign server the attacker
touched all carried the same positive label. That is defensible for this product but it was never
chosen. Now `attack_role ∈ {either, target, source}`, default `either` (unchanged behaviour),
recorded in the results JSON so a benchmark states which target it scored.

### 1.4 Unmapped labels no longer train as confident negatives

`label_resolver` returns the third state `UNKNOWN` with `is_attack=False` and says in its own
comment that callers "must exclude these rows rather than treat them as benign".
`is_unresolved()` and `unresolved_report()` had **zero call sites in the repository**, so every
label the maps missed — a new attack name, a typo, a corpus the maps were never written for —
arrived at the loss as a confident negative.

New `data_unification/label_filter.py`, wired into `scripts/train_v4.py` and
`scripts/retrain_branch_a_live.py`. Coverage is printed per split, named labels and all, and an
unresolved rate above 5% is a credibility failure rather than a footnote.

### 1.5 PCAP labels can now name hosts, not just instants

`pcap_bridge` took `label_at(mid_ts)` and stamped it on every flow of every host in the window. A
CIC-2018 day directory holds ~445 per-host captures, so an attack hour labelled ~445 hosts when one
or two were involved — making the target nearly a function of the clock, which a model can learn
without looking at traffic.

- `AttackInterval` gains `participants`; `derive_windows` now recovers them **from the CSV** wherever
  the CSV has Src/Dst IP columns (all of CIC-2017, and `tue_20` of CIC-2018).
- `data_unification/attack_participants.py` supplies an auditable map for the nine CIC-2018 days
  whose CSVs have no addresses at all, plus `scripts/verify_attack_participants.py` to check a
  transcription against a day that *can* answer for itself.
- **`config/attack_participants.json` ships empty on purpose.** A guessed attacker/victim table
  would put real attack labels on the wrong hosts, which is worse than the defect it fixes. Days
  without an entry stay time-only, are logged as such, and carry `label_scope: "time_only"` in
  their metadata — the previous behaviour, now visible.

---

## 2. Features

### 2.1 The fan-out attributes no longer saturate at 147 — `host_attributes.py`

`unique_peers` and `unique_dst_ports` were `min(1, log1p(x) / 5.0)`, which reaches 1.0 at **147**
and stays there. A port scan covering 1,000 ports and one covering 148 were the identical feature
value: the most direct scan signature in the vector was flat across its whole discriminative range.

`PEER_COUNT_LOG_SCALE = 10.0` (saturates ~22,026, matching `flow_count`) and
`PORT_COUNT_LOG_SCALE = log1p(65535)` (reaches 1.0 exactly when the host has touched the whole port
space). Every divisor is now a named constant in `host_attributes.py`, imported by the computation
and asserted by the test, so a value can no longer drift from its documentation.

### 2.2 Destination port is log-scaled — `tgne_features.py`

`dst_port / 65535` put every service that carries signal into the bottom 1.5% of the feature:

| port | old | new |
|---|---|---|
| 22 | 0.00034 | 0.283 |
| 53 | 0.00081 | 0.360 |
| 80 | 0.00122 | 0.396 |
| 443 | 0.00676 | 0.550 |
| 3389 | 0.05171 | 0.733 |

SSH and HTTP were 0.0009 apart while two meaningless ephemeral ports 40,000 apart were 0.6 apart.
A port is really categorical and deserves indicator features or an embedding, which the 12-D edge
contract has no slots for; log scaling is the best available use of one slot, and it is monotone so
nothing that assumed ordering breaks.

### 2.3 The 30 packet-level features can now reach a model

`telemetry/packet/pcap_engine.py` has computed thirty per-host-window features since the sensor was
written — TTL variance, TCP window, `pkt_iat_cv`, `syn_no_response_ratio`, vertical/horizontal scan
scores, retransmission ratio, DNS domain entropy. `pcap_adapter` and `pcap_bridge` both called it
and then dropped the result:

```python
for host_ip, (flows, _packet_features) in per_host.items():
```

The compute cost was being paid on every PCAP run and no model had ever seen one, while the 15 flow
attributes that did reach the model **structurally cannot** express periodicity or scan shape.

They now travel on `record.metadata["packet_features"]` — one shared dict per host-window, which
also removes the per-record dict literal the schema's own docstring calls out as a 64 B/record cost.
`HostTrajectoryExtractor(include_packet_features=True)` produces a 45-wide attribute vector and a
57-D model input; `TrajectoryStoreBuilder` takes the width as a parameter instead of the hardcoded
`FEAT_DIM = 27` that silently truncated it, and refuses a mismatch loudly.

**Off by default**, because it changes the contract. Flipping it on is a retrain and
`CyberWorldConfig(host_attr_dim=45)`.

---

## 3. Measurement — `scripts/train_v4.py`, `cyberworld_v4/`

### 3.1 Host identity is no longer collapsed

Every HostId was built as `offline_host("mixed", "mixed", "mixed", ip)`, defeating the entire point
of `cyberworld_v4/identity.py` — whose docstring exists to stop 192.168.1.10 in CIC-2018 and the
same address in CTU-13 becoming one host — and leaving the bare IP as the only available split
group. `load_captures()` keeps `(dataset, scenario, capture)` per record and extracts trajectories
**per capture**, which also stops a CIC-2018 day and a CTU-13 scenario becoming each other's
temporal neighbours in one merged TGNE graph. Branch A already did this; v4 did not.

### 3.2 Splits come from the frozen lock

`data_unification/splits.lock.json` assigns 41 captures, stratified by measured attack fraction and
dealt 4:1:1. Every v3 trainer reads it; v4 recomputed its own from bare IPs, so v3 and v4 numbers
were never comparable. New `frozen_capture_split()` applies the lock and carves **CALIBRATION out of
the lock's train captures** — never validation, which would inherit model-selection optimism —
taking the most recent ones in time order. `--split chronological` keeps the old strategy, now
grouped by capture. The group bootstrap resamples captures too, which is what makes the confidence
interval exist at all.

### 3.3 Lead time and conformal coverage are actually reported

`cyberworld_v4/metrics/earlywarning.py` was written, tested, and **never called**; every "lead time"
the project reported was `forecast_steps × window_seconds`, a constant that describes the question
asked rather than any answer. `cyberworld_v4/conformal.py` likewise had no caller outside tests.

New `lead_time_from_samples()` turns a split's samples plus `[N, K]` forecasts into episodes at each
0→1 onset, scoring on `max_k P(A_t+k)`, with each episode seeing only the event-free windows since
the previous episode ended. The benchmark now prints detection rate, median/p10/p90 lead and
recall-at-lead, and fits a split-conformal interval on calibration only.

The forecast head also gets **its own** operating point, chosen on validation:
`thr` was fitted for P(attack now) and was being applied to P(attack at t+k).

### 3.4 The credibility gate covers the new failure modes

Added: unresolved-label rate above 5%; no onset ever warned about before it happened; conformal
coverage missing its target by more than 0.05. The "1 host group" message now says "capture group",
which is the unit actually being counted.

---

## 4. What must be retrained

`SCHEMA_VERSION` is **2.0.0**. §2.1 and §2.2 change the value at existing indices, which is the
silent kind of breakage — shapes still match and nothing raises, the encoder just receives an input
distribution it never saw. `build_or_load_tgne_ta` now refuses a 1.0.0 checkpoint and says which
features changed and why there is no conversion.

In order:

1. **TGNE encoder** — `bita/train.py`. Everything downstream consumes its latent.
2. **Branch A** — `scripts/retrain_branch_a_live.py`, now with label coverage reported per split.
3. **Branch B / DeepOP** — `scripts/retrain_future_models_live.py`.
4. **v4 benchmark** — `scripts/train_v4.py --split frozen`, which is the first run whose numbers can
   be reported at all.

Expect the measured attack rate to fall sharply once §1.1 is in effect. That is the point: it was
0.677 because the pipeline was inventing positives, and a lower, honest base rate is what makes
PR-AUC and lead time meaningful.

---

## 5. Deliberately not done

- **Encoder fine-tuning, latent width, TGN memory.** The 12-D latent is trained on link prediction
  and frozen, its width is pinned to the node-feature count by `tgn.py:82`, and `use_memory=False`
  disables the per-node state that is the whole premise of a host-trajectory model. All three are
  real levers and all three are only measurable against a working benchmark.
- **The forecast horizon.** Δt=2s × K=5 is a 10-second horizon; attack *stage* progression takes
  minutes to hours, which is why persistence is unbeatable and the gate fires. Changing it is a
  contract change that touches every checkpoint and deserves its own decision.
- **The IP ablation on the real corpus.** The harness is built and calibrated
  (`scripts/ablate_ip_features.py`, section 7 below) but the corpora are not on this machine, so
  the corpus arm has not been run against real addresses.

---

## 6. Test baseline

| | before | after |
|---|---|---|
| passed | 630 | **729** |
| failed | 7 | 7 |

The seven are identical before and after and all environmental in this copy:
`test_alert_thresholding` (the canonical retrained encoder checkpoint is not in the copy),
`test_split_manager_uses_the_lock` ×2 (the corpora are not on this machine),
`test_rollout_cache` ×2 and `test_spill_file_is_reclaimed` ×2 (they assert POSIX unlink-while-mapped
semantics, which Windows does not provide).

New test files:

| file | what it pins |
|---|---|
| `test_labels_are_ground_truth.py` | the heuristic overrides, the technique union, `attack_role` |
| `test_unresolved_labels_are_excluded.py` | UNKNOWN filtering, coverage reporting, and that the trainers call it |
| `test_packet_features_reach_the_model.py` | the 30 features, their normalisation, and participant scoping |
| `test_v4_capture_splits_and_leadtime.py` | the frozen split, four-way disjointness, episodes and lead time |
| `test_port_encoding_is_discriminative.py` | port resolution, monotonicity, and the schema bump |
| `test_ip_ablation_harness.py` | the ablation detects both leak types, names the carrier, and stays silent on clean data |


---

## 7. The IP-octet ablation

`scripts/ablate_ip_features.py`. **The corpora are not on this machine, so the corpus arm has not
been run.** What follows is the harness, its calibration on data where the answer is known, and the
one measurement that could be made against real trained weights.

### 7.1 The harness

Four arms. A/A2/A3 need no encoder and no GPU and are properties of the *data*; B and C need a
trained pipeline.

| arm | what it asks |
|---|---|
| A `ip_only` | train a classifier on the 12 IP features alone against the label — an upper bound on how much of the label the address recovers |
| A2 `subnet_only` | same with `octet3`/`octet4` masked — separates subnet topology from host identity |
| A3 `no_octets` | same with all four octets masked — what survives on the address-class flags alone |
| C `permuted` | consistently relabel addresses in the test split only; a behaviour model is unaffected, an identity model collapses |

Each arm runs twice: **transductive** (same hosts on both sides, which is the regime any row-level
offline split is really measuring) and **inductive** (disjoint hosts). The gap between them is the
memorisation exposure. Every arm also reports **permutation importance per IP feature**, because
"the address predicts the label" has two causes that want opposite responses.

Run it as:

```bash
python scripts/ablate_ip_features.py   --cic-dir ~/Documents/SIH/DATA/CSV   --ctu-dir ~/Documents/SIH/CTU-13-Dataset   --arms a --rows-per-file 200000 --stride 20
```

### 7.2 Calibration — does the instrument work?

Three corpora where the answer is known by construction, 40,000 host-rows each. Transductive ROC-AUC:

| scenario | A (all 12) | A2 (octet3/4 masked) | A3 (all octets masked) | attributed carrier |
|---|---|---|---|---|
| label independent of address | 0.4913 | 0.5000 | 0.5000 | none (max importance < 0.001) |
| public attacker, private victims — **CIC-2018's actual topology** | 1.0000 | 1.0000 | 1.0000 | **`is_private` (+0.496)** |
| attacker inside the victims' /24 | 0.9979 | **0.5000** | 0.5000 | **`octet4` (+0.493)** |

The instrument does not fire on clean data, fires on both leaks, and names the right carrier each
time. Masking `octet3`/`octet4` collapses an octet-carried leak to exactly chance and does nothing
to a class-flag-carried one.

### 7.3 What the calibration already settles

**On this corpus the octets are not the main address leak — `is_private` / `is_global` are.**

CIC-2018 stages its attacks from public AWS addresses against RFC1918 victims in 172.31.x.x. A
single boolean, IP feature index 2, separates attacker traffic from benign internal traffic
perfectly, and masking all four octets changes nothing. That is a property of how the capture was
built, not of the model, and no amount of octet masking touches it.

This **corrects the recommendation in the original review**, which proposed keeping the
address-class bits and dropping `octet3`/`octet4`. That is backwards for this corpus: it would
remove the smaller leak and keep the larger one. The octets matter where an attacker shares its
victims' subnet — lateral movement, insider scenarios, and the live SPAN deployment — which is
exactly where this product is meant to work and where the offline corpus provides least evidence.

### 7.4 The one real-weights measurement available

The shipped `branch_a_lstm.pt` loads. Its input layer is `lstm.weight_ih_l0`, shape (256, 27);
column *j* is every gate's sensitivity to input dimension *j*. Dimensions 0–11 are the TGNE latent —
the only path by which any IP feature can reach Branch A — and 12–26 are the flow attributes.

```
fresh init   : mean 1.1549  sd 0.0311   (8 seeds, same architecture)
trained      : mean 5.3732  sd 0.9698   -> 4.65x, so the layer did learn

TGNE latent  dims 0-11   share 44.3%   mean 5.3600
flow attrs   dims 12-26  share 55.7%   mean 5.3837
Welch t = -0.061, p = 0.952
```

**No detectable preference for either block.** The identity-bearing path is neither ignored nor
dominant at the input layer. This bounds the concern rather than resolving it: if latent reliance
had been ~0 the octets could not matter downstream, and it is not ~0.

Two incidental findings from the same ranking, both corroborating changes made elsewhere in this
document:

- `unique_dst_ports` ranks **26th of 27** and `flow_count` 27th. `unique_dst_ports` is the attribute
  that saturated at 147 (section 2.1) — the model learned to ignore a feature that was
  near-constant across its whole discriminative range. That is independent evidence for the fix.
- `unique_peers` ranks **3rd**, and it saturated at the same 147. The fan-out signal was being used
  as hard as a flattened feature allows.

### 7.5 What is still unmeasured

`concentration_report()` in the harness answers the question the audits raised — is Recon really
carried by one host, is C2 really about ten — and it needs the corpus. So does arm C, which is the
one that predicts live transfer. Until those run, the honest statement is: a class-flag shortcut is
**certain** from the corpus topology, an octet shortcut is **plausible and unquantified**, and
Branch A's architecture leaves room for both.


---

## 8. The three training fixes

Applied after the ablation, in ascending order of risk.

### 8.1 Samples are no longer mostly padding

`create_host_sequence_samples` and `LazyHostSequenceDataset` built a sample for
any host with two windows, left-padding the rest of the 15-step input with
zeros. The shipped checkpoint records `val_traj_len_median = 1.0` over
2,172,277 hosts, so **the median training example was fourteen-fifteenths
zeros**, and a zero vector is a valid point in feature space rather than a
"missing" marker -- the model cannot tell an absent step from a silent host.

Both builders now take `min_history_steps`, defaulting to the full window, and
both fill a `report` dict that the trainer prints. Padding is still reachable
with `min_history_steps=1`. `--min-history-steps` exposes it on the Branch A
trainer.

Expect the sample count to fall sharply. That is the fix working: those samples
were mostly zeros. The report says exactly how many went and warns when it is
more than half.

### 8.2 The forecast horizon is 150 seconds, not 10

`TemporalContract` gains `forecast_window_seconds` (default **30.0**), so a
forecast step covers fifteen 2-second input windows. History stays at 2s --
detection still sees fine detail, only the targets are coarsened, by
OR-aggregating each bucket (an attack anywhere in the bucket makes the bucket
an attack; techniques take the union; the latent state takes the mean).

Raising the step rather than `FORECAST_STEPS` is deliberate. K=150 at a 2s step
reaches the same horizon, but Branch B rolls out autoregressively and 150
compounding steps is a far worse estimator than 10 coarse ones -- and the
technique and hazard heads would each need 150 outputs.

`forecast_stride=1` reproduces the old behaviour exactly, and a checkpoint
written before the field existed is read as single-scale, which is what those
checkpoints were.

### 8.3 TGN memory: works, and is not free

Verified the path runs rather than assuming it: `use_memory=True` produces a
valid 12-D embedding and takes the encoder from 5,430 to 32,754 parameters.
The latent width is unchanged, so the downstream contract holds.

**But it is mutually exclusive with batch shuffling**, and shuffling is load
bearing. TGN slices batches contiguously in time, attacks are time-localised,
and 58.6% of 128-sample batches on `fri_16` hold a single class -- which is
what collapsed the category head in the first place. Turning memory on without
handling that gives a *worse* encoder.

The mitigation is `--backprop_every`: accumulating over consecutive batches
makes one update span `backprop_every x batch_size` samples and much more time,
restoring class contrast per update, while batch ORDER stays chronological so
the memory state is the one that actually existed.

    python bita/train.py --use_memory --no_shuffle_batches                          --backprop_every 8 --batch_size 128

The error now says all of this instead of only forbidding the combination, and
a warning fires on `--use_memory` with `--backprop_every < 4`.

### 8.4 Correction to section 5

Section 5 said the encoder has "no gradient path from the detection loss into
the representation". **That is wrong.** `bita/train.py` trains it with an
auxiliary category head over the five coarse classes at `cat_loss_weight=15.0`,
so the encoder does receive attack supervision during pretraining.

What is true is narrower: the *downstream* Branch A loss never reaches it,
because trajectories are extracted once into a memmapped `TrajectoryStore` and
training reads from that. End-to-end fine-tuning would mean re-running the
graph encoder every epoch, which is a restructure of the data path rather than
a flag, and was not attempted here.

### 8.5 Three checkpoints are now xfail

`test_served_checkpoint_contracts` marks Branch A, Branch B and DeepOP
`xfail(strict=True)`. They were trained at a 10-second horizon under feature
schema 1.0.0 and both have changed; the shapes are identical, so the contract
check is the only thing that can catch it. Strict markers mean a retrained
checkpoint fails the suite until its marker is removed, which is how the
finished retrain announces itself.


---

## 9. Four unused levers, wired

All four already existed as capability the code never exercised.

| | was | now |
|---|---|---|
| input scaling | none | per-feature standardisation, fitted on train, carried as buffers |
| `pos_weight` | a parameter nothing passed | computed from the train split, clamped |
| epoch selection | lowest multi-task val loss | highest mean val ROC-AUC (nowcast + forecast) |
| early stopping | none | `--patience`, default 3 |

### 9.1 The 27-D state had no common scale

Dims 0-11 are the TGNE graph-attention output and are unbounded. Dims 12-26
are the host attributes, every one clipped to [0, 1] by construction. An LSTM
gate sums them, so whichever block is larger dominates every gate and the
other is compressed against it. Nothing normalised the input; the existing
`LayerNorm` is on the encoder's OUTPUT.

Fitted statistics are held as buffers rather than a `LayerNorm` or
`BatchNorm`:

- `LayerNorm` here would normalise across the 27 features *within* a sample,
  mixing the two blocks and destroying overall magnitude -- and magnitude is
  signal in network traffic; a 100-flow window is not a 2-flow window.
- `BatchNorm` would make one inference depend on the rest of its batch, which
  serving cannot guarantee.
- Buffers land in `state_dict`, so serving inherits exactly the statistics
  training used. A normaliser refitted at serve time would be the same class
  of silent mismatch as the zeroed node features were.

Identity until fitted, so old checkpoints stay loadable and unchanged.

### 9.2 Measured effect

Controlled fixture, 27-D state with the real one's shape -- unbounded latent,
attributes squashed to (0, 1), signal split equally across both blocks, base
rate 0.12, latent block 60x the attribute block. Test ROC-AUC, mean of three
seeds:

| configuration | test ROC-AUC | delta |
|---|---|---|
| no normalisation, no pos_weight, select on val loss | 0.9787 +/- 0.0039 | — |
| + input normalisation | 0.9993 +/- 0.0003 | **+0.0206** |
| + pos_weight | 0.9995 +/- 0.0001 | +0.0209 |
| + select on val AUC | 0.9996 +/- 0.0000 | +0.0209 |

**Read this for what it is.** The fixture was built with the scale mismatch,
so it shows the mechanism works when the mismatch exists, not that the real
corpus has a 60x one. The mismatch is structurally guaranteed -- unbounded
against [0, 1] -- but its magnitude on real data is unmeasured. The trainer
now prints the fitted latent/attr ratio at startup and flags it when the two
blocks are more than 3x apart, so the first retrain answers this directly.

The last two rows are nearly flat here because the fixture is close to
separable once scaled. They are kept because they are principled rather than
because this fixture rewards them: `pos_weight` is the standard BCE correction
for a base rate the shipped run measured at 0.2263, and selecting on the
metric being reported is what Branch A already does
(`claude_latest_analysis/27`) and this trainer did not.

### 9.3 Node-feature ablation

`CYBERWORLD_ABLATE_NODE_FEATURES`, mirroring the existing
`CYBERWORLD_ABLATE_EDGE_FEATURES`. It exists because section 7 found the
address leak is **not** where it was assumed: masking all four octets changes
nothing, while `is_private` alone carries ROC-AUC 1.0 on CIC-2018's topology.

Group shorthands: `address_class` (is_private, is_global), `host_identity`
(octet3, octet4), `octets` (all four).

    CYBERWORLD_ABLATE_NODE_FEATURES=address_class python bita/train.py ...

Zeroing rather than dropping keeps the 12-D node contract. `bias` is refused:
an all-zero row is how the encoder recognises an unknown node, so zeroing it
changes the meaning of every other row instead of removing one feature.

Expect this one to **lower** offline AUC and raise live AUC. It removes a
shortcut that works on the corpus and not on a network where every host is
internal.

### 9.4 Still not done

Decoupling the latent width from the node-feature count (`tgn.py:82`,
`embedding_dimension = n_node_features`) is the remaining capacity lever. It
changes the graph-attention output dimension, which cannot be validated
without the corpora, so it was left alone rather than changed blind.
