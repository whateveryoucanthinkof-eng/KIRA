# CyberWorld

## Final Architecture, Scientific Corrections, and Implementation Specification

This document follows the deep architecture and scientific audit of the current repository.

The objective is to turn CyberWorld into a **scientifically coherent, reproducible, technically strong short-horizon cyber-state forecasting system** whose claims are supported by its data construction, prediction targets, architecture, evaluation protocol, calibration, and deployment path.

The canonical temporal contract for the entire project is now:

```yaml
window_seconds: 2
history_steps: 15
history_seconds: 30
forecast_steps: 5
forecast_seconds: 10

```

That means:

```text
30 seconds of causal network history
        ↓
predict the next 5 × 2-second states
        ↓
10-second forecast horizon

```

This contract is authoritative and must be enforced throughout training, validation, calibration, testing, replay, checkpoint loading, and live inference.

---

# 1. What exactly is CyberWorld?

### Question

What is the precise research and engineering problem CyberWorld should solve?

### Answer

CyberWorld should be defined as:

> **A short-horizon probabilistic cyber-state forecasting system that consumes the recent temporal evolution of network telemetry and predicts current threat state, future attack onset risk, future network/host state, and likely future MITRE ATT&CK techniques.**

The central research problem is **forecasting evolving cyber behavior**, not simply classifying an already-observed attack.

The system therefore has four distinct capabilities:

1. **Current-state estimation**
   - Is an attack happening now?
   - What attack/technique is currently observable?
2. **Future attack forecasting**
   - Is an attack likely to occur in the next few seconds?
   - At which future horizon is the probability increasing?
3. **Cyber-state/world forecasting**
   - What will the host/network state look like over the next five 2-second steps?
4. **Behavior/ATT&CK forecasting**
   - Given the predicted future state, which attack techniques are likely to appear?

The project must not claim that it “predicts attacks” merely because it recognizes currently active attack traffic.

---

# 2. What is the fundamental mathematical contract?

### Question

What should the entire model mathematically represent?

### Answer

The canonical formulation should be:

```math
X_{t-L+1:t} \rightarrow \left\{ \begin{array}{l} P(A_t=1) \\ P(T_{\text{onset}}\le t+k) \\ \hat{S}_{t+k} \\ P(Y_{t+k,j}=1) \end{array} \right. \quad k=1,\dots,5
```

where:

- `X` = observed network telemetry/graph,
- `L=15` = history length,
- `A_t` = current attack state,
- `T_{\text{onset}}` = future attack onset time,
- `S` = latent/world state,
- `Y_j` = ATT&CK technique `j`,
- `K=5` = forecast horizon.

The canonical timing is therefore:

```text
State t-14
State t-13
...
State t-1
State t
        ↓
15 states × 2 seconds
        ↓
30 seconds causal history
        ↓
predict:
t+1
t+2
t+3
t+4
t+5
        ↓
5 × 2 seconds
        ↓
10-second forecast horizon

```

Every training, validation, checkpoint, replay, and live inference component must obey this same contract.

---

# 3. Should 60-second windows remain anywhere in the main pipeline?

### Question

The existing training pipeline uses approximately 60-second windows in places while live inference uses 2-second windows. Should this be retained?

### Answer

**No.**

The promoted research model must use:

```text
2-second temporal windows

```

throughout the end-to-end pipeline.

No Branch A, Branch B, DeepOP, dataset builder, validation path, or live adapter should silently use 60-second windows.

If 60-second aggregate behavior is later found scientifically useful, it must be introduced as an explicitly separate macro-timescale feature/model, not as an accidental difference between training and deployment.

The primary model must be:

```text
TRAIN       = 2s
VALIDATION  = 2s
CALIBRATION = 2s
TEST        = 2s
LIVE        = 2s
REPLAY      = 2s

```

---

# 4. How much history should the model see?

### Question

Why 15 × 2 seconds?

### Answer

Use:

```text
L = 15

```

giving:

```text
30 seconds of causal history

```

This is now the canonical history length.

The model therefore receives:

```text
15 consecutive 2-second states

```

before producing a future forecast.

The key requirement is that all components must use exactly the same history semantics.

Do not allow one component to use:

```text
5 states

```

while another silently uses:

```text
4 states

```

or:

```text
60-second aggregated context

```

The architecture must have one authoritative history contract.

---

# 5. What should the forecast horizon be?

### Question

How far into the future should CyberWorld predict?

### Answer

Use:

```text
K = 5

```

with:

```text
2 seconds per future step

```

Therefore:

```text
Step 1 = +2 sec
Step 2 = +4 sec
Step 3 = +6 sec
Step 4 = +8 sec
Step 5 = +10 sec

```

The primary model is therefore:

```text
30 seconds observed history
        ↓
10 seconds future forecast

```

The model must be **trained for K=5**, not trained for a smaller horizon and then extended during deployment.

The evaluation must be horizon-aware:

```text
@2s
@4s
@6s
@8s
@10s

```

---

# 6. What exactly should “forecasting” mean?

### Question

How do we distinguish forecasting from current-state classification?

### Answer

The target must occur strictly after the prediction cutoff.

Given:

```text
Input:
t-14, t-13, ..., t-1, t

```

the forecast target is:

```text
t+1
t+2
t+3
t+4
t+5

```

where each state represents a 2-second interval.

Never create a future-forecasting experiment where:

```text
input includes state t
target = label(t)

```

That is current-state classification/nowcasting, not future forecasting.

The current-state detector can still exist, but it must be explicitly named:

```text
Current-State Head

```

and separated from:

```text
Future Forecast Head

```

---

# 7. What should be the central predictive target?

### Question

What should demonstrate that CyberWorld genuinely provides early warning?

### Answer

The central future target should be:

```math
P(T_{\text{attack onset}}\le t+k\mid X_{\le t})
```

for each:

```math
k\in\{1,2,3,4,5\}.
```

This means the model answers:

> Given the previous 30 seconds of network behavior, how likely is attack onset/activity within the next 2, 4, 6, 8, and 10 seconds?

This is much more meaningful than simply predicting whether the current state is malicious.

---

# 8. Should “risk” remain one scalar?

### Question

Can the existing manually constructed 0–1 severity score remain the primary risk target?

### Answer

No.

There should be separate variables:

```text
current_attack_probability
future_attack_hazard
cumulative_future_attack_probability
severity_score
current_attack_state

```

Each must have explicit statistical meaning.

The model must not use a hand-authored severity value and call it:

```text
probability
hazard
confidence

```

unless it genuinely has that interpretation.

---

# 9. What should hazard mean?

### Question

If hazard terminology is retained, what exactly does it mean?

### Answer

For future attack onset:

```math
h_k = P(T_{\text{onset}}=t+k\mid T_{\text{onset}}>t+k-1,X_{\le t})
```

Then:

```math
P(T_{\text{onset}}\le t+K) = 1-\prod_{k=1}^{K}(1-h_k)
```

with:

```math
K=5.
```

Therefore:

```text
h1 = onset hazard at +2s
h2 = onset hazard at +4s
h3 = onset hazard at +6s
h4 = onset hazard at +8s
h5 = onset hazard at +10s

```

Do not compute:

```python
max(step_risks)

```

and call the result cumulative probability.

If the maximum is useful, name it:

```text
peak_hazard

```

---

# 10. Should severity disappear?

### Question

Is there still a place for severity?

### Answer

Yes, but as a separate task.

The system may output:

```text
attack probability
attack hazard
severity

```

where:

```text
probability:
"How likely is the event?"

hazard:
"At which future step is onset likely?"

severity:
"How severe is the predicted event?"

```

If severity is based on a manually defined ontology, explicitly document it as an operational score rather than empirical probability.

---

# 11. What should Branch A actually do?

### Question

What is the correct role of Branch A?

### Answer

Branch A should be the:

> **Current-state and short-horizon threat estimation module operating over the 30-second temporal context.**

It should have explicit heads such as:

```text
Head 1:
current attack state

Head 2:
future attack hazard

Head 3:
future attack probability

Head 4:
current/future ATT&CK multilabel prediction

Head 5:
optional auxiliary state variables

```

Do not collapse everything into one ambiguous “risk head.”

---

# 12. Should Branch A predict current or future ATT&CK techniques?

### Question

Should technique prediction refer to the current state or the future?

### Answer

Both can exist, but they must be different tasks.

Current:

```math
P(Y_{t,j}=1\mid X_{\le t})
```

Future:

```math
P(Y_{t+k,j}=1\mid X_{\le t})
```

for:

```math
k=1,\dots,5.
```

The future technique head is the more important research capability.

---

# 13. Should ATT&CK be single-label or multilabel?

### Question

Can the existing `technique_ids[0]`-style behavior remain?

### Answer

No.

ATT&CK should be represented as **multilabel**.

A time step can correspond to several techniques.

Represent the target as:

```math
y\in\{0,1\}^{N_{\text{techniques}}}
```

with multilabel loss.

Primary evaluation:

```text
micro-F1
macro-F1
mAP
precision@k
recall@k

```

Accuracy should not be the primary technique metric.

---

# 14. How should ATT&CK tactics be represented?

### Question

Can ATT&CK tactics be treated as a fixed attack progression?

### Answer

No.

Do not encode ATT&CK itself as:

```text
Recon
→ Initial Access
→ Execution
→ C2
→ Impact

```

as though every attack follows one mandatory ordinal path.

Instead maintain separate concepts:

```text
CyberWorld operational progression
ATT&CK tactic(s)
ATT&CK technique(s)
attack family
severity

```

If CyberWorld uses an internal ordinal progression score, clearly identify it as a CyberWorld-defined ontology rather than ATT&CK ground truth.

---

# 15. What happens to labels that cannot be resolved?

### Question

Should unknown labels become attack labels?

### Answer

Absolutely not.

Use:

```text
BENIGN
ATTACK
UNKNOWN
UNLABELED

```

An unknown label must never silently become:

```text
is_attack = True

```

Dataset preparation must generate a mapping report containing:

```text
total labels
mapped labels
unmapped labels
ambiguous labels
discarded labels

```

This makes label uncertainty explicit.

---

# 16. Are all ATT&CK techniques observable from network telemetry?

### Question

Can network flow alone directly identify every ATT&CK technique?

### Answer

No.

Techniques should be categorized as:

```text
directly observable
indirectly inferable
not observable from current telemetry

```

Some behaviors primarily require:

```text
process telemetry
filesystem telemetry
authentication telemetry
endpoint telemetry
user interaction telemetry

```

rather than network-flow data.

CyberWorld may infer some of them, but the claim must be:

> inferred from network behavior

rather than:

> directly observed.

---

# 17. What is the canonical state representation?

### Question

What exactly is one 2-second state?

### Answer

Define:

```text
State_t =
{
    timestamp,
    host nodes,
    communication edges,
    flow features,
    packet/byte statistics,
    protocol information,
    port information,
    attack labels,
    ATT&CK labels,
    metadata
}

```

Every dataset must be normalized into this canonical 2-second representation before model-specific processing.

There must be exactly one authoritative preprocessing implementation:

```text
CanonicalFlowProcessor

```

and both offline and live inference must use it.

---

# 18. How should host identity work across datasets?

### Question

Can raw IP addresses be used as global host identity?

### Answer

No.

Use namespaced identity:

```text
dataset_id
+
scenario_id
+
capture_id
+
native_host_id

```

then map this to a deterministic internal identifier.

A private IP such as:

```text
192.168.1.10

```

must not automatically represent the same host across unrelated captures.

---

# 19. Should Python `hash()` be used for persistent IDs?

### Question

Can Python's built-in `hash()` be used?

### Answer

No.

Use deterministic identifiers such as:

```text
SHA-256
UUID5
canonical-string deterministic integer

```

The same source entity must produce the same identifier across:

```text
machines
processes
runs
environments

```

---

# 20. How should temporal gaps be handled?

### Question

What if activity has a large gap between observations?

### Answer

Every state should carry temporal information such as:

```text
delta_t

```

If a gap exceeds the allowed continuity threshold:

```text
terminate the sequence

```

Do not treat:

```text
12:00:04 → 13:42:00

```

as an ordinary 2-second transition.

The 15-state history must represent 15 genuinely consecutive 2-second states, or explicitly masked/gapped states according to a documented policy.

---

# 21. What should the graph mean?

### Question

What exactly should TGNE model?

### Answer

The canonical graph should represent:

```text
nodes = hosts/entities
edges = observed communications/interactions

```

with consistent semantics across:

```text
training
validation
calibration
test
replay
live

```

Do not use one graph interpretation for training and another for live inference without an explicit mathematical mapping.

---

# 22. Should Warden pretraining remain?

### Question

Is pretraining useful?

### Answer

Potentially yes.

But it must be treated as:

```text
pretraining
→ domain adaptation
→ target-domain evaluation

```

not as though the Warden representation is automatically equivalent to the target production representation.

Run:

```text
without pretraining
vs
with pretraining

```

and keep pretraining only if it demonstrably improves useful generalization.

---

# 23. What should Branch B actually model?

### Question

What makes Branch B a real cyber-world model?

### Answer

It should forecast future latent cyber state:

```math
\hat S_{t+1:t+5} = f_\phi(S_{t-14:t})
```

and ideally use structured state components:

```math
S_t=(H_t,E_t,A_t,C_t)
```

where:

- `H_t` = host state,
- `E_t` = network edges,
- `A_t` = attack state,
- `C_t` = communication/traffic state.

The main minimum target is:

```text
future host state
future attack state

```

with future topology/edge prediction added if genuinely implemented and evaluated.

---

# 24. Is per-host prediction enough to call it a cyber-world model?

### Question

Can independent host forecasting be called a full cyber-world model?

### Answer

Not strongly.

A host-centric latent world model is defensible.

A stronger multi-host world model should model:

```math
G_t\rightarrow G_{t+k}
```

and be able to reason about interactions such as:

```text
Host A compromised
       ↓
new communication
       ↓
Host B
       ↓
Host B becomes affected

```

Do not claim full multi-host cyber-world modeling until that capability is actually trained and evaluated.

---

# 25. What should happen to MultiHostInteractionLayer?

### Question

Should it remain because it exists in the codebase?

### Answer

No.

Either:

```text
A. integrate it into the actual training/evaluation path

```

or:

```text
B. remove it from the promoted architecture

```

Unused architectural claims should not appear in the final system diagram.

---

# 26. How should DeepOP/CWA be trained?

### Question

What future states should the ATT&CK decoder see during training?

### Answer

It should ultimately consume the same distribution of future states produced during inference.

Avoid:

```text
TRAIN:
ground-truth future latent + noise

LIVE:
predicted future latent

```

Instead move toward:

```text
observed 30s history
        ↓
World Model
        ↓
predicted t+1 ... t+5 states
        ↓
DeepOP/CWA
        ↓
future ATT&CK prediction

```

This removes the largest train/serve discrepancy.

---

# 27. Should teacher forcing be used?

### Question

Should training always use ground-truth previous state?

### Answer

Not for the final deployment-style model.

A reasonable training strategy is:

```text
early:
teacher forcing for stability

later:
scheduled sampling / predicted-state rollout

final:
deployment-style autoregressive rollout

```

Then evaluate:

```text
oracle rollout
vs
predicted rollout

```

to measure error accumulation.

---

# 28. What should the world-model forecast length be?

### Answer

Exactly:

```text
K = 5

```

for the promoted experiment.

Thus:

```text
S_t+1
S_t+2
S_t+3
S_t+4
S_t+5

```

with each state representing another 2-second interval.

Do not train with K=4 and call the deployed system a five-step forecaster.

If K=1–4 is used for debugging or ablation, it must be explicitly identified as such.

---

# 29. How should Branch B be evaluated?

### Question

Is latent-space MSE enough?

### Answer

No.

Evaluate:

### Latent prediction

```text
MAE
RMSE
cosine similarity

```

### Security-state prediction

```text
attack-state PR-AUC
future attack probability
future technique performance

```

### Graph prediction, if enabled

```text
edge precision/recall
edge AP
next-victim recall

```

A world model should ultimately be judged by whether its predicted future state improves meaningful downstream cyber forecasting.

---

# 30. What should the early-warning metric actually be?

### Question

Does a 10-second forecast horizon mean 10 seconds of warning?

### Answer

No.

10 seconds is the maximum forecast horizon.

Actual lead time is:

```math
L_i = t_{\text{attack onset},i} - t_{\text{first valid alert},i}.
```

For every attack episode calculate:

```text
first valid alert
attack onset
lead time

```

Then report:

```text
median lead time
mean lead time
P10
P90
false alarms/hour

```

---

# 31. What should count as a valid alert?

### Question

How should the first alert be defined?

### Answer

Define an explicit policy such as:

```text
alert if cumulative attack probability ≥ threshold
for N consecutive 2-second prediction steps

```

or another precisely documented policy.

The threshold and persistence rule must be chosen using validation/calibration data and frozen before test evaluation.

---

# 32. How should false alarms be measured?

### Answer

Use:

```math
false\ alarms/hour
```

alongside:

```text
precision
recall
lead time

```

A model that predicts attacks constantly is not operationally useful even if recall is high.

Important operating-point results should include:

```text
Recall at fixed false alarms/hour

```

and/or:

```text
Median lead time at fixed false alarms/hour

```

---

# 33. What should the calibration system be?

### Question

Can the current fixed ±0.05 interval remain?

### Answer

No.

Probability calibration must be fitted on a dedicated calibration set.

Evaluate methods such as:

```text
temperature scaling
isotonic regression

```

for probabilities.

For predictive intervals, implement a real conformal calibration process rather than a fixed-width interval.

The pipeline must explicitly contain:

```text
TRAIN
CALIBRATION
TEST

```

and the test set must not be used to fit calibration.

---

# 34. How should conformal prediction work?

### Question

Can temporal samples simply be shuffled into calibration/test sets?

### Answer

No.

Use a time-respecting protocol:

```text
earlier calibration period
        ↓
later untouched test period

```

and document:

```text
nonconformity score
quantile
target coverage
empirical coverage
interval width
coverage by horizon

```

The claim should be about measured coverage under the defined protocol rather than an unconditional guarantee.

---

# 35. How should probability calibration be evaluated?

### Answer

Report:

```text
ECE
Brier score
NLL
reliability diagrams

```

for:

```text
current attack probability
future attack probability
future technique probability

```

before and after calibration.

---

# 36. What should the dataset split look like?

### Question

Should we randomly split rows?

### Answer

No.

Use meaningful independent units:

```text
capture
scenario
campaign
host session
temporal block

```

The canonical split is:

```text
TRAIN
VALIDATION
CALIBRATION
TEST

```

with the final test set untouched until all modeling decisions are frozen.

---

# 37. How should cross-dataset generalization be evaluated?

### Answer

Run at least:

### A. In-domain temporal generalization

Train and test on different chronological/scenario partitions.

### B. Cross-dataset transfer

Train on one dataset family and test on another.

### C. Cross-scenario generalization

Hold out complete attack scenarios/campaigns.

This distinguishes memorization from actual generalization.

---

# 38. What should the strongest generalization experiment be?

### Answer

Prefer:

```text
TRAIN:
Dataset/Scenario group A

VALIDATION:
Dataset/Scenario group B

CALIBRATION:
Later unseen validation period

TEST:
Completely held-out scenario/capture/domain C

```

The checkpoint must be frozen before accessing the final test set.

---

# 39. What baselines are mandatory?

### Answer

At minimum:

```text
1. Majority-class baseline
2. Persistence / last-state baseline
3. Logistic Regression
4. XGBoost
5. Small MLP
6. GRU/LSTM
7. Temporal-only model
8. Graph-only model
9. Full CyberWorld

```

For world forecasting additionally:

```text
last-state persistence
moving average
simple autoregressive model
GRU/LSTM forecaster

```

For future ATT&CK:

```text
last-technique baseline
most-frequent technique
Markov transition baseline
simple sequence model

```

---

# 40. What ablation study is mandatory?

### Answer

At minimum:

```text
A0:
XGBoost baseline

A1:
Temporal LSTM

A2:
TGNE + temporal model

A3:
TGNE + threat forecasting

A4:
TGNE + world model

A5:
TGNE + world model + ATT&CK decoder

A6:
Full ML model + SOC rules

```

Measure:

```text
PR-AUC
Brier
ECE
false alarms/hour
lead time
future ATT&CK F1/mAP

```

This demonstrates which components actually matter.

---

# 41. Should the rules engine remain?

### Answer

Yes, but as a separate layer:

```text
Research Model
      ↓
Probabilistic Predictions
      ↓
SOC Policy / Rules Layer
      ↓
Alerting / Operator / UI

```

Evaluate separately:

```text
ML-only
Rules-only
Hybrid

```

The hybrid system can be the production system, but neural predictions and deterministic rule contributions must remain distinguishable.

---

# 42. What about the “armed attack” flag?

### Answer

It may exist in:

```text
simulation mode
demo mode
replay mode

```

but must never be available to the autonomous research benchmark.

The benchmark path must contain no oracle:

```text
attack_active = ground_truth

```

input.

The experiment must evaluate what the model knows from telemetry alone.

---

# 43. How should offline/live parity be validated?

### Answer

Given the same raw telemetry:

```text
offline pipeline
→ canonical feature tensor

```

and:

```text
live pipeline
→ canonical feature tensor

```

must produce equivalent outputs.

Tests must compare:

```text
window construction
history length
future alignment
feature ordering
normalization
missing-value handling
timestamps
node IDs
edge IDs
tensor shapes

```

This should become a CI test.

---

# 44. How should the temporal contract be enforced?

### Answer

Create a central executable configuration:

```python
TemporalContract(
    window_seconds=2,
    history_steps=15,
    forecast_steps=5
)

```

with derived values:

```python
history_seconds = 30
forecast_seconds = 10

```

Every component should validate itself against this contract.

If any component attempts:

```text
60-second window
history != 15
K != 5

```

it should fail loudly rather than silently continuing.

---

# 45. What should checkpoint metadata contain?

### Answer

Every checkpoint must contain:

```yaml
experiment_id:
git_commit:
dataset_manifest_hash:
config_hash:
seed:
python_version:
torch_version:
cuda_version:

window_seconds: 2
history_steps: 15
history_seconds: 30
forecast_steps: 5
forecast_seconds: 10

feature_schema_hash:
label_schema_hash:
host_identity_scheme:

optimizer:
learning_rate:
batch_size:
weight_decay:

train_split_hash:
validation_split_hash:
calibration_split_hash:
test_split_hash:

```

A checkpoint without this metadata is not considered a reproducible research artifact.

---

# 46. What should the experiment registry look like?

### Answer

Use a single immutable registry:

```text
experiments/
  EXP-001/
    config.yaml
    dataset_manifest.json
    metrics.json
    calibration.json
    checkpoint.pt
    git_commit.txt

  EXP-002/
    ...

```

Every reported result must map to an experiment ID.

---

# 47. How should random seeds be handled?

### Answer

Use multiple seeds for the final benchmark, for example:

```text
42
123
2024
3407
9001

```

and report:

```text
mean
standard deviation
confidence intervals

```

Seed values must be stored in experiment metadata.

---

# 48. How should statistics handle overlapping sliding windows?

### Answer

Do not treat every overlapping window as statistically independent.

For example:

```text
[t0 ... t14]
[t1 ... t15]
[t2 ... t16]

```

are highly correlated.

Bootstrap confidence intervals at meaningful independent units such as:

```text
campaign
capture
host session
attack episode

```

rather than individual overlapping windows.

---

# 49. What should happen to extremely high technique accuracy?

### Answer

Treat any existing value such as:

```text
99.9% accuracy

```

as a diagnostic result until all of the following are ruled out:

```text
same-time target leakage
class imbalance
duplicate overlap
scenario leakage
feature leakage
label leakage

```

Headline metrics should instead be:

```text
macro-F1
micro-F1
mAP
per-class recall
PR-AUC
class support

```

---

# 50. What should explainability actually explain?

### Answer

Explanations must use the actual causal model input:

```text
15 × 2-second states

```

not a single synthetic snapshot.

Use:

```python
model.eval()

```

during explanation.

The explanation should ideally identify:

```text
important time steps
important graph nodes
important graph edges
important features
important forecast horizons

```

and clearly separate:

```text
ML contribution
SOC-rule contribution

```

---

# 51. Should dashboard metrics be labeled?

### Answer

Yes.

Every dashboard value must be explicitly marked as one of:

```text
OBSERVED
PREDICTED
ESTIMATED
SIMULATED

```

Synthetic telemetry must never visually imply direct measurement.

---

# 52. What should the final architecture look like?

```text
                     RAW NETWORK TELEMETRY
                              │
                              ▼
                   ┌────────────────────┐
                   │ Canonical 2s State │
                   └─────────┬──────────┘
                             │
                             ▼
                     Temporal Graph
                       Construction
                             │
                             ▼
                   ┌────────────────────┐
                   │ Temporal Graph     │
                   │ Encoder (TGNE)     │
                   └─────────┬──────────┘
                             │
                    30-second state
                        representation
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
       Current State    Future Attack     Future ATT&CK
          Head             Hazard             Head
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    ┌────────────────┐
                    │ Cyber World    │
                    │ Dynamics Model │
                    └───────┬────────┘
                            │
                predicted future states
                 t+1 ... t+5
                            │
                  ┌─────────┴──────────┐
                  │                    │
                  ▼                    ▼
          Future Network/Host     DeepOP / CWA
               State              ATT&CK Decoder
                  │                    │
                  └─────────┬──────────┘
                            ▼
                    Future Forecast Trace
                            │
                            ▼
                       Calibration
                            │
                            ▼
                     SOC Policy Layer
                            │
                            ▼
                         UI / API

```

The key architectural principle is:

> **Prediction first, policy second.**

---

# 53. What should be removed from the promoted architecture?

Remove or demote anything that is merely legacy complexity:

```text
unused V3.1 / 72-D path
unused model variants
unused losses
duplicate preprocessing
unused graph modules
dead calibration implementations
stale temporal configurations
unused feature definitions

```

Keep audit/research tooling that is actually used for scientific validation.

The repository should have one obvious promoted path.

---

# 54. What should the research-production separation be?

## Research model

```text
canonical input
→ TGNE
→ current detector
→ future hazard
→ world model
→ future ATT&CK decoder
→ calibrated predictions

```

## Production/SOC layer

```text
predictions
+
rules
+
thresholding
+
alert suppression
+
operator policy
+
routing

```

This separation preserves both scientific validity and production usefulness.

---

# 55. What should the primary benchmark table contain?

The final benchmark should contain horizon-aware metrics:

| ModelCurrent PR-AUCFuture PR-AUC @2s\@4s\@6s\@8s\@10sECEFalse alarms/hrMedian lead time |   |   |   |   |   |   |   |   |   |
| --------------------------------------------------------------------------------------- | - | - | - | - | - | - | - | - | - |
| Persistence                                                                             |   |   |   |   |   |   |   |   |   |
| XGBoost                                                                                 |   |   |   |   |   |   |   |   |   |
| LSTM                                                                                    |   |   |   |   |   |   |   |   |   |
| TGNE                                                                                    |   |   |   |   |   |   |   |   |   |
| TGNE + World Model                                                                      |   |   |   |   |   |   |   |   |   |
| Full CyberWorld                                                                               |   |   |   |   |   |   |   |   |   |

A separate table should report future ATT&CK multilabel performance.

---

# 56. What should the strongest scientific figure be?

A central figure should show:

```text
Forecast horizon

2s → 4s → 6s → 8s → 10s

```

versus:

```text
PR-AUC
Brier score
calibration error

```

for:

```text
baseline
LSTM
TGNE
full CyberWorld

```

Another key figure should show:

```text
false alarms/hour
        vs
median early-warning lead time

```

This directly communicates the practical value of the model.

---

# 57. What would convince a skeptical research reviewer?

Not:

```text
99.9% accuracy

```

and not:

```text
10-second forecast = 10 seconds guaranteed warning

```

Instead:

> “Using the previous 30 seconds of network telemetry, CyberWorld estimates current attack state and forecasts attack onset, future network state, and likely ATT&CK behavior over the next 10 seconds. The model is evaluated on previously unseen temporal scenarios using leakage-resistant chronological splits, held-out calibration, multiple baselines, component ablations, horizon-wise metrics, attack-episode lead-time measurements, and reproducible experiment manifests.”

That is a defensible research statement.

---

# 58. What is the central research hypothesis?

The first hypothesis should be:

> **Temporal graph representations of the previous 30 seconds of network activity contain predictive information about near-future cyber-state transitions beyond static or purely temporal baselines.**

The second should be:

> **Explicit prediction of future cyber-state evolution improves future attack/ATT&CK forecasting compared with directly predicting future behavior without an explicit world-state model.**

These are falsifiable hypotheses.

That is important.

---

# 59. What is the most important implementation principle?

> **Do not add architectural complexity until the labels, temporal alignment, and evaluation protocol are correct.**

CyberWorld already has sufficient architectural sophistication.

The next gains should come from:

```text
better target definition
better temporal consistency
better dataset hygiene
better calibration
better evaluation
better reproducibility
better train/serve parity

```

not from adding additional Transformer blocks merely to make the model sound more advanced.

---

# 60. What is the ideal end-to-end data flow?

```text
RAW FLOWS / PCAP / TELEMETRY
            │
            ▼
       CANONICALIZER
            │
            ▼
      2-SECOND STATES
            │
            ▼
15 STATE HISTORY = 30 SECONDS
            │
            ▼
       HOST/EDGE GRAPH
            │
            ▼
      TEMPORAL ENCODER
            │
            ▼
          S_t
            │
     ┌──────┼──────┐
     │      │      │
     ▼      ▼      ▼
 Current  Future  Future
 Threat   Hazard  Techniques
     │      │      │
     └──────┼──────┘
            ▼
       WORLD MODEL
            │
            ▼
      S_t+1 ... S_t+5
            │
            ▼
      ATT&CK DECODER
            │
            ▼
       FORECAST TRACE
            │
            ▼
        CALIBRATION
            │
            ▼
       POLICY / RULES
            │
            ▼
          SOC/UI

```

---

# 61. What should the training pipeline be?

The canonical experiment pipeline must be:

```text
Dataset construction
        ↓
Schema validation
        ↓
Leakage audit
        ↓
Temporal/scenario split
        ↓
TRAIN
        ↓
VALIDATION
        ↓
CALIBRATION
        ↓
Freeze model/thresholds
        ↓
UNTOUCHED TEST
        ↓
Multi-seed evaluation
        ↓
Confidence intervals
        ↓
Experiment registry

```

No test-set tuning.

No calibration on test.

No threshold optimization on test.

No architecture selection based on final test performance.

---

# 62. What should happen if CyberWorld loses to a baseline?

Keep the result.

If:

```text
XGBoost = 0.89 PR-AUC
CyberWorld = 0.86

```

that is scientifically useful.

CyberWorld should investigate why rather than modifying the evaluation until CyberWorld wins.

Research credibility comes from a trustworthy methodology, not from forcing a favorable result.

---

# 63. What constitutes a successful CyberWorld v4?

CyberWorld should not be considered scientifically complete until:

### Temporal contract

```text
2s state
15-step history
30s history
5-step forecast
10s horizon

```

is enforced everywhere.

### Forecasting

Future targets are strictly after the prediction cutoff.

### Labels

ATT&CK is multilabel and unknown labels are explicit.

### Identity

Dataset/scenario/capture identity is preserved.

### Evaluation

Chronological/scenario-held-out test exists.

### Calibration

Calibration is performed on held-out calibration data.

### Reproducibility

Checkpoint → config → dataset → code commit is traceable.

### Train/serve parity

Offline replay and live preprocessing are equivalent.

### Baselines

At least XGBoost + LSTM + persistence baselines exist.

### Ablation

World model / encoder / decoder contribution is measured.

### Early warning

Actual attack-onset lead time is calculated from timestamps.

### Hybrid rules

ML-only, rules-only, and hybrid results are separable.

### Horizon evaluation

Performance is reported at:

```text
2s
4s
6s
8s
10s

```

---

# 64. Canonical configuration

This is the authoritative starting configuration:

```yaml
project:
  name: CyberWorld
  version: v4

temporal:
  window_seconds: 2
  history_steps: 15
  history_seconds: 30
  forecast_steps: 5
  forecast_seconds: 10

targets:
  current_attack: true
  future_attack_hazard: true
  future_attack_probability: true
  future_techniques: true
  future_host_state: true
  future_edges: optional

attack_labels:
  representation: multilabel
  unknown_policy: explicit_unknown

identity:
  namespace:
    - dataset
    - scenario
    - capture
    - host

splits:
  strategy: chronological_and_scenario_held_out
  train: true
  validation: true
  calibration: true
  test: untouched

evaluation:
  seeds:
    - 42
    - 123
    - 2024
    - 3407
    - 9001

  detection:
    - pr_auc
    - roc_auc
    - precision
    - recall
    - f1
    - false_alarms_per_hour

  forecasting:
    - pr_auc_by_horizon
    - brier_by_horizon
    - nll_by_horizon

  technique:
    - micro_f1
    - macro_f1
    - map
    - precision_at_k
    - recall_at_k

  calibration:
    - ece
    - brier
    - nll

  early_warning:
    - median_lead_time
    - mean_lead_time
    - p10_lead_time
    - p90_lead_time

```

---

# 65. What should the final research claim be?

Do not claim:

> “CyberWorld predicts cyberattacks with 99.9% accuracy.”

Do not claim:

> “CyberWorld guarantees 10 seconds of warning.”

Do not claim:

> “CyberWorld provides 95% conformal confidence.”

Do not claim:

> “CyberWorld understands the complete MITRE ATT&CK kill chain.”

The target claim should be closer to:

> **CyberWorld is a short-horizon cyber-state forecasting architecture that uses the previous 30 seconds of network telemetry to estimate current attack state and forecast future attack onset, network/host-state evolution, and likely ATT&CK behavior over the next 10 seconds. The system is evaluated using leakage-resistant temporal/scenario splits, held-out probability calibration, horizon-wise forecasting metrics, attack-episode lead-time measurements, and explicit baseline and ablation experiments.**

Every part of this statement must be demonstrably true in the repository.

---

# 66. What Claude must NOT do

Claude must **not**:

```text
- blindly preserve all legacy architecture
- add new models before fixing data semantics
- keep 60-second windows anywhere in the promoted 2-second pipeline
- use a history other than 15 states in the canonical path
- train K=4 and claim K=5
- treat severity scores as probabilities
- call fixed intervals conformal
- treat ATT&CK as single-label
- merge private IPs across datasets
- convert unknown labels into attack labels
- use oracle attack state in benchmark evaluation
- optimize thresholds against the final test set
- report accuracy as the main forecasting metric
- fabricate benchmark results
- preserve dead modules merely because they sound advanced
- change the temporal contract independently in different modules

```

---

# 67. What Claude must do

Claude should:

```text
1. Audit the entire repository against this specification.

2. Produce a dependency graph of the actual promoted
   training/inference path.

3. Produce a current-vs-required discrepancy matrix.

4. Audit every dataset builder for:
   - temporal leakage
   - target leakage
   - duplicate overlap
   - host identity leakage
   - label contamination

5. Enforce:
   window_seconds = 2
   history_steps = 15
   forecast_steps = 5

6. Ensure:
   history_seconds = 30
   forecast_seconds = 10

7. Build one canonical preprocessing path shared by:
   - offline training
   - offline evaluation
   - replay
   - live inference

8. Build properly timestamped future attack-onset labels.

9. Build multilabel ATT&CK targets.

10. Replace ambiguous risk semantics with explicit:
    - current probability
    - future hazard
    - cumulative future probability
    - optional severity score

11. Make Branch B genuinely predict K=5 future states.

12. Ensure DeepOP/CWA receives deployment-style predicted future states.

13. Build proper chronological/scenario-held-out splits.

14. Add train/validation/calibration/test separation.

15. Implement baseline models.

16. Implement ablation experiments.

17. Implement multi-seed evaluation.

18. Implement statistically valid confidence intervals.

19. Implement actual probability calibration.

20. Implement valid conformal prediction only if its
    assumptions/procedure are correctly defined.

21. Implement deterministic offline/live parity tests.

22. Generate immutable experiment manifests.

23. Refactor/remove legacy code only after confirming it
    is not part of the promoted path.

24. Make the repository reproducible from a clean environment.

25. Update README/documentation so every research claim
    exactly matches the implementation.

```

---

# 68. Required implementation workflow

Do not immediately rewrite the entire repository.

First produce:

```text
A. Current architecture map
B. Current-vs-required discrepancy matrix
C. Temporal-contract violation report
D. Data/label leakage audit
E. Train/serve mismatch report
F. Exact files requiring modification
G. Exact files safe to delete
H. Migration plan
I. Evaluation implementation plan
J. Reproducibility implementation plan

```

Then implement changes in dependency order.

For every major change document:

```text
OLD BEHAVIOR
NEW BEHAVIOR
WHY OLD BEHAVIOR WAS SCIENTIFICALLY INVALID
HOW NEW BEHAVIOR IS VALIDATED
TESTS ADDED

```

Do not optimize for preserving the existing code.

Optimize for producing a:

> **clean, reproducible, scientifically defensible, technically advanced cyber-state forecasting system.**

The final repository should allow an external researcher to:

```text
clone
→ install
→ prepare dataset manifest
→ reproduce preprocessing
→ train
→ calibrate
→ evaluate
→ reproduce metrics
→ reproduce figures
→ replay inference
→ run live mode

```

without manually determining which version of the architecture is authoritative.

---

# FINAL PRIORITY

The current architecture already contains enough complexity.

The next iteration should focus on:

```text
1. Correct temporal semantics
2. Correct future targets
3. Correct dataset isolation
4. Correct probability/hazard semantics
5. Correct world-model training
6. Correct calibration
7. Correct held-out evaluation
8. Correct baseline/ablation analysis
9. Exact offline/live parity
10. Reproducibility

```

The authoritative temporal contract is:

```text
2-second states
15 historical states
30-second history
5 future states
10-second forecast horizon

```

**Every module, checkpoint, dataset builder, training script, evaluation script, replay path, and live adapter must conform to this contract.**