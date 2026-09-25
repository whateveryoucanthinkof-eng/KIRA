# RL at Inference / Runtime — Test-Time Adaptation, Online Learning, Test-Time Memory

**Date:** 2026-09-19
**Scope:** INFERENCE / RUNTIME only. Training-time RL (reward shaping, offline RL, Dreamer-style latent-imagination training) is deliberately out of scope and covered separately.
**Repo:** `cyber_network_predictor`, branch `testing-prod`
**Inputs trusted as verified:** `claude_latest_analysis/01_training_data_pcap_audit.md`, `claude_latest_analysis/02_ps_26153_gap_analysis.md`

---

## (a) Verdict

**Three sentences:**

1. **MIRAS does not apply here.** It is an architecture-design framework for building sequence-model *layers* whose internal memory is updated by an online optimization step during the forward pass. Adopting it means throwing away all four checkpoints and retraining from scratch to solve a long-context problem this system does not have — its context is **5 timesteps × 12 dimensions = 60 numbers** (`model_adapter.py:91`, `DEFAULT_HISTORY_STEPS`; `rollout_encoder_decoder.py:194`, `max_context_len=10`). Titans/MIRAS earn their keep at 10M-token BABILong contexts. There is nothing to gain and a checkpoint to lose.

2. **Weight-updating online learning of any kind is the wrong move for this system on this timeline, and is genuinely dangerous in a security product** — but the *structural insight* behind it is correct and valuable: this system's rollout predicts `h_{t+K}`, and 16 seconds later the real `h_{t+K}` arrives. That is a free, label-free, continuously-available supervision signal that the repo currently throws away entirely.

3. **The right thing to build with that signal is not gradient descent — it is online conformal calibration.** The hardcoded radii `[0.01890, 0.03661, 0.05327, 0.06904]` at `branch_b_world_model/rollout_encoder_decoder.py:262` are a real weakness, they are **discarded at the call site anyway** (`model_adapter.py:379` binds them to `_radii` and never uses them), and replacing them with an Adaptive Conformal Inference loop is ~1.5 days of work, requires **no labels and no retraining**, carries a published distribution-free coverage guarantee, and directly discharges the PS clause the gap analysis marks PARTIAL ("probability distribution over the next network state"). **This is the only item in this entire research thread I recommend building before the SIH deadline**, and it should still queue *behind* gap-analysis items 1–3.

**Is there a genuine RL-shaped decision at runtime?** Structurally yes (which host to focus on, when to escalate, where to point sensors — `model_adapter.py:532` `_focus_binding`, `:555` `select_primary_target`, both currently heuristic). **But there is no reward signal**: no analyst disposition loop, no ticket outcomes, no feedback channel of any kind in the product. A contextual bandit with no reward is a random policy with extra code. Verdict: honest "future work" slide item, not a build item.

---

## (b) What MIRAS actually is

### The paper

**"It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization"** — Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research). arXiv:2504.13173, submitted **17 April 2025**, v1 only.
<https://arxiv.org/abs/2504.13173>

Abstract, verbatim (excerpt of the load-bearing sentences):

> "Inspired by the human cognitive phenomenon of attentional bias—the natural tendency to prioritize certain events or stimuli—we reconceptualize neural architectures, including Transformers, Titans, and modern linear recurrent neural networks as associative memory modules that learn a mapping of keys and values using an internal objective, referred to as attentional bias. Surprisingly, we observed that most existing sequence models leverage either (1) dot-product similarity, or (2) L2 regression objectives as their attentional bias. […] We then reinterpret forgetting mechanisms in modern deep learning architectures as a form of retention regularization, providing a novel set of forget gates for sequence models. Building upon these insights, we present Miras, a general framework to design deep learning architectures based on four choices of: (i) associative memory architecture, (ii) attentional bias objective, (iii) retention gate, and (iv) memory learning algorithm. We present three novel sequence models—Moneta, Yaad, and Memora—that go beyond the power of existing linear RNNs while maintaining a fast parallelizable training process."

Google's own write-up (**"Titans + MIRAS: Helping AI have long-term memory"**, Google Research blog, **4 December 2025**) describes MIRAS as *"the theoretical framework (the blueprint) for generalizing these approaches"* and characterises the three variants as: **Yaad** (Huber loss — robustness to outliers), **Moneta** (generalized norms), **Memora** (probabilistic/normalized mapping for stability).
<https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/>

### What "test-time memorization" actually means in this line of work — and what it does not

This is the point where the name misleads, so state it precisely:

- In **Titans** (Behrouz, Zhong, Mirrokni, arXiv:2501.00663, <https://arxiv.org/abs/2501.00663>), a neural long-term memory module updates **its own memory parameters during the forward pass** as tokens stream in, using a gradient-based *"surprise"* signal (the gradient of an associative-memory loss w.r.t. the input), with momentum and an adaptive forgetting gate.
- MIRAS generalises exactly that inner update: `attentional bias` is the objective the memory minimises, `retention gate` is the regulariser trading new information against retained information, and `memory learning algorithm` is the optimiser used for that inner step.
- **The outer network weights are frozen at test time.** The "learning at test time" is an *architectural* mechanism whose behaviour is trained in offline. It is an inner loop, not a deployment-time retraining loop.
- The same idea is formalised generically in **"Test-time regression: a unifying framework for designing sequence models with associative memory"** (arXiv:2501.12352, <https://arxiv.org/pdf/2501.12352>): these layers are all solving an online regression problem over (key, value) pairs.
- Follow-ups in the same group: **ATLAS** (Behrouz et al., arXiv:2505.23735, <https://arxiv.org/abs/2505.23735>) extends memory capacity and fixes the "optimise only w.r.t. the last token" limitation via the Omega rule; **Nested Learning / HOPE** (arXiv:2512.24695, Google Research blog 7 Nov 2025, <https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/>) adds a Continuum Memory System of banks updating at different frequencies.

**One property is genuinely attractive for this system:** the memory update is *self-supervised* — it reconstructs values from keys. **No labels at test time.** That is the right shape for a system with no runtime labels.

### Why it still does not fit

| MIRAS/Titans assumption | This system |
|---|---|
| The bottleneck is long context (BABILong up to 10M tokens; needle-in-haystack) | Context is `history_steps = 5` latents of dim 12. `rollout()` caps attention context at `max_context_len=10`. There is no long-context problem to solve. |
| Trained from scratch, 170M–1.3B params, language-scale corpora | `HostWorldDynamicsTransformer` is a small model trained on ~9 GB of flow records; adopting a MIRAS layer means discarding `saved_models/branch_b/host_wdt.pt` and retraining Branch B, Branch A's downstream consumers, DeepOP and the risk head. |
| Gains measured as perplexity / recall accuracy vs strong Transformer baselines | The repo has **zero detection metrics** (gap analysis §B3 — the only quantitative artifact is a 1-epoch BiTA link-prediction row with inductive AUC 0.485, below chance). **You cannot demonstrate that a MIRAS layer helped, because you have no instrument that would show it.** |
| Evaluation domains: language modeling, commonsense reasoning, recall-intensive tasks (arXiv abstract). Google's blog additionally names genomics and time-series forecasting for the Titans+MIRAS line | The blog's time-series claim is not broken out per-benchmark in the abstract and I did not verify which of the two papers it attaches to. **Do not cite "MIRAS works on time series" in the submission** — it is not supported at the granularity you would need. |

**Assessment:** transferring MIRAS here is a negative-expected-value move on an SIH timeline. It is architecture surgery with no measurable payoff, aimed at a bottleneck that does not exist in a 60-number context window.

**What MIRAS *is* worth here:** exactly one sentence of intellectual lineage in the architecture document, and a piece of useful vocabulary for §(e) below — MIRAS's **retention gate** is a clean formalisation of the stability/plasticity trade-off that is the central risk of *any* runtime adaptation. Cite it as related work. **Do not claim to implement it.** An SIH judge can check `git grep -i miras` in about four seconds, and a claimed-but-absent component is exactly the kind of finding that costs more than the claim gained.

### "RIC/RIS from Google" — clarification

**No Google Research ML framework named RIC or RIS exists in this line of work.** I searched specifically and found nothing. The two real "RIC"/"RIS" acronyms are from telecom and are unrelated: **RIC = RAN Intelligent Controller** (O-RAN Near-RT/Non-RT controller) and **RIS = Reconfigurable Intelligent Surfaces** (6G wireless). The RIC one is a plausible source of confusion given the networking domain.

Most likely intended, in descending order of probability:

1. **MIRAS itself** — same thing, half-remembered as a second item.
2. **Titans** (arXiv:2501.00663) — the architecture MIRAS generalises; always discussed alongside it.
3. **Mixture-of-Recursions (MoR)** — Google DeepMind + KAIST + Mila, arXiv:2507.10524, NeurIPS 2025, <https://arxiv.org/abs/2507.10524>. Adaptive per-token recursion depth with lightweight routers; 135M–1.7B scale. An *inference-efficiency* technique, not a memory/adaptation one. **Also does not apply here** — this model is far too small for adaptive-depth routing to pay for itself.
4. **ATLAS** (arXiv:2505.23735) or **Nested Learning / HOPE** (arXiv:2512.24695).

Recommend asking the user which they meant before building anything on it.

---

## (c) Label-free runtime adaptation — what actually applies, ranked

### The distinction that decides everything: **label ≠ outcome**

`TECHNICAL_REVIEW.md` is right that live labels are unavailable. But the repo conflates two different things:

- **Labels** (is this host under attack? which ATT&CK technique?) — genuinely unavailable at runtime, forever.
- **Outcomes** (what was the actual host latent `h` at t+K?) — **available, for free, 16 seconds after every single prediction.** `h_state_history_by_target` (`model_adapter.py:86`) already holds the realized latents. The rollout's target is *literally in the dict already*.

This splits the three model heads cleanly:

| Component | Runtime supervision available? | Why |
|---|---|---|
| **Branch B** `HostWorldDynamicsTransformer` — latent dynamics | **YES** (delayed 2K seconds) | It predicts `h_{t+k}`; the real `h_{t+k}` is observed and stored. |
| **Branch A** `MultiTaskLSTM` — risk score + technique | **NO** | Its training target is a lookup keyed on the ground-truth tactic (`multi_dataset_stream.py:290-311`, `TACTIC_BASE_SEVERITY`). There is no runtime analogue of that key. |
| **DeepOP** CWA decoder — future ATT&CK tokens | **NO, and worse** | Its "observed token" is derived from Branch A's output — and currently from the **heuristic override** at `model_adapter.py:350-362`, not the classifier (gap analysis §B2). Self-training on it is a closed loop that amplifies a hand-written rule. |

**Therefore: runtime adaptation is legitimate only for the latent-dynamics path and for calibration. It is illegitimate for the risk score and for the attack-stage classifier.** That sentence is the most important technical conclusion in this document.

### The literature on "predict, then observe"

This structure is well-studied, and the relevant papers are explicit that the revealed value is an *outcome*, not an annotation:

- **TAFAS — "Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation"** (Kim, Kim, Mok, Yoon; AAAI 2025; arXiv:2501.04970, <https://arxiv.org/abs/2501.04970>). Explicitly the pioneering TSF-TTA framework; its adaptation signal is *"partially-observed ground truth"* — the part of the forecast horizon that has already elapsed — fed through a gated calibration module, with the source forecaster's core **frozen**. This is precisely the structure this repo has and does not use.
- **FSNet — "Learning Fast and Slow for Online Time Series Forecasting"** (Pham, Tran, Phung, Venkatesh; ICLR 2023; arXiv:2202.11672, <https://arxiv.org/abs/2202.11672>). Online gradient descent on realized forecast error, with per-layer adapters (fast) over a slow backbone plus an associative memory for recurring regimes. Notably, FSNet *"does not explicitly detect distribution shifts"* — it continuously improves on current samples. Relevant as the canonical "online gradient on forecast error" reference.
- **Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting** (arXiv:2412.08435) — the successor line, addressing the delay between prediction and feedback.

### Which TTA methods need labels — the taxonomy that matters here

| Method | Needs labels at test time? | Needs source data? | Applicable here? |
|---|---|---|---|
| **TTT** (Sun, Wang, Liu, Miller, Efros, Hardt; ICML 2020; arXiv:1909.13231) — self-supervised auxiliary task per test sample | **No** | No | Conceptually yes; requires designing and *pre-training with* an auxiliary head. Architecture change + full retrain. |
| **TTT layers** (Sun et al., arXiv:2407.04620) — hidden state *is* a model, update rule *is* a self-supervised step | **No** | No | Same category as Titans/MIRAS: architecture-level, full retrain. No. |
| **TENT** (Wang, Shelhamer, Liu, Olshausen, Darrell; ICLR 2021; arXiv:2006.10726) — entropy minimization over test batches, updating BatchNorm affine params | **No** | No | **Mechanically impossible as-is.** Verified: `branch_a_gnn_lstm/lstm_multitask.py` contains **no `BatchNorm` and no `LayerNorm`** — only `nn.LSTM` and `nn.Dropout` (lines 120, 135, 144, 152). TENT's entire mechanism (re-estimating normalization statistics + optimizing channel-wise affine transforms) has nothing to attach to. And see §(e) for why you should not add them. |
| **MADCAT** (Roh, Kaya, Kruegel, Vigna, Hong; arXiv:2505.18734, <https://arxiv.org/abs/2505.18734>) — TTA for malware detection under concept drift; encoder-decoder, self-supervised test-time training of the encoder on a balanced subset of test data | **No** | No | The closest security-domain precedent. Demonstrates the idea is not crazy in security — but note it trains an *encoder* on a *curated balanced subset*, which presumes a batch-oriented, not stream-oriented, deployment. |
| **ACI / online conformal** (Gibbs & Candès, NeurIPS 2021) | **No — needs only the realized outcome** | No | **Yes. This is the one.** See §(d). |

### Ranked opportunities — value ÷ effort

**#1 — Compute and emit `forecast_error`. (~0.5 day. Value: high. Effort: trivial.)**

`control_backend/schema.py:66` already declares `forecast_error: Optional[float]`. It is **never set anywhere in the repo**. `web_dashboard/src/api/adapter.ts:245` already renders it as a bar labelled "Forecast Error" — which is therefore **permanently zero on screen** (gap analysis §B1).

The fix: in `predict_window` (`model_adapter.py:262`), keep a small per-host ring buffer of `(window_id, h_future)` rollouts. When window `t+k` arrives and `curr_h` is computed (`:363`), look up the prediction made at `t` for step `k` and emit `||ĥ − h||₂`. Everything needed is already in scope: `h_state_history_by_target` holds the realized latents, `h_future` holds the predictions.

This is the highest value-per-line item in this entire document. It fixes a visibly broken UI element, it gives the demo a "the model checks its own work" moment, and it is the substrate every other item below depends on.

**#2 — Online conformal calibration of the rollout radii. (~1.5 days. Value: high.)** Full treatment in §(d).

**#3 — Conformal drift / abstention gate. (~0.5 day on top of #2. Value: high for credibility.)**

Once you are tracking realized miscoverage, a sustained excursion above the target rate is a *model-is-now-wrong* detector that needs no labels. Surface it as a badge: `CALIBRATION DRIFT — predictions outside guaranteed coverage`.

This has direct, citable precedent in security ML: **Transcend** (Jordaney et al., USENIX Security 2017, <https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/jordaney>) uses conformal evaluation to detect aging classifiers *in vivo* before performance degrades, via per-class p-values from non-conformity measures rather than classifier confidence; **TRANSCENDENT** (arXiv:2010.03856, <https://arxiv.org/pdf/2010.03856v4>) extends it into a rejection framework — quarantine samples likely to be misclassified. "The system knows when it does not know" is a strong story for an NTRO evaluator, and it is the honest alternative to a confident-but-wrong number.

**#4 — Alert-rate controller for the hardcoded threshold. (~0.5 day. Value: medium. Ships off-by-default.)**

`model_adapter.py:88`: `self.alert_threshold = 0.65`. Hardcoded, used at `:258`, `:435`, `:477`, `:495`.

A label-free adaptation is available — but **only for the workload target, not the detection target**. You cannot tune for detection rate without labels. You *can* tune for **alert budget**: hold the alert rate at ≤ N per hour. That is the identical single-parameter online update as ACI, with the alert indicator in place of the miscoverage indicator:

```
θ_{t+1} = θ_t + γ · (1{alert_t} − target_rate)
```

Ship it clearly labelled as an **alert-volume controller**, explicitly *not* a detection-quality improvement. Mislabelling it is the single easiest way to turn a defensible engineering feature into a fabricated claim.

**#5 — Online gradient descent on Branch B only, FSNet/TAFAS-style. (~3–5 days. Value: medium. Risk: high.) NOT for this timeline.**

Genuinely applicable and genuinely in the literature. If ever built, the non-negotiable constraints are: update **only** `out_head` (the residual-delta head at `rollout_encoder_decoder.py:182`) or a small inserted adapter, never the transformer backbone; freeze all adaptation whenever an alert is active; rate-limit the step size; checkpoint adaptation state per window for forensic replay; and keep a frozen shadow model running in parallel so you can always answer "what would the accredited model have said?". Every one of those is a requirement, not a nicety — see §(e).

**#6 — Contextual bandit for host focus / sensor allocation. (Not now.)**

There *is* a real decision surface: `_focus_binding` (`model_adapter.py:532`) and `select_primary_target` (`:555`) are pure heuristics that pick which host gets scored and highlighted, and `predict_window` scores one primary target. In a real SOC this is a textbook contextual bandit — the survey literature confirms multi-armed and contextual bandits are an established tool for SOC triage and prioritization (**"AI-Driven Security Alert Screening and Alert Fatigue Mitigation in Security Operations Centers: A Survey"**, arXiv:2605.08316, <https://arxiv.org/html/2605.08316>, which places risk scoring, RL, RLHF, learning-to-defer and multi-armed bandits in the triage/prioritization stage).

**But this product has no reward channel.** No analyst disposition, no ticket outcome, no true/false-positive feedback, no API for one. A bandit without reward is a random policy plus a dependency. Correct treatment: one bullet on the "future work" slide, framed as *"the architecture exposes a triage decision point that a contextual bandit would occupy once analyst-disposition feedback exists."* That is an honest, defensible, and actually rather good thing to say to an NTRO panel — it shows you understand the deployment loop.

**#7 — MIRAS / Titans memory layer in Branch B. (Not now. See §(b).)**

**#8 — TENT / entropy minimization on Branch A. (Never. See §(e).)**

---

## (d) Online conformal prediction — the concrete near-term win

### The current state is worse than "hardcoded"

`branch_b_world_model/rollout_encoder_decoder.py:238-269`:

```python
h_future = self.rollout(h_seq, K=K, ...)
if empirical_radii is None:
    base_radii = [0.01890, 0.03661, 0.05327, 0.06904]
    empirical_radii = list(base_radii[:K])
    for step in range(len(empirical_radii) + 1, K + 1):
        # Sub-exponential empirical error growth across extended horizons K in [5..12]
        r_step = base_radii[-1] + 0.015 * math.sqrt(float(step - 3))
        empirical_radii.append(round(float(r_step), 5))
```

Five separate problems, all verified:

1. **The radii are never used.** `model_adapter.py:375-381` calls `rollout_with_uncertainty` and binds the result to `_radii` — an underscore-prefixed throwaway. The in-code comment admits it: *"They are computed but not yet carried in the prediction payload."* The live system emits a **point forecast with no uncertainty at all.**
2. **Only 4 of 8 steps are real.** `DEFAULT_ROLLOUT_HORIZON_LIVE = 8`, but `base_radii` has 4 entries. Steps 5–8 come from an invented closed-form `0.015·√(step−3)` with no empirical basis. That formula is a guess wearing a comment that says "empirical".
3. **A scalar radius on a 12-D latent.** One number for a 12-dimensional ball. The latent's dimensions are not isotropic — 3 of the 12 canonical edge dims are constant-zero in TGNE pretraining (audit §2, `bita/train.py:163-166`), so the geometry is demonstrably degenerate.
4. **Calibrated on a different distribution.** Derived from offline validation rollouts over CIC-2018/CTU-13 flow records; applied to live SPAN capture in a containerlab topology. There is no reason the residual distribution should match, and no mechanism that would notice if it did not.
5. **No coverage claim.** Nobody knows whether 95% of live residuals fall inside 0.01890 at k=1. Nobody can know, because nothing measures it.

### The replacement

**Adaptive Conformal Inference** — Isaac Gibbs and Emmanuel J. Candès, *"Adaptive Conformal Inference Under Distribution Shift"*, NeurIPS 2021, arXiv:2106.00170, <https://arxiv.org/abs/2106.00170>.

The update rule is one line:

```
α_{t+1} = α_t + γ · ( α − 1{ Y_t ∉ C_{α_t}(X_t) } )
```

Properties that matter here:

- **It is a wrapper.** It "can be combined with any black box method that produces point predictions" — no retraining, no architecture change, no touching `host_wdt.pt`.
- **It does not assume exchangeability.** The classical conformal assumption is dropped; the method "provably achieves the desired long-term coverage frequency irrespective of the true data generating process" by treating the shift as a single-parameter online learning problem. The paper bounds the deviation of realized miscoverage from α by a term of order 1/(γT).
- **It needs only a binary coverage indicator per step** — i.e. did the realized value land inside the region. Here the realized value is `h_{t+k}`, already in `h_state_history_by_target`. **No labels. Ever.**

### Concrete design for this repo

New module `control_backend/conformal.py` (~180 lines), plus ~25 lines of wiring in `model_adapter.py`.

1. **Nonconformity score, per horizon k:** `s_k = ||ĥ_{t+k|t} − h_{t+k}||₂`. Upgrade path: Mahalanobis distance under a running covariance, giving ellipsoidal rather than spherical regions — see **MultiDimSPCI**, Xu, Jiang & Xie, *"Conformal prediction for multi-dimensional time series by ellipsoidal sets"*, ICML 2024, arXiv:2403.03850, <https://arxiv.org/abs/2403.03850>, which is distribution-free, model-agnostic, sequential, and produces smaller regions than spherical baselines. This is the correct treatment for a 12-D latent with degenerate dimensions.
2. **Per-horizon sliding window** of the last W scores. W ≈ 500 windows ≈ 17 minutes of wall clock at 2.0 s.
3. **Radius:** `r_k(t) = Quantile_{1−α_k(t)}(window_k)`.
4. **ACI update** of `α_k(t)` as above, one independent tracker per k ∈ {1..8}. If you do not want to hand-pick γ, use **DtACI** — Gibbs & Candès, *"Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"*, arXiv:2208.08401, <https://arxiv.org/abs/2208.08401> — which tunes the step size over time and, unlike ACI, is "adaptive to both the size and type of the distribution shift" without requiring prior knowledge of the drift rate.
5. **Offline warm start:** seed each window from the existing validation rollouts so the system is not uncalibrated for its first 17 minutes. The current `base_radii` are the right seed values; they become an initialization instead of a hardcoded answer.
6. **Safety floor (see §(e)):** never allow the online radius to exceed `floor_multiplier × offline_radius`. Online calibration may make the system *more* sensitive; it must never be able to make it less sensitive than the accredited baseline. This is the specific control that blocks the boiling-frog attack.

### Turning a latent region into a risk band

The dashboard shows *risk*, not latents. Two options:

- **(i) Monte Carlo through the risk head — recommended.** Sample ~32 perturbations of `ĥ_{t+k}` on the conformal ball and push them through `infiltration_head.forward_trajectory` (`branch_b_world_model/infiltration_head.py:46`). Take the empirical min/max or 5th/95th percentile. Cost: 32 × 8 × 12 floats through a small MLP — negligible against the existing per-window budget. Honest description: *"the range of risk values consistent with the conformal region on the predicted network state."*
- **(ii) Conformalize the risk residual directly.** Compare `risk_head(ĥ_{t+k|t})` against `risk_head(h_{t+k})` once observed. Also label-free and arguably cleaner, **but say exactly what it measures**: self-consistency of the forecast against the model's own later assessment, *not* accuracy against truth. Both quantities come from the same model. Do not let this get described as accuracy.

Do **not** attempt to conformalize risk against the *displayed* `obs_risk` — that value is the heuristic override (`model_adapter.py:350-362`), so you would be calibrating the world model against a hand-written rule. Delete the override first (gap-analysis fix #3, ~1 hour) or this item is built on sand.

### Schema and UI

`ForecastPoint` (`control_backend/schema.py:37-41`) has a `confidence` field — **but it is already occupied** by the DeepOP token confidence (`model_adapter.py:446`). Add distinct fields rather than overloading it:

```python
risk_lo: Optional[float] = None
risk_hi: Optional[float] = None
latent_radius: Optional[float] = None
coverage_target: Optional[float] = None
```

UI: a shaded band on the forecast chart in `web_dashboard/src/pages/Predictions.tsx`, plus a small live coverage tracker ("realized coverage: 91.3% / target 90%"). Combine with the already-required fix to `:188` (`+{i+1}m` → `f.horizon_seconds`, gap analysis §C.3 — the UI currently mislabels a 16-second horizon by 30×).

### How this compares to the gap analysis's own suggestion

Gap analysis §C.2 proposes adding a log-variance head and Gaussian NLL training (~15 lines, ~1 day, ranked #17/low leverage). Honest comparison:

| | Gaussian NLL variance head | Online conformal |
|---|---|---|
| Retraining required | **Yes** — Branch B retrain | **No** |
| Uncertainty varies with input | **Yes** (heteroscedastic) | No (marginal, by default) |
| Calibration guarantee | None unless separately verified | **Yes** — distribution-free, long-run, no exchangeability assumption |
| Adapts at runtime | No | **Yes** |
| Answers PS "probability distribution over next state" | Parametrically | **Yes** — a distribution-free prediction *region* for `S_{t+1..t+K}` is a literal, defensible reading of the clause |
| Risk if it goes wrong | Silent miscalibration | Visible — the coverage tracker shows it |

**They compose.** Best answer if time allows: variance head first, then **Conformalized Quantile Regression** (Romano, Patterson & Candès, NeurIPS 2019, arXiv:1905.03222, <https://arxiv.org/abs/1905.03222>) on top, which is "fully adaptive to heteroscedasticity" and yields shorter intervals — giving input-dependent width *and* a coverage guarantee. On the SIH timeline: **do conformal alone.** It is cheaper, needs no retrain, and produces the stronger claim.

### Prior art in exactly this domain

Conformal prediction is already established in security ML, which makes this defensible rather than exotic: Transcend / TRANSCENDENT (above); conformal prediction applied to intrusion detection and botnet detection; and recent work on conformal anomaly detection with false-discovery-rate guarantees evaluated on 5G network intrusion detection (**"Online Conformal Anomaly Detection with Prediction-Powered Data Acquisition"**, arXiv:2505.01783, <https://arxiv.org/html/2505.01783>). You are not inventing a technique; you are applying a standard one to a place where a hardcoded constant currently sits.

---

## (e) Risks — and why some of this is genuinely dangerous

### E1. The boiling-frog attack — the one that matters most

**This is the central objection to runtime adaptation in this system, and it is specific to it.**

The supervision signal is "the observed `h_{t+K}`". `h` is computed from traffic. **An attacker who can shape traffic can shape the supervision signal.** A patient adversary generates low-rate traffic that is slightly unusual but individually harmless, driving the online residual distribution steadily upward. The conformal radius widens to maintain its coverage target — *correctly, by the algorithm's own definition*. When the real attack runs, its residual falls comfortably inside the now-wide "normal" band.

The algorithm has not failed. It did exactly what it guarantees: long-run marginal coverage. **The guarantee is coverage, not detection.** Nothing in ACI's theorem says anything about catching attacks; it says the realized miscoverage frequency converges to α. Under an adversary who controls the input stream, those are different objectives and the adversary chooses which one you get.

A **frozen** model cannot be trained by the adversary at all. Adaptivity converts the detector from a fixed target into a trainable one. That is a qualitative increase in attack surface, not a quantitative one.

The literature confirms the mechanism generalises: **"Adversarial concept drift detection under poisoning attacks for robust data stream mining"** (Korycki & Krawczyk, *Machine Learning*, 2023; arXiv:2009.09497, <https://arxiv.org/pdf/2009.09497>) shows existing drift detectors "are not capable of differentiating between real and adversarial concept drift" and that standard drift statistics "offer no robustness to poisoning attacks." And specifically for the technique recommended in §(d): **"Online Conformal Prediction with Corrupted Feedback"** (arXiv:2605.20515, <https://arxiv.org/html/2605.20515>) shows OCP's guarantees "hinge on the assumption of perfect feedback about the coverage of past prediction sets," and that corruption — including adversarial manipulation — "can severely degrade OCP's calibration guarantees." Related work on conformal calibration under adversarial attack finds that coverage "varies monotonically with the calibration-time attack strength, enabling the use of nonzero calibration-time attack to predictably control coverage."

**Mitigations, all mandatory if you ship §(d):**
- **Radius floor** — cap the online radius at a small multiple of the offline-calibrated value. Adaptation may tighten, never loosen past the accredited baseline.
- **Rate limit** — cap per-window change in `α_k` and in `r_k`.
- **Freeze during incident** — no calibration updates while any alert is active. This also kills the "model changed its mind mid-incident" problem (E4).
- **Log every adaptation step** with window id and timestamp, so post-incident forensics can reconstruct the calibration trajectory and spot a slow poisoning ramp.
- **Alarm on the adaptation itself** — a sustained monotone widening of the radius is *itself* a detection signal and should raise its own event, not just silently widen the band.

### E2. Catastrophic forgetting

Standard for any online weight update: gradient steps on the current distribution overwrite representations for distributions not currently present (McCloskey & Cohen 1989; French 1999; Kirkpatrick et al., EWC, PNAS 2017). In a network context the "absent distribution" is *every attack class not currently running* — which, by construction, is all of them most of the time. A model that quietly adapts to two hours of benign office traffic has been optimising away from exactly the states it exists to recognise.

This is precisely the stability/plasticity trade-off MIRAS formalises as its **retention gate** — worth citing in the architecture doc as the reason you *didn't* do online weight updates. It is a stronger use of the reference than pretending to implement it.

**Note the asymmetry:** online conformal calibration updates a **scalar per horizon**. It has no representation to forget. This is the main reason it is safe and online gradient descent is not.

### E3. Co-adaptation feedback loop

Model adapts to attacker; attacker observes changed behaviour and adapts. Two learning systems in closed loop with opposing objectives and no equilibrium guarantee. In a competition demo this is invisible; in a Critical Information Infrastructure deployment — which the PS explicitly names — it is a stability question nobody has answered.

### E4. Operational non-reproducibility — the deployment blocker

Incident response requires answering **"what did the system know at 03:14:22, and why?"** If weights or calibration state changed between then and the investigation, you cannot replay it. A self-modifying detector is:

- **un-auditable** — the explanation you generate today is not the explanation the system would have generated then;
- **a change-control problem** — for CII deployment the model is a controlled artifact; silent runtime mutation is a compliance failure before it is a technical one;
- **untestable against a fixed benchmark** — every evaluation becomes path-dependent on the exact stream that preceded it.

**Minimum bar if any runtime state mutates:** version and checkpoint it per window, make it replayable, and run a frozen shadow model in parallel so the question "what would the accredited model have said?" always has an answer.

For an NTRO evaluator this is not a footnote. It is likely the first question asked about any adaptive component.

### E5. Why TENT-style entropy minimization is specifically wrong here

Beyond being mechanically impossible (no norm layers — verified, §(c)): entropy minimization optimises the model for **confidence**, not correctness. Under class imbalance it collapses toward the majority class — and gap analysis §A/§C.3 documents exactly that pathology already (99.94% technique accuracy at epoch 5 alongside 0.32 risk MAE is the signature of imbalance, not skill). Entropy minimization on a security classifier would drive it toward confidently predicting "Benign", and an attacker who can inject traffic can steer that. It is the wrong objective in the wrong place with the wrong threat model.

### E6. The meta-risk: adaptivity on an unmeasured system

The repo has **zero detection metrics** (gap analysis §B3: no F1, no precision, no recall, no FPR; 27/27 tests pass but assert only contracts and infrastructure). **"Dos and Don'ts of Machine Learning in Computer Security"** (Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro, Rieck; USENIX Security 2022; <https://www.usenix.org/conference/usenixsecurity22/presentation/arp>) catalogues the pitfalls that make security ML look better in papers than in deployment — inappropriate baselines, lab-only evaluation, sampling bias, spurious correlations.

Adding runtime adaptation to a system with no measurement makes every one of those worse, because the system's behaviour now depends on the deployment stream and is therefore *less* reproducible, not more. **Never add adaptivity to a system you cannot measure.** Build the benchmark first (gap analysis #2, ~1 day). Everything in §(d) is more valuable, more defensible and more demoable once there is a number to move.

---

## (f) SIH-timeline recommendation

### Priority relative to the gap analysis

**Nothing in this document belongs in the critical path.** The gap analysis's top three — explainability/ATT&CK not rendered in the UI, the heuristic risk override on the live path, and the total absence of F1/precision/recall/FPR — are each worth more than everything here combined. Those are the findings a judge stumbles into. Do them first.

There is also a **hard dependency**: item #2 below is partly built on sand until the heuristic override at `model_adapter.py:350-362` is deleted, because you would otherwise be calibrating the world model against a hand-written rule. Gap-analysis fix #3 is ~1 hour and is a prerequisite.

### Build list (in order, after gap-analysis items 1–3)

| # | Item | Effort | Files |
|---|---|---|---|
| 1 | **Compute and emit `forecast_error`** — ring buffer of past rollouts; on each new window, `‖ĥ − h‖₂` vs the k-step-ago prediction | **0.5 d** | `control_backend/model_adapter.py:262-400` (`predict_window`); field already exists at `schema.py:66`; UI bar already exists at `web_dashboard/src/api/adapter.ts:245` |
| 2 | **Online conformal radii** replacing the hardcoded list — per-horizon sliding windows + ACI update + offline warm start + **radius floor** | **1.5 d** | new `control_backend/conformal.py`; `rollout_encoder_decoder.py:238-269`; `model_adapter.py:375-381` (stop discarding `_radii`); new fields on `schema.py:37` `ForecastPoint` |
| 3 | **Risk band on the forecast chart** — MC through `infiltration_head.forward_trajectory:46`; shaded band; live coverage tracker | **0.5 d** | `web_dashboard/src/pages/Predictions.tsx` (fix the `+{i+1}m` label at `:188` in the same pass) |
| 4 | **Calibration-drift badge** — sustained miscoverage above target → `CALIBRATION DRIFT` indicator | **0.5 d** | `conformal.py` + dashboard |
| 5 | *(optional)* **Alert-budget controller** for `alert_threshold = 0.65` — same single-parameter update, honestly labelled as volume control, **off by default** | **0.5 d** | `model_adapter.py:88, 258, 435` |

**Total: 3 days, or 3.5 with the optional item.** Everything is additive, nothing touches a checkpoint, nothing requires retraining, and every piece degrades gracefully to current behaviour if disabled.

### Do not build

- Titans / MIRAS / ATLAS / MoR memory layers — no long-context problem, full retrain, no instrument to measure the result.
- TENT or any entropy minimization on Branch A — mechanically impossible and conceptually wrong.
- Online gradient updates to any model weights.
- A contextual bandit — no reward signal exists.
- Anything that adapts the risk score or the ATT&CK stage classifier. There is no runtime supervision for either.

### Demo — how it looks in the 2-minute video (~35 seconds)

1. **Baseline.** Forecast chart with a visible confidence band and a live counter: *"realized coverage 90.4% / target 90%."* The band is narrow and stable.
2. **Attack injection.** (`workloads/attacker_scenario.py`.) The rollout residual spikes. The `forecast_error` readout climbs. The band widens in real time.
3. **The moment.** The `CALIBRATION DRIFT` badge lights up **before** the risk score crosses threshold — the world model detects that it has stopped being able to predict the network *before* the classifier says "attack". Say out loud: *"the model noticed it could no longer predict the future of this host, four seconds before it could name what was happening."* That is the world-model thesis of PS 26153 demonstrated in one visual, and it is an argument no logistic-regression baseline can make.
4. **The claim.** *"These bands are not hand-tuned. They are recalibrated every two seconds against the model's own realized forecast error, with a distribution-free long-run coverage guarantee — Gibbs and Candès, NeurIPS 2021 — and they use no labels, because at runtime there are none."*

That last sentence is the highest-value artifact this research thread produces.

### What to write in the architecture document

One sentence of related work, and no more:

> Runtime uncertainty is calibrated online by Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) against the model's own realized forecast error; model weights are frozen at inference. Architectural test-time-memorization approaches (Titans, arXiv:2501.00663; MIRAS, arXiv:2504.13173) were evaluated and rejected: they address long-context retention, whereas this model's context is five 12-dimensional host states, and their adoption would require full retraining with no available benchmark to measure the effect.

That paragraph is worth more than a claimed MIRAS integration, because it is true, it is checkable, and it demonstrates that the architecture choices were made rather than defaulted into.

---

## Sources

**MIRAS / Titans line (Google Research)**
- Behrouz, Razaviyayn, Zhong, Mirrokni — *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization* (MIRAS), arXiv:2504.13173, 17 Apr 2025 — <https://arxiv.org/abs/2504.13173>
- Google Research blog — *Titans + MIRAS: Helping AI have long-term memory*, 4 Dec 2025 — <https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/>
- Behrouz, Zhong, Mirrokni — *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663 — <https://arxiv.org/abs/2501.00663>
- Behrouz et al. — *ATLAS: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735 — <https://arxiv.org/abs/2505.23735>
- *Nested Learning: The Illusion of Deep Learning Architectures* (HOPE, Continuum Memory System), arXiv:2512.24695; Google Research blog 7 Nov 2025 — <https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/>
- *Test-time regression: a unifying framework for designing sequence models with associative memory*, arXiv:2501.12352 — <https://arxiv.org/pdf/2501.12352>
- Mixture-of-Recursions (Google DeepMind / KAIST / Mila), arXiv:2507.10524, NeurIPS 2025 — <https://arxiv.org/abs/2507.10524>

**Test-time training / adaptation**
- Sun, Wang, Liu, Miller, Efros, Hardt — *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*, ICML 2020, arXiv:1909.13231 — <https://arxiv.org/abs/1909.13231>
- Sun et al. — *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (TTT layers), arXiv:2407.04620 — <https://arxiv.org/abs/2407.04620>
- Wang, Shelhamer, Liu, Olshausen, Darrell — *Tent: Fully Test-time Adaptation by Entropy Minimization*, ICLR 2021, arXiv:2006.10726 — <https://arxiv.org/abs/2006.10726>
- Roh, Kaya, Kruegel, Vigna, Hong — *MADCAT: Combating Malware Detection Under Concept Drift with Test-Time Adaptation*, arXiv:2505.18734 — <https://arxiv.org/abs/2505.18734>

**Online forecasting / predict-then-observe**
- Kim, Kim, Mok, Yoon — *Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation* (TAFAS), AAAI 2025, arXiv:2501.04970 — <https://arxiv.org/abs/2501.04970>
- Pham, Tran, Phung, Venkatesh — *Learning Fast and Slow for Online Time Series Forecasting* (FSNet), ICLR 2023, arXiv:2202.11672 — <https://arxiv.org/abs/2202.11672>
- *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting*, arXiv:2412.08435 — <https://arxiv.org/pdf/2412.08435>

**Conformal prediction**
- Gibbs, Candès — *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021, arXiv:2106.00170 — <https://arxiv.org/abs/2106.00170>
- Gibbs, Candès — *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts* (DtACI), arXiv:2208.08401 — <https://arxiv.org/abs/2208.08401>
- Angelopoulos, Candès, Tibshirani — *Conformal PID Control for Time Series Prediction*, NeurIPS 2023 — <https://neurips.cc/virtual/2023/poster/69896>
- Xu, Jiang, Xie — *Conformal prediction for multi-dimensional time series by ellipsoidal sets* (MultiDimSPCI), ICML 2024, arXiv:2403.03850 — <https://arxiv.org/abs/2403.03850>
- Romano, Patterson, Candès — *Conformalized Quantile Regression*, NeurIPS 2019, arXiv:1905.03222 — <https://arxiv.org/abs/1905.03222>
- *Optimization-based Online Conformal Prediction for Multi-step Forecasting*, arXiv:2508.13362 — <https://arxiv.org/pdf/2508.13362>

**Security ML: drift, calibration, adversarial adaptation, pitfalls**
- Jordaney et al. — *Transcend: Detecting Concept Drift in Malware Classification Models*, USENIX Security 2017 — <https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/jordaney>
- *Transcending TRANSCEND: Revisiting Malware Classification with Conformal Evaluation*, arXiv:2010.03856 — <https://arxiv.org/pdf/2010.03856v4>
- Korycki, Krawczyk — *Adversarial concept drift detection under poisoning attacks for robust data stream mining*, Machine Learning (2023), arXiv:2009.09497 — <https://arxiv.org/pdf/2009.09497>
- *Online Conformal Prediction with Corrupted Feedback*, arXiv:2605.20515 — <https://arxiv.org/html/2605.20515>
- *Online Conformal Anomaly Detection with Prediction-Powered Data Acquisition*, arXiv:2505.01783 — <https://arxiv.org/html/2505.01783>
- *AI-Driven Security Alert Screening and Alert Fatigue Mitigation in Security Operations Centers: A Survey*, arXiv:2605.08316 — <https://arxiv.org/html/2605.08316>
- Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro, Rieck — *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022 — <https://www.usenix.org/conference/usenixsecurity22/presentation/arp>
