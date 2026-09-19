# Architectural & Engineering Design Decisions: World Dynamics Transformer (WDT)

This document formalizes the key architectural, mathematical, and systems design decisions made for the World Dynamics Transformer.

---

## 1. Causal Decoder-Only Transformer (vs Encoder-Decoder or SSM)
- **Decision**: Adopt a causal decoder-only architecture ($L=4, d_{\text{model}}=256, n_{\text{heads}}=4, d_{\text{ff}}=512$, ~2.42M parameters) with strict lower-triangular masking.
- **Rationale**:
  1. *Single-Pass Training*: With teacher forcing across sequences of length $H+K-1$, every position simultaneously computes next-state predictions in parallel.
  2. *Native Rollout*: At inference, autoregressive generation appends predictions iteratively.
  3. *Temporal Semantics*: In network traffic, past events causally precede future events. Bidirectional history encoding violates domain causality.
  4. *Parameter Efficiency*: One stack serves both history ingestion and trajectory rollout.

---

## 2. Hybrid Latent State Representation ($d_z = 128$)
- **Decision**: Decompose $z_t \in \mathbb{R}^{128}$ into:
  - $z_t^{\text{structural}} \in \mathbb{R}^{114}$: Learned graph topological representation pooled via attention from node embeddings or flow/PCAP telemetry.
  - $z_t^{\text{statistical}} \in \mathbb{R}^{14}$: Hand-crafted domain-grounded statistics $\Phi_t$ (active nodes/edges, byte/packet rates, protocol mix, port entropy, malicious edge fraction).
  - Normalized jointly via `LayerNorm(Concat(z_struct, z_stats))`.
- **Rationale**: Prevents uninterpretable latent drift. Hand-crafted statistical features provide an anchor that allows auxiliary reconstruction heads to ground the latent space.

---

## 3. Output Residual Connection (Delta Prediction)
- **Decision**: Formulate transition as $\hat{z}_{t+1} = z_t + \Delta z_t$, where the Transformer outputs the state delta $\Delta z_t = W_{\text{out}} \cdot h_t$.
- **Rationale**: Telemetry states change incrementally across 2-second windows. Predicting the delta acts as an identity-mapping inductive bias (analogous to ResNets and Neural ODEs), preventing explosive error drift during $K$-step rollouts.

---

## 4. Continuous Harmonic Time-Delta Encoding with Gap Clamping
- **Decision**: Combine learnable absolute positional embeddings $\text{PE}(i)$ with TGAT continuous harmonic cosine encodings $\text{TE}(\Delta \tau_i)$ initialized logarithmically across frequencies $10^0 \to 10^{-9}$, with $\Delta \tau$ clamped to $[0, 300\text{s}]$.
- **Rationale**: Telemetry intervals are irregular (e.g., 2.0s normal, 0.1s bursts, 60s idle periods). Time gaps between distinct attack scenarios can exceed 90,000s; clamping to 300s preserves sensitivity to local burstiness while preventing cross-scenario numerical instability.

---

## 5. PyTorch `scaled_dot_product_attention` & BFloat16 Training
- **Decision**: Utilize PyTorch's native `F.scaled_dot_product_attention` and `bfloat16` mixed precision on RTX 4050.
- **Rationale**: Traditional additive masking with `float("-inf")` in half-precision (FP16) causes NaN overflow during softmax and backward passes. Native scaled dot product attention utilizes FlashAttention / memory-efficient C++ kernels, which are mathematically stable and 3x faster.

---

## 6. KV-Cache Autoregressive Rollout ($O(1)$ Step Complexity)
- **Decision**: Cache past key ($K$) and value ($V$) projections across layers during inference rollout.
- **Rationale**: Without KV caching, stepping $k=1 \to K$ requires recomputing self-attention over the expanding context at $O(k^2)$ cost. With KV caching, only the single newly generated vector is projected, reducing per-step complexity to $O(1)$ and verified to achieve $4.77 \times 10^{-7}$ numerical equivalence with full-context computation.

---

## 7. Scheduled Sampling for Exposure Bias Mitigation
- **Decision**: Anneal sampling probability $p_{\text{sample}}$ linearly from 0.0 to 0.5 over training epochs, detaching model-generated tokens before mixing.
- **Rationale**: Pure teacher forcing shields the model from its own mistakes, causing compounding errors at test-time rollout. Scheduled sampling exposes the model to self-generated states while keeping gradients bounded.

---

## 8. Auxiliary Direct Multi-Horizon Prediction Head
- **Decision**: Include a secondary parallel MLP head predicting $\hat{z}_{t+1:t+K}$ directly from $h_{t}$ with auxiliary loss weight $\lambda_{\text{direct}} = 0.1$.
- **Rationale**: Serves as a multi-step anchor that does not suffer from sequential error accumulation, stabilizing the primary autoregressive pathway.

---

## 9. Strict Chronological Temporal Splitting (Zero Leakage)
- **Decision**: Strictly split sequences by timestamp quantiles: Train (first 60%), Validation (next 15%), Holdout Test (final 25%).
- **Rationale**: Random splitting of temporal telemetry creates massive data leakage (future windows predicting past windows). Strictly enforcing $\max(\tau_{\text{train}}) < \min(\tau_{\text{val}}) < \min(\tau_{\text{test}})$ ensures valid evaluation.

---

## 10. Decoupled Downstream Architectural Decoders
- **Decision**: Keep the WDT backbone generic and predictive of latent dynamics, while delegating interpretation to lightweight specialized heads:
  - `StateDecoder`: Reconstructs human-interpretable traffic metrics for analysts.
  - `AttackStageDecoder`: Maps rollout trajectories to MITRE ATT&CK tactics via temporal attention pooling.
  - `RiskHead`: Computes per-step hazard and cumulative multi-step infiltration probabilities.
- **Rationale**: Decoupling representation and dynamics from task-specific classification allows the world model to serve multiple security operations center (SOC) tasks simultaneously without catastrophic forgetting.
