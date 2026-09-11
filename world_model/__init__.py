# World Dynamics Transformer (WDT)
# Causal decoder-only Transformer for latent network-state dynamics prediction
# with K-step autoregressive rollout.
#
# Architecture: TGN → Readout → z_t → WDT → ẑ_{t+1:t+K} → {State, ATT&CK, Risk} decoders
