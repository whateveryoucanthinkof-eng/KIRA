# ML review: do this in the next session

This file lists what is already known to need an ML-engineering review. It was written on 2026-09-25, while the full-scale
run was paused, after epoch 1 exposed a category-head collapse. Review everything below
as an ML engineer before trusting any metric from the run on branch v5.5o.

## 1. Category head collapses at full scale (highest priority)

- **Observed:** epoch 1 at full scale (130.6M edges, 3.49M nodes, 4 classes). Both encoders
  (`full` and `cross_network`) predict **InitialAccess for every validation edge**:
  - per-class val accuracy: Benign 0, C2 0, Impact 0, InitialAccess 1.0;
  - CatAcc 0.0886, which equals the InitialAccess share of validation;
  - val macro-F1 0.054.
- **Link prediction is healthy:** val AUC 0.98 and AP 0.98.
- **The training category loss is low** (0.018 × `cat_loss_weight` 15), so the head fits in
  training but outputs a constant at validation.
- **Ruled out (measured or checked, 2026-09-25):**
  - the shared setup store: CTU-13 1+2 with and without it, both healthy;
  - the edge-feature gather at indices past 2^31 bytes (`np.take` with int64);
  - BatchNorm (there is none; the head is Linear→ReLU→Dropout→Linear).
- **Does NOT reproduce** on CTU-13 1+2 (2 classes, ~4.6M edges): Benign 0.976, C2 0.986.
- **Hypotheses to test:**
  - train→val label and feature shift from the per-capture 70/15/15 **time** split:
    CIC-2018 attacks sit in specific time windows, so val windows may hold different
    attack types or none;
  - focal loss (gamma 2) with inverse-frequency alpha (Benign 0.28, C2 5.0,
    Impact 0.97, InitialAccess 0.64) combined with `cat_loss_weight` 15;
  - memory or embedding saturation over a 59M-edge epoch;
  - edge-feature scaling (what `extract_canonical_edge_features` feeds the head,
    raw versus normalised);
  - whether training category accuracy is really high: log it per class during training;
  - the category head sees `[src_emb, dst_emb, edge_feat]`. Check how much of each it uses.

## 2. Evaluation and model selection

- **Selection score** = mean(inductive AP, val macro-F1) (about 0.52 at epoch 1). The link
  term dominates, so a collapsed category head still "IMPROVED". Decide whether that is
  the right selection metric.
- **Epoch length:** validation runs once per ~70-minute epoch (461k batches), and the guard's
  patience is 3 epochs. Consider mid-epoch validation checkpoints.
- **`UndefinedMetricWarning`** (single-class AUC slices): find which slices, and whether a
  metric silently excludes them.
- **Leakage fix (009f382):** the category head embedded twice, and the second pass saw the
  batch's own future. Audit the eval code, Branch A and downstream for any analogous
  within-batch or future leakage.

## 3. Data and labels

- **Train/test domain mismatch:** training flows come from PCAPs through our own flow
  extractor (Rust port of `telemetry/`). The CIC-2017 **test** flows come from CICFlowMeter
  CSVs. Different flow definitions mean a feature distribution shift. Quantify it, and
  decide whether the cross-year headline is measuring the model or the extractor.
- **Class coverage:** C2 exists only in CTU-13 (112k edges, against 46M Benign), and
  InitialAccess and Impact only in CIC-2018. Check which classes exist in val and test per
  corpus, and whether macro-F1 over absent classes is meaningful.
- **Label scoping:** PCAP labels come from time windows in the CSVs, scoped to known
  participants, and fall back to `time_only` when there are no participants. Measure how
  many edges carry time-only labels (noise).
- **263 corrupt PCAPs:** their days are truncated at the damaged record. Check whether
  attack windows fall in the missing parts, which would leave labels without traffic.
- **Node identity:** nodes are keyed per (capture, IP). The IP ablation (`cross_network`
  zeroes the octets and address-class flags) is the served variant. Confirm that no other
  identity shortcut remains (ports, per-capture node ids).

## 4. Training dynamics

- Batches are time-ordered and never shuffled (TGN memory requires it). Check the
  consequences for Adam, focal loss and the guard's LR step-backs.
- The inductive new-node draw holds out about 10% of the nodes seen in val/test, with a
  class-protection re-draw. Check its effect at this scale (Train=59M of 91M train-window
  edges).
- **Numerics:** fast path level 3 is bit-identical to level 2. Level 2 differs from the
  reference at fp32 rounding, and the reference's cuDNN GRU uses TF32 for weight gradients.
  Seeds: 3 planned (42, 123, 2024), and GPU atomics are nondeterministic.

## 5. Downstream (not yet run at PCAP scale)

- **Branch A:** check the hazard target, `hazard_tau`, the `budgeted_f1` operating point
  with alert budget 2.0, and the new columnar extraction path (bit-identical on tests, and
  5.5 GB for one full day).
- **Branch B:** check the credibility gate (must beat persistence by 2%) and what happens
  to DeepOP when it fails. The dry run shows Branch B is not credible on synthetic data,
  which is expected there.

## 6. Unused data worth a decision

- `DATA/logs`: Windows `.evtx` host event logs per host per day (1.8 GB), not used anywhere.
  They are a possible host-activity signal alongside the network graph.
