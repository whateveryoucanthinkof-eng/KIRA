# 29 — An external critique, checked against the code, and what was fixed

**Date:** 2026-09-24
**Input:** a long review titled "Architectural and Algorithmic Catastrophes in the CyberWorld
v5.5o Intrusion Detection Framework". "v5.5o" appears nowhere in this repo. The copy received was
missing a chunk (end of §2.1 through the start of §3.2).
**Verdict:** most specific claims are wrong about this code. Five general concerns were fair. Four
of them were fixed or instrumented here. The fifth, adversarial robustness, is only partly covered
by the flooding test in §2.3; the rest needs an evaluation this tree cannot run.

---

## 1. Claims the code contradicts

| Claim | What the repo shows |
|---|---|
| Reports 24/25/26 prove data leakage and "thrashing in deployment" | 24: accuracy could not reveal majority-class collapse. 25: the val/test gap was inflated by a loss offset and two task weights frozen by `clamp`. 26: the training job exhausted RAM. None is leakage; none is deployment. |
| Random node/edge splits leak labels through message passing | Splits are whole captures (`splits.lock.json`); CTU-13 held out by scenario; a cross-year protocol exists. TGN pretraining uses chronological quantiles plus an inductive node holdout. |
| `message_aggregator.py` over-smooths malicious edges | Never executed: TGN builds an aggregator only when `use_memory=True`, and every shipped encoder config has `use_memory: false`. |
| Poisoned memory carries forward; attackers poison through continuous learning | Memory is off. Nothing retrains on live traffic. |
| Branch B is trained with teacher forcing | Its loss is on `wdt.rollout()` output, which feeds back its own predictions; a persistence skill gate blocks DeepOP otherwise. (DeepOP *is* teacher-forced in training, and its free-running score is already reported as the served number.) |
| No shallow baseline | Nine exist (`cyberworld_v4/baselines/`, `world_model/baselines/`). On the one committed run gradient boosting beat the model, 0.780 vs 0.690 ROC-AUC. |
| Conformal intervals auto-close alerts | Alerts are a point threshold (`model_adapter._alert_level`). No interval gates an alert; Branch B's radii are computed and discarded. |
| Retrain output silently ignored by serving | Fixed in report 23; the adapter refuses contract-mismatched checkpoints. |
| `nodes/services/app_server.py` is the inference engine | It is a simulated victim service in the range. |
| Loader starvation (report 21) makes the live sensor drop packets | Report 21 is offline training throughput; `bita/.../DataLoading.py` is not on the live path. |
| Branch A has a benign-reconstruction task that swamps the others | Its tasks are risk, technique, gradation. |
| "Benign wrapper" payloads anchor the embedding | No payload is ever seen; features are flow statistics. |

## 2. Fair concerns, and what was done

### 2.1 Conformal coverage on the attack class — fixed

`SplitConformal` guarantees marginal coverage only, and `scripts/train_v4.py` fits it on a binary
future-attack target. The benign majority sets the quantile. On a synthetic stream with a 17% base
rate, 0.95 marginal coverage came with **~0.7 coverage on attack windows** (pinned in the test).

- `SplitConformal.evaluate(..., groups=label)` reports coverage per label and the worst label.
- New `LabelConditionalConformal` (Mondrian): one quantile per class, a guarantee per class, paid
  for with ambiguous two-label sets where the model cannot separate the classes.
- `train_v4.py` prints both, and the credibility gate fails a run when either the interval's
  worst-label coverage or the label-conditional worst-class coverage misses target by > 0.05.

Tests: `tests/test_conformal_is_label_conditional.py` (9).

### 2.2 Blind spots under a flood — instrumented

The review blamed the wrong module, but the gap was real. The sensor is one Python `recvfrom` loop.
The kernel drops frames once its buffer fills, and nothing read the counter. Separately,
`AsyncStateRecorder` is the only route from sensor to models, because the backend tails its file,
and it dropped whole windows silently when its queue was full.

- `StreamingPacketSniffer.read_kernel_stats()` reads `PACKET_STATISTICS` (per interval, totalled);
  `start()` asks for a 32 MiB receive buffer and reports what the kernel granted.
- Both recorders count drops. Every live window carries `state["capture"]`: kernel packets,
  drops, drop ratio, capture-loop errors, undelivered windows. The window log line flags them.
- `control_backend/capture_accounting.py` detects window-id gaps (windows never scored) and
  incomplete windows. `system_status` now carries `sensorKernelDrops`, `incompleteWindows` and
  `windowsMissed`, shown as the "Sensor gaps" tile on the Overview (analysis 30).

Tests: `tests/test_capture_blind_spots_are_counted.py` (8).

### 2.3 Benign-flow flooding — measured, not "fixed"

The mechanism the review named does not run. The one that does is sharper. At serve time each
host's 12-D latent is built from its **last `n_neighbors` (10) flows in the 2 s window**. Verified
with a real encoder:
**10 benign flows after an attack flow make the target's latent identical (atol 1e-6) to the same
window with no attack.** That is eviction, not dilution. The 15 attributes still count the attack
flow, so the model as a whole is not blind, only its graph half.

Training uses the same cut-off (`--n_degree 10`, most-recent), so there is no train/serve mismatch,
and changing it at serve time would create one. The done changes:

- `bita/train.py` writes `n_neighbors` and `neighbor_sampling` into the encoder config, and
  `scripts/write_encoder_config.py` writes the defaults every shipped encoder used.
  `build_or_load_tgne_ta` attaches them to the model. `HostTrajectoryExtractor` reads them instead
  of the literal `10` / `uniform=False`, which matched training only by coincidence.
- `HostTrajectoryExtractor.neighbor_exposure_report()` counts truncated host-windows, flows outside
  the latent, and attack host-windows where **every** attack flow fell outside it.
  `retrain_branch_a_live.py` prints it per split and saves it in the checkpoint as
  `neighbor_exposure`.

The next retrain answers how often this happens on the real corpus. A real mitigation (e.g. a
neighbour sample stratified over the window, or a count-invariant aggregate) is a retrain with a
measured A/B, not a serve-time switch.

Tests: `tests/test_neighbor_cutoff_is_measured.py` (8).

### 2.4 Gradient conflict between Branch A's tasks — instrumented

Partly right: the default `--architecture paper` uses fixed 0.5/0.3/0.2 weights. Nobody had measured
whether the tasks conflict, and PCGrad/GradNorm without that measurement would be an unvalidated
architecture change.

- `compute_loss` now delegates to `task_losses()` (same objective, pinned by test).
- `MultiTaskLSTM.task_gradient_conflict(x, targets)`: pairwise cosine of per-task gradients on the
  shared trunk, raw and objective-weighted norms, dominant task. No `.grad` is written.
- The Branch A trainer prints it on the first batch of every epoch and stores it in
  `metrics["task_gradients"]`. Persistently negative cosines would justify gradient surgery;
  near-zero or positive would not.

Tests: `tests/test_task_gradient_conflict.py` (11).

### 2.5 No adversarial evaluation — partly addressed

§2.3's test is the first evasion test in the repo. There is still no evaluation of feature-space
evasion or slow drift against trained weights. That needs the corpora and a trained encoder, and
neither is on this machine.

## 3. Also fixed

`model_contract.DEEPOP_WINDOW_SIZES` said `[2, 4, 7]`. `cwa.py`, both trainers and
`deepop.manifest.json` say `[2, 4, 8]`. The contract was wrong; `test_model_contract` had been
failing on it.

## 4. Not fixable in this tree

Everything below needs the corpora or a retrain, and neither is available here:

- **Retrain + a credible v4 benchmark.** The committed `results/v4_benchmark.json` is not a result
  (see `results/README.md`).
- **CIC-2018 `is_private` shortcut, fabricated host IPs, 2-vs-7 technique classes in test.** These
  are data limits (reports 27, 28).
- **The 30 s history / 10 s horizon scope.** This is a design decision (report 28, concern 2).

## 5. Tests

| | passed | failed | skipped | xfailed |
|---|---|---|---|---|
| before | 848 | 6 | 2 | 3 |
| after | **885** | **5** | 2 | 3 |

+36 new tests and one fixed (`test_model_contract`). The five remaining failures are the known
environmental ones listed in report 28 §6: `test_split_manager_uses_the_lock` ×2 fail because the
corpora are absent. `test_rollout_cache` ×1 and `test_spill_file_is_reclaimed` ×2 fail because
Windows lacks POSIX unlink-while-mapped semantics. Run on Windows, Anaconda Python 3.13.5,
torch 2.6.0.
