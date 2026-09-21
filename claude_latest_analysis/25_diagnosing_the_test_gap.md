# 25 — Diagnosing Branch A's val 0.66 vs test 26.36

**Date:** 2026-09-22
**Method:** read the learned parameters out of the checkpoint, decompose the loss.

---

## First: the "40× gap" was overstated

I reported a 40× generalisation gap from the raw reported losses. That framing
was wrong, and the checkpoint says why.

The objective is Kendall & Gal homoscedastic weighting:

```
L = sum_i [ exp(-s_i) * L_i + s_i ]
```

The `+ s_i` terms are a **constant offset** that does not depend on the data.
Reading them out of the saved checkpoint:

| parameter | raw | clamped | weight `exp(-s)` | |
|---|---|---|---|---|
| `log_var_risk` | −3.0090 | −3.0000 | **20.086** | saturated |
| `log_var_tech` | −3.0243 | −3.0000 | **20.086** | saturated |
| `log_var_grad` | −1.0374 | −1.0374 | 2.822 | |
| | | **Σ = −7.0374** | | |

Subtracting that offset:

| | reported | Σ weight·task_loss | mean raw task loss |
|---|---|---|---|
| train | −4.3365 | 2.70 | 0.086 |
| val (ep 6) | 0.6601 | 7.70 | 0.246 |
| held-out test | 26.3618 | 33.40 | 1.066 |

The real degradation is **4.3×**, not 40×. Still a large gap — but "40×" was
an artefact of comparing numbers that contain a −7.04 constant and a 20×
multiplier. `risk_mae` 0.2268 → 0.3381 (1.5×) and `tech_accuracy` 0.8818 →
0.7807 are the honest headline figures.

---

## Defect 1 — the selection metric was not comparable across epochs

Every `s_i` is a **learned parameter that moves during training**. They start
at 0 and ended at Σ = −7.04. So the validation loss reported at epoch 1 and at
epoch 8 are not the same quantity: the weighting differs and the additive
constant has drifted by 7 units.

A large part of the apparent fall in validation loss over the run was that
offset moving, not the model getting better. And selecting `min(val_loss)`
picked epoch 6 at 0.6601 — roughly half the median of every other epoch — which
then scored worst on the held-out test.

**Fix.** `--select-on`, defaulting to `composite`:

```
composite = 0.5 * tech_macro_f1 + 0.5 * (1 - min(risk_mae, 1))
```

Both terms are bounded in [0, 1] and independent of the loss weighting, so they
mean the same thing at every epoch. `macro_f1` and the previous `val_loss`
remain available; `val_loss` is kept so an old run can be reproduced, not
because it is sound.

This changes which checkpoint a run keeps, which is a modelling decision — it
is stated here and on the CLI rather than made silently.

## Defect 2 — two of the three task weights were frozen

`x.clamp(lo, hi)` passes **no gradient** where `x` is outside `[lo, hi]`.
`log_var_risk` settled at −3.0090 and `log_var_tech` at −3.0243: both just past
the bound, where their gradient is identically zero. They were dead for the
rest of training, and could not have recovered even if the balance they implied
stopped being right. Two of three task weights had silently become constants
pinned at the maximum 20.086.

**Fix.** The parameters are projected back into range in place
(`project_()`), which is ordinary projected gradient descent: the value stays
bounded and the gradient stays live at the boundary. Verified — at the
checkpoint's own values the gradients go from exactly 0 to 0.799 and −5.026,
and `log_var_tech`'s is *negative*, meaning it can now climb back off the bound
and rebalance the technique task.

`project_()` is also called after `optimizer.step()`, because a step can leave
a parameter epsilon outside the bound and a checkpoint saved in that window
records the out-of-range value — which is precisely how −3.0090 came to be
stored.

## Defect 3 — the loss was uninterpretable, and cost 1.1M syncs an epoch

`compute_loss` returned seven metrics, each built with `.item()`: a
host-device sync per call, **seven per batch, ~1.1M per epoch** at 161,439
batches. Nothing consumed them — the only caller that binds the dict
(`train_branch_a.py:306`) never reads it. That also means my earlier claim of
"removed ~161k syncs per epoch" was incomplete: the loop-level sync went, but
`compute_loss` kept syncing regardless, capping the benefit of the loader work.

They are now detached 0-dim tensors, which format and `float()` like scalars,
so a caller pays only when it asks. `_evaluate` now accumulates them on-device
and the epoch line reports per-task losses **and** the learned weights — with
those printed, the saturation above would have been obvious on epoch 1 instead
of being dug out of a checkpoint eight epochs later.

---

## Still open

Whether the technique head has partly collapsed. The decomposition rules out
"the loss exploded" as a mystery, but `tech_accuracy` 0.7807 on the test split
is only meaningful against that split's own majority baseline, which nothing
has measured yet. `--eval-only` now scores **validation and held-out test side
by side** with macro F1, baseline and lift; it is queued behind the downstream
retrain.

**Tests:** `tests/test_selection_and_uncertainty.py` (12) — the composite score
is provably invariant to the loss offset while `val_loss` is moved entirely by
it; composite prefers the better model where `val_loss` prefers epoch 6; a
parameter pinned at the bound keeps a live gradient and can climb back off it;
a post-step parameter *is* out of range until projected; and the metrics dict
holds no graph and forces no sync.

Suite: **480 passed, 2 xfailed**.
