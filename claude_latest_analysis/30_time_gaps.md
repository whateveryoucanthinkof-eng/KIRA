# 30 — The "10-second forecast" is not what the models are trained on

**Date:** 2026-09-23
**Status:** decision needed. Machinery built, **off by default**.

## The finding

The contract says 2 s windows, 15 history steps, 5 forecast steps — a 10 s
horizon. But a trajectory's rows are a host's **active** windows, not its clock
ticks. "The next window" is whenever that host next appears.

Measured across **all 17 train captures at full density**, through the real
adapters (19,885,447 Branch A targets):

| target further than G from its history | CIC-2018 | CTU-13 | **all** |
|---|---:|---:|---:|
| 2 s | 63.4% | 78.9% | 70.1% |
| 6 s | 35.4% | 68.9% | 49.9% |
| **10 s — the contract horizon** | **23.3%** | **63.9%** | **40.9%** |
| 30 s | 7.8% | 55.3% | 28.3% |
| 60 s | 3.9% | 49.6% | 23.7% |
| 5 min | 1.2% | 40.1% | 18.1% |
| 1 hour | 0.2% | 24.8% | 10.8% |

**41% of training targets lie beyond the 10 s horizon the system claims.** For
CTU-13 it is 64%, and a quarter of CTU-13 "next windows" are over an hour
away. So for most CTU-13 samples the model is not forecasting the next 10
seconds; it is predicting what a host looks like **the next time it shows up**,
which can be hours later. "15 steps of history" likewise spans hours.

CIC-2018 is dense (96% of targets within a minute); CTU-13 is sparse — its
hosts come and go.

Branch B and DeepOP inherit the same structure: a "5-step rollout" is the next
five activity bursts, not the next 10 seconds.

## The options

| | what it means | cost |
|---|---|---|
| **A. Enforce the contract** (`--max-gap-seconds 10`) | Samples only where history and target sit within 10 s of each other. The model genuinely forecasts 10 s ahead. | **Drops 40.9% of targets** — 8.1M of 19.9M; 64% of CTU-13 |
| **B. Middle ground** (`--max-gap-seconds 60`) | Horizon becomes "within a minute". Cuts the multi-hour bridges. | Drops 23.7% |
| **C. Keep everything, change the claim** | Rename the task honestly: *next-activity* forecasting, horizon unbounded. No data lost. | Every "10-second" statement in docs, the dashboard and the pitch is wrong and must change |
| D. Dense clock | Emit an idle window every 2 s whether or not the host is active. The contract becomes literally true. | Explodes the data (CTU-13 spans hours × 600k hosts); major re-architecture |

**Not an option:** leaving it as-is while continuing to describe the system as
a 10-second forecaster. That is the one choice that is actually wrong.

## On "dilution"

The no-dilution rule was about thinning data *uniformly* — strides and row caps
that discard information for no reason. Option A removes a specific, definable
class of sample: those whose target does not satisfy the task definition. I read
that as correctness rather than dilution — but it removes 41% of samples, so it
is your call, not mine.

## Original recommendation (superseded above)

B (60 s), then A as a comparison. B removes every multi-hour
bridge (the samples that are unambiguously not forecasting) while keeping 76% of
the data, and it gives an honest statement — "forecasts within a minute" — that
the evidence supports. Running A alongside tells us what the true 10 s task
looks like and whether the model can do it at all.

Whatever is chosen, the stated horizon must match it.

## Branch B and DeepOP: solved differently, and without dropping anything

Branch B already had the better answer built — and it was never connected.
`LazyHostRolloutDataset` emits each step's **real elapsed time**, and
`HostWorldDynamicsTransformer.rollout()` encodes it (`dt → time_encoder →
x + t_emb`). But `retrain_future_models_live.py`, the trainer that produces
the **served** checkpoint, called `wdt.rollout(h, K=...)` with no times. The
model was told every step was 2 s apart when the median is 14 s. A comment in
the standalone trainer said so: *"Passing these two tensors is the whole change
it needs."* It never landed.

Now wired end to end:

| link | passes |
|---|---|
| Branch B train + validation | real `t_history`, real `t_future` |
| DeepOP dataset | emits both, same convention as Branch B (pinned equal) |
| DeepOP rollout precompute + live paths | real `t_history`, real `t_future` |
| serving adapter | real `t_history` (per-host window times, kept in lockstep with states) |
| trajectory_assembler | real `t_history` from snapshot `window_start` |

At serving `t_future` is deliberately `None`: it falls back to the uniform
2/4/6/8/10 s grid, which **is** the contract's question. Training teaches
h(t + Δt) across real Δt; serving chooses which Δt to ask about. That is a
query, not a leak.

**This makes the gap problem a non-issue for Branch B and DeepOP with zero
samples dropped**, which is what the no-dilution rule wants.

## Branch A: given the same time channel

None of Branch A's 15 temporal attributes spans windows — all are per-window
aggregates — so it saw fifteen feature vectors with no way to tell 2 s from
2 hours. It now has the same fix as Branch B, without touching the 27-D
contract:

- `MultiTaskLSTM.forward(x, t_history=None)` adds `time_proj(time_encoder(log t))`
  to the input. Time is **log-compressed** first, because gaps run from 2 s to
  hours and the encoder's learned `cos(Linear(t))` turns hour-scale raw seconds
  into noise.
- `time_proj` is **zero-initialised**, so the output is bit-identical to the
  old model until training moves it.
- `load_state_dict` backfills **only** the new `time_*` keys, so every existing
  checkpoint still loads — and every other key is still checked strictly
  (pinned by a test that deletes an LSTM key and expects a failure).
- The dataset, trainer, `_evaluate`, the serving adapter (including its
  saliency path, so attributions describe the forward actually served) and
  the assembler all pass it.

**So Branch A no longer needs to drop anything either.** `--max-gap-seconds`
stays as an option, off by default, for anyone who wants the strict 10 s task
as a comparison — but the gap problem is now solved everywhere without
dilution.

## Updated recommendation

Train all three with the time channels on and **no gap cut**. That keeps all
19.9M samples, and it is the honest version of the task: the model forecasts a
host's next activity *and knows how far away it is*. At serving, `t_future`
asks about +2 … +10 s specifically.

The strict 10 s cut (option A) is still worth one comparison run later, to see
how the model does on the contract's literal task.

## What is built

- `LazyHostSequenceDataset(..., max_gap_seconds=G)` — cuts each trajectory at
  gaps > G; no sample's history or target crosses a cut; history is clipped to
  its segment and left-padded. Reports `n_dropped_by_gap`.
- `retrain_branch_a_live.py --max-gap-seconds G` — default `None`, prints kept
  and dropped counts per split when set.
- `tests/test_gap_segmentation.py` — five tests; mutation-checked (a history
  that ignores the segment start fails the history test).
- Branch B and DeepOP: time-conditioned instead (above) — no gap option needed.

## Related fix already landed

Before this, trajectories also merged **across captures** — the same address in
two captures was one trajectory (all 350 fabricated CIC-2018 hosts across days;
28,474 CTU-13 hosts across scenarios; 2018 rows preceding 2011 rows). Fixed
unconditionally in `546d459`: that was never a design choice.
