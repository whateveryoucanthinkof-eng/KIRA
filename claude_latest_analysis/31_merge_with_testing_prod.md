# 31 — Merging testing-prod into v5.5o

**Date:** 2026-09-24
**Why:** v5.5o was forked from `5060d62` (testing-prod, 2026-09-22 16:31). testing-prod then gained
16 commits of fixes. Training on v5.5o without them would lose those fixes; overwriting testing-prod
with v5.5o would lose v5.5o's. This merge keeps both.

Before merging, 45 files that a Windows copy had flipped to CRLF were restored to LF, with a
`.gitattributes` added to pin LF. That dropped the conflicts from 15 files to 12 (33 blocks).

## How each conflict was resolved

| File | Both sides | Resolution |
|---|---|---|
| `run.sh` | testing-prod: sequential stages, 15G caps (measured throttling), status capture under `set -e`. v5.5o: hazard target for Branch A | testing-prod's structure; Branch A `--risk-objective soft_bce --risk-target hazard` (analysis 30) |
| `sequence_dataset.py` | v5.5o: `min_history_steps` (no zero-padded samples). testing-prod: `max_gap_seconds` (no sample crosses a time gap) | both: a target must not open a segment AND needs `need` real steps inside its segment; `padded` is now counted, not assumed 0 |
| `model_adapter.py` | v5.5o: verdict = model only, advisory rules, mitigation display-only, forecast bands. testing-prod: same-window clipping for both feature halves, per-host time history, `rule_risk` provenance, lead-time fix, "alerting unreachable" warning | v5.5o's verdict design + testing-prod's clipping and time history. The "unreachable" check is removed: its premise (rules capping internal traffic at 0.40) no longer exists. Lead time = 0.0 when the current window crosses (testing-prod), in forecast steps (v5.5o) |
| `model_adapter.py` (not a conflict, found while merging) | testing-prod serves Branch B with `t_future=None`, i.e. a 2 s grid, "the contract's question"; v5.5o's contract asks about 5 × 30 s | serving now passes `t_future = (k+1) × step_seconds`, so the forecast answers the horizon it is labelled with |
| `telemetry_service.py` | both removed fake throughput / loss / latency | testing-prod's text; v5.5o's capture accounting kept |
| `main.py` | testing-prod: replay on a private adapter instance. v5.5o: rules never move risk | private adapter, rules forced off for replay |
| `causal_edge_scorer.py` | both fixed the untrained MLP: v5.5o deleted it (transparent heuristic), testing-prod guarded it | deletion (stronger); `test_serving_causal_edge_guard.py` rewritten to pin that no learned scorer exists |
| `multi_dataset_stream.py` | both turned the heuristic relabelling off: v5.5o by constructor flag, testing-prod by env var + a measurement note | either switch enables it; the note is kept |
| `retrain_branch_a_live.py` | label filter + neighbour exposure + gradient diagnostic vs capture namespaces + time channel + gap segmentation | all; the gradient diagnostic now receives `t_history` |
| `retrain_future_models_live.py` | forecast bands vs real step times; DeepOP selection on free-running macro F1 | all, with v5.5o's extra checkpoint fields |
| `chunked_extraction.py`, `flow_to_temporal_event.py`, `trajectory_store.py` | the same bug fixed twice / two new attributes | testing-prod's code, v5.5o's comments; both attributes |

## Tests adjusted

- testing-prod's gap, time-channel and left-padding tests build short histories. v5.5o's default
  now requires a full window, so they pass `min_history_steps=1`. They test gaps and padding, not
  the floor.
- v5.5o's adapter stubs gained the new time history, and their signatures accept `t_history`.
- `test_alert_thresholding.py`: the three tests of the removed "unreachable" check are replaced by
  one test that pins the ceiling cannot return.
- A test reading source now opens it as UTF-8 (Windows defaults to cp1252).

**Result on this Windows copy:** 985 passed, 3 failed, 1 collection error. All four are
environmental: the corpora are absent (2), Windows unlink semantics (1), and
`test_serving_replay_isolation.py` needs the retrained encoder at import. That test fails
identically on untouched testing-prod here.
