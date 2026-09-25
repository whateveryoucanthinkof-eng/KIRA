# The CSV path cannot support a forecasting benchmark

**Date:** 2026-09-20
**Status:** settled by three full training runs on an RTX 4060, not by argument

This is a negative result. It is the most useful thing produced by actually running the v4 trainer, and
it is worth more than the numbers those runs printed — because those numbers looked excellent.

---

## What the runs reported

```
NOWCAST  P(attack now)   PR-AUC 0.999
FORECAST +2s..+10s       PR-AUC 0.9998 at every horizon
```

Any of that on a slide would be a serious misrepresentation. Here is why.

## Why it means nothing

### 1. The test base rate is 0.9997

The final test split was 99.97% attack (`tp=0 fp=0 tn=5 fn=15255`). **PR-AUC is bounded below by the
base rate**, so at that rate *any* ranking scores ~1.0. Demonstrated directly: feeding **pure random
noise** into that split gives PR-AUC **0.9999**.

Persistence scored 0.9997 for exactly the same reason.

### 2. Label churn is 0.0007

`A_{t+K}` equals `A_t` for 99.93% of samples. A model that ignores the future entirely is almost
perfect. There is no forecasting content to learn, so "forecast PR-AUC" is nowcast PR-AUC wearing a
different label.

### 3. The test split contains one host

`CI undefined, n_groups=1`. A group bootstrap needs groups; one host gives no interval at all. Any
number quoted from it has no stated uncertainty because none can be computed.

### 4. Supporting symptoms

| Signal | Value | Reading |
|---|---|---|
| Brier | **0.98–0.99** | probabilities are near-maximally wrong while ranking is perfect |
| Temperature | pinned at **10.0**, then **0.05** | boundary hit, not a fit |
| ECE after scaling | **0.0000** on 154 samples | split is single-class; calibration trivially satisfied |
| F1 | **0.000** | model predicts one class at any threshold |
| Validation split | **128 samples** | threshold selection is noise |

## The root cause is the data, not the model

```
66,417 nominal hosts
        ↓  require 20 consecutive 2-second windows (15 history + 5 forecast)
        14 hosts
        ↓  those 14 are the DENSE hosts — i.e. the fabricated ones
        one host holds most of the samples
        ↓
test = 1 host, ~100% attack, churn 0.0007
```

Each step is a consequence of a defect already recorded in the audit:

- **D1** — `cic2018_adapter.py:65,70` fabricates host IPs from the row index for 9 of 10 days. Hosts are
  `192.168.10.{i % 250 + 1}`, so a "host trajectory" is every 250th row. The hosts dense enough to
  survive the 20-window requirement are precisely the synthetic ones.
- **D6** — `--rows-per-file` is a prefix. The first 60k rows of a day are 87–100% single-label. `--stride`
  lifted onset-within-horizon from 0.000 to **0.072**, which is real progress, but churn over the full
  horizon stayed at 0.0007 because the surviving hosts still do not transition.

## This is now enforced, not remembered

`scripts/train_v4.py` refuses to present such a run as a result:

```
====================================================================
BENCHMARK NOT CREDIBLE — do not report these numbers
====================================================================
  - persistence is near-perfect: the label does not change over the horizon
  - label churn 0.0007: almost nothing to forecast
  - test split has 1 host group(s): no usable confidence interval
  - test base rate 0.9997 is extreme: PR-AUC is near 1.0 for any ranking
  - temperature hit the search boundary: calibration split is unrepresentative
  - validation split has 128 samples: threshold selection is noise
====================================================================
```

`results/v4_benchmark.json` carries `"credible": false` and the list. Supporting guards:
`detection_metrics` reports `pr_auc_lift` beside PR-AUC at extreme base rates; `TemperatureScaler`
reports `at_grid_boundary`; `chronological_split` reports `dominated_by_one_group`.

## What would fix it

**The PCAP corpus, which is why `data_unification/pcap_adapter.py` exists.**

| | CSV path | PCAP path |
|---|---|---|
| Host identity | fabricated from row index (9/10 days) | **real**, in the filename |
| Hosts with long trajectories | 14 | ~445 per day |
| Continuous data per host | sparse, interleaved | **~9 hours** |
| Packet features | none | 30, PS-required |
| Label churn | 0.0007 | spans full attack windows |

Verified working on `wed_14/UCAP172.31.69.18`: `ttl_std 22.27`, `tcp_window_mean 253.14`,
`payload_size_std 68.42`, real scan scores — with `host=172.31.69.18` taken from the filename rather
than invented.

**Caveat, stated rather than buried:** 263 of 4,457 captures (5.9%, 56.2 GB) carry mid-file record
corruption (report 06). The adapter keeps each file's valid prefix instead of aborting, so the corpus is
usable today, but the affected days should be re-fetched before final numbers.

## The honest position

There is no v4 benchmark result, and there should not be one yet. Three runs produced numbers that
would have passed casual review and were discarded because the guards caught them.

A negative result with a stated cause is a stronger artifact than a 0.9998 that does not survive a
question about its base rate.
