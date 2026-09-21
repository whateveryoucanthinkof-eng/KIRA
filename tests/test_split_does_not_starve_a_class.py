"""No attack class may be starved of training data by the temporal split.

CIC-2017 and CIC-2018 organise attack types BY DAY: Monday is benign, Friday
afternoon is PortScan, and so on. Cutting a whole corpus at one time quantile
therefore segregates entire attack classes into the future half.

Measured on the real corpus: **Recon had SEVEN training samples out of
11,792,781.** CIC-2017's 70% cut fell at 13:13 UTC and all 158,930 PortScan
rows run 13:00-15:59 on the Friday-afternoon capture, so essentially every
Recon record landed in val/test. A class with 7 samples cannot be learned, and
Recon is the earliest attack stage -- precisely what a forecaster exists to
catch.

Cutting inside each CAPTURE keeps the causal property that matters (train
precedes val precedes test within every capture) while letting every attack
type appear on all three sides.
"""

import numpy as np
import pandas as pd
import pytest

from bita.train import split_data

# Three captures on one day. The third is "Friday afternoon PortScan": it is
# late in the corpus timeline AND holds the only Recon records.
BASE = 1499400000.0
BENIGN, RECON = 0, 1


def _frame(n_each=3000):
    rows = []
    for cap, (t0, label) in enumerate([
        (BASE, BENIGN),                  # morning, benign
        (BASE + 10_000, BENIGN),         # midday, benign
        (BASE + 20_000, RECON),          # late, the ONLY Recon capture
    ]):
        for i in range(n_each):
            rows.append((cap, t0 + i, label))
    df = pd.DataFrame(rows, columns=["capture", "ts", "label"]).sort_values("ts")
    df = df.reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(0)
    df["u"] = rng.integers(1, 200, n)
    df["i"] = rng.integers(1, 200, n)
    df["idx"] = np.arange(1, n + 1)
    df["source"] = 0                      # all one corpus
    return df


def _split(df):
    ef = np.zeros((len(df) + 1, 12), dtype=np.float32)
    nf = np.zeros((300, 12), dtype=np.float32)
    out = split_data(df, ef, nf)
    return out[2], out[3], out[4], out[5]


def test_the_rare_class_reaches_training():
    """The actual regression: Recon must not be starved."""
    _full, train, _val, _test = _split(_frame())
    counts = np.bincount(train.labels.astype(int), minlength=2)
    assert counts[RECON] > 100, (
        f"Recon got {counts[RECON]} training samples -- the split is starving it"
    )


def test_a_per_corpus_cut_would_have_starved_it():
    """Pin the bug, so this test is known to discriminate."""
    df = _frame()
    v = np.quantile(df.ts.values, 0.70)
    starved = int((df.label.values[df.ts.values <= v] == RECON).sum())
    total = int((df.label.values == RECON).sum())
    assert starved < total * 0.2, (
        "a single corpus-wide cut should starve the late class -- if it does "
        "not, this fixture no longer reproduces the bug"
    )


def test_every_class_appears_in_every_split():
    _full, train, val, test = _split(_frame())
    for name, part in (("train", train), ("val", val), ("test", test)):
        present = set(np.unique(part.labels).astype(int))
        assert present == {BENIGN, RECON}, f"{name} is missing a class: {present}"


def test_each_capture_is_still_cut_chronologically():
    """Causality within a capture is the property worth keeping."""
    df = _frame()
    _full, train, val, test = _split(df)
    ts_to_cap = dict(zip(df.ts.values, df.capture.values))
    for cap in df.capture.unique():
        tr = [t for t in train.timestamps if ts_to_cap[t] == cap]
        va = [t for t in val.timestamps if ts_to_cap[t] == cap]
        te = [t for t in test.timestamps if ts_to_cap[t] == cap]
        assert tr and va and te, f"capture {cap} missing from a split"
        assert max(tr) <= min(va), f"capture {cap}: train overlaps val"
        assert max(va) <= min(te), f"capture {cap}: val overlaps test"


def test_capture_tag_is_preferred_over_corpus_tag():
    """Both columns present -> capture wins, because it is the finer grain."""
    df = _frame()
    assert "capture" in df.columns and "source" in df.columns
    _full, train, _val, _test = _split(df)
    assert np.bincount(train.labels.astype(int), minlength=2)[RECON] > 100


def test_falls_back_to_corpus_then_global_when_capture_is_absent():
    df = _frame().drop(columns=["capture"])
    _full, train, val, test = _split(df)
    assert train.n_interactions > 0 and val.n_interactions > 0 and test.n_interactions > 0

    df2 = _frame().drop(columns=["capture", "source"])
    _full2, train2, val2, test2 = _split(df2)
    assert train2.n_interactions > 0 and val2.n_interactions > 0 and test2.n_interactions > 0
