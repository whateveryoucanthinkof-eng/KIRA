"""The temporal split must be taken per corpus, never over the concatenation.

CTU-13 is 2011, CIC-2017 is July 2017, CIC-2018 is Feb/Mar 2018 -- three
disjoint absolute time ranges. A global `np.quantile(ts, [.70, .85])` over
records sorted by absolute time therefore does not split time, it splits by
CORPUS: train on 2011 Czech university botnet traffic, validate and test on
2018 AWS enterprise traffic.

That produced all three symptoms at once: 3 of 5 categories absent from
training (CTU-13's vocabulary is only Benign and C2, which is why the focal
alpha came back with weights of exactly 1.0 for the other three), inductive
AUC at chance on a disjoint host population, and a frozen CatAcc.
"""

import numpy as np
import pandas as pd
import pytest

from bita.train import split_data

CORPORA = {0: 1.31e9, 1: 1.499e9, 2: 1.518e9}   # 2011, 2017, 2018

# Sized like the real corpora, where CTU-13 dominates. That is what makes a
# global 70% cut land INSIDE CTU-13 and starve training of the other two -- with
# equal-sized corpora the bug does not reproduce, which
# test_a_global_split_would_have_failed_this exists to keep honest.
CORPUS_ROWS = {0: 7500, 1: 1200, 2: 1300}


def _frame():
    rows = []
    for tag, base in CORPORA.items():
        for i in range(CORPUS_ROWS[tag]):
            rows.append((tag, base + i))
    df = pd.DataFrame(rows, columns=["source", "ts"]).sort_values("ts").reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(0)
    df["u"] = rng.integers(1, 400, n)
    df["i"] = rng.integers(1, 400, n)
    df["idx"] = np.arange(1, n + 1)
    # Each corpus has its own label vocabulary, as the real ones do.
    df["label"] = df["source"].map({0: 1, 1: 2, 2: 3})
    return df


def _split(df):
    """-> (full, train, val, test). split_data returns
    (node_features, edge_features, full, train, val, test, ind_val, ind_test)."""
    ef = np.zeros((len(df) + 1, 12), dtype=np.float32)
    nf = np.zeros((500, 12), dtype=np.float32)
    out = split_data(df, ef, nf)
    return out[2], out[3], out[4], out[5]


def test_every_corpus_appears_in_train_val_and_test():
    df = _frame()
    full, train, val, test = _split(df)
    ts_to_src = dict(zip(df.ts.values, df.source.values))
    for name, part in (("train", train), ("val", val), ("test", test)):
        got = {ts_to_src[t] for t in part.timestamps}
        assert got == set(CORPORA), f"{name} holds corpora {sorted(got)}, not all three"


def test_every_label_appears_in_training():
    """The real failure: 3 of 5 categories had ZERO training samples."""
    df = _frame()
    _full, train, _val, _test = _split(df)
    assert set(np.unique(train.labels)) == {1, 2, 3}


def test_a_global_split_would_have_failed_this():
    """Pin the bug itself, so the test is known to discriminate."""
    df = _frame()
    v, t = np.quantile(df.ts.values, [0.70, 0.85])
    global_train_labels = set(np.unique(df.label.values[df.ts.values <= v]))
    assert global_train_labels != {1, 2, 3}, (
        "a global quantile split should starve training of some corpora -- "
        "if it does not, this fixture no longer reproduces the bug"
    )


def test_each_corpus_is_cut_chronologically_within_itself():
    """Within a corpus, train must precede val must precede test."""
    df = _frame()
    _full, train, val, test = _split(df)
    ts_to_src = dict(zip(df.ts.values, df.source.values))
    for tag in CORPORA:
        tr = [t for t in train.timestamps if ts_to_src[t] == tag]
        va = [t for t in val.timestamps if ts_to_src[t] == tag]
        te = [t for t in test.timestamps if ts_to_src[t] == tag]
        assert tr and va and te, f"corpus {tag} missing from a split"
        assert max(tr) <= min(va), f"corpus {tag}: train overlaps val in time"
        assert max(va) <= min(te), f"corpus {tag}: val overlaps test in time"


def test_the_three_splits_partition_the_data():
    df = _frame()
    full, train, val, test = _split(df)
    # train drops edges touching inductive nodes, so it is a subset, not exact
    assert train.n_interactions + val.n_interactions + test.n_interactions <= full.n_interactions
    assert val.n_interactions > 0 and test.n_interactions > 0
    assert not (set(val.timestamps) & set(test.timestamps))


def test_single_corpus_data_still_uses_a_global_cut():
    """The Warden loader emits no `source` column; that path must keep working."""
    df = _frame()
    df = df.drop(columns=["source"])
    full, train, val, test = _split(df)
    assert full.n_interactions == len(df)
    assert train.n_interactions > 0 and val.n_interactions > 0 and test.n_interactions > 0
