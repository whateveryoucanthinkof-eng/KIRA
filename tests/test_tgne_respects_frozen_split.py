"""TGNE must not see the captures reserved for downstream validation and test.

TGNE is NOT purely self-supervised. Alongside link prediction it trains a
category head on each edge's `coarse_category`. Training it over the whole
corpus therefore pushed LABEL information from the frozen val and test captures
into the encoder -- and Branch A, Branch B and DeepOP all consume its
embeddings.

That contaminates every downstream "held-out" number at the encoder, where it
is invisible to any check performed on the branches themselves. It is the
subtlest leak in the pipeline and the easiest to publish by accident.

`load_and_preprocess_unified_dataset(splits=("train",))` is the guard, and
"train" is the default.
"""

import inspect

import pytest

import bita.train as bt
from data_unification.split_policy import load_lock, split_of_path


def test_the_default_is_train_only():
    """A permissive default would reintroduce the leak silently."""
    sig = inspect.signature(bt.load_and_preprocess_unified_dataset)
    assert sig.parameters["splits"].default == ("train",)


def test_the_cli_defaults_to_train_only():
    src = inspect.getsource(bt)
    assert "'--train_splits'" in src or '"--train_splits"' in src
    assert "default='train'" in src or 'default="train"' in src


def test_val_and_test_captures_are_excluded_by_the_default():
    """Every capture the lock reserves must be rejected by the train filter."""
    lock = load_lock()
    held_out = [
        (ds, cap)
        for ds, m in lock.items() if ds != "PCAP2018"
        for cap, sp in m.items() if sp in ("val", "test")
    ]
    assert held_out, "the lock reserves nothing -- that is itself wrong"
    for _ds, cap in held_out:
        assert split_of_path("/anywhere/" + cap) != "train"


def test_train_captures_are_admitted():
    lock = load_lock()
    train_caps = [
        cap for ds, m in lock.items() if ds != "PCAP2018"
        for cap, sp in m.items() if sp == "train"
    ]
    # 23 = CIC2017 6 + CIC2018 8 + CTU13 9. (17 is the CIC2018+CTU13 subset that
    # Branch A alone reads; the encoder reads CIC-2017 as well.)
    assert len(train_caps) == 23
    for cap in train_caps:
        assert split_of_path("/anywhere/" + cap) == "train"


def test_the_split_partition_is_exhaustive_for_the_encoder():
    """Nothing may fall between the filters -- a capture that matches no split
    would be dropped from training without appearing in any held-out set."""
    lock = load_lock()
    for ds, m in lock.items():
        if ds == "PCAP2018":
            continue
        for cap, sp in m.items():
            assert sp in ("train", "val", "test"), f"{ds}/{cap} has split {sp!r}"
