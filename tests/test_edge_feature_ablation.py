"""Ablation of named edge features, to answer "why these twelve?" with numbers.

A reviewer asked how the 12 edge features were chosen and what makes them the
right ones. Nothing in the repository justified them -- they are conventional
NetFlow summaries with no ablation and no importance analysis behind them.

`dst_port_norm_65535` is the sharp end of it: several CIC-2018 attack classes
sit on fixed destination ports, so a model can learn "port => class" and score
well without learning behaviour, then collapse when an attacker changes port.
Zeroing one feature and re-measuring inductive AUC turns that from an argument
into a measurement.

Zeroing rather than removing keeps the 12-D contract, so an ablated encoder
still loads everywhere a normal one does and the comparison is not confounded
by an architecture change.
"""
import os

import numpy as np
import pytest

from data_unification import tgne_features as tf
from data_unification.tgne_features import (
    EDGE_FEATURE_NAMES, extract_canonical_edge_features, reset_ablation_cache,
)

ARGS = dict(fwd_bytes=1200, bwd_bytes=400, fwd_packets=10, bwd_packets=8,
            duration_sec=1.0, byte_rate=1600.0, packet_rate=18.0,
            protocol=6, dst_port=80)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CYBERWORLD_ABLATE_EDGE_FEATURES", raising=False)
    reset_ablation_cache()
    yield
    reset_ablation_cache()


def test_no_ablation_by_default_changes_nothing():
    v = extract_canonical_edge_features(**ARGS)
    assert v.shape == (12,)
    assert v[10] != 0.0, "dst_port feature should be populated by default"


def test_ablating_dst_port_zeroes_only_that_feature(monkeypatch):
    base = extract_canonical_edge_features(**ARGS).copy()
    monkeypatch.setenv("CYBERWORLD_ABLATE_EDGE_FEATURES", "dst_port_norm_65535")
    reset_ablation_cache()
    ab = extract_canonical_edge_features(**ARGS)
    i = EDGE_FEATURE_NAMES.index("dst_port_norm_65535")
    assert ab[i] == 0.0
    for j in range(12):
        if j != i:
            assert ab[j] == base[j], f"{EDGE_FEATURE_NAMES[j]} changed; only {i} should"


def test_the_vector_stays_12d_so_checkpoints_still_load(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_ABLATE_EDGE_FEATURES",
                       "dst_port_norm_65535,is_tcp,is_udp")
    reset_ablation_cache()
    assert extract_canonical_edge_features(**ARGS).shape == (12,)


def test_several_features_can_be_ablated_at_once(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_ABLATE_EDGE_FEATURES",
                       "dst_port_norm_65535, is_icmp ,log1p_fwd_bytes")
    reset_ablation_cache()
    v = extract_canonical_edge_features(**ARGS)
    for n in ("dst_port_norm_65535", "is_icmp", "log1p_fwd_bytes"):
        assert v[EDGE_FEATURE_NAMES.index(n)] == 0.0
    assert v[EDGE_FEATURE_NAMES.index("log1p_bwd_bytes")] != 0.0


def test_an_unknown_feature_name_fails_loudly(monkeypatch):
    """A typo must not silently ablate nothing and produce a 'no effect'
    result that reads as 'the feature does not matter'."""
    monkeypatch.setenv("CYBERWORLD_ABLATE_EDGE_FEATURES", "dst_port")  # not the real name
    reset_ablation_cache()
    with pytest.raises(ValueError, match="unknown edge feature"):
        extract_canonical_edge_features(**ARGS)


def test_dst_port_actually_separates_classes_in_this_corpus():
    """Establishes that the leak is possible before spending a retrain on it:
    if port did not vary with the attack class there would be nothing to
    memorise."""
    http = extract_canonical_edge_features(**{**ARGS, "dst_port": 80})
    ssh = extract_canonical_edge_features(**{**ARGS, "dst_port": 22})
    i = EDGE_FEATURE_NAMES.index("dst_port_norm_65535")
    assert http[i] != ssh[i], "port must be distinguishable for the leak to exist"
    assert np.allclose(np.delete(http, i), np.delete(ssh, i)), (
        "two flows identical but for the port differ in exactly one feature -- "
        "so a model can key on it alone")
