"""Ablation of named NODE features -- the counterpart to the edge-feature one.

It exists because the IP-feature ablation found the leak is not where it was
assumed to be. The suspicion was the octets, since an octet is an identifier
and on this corpus one host carries a whole attack class. Measured on a
fixture reproducing CIC-2018's topology -- attacks staged from public AWS
addresses against RFC1918 victims -- the address recovers the label at ROC-AUC
1.0000, permutation importance attributes it to `is_private` (+0.496), and
masking all four octets changes nothing.

One boolean separates attacker traffic from benign internal traffic perfectly.
That is a property of how the capture was staged, and a model keying on it
scores well offline and transfers nothing to a live network where every host
is internal.

Zeroing rather than dropping keeps the 12-D node contract, so an ablated
encoder loads everywhere a normal one does.
"""

import numpy as np
import pytest

from data_unification.ip_features import (
    ABLATION_GROUPS,
    IP_FEATURE_DIM,
    IP_FEATURE_NAMES,
    build_node_feature_matrix,
    ip_node_features,
    node_ablation_mask,
    reset_node_ablation_cache,
)

VICTIM, ATTACKER = "172.31.69.25", "18.221.219.4"
IPS = {VICTIM: 1, ATTACKER: 2}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CYBERWORLD_ABLATE_NODE_FEATURES", raising=False)
    reset_node_ablation_cache()
    yield
    reset_node_ablation_cache()


def _matrix():
    return build_node_feature_matrix(IPS, n_nodes=3)


def _col(name):
    return IP_FEATURE_NAMES.index(name)


def test_no_ablation_by_default():
    assert node_ablation_mask() is None
    m = _matrix()
    assert m[1][_col("is_private")] == 1.0
    assert m[2][_col("is_global")] == 1.0


def test_the_leak_this_exists_for_is_real():
    """Establishes the premise: one boolean separates the two hosts."""
    m = _matrix()
    assert m[1][_col("is_private")] != m[2][_col("is_private")]
    assert m[1][_col("is_global")] != m[2][_col("is_global")]


def test_ablating_the_address_class_removes_exactly_that_signal(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "is_private,is_global")
    reset_node_ablation_cache()
    m = _matrix()
    assert m[1][_col("is_private")] == 0.0 and m[2][_col("is_private")] == 0.0
    assert m[1][_col("is_global")] == 0.0 and m[2][_col("is_global")] == 0.0
    # everything else is untouched
    for n in ("octet1", "octet2", "octet3", "octet4", "is_ipv4", "bias"):
        assert m[1][_col(n)] == pytest.approx(ip_node_features(VICTIM)[_col(n)])


def test_the_group_shorthand_matches_the_explicit_list(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "address_class")
    reset_node_ablation_cache()
    grouped = _matrix().copy()
    reset_node_ablation_cache()
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "is_private,is_global")
    reset_node_ablation_cache()
    assert np.array_equal(grouped, _matrix())


@pytest.mark.parametrize("group", sorted(ABLATION_GROUPS))
def test_every_group_names_real_features(group):
    for n in ABLATION_GROUPS[group]:
        assert n in IP_FEATURE_NAMES, f"{group} names {n}, which is not a feature"


def test_ablating_the_octets_leaves_the_class_flags(monkeypatch):
    """The arm that shows the octets are NOT the carrier on this corpus."""
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "octets")
    reset_node_ablation_cache()
    m = _matrix()
    for n in ("octet1", "octet2", "octet3", "octet4"):
        assert m[1][_col(n)] == 0.0 and m[2][_col(n)] == 0.0
    assert m[1][_col("is_private")] == 1.0, "the class flag should survive"
    assert m[2][_col("is_global")] == 1.0


def test_the_vector_stays_12d_so_encoders_still_load(monkeypatch):
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "address_class,octets")
    reset_node_ablation_cache()
    m = _matrix()
    assert m.shape == (3, IP_FEATURE_DIM)


def test_an_unknown_name_fails_loudly(monkeypatch):
    """A typo must not silently ablate nothing and produce a 'no effect'
    result that reads as 'the feature does not matter'."""
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "is_privte")
    reset_node_ablation_cache()
    with pytest.raises(ValueError, match="unknown node feature"):
        _matrix()


def test_the_bias_cannot_be_ablated(monkeypatch):
    """An all-zero row is how the encoder recognises an unknown node, so
    zeroing the bias changes the meaning of every other row."""
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "bias")
    reset_node_ablation_cache()
    with pytest.raises(ValueError, match="refusing to ablate 'bias'"):
        _matrix()


def test_the_per_ip_cache_is_not_poisoned(monkeypatch):
    """ip_node_features is lru_cached; an ablated value written into that cache
    would survive a reset and silently affect an un-ablated run."""
    clean = tuple(ip_node_features(VICTIM))
    monkeypatch.setenv("CYBERWORLD_ABLATE_NODE_FEATURES", "address_class")
    reset_node_ablation_cache()
    _matrix()
    monkeypatch.delenv("CYBERWORLD_ABLATE_NODE_FEATURES")
    reset_node_ablation_cache()
    assert tuple(ip_node_features(VICTIM)) == clean
    assert _matrix()[1][_col("is_private")] == 1.0


def test_it_mirrors_the_edge_ablation_interface():
    """Two knobs for the same kind of experiment should not behave differently."""
    from data_unification import tgne_features as tf

    assert callable(tf.reset_ablation_cache) and callable(reset_node_ablation_cache)
    assert callable(tf.ablation_mask) and callable(node_ablation_mask)
