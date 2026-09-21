"""Intrinsic IP node features -- the fix for chance-level inductive AUC.

`bita/train.py` built node_features as all zeros. Combined with TGN's zero
memory for unseen nodes, an inductive host carried no signal at all:
transductive val AUC 0.9981 against inductive val AUC 0.5043.

These tests pin the two properties that make the fix work: the features are
informative (same-subnet hosts are close, different networks are far), and
they are leak-free (a pure function of the address, so no dataset statistics
and no temporal information enter).
"""

import numpy as np
import pytest

from data_unification.ip_features import (
    IP_FEATURE_DIM,
    IP_FEATURE_NAMES,
    build_node_feature_matrix,
    ip_node_features,
    subnet_cohesion,
)


def v(ip):
    return np.asarray(ip_node_features(ip), dtype=np.float32)


def test_width_is_pinned_to_the_latent_contract():
    """tgn.py sets embedding_dimension = n_node_features, so 12 is not optional."""
    assert IP_FEATURE_DIM == 12 == len(IP_FEATURE_NAMES)
    assert v("10.0.0.1").shape == (12,)


def test_no_feature_is_all_zero_for_a_real_address():
    """An all-zero row is reserved to mean 'unknown'; a real IP must not look like one."""
    assert v("172.31.69.25").any()


def test_unparseable_input_is_all_zero_but_for_the_bias():
    for bad in ("nan", "", "   ", "not-an-ip", "999.999.999.999"):
        f = v(bad)
        assert f[-1] == 1.0, "bias must stay set"
        assert not f[:-1].any(), f"{bad!r} produced signal it should not"


# ------------------------------------------------- the inductive property

def test_same_subnet_hosts_are_close_in_feature_space():
    """This is the whole point: an unseen host in a known /24 must land near it."""
    a, b = v("172.31.69.25"), v("172.31.69.30")
    far = v("8.8.8.8")
    assert np.linalg.norm(a - b) < np.linalg.norm(a - far)


def test_the_first_three_octets_are_shared_within_a_24():
    a, b = v("192.168.10.5"), v("192.168.10.200")
    assert np.array_equal(a[7:10], b[7:10]), "/24 identity must be shared"
    assert a[10] != b[10], "host identity must still differ"


def test_subnet_cohesion_grades_prefix_overlap():
    assert subnet_cohesion("172.31.69.25", "172.31.69.30") == 0.75   # same /24
    assert subnet_cohesion("172.31.69.25", "172.31.70.30") == 0.5    # same /16
    assert subnet_cohesion("172.31.69.25", "172.30.70.30") == 0.25   # same /8
    assert subnet_cohesion("172.31.69.25", "8.8.8.8") == 0.0
    assert subnet_cohesion("172.31.69.25", "172.31.69.25") == 1.0


def test_the_real_corpus_networks_separate():
    """CIC-2018 is 172.31.x, CIC-2017 is 192.168.10.x, external is public."""
    c18, c17, ext = v("172.31.69.25"), v("192.168.10.50"), v("8.8.8.8")
    assert c18[2] == c17[2] == 1.0, "both corpora are private space"
    assert ext[2] == 0.0 and ext[3] == 1.0, "external must read as public"
    assert np.linalg.norm(c18 - c17) > 0.1, "the two corpora must not collapse together"


# ------------------------------------------------------- leakage safety

def test_features_depend_only_on_the_address():
    """No dataset statistics, no ordering, no time -- so nothing can leak."""
    assert ip_node_features("10.1.2.3") == ip_node_features("10.1.2.3")
    assert ip_node_features(" 10.1.2.3 ") == ip_node_features("10.1.2.3")


def test_classification_flags_are_mutually_sensible():
    priv, pub, mcast, loop = v("10.0.0.1"), v("1.1.1.1"), v("224.0.0.1"), v("127.0.0.1")
    assert priv[2] == 1.0 and priv[3] == 0.0
    assert pub[3] == 1.0 and pub[2] == 0.0
    assert mcast[4] == 1.0
    assert loop[5] == 1.0


def test_ipv6_is_handled_without_crashing():
    f = v("2001:db8::1")
    assert f[0] == 1.0 and f[1] == 0.0


# ------------------------------------------------------------- the matrix

def test_matrix_reserves_row_zero_for_padding():
    m = build_node_feature_matrix({"10.0.0.1": 1, "10.0.0.2": 2})
    assert m.shape == (3, 12)
    assert not m[0].any(), "node id 0 is padding and must stay zero"
    assert m[1].any() and m[2].any()


def test_matrix_rows_match_the_per_ip_function():
    ips = {"172.31.69.25": 1, "8.8.8.8": 2, "192.168.10.50": 3}
    m = build_node_feature_matrix(ips)
    for ip, i in ips.items():
        np.testing.assert_allclose(m[i], v(ip), rtol=0, atol=0)


def test_matrix_is_not_degenerate_across_many_hosts():
    """A matrix whose rows are all identical is as useless as all zeros."""
    ips = {f"172.31.{i // 256}.{i % 256}": i + 1 for i in range(500)}
    m = build_node_feature_matrix(ips)[1:]
    assert len(np.unique(m, axis=0)) == len(m), "every host must be distinguishable"
    assert m.std(axis=0).sum() > 0.0
