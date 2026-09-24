"""The IP ablation must detect address-based leakage, name its carrier, and
stay silent when there is none.

An instrument that reports leakage is only useful if it has been shown not to
report it on clean data, and only actionable if it says WHICH address feature
carries the signal -- `is_private` separating an external attacker from internal
victims is a property of how a capture was staged, while `octet4` picking out
one machine is memorisation, and the two want opposite responses.
"""

import numpy as np
import pytest

from scripts.ablate_ip_features import (
    HOST_IDENTITY_IDX,
    OCTET_IDX,
    arm_ip_only,
    concentration_report,
    ip_matrix,
    permute_addresses,
)

N = 6000
SEED = 0


def _corpus(kind, seed=SEED):
    rng = np.random.default_rng(seed)
    ips, lab = [], []
    if kind == "clean":
        hosts = [f"172.31.{rng.integers(64, 80)}.{rng.integers(1, 254)}" for _ in range(200)]
        for _ in range(N):
            ips.append(hosts[rng.integers(0, len(hosts))])
            lab.append(int(rng.integers(0, 2)))
    elif kind == "external":
        # The real CIC-2018 topology: public attacker, RFC1918 victims.
        ben = [f"172.31.69.{i}" for i in range(10, 200)]
        atk = ["18.221.219.4"] + [f"18.219.211.{i}" for i in range(1, 10)]
        for _ in range(N):
            if rng.random() < 0.25:
                ips.append(atk[rng.integers(0, len(atk))]); lab.append(1)
            else:
                ips.append(ben[rng.integers(0, len(ben))]); lab.append(0)
    else:  # internal attacker, same /24 as the victims -> only octets separate
        ben = [f"172.31.69.{i}" for i in range(10, 200)]
        for _ in range(N):
            if rng.random() < 0.25:
                ips.append("172.31.69.25"); lab.append(1)
            else:
                ips.append(ben[rng.integers(0, len(ben))]); lab.append(0)
    return ips, lab, ["cap0"] * N


def _auc(res):
    return res["transductive"]["roc_auc"]


def _carrier(res):
    return res["transductive"]["top_features"][0][0]


# ---------------------------------------------------------------------------
# No false positives
# ---------------------------------------------------------------------------


def test_clean_data_reports_chance():
    ips, lab, grp = _corpus("clean")
    r = arm_ip_only(ips, lab, grp, seed=SEED)
    assert _auc(r) == pytest.approx(0.5, abs=0.05), (
        "the harness claims the address predicts a label that is independent of it"
    )
    assert r["inductive"]["roc_auc"] == pytest.approx(0.5, abs=0.05)


def test_clean_data_attributes_nothing():
    ips, lab, grp = _corpus("clean")
    imp = arm_ip_only(ips, lab, grp, seed=SEED)["transductive"]["permutation_importance"]
    assert max(imp.values()) < 0.05, f"spurious attribution on clean data: {imp}"


# ---------------------------------------------------------------------------
# It detects leakage and names the carrier
# ---------------------------------------------------------------------------


def test_an_octet_carried_leak_is_detected_and_named():
    ips, lab, grp = _corpus("internal")
    r = arm_ip_only(ips, lab, grp, seed=SEED)
    assert _auc(r) > 0.9
    assert _carrier(r) in ("octet4", "octet3"), (
        f"leak is carried by the host octets but attributed to {_carrier(r)}"
    )


def test_masking_the_identity_octets_removes_an_octet_carried_leak():
    ips, lab, grp = _corpus("internal")
    full = arm_ip_only(ips, lab, grp, seed=SEED)
    masked = arm_ip_only(ips, lab, grp, seed=SEED, mask=HOST_IDENTITY_IDX)
    assert _auc(full) > 0.9 and _auc(masked) == pytest.approx(0.5, abs=0.02), (
        "masking octet3/octet4 should collapse a leak that only they can carry"
    )


def test_an_address_class_leak_is_attributed_to_the_class_flag_not_the_octets():
    """The finding that matters for this corpus: with a public attacker and
    private victims, `is_private` alone separates them, and masking every octet
    changes nothing."""
    ips, lab, grp = _corpus("external")
    full = arm_ip_only(ips, lab, grp, seed=SEED)
    no_octets = arm_ip_only(ips, lab, grp, seed=SEED, mask=OCTET_IDX)
    assert _auc(full) > 0.9
    assert _carrier(full) in ("is_private", "is_global")
    assert _auc(no_octets) > 0.9, (
        "masking the octets removed a leak the octets were not carrying"
    )


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_masking_actually_zeroes_the_named_columns():
    X = ip_matrix(["172.31.69.25"], mask=HOST_IDENTITY_IDX)
    assert X[0, HOST_IDENTITY_IDX[0]] == 0.0 and X[0, HOST_IDENTITY_IDX[1]] == 0.0
    assert X[0, 2] != 0.0, "masking must not disturb the address-class flags"
    assert ip_matrix(["172.31.69.25"])[0, HOST_IDENTITY_IDX[0]] != 0.0


def test_a_single_class_split_says_so_instead_of_inventing_an_auc():
    ips = ["10.0.0.1"] * 50
    r = arm_ip_only(ips, [0] * 50, ["cap0"] * 50, seed=SEED)
    assert np.isnan(r["transductive"]["roc_auc"])
    assert "note" in r["transductive"]


def test_concentration_report_finds_a_single_carrier_host():
    """The mechanism behind any leak: a class carried by one host is perfectly
    predictable from that host's address."""
    ips = ["172.16.0.1"] * 900 + [f"10.0.0.{i}" for i in range(100)]
    cats = ["Recon"] * 900 + ["Benign"] * 100
    rep = concentration_report(ips, cats)
    assert rep["Recon"]["distinct_hosts"] == 1
    assert rep["Recon"]["top_host"] == "172.16.0.1"
    assert rep["Recon"]["hosts_covering_90pct"] == 1
    assert rep["Benign"]["distinct_hosts"] == 100


def test_permutation_is_a_bijection():
    ips = [f"172.31.69.{i}" for i in range(10, 60)]
    m = permute_addresses(ips, seed=SEED)
    assert set(m) == set(ips) and set(m.values()) == set(ips)
    assert any(k != v for k, v in m.items()), "permutation left every address in place"


def test_subnet_preserving_permutation_keeps_hosts_inside_their_own_24():
    ips = [f"172.31.69.{i}" for i in range(10, 40)] + [f"10.2.3.{i}" for i in range(1, 30)]
    m = permute_addresses(ips, seed=SEED, preserve_subnet=True)
    for src, dst in m.items():
        assert src.rsplit(".", 1)[0] == dst.rsplit(".", 1)[0], (
            "a subnet-preserving permutation moved a host to another subnet"
        )
    assert any(k != v for k, v in m.items())
