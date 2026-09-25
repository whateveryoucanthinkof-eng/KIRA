"""The attack label must come from the corpus, not from the pipeline's own heuristics.

`HostTrajectoryExtractor` used to overwrite `is_attack` in two places:

    auth:        auth_failed_count >= 5     -> CredentialAccess / T1110
    behavioural: port_mismatch_count >= 2   -> Execution / T1059

The second one is the damaging one, and it is not a corner case.
`BehavioralFlowFingerprinter` calls a flow "C2 beaconing" when it carries
<= 1200 bytes in 1-3 packets each way with mean packet size < 180, and calls
that a "port mismatch" whenever the destination port is not one of
{80, 443, 8000, 8080, 8443}. Two small UDP service exchanges lasting longer
than half a second -- mDNS, NetBIOS name service, NTP, or a DNS lookup that
retries -- satisfy both, so an entirely benign host was written down as under
attack. That traffic is on every LAN continuously. (Verified against the
fingerprinter directly; a sub-0.5s single-exchange DNS lookup takes the
BurstRecon branch instead and does not trip it.)

Downstream that produced a 0.677 measured attack rate and a 0.9997 test-split
base rate (results/v4_benchmark.json) on corpora whose published attack
fractions run from 0.03% to 57%. It also made the target a function of the same
flow statistics handed to the model as input, so accuracy measured the model's
ability to re-derive its own features.

These tests pin the label to ground truth and keep the heuristics available
only behind an explicit opt-in.
"""

import pytest
import torch

from data_unification.multi_dataset_stream import (
    ATTACK_ROLES,
    HostTrajectoryExtractor,
    _window_attack_label,
)
from data_unification.unified_schema import LabelSource, UnifiedFlowRecord

WINDOW = 2.0
EMB_DIM = 12


class _StubTGN:
    """Deterministic embeddings. No `embedding_module`, so extract_trajectories
    skips the graph rebuild that is not under test here."""

    device = "cpu"

    def get_host_embeddings(self, node_ids, timestamp, n_neighbors=10):
        return torch.zeros((len(node_ids), EMB_DIM), dtype=torch.float32)


def _rec(src, dst, *, dport, proto=17, fb=70, bb=120, fp=1, bp=1, t=0.0,
         is_attack=False, coarse="Benign", techs=(), label="BENIGN"):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=41234, dst_port=dport, protocol=proto,
        start_time=t, end_time=t + 0.01,
        fwd_bytes=fb, bwd_bytes=bb, fwd_packets=fp, bwd_packets=bp,
        raw_label=label, raw_label_source=LabelSource.CIC2018.value,
        is_attack=is_attack, coarse_category=coarse,
        attck_technique_ids=list(techs),
    )


def _run(records, **kw):
    ex = HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), window_size_sec=WINDOW, **kw)
    return ex.extract_trajectories(records)


def _udp_chatter(t):
    """One benign small UDP service exchange, of the shape the fingerprinter
    scores as C2Beaconing + port mismatch: <=1200 bytes, 2 packets each way,
    mean packet size < 180, on a non-web port, lasting > 0.5s."""
    r = _rec("10.0.0.5", "10.0.0.53", dport=5353, proto=17,
             fb=160, bb=200, fp=2, bp=2, t=t)
    r.end_time = t + 0.8
    return r


#: Two mDNS exchanges in one window -- ubiquitous benign LAN chatter, and
#: exactly what used to trip `port_mismatch_count >= 2`.
BENIGN_DNS = [_udp_chatter(0.0), _udp_chatter(0.9)]


def test_benign_udp_chatter_is_not_an_attack():
    store = _run(BENIGN_DNS)
    snaps = list(store["10.0.0.5"])
    assert snaps, "host produced no snapshot"
    assert not any(s.is_attack for s in snaps), (
        "a benign host doing ordinary mDNS chatter was labelled under attack"
    )
    assert all(s.coarse_category == "Benign" for s in snaps)
    assert all(s.technique_ids == [] for s in snaps)
    assert all(s.risk_score == 0.0 for s in snaps)


def test_the_heuristic_is_still_available_behind_the_opt_in():
    """Removing the default must not remove the capability."""
    store = _run(BENIGN_DNS, heuristic_label_augmentation=True)
    snaps = list(store["10.0.0.5"])
    assert any(s.is_attack for s in snaps), (
        "heuristic_label_augmentation=True no longer reproduces the old behaviour"
    )
    assert any("T1059" in s.technique_ids for s in snaps)


def test_a_real_corpus_attack_is_still_labelled():
    """Turning the heuristics off must not suppress genuine labels."""
    recs = BENIGN_DNS + [
        _rec("10.0.0.9", "10.0.0.5", dport=80, proto=6, t=1.0,
             is_attack=True, coarse="Impact", techs=("T1498",), label="DDoS"),
    ]
    store = _run(recs)
    assert any(s.is_attack for s in store["10.0.0.5"])
    assert any(s.coarse_category == "Impact" for s in store["10.0.0.5"])


# ---------------------------------------------------------------------------
# Technique labels are the union, not whichever record sorted first
# ---------------------------------------------------------------------------


def test_window_label_takes_the_union_of_techniques():
    recs = [
        _rec("a", "b", dport=80, is_attack=True, coarse="Recon", techs=("T1046",)),
        _rec("a", "b", dport=80, is_attack=True, coarse="Impact", techs=("T1498", "T1046")),
    ]
    coarse, techs = _window_attack_label(recs)
    assert set(techs) == {"T1046", "T1498"}, "techniques beyond the first record were discarded"
    assert techs == ["T1046", "T1498"], "union must be ordered by first appearance, for determinism"


def test_window_label_takes_the_most_severe_category():
    recon_first = [
        _rec("a", "b", dport=80, is_attack=True, coarse="Recon", techs=("T1046",)),
        _rec("a", "b", dport=80, is_attack=True, coarse="Exfiltration", techs=("T1005",)),
    ]
    assert _window_attack_label(recon_first)[0] == "Exfiltration"
    # and the answer does not depend on record order
    assert _window_attack_label(list(reversed(recon_first)))[0] == "Exfiltration"


def test_multilabel_survives_into_the_snapshot():
    recs = [
        _rec("10.1.1.1", "10.1.1.2", dport=80, proto=6, t=0.1, is_attack=True,
             coarse="Recon", techs=("T1046",), label="PortScan"),
        _rec("10.1.1.1", "10.1.1.3", dport=443, proto=6, t=0.2, is_attack=True,
             coarse="C2", techs=("T1071",), label="Bot"),
    ]
    snaps = list(_run(recs)["10.1.1.1"])
    hit = [s for s in snaps if s.is_attack]
    assert hit, "no attack snapshot produced"
    assert set(hit[0].technique_ids) == {"T1046", "T1071"}, (
        "downstream multilabel targets can only be as multilabel as the snapshot"
    )
    assert hit[0].coarse_category == "C2", "most severe category should win"


# ---------------------------------------------------------------------------
# attack_role
# ---------------------------------------------------------------------------


ATTACKER, VICTIM, BYSTANDER = "203.0.113.7", "172.31.69.25", "172.31.69.90"

ONE_ATTACK = [
    _rec(ATTACKER, VICTIM, dport=80, proto=6, t=0.1, is_attack=True,
         coarse="Impact", techs=("T1498",), label="DDoS"),
    _rec(BYSTANDER, "10.0.0.53", dport=53, t=0.2),
]


@pytest.mark.parametrize("role", ATTACK_ROLES)
def test_attack_role_is_accepted(role):
    _run(ONE_ATTACK, attack_role=role)


def test_attack_role_target_labels_only_the_victim():
    store = _run(ONE_ATTACK, attack_role="target")
    assert any(s.is_attack for s in store[VICTIM])
    assert not any(s.is_attack for s in store[ATTACKER])


def test_attack_role_source_labels_only_the_attacker():
    store = _run(ONE_ATTACK, attack_role="source")
    assert any(s.is_attack for s in store[ATTACKER])
    assert not any(s.is_attack for s in store[VICTIM])


def test_attack_role_either_is_the_documented_default():
    store = _run(ONE_ATTACK)
    assert any(s.is_attack for s in store[ATTACKER])
    assert any(s.is_attack for s in store[VICTIM])


def test_a_bystander_is_never_labelled_in_any_role():
    for role in ATTACK_ROLES:
        store = _run(ONE_ATTACK, attack_role=role)
        assert not any(s.is_attack for s in store[BYSTANDER]), role


def test_an_unknown_role_is_refused_at_construction():
    with pytest.raises(ValueError, match="attack_role"):
        HostTrajectoryExtractor(tgne_ta_model=_StubTGN(), attack_role="victim")
