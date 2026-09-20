"""v4 contract, identity and target-construction guarantees.

These tests encode the scientific properties, not just the code paths. If one
fails, a claim the project makes has stopped being true.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import List

from cyberworld_v4.config import DEFAULT_CONFIG, CyberWorldConfig, get_contract
from cyberworld_v4.contract import ContractViolation, validate, validate_checkpoint
from cyberworld_v4.identity import HostId, offline_host, live_host, stable_id
from cyberworld_v4.targets import (
    NO_ONSET, build_samples, cumulative_from_hazard, split_on_gaps,
)


@dataclass
class Snap:
    window_end: float
    embedding: np.ndarray
    temporal_attrs: np.ndarray
    is_attack: bool
    technique_ids: List[str]
    coarse_category: str = "X"
    risk_score: float = 0.0


def make_snaps(n, attack_from=None, step=2.0, t0=0.0):
    return [
        Snap(
            t0 + i * step,
            np.zeros(12, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            attack_from is not None and i >= attack_from,
            ["T1046"] if (attack_from is not None and i >= attack_from) else [],
        )
        for i in range(n)
    ]


# --- contract --------------------------------------------------------------

def test_contract_is_the_v4_values():
    c = get_contract()
    assert (c.window_seconds, c.history_steps, c.forecast_steps) == (2.0, 15, 5)
    assert c.history_seconds == 30.0 and c.forecast_seconds == 10.0


def test_contract_refuses_mismatch():
    with pytest.raises(ContractViolation):
        validate({"window_seconds": 2.0, "history_steps": 5, "forecast_steps": 8})
    with pytest.raises(ContractViolation):
        validate({"window_seconds": 60.0, "history_steps": 15, "forecast_steps": 5})


def test_contract_accepts_exact_match():
    validate(get_contract().to_dict())


def test_v3_checkpoint_shape_is_refused():
    """A v3 checkpoint must not load silently under v4."""
    with pytest.raises(ContractViolation):
        validate_checkpoint({"window_seconds": 2.0, "history_steps": 5, "forecast_steps": 8})


def test_config_roundtrip():
    d = DEFAULT_CONFIG.to_dict()
    assert CyberWorldConfig.from_dict(d).temporal == DEFAULT_CONFIG.temporal


# --- identity --------------------------------------------------------------

def test_same_ip_in_two_datasets_is_two_hosts():
    a = offline_host("CIC2018", "wed14", "cap1", "192.168.1.10")
    b = offline_host("CTU13", "scen3", "cap9", "192.168.1.10")
    assert a != b and a.key != b.key and a.node_index() != b.node_index()


def test_ids_are_stable_across_processes():
    """Python hash() is salted per run; SHA-256 is not."""
    import subprocess, sys
    code = (
        "from cyberworld_v4.identity import stable_id;"
        "print(stable_id('scenario-3','src',modulo=250))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONHASHSEED": s, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for s in ("0", "1", "12345")
    }
    assert len(outs) == 1, f"stable_id varied with PYTHONHASHSEED: {outs}"


def test_live_host_namespace():
    assert live_host("10.0.2.30").is_live
    assert not offline_host("CIC2018", "s", "c", "10.0.2.30").is_live


# --- targets ---------------------------------------------------------------

def test_no_target_comes_from_the_input_window():
    """The v3 defect: target_snap = window_slice[-1]."""
    snaps = make_snaps(40, attack_from=25)
    ss = build_samples(snaps, offline_host("d", "s", "c", "h"), ["Benign", "T1046"])
    L, K = DEFAULT_CONFIG.temporal.history_steps, DEFAULT_CONFIG.temporal.forecast_steps
    for s in ss:
        assert s.features.shape == (L, DEFAULT_CONFIG.state_dim)
        assert s.future_attack.shape == (K,)
        # the last input state is at index t_index; targets start at t_index+1
        assert s.t_end == snaps[s.t_index].window_end


def test_future_targets_are_actually_future():
    snaps = make_snaps(40, attack_from=25)
    ss = build_samples(snaps, offline_host("d", "s", "c", "h"), ["Benign", "T1046"])
    for s in ss:
        expected = [int(snaps[s.t_index + 1 + k].is_attack) for k in range(len(s.future_attack))]
        assert list(s.future_attack) == expected


def test_hazard_marks_onset_once_and_censors_ongoing():
    snaps = make_snaps(40, attack_from=25)
    ss = build_samples(snaps, offline_host("d", "s", "c", "h"), ["Benign", "T1046"])
    onsets = [s for s in ss if s.onset_step != NO_ONSET]
    assert onsets, "no onset detected in a trajectory that contains one"
    for s in onsets:
        assert s.hazard_target.sum() == 1.0          # exactly one event step
        assert not s.onset_censored
    for s in [x for x in ss if x.onset_censored]:
        # already under attack: contributes nothing to the hazard likelihood
        assert s.at_risk.sum() == 0.0
        assert s.onset_step == NO_ONSET


def test_cumulative_is_product_not_max():
    h = np.array([0.1, 0.2, 0.0, 0.0, 0.0])
    cum = cumulative_from_hazard(h)
    assert np.isclose(cum[1], 1 - 0.9 * 0.8)
    assert not np.isclose(cum[-1], h.max())
    assert np.all(np.diff(cum) >= -1e-6)             # monotone non-decreasing


def test_techniques_are_multilabel():
    """v3 kept technique_ids[0] and discarded the rest."""
    snaps = make_snaps(30, attack_from=10)
    for s in snaps[10:]:
        s.technique_ids = ["T1046", "T1071"]
    ss = build_samples(snaps, offline_host("d", "s", "c", "h"), ["Benign", "T1046", "T1071"])
    multi = [s for s in ss if s.future_techniques.sum() > 0]
    assert multi, "no technique targets produced"
    assert max(s.future_techniques.sum(axis=1).max() for s in multi) >= 2


def test_gaps_split_trajectories():
    """A 102-minute gap is not a 2-second transition."""
    a = make_snaps(20, t0=0.0)
    b = make_snaps(20, t0=6000.0)
    segs = split_on_gaps(a + b, DEFAULT_CONFIG.max_gap_seconds)
    assert len(segs) == 2 and len(segs[0]) == 20 and len(segs[1]) == 20


def test_short_trajectory_yields_nothing():
    """Fewer than history+horizon states cannot form a sample."""
    assert build_samples(make_snaps(10), offline_host("d", "s", "c", "h"), ["Benign"]) == []
