"""
tests/test_serving_path_defects.py

Regression pins for semantic defects found in the LIVE SERVING PATH -- the code
between a captured flow window and the number an operator reads. Each test
names the wrong answer it prevents, because none of these were type errors,
shape errors or lint findings: every one of them ran cleanly and returned
something plausible.

Scope: control_backend/, correlation/, explainability/, telemetry/.
"""

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "bita"))

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from branch_a_gnn_lstm.sequence_dataset import create_host_sequence_samples
from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from correlation.trajectory_assembler import (
    AttackTrajectoryAssembler,
    HostAttackTrajectory,
    Provenance,
    TrajectoryEntry,
)
from cyberworld_v4.config import get_contract
from data_unification.multi_dataset_stream import HostWindowSnapshot
from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
from deepop_decoder.joint_vocab import get_joint_vocab
from explainability.unified_explanation import FEATURE_NAMES, UnifiedExplainer

CONTRACT = get_contract()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _snap(i, host="10.0.0.5"):
    """A snapshot whose embedding/attrs encode its index, so padding is visible."""
    return HostWindowSnapshot(
        host_ip=host,
        host_id=1,
        window_idx=i,
        window_start=1000.0 + CONTRACT.window_seconds * i,
        window_end=1000.0 + CONTRACT.window_seconds * (i + 1),
        embedding=np.full(12, 0.1 * (i + 1), dtype=np.float32),
        temporal_attrs=np.full(15, 0.05 * (i + 1), dtype=np.float32),
        is_attack=False,
        coarse_category="Benign",
        technique_ids=[],
        risk_score=0.0,
    )


def _entry(i, risk=0.5):
    return TrajectoryEntry(
        host_ip="10.0.0.5",
        window_idx=i,
        timestamp=1000.0 + CONTRACT.window_seconds * i,
        provenance=Provenance.OBSERVED.value,
        coarse_category="C2",
        technique_id="T1071",
        confidence=0.9,
        risk_score=risk,
    )


@pytest.fixture(scope="module")
def assembler():
    torch.manual_seed(0)
    vocab = get_joint_vocab()
    return AttackTrajectoryAssembler(
        MultiTaskLSTM(input_dim=27, hidden_dim=64).eval(),
        HostWorldDynamicsTransformer(d_latent=12, d_model=64).eval(),
        InfiltrationRiskHead(d_latent=12, hidden_dim=32).eval(),
        DeepOPForecastDecoder(d_latent=12, d_model=72, vocab_size=vocab.vocab_size).eval(),
        device="cpu",
    )


# --------------------------------------------------------------------------
# explainability/unified_explanation.py
# --------------------------------------------------------------------------
class TestExplainerDoesNotInventAttributions:
    """The explainer used to attribute a prediction to an all-zero input.

    `for e in obs_entries[-5:]` never read `e`; every timestep was
    `np.zeros(27)`. `|grad * input|` was therefore identically zero, the
    `if total_attr > 0` normalisation guard skipped, and `sorted()` returned
    whatever FEATURE_NAMES lists first. The operator narrative read
    "Primary driving indicators: H_emb_0 (0.0%), H_emb_1 (0.0%), H_emb_2 (0.0%)"
    for every host, every alert, forever.
    """

    @pytest.fixture
    def explainer(self):
        torch.manual_seed(0)
        return UnifiedExplainer(MultiTaskLSTM(input_dim=27, hidden_dim=64), device="cpu")

    @pytest.fixture
    def trajectory(self):
        return HostAttackTrajectory(
            host_ip="10.0.0.5",
            observed_entries=[_entry(i) for i in range(CONTRACT.history_steps)],
            forecast_entries=[],
            cumulative_forecast_risk=0.0,
        )

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_absent_window_yields_no_attribution_rather_than_ranked_zeros(
        self, explainer, trajectory, fast_mode
    ):
        exp = explainer.explain_host_trajectory(
            trajectory, fast_mode=fast_mode, use_cache=False
        )
        assert exp.top_feature_attributions == [], (
            "with no scored window the explainer must report nothing, not a "
            "ranking of zeros that looks like a real attribution"
        )
        assert "Primary driving indicators" not in exp.operator_narrative
        assert "unavailable" in exp.operator_narrative

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_supplied_window_produces_real_attribution(
        self, explainer, trajectory, fast_mode
    ):
        seq = np.random.RandomState(0).rand(CONTRACT.history_steps, 27).astype(np.float32)
        exp = explainer.explain_host_trajectory(
            trajectory, fast_mode=fast_mode, use_cache=False, feature_sequence=seq
        )
        scores = [v for _, v in exp.top_feature_attributions]
        assert len(scores) == 5
        assert sum(scores) > 0.0, "attribution over a non-zero window must be non-zero"
        assert scores == sorted(scores, reverse=True)
        assert all(n in FEATURE_NAMES for n, _ in exp.top_feature_attributions)

    def test_attention_spans_the_contract_history_not_a_hardcoded_five(
        self, explainer, trajectory
    ):
        seq = np.random.RandomState(1).rand(CONTRACT.history_steps, 27).astype(np.float32)
        exp = explainer.explain_host_trajectory(
            trajectory, use_cache=False, feature_sequence=seq
        )
        assert len(exp.temporal_attention_weights) == CONTRACT.history_steps

    def test_attribution_is_reproducible(self, explainer, trajectory):
        """The gradient path used to call .train(), leaving dropout=0.2 live, so
        explaining the same alert twice gave two different answers."""
        seq = np.random.RandomState(2).rand(CONTRACT.history_steps, 27).astype(np.float32)
        a = explainer.explain_host_trajectory(
            trajectory, fast_mode=False, use_cache=False, feature_sequence=seq
        )
        b = explainer.explain_host_trajectory(
            trajectory, fast_mode=False, use_cache=False, feature_sequence=seq
        )
        assert a.top_feature_attributions == b.top_feature_attributions
        assert not explainer.branch_a.training, "the model must be left in eval()"

    def test_wrong_width_is_refused_not_silently_reshaped(self, explainer, trajectory):
        with pytest.raises(ValueError):
            explainer.explain_host_trajectory(
                trajectory, use_cache=False,
                feature_sequence=np.zeros((CONTRACT.history_steps, 26), dtype=np.float32),
            )


# --------------------------------------------------------------------------
# correlation/trajectory_assembler.py
# --------------------------------------------------------------------------
class TestAssemblerMatchesTrainingLayout:
    """The assembler right-padded; every trainer left-pads.

    sequence_dataset.create_host_sequence_samples builds each window as
    `padding + feature_vectors`, and LazyHostSequenceDataset as
    `concatenate([pad, feats])` -- zeros first, the newest step last -- with no
    mask. The assembler filled from index 0 and masked the tail, which is the
    opposite arrangement and a different function.
    """

    def test_branch_a_input_is_left_padded_like_training(self, assembler):
        short = [_snap(i) for i in range(3)]
        long_ = [_snap(i, "10.0.0.9") for i in range(CONTRACT.history_steps)]

        seen = {}
        original = assembler.branch_a.forward

        def spy(x, mask=None, t_history=None):
            seen["x"] = x.detach().clone()
            seen["mask"] = mask
            return original(x, mask=mask)

        assembler.branch_a.forward = spy
        try:
            assembler.assemble_trajectories_batch(
                {"10.0.0.5": short, "10.0.0.9": long_}, batch_size=8
            )
        finally:
            assembler.branch_a.forward = original

        rows = seen["x"][0].abs().sum(-1).numpy()
        n_real = 3
        assert np.allclose(rows[:-n_real], 0.0), "padding must be at the START"
        assert (rows[-n_real:] > 0).all(), "the host's real steps must be LAST"

        # and the training builder agrees about which end the padding goes on
        training = create_host_sequence_samples(
            {"10.0.0.5": short}, seq_len=CONTRACT.history_steps, min_history_steps=1
        )[-1]["features"]
        t_rows = np.abs(training).sum(-1)
        assert t_rows[0] == 0.0 and t_rows[-1] > 0.0

    def test_no_mask_is_passed_because_training_passes_none(self, assembler):
        short = [_snap(i) for i in range(3)]
        seen = {}
        original = assembler.branch_a.forward

        def spy(x, mask=None, t_history=None):
            seen["mask"] = mask
            return original(x, mask=mask)

        assembler.branch_a.forward = spy
        try:
            assembler.assemble_trajectories_batch({"10.0.0.5": short}, batch_size=8)
        finally:
            assembler.branch_a.forward = original
        assert seen["mask"] is None

    def test_world_model_rollout_starts_from_the_hosts_real_latest_state(self, assembler):
        """The defect with the worst consequence.

        h_hist was right-padded, so for any host with fewer snapshots than the
        longest host in its minibatch the LAST history step -- the state the
        autoregressive residual-delta rollout extends -- was the zero vector.
        The host was forecast from the origin. Measured before the fix:
        |h[short, -1]| = 0.0 against |h[long, -1]| = 18.0.
        """
        short = [_snap(i) for i in range(3)]
        long_ = [_snap(i, "10.0.0.9") for i in range(CONTRACT.history_steps)]

        seen = {}
        original = assembler.wdt.rollout

        def spy(h_seq, *a, **k):
            seen["h"] = h_seq.detach().clone()
            return original(h_seq, *a, **k)

        assembler.wdt.rollout = spy
        try:
            assembler.assemble_trajectories_batch(
                {"10.0.0.5": short, "10.0.0.9": long_}, batch_size=8
            )
        finally:
            assembler.wdt.rollout = original

        last_step = seen["h"][0, -1].numpy()
        assert float(np.abs(last_step).sum()) > 0.0, (
            "the short-history host's rollout seed must not be the zero vector"
        )
        np.testing.assert_allclose(last_step, short[-1].embedding, rtol=0, atol=1e-6)


class TestAssemblerHonoursTheContract:
    def test_single_host_wrapper_defaults_to_the_contract_not_60s(self, assembler):
        """`assemble_host_trajectory` still defaulted to K=4, window=60.0 -- the
        MACRO granularity no model here is trained at -- long after the batch
        entry point was fixed. It produced a 4-step timeline whose steps were
        stamped 60 s apart from weights that see 2 s and 5 steps."""
        snaps = [_snap(i) for i in range(CONTRACT.history_steps)]
        traj = assembler.assemble_host_trajectory("10.0.0.5", snaps)

        assert len(traj.forecast_entries) == CONTRACT.forecast_steps
        spacing = (
            traj.forecast_entries[1].timestamp - traj.forecast_entries[0].timestamp
        )
        # the forecast grid: forecast_window_seconds per step (30 s under the
        # multi-scale contract), the grid serving and the hazard horizon use
        assert spacing == pytest.approx(CONTRACT.forecast_window_seconds)

    def test_forecast_confidence_comes_from_the_decoder(self, assembler):
        """It was `max(0.4, 0.9 - 0.1 * k)`: a fixed decay that never consulted
        the model, so step 0 of every forecast for every host was reported at
        exactly 90% confidence. The same value feeds CausalEdgeScorer's
        min_confidence feature."""
        snaps = [_snap(i) for i in range(CONTRACT.history_steps)]
        confs = [
            e.confidence
            for e in assembler.assemble_host_trajectory("10.0.0.5", snaps).forecast_entries
        ]
        fabricated = [max(0.4, 0.9 - 0.1 * k) for k in range(CONTRACT.forecast_steps)]
        assert confs != fabricated
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_empty_technique_is_not_relabelled_as_c2(self, assembler):
        """`tech if tech else "T1071"` renamed an unknown forecast to
        Application Layer Protocol -- a specific C2 claim, invented."""
        snaps = [_snap(i) for i in range(CONTRACT.history_steps)]
        traj = assembler.assemble_host_trajectory("10.0.0.5", snaps)
        import inspect
        src = inspect.getsource(AttackTrajectoryAssembler.assemble_trajectories_batch)
        assert 'else "T1071"' not in src
        assert all(isinstance(e.technique_id, str) for e in traj.forecast_entries)

    def test_empty_and_single_window_hosts_do_not_crash(self, assembler):
        res = assembler.assemble_trajectories_batch(
            {"empty": [], "one": [_snap(0, "one")]}, batch_size=8
        )
        assert res["empty"].forecast_entries == []
        assert res["empty"].cumulative_forecast_risk == 0.0
        assert len(res["one"].forecast_entries) == CONTRACT.forecast_steps


# --------------------------------------------------------------------------
# control_backend/telemetry_service.py -- source-level guards
# --------------------------------------------------------------------------
class TestTelemetryServiceReportsMeasurements:
    """These are source guards, deliberately.

    The values live inside `_tail_worker`'s read loop and are not reachable
    without a running sensor, so the pin is against the exact expressions that
    fabricated them rather than against a return value.
    """

    @pytest.fixture
    def source(self):
        """Executable lines only.

        The comments at each fixed site quote the expression that used to be
        there, so a raw text search would match the explanation of the defect
        as readily as the defect. Stripping comments keeps the guard pointed at
        code.
        """
        path = os.path.join(PROJECT_ROOT, "control_backend", "telemetry_service.py")
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        return "\n".join(
            line for line in lines if not line.lstrip().startswith("#")
        )

    def test_packet_loss_is_not_derived_from_the_wall_clock(self, source):
        # `if loss_pct == 0.0: loss_pct = round((int(time.time()) % 4) * 0.1, 1)`
        # overwrote a measured zero with a 0.0-0.3% figure from the clock, so a
        # healthy link could never report healthy.
        assert "(int(time.time()) % 4)" not in source

    def test_throughput_has_no_invented_floor(self, source):
        # `max(raw_mbps, len(flows) * 0.35 + 20.0)` never reported below
        # 20 Mbps whatever the wire was carrying.
        assert "len(flows) * 0.35 + 20.0" not in source

    def test_latency_is_not_padded_with_synthetic_jitter(self, source):
        assert "jitter" not in source
        assert "len(flows) * 0.25 + 8.0" not in source

    def test_throughput_uses_the_served_window_not_a_literal_2(self, source):
        assert "(2.0 * 1_000_000.0)" not in source
        assert "window_s" in source

    def test_attack_state_uses_the_fitted_threshold_not_a_literal_65(self, source):
        # alert_level/alert use adapter.alert_threshold, which _adopt_risk_semantics
        # takes from the checkpoint. A literal 65 here meant that at a fitted
        # threshold of 0.40 a risk of 0.55 raised alert=True / ELEVATED / "high"
        # while this field on the same bus still said "stopped".
        assert "self.current_anomaly_score >= 65" not in source
        assert "alert_threshold" in source
