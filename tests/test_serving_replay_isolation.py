"""
tests/test_serving_replay_isolation.py

Pin for control_backend/main.py's /api/replay endpoint (serving-path review,
"Reported, not fixed"): replay used to call predict_window() on `model_adapter`,
the SAME instance the live tail worker scores on concurrently, and mutated
its shared state:

  * `rules_enabled = False`, restored in `finally` -- but only *after* the
    live tail worker could have read the wrong value mid-replay.
  * `reset_history()`, never restored. `AntigravityModelAdapter.reset_history()`
    clears `h_state_history_by_target` and `feature_history_by_target` IN
    FULL (control_backend/model_adapter.py:366-370) -- one `/api/replay` call
    wiped the rolling window for EVERY host the live tail worker was
    tracking, not just whatever the upload analysed. And for the rest of the
    replay run, any window whose target IP happens to match a live host's IP
    (likely, since both draw from the same site's asset IPs) would splice
    replayed windows into that host's real history as it rebuilds.

Fix: /api/replay now scores on a private, lazily-built `AntigravityModelAdapter`
instance obtained from `_get_replay_adapter()`, with its own history dicts
from construction, so it can never read or clear the live singleton's state.
"""
import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "bita"))

os.environ.setdefault("CYBERWORLD_ALLOW_CONTRACT_MISMATCH", "1")

import control_backend.main as main_mod
from control_backend.model_adapter import model_adapter as live_adapter


# --------------------------------------------------------------------------
# source guard -- pins the actual shape of the fix, so a regression back to
# scoring on the bare `model_adapter` singleton is caught even if the
# behavioural tests below are ever weakened.
# --------------------------------------------------------------------------
def _replay_file_source() -> str:
    import inspect
    return inspect.getsource(main_mod.replay_file)


class TestReplayEndpointSourceNeverTouchesLiveSingleton:
    def test_replay_file_has_no_bare_model_adapter_reference(self):
        src = _replay_file_source()
        # only the isolated `replay_adapter` may be scored against inside
        # the endpoint body -- not the shared `model_adapter` singleton.
        assert "model_adapter." not in src, (
            "replay_file() references the live `model_adapter` singleton "
            "directly again -- this is exactly the bug that let replay "
            "corrupt live scoring"
        )

    def test_get_replay_adapter_exists_and_is_used(self):
        assert hasattr(main_mod, "_get_replay_adapter")
        assert "_get_replay_adapter()" in _replay_file_source()


# --------------------------------------------------------------------------
# behavioural: exercises the real AntigravityModelAdapter class (fast --
# measured ~0.03s to construct a second instance once the first is warm,
# well under the memory cap) so the isolation is proven against the real
# defect mechanism, not a reimplementation of it.
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_live_adapter_state():
    """Snapshot + restore the live singleton's history around each test, so
    this file's sentinels never leak into a test that runs after it in the
    same pytest session."""
    saved_h = dict(live_adapter.h_state_history_by_target)
    saved_f = dict(live_adapter.feature_history_by_target)
    saved_rules = live_adapter.rules_enabled
    main_mod._replay_adapter = None
    yield
    live_adapter.h_state_history_by_target.clear()
    live_adapter.h_state_history_by_target.update(saved_h)
    live_adapter.feature_history_by_target.clear()
    live_adapter.feature_history_by_target.update(saved_f)
    live_adapter.rules_enabled = saved_rules
    main_mod._replay_adapter = None


class TestReplayAdapterIsIsolatedFromLiveSingleton:
    def test_get_replay_adapter_is_not_the_live_singleton(self):
        replay_adapter = main_mod._get_replay_adapter()
        assert replay_adapter is not live_adapter

    def test_get_replay_adapter_is_cached_across_calls(self):
        a = main_mod._get_replay_adapter()
        b = main_mod._get_replay_adapter()
        assert a is b, "a fresh model load on every /api/replay call is wasteful and untested for"

    def test_replay_history_dicts_are_not_shared_with_live(self):
        replay_adapter = main_mod._get_replay_adapter()
        assert (
            replay_adapter.h_state_history_by_target
            is not live_adapter.h_state_history_by_target
        )
        assert (
            replay_adapter.feature_history_by_target
            is not live_adapter.feature_history_by_target
        )


class TestReplayCannotCorruptLiveHistory:
    def test_replay_reset_history_does_not_wipe_every_live_host(self):
        """The exact old bug: reset_history() on the shared singleton wiped
        EVERY live host's history, not just the one being replayed."""
        live_adapter.feature_history_by_target["10.0.0.50"] = [torch.zeros(27)]
        live_adapter.feature_history_by_target["10.0.0.51"] = [torch.zeros(27)]
        live_adapter.h_state_history_by_target["10.0.0.50"] = [torch.zeros(12)]
        live_adapter.rules_enabled = True

        replay_adapter = main_mod._get_replay_adapter()
        # exactly what replay_file() does at the top of its try-block
        replay_adapter.rules_enabled = False
        replay_adapter.reset_history()

        assert set(live_adapter.feature_history_by_target) == {"10.0.0.50", "10.0.0.51"}, (
            "replay's reset_history() wiped live hosts' history"
        )
        assert len(live_adapter.feature_history_by_target["10.0.0.50"]) == 1
        assert "10.0.0.50" in live_adapter.h_state_history_by_target
        assert live_adapter.rules_enabled is True, (
            "forcing rules off on the replay adapter must not affect the "
            "live singleton's rules_enabled"
        )

    def test_replay_scoring_the_same_target_ip_as_a_live_host_does_not_touch_it(self):
        """Worst case: the uploaded capture's target IP matches a host the
        live tail worker is actively tracking. Replay must still only write
        into its OWN adapter's dict entry for that IP."""
        target_ip = "10.0.0.50"
        live_adapter.feature_history_by_target[target_ip] = [torch.zeros(27)]

        replay_adapter = main_mod._get_replay_adapter()
        replay_adapter.reset_history()
        # what predict_window() does to the per-target list, for the same IP
        replay_adapter.feature_history_by_target.setdefault(target_ip, []).append(torch.ones(27))

        assert len(live_adapter.feature_history_by_target[target_ip]) == 1
        assert torch.equal(
            live_adapter.feature_history_by_target[target_ip][0], torch.zeros(27)
        ), "a replay window for the same target IP as a live host leaked into its real history"
