"""Shared test configuration.

## Why the contract-mismatch escape hatch is set here

`control_backend/model_adapter.py` now REFUSES to serve checkpoints whose
temporal contract disagrees with `cyberworld_v4.config.get_contract()`. That is
deliberate: serving a model trained for a 5-step history and an 8-step horizon
under a contract that says 15 and 5 produces confident, wrong forecasts.

The checkpoints currently in `saved_models/` are exactly that stale:

    contract   : window 2.0s | history 15 | forecast 5
    branch_a   : window 2.0s | history  5 | forecast  -
    branch_b   : window 2.0s | history  5 | forecast  8
    deepop     : window 2.0s | history  -  | forecast  8

`history 5 / forecast 8` are the values the old, now-deleted second contract in
`data_unification/temporal_config.py` declared. So the drift is not theoretical;
it is baked into the shipped weights.

Until the retrain lands, the backend tests would fail at import. Setting the
opt-in here keeps them running while `test_served_checkpoints_match_the_contract`
below stays RED as the tracked reminder. When the retrain completes, that test
goes green and this whole block should be deleted.
"""

import os

os.environ.setdefault("CYBERWORLD_ALLOW_CONTRACT_MISMATCH", "1")


# ---------------------------------------------------------------------------
# Backend tests need an encoder checkpoint that can actually load
# ---------------------------------------------------------------------------
#
# The category head changed shape. It was a single Linear over
# `src_emb + dst_emb` -- direction-blind, edge-blind, and provably unable to
# separate two flows between the same host pair, which is why it collapsed to
# predicting one class for everything. It is now an MLP over
# [src_emb ; dst_emb ; edge_features].
#
# The checkpoints in saved_models/ predate that (Sep 10 v3 weights, which
# already fail validate_checkpoint), so they cannot be loaded at all. Skip the
# three modules that construct the serving adapter, with a stated reason,
# rather than letting one collection error take the whole suite down.
#
# These skips disappear the moment the retrain lands. If they are still here
# after a successful retrain, that is a bug, not housekeeping.

_ADAPTER_TESTS = [
    "test_control_backend.py",
    "test_site_config.py",
    "test_topology_service.py",
]

collect_ignore = []


def _adapter_is_constructible():
    """Why the serving adapter cannot be built right now, or None if it can.

    Two separate preconditions have to hold, and they clear at different times
    during a staged retrain, so they are reported separately:

      1. The TGNE encoder checkpoint has to match the current architecture.
         Cleared 2026-09-21 when the encoder was retrained with the edge-aware
         category head.
      2. Every served checkpoint has to carry the *same* temporal contract.
         Branch A is retrained under v4 (history 15 / forecast 5) while Branch
         B and DeepOP are still v3 (history 5 / forecast 8), so the adapter
         refuses to compose them -- correctly. This clears when
         scripts/retrain_future_models_live.py finishes.

    Returning a reason rather than a bool keeps the printed message specific;
    "adapter tests skipped" with no cause is how a skip becomes permanent.
    """
    try:
        from branch_a_gnn_lstm.train_branch_a import (
            StaleEncoderArchitecture, build_or_load_tgne_ta,
        )
    except Exception:
        return None          # cannot tell; let the tests run and report
    try:
        build_or_load_tgne_ta()
    except StaleEncoderArchitecture as exc:
        return f"the TGNE checkpoint predates the edge-aware category head ({exc})"
    except Exception:
        return None          # a different failure is the tests' to report

    # The module-level adapter is built on first access, so touch it: that is
    # where a contract conflict surfaces.
    try:
        from control_backend.model_adapter import model_adapter  # noqa: F401
    except RuntimeError as exc:
        if "contract conflict" in str(exc):
            return str(exc)
        return None
    except Exception:
        return None
    return None


_ADAPTER_SKIP_REASON = _adapter_is_constructible()
if _ADAPTER_SKIP_REASON:
    collect_ignore.extend(_ADAPTER_TESTS)
    print(f"\nSKIPPING adapter tests: {_ADAPTER_SKIP_REASON}\n"
          f"These re-enable themselves once the staged retrain completes.\n")
