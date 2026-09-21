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
