"""A retrain that writes where nothing reads is a silent no-op.

`scripts/retrain_future_models_live.py` used to save Branch B and DeepOP as
`<out_dir>/host_wdt.canonical-tgne.pt` and
`<out_dir>/cwa_forecast_decoder.canonical-tgne.pt`. Nothing in the repo ever
read either name. `control_backend/model_adapter.py` loads
`saved_models/branch_b/host_wdt.pt` and
`saved_models/deepop/cwa_forecast_decoder.pt`.

So the whole downstream retrain could finish successfully and leave the served
v3 checkpoints in place -- the adapter would go on refusing to compose them
with a v4 Branch A, the adapter tests would stay skipped and the offline/live
parity check would stay blocked, with nothing reporting a failure anywhere.

These tests compare the two files' path literals directly, because the failure
is a disagreement between two source files that no runtime check catches.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADAPTER = (REPO / "control_backend" / "model_adapter.py").read_text()
RETRAIN = (REPO / "scripts" / "retrain_future_models_live.py").read_text()
BRANCH_A = (REPO / "scripts" / "retrain_branch_a_live.py").read_text()


def _code_only(src: str) -> str:
    """Drop comment lines. The comment explaining why the old names were
    abandoned legitimately names them; only executable code must not."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


def _served_paths():
    """The saved_models/... paths model_adapter.py joins onto the repo root."""
    return set(re.findall(r'os\.path\.join\(repo,\s*"(saved_models/[^"]+)"\)', ADAPTER))


def test_adapter_declares_the_three_served_checkpoints():
    served = _served_paths()
    assert "saved_models/branch_a/branch_a_lstm.pt" in served
    assert "saved_models/branch_b/host_wdt.pt" in served
    assert "saved_models/deepop/cwa_forecast_decoder.pt" in served


def test_branch_b_and_deepop_retrain_writes_to_the_served_paths():
    served = _served_paths()
    for sub, fname in (("branch_b", "host_wdt.pt"), ("deepop", "cwa_forecast_decoder.pt")):
        assert f"saved_models/{sub}/{fname}" in served, "adapter path changed"
        # the retrain builds the path as out_dir / "<sub>" / "<fname>"
        pattern = rf'args\.out_dir\s*/\s*"{sub}"\s*/\s*"{re.escape(fname)}"'
        assert re.search(pattern, RETRAIN), (
            f"retrain_future_models_live.py no longer writes {sub}/{fname}; "
            f"if the output path moved, model_adapter.py must move with it"
        )


def test_no_write_only_checkpoint_names_remain():
    """The dead `.canonical-tgne.pt` outputs must not come back."""
    assert "canonical-tgne.pt" not in _code_only(RETRAIN), (
        "a checkpoint name nothing reads was reintroduced"
    )


def test_the_existing_served_checkpoint_is_backed_up_before_being_replaced():
    """Both trainers save on every improving epoch, so the copy has to happen
    before training, not after."""
    assert "superseded-" in RETRAIN, "no backup of the checkpoint being replaced"
    backup_at = RETRAIN.index("superseded-")
    # the call sites, not the `def` lines -- those come earlier in the file
    calls = [m.start() for m in re.finditer(
        r"=?\s*train_(?:branch_b|deepop)_live\(train_traj,val_traj,", RETRAIN)]
    assert calls, "could not locate the trainer call sites"
    assert backup_at < min(calls), "backup must be taken before training starts"


def test_branch_a_retrain_also_targets_a_served_path():
    """Branch A takes --output rather than hardcoding it; the runbook and the
    systemd unit pass the served path. Pin that the adapter still expects it."""
    assert "saved_models/branch_a/branch_a_lstm.pt" in _served_paths()
    assert '--output' in BRANCH_A
