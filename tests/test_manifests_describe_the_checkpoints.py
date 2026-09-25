"""The manifest next to a checkpoint must describe that checkpoint.

All three manifests used to be hand-written, still described the retired v3
contract (5 history / 8 forecast) after the weights had been retrained under
15 / 5, and carried no metrics. They are now generated from the .pt files by
scripts/write_model_manifests.py; this fails if someone retrains without
regenerating them.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.write_model_manifests import SERVED, _sha256, temporal_contract

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", list(SERVED))
def test_manifest_matches_checkpoint(name):
    folder, ckpt_file, manifest_file = SERVED[name]
    ckpt_path = REPO / folder / ckpt_file
    if not ckpt_path.exists():
        pytest.skip(f"{ckpt_path} not present")
    manifest = json.loads((REPO / folder / manifest_file).read_text())
    assert manifest["checkpoint_sha256"] == _sha256(ckpt_path), (
        "checkpoint changed since the manifest was written; "
        "run python scripts/write_model_manifests.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    want = temporal_contract(ckpt)
    got = manifest["temporal_contract"]
    for k in ("window_seconds", "history_steps", "forecast_steps", "forecast_step_seconds"):
        assert got[k] == want[k], k
    assert "metrics" in manifest and "not_recorded" in manifest["metrics"]
