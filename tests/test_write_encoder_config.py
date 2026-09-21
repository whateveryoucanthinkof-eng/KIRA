"""A mid-run checkpoint needs a config, and num_categories must not be guessed.

bita/train.py writes <checkpoint>_config.json only when a run FINISHES.
scripts/select_best_encoder.py routinely picks an earlier epoch (the two
objectives diverge on this corpus), and that epoch's checkpoint has no config.
build_or_load_tgne_ta then falls back to its defaults -- including
num_categories=4, while this corpus has 5 -- and the load fails on a shape
mismatch in the category head.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/write_encoder_config.py")

LOG = """2026-09-21 16:00:00 [INFO] Detected coarse categories: {0: 'Benign', 1: 'C2', 2: 'Impact', 3: 'InitialAccess', 4: 'Recon'}
"""


def _run(tmp_path, log_text):
    ckpt = tmp_path / "enc-3.pth"
    ckpt.write_bytes(b"not a real checkpoint")
    log = tmp_path / "train.out"
    log.write_text(log_text)
    r = subprocess.run([sys.executable, str(SCRIPT), str(ckpt), "--log", str(log)],
                       capture_output=True, text=True)
    return ckpt, r


def test_num_categories_is_read_from_the_log(tmp_path):
    ckpt, r = _run(tmp_path, LOG)
    assert r.returncode == 0, r.stderr
    cfg = json.loads((tmp_path / "enc-3_config.json").read_text())
    assert cfg["num_categories"] == 5, "must reflect the run, not the default 4"


def test_it_refuses_to_guess_when_the_log_lacks_the_line(tmp_path):
    """Guessing here produces a checkpoint that cannot load."""
    _ckpt, r = _run(tmp_path, "nothing useful here\n")
    assert r.returncode != 0
    assert "refusing to guess" in (r.stdout + r.stderr)


def test_the_canonical_dims_are_pinned(tmp_path):
    _ckpt, r = _run(tmp_path, LOG)
    cfg = json.loads((tmp_path / "enc-3_config.json").read_text())
    assert cfg["edge_feat_dim"] == 12 and cfg["node_feat_dim"] == 12
    assert cfg["feature_schema_version"] == "1.0.0"


def test_config_sits_beside_the_checkpoint(tmp_path):
    """build_or_load_tgne_ta looks for <stem>_config.json next to the file."""
    ckpt, _r = _run(tmp_path, LOG)
    assert (ckpt.parent / (ckpt.stem + "_config.json")).exists()
