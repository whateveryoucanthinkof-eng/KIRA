"""A run on a shared setup must train exactly like a run that built its own.

Three encoder runs on the synthetic corpus, CPU and single-threaded: (a) no
shared setup, (b) building it, (c) reusing it read-only. Epoch results and
all final weights must be identical.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location("drp", REPO / "scripts" / "dry_run_plan.py")
    drp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drp)
    return drp.build_corpus(tmp_path_factory.mktemp("synthetic"), seed=7)


def _run(tag, corpus, root, *extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=f"{REPO}:{REPO / 'bita'}")
    cmd = [sys.executable, str(REPO / "bita" / "train.py"), "--prefix", tag, "--data_name", "cic2018",
           "--save_dir", str(root / tag / "save"), "--checkpoint_dir", str(root / tag / "ckpt"),
           "--log_dir", str(root / tag / "logs"), "--seed", "42",
           "--pcap2018_root", str(corpus["pcap"]), "--pcap2018_label_dir", str(corpus["csv"]),
           "--ctu13_dir", str(corpus["ctu13"]), "--split_scheme", "cross_year_ctu",
           "--train_splits", "train", "--use_memory", "--n_degree", "10", "--n_epoch", "1",
           "--batch_size", "64", "--ingest_workers", "2", "--ingest_cache", str(root / "cache"), *extra]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    log = out.stdout + out.stderr
    epochs = [l.split("]", 1)[-1] for l in log.splitlines() if "Epoch 00 [" in l]
    epochs = [e.split("s]", 1)[-1] for e in epochs]          # drop the wall time
    sd = torch.load(glob.glob(str(root / tag / "save" / "*.pth"))[0], map_location="cpu",
                    weights_only=False)
    return log, epochs, sd.get("model_state_dict", sd)


def test_shared_setup_trains_identically(corpus, tmp_path):
    shared = tmp_path / "shared"
    _, ea, a = _run("a", corpus, tmp_path)
    lb, eb, b = _run("b", corpus, tmp_path, "--shared_setup_dir", str(shared))
    lc, ec, c = _run("c", corpus, tmp_path, "--shared_setup_dir", str(shared))
    assert "Shared setup" in lb and "building" in lb
    assert "reused (read-only)" in lc
    assert ea and ea == eb == ec
    keys = [k for k in a if torch.is_tensor(a[k])]
    assert keys and all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in keys)


def test_prune_only_removes_setup_entries(tmp_path):
    from bita.utils import setup_store as ss
    root = tmp_path / "shared"
    (root / "ckpt").mkdir(parents=True)                       # not an entry
    (root / "ckpt" / "model.pth").write_bytes(b"x")
    old = root / ("a" * 24); (old / "store").mkdir(parents=True)
    keep = root / ("b" * 24); (keep / "store").mkdir(parents=True)
    ss.prune(str(root), keep="b" * 24)
    assert (root / "ckpt" / "model.pth").exists()
    assert not old.exists() and keep.exists()
