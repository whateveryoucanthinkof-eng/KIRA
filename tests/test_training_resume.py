"""Crash recovery: a run killed after epoch N continues at epoch N+1.

Before this, a crash at hour five of the encoder restarted it from epoch 0.
The end-to-end version (every real trainer killed and resumed on the dry-run
corpus) is in analysis 32; these pin the mechanism.
"""

import argparse
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from cyberworld_v4.training_guard import ResumePoint, TrainingGuard, run_fingerprint  # noqa: E402


def _setup(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    g = TrainingGuard("t", [m], opt, mode="min", patience=3, step_back_after=2,
                      warmup_steps=3, clip_norm=1.0, log=lambda _m: None)
    return m, opt, g


def _epoch(m, g, epoch):
    gen = torch.Generator().manual_seed(100 + epoch)
    x = torch.randn(32, 4, generator=gen)
    y = x.sum(1, keepdim=True)
    for i in range(0, 32, 8):
        g.backward_step(((m(x[i:i + 8]) - y[i:i + 8]) ** 2).mean())
        g.optimizer.zero_grad()
    with torch.no_grad():
        val = float(((m(x) - y) ** 2).mean())
    return g.end_epoch(val, train_loss=val)


def _fp():
    return run_fingerprint(argparse.Namespace(lr=1e-2, epochs=4, data="x"), ignore=("epochs",))


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    # uninterrupted: 4 epochs
    m, opt, g = _setup()
    for e in range(4):
        _epoch(m, g, e)
    want = [p.detach().clone() for p in m.parameters()]

    # interrupted after 2, then a fresh process-equivalent resumes
    rp = ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None)
    m1, opt1, g1 = _setup()
    for e in range(2):
        _epoch(m1, g1, e)
        rp.save(e + 1, model=m1.state_dict(), optimizer=opt1.state_dict(), guard=g1.state_dict())
    del m1, opt1, g1

    m2, opt2, g2 = _setup(seed=123)            # different init: must be overwritten
    st = ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None).load()
    m2.load_state_dict(st["model"]); opt2.load_state_dict(st["optimizer"]); g2.load_state_dict(st["guard"])
    assert st["done_epochs"] == 2 and g2.epoch == 2 and len(g2.history) == 2
    for e in range(st["done_epochs"], 4):
        _epoch(m2, g2, e)
    for a, b in zip(want, m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6), "resumed run diverged from the uninterrupted one"
    assert g2.global_step == g.global_step and g2.best == g.best


def test_a_different_run_is_set_aside_not_loaded(tmp_path):
    rp = ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None)
    rp.save(1, x=1)
    other = run_fingerprint(argparse.Namespace(lr=5e-3, epochs=4, data="x"), ignore=("epochs",))
    assert ResumePoint(tmp_path / "r.pt", other, log=lambda _m: None).load() is None
    assert (tmp_path / "r.pt.stale").exists() and not (tmp_path / "r.pt").exists()


def test_the_epoch_ceiling_may_change_between_attempts(tmp_path):
    ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None).save(1, x=1)
    more = run_fingerprint(argparse.Namespace(lr=1e-2, epochs=9, data="x"), ignore=("epochs",))
    assert ResumePoint(tmp_path / "r.pt", more, log=lambda _m: None).load()["x"] == 1


def test_a_truncated_file_is_set_aside(tmp_path):
    (tmp_path / "r.pt").write_bytes(b"not a checkpoint")
    assert ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None).load() is None
    assert (tmp_path / "r.pt.corrupt").exists()


def test_disabled_and_cleared(tmp_path):
    rp = ResumePoint(tmp_path / "r.pt", _fp(), enabled=False, log=lambda _m: None)
    rp.save(1, x=1)
    assert not (tmp_path / "r.pt").exists() and rp.load() is None
    rp2 = ResumePoint(tmp_path / "r.pt", _fp(), log=lambda _m: None)
    rp2.save(1, x=1)
    rp2.clear()
    assert not (tmp_path / "r.pt").exists()


def test_the_crash_hook_exits_after_the_file_is_written(tmp_path):
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {repr(str(__import__('pathlib').Path(__file__).resolve().parents[1]))})
        os.environ["CYBERWORLD_TEST_CRASH_AFTER_EPOCH"] = "2"
        from cyberworld_v4.training_guard import ResumePoint
        rp = ResumePoint({repr(str(tmp_path / 'r.pt'))}, {{}}, log=print)
        rp.save(1, x=1)
        rp.save(2, x=2)
        print("NOT REACHED")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 99 and "NOT REACHED" not in r.stdout
    assert ResumePoint(tmp_path / "r.pt", {}, log=lambda _m: None).load()["x"] == 2


def test_every_trainer_is_wired():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    for f, flag in (("bita/train.py", "'--no_resume'"),
                    ("scripts/retrain_branch_a_live.py", '"--no-resume"'),
                    ("scripts/retrain_future_models_live.py", '"--no-resume"')):
        src = (repo / f).read_text(encoding="utf-8")
        assert flag in src and "resume.load()" in src and "resume.save(" in src, f
    ds = (repo / "scripts/retrain_future_models_live.py").read_text(encoding="utf-8")
    assert ds.count("resume.save(") == 2 and "resume=bb_resume" in ds and "resume=dp_resume" in ds


def test_downstream_clears_its_resume_points_inside_main():
    """The first version cleared them in _score_cross_year, where they are not
    defined: NameError at the very end of a successful run (found by the
    end-to-end crash test)."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts/retrain_future_models_live.py").read_text(encoding="utf-8")
    for fn in ast.parse(src).body:
        if isinstance(fn, ast.FunctionDef):
            names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            if "bb_resume" in names or "dp_resume" in names:
                assert fn.name == "main", f"{fn.name} uses main's resume points"
