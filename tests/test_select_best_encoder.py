"""Encoder selection must not follow a metric that diverges from the goal.

Observed over six epochs of the real run:

    epoch      0       1       2       3       4       5
    val_ap    .9971   .9968   .9968   .9973   .9974   .9975   <- still rising
    ind AP    .9926   .9921   .9918   .9935   .9936   .9934
    C2 ind    .325    .319    .517    .565    .586    .480    <- peaks at 4
    IA        .505    .714    .578    .612    .561    .510

val_ap rises monotonically at the end while classification degrades, so
selecting on it picks epoch 5 and throws away the better classifier. The
combined criterion picks epoch 3.
"""

import pytest

from scripts.select_best_encoder import score_log

REAL_LOG = """
Epoch 00 [700s] Loss: 0.28 | Val AUC: 0.9970, AP: 0.9971, CatAcc: 0.97, MRR: 0.98 | Inductive Val AUC: 0.9923, AP: 0.9926, CatAcc: 0.97
           per-class val acc: {'Benign': 0.994, 'C2': 0.677, 'Impact': 1.0, 'InitialAccess': 0.505, 'Recon': 0.995} | inductive: {}
Epoch 01 [700s] Loss: 0.18 | Val AUC: 0.9967, AP: 0.9968, CatAcc: 0.98, MRR: 0.99 | Inductive Val AUC: 0.9918, AP: 0.9921, CatAcc: 0.97
           per-class val acc: {'Benign': 0.993, 'C2': 0.674, 'Impact': 1.0, 'InitialAccess': 0.714, 'Recon': 0.994} | inductive: {}
Epoch 02 [700s] Loss: 0.16 | Val AUC: 0.9967, AP: 0.9968, CatAcc: 0.98, MRR: 0.99 | Inductive Val AUC: 0.9916, AP: 0.9918, CatAcc: 0.98
           per-class val acc: {'Benign': 0.994, 'C2': 0.761, 'Impact': 1.0, 'InitialAccess': 0.578, 'Recon': 0.994} | inductive: {}
Epoch 03 [700s] Loss: 0.16 | Val AUC: 0.9973, AP: 0.9973, CatAcc: 0.98, MRR: 0.99 | Inductive Val AUC: 0.9936, AP: 0.9935, CatAcc: 0.98
           per-class val acc: {'Benign': 0.993, 'C2': 0.779, 'Impact': 1.0, 'InitialAccess': 0.612, 'Recon': 0.996} | inductive: {}
Epoch 04 [700s] Loss: 0.15 | Val AUC: 0.9974, AP: 0.9974, CatAcc: 0.98, MRR: 0.99 | Inductive Val AUC: 0.9937, AP: 0.9936, CatAcc: 0.98
           per-class val acc: {'Benign': 0.994, 'C2': 0.796, 'Impact': 1.0, 'InitialAccess': 0.561, 'Recon': 0.996} | inductive: {}
Epoch 05 [700s] Loss: 0.15 | Val AUC: 0.9974, AP: 0.9975, CatAcc: 0.98, MRR: 0.99 | Inductive Val AUC: 0.9936, AP: 0.9934, CatAcc: 0.98
           per-class val acc: {'Benign': 0.995, 'C2': 0.741, 'Impact': 1.0, 'InitialAccess': 0.51, 'Recon': 0.998} | inductive: {}
"""


def test_all_epochs_are_parsed():
    rows = score_log(REAL_LOG)
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5]


def test_it_picks_epoch_three_not_five():
    rows = score_log(REAL_LOG)
    best = max(rows, key=lambda r: r[3])
    assert best[0] == 3, f"selected epoch {best[0]}, expected 3"


def test_val_ap_alone_would_have_picked_five():
    """Establish that the two criteria really disagree on this run."""
    import re
    # the leading pipe matters: "Inductive Val AUC: ..., AP: ..." also
    # matches the bare pattern, interleaving the two series.
    aps = [float(x) for x in re.findall(r"\| Val AUC: [\d.]+, AP: ([\d.]+)", REAL_LOG)]
    assert aps.index(max(aps)) == 5, "fixture no longer shows the divergence"


def test_macro_recall_is_used_not_aggregate():
    """Epoch 5 has the best Benign and Recon but the worst C2 and IA; a macro
    average must rank it below epoch 3."""
    rows = {r[0]: r for r in score_log(REAL_LOG)}
    assert rows[5][2] < rows[3][2], "macro recall should penalise epoch 5"


def test_transductive_and_inductive_ap_are_not_confused():
    """The regex trap: "Inductive Val AUC: ..., AP: ..." also matches a bare
    "Val AUC: ..., AP: ..." pattern, which silently interleaves the two
    series. My first version of this script had exactly that bug."""
    import re
    bare = re.findall(r"Val AUC: [\d.]+, AP: ([\d.]+)", REAL_LOG)
    piped = re.findall(r"\| Val AUC: [\d.]+, AP: ([\d.]+)", REAL_LOG)
    assert len(bare) == 2 * len(piped), "the bare pattern should over-match"
    assert len(piped) == 6


def test_empty_log_yields_nothing():
    assert score_log("") == []


def test_partial_epoch_without_per_class_is_skipped():
    partial = REAL_LOG + "\nEpoch 06 [700s] Loss: 0.14 | Val AUC: 0.999, AP: 0.999, CatAcc: 0.9, MRR: 0.9 | Inductive Val AUC: 0.99, AP: 0.99, CatAcc: 0.9\n"
    rows = score_log(partial)
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5], "an epoch with no per-class line must not score"


# ---------------------------------------------------------------------------
# Selection must use the RAW score, not a smoothed one
# ---------------------------------------------------------------------------

def test_smoothing_is_informational_not_the_selector():
    """An earlier version selected on the smoothed score and picked epoch 2
    (raw 0.9286) over epoch 6 (raw 0.9357) -- a demonstrably worse checkpoint.

    Smoothing identifies a good REGION of training, but we deploy one specific
    checkpoint and its quality is its own raw score, not its neighbours'
    average.
    """
    src = open("scripts/select_best_encoder.py").read()
    assert "best = max(rows, key=lambda r: r[3])" in src, "must select on the raw score"
    assert "INFORMATIONAL ONLY" in src


def test_it_reports_the_tie_set():
    """The winner's curse is real -- C2 inductive std is 0.119 -- so the honest
    remedy is to name the epochs that are statistically tied, not to pick a
    lower-scoring one."""
    src = open("scripts/select_best_encoder.py").read()
    assert "tie_tolerance" in src
    assert "partly luck" in src


def test_smooth_handles_edges_and_trivial_windows():
    from scripts.select_best_encoder import smooth
    v = [1.0, 2.0, 3.0, 4.0]
    assert smooth(v, 1) == v
    out = smooth(v, 3)
    assert len(out) == len(v)
    assert out[0] == pytest.approx(1.5)      # clipped at the start
    assert out[-1] == pytest.approx(3.5)     # clipped at the end


def test_promotion_uses_the_trainers_harmonic_rule():
    """A collapsed head (macro recall ~0.25 = one of four classes) must not be
    promoted over a working one on link AP alone."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sbe", root / "scripts" / "select_best_encoder.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    collapsed, working = m._harmonic(0.99, 0.25), m._harmonic(0.95, 0.60)
    assert working > collapsed
