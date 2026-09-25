"""The paper architecture's discrete gradation comes from the technique head.

Found by scripts/dry_run_plan.py: with gradation_mode="scalar" the level was
the nearest of {0, 1/3, 2/3, 1} to an MSE-trained sigmoid. MSE predicts the
conditional mean, so an uncertain window between Benign (0) and the attack
levels decodes to 1/3 = "Recon/Unknown" -- the held-out test read 100% Recon
while the technique head already separated 3 of 3 classes. A level is a fixed
function of the technique, so its probability is the sum of its techniques'.
"""

import csv
import glob
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM  # noqa: E402
from branch_a_gnn_lstm.sequence_dataset import (  # noqa: E402
    GRADATION_LEVELS, TECH_TO_IDX, TECHNIQUE_GRADATION, TECHNIQUE_VOCAB)

REPO = Path(__file__).resolve().parents[1]


def _paper(**kw):
    torch.manual_seed(0)
    return MultiTaskLSTM(input_dim=27, **{**MultiTaskLSTM.PAPER_ARCH, **kw}).eval()


def test_the_table_matches_every_label_map():
    rows = 0
    for f in glob.glob(str(REPO / "data_unification" / "label_maps" / "*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            techs = [t.strip() for t in (r["attck_technique_ids"] or "").split(";") if t.strip()]
            if not techs:
                continue
            primary = techs[0]           # what Branch A's technique target takes
            assert primary in TECH_TO_IDX, (f, r["raw_label"], primary)
            assert TECHNIQUE_GRADATION[primary] == GRADATION_LEVELS[r["coarse_category"]], \
                (Path(f).name, r["raw_label"], primary, r["coarse_category"])
            rows += 1
    assert rows > 20


def test_gradation_is_the_technique_distribution_summed_per_level():
    m = _paper()
    out = m(torch.randn(8, 5, 27))
    p_tech = torch.softmax(out["technique_logits"], -1)
    p_level = out["gradation_logits"].exp()
    assert torch.allclose(p_level.sum(-1), torch.ones(8), atol=1e-5)
    for lv in range(4):
        idx = [TECH_TO_IDX[t] for t in TECHNIQUE_VOCAB if TECHNIQUE_GRADATION[t] == lv]
        assert torch.allclose(p_level[:, lv], p_tech[:, idx].sum(-1), atol=1e-5)
    assert out["gradation_score"] is not None and out["gradation_score"].shape == (8,)


def test_a_confident_technique_gives_its_own_level():
    m = _paper()
    with torch.no_grad():
        m.technique_head.weight.zero_()
        m.technique_head.bias.zero_()
        for t, want in (("T1498", 3), ("T1110", 2), ("T1046", 1), ("Benign", 0)):
            m.technique_head.bias.zero_()
            m.technique_head.bias[TECH_TO_IDX[t]] = 20.0
            g = m(torch.randn(4, 3, 27))["gradation_logits"].argmax(-1)
            assert g.tolist() == [want] * 4, t


def test_state_dict_and_loss_are_unchanged():
    m = _paper()
    assert not [k for k in m.state_dict() if "tech_level" in k], "serving loads strictly"
    batch = {"risk": torch.rand(6), "technique": torch.randint(0, len(TECHNIQUE_VOCAB), (6,)),
             "gradation": torch.randint(0, 4, (6,))}
    x = torch.randn(6, 4, 27)
    pred = m(x)
    _r, _t, g = m.task_losses(pred, batch)
    want = torch.nn.functional.mse_loss(pred["gradation_score"], batch["gradation"].float() / 3)
    assert torch.allclose(g, want), "the scalar head still trains on the paper's MSE"


def test_the_legacy_class_head_is_untouched():
    torch.manual_seed(0)
    m = MultiTaskLSTM(input_dim=27, **MultiTaskLSTM.LEGACY_ARCH).eval()
    out = m(torch.randn(3, 4, 27))
    assert out["gradation_score"] is None
    assert out["gradation_logits"].shape == (3, 4)
