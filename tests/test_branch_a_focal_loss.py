"""Branch A's focal loss did not focus and did not balance.

Two independent defects, both present in the same twenty lines:

1. **alpha was never set.** `MultiTaskLSTM.__init__` built
   `MultiClassFocalLoss(gamma=2.0)` -- alpha=None -- so the loss applied no
   per-class weighting at all, on a 14-class vocabulary whose majority class
   is 82.5% of the corpus. The docstring described per-class weighting as the
   reason the class exists. The geometric-mean-normalised weighting had been
   written, debugged through two failed versions and tested -- in
   `bita/train.py`, for the TGNE encoder. It was never carried across.

2. **the modulating factor was computed from the label-smoothed loss.**
   `pt = exp(-ce_loss)` with `label_smoothing=0.04` is not p_t. PyTorch's
   smoothed loss is `(1-eps)*nll + eps*mean_j(-log p_j)`, whose second term
   grows without bound as the model becomes confident -- so `exp(-ce_loss)`
   never approaches 1 and `(1-pt)^gamma` never approaches 0.

The second is the more serious, because past p_t ~ 0.99 it does not merely
weaken the down-weighting, it reverses it.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from branch_a_gnn_lstm.lstm_multitask import MultiClassFocalLoss, MultiTaskLSTM

K = 14          # len(TECHNIQUE_VOCAB)
EPS = 0.04      # the head's label_smoothing
GAMMA = 2.0


def _logits_for(p_t, k=K):
    """Logits whose true-class (index 0) probability is exactly p_t."""
    rest = (1.0 - p_t) / (k - 1)
    probs = torch.full((1, k), rest, dtype=torch.double)
    probs[0, 0] = p_t
    return probs.log().float()


def _old_focal_term(p_t, k=K, eps=EPS, gamma=GAMMA):
    """What `pt = exp(-ce_loss)` produced."""
    logits = _logits_for(p_t, k)
    ce = F.cross_entropy(logits, torch.tensor([0]), reduction="none",
                         label_smoothing=eps)
    return float((1.0 - torch.exp(-ce)) ** gamma)


# --- defect 2: the modulating factor ---------------------------------------

def test_the_old_modulating_factor_grows_as_the_model_gets_more_confident():
    """The regression, stated as the inversion it is. Focal loss exists to
    shrink the weight on easy examples; past ~0.99 the old form grew it."""
    at_99 = _old_focal_term(0.99)
    at_1m9 = _old_focal_term(1 - 1e-9)
    assert at_1m9 > at_99, (at_99, at_1m9)
    assert at_1m9 / at_99 > 5, "the inversion was ~6x, not a rounding artefact"


def test_the_fixed_modulating_factor_is_monotone_decreasing():
    terms = []
    for p in (0.5, 0.9, 0.99, 0.999, 1 - 1e-6):
        logits = _logits_for(p)
        logpt = F.log_softmax(logits, -1)[0, 0]
        terms.append(float((1 - logpt.exp()) ** GAMMA))
    assert terms == sorted(terms, reverse=True), terms
    assert terms[-1] < 1e-10


def test_the_measured_88000x_gap_at_p_999():
    """The number quoted in the comment. Pinned so it cannot rot."""
    old = _old_focal_term(0.999)
    new = (1 - 0.999) ** GAMMA
    assert old / new == pytest.approx(8.836e4, rel=0.05)


def test_an_easy_example_now_contributes_almost_nothing():
    """The whole purpose: 82.5% of this corpus is easy Benign windows."""
    loss = MultiClassFocalLoss(gamma=GAMMA, label_smoothing=EPS)
    easy = float(loss(_logits_for(0.999), torch.tensor([0])))
    hard = float(loss(_logits_for(0.10), torch.tensor([0])))
    assert easy < hard / 1000, (easy, hard)


def test_label_smoothing_still_reaches_the_magnitude_term():
    """Only the modulating factor was wrong; smoothing must keep regularising."""
    logits = _logits_for(0.80)
    t = torch.tensor([0])
    smoothed = MultiClassFocalLoss(gamma=GAMMA, label_smoothing=0.04)(logits, t)
    plain = MultiClassFocalLoss(gamma=GAMMA, label_smoothing=0.0)(logits, t)
    assert float(smoothed) != pytest.approx(float(plain))


def test_gamma_zero_reduces_to_weighted_cross_entropy():
    """A sanity anchor on the new form: with no focusing the loss must be the
    (smoothed) cross-entropy exactly."""
    logits = torch.randn(64, K)
    t = torch.randint(0, K, (64,))
    got = MultiClassFocalLoss(gamma=0.0, label_smoothing=EPS)(logits, t)
    want = F.cross_entropy(logits, t, label_smoothing=EPS)
    assert float(got) == pytest.approx(float(want), abs=1e-6)


# --- defect 1: alpha -------------------------------------------------------

def _labels(counts):
    return np.concatenate([np.full(n, c, dtype=np.int64) for c, n in counts.items()])


def test_the_model_actually_installs_an_alpha_when_given_one():
    """A weight vector that never reaches the loss is decoration -- and for
    Branch A it never reached it at all, because none was ever built."""
    m = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1, num_techniques=3)
    logits = torch.zeros(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    m.tech_focal_loss.alpha = torch.ones(3)
    flat = float(m.tech_focal_loss(logits, targets))
    m.tech_focal_loss.alpha = torch.tensor([5.0, 1.0, 1.0])
    skewed = float(m.tech_focal_loss(logits, targets))
    assert skewed > flat


def test_alpha_agrees_with_the_encoder_implementation_it_was_copied_from():
    """bita/train.py::FocalLoss.inverse_frequency_alpha is the original. Two
    copies of a formula drift; this is what stops them."""
    bita = pytest.importorskip("bita.train")
    y = _labels({0: 9_000_000, 1: 1_500_000, 2: 1_000_000, 3: 300_000, 4: 120})
    mine = MultiClassFocalLoss.inverse_frequency_alpha(y, 5).numpy()
    theirs = bita.FocalLoss.inverse_frequency_alpha(y, 5).numpy()
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=0)


def test_alpha_from_counts_equals_alpha_from_labels():
    """The training split is 20.7M samples and its histogram is read off the
    store's columns; expanding it back to labels would be the expensive way to
    compute a 14-element vector. The two routes must agree exactly."""
    counts = {0: 9_000_000, 1: 1_500_000, 2: 1_000_000, 3: 300_000, 4: 120}
    from_labels = MultiClassFocalLoss.inverse_frequency_alpha(_labels(counts), 6)
    hist = np.zeros(6, dtype=np.int64)
    for c, n in counts.items():
        hist[c] = n
    from_counts = MultiClassFocalLoss.alpha_from_counts(hist, 6)
    np.testing.assert_allclose(from_labels.numpy(), from_counts.numpy(), atol=0)


def test_alpha_is_scale_invariant():
    """It depends only on the ratios, which is what lets a histogram stand in
    for the labels."""
    hist = np.array([9_000_000, 1_500_000, 1_000_000, 300_000, 120])
    a = MultiClassFocalLoss.alpha_from_counts(hist, 5).numpy()
    b = MultiClassFocalLoss.alpha_from_counts(hist * 7, 5).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_rarer_classes_get_larger_weights():
    w = MultiClassFocalLoss.alpha_from_counts([100_000, 10_000, 1_000, 100], 4).numpy()
    assert w[0] < w[1] < w[2] < w[3], w


def test_an_absent_class_keeps_weight_exactly_one():
    """Weight 1.0 is the diagnostic that found a broken split. It must survive
    and must not skew the present classes' normalisation."""
    w = MultiClassFocalLoss.alpha_from_counts([10_000, 1_000, 0, 0], 4).numpy()
    assert w[2] == pytest.approx(1.0) and w[3] == pytest.approx(1.0)
    assert w[0] != pytest.approx(1.0)


def test_the_real_corpus_shape_does_not_collapse_onto_the_clip_floor():
    """The arithmetic-mean version put four of five classes on the floor, i.e.
    weighted them identically -- no balancing among the classes that carry the
    data. The geometric mean is what prevents that."""
    w = MultiClassFocalLoss.alpha_from_counts(
        [9_000_000, 1_500_000, 1_000_000, 300_000, 120], 5).numpy()
    assert len(set(np.round(w[:4], 4))) == 4, w


def test_alpha_never_enters_the_state_dict():
    """The serving adapter builds `MultiTaskLSTM(input_dim=27, hidden_dim=64)`
    and calls `load_state_dict` strictly. A `tech_focal_loss.alpha` key in a
    checkpoint would be an unexpected key and would take serving down."""
    m = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1, num_techniques=5)
    m.tech_focal_loss.alpha = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert not [k for k in m.state_dict() if "alpha" in k]
    fresh = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1, num_techniques=5)
    fresh.load_state_dict(m.state_dict())     # strict


def test_gradation_weights_are_applied_and_also_stay_out_of_state_dict():
    torch.manual_seed(0)
    x = torch.randn(32, 5, 27)
    batch = {"risk": torch.rand(32),
             "technique": torch.zeros(32, dtype=torch.long),
             "gradation": torch.tensor([0] * 24 + [1, 2, 3] * 2 + [0, 0])}
    flat = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1, num_techniques=5)
    flat.eval()
    weighted = MultiTaskLSTM(input_dim=27, hidden_dim=16, num_layers=1,
                             num_techniques=5,
                             gradation_class_weights=torch.tensor([0.2, 5.0, 5.0, 5.0]))
    weighted.load_state_dict(flat.state_dict())
    weighted.eval()
    p = flat(x)
    _, m_flat = flat.compute_loss(p, batch)
    _, m_w = weighted.compute_loss(p, batch)
    assert float(m_flat["loss_grad"]) != pytest.approx(float(m_w["loss_grad"]))
    assert not [k for k in weighted.state_dict() if "gradation_class_weights" in k]
