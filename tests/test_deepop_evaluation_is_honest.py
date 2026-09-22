"""DeepOP's reported numbers must describe DeepOP.

Every test here pins a defect that made a printed number mean something other
than what it says. Each one is stated with the measurement that found it.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepop_decoder.forecast_decoder import (
    CONTINUITY_BONUS_BENIGN,
    DeepOPForecastDecoder,
    DeepOPTokenScorer,
    smoothed_and_plain_ce,
)
from deepop_decoder.joint_vocab import get_joint_vocab

V = get_joint_vocab().vocab_size
BOS = get_joint_vocab().bos_idx
BENIGN = get_joint_vocab().encode("Benign", None)
C2 = get_joint_vocab().encode("C2", "T1071")


def _decoder(**kw):
    torch.manual_seed(0)
    return DeepOPForecastDecoder(
        d_latent=12, d_model=72, vocab_size=V, n_heads=6, num_layers=2,
        window_sizes=[2, 4, 8], dim_feedforward=144, **kw)


# ---------------------------------------------------------------------------
# 1. the single highest-value property: the decoder cannot see ahead
# ---------------------------------------------------------------------------

def test_no_target_token_influences_its_own_or_an_earlier_prediction():
    """Whole-decoder causality, not just the attention module.

    tests/test_deepop_attention_is_causal.py exercises CausalWindowAttention in
    isolation. DeepOPForecastDecoder adds a cross-attention, a future-state
    residual gate and two heads that read h_future directly, any of which could
    reintroduce a path from a later input token to an earlier logit. This
    differentiates the logits at position s with respect to the embedding of
    every input position and asserts the future half is exactly zero.

    Measured on the shipped configuration: the future gradient mass is 0.000e+00
    at every position, so the flat validation loss is NOT a leak artefact.
    """
    d = _decoder().eval()
    B, S = 3, 5
    h = torch.randn(B, S, 12)
    tok = torch.randint(3, V, (B, S))

    emb = d.token_embed(tok).detach().clone().requires_grad_(True)

    class _Fixed(torch.nn.Module):
        def forward(self, x):
            return emb

    real, d.token_embed = d.token_embed, _Fixed()
    try:
        for s in range(S):
            emb.grad = None
            d(h, tok)[:, s, :].sum().backward()
            future = emb.grad.abs().sum(dim=(0, 2))[s + 1:]
            assert float(future.sum()) == 0.0, (
                f"logit at position {s} depends on input tokens {s+1}..{S-1} "
                f"(mass {float(future.sum()):.3e}) -- the decoder can see ahead")
            past = emb.grad.abs().sum(dim=(0, 2))[: s + 1]
            assert float(past.sum()) > 0.0, "vacuous: no gradient anywhere"
    finally:
        d.token_embed = real


# ---------------------------------------------------------------------------
# 2. the prototype head was born dead
# ---------------------------------------------------------------------------

def test_prototype_head_receives_gradient_from_a_fresh_model():
    """`self.prototypes = torch.zeros(...)` gated itself out of the graph.

    forward() only computes proto_logits when some prototype has norm > 1e-4;
    otherwise it returns a freshly-built zeros tensor, which is a constant.
    With an all-zero init, `prototypes.grad` stayed None and the parameter was
    still exactly 0.0 after 50 AdamW steps -- so `use_prototypes=True` was a
    no-op in every checkpoint train_deepop_live produced.
    """
    d = _decoder()
    assert not bool((d.prototypes == 0).all()), "prototypes must not start at zero"
    h, tok, tgt = torch.randn(4, 5, 12), torch.randint(3, V, (4, 5)), torch.randint(3, V, (4, 5))
    F.cross_entropy(d(h, tok).reshape(-1, V), tgt.reshape(-1)).backward()
    assert d.prototypes.grad is not None, "the prototype head is not in the autograd graph"
    assert float(d.prototypes.grad.abs().sum()) > 0.0


def test_an_unseeded_prototype_is_floored_not_promoted():
    """Masking an inactive class to logit 0 PROMOTED it.

    An active class gets proto_scale * cos(h, p) in [-2.5, +2.5] centred near
    zero, so a literal 0 for an unseeded class sits mid-range: better evidence
    than every class the prototype head actively dislikes. With 6 of the 10
    joint tokens absent from the corpus that is six free votes per step. They
    are now floored at -proto_scale, the least a cosine can award.
    """
    d = _decoder().eval()
    with torch.no_grad():
        d.prototypes.normal_(0, 0.5)
        d.prototypes[7:] = 0.0                     # simulate unseeded classes
        d.fc_out.weight.zero_(); d.fc_out.bias.zero_()
        d.direct_head.weight.zero_(); d.direct_head.bias.zero_()
    h = torch.randn(8, 5, 12)
    logits = d(h, torch.randint(3, V, (8, 5)))
    dead = logits[..., 7:]
    live = logits[..., 3:7]
    dead, live = dead.detach(), live.detach()
    assert float(dead.max()) <= float(live.min()) + 1e-4, (
        "an unseeded class outscores a seeded one; the mask is promoting "
        "classes the prototype head knows nothing about")


# ---------------------------------------------------------------------------
# 3. baselines must have the same information as the model
# ---------------------------------------------------------------------------

def _corpus_like_batch(n=40000, churn=0.0524, p_atk=0.175, seed=0):
    """Sequences at the statistics the retrain logged for the val split:
    val_label_churn 0.0524, val_positive_rate 0.175. At these values the
    free-running persistence accuracy reproduces the logged
    val_persistence_accuracy of 0.8248."""
    rng = np.random.default_rng(seed)
    K = 5
    cur = np.where(rng.random(n) < p_atk, C2, BENIGN)
    prev = np.where(rng.random(n) < churn, np.where(cur == C2, BENIGN, C2), cur)
    seq = np.empty((n, K), dtype=np.int64)
    for k in range(K):
        flip = rng.random(n) < churn
        cur = np.where(flip, np.where(cur == C2, BENIGN, C2), cur)
        seq[:, k] = cur
    tgt = torch.from_numpy(seq)
    obs = torch.from_numpy(prev)
    inp = torch.cat([torch.full((n, 1), BOS, dtype=torch.long), tgt[:, :-1]], 1)
    return tgt, inp, obs


def test_a_zero_parameter_echo_model_does_not_look_like_it_beat_persistence():
    """The defect this whole file exists for.

    train_deepop_live scores a TEACHER-FORCED model (given y_{s-1} at step s)
    against a FREE-RUNNING baseline (given y_{-1}, held for all 5 steps).
    Measured on 200k corpus-statistics sequences, a model with no parameters
    that echoes its own input token scores:

        acc 0.9165  macro_f1 0.8829  persistence(printed) 0.8251
        -> lift +0.0914   "beats both baselines"

    while the information-matched baseline is 0.9382, which it does not reach.
    The scorer must report the matched pair so the echo shows a negative lift.
    """
    tgt, inp, obs = _corpus_like_batch()
    echo = inp.clone()
    echo[:, 0] = BENIGN                       # <BOS> is not a legal output

    sc = DeepOPTokenScorer(V)
    sc.update(tgt, inp, obs, pred_tf=echo)
    r = sc.result()

    assert r["acc_persistence_free"] == pytest.approx(0.825, abs=0.02), (
        "the synthetic corpus no longer matches the logged val_persistence_accuracy")
    assert r["acc_persistence_fed"] > r["acc_persistence_free"] + 0.08, (
        "the matched baseline must be much stronger than the free-running one")
    assert r["lift_teacher_forced"] < 0.0, (
        f"an echo model shows lift {r['lift_teacher_forced']:+.4f} against its "
        f"matched baseline; it must not look like it learned anything")


def test_the_free_running_baseline_is_still_reported_for_free_running_preds():
    """Guard the guard: the matched pairing must not simply make everything
    negative. A genuinely-better-than-persistence free-running prediction has
    to show a positive free-running lift."""
    tgt, inp, obs = _corpus_like_batch()
    sc = DeepOPTokenScorer(V)
    sc.update(tgt, inp, obs, pred_free=tgt.clone())   # an oracle
    r = sc.result()
    assert r["acc_free"] == 1.0
    assert r["lift_free"] > 0.0


def test_macro_f1_ignores_classes_with_no_support():
    """6 of the 10 joint tokens (<PAD>, <BOS>, <EOS>, CredentialAccess.T1110,
    Exfiltration.T1005, Recon.T1595) never occur as a target in this corpus,
    and Impact.T1498 occurs in train but not val. Averaging F1 over the union
    of true and predicted classes lets one stray prediction into an absent
    class add a hard 0.0 to the mean."""
    tgt, inp, obs = _corpus_like_batch(n=1000)
    perfect = tgt.clone()
    sc = DeepOPTokenScorer(V); sc.update(tgt, inp, obs, pred_tf=perfect)
    assert sc.result()["macro_f1_teacher_forced"] == pytest.approx(1.0)
    assert sc.result()["classes_present"] == 2

    stray = tgt.clone(); stray[0, 0] = 9        # Recon.T1595, absent from targets
    sc2 = DeepOPTokenScorer(V); sc2.update(tgt, inp, obs, pred_tf=stray)
    got = sc2.result()["macro_f1_teacher_forced"]
    assert got > 0.98, (
        f"one stray prediction into an absent class dropped macro-F1 to {got:.4f}; "
        f"it is being averaged over classes with no support")


# ---------------------------------------------------------------------------
# 4. train and val losses must be the same function
# ---------------------------------------------------------------------------

def test_smoothed_and_plain_losses_are_reported_as_a_pair():
    """train_deepop_live optimises cross_entropy(label_smoothing=0.04) and
    prints plain cross_entropy as val_loss, then shows them side by side.

    The relation is the identity
        L_smooth = (1 - eps) * plain + eps * U,  U = mean_c NLL(c) >= ln(V)
    so the two are only equal when the model is no better than the mean over
    all classes. With eps*U >= 0.0921 the comparable train CE behind the
    reported 0.6644 is at most 0.5961, making the real degradation at least
    0.2087 nats against the printed 0.1404.

    NOTE the sign: smoothing RAISES the train number, so this correction makes
    the train/val gap larger, not smaller.
    """
    torch.manual_seed(0)
    # a distribution in the regime the run is actually in: confident, mostly
    # right. Random logits on random targets sit at U ~= plain, where the two
    # losses coincide and the test would be vacuous.
    n = 8192
    logits = torch.full((n, V), -6.0)
    logits[:, BENIGN] = 3.0
    logits[:, C2] = 0.0
    tgt = torch.where(torch.rand(n) < 0.9, torch.tensor(BENIGN), torch.tensor(C2))

    sm, plain = smoothed_and_plain_ce(logits, tgt, label_smoothing=0.04)
    assert float(plain) == pytest.approx(float(F.cross_entropy(logits, tgt)), abs=1e-6)
    assert not plain.requires_grad, "the plain CE must not be part of the graph"

    # the exact identity, so the helper cannot drift from what PyTorch computes
    lp = F.log_softmax(logits, -1)
    U = float(-lp.mean(-1).mean())
    assert float(sm) == pytest.approx(0.96 * float(plain) + 0.04 * U, abs=1e-5)
    assert 0.04 * U >= 0.04 * np.log(V) - 1e-9, "eps*U cannot fall below eps*ln(V)"
    assert float(sm) - float(plain) > 0.1, (
        f"in the confident regime the smoothed loss must sit well above the "
        f"plain one; got {float(sm)-float(plain):.4f}")


def test_label_smoothing_spends_mass_on_tokens_that_cannot_occur():
    """Only 4 of 10 tokens occur in train. eps/V on each of the other 6 is
    2.4% of every target's probability mass assigned to impossible classes --
    a floor on the loss, not regularisation."""
    eps, dead = 0.04, 6
    assert dead * eps / V == pytest.approx(0.024)


# ---------------------------------------------------------------------------
# 5. knobs that do nothing must not look like knobs that do something
# ---------------------------------------------------------------------------

def test_temperature_cannot_change_the_output():
    """Decoding is greedy and argmax(x/T) == argmax(x) for every T > 0."""
    d = _decoder().eval()
    h = torch.randn(32, 5, 12)
    ref, _ = d.forecast_sequence(h, max_steps=5, temperature=1.0)
    for T in (0.05, 0.8, 4.0):
        got, _ = d.forecast_sequence(h, max_steps=5, temperature=T)
        assert torch.equal(ref, got)


def test_the_continuity_bonus_can_be_switched_off_for_measurement():
    """`forecast_sequence` adds +1.8 (Benign) / +1.2 (attack) to the observed
    token at step 0. It is never in the training objective. Measured on an
    untrained decoder over 4,000 samples at the corpus's 82.5% Benign observed
    mix, it changes 71.9% of step-0 decisions and takes step-0 agreement with
    the observed token from 0.177 to 0.896 -- so with it on, step 0 is
    substantially the persistence baseline rather than the model. Serving
    relies on the default, so the default must stay 1.0 and evaluation must be
    able to turn it off."""
    d = _decoder().eval()
    h = torch.randn(2000, 5, 12)
    obs = torch.where(torch.rand(2000) < 0.825, torch.tensor(BENIGN), torch.tensor(C2))

    on, _ = d.forecast_sequence(h, max_steps=5, observed_token=obs, continuity_bonus=1.0)
    off, _ = d.forecast_sequence(h, max_steps=5, observed_token=obs, continuity_bonus=0.0)
    none_, _ = d.forecast_sequence(h, max_steps=5, observed_token=None)

    assert torch.equal(off, none_), "continuity_bonus=0.0 must equal passing no observation"
    assert not torch.equal(on, off), "the bonus must actually be doing something"
    agree_on = float((on[:, 0] == obs).float().mean())
    agree_off = float((off[:, 0] == obs).float().mean())
    assert agree_on > agree_off + 0.2, (
        f"the bonus should visibly pull step 0 onto the observed token "
        f"({agree_off:.3f} -> {agree_on:.3f})")


def test_the_bonus_is_applied_to_the_right_sample_and_token():
    """The vectorised rewrite must reproduce the per-sample loop exactly:
    the bonus goes on sample b's own observed token, and on no other."""
    d = _decoder().eval()
    h = torch.randn(6, 5, 12)
    obs = torch.tensor([BENIGN, C2, BENIGN, C2, BOS, 0])
    with torch.no_grad():
        base = d(h, torch.full((6, 1), BOS, dtype=torch.long))[:, -1, :].clone()
    base[:, d.vocab.bos_idx] = -1e9
    base[:, d.vocab.pad_idx] = -1e9
    expect = base.clone()
    for b in range(6):
        ot = int(obs[b])
        if ot not in (d.vocab.bos_idx, d.vocab.pad_idx):
            expect[b, ot] += CONTINUITY_BONUS_BENIGN if ot == BENIGN else 1.2
    got, _ = d.forecast_sequence(h, max_steps=1, observed_token=obs)
    assert torch.equal(got[:, 0], expect.argmax(dim=-1))


def test_observed_tokens_may_be_a_column_vector():
    """`forecast_sequence` documents [B] or [B, 1]; evaluate_forecast_rigor's
    old persistence baseline built a [B, 1, K] tensor from the [B, 1] form."""
    d = _decoder().eval()
    h = torch.randn(8, 5, 12)
    flat = torch.full((8,), BENIGN, dtype=torch.long)
    a, _ = d.forecast_sequence(h, max_steps=5, observed_token=flat)
    b, _ = d.forecast_sequence(h, max_steps=5, observed_token=flat.unsqueeze(1))
    assert torch.equal(a, b)

    from deepop_decoder.forecast_decoder import evaluate_forecast_rigor
    tgt = torch.randint(3, V, (8, 5))
    r = evaluate_forecast_rigor(d, h, tgt, observed_tokens=flat.unsqueeze(1))
    assert 0.0 <= r["acc_persistence_free"] <= 1.0


def test_evaluate_forecast_rigor_reports_both_modes_and_both_baselines():
    d = _decoder().eval()
    tgt, _, obs = _corpus_like_batch(n=512)
    h = torch.randn(512, 5, 12)
    from deepop_decoder.forecast_decoder import evaluate_forecast_rigor
    r = evaluate_forecast_rigor(d, h, tgt, observed_tokens=obs, batch_size=128)
    for k in ("acc_teacher_forced", "acc_free", "acc_free_no_continuity_bonus",
              "acc_persistence_fed", "acc_persistence_free", "acc_majority",
              "lift_teacher_forced", "lift_free", "lift_free_no_continuity_bonus"):
        assert k in r, f"{k} missing -- a number cannot be read without its baseline"
    # batching must not change the answer
    r2 = evaluate_forecast_rigor(d, h, tgt, observed_tokens=obs, batch_size=512)
    assert r["acc_free"] == pytest.approx(r2["acc_free"])
    assert r["acc_teacher_forced"] == pytest.approx(r2["acc_teacher_forced"])
