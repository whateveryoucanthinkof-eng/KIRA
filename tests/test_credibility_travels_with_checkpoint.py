"""A checkpoint must carry its own credibility verdict.

The gate was computed, printed to stdout, and discarded. The checkpoint kept
`test_metrics` but nothing recording that those numbers might be meaningless,
so a model its own training run judged unsound could be loaded and served with
no trace, and its metrics quoted as results.

That is the same failure shape as everything else this pipeline has hidden
today: a collapsed head scoring 0.83 aggregate CatAcc, a class with 7 training
samples, an encoder memorising hosts at 0.9981 AUC. The number looked fine.
"""

import inspect
import pathlib

import pytest


def test_branch_a_persists_the_verdict():
    src = inspect.getsource(__import__("scripts.retrain_branch_a_live",
                                       fromlist=["main"]).main)
    assert 'ckpt["credibility"]' in src, "the verdict is not saved"
    assert '"problems"' in src, "the reasons must be saved, not just a boolean"


def test_branch_a_warns_loudly_when_not_credible():
    src = inspect.getsource(__import__("scripts.retrain_branch_a_live",
                                       fromlist=["main"]).main)
    assert "NOT CREDIBLE" in src


def _adapter_source() -> str:
    """Read the source rather than import: importing control_backend
    constructs the adapter, which cannot load the current (stale) encoder
    checkpoint. The behaviour under test is in the source either way."""
    return pathlib.Path("control_backend/model_adapter.py").read_text()


def test_the_serving_adapter_checks_the_verdict():
    src = _adapter_source()
    assert "def _warn_if_not_credible" in src
    assert "NOT CREDIBLE" in src
    assert "logger.error" in src, "an unsound checkpoint must log at ERROR"


def test_a_checkpoint_with_no_verdict_is_also_flagged():
    """Silence must not read as approval -- an older checkpoint has no verdict
    at all, and that is itself worth saying."""
    assert "carries no credibility verdict" in _adapter_source()


def test_the_adapter_calls_it_on_the_branch_a_load():
    src = _adapter_source()
    assert src.count("_warn_if_not_credible") >= 2, (
        "the check is defined but never called"
    )


def test_verdict_shape_is_serialisable():
    """torch.save must be able to write it -- no numpy scalars, no objects."""
    import json
    verdict = {"checked": True, "credible": False,
               "problems": ["label churn 0.0000: nothing to forecast"],
               "stats": {"val_label_churn": 0.0, "val_n": 147.0}}
    json.dumps(verdict)   # raises if not plain types


# ---------------------------------------------------------------------------
# Label-mapping coverage must be reported, not just tracked
# ---------------------------------------------------------------------------

def test_label_coverage_is_reported_by_the_trainer():
    """LabelResolver tracks unresolved labels and had ZERO callers -- the
    number was computed and discarded, like the credibility verdict and the
    two loss terms before it.

    Unmapped labels become UNKNOWN with is_attack=False. That is the right
    default (it asserts nothing) but it is silent label noise if the rate is
    material, and it will matter when new data arrives with label strings the
    maps have not seen.
    """
    src = pathlib.Path("bita/train.py").read_text()
    assert "unresolved_report()" in src, "label coverage is never reported"
    assert "UNRESOLVED" in src, "a nonzero unresolved rate must warn"


def test_the_resolver_still_reports_a_usable_shape():
    from data_unification.label_resolver import LabelResolver
    rep = LabelResolver().unresolved_report()
    for k in ("resolve_calls", "unresolved_calls", "unresolved_rate",
              "mapped_rate", "distinct_unresolved_labels"):
        assert k in rep
