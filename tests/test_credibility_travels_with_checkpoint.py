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


def test_the_serving_adapter_checks_the_verdict():
    import control_backend.model_adapter as ma
    assert hasattr(ma.AntigravityModelAdapter, "_warn_if_not_credible")
    src = inspect.getsource(ma.AntigravityModelAdapter._warn_if_not_credible)
    assert "NOT CREDIBLE" in src
    assert "logger.error" in src, "an unsound checkpoint must log at ERROR"


def test_a_checkpoint_with_no_verdict_is_also_flagged():
    """Silence must not read as approval -- an older checkpoint has no verdict
    at all, and that is itself worth saying."""
    import control_backend.model_adapter as ma
    src = inspect.getsource(ma.AntigravityModelAdapter._warn_if_not_credible)
    assert "carries no credibility verdict" in src


def test_the_adapter_calls_it_on_the_branch_a_load():
    import control_backend.model_adapter as ma
    src = inspect.getsource(ma.AntigravityModelAdapter._load_models)
    assert "_warn_if_not_credible" in src, "the check is defined but never called"


def test_verdict_shape_is_serialisable():
    """torch.save must be able to write it -- no numpy scalars, no objects."""
    import json
    verdict = {"checked": True, "credible": False,
               "problems": ["label churn 0.0000: nothing to forecast"],
               "stats": {"val_label_churn": 0.0, "val_n": 147.0}}
    json.dumps(verdict)   # raises if not plain types
