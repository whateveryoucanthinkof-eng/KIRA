"""Regenerate saved_models/*/*.manifest.json from the served checkpoints.

The manifest is what an outsider opens to find out what a checkpoint is. All
three were hand-written, still described the retired v3 contract (5 history /
8 forecast / 16 s) and contained no accuracy numbers at all, while the weights
next to them had been retrained under 15 history / 5 forecast.

Everything below is read out of the .pt file. Nothing is typed in by hand, so
rerunning this after a retrain is the whole maintenance procedure:

    python scripts/write_model_manifests.py

A metric the checkpoint does not carry is listed under `metrics.not_recorded`
rather than omitted, so a missing AUC reads as missing, not as forgotten.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cyberworld_v4.config import get_contract  # noqa: E402

MANIFEST_VERSION = "2.0.0"

#: Metrics a reader should be able to find for each model. Accuracy alone is
#: meaningless on a corpus that is ~82.5% Benign.
EXPECTED = {
    "branch_a": ["risk_auc", "risk_brier", "tech_macro_f1"],
    "branch_b": ["skill", "mse_model", "mse_persistence", "risk_mae_model", "risk_mae_zero"],
    "deepop": ["macro_f1", "token_acc", "acc_persistence", "acc_majority"],
}

SERVED = {
    "branch_a": ("saved_models/branch_a", "branch_a_lstm.pt", "branch_a.manifest.json"),
    "branch_b": ("saved_models/branch_b", "host_wdt.pt", "branch_b.manifest.json"),
    "deepop": ("saved_models/deepop", "cwa_forecast_decoder.pt", "deepop.manifest.json"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(v: Any) -> Any:
    """JSON-safe: NaN/inf become null, tensors and numpy scalars become floats."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and callable(v.item):
        try:
            v = v.item()
        except Exception:
            return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def temporal_contract(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    tc = ckpt.get("training_contract") or {}
    cfg = (ckpt.get("config") or {}).get("temporal") or {}
    src = cfg or tc or ckpt
    w = src.get("window_seconds", src.get("window_size_sec"))
    h, k = src.get("history_steps"), src.get("forecast_steps")
    # Checkpoints that predate the field were single-scale.
    fw = src.get("forecast_window_seconds", w)
    out = {
        "window_seconds": w,
        "history_steps": h,
        "forecast_steps": k,
        "forecast_step_seconds": fw,
        "history_seconds": (w * h) if (w and h) else None,
        "forecast_seconds": (fw * k) if (fw and k) else None,
    }
    auth = get_contract()
    out["matches_authoritative_contract"] = auth.matches(
        {"window_seconds": w, "history_steps": h, "forecast_steps": k,
         "forecast_window_seconds": fw}) if (w and h and k) else False
    out["authoritative_contract"] = auth.to_dict()
    return out


def metrics_block(name: str, ckpt: Dict[str, Any]) -> Dict[str, Any]:
    history = ckpt.get("epoch_history") or []
    best_epoch = ckpt.get("epoch")
    best = next((e for e in history if e.get("epoch") == best_epoch), None)
    validation = dict(ckpt.get("metrics") or {})
    if best:
        validation.update(best)
    for k in ("accuracy", "k_last_mse", "best_score"):
        if k in ckpt:
            validation.setdefault(k, ckpt[k])
    found = set(validation) | set(ckpt.get("test_metrics") or {})
    return {
        "selected_epoch": best_epoch,
        "validation": validation or None,
        "test": ckpt.get("test_metrics") or None,
        "baselines": ckpt.get("baselines") or None,
        "operating_point": ckpt.get("operating_point") or None,
        "not_recorded": [m for m in EXPECTED[name] if m not in found],
    }


def warnings_for(name: str, ckpt: Dict[str, Any], tcon: Dict[str, Any]) -> list:
    w = []
    if not tcon.get("matches_authoritative_contract"):
        w.append("Temporal contract differs from cyberworld_v4/config.py; the serving "
                 "adapter refuses this checkpoint unless CYBERWORLD_ALLOW_CONTRACT_MISMATCH=1.")
    cred = ckpt.get("credibility") or {}
    if not cred.get("checked"):
        w.append("No credibility verdict recorded; metrics are unvalidated.")
    elif not cred.get("credible", True):
        w.append("Marked NOT CREDIBLE by its own training run: "
                 + "; ".join(cred.get("problems") or []))
    stats = cred.get("stats") or {}
    tr, va = stats.get("train_technique_classes"), stats.get("val_technique_classes")
    if tr and va and va < tr:
        w.append(f"Validation split has {int(va)} technique classes vs {int(tr)} in training; "
                 f"per-class metrics cannot be measured for the missing classes.")
    m = ckpt.get("metrics") or {}
    if "tech_accuracy" in m and "tech_macro_f1" not in m:
        w.append("Only technique ACCURACY is recorded. The corpus is ~82.5% Benign, so "
                 "accuracy cannot visibly fail; quote macro F1 once a retrain records it.")
    # Architecture relative to docs/PAPER_CONFORMANCE.md.
    if name in ("branch_a", "deepop") and "arch" not in ckpt:
        w.append("Legacy architecture: trained before this model was aligned with its paper "
                 "(see docs/PAPER_CONFORMANCE.md). Loads via from_checkpoint; retrain for the "
                 "paper model.")
    if name == "branch_b":
        d_state = ckpt.get("d_state") or ckpt["wdt_state_dict"]["in_proj.weight"].shape[1]
        if int(d_state) == 12:
            w.append("Legacy world state: models the 12-D TGNE latent only, not the 27-D "
                     "state s(t) Branch A reads. Retrain for the current pipeline.")
    if name == "branch_b" and not ckpt.get("epoch_history"):
        w.append("No per-epoch history: whether this world model beats persistence "
                 "(copying the last embedding) was never measured for these weights.")
    return w


def build(name: str) -> Optional[Dict[str, Any]]:
    import torch

    folder, ckpt_file, manifest_file = SERVED[name]
    ckpt_path = REPO / folder / ckpt_file
    if not ckpt_path.exists():
        print(f"skip {name}: {ckpt_path} not found")
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    manifest_path = REPO / folder / manifest_file
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    embedded = ckpt.get("manifest") or {}
    tcon = temporal_contract(ckpt)

    out = {
        "manifest_version": MANIFEST_VERSION,
        "generated_by": "scripts/write_model_manifests.py",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_name": previous.get("model_name", name),
        "checkpoint_file": ckpt_file,
        "checkpoint_sha256": _sha256(ckpt_path),
        # Static layout of the module; unchanged by retraining, so carried over.
        "architecture": previous.get("architecture"),
        "temporal_contract": tcon,
        "metrics": metrics_block(name, ckpt),
        "credibility": ckpt.get("credibility") or {"checked": False},
        "risk_target": ckpt.get("risk_target") or (ckpt.get("training_contract") or {}).get("risk_target"),
        "provenance": {
            "git": embedded.get("git"),
            "created_utc": embedded.get("created_utc"),
            "command": embedded.get("command"),
            "dataset_sources": embedded.get("dataset_sources")
                or (ckpt.get("training_contract") or {}).get("sources"),
            "dependencies": embedded.get("dependencies"),
            "fingerprint": ckpt.get("fingerprint"),
        },
        "warnings": warnings_for(name, ckpt, tcon),
    }
    out = _clean(out)
    manifest_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {manifest_path.relative_to(REPO)}")
    return out


def main() -> int:
    for name in SERVED:
        build(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
