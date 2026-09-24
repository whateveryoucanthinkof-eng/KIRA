#!/usr/bin/env python3
"""Write the `<checkpoint>_config.json` that build_or_load_tgne_ta requires.

bita/train.py writes this only when a run finishes. Selecting an EARLIER epoch
(which scripts/select_best_encoder.py routinely does, because the two
objectives diverge) leaves that epoch's checkpoint without one, and
build_or_load_tgne_ta then falls back to defaults -- including
num_categories=4 when this corpus has 5, which fails to load with a shape
mismatch.

The values must match the run that produced the checkpoint. They are read from
the training log where possible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Run directly as a script (that is how the tests and the runbook invoke it),
# so the repo root has to be on the path before the import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_unification.tgne_features import SCHEMA_VERSION

DEFAULTS = {
    "n_layers": 1, "n_heads": 2, "dropout": 0.1, "use_memory": False,
    "message_dimension": 100, "memory_dimension": 9,
    "embedding_module_type": "graph_attention", "message_function": "identity",
    "aggregator_type": "bigru_transformer", "memory_updater_type": "gru",
    "edge_feat_dim": 12, "node_feat_dim": 12,
    "feature_schema_version": SCHEMA_VERSION,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--log", type=Path, default=Path("logs/tgne_final.out"))
    a = ap.parse_args()

    cfg = dict(DEFAULTS)

    # num_categories comes from the run, never a guess: the loader builds the
    # category head with it and a wrong value is a silent shape mismatch.
    text = a.log.read_text() if a.log.exists() else ""
    m = re.search(r"Detected coarse categories: \{([^}]*)\}", text)
    if m:
        cfg["num_categories"] = len(re.findall(r"\d+\s*:", m.group(1)))
    else:
        raise SystemExit(
            f"could not read 'Detected coarse categories' from {a.log}; "
            f"refusing to guess num_categories"
        )

    out = a.checkpoint.with_name(a.checkpoint.stem + "_config.json")
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")
    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
