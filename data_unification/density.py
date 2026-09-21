"""Full data density is the default, and thinning it requires saying so out loud.

## Why this module exists

Training data kept getting silently thinned in this pipeline, in eleven
different places, by three different mechanisms:

* **row caps** -- `max_per_source`, `rows_per_file`, `max_rows`: take the first
  N rows of a capture. These captures are chronological and benign in the
  morning, so a prefix cap is also a *label* bias, not just less data. Measured
  on the frozen val split: a 2,000-row cap produced **0% attack**.
* **strides** -- keep every Nth record. `stride=20` discards 95%.
* **structural truncation** -- `snapshot_flows(max_flows=256)` silently drops
  every flow past the 256th for a host in a window, and
  `max_packets_per_host=20000` stops reading a capture partway.

Each was individually defensible as a smoke-test or memory workaround. Together
they meant no run was ever on the full corpus, and the caps outlived the
reasons for them. The memory pressure that justified them turned out to be a
single 13.91 GiB pandas call in the clock-detection pre-pass, plus a
293-bytes-per-edge Python adjacency list -- both since fixed.

## The rule

**Every default in the training path is now full density.** A cap is not
refused outright -- smoke tests legitimately need one -- but it must be
declared, and a run that declares one is marked as non-full-density so its
numbers can never be mistaken for a real result.

Set `CYBERWORLD_ALLOW_SUBSAMPLING=1` to permit caps. Without it,
`require_full_density()` raises.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

ENV_FLAG = "CYBERWORLD_ALLOW_SUBSAMPLING"


def subsampling_allowed() -> bool:
    return os.environ.get(ENV_FLAG, "") in ("1", "true", "True", "yes")


def active_caps(**caps: Any) -> Dict[str, Any]:
    """The subset of `caps` that would actually thin the data.

    A cap is inactive when it is None (no limit) or, for a stride, 1.
    """
    out: Dict[str, Any] = {}
    for name, value in caps.items():
        if value is None:
            continue
        if "stride" in name and int(value) <= 1:
            continue
        if "stride" not in name and value is False:
            continue
        out[name] = value
    return out


def require_full_density(context: str, **caps: Any) -> None:
    """Raise unless this run reads every record of every capture it opens.

    Call it once, at the top of a trainer, with every cap that trainer honours.
    """
    active = active_caps(**caps)
    if not active:
        logger.info("%s: FULL DENSITY -- no row cap, no stride, no truncation", context)
        return

    msg = (
        f"{context} would run on thinned data: {active}. Full density is the "
        f"default and the requirement here. A row cap takes a chronological "
        f"PREFIX of a capture, which biases labels as well as reducing volume "
        f"(a 2,000-row cap on the frozen val split measured 0% attack). "
        f"If this is a deliberate smoke test, set {ENV_FLAG}=1 -- and do not "
        f"report the resulting numbers as a result."
    )
    if subsampling_allowed():
        logger.warning("SUBSAMPLED RUN, NOT A RESULT. %s", msg)
        return
    raise ValueError(msg)
