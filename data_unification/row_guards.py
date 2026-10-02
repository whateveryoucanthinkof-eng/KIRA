"""Row-level validity guards shared by the CSV adapters.

Every rule here rejects a row the *upstream corpus* is wrong about, not a row we
dislike. Each is documented with the evidence that justifies it, because a guard
that silently drops real traffic is worse than the defect it fixes.

Measured on the corpora in this repository.
"""

from __future__ import annotations

import numpy as np

#: Nothing in these corpora predates 2016. CIC-2017 captures July 2017,
#: CIC-2018 February/March 2018, CTU-13 2011-2013. A stamp below this floor is
#: a parse failure or a CICFlowMeter bug, never real traffic.
#: 2010-01-01T00:00:00Z -- deliberately loose so CTU-13 (2011+) still passes.
MIN_PLAUSIBLE_EPOCH = 1262304000.0

#: 2030-01-01T00:00:00Z. Above this is a sentinel or an overflow.
MAX_PLAUSIBLE_EPOCH = 1893456000.0


def invalid_timestamp_mask(epoch_seconds: np.ndarray) -> np.ndarray:
    """Rows whose timestamp cannot be real.

    Catches three distinct upstream defects:

    * **NaT sentinels.** Casting NaT yields -9223372036854775808, which becomes
      -9.223e9 -- a plausible-looking negative epoch that slipped past an
      earlier `> 0` threshold guard.
    * **Blank padding rows.** One CIC-2017 file (Thursday WebAttacks) is 63%
      empty rows appended by a spreadsheet; they produced 288,602 phantom
      records.
    * **Epoch-1970 rows.** Fourteen CIC-2018 rows carry stamps like
      `10/01/1970 03:04:26`. These are ~2.3e4 epoch seconds, i.e. comfortably
      **positive**, so a `<= 0` check misses them entirely. One such row
      stretches a 2-second window grid across 48 years.
    """
    e = np.asarray(epoch_seconds, dtype=np.float64)
    return ~np.isfinite(e) | (e < MIN_PLAUSIBLE_EPOCH) | (e > MAX_PLAUSIBLE_EPOCH)


def tso_failure_mask(protocol: np.ndarray, dst_port: np.ndarray) -> np.ndarray:
    """Rows where CICFlowMeter's TCP-Segmentation-Offload handling failed.

    Signature: ``Protocol == 0 AND Dst Port == 0``, usually with a negative
    FlowDuration. The DistriNet errata for CSE-CIC-IDS2018 documents this as
    "causes a lot of flows to have protocol/src and dst port = 0". 94,181 rows
    across CIC-2018 match.

    Protocol 0 is HOPOPT, which does not appear as a flow protocol in these
    captures, so the combination is unambiguous. Requiring *both* fields keeps
    the rule narrow: a genuine flow to port 0 over a real protocol survives.
    """
    return (np.asarray(protocol) == 0) & (np.asarray(dst_port) == 0)


def blank_host_mask(src_ips: np.ndarray) -> np.ndarray:
    """Rows whose source host is a literal "nan" or empty string.

    Blank padding rows stringify to "nan" and would otherwise become phantom
    hosts with their own trajectories.
    """
    s = np.asarray(src_ips).astype(str)
    lowered = np.char.lower(np.char.strip(s))
    return np.isin(lowered, np.array(["nan", "", "none", "0"]))


def valid_row_mask(
    epoch_seconds: np.ndarray,
    protocol: np.ndarray,
    dst_port: np.ndarray,
    src_ips: np.ndarray,
) -> np.ndarray:
    """True for rows safe to emit as UnifiedFlowRecords."""
    bad = (
        invalid_timestamp_mask(epoch_seconds)
        | tso_failure_mask(protocol, dst_port)
        | blank_host_mask(src_ips)
    )
    return ~bad


def rejection_breakdown(
    epoch_seconds: np.ndarray,
    protocol: np.ndarray,
    dst_port: np.ndarray,
    src_ips: np.ndarray,
) -> dict:
    """Per-rule counts, so a drop is always attributable rather than mysterious."""
    return {
        "invalid_timestamp": int(invalid_timestamp_mask(epoch_seconds).sum()),
        "tso_failure": int(tso_failure_mask(protocol, dst_port).sum()),
        "blank_host": int(blank_host_mask(src_ips).sum()),
        "total_rows": int(len(np.asarray(epoch_seconds))),
    }
