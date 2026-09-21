"""
Scientific Split Manager -- reads the FROZEN capture-level split.

## What this replaced, and why it mattered

The previous version hardcoded its own train/val/test assignment *and* its own
dataset paths. Both were wrong:

* Every path was unreachable on this machine, including a literal Windows path
  ``C:\\SIH_DATA\\dump\\cic2018(csv+pcap+logs)raw\\CSV``. Each lookup was
  guarded by ``if os.path.exists(...)``, so nothing raised -- all three splits
  simply returned **zero records**. The standalone Branch B and DeepOP trainers
  were training on nothing, silently.
* Its assignment contradicted ``splits.lock.json``. It put CIC-2017 Wednesday
  in *train* where the lock puts it in *test*, and CIC-2017 Friday-Morning in
  *val* where the lock puts it in *train*. Had the paths worked, that would
  have leaked test data into training.

The split is now read from ``data_unification/splits.lock.json`` -- one frozen,
capture-level, attack-fraction-stratified 4:1:1 assignment, the single thing
that must never drift if results are to stay comparable across retrains.
Paths come from ``data_unification/dataset_paths.py``.

## Leakage properties

Splitting is by **capture**, never by row. A capture is one contiguous
recording, so no host trajectory, no time window and no flow can appear on both
sides of a split boundary. This is the property row-level shuffling destroys.

Warden is gone entirely -- that corpus was dropped from the project.
"""

from __future__ import annotations

import logging
import os
from itertools import islice
from typing import Dict, Iterator, List, Optional

from data_unification.cic2017_adapter import CIC2017Adapter
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.dataset_paths import ROOT, missing_roots, resolve
from data_unification.split_policy import Split, load_lock
from data_unification.unified_schema import UnifiedFlowRecord

logger = logging.getLogger(__name__)


class EmptySplitError(RuntimeError):
    """A split resolved to no records. Always a defect, never a valid state."""


class ScientificSplitManager:
    """Serves records for a split, exactly as frozen in splits.lock.json."""

    def __init__(self, *, strict: bool = True):
        """`strict` makes an empty split raise instead of returning []. Keep it
        on: silently training on nothing is the failure this class was rewritten
        to make impossible."""
        self.strict = strict
        self.assignment: Dict[str, Dict[str, str]] = load_lock()
        self._adapters = {
            "CIC2017": CIC2017Adapter(),
            "CIC2018": CIC2018Adapter(),
            "CTU13": CTU13Adapter(),
        }
        missing = missing_roots()
        if missing:
            logger.warning(
                "dataset roots not found (their captures will be skipped): %s", missing
            )

    # ------------------------------------------------------------------ core

    def captures_in(self, split: Split) -> List[tuple]:
        """[(dataset, capture_name)] assigned to `split`, PCAP2018 excluded.

        PCAP captures are frozen in the same lock but are consumed through
        `data_unification/pcap_bridge.py`, not as flow CSVs.
        """
        out = []
        for dataset, mapping in sorted(self.assignment.items()):
            if dataset == "PCAP2018":
                continue
            for capture, assigned in sorted(mapping.items()):
                if assigned == split:
                    out.append((dataset, capture))
        return out

    def _parse_all(self, dataset: str, capture: str) -> Iterator:
        """Every record in a capture, unbounded."""
        path = resolve(dataset, capture)
        if not path.exists():
            logger.warning("%s/%s missing at %s -- skipped", dataset, capture, path)
            return iter(())
        adapter = self._adapters[dataset]
        if dataset == "CTU13":
            return adapter.parse_netflow_csv(str(path), max_rows=None)
        return adapter.parse_file(str(path), max_rows=None)

    def _parse(
        self, dataset: str, capture: str, max_rows: Optional[int], stride: int = 1
    ) -> Iterator:
        """Records from one capture, subsampled WITHOUT prefix bias.

        Passing `max_rows` straight to the adapter takes the first N rows of the
        file, and these captures are chronological: the morning is benign and
        the attacks run later in the day. Measured on the frozen val split at
        max_per_source=2000, a prefix sample produced **0% attack** -- a
        validation set with no positives, useless for early stopping and
        silently so.

        `stride` walks the whole capture and keeps every k-th record, so the
        sample spans the full day and preserves the label mix.
        """
        it = self._parse_all(dataset, capture)
        if stride > 1:
            it = islice(it, 0, None, stride)
        if max_rows is not None:
            it = islice(it, max_rows)
        return it

    def records_for(
        self,
        split: Split,
        max_per_source: Optional[int] = None,
        stride: int = 1,
    ) -> List[UnifiedFlowRecord]:
        """Every record assigned to `split`, in a stable capture order.

        `stride > 1` samples across the whole capture instead of taking a
        chronological prefix. Use it whenever `max_per_source` is set.
        """
        if max_per_source is not None and stride == 1:
            logger.warning(
                "split=%s is capped at %d records per capture with stride=1: this "
                "takes a chronological PREFIX, and these captures are benign in the "
                "morning. Pass stride>1 for a label-representative sample.",
                split, max_per_source,
            )

        records: List[UnifiedFlowRecord] = []
        per_capture: Dict[str, int] = {}

        for dataset, capture in self.captures_in(split):
            before = len(records)
            records.extend(self._parse(dataset, capture, max_per_source, stride))
            per_capture[f"{dataset}/{capture}"] = len(records) - before

        n_attack = sum(1 for r in records if r.is_attack)
        logger.info(
            "split=%s captures=%d records=%d attack=%d (%.2f%%)",
            split, len(per_capture), len(records), n_attack,
            100.0 * n_attack / max(1, len(records)),
        )
        if records and n_attack == 0:
            logger.error(
                "split=%s has ZERO attack records. Early stopping and every "
                "threshold on it are meaningless. Check stride/max_per_source.",
                split,
            )
        if not records:
            msg = (
                f"split '{split}' resolved to ZERO records from "
                f"{len(self.captures_in(split))} frozen captures. Roots: "
                f"{ {k: str(v) for k, v in ROOT.items()} }. This is the exact "
                f"failure the previous hardcoded-path split manager hid."
            )
            if self.strict:
                raise EmptySplitError(msg)
            logger.error(msg)
        return records

    def counts_for(self, split: Split, max_per_source: Optional[int] = None,
                   stride: int = 1) -> Dict[str, int]:
        """Per-capture record counts, for auditing a split without holding it."""
        out: Dict[str, int] = {}
        for dataset, capture in self.captures_in(split):
            out[f"{dataset}/{capture}"] = sum(
                1 for _ in self._parse(dataset, capture, max_per_source, stride)
            )
        return out

    # -------------------------------------------------------------- legacy API

    def get_train_records(self, max_per_source: int = 1500, stride: int = 20) -> List[UnifiedFlowRecord]:
        return self.records_for("train", max_per_source, stride)

    def get_val_records(self, max_per_source: int = 500, stride: int = 20) -> List[UnifiedFlowRecord]:
        return self.records_for("val", max_per_source, stride)

    def get_heldout_test_records(self, max_per_source: int = 1500, stride: int = 20) -> List[UnifiedFlowRecord]:
        """The held-out test split. Score it ONCE, at the end, on the restored
        best checkpoint -- never for model selection."""
        return self.records_for("test", max_per_source)


_GLOBAL_SPLIT_MANAGER: Optional[ScientificSplitManager] = None


def get_split_manager() -> ScientificSplitManager:
    global _GLOBAL_SPLIT_MANAGER
    if _GLOBAL_SPLIT_MANAGER is None:
        _GLOBAL_SPLIT_MANAGER = ScientificSplitManager()
    return _GLOBAL_SPLIT_MANAGER
