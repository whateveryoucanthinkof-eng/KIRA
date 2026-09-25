"""Blind-spot accounting for the live sensor stream.

Two ways a window can fail to be scored without anything noticing:

  * it never arrives -- the sensor's state recorder dropped it because its
    queue was full, which shows up here as a gap in window_id;
  * it arrives incomplete -- the kernel dropped frames because the Python
    capture loop could not keep up, reported by the sensor as
    record["capture"]["kernel_drops"].

A volumetric flood produces both, and to the models either looks like quiet
traffic. Kept free of model imports so it can be tested on its own.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("antigravity.capture_accounting")


class CaptureAccounting:
    def __init__(self) -> None:
        self.last_window_id: Optional[int] = None
        self.windows_missed = 0
        self.sensor_kernel_drops = 0
        self.incomplete_windows = 0

    def reset_sequence(self) -> None:
        """A new sensor process numbers its windows from 0 again."""
        self.last_window_id = None

    def observe(self, record: Dict[str, Any]) -> Dict[str, int]:
        """Account for one state record; returns what this record revealed."""
        missed_now = 0
        wid = record.get("window_id")
        if isinstance(wid, int) and not isinstance(wid, bool):
            if self.last_window_id is not None and wid > self.last_window_id + 1:
                missed_now = wid - self.last_window_id - 1
                self.windows_missed += missed_now
                logger.warning(
                    "Sensor window gap: %d window(s) between #%d and #%d were never "
                    "scored (%d missed in total).", missed_now, self.last_window_id,
                    wid, self.windows_missed)
            if self.last_window_id is None or wid > self.last_window_id:
                self.last_window_id = wid

        cap = record.get("capture") or {}
        drops = int(cap.get("kernel_drops", 0) or 0)
        if drops > 0:
            self.sensor_kernel_drops += drops
            self.incomplete_windows += 1
            logger.warning(
                "Window #%s was built from an incomplete capture: the kernel dropped "
                "%d frame(s) (%.1f%%). Its flows and host attributes undercount.",
                wid, drops, 100.0 * float(cap.get("kernel_drop_ratio", 0.0) or 0.0))
        return {"windows_missed": missed_now, "kernel_drops": drops}

    def status(self) -> Dict[str, int]:
        return {
            "sensorKernelDrops": self.sensor_kernel_drops,
            "incompleteWindows": self.incomplete_windows,
            "windowsMissed": self.windows_missed,
        }
