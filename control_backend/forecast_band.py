"""The uncertainty band served with each forecast step.

The band comes from the Branch B checkpoint's `forecast_risk_conformal`, fitted
per step on validation residuals by scripts/retrain_future_models_live.py. It
replaces a band the dashboard used to invent as risk +/- 20 * (DeepOP token
confidence), which measured nothing and widened as confidence ROSE.

No fitted band means no band: `risk_lower` / `risk_upper` are None and the
dashboard draws the point alone. A made-up interval is worse than none,
because it is believed.

Kept free of model imports so it can be tested without loading checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def forecast_band_halfwidths(ckpt: Dict[str, Any]) -> Optional[List[Optional[float]]]:
    """Per-step half-widths from a Branch B checkpoint; None when never fitted."""
    conf = ckpt.get("forecast_risk_conformal") or {}
    hw = conf.get("half_width_by_step")
    if not hw or all(h is None for h in hw):
        return None
    return [None if h is None or not math.isfinite(float(h)) else float(h) for h in hw]


def forecast_band(risk: float, halfwidths: Optional[List[Optional[float]]], step: int) -> Dict[str, Optional[float]]:
    """{'risk_lower', 'risk_upper'} for forecast step `step` (0-based), clipped to [0, 1]."""
    if not halfwidths or step >= len(halfwidths) or halfwidths[step] is None:
        return {"risk_lower": None, "risk_upper": None}
    h = halfwidths[step]
    return {"risk_lower": round(max(0.0, risk - h), 4),
            "risk_upper": round(min(1.0, risk + h), 4)}
