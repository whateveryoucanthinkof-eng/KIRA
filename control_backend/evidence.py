"""
control_backend/evidence.py
The evidence behind a verdict, in the shape the console lists it.

  flow_records()   the sensor's flow snapshot for the window (telemetry/flow/
                   flow_table.py:snapshot_flows), with direction from the site
                   CIDRs. Display only: the model reads the same flows through
                   flows_from_span_dicts(), not through this.
  state_vector()   all 27 dimensions Branch A read for the last timestep, with
                   their Input x Gradient share (model_adapter._explain_full).
"""

from typing import Any, Dict, List, Optional, Sequence

from control_backend.schema import FlowFlags, FlowRecordOut, StateDim

_PROTOCOLS = {6: "TCP", 17: "UDP", 1: "ICMP"}
#: Flows listed per window. The sensor already caps its export at 256.
MAX_FLOWS = 256


def _direction(site, src: str, dst: str) -> str:
    src_ext = site.classify_ip(src) == "external"
    dst_ext = site.classify_ip(dst) == "external"
    if src_ext and not dst_ext:
        return "inbound"
    if dst_ext and not src_ext:
        return "outbound"
    return "internal"


def flow_records(
    raw_flows: Sequence[Dict[str, Any]],
    window_id: int,
    target_ip: Optional[str],
    site=None,
) -> List[FlowRecordOut]:
    """The window's exported flows, first-packet order.

    `on_path` marks flows touching the host scored this window -- the only
    host a verdict is about. It is not a predicted attack path.
    """
    if site is None:
        from control_backend.site_config import get_site_config
        site = get_site_config()
    ordered = sorted(raw_flows or [], key=lambda f: float(f.get("start_time", 0.0)))[:MAX_FLOWS]
    out: List[FlowRecordOut] = []
    for i, f in enumerate(ordered):
        src, dst = str(f.get("src_ip", "")), str(f.get("dst_ip", ""))
        start, end = float(f.get("start_time", 0.0)), float(f.get("end_time", 0.0))
        out.append(FlowRecordOut(
            id=f"w{window_id}-{i}",
            window=int(window_id),
            ts_us=int(round(start * 1e6)),
            src_ip=src,
            src_port=int(f.get("src_port", 0)),
            dst_ip=dst,
            dst_port=int(f.get("dst_port", 0)),
            protocol=_PROTOCOLS.get(int(f.get("protocol", 0) or 0), "OTHER"),
            fwd_bytes=int(f.get("fwd_bytes", 0)),
            bwd_bytes=int(f.get("bwd_bytes", 0)),
            fwd_packets=int(f.get("fwd_packets", 0)),
            bwd_packets=int(f.get("bwd_packets", 0)),
            duration_ms=round(max(0.0, end - start) * 1000.0, 3),
            flags=FlowFlags(**(f.get("flags") or {})),
            direction=_direction(site, src, dst),
            on_path=bool(target_ip) and target_ip in (src, dst),
        ))
    return out


def state_vector(
    names: Sequence[str],
    groups: Dict[str, str],
    values: Sequence[float],
    attributions: Sequence[float],
) -> List[StateDim]:
    return [
        StateDim(
            index=i,
            feature=name,
            group=groups.get(name, "General"),
            value=round(float(values[i]), 6),
            attribution=round(float(attributions[i]), 6),
        )
        for i, name in enumerate(names)
    ]
