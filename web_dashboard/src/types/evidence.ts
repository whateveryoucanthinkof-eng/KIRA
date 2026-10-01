/**
 * Evidence-level view models: the raw flows behind a verdict, and the full
 * state vector the models actually read. Both arrive on every scored window:
 *
 *   FlowRecord   control_backend/evidence.py:flow_records — the sensor's
 *                per-window flow export (telemetry/flow/flow_table.py, up to
 *                256 per 2 s window), TCP flag counts included, direction
 *                from the site's CIDRs. `on_path` marks flows touching the
 *                scored host.
 *   StateDim     control_backend/model_adapter.py:_explain_full — Input x
 *                Gradient over all 27 dims of the last timestep.
 */

export type FlowProtocol = "TCP" | "UDP" | "ICMP" | "OTHER";

/** TCP flag counts across all packets in the flow (flow_table.py). */
export interface FlowFlags {
  syn: number;
  ack: number;
  psh: number;
  rst: number;
  fin: number;
  urg: number;
}

export interface FlowRecord {
  /** Stable per window: `w<window>-<index>`. */
  id: string;
  window: number;
  /** Epoch microseconds of first packet — integer, within 2^53. */
  ts_us: number;
  src_ip: string;
  src_port: number;
  dst_ip: string;
  dst_port: number;
  protocol: FlowProtocol;
  fwd_bytes: number;
  bwd_bytes: number;
  fwd_packets: number;
  bwd_packets: number;
  duration_ms: number;
  flags: FlowFlags;
  /** Relative to the site's enterprise CIDRs. */
  direction: "inbound" | "outbound" | "internal";
  /** True when either endpoint is on the attributed attack path. */
  on_path: boolean;
}

/** One dimension of the 27-D state the models consume. */
export interface StateDim {
  index: number;
  /** H_emb_0 … H_emb_11, then the 15 host attributes. */
  feature: string;
  /** "TGNE Latent" | "Connectivity" | "Volume" | "Protocol" | "Timing" | "Rate" */
  group: string;
  /**
   * TGNE latent dims are unbounded and signed. Host attributes are clipped
   * to [0, 1] by data_unification/host_attributes.py.
   */
  value: number;
  /** Share of total |input x gradient| attribution, 0–1, sums to 1 across 27. */
  attribution: number;
}
