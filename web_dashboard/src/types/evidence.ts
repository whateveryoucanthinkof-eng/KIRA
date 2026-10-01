/**
 * Evidence-level view models: the raw flows behind a verdict, and the full
 * state vector the models actually read.
 *
 * Both exist in the pipeline today but are not yet forwarded over /ws:
 *
 *   FlowRecord   telemetry/state/state_builder.py writes up to 256 flow dicts
 *                per 2.0s window to the JSONL stream; flag counts come from
 *                telemetry/flow/flow_table.py (syn_count, ack_count, …), which
 *                aggregates the per-packet flags parsed in
 *                telemetry/capture/sniffer.py.
 *   StateDim     control_backend/model_adapter.py:_explain() computes
 *                Input x Gradient over all 27 dims of the last timestep, then
 *                sends only the top 8. This carries all 27.
 *
 * The demo stream populates both. Against a live backend the fields are
 * absent and the panels say so rather than rendering empty tables.
 */

export type FlowProtocol = "TCP" | "UDP" | "ICMP";

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
