/**
 * TGNE temporal attention.
 *
 * bita/model/temporal_attention.py: each host's query attends over its
 * sampled temporal neighbours (n_neighbors=20, 2 heads); the key is
 * [neighbour memory ; 12 edge features ; time encoding]. Collapsing the
 * neighbour slots by host gives a host × host matrix α — rows are the host
 * attending, columns the neighbour attended to, each row summing to 1.
 *
 * Saliency is Input × Gradient on the attention logit over the key's inputs:
 * the 12 edge features by their schema names (data_unification/
 * tgne_features.py:EDGE_FEATURE_NAMES), the time encoding and the
 * neighbour's memory. Contributions sum from the baseline logit to the cell's
 * logit.
 *
 * Not yet on /ws: the adapter extracts the TGNE embedding but not the
 * layer's weights. The demo stream carries this shape.
 */

export type AttentionRole = "external" | "dmz" | "server" | "workstation";

export interface AttentionNode {
  ip: string;
  name: string;
  role: AttentionRole;
}

export interface SaliencyTerm {
  feature: string;
  group: "edge" | "time" | "memory";
  /** The input's normalised value on this edge. */
  value: number;
  /** Signed contribution to the attention logit. */
  contribution: number;
}

export interface AttentionMatrix {
  layer: string;
  heads: number;
  neighbors: number;
  nodes: AttentionNode[];
  /** [query][neighbour]; rows sum to 1, the diagonal is 0. */
  alpha: number[][];
  /** Per-head weights; α is their mean, as nn.MultiheadAttention averages. */
  alpha_heads: number[][][];
  logits: number[][];
  baseline_logit: number;
  /** [query][neighbour] → terms, largest |contribution| first. Empty on the diagonal. */
  saliency: SaliencyTerm[][][];
}
