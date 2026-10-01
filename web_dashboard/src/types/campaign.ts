/**
 * Campaign / causal-graph view model.
 *
 * Built live by control_backend/correlation_service.py, which runs the
 * existing `correlation/` pipeline over the scored windows. That pipeline is
 * heuristic — every parameter is hand-set, nothing is learned — and the
 * payload says so in `method`.
 *
 *   Campaign      correlation/campaign_merge.py:AttackCampaign
 *   CampaignNode  correlation/graph_compaction.py:CompactedAlertNode
 *   Provenance    correlation/trajectory_assembler.py:Provenance
 *   lanes         control_backend/tactics.py:LANES
 */

export type Provenance = "OBSERVED" | "FORECAST";

/** A consolidated alert node surviving semantics skipping and compaction. */
export interface CampaignNode {
  node_id: number;
  host_ip: string;
  /** One of the kill-chain lanes — see `Campaign.lanes`. */
  coarse_category: string;
  technique_id: string;
  provenance: Provenance;
  start_time: number;
  end_time: number;
  hit_count: number;
  max_risk_score: number;
  mean_confidence: number;
}

/** A scored causal link between two compacted nodes. */
export interface CampaignEdge {
  src: number;
  dst: number;
  causality_score: number;
}

/** A reconstructed multi-step, multi-host attack campaign. */
export interface Campaign {
  campaign_id: number;
  involved_hosts: string[];
  root_cause_node_ids: number[];
  nodes: CampaignNode[];
  edges: CampaignEdge[];
  max_risk_score: number;
  start_time: number;
  end_time: number;
  duration_sec: number;
  attack_techniques: string[];
  has_forecast_components: boolean;
  /**
   * Kill-chain lane order for layout. Carried on the payload rather than
   * hardcoded in the view so the backend stays the source of truth.
   */
  lanes: string[];
  /** How the campaign was assembled. */
  method?: string;
}
