/**
 * Incident triage model.
 *
 * An incident is a correlated group of alerts on one host, carried through an
 * analyst workflow rather than left as a flat log line. The backend does not
 * own this concept yet — `control_backend/` emits predictions and events, and
 * `correlation/` assembles campaigns, but nothing assigns, acknowledges or
 * closes. This is the contract to serve against when it does.
 *
 * Status transitions are the standard SOC ladder:
 *   new → triaging → contained → closed
 */

export type IncidentStatus = "new" | "triaging" | "contained" | "closed";

export const INCIDENT_STATUSES: IncidentStatus[] = ["new", "triaging", "contained", "closed"];

export interface Incident {
  /** Human reference, e.g. INC-0142. */
  id: string;
  host: string;
  hostLabel: string;
  /** Short title — usually the driving technique. */
  title: string;
  technique: string | null;
  tactic: string | null;
  /** Mirrors PredictionData.alert_level, lowercased. */
  severity: "critical" | "elevated" | "warning" | "nominal";
  status: IncidentStatus;
  /** epoch seconds */
  opened: number;
  lastSeen: number;
  peakRisk: number;
  alertCount: number;
  assignee: string | null;
  /** early_warning.lead_time_seconds at the time the incident opened. */
  leadTimeSeconds: number | null;
  campaignId: number | null;
  /** Analyst-visible audit trail. */
  notes: { at: number; text: string }[];
}

/** Queue ordering: unresolved first, then severity, then most recent. */
const SEV_RANK: Record<Incident["severity"], number> = {
  critical: 0,
  elevated: 1,
  warning: 2,
  nominal: 3,
};

const STATUS_RANK: Record<IncidentStatus, number> = {
  new: 0,
  triaging: 1,
  contained: 2,
  closed: 3,
};

export function rankIncidents(a: Incident, b: Incident): number {
  const s = STATUS_RANK[a.status] - STATUS_RANK[b.status];
  if (s !== 0) return s;
  const v = SEV_RANK[a.severity] - SEV_RANK[b.severity];
  if (v !== 0) return v;
  return b.lastSeen - a.lastSeen;
}
