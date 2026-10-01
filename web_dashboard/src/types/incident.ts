/**
 * Incident triage model.
 *
 * An incident is a correlated group of model alerts on one host
 * (control_backend/correlation_service.py): it opens on a host's first alert,
 * or on one more than the evidence horizon after its last. The server never
 * acknowledges or closes one; it marks an incident contained only while an
 * isolation is recorded for its host. Assignment, notes and status changes
 * are the analyst's, held in the console.
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

/**
 * An analyst's edits on top of the stream. Status and assignee replace the
 * stream's; notes are appended to its trail. `since` is the stream's alert
 * count when the first edit was made — an incident whose count falls below it
 * has restarted (the same host attacked again), and the old edits no longer
 * apply to it.
 */
export interface IncidentEdit {
  status?: IncidentStatus;
  assignee?: string | null;
  notes?: Incident["notes"];
  since?: number;
}

/** The stream's incident with the analyst's edits applied, if they still belong to it. */
export function applyEdit(i: Incident, e: IncidentEdit | undefined): Incident {
  if (!e || (e.since != null && i.alertCount < e.since)) return i;
  return {
    ...i,
    status: e.status ?? i.status,
    assignee: e.assignee !== undefined ? e.assignee : i.assignee,
    notes: e.notes?.length ? [...i.notes, ...e.notes].sort((a, b) => a.at - b.at) : i.notes,
  };
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
