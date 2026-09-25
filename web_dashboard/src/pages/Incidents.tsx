import { useMemo, useState } from "react";
import type { Incident, IncidentStatus } from "../types/incident";
import { INCIDENT_STATUSES, rankIncidents } from "../types/incident";
import {
  Btn,
  Chip,
  Data,
  Empty,
  Field,
  Meter,
  Micro,
  Panel,
  PanelBody,
  PanelHead,
  Readout,
  Square,
  sevColor,
} from "../design/primitives";

/**
 * Incident triage queue.
 *
 * Events is a log; this is the job. An incident is a correlated group of
 * alerts on one host carried through new → triaging → contained → closed,
 * with an assignee and an audit trail.
 *
 * Status changes are local state. The backend has no incident store yet —
 * `control_backend/` emits predictions and events, `correlation/` assembles
 * campaigns, and nothing acknowledges or closes. `src/types/incident.ts` is
 * the contract to serve against when that lands.
 */

interface IncidentsProps {
  incidents: Incident[];
  /** Applied on top of the stream so analyst actions survive re-renders. */
  overrides: Record<string, Partial<Incident>>;
  onUpdate: (id: string, patch: Partial<Incident>) => void;
  onOpenHost: (ip: string) => void;
}

const ANALYST = "a.rao";

function hhmm(epochSeconds: number): string {
  try {
    return new Date(epochSeconds * 1000).toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return "—";
  }
}

function ago(epochSeconds: number): string {
  const d = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (d < 60) return `${d}s`;
  if (d < 3600) return `${Math.floor(d / 60)}m`;
  return `${Math.floor(d / 3600)}h ${Math.floor((d % 3600) / 60)}m`;
}

function duration(from: number, to: number): string {
  const d = Math.max(0, Math.floor(to - from));
  if (d < 60) return `${d}s`;
  if (d < 3600) return `${Math.floor(d / 60)}m ${d % 60}s`;
  return `${Math.floor(d / 3600)}h ${Math.floor((d % 3600) / 60)}m`;
}

export default function Incidents({ incidents, overrides, onUpdate, onOpenHost }: IncidentsProps) {
  const [filter, setFilter] = useState<Set<IncidentStatus>>(new Set());
  const [sel, setSel] = useState<string | null>(null);

  const merged = useMemo(
    () => incidents.map((i) => ({ ...i, ...(overrides[i.id] ?? {}) })).sort(rankIncidents),
    [incidents, overrides]
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = { new: 0, triaging: 0, contained: 0, closed: 0 };
    for (const i of merged) c[i.status] = (c[i.status] ?? 0) + 1;
    return c;
  }, [merged]);

  const rows = filter.size ? merged.filter((i) => filter.has(i.status)) : merged;
  const selected = merged.find((i) => i.id === sel) ?? null;
  const open = merged.filter((i) => i.status !== "closed");

  /** Median time to contain, over incidents that actually reached contained. */
  const mttc = useMemo(() => {
    const done = merged.filter((i) => i.status === "contained" || i.status === "closed");
    if (!done.length) return null;
    const secs = done.map((i) => i.lastSeen - i.opened).sort((a, b) => a - b);
    return secs[Math.floor(secs.length / 2)];
  }, [merged]);

  function toggle(s: IncidentStatus) {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  function act(inc: Incident, status: IncidentStatus, text: string) {
    onUpdate(inc.id, {
      status,
      assignee: status === "triaging" ? (inc.assignee ?? ANALYST) : inc.assignee,
      notes: [...inc.notes, { at: Date.now() / 1000, text }],
    });
  }

  return (
    <div className="in">
      <div className="in-main sheet">
        {/* ── Queue stats ──────────────────────────────────────────── */}
        <Panel flush>
          <PanelHead
            title="Queue"
            note={`${open.length} open · ${merged.length} total`}
            aside={
              <div style={{ display: "flex", gap: "var(--s-2)" }}>
                {INCIDENT_STATUSES.map((s) => {
                  const on = filter.has(s);
                  const level = s === "new" ? "critical" : s === "triaging" ? "elevated" : s === "contained" ? "warning" : "nominal";
                  return (
                    <button
                      key={s}
                      onClick={() => toggle(s)}
                      aria-pressed={on}
                      className="t-micro"
                      style={{
                        padding: "3px var(--s-2)",
                        border: "var(--hair)",
                        borderLeft: `3px solid ${sevColor(level)}`,
                        background: on ? sevColor(level) : "transparent",
                        color: on ? "var(--ink-000)" : sevColor(level),
                        transition: "background 90ms steps(2,end), color 90ms steps(2,end)",
                      }}
                    >
                      {s} {counts[s] ?? 0}
                    </button>
                  );
                })}
              </div>
            }
          />
          <PanelBody>
            <div className="in-stats">
              <Readout label="Open" value={String(open.length)} level={open.length ? "elevated" : "nominal"} sub="requiring action" />
              <Readout
                label="Unassigned"
                value={String(open.filter((i) => !i.assignee).length)}
                sub="no analyst"
                level={open.some((i) => !i.assignee) ? "warning" : "nominal"}
              />
              <Readout
                label="Median contain"
                value={mttc != null ? duration(0, mttc) : "—"}
                sub="this shift"
              />
              <Readout
                label="Mean lead time"
                value={
                  merged.some((i) => i.leadTimeSeconds != null)
                    ? (
                        merged.reduce((s, i) => s + (i.leadTimeSeconds ?? 0), 0) /
                        merged.filter((i) => i.leadTimeSeconds != null).length
                      ).toFixed(1)
                    : "—"
                }
                unit="s"
                level="nominal"
                sub="ahead of milestone"
              />
            </div>
          </PanelBody>
        </Panel>

        {/* ── Queue table ──────────────────────────────────────────── */}
        <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 220 }}>
          <PanelHead title="Incidents" note={`${rows.length} shown`} />
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 92 }}>ID</th>
                  <th style={{ width: 96 }}>Status</th>
                  <th style={{ width: 88 }}>Severity</th>
                  <th style={{ width: 130 }}>Host</th>
                  <th>Title</th>
                  <th className="num" style={{ width: 76 }}>Peak</th>
                  <th className="num" style={{ width: 70 }}>Alerts</th>
                  <th style={{ width: 92 }}>Assignee</th>
                  <th style={{ width: 72 }}>Age</th>
                </tr>
              </thead>
              <tbody>
                {rows.length ? (
                  rows.map((i) => (
                    <tr
                      key={i.id}
                      className={i.id === sel ? "is-selected" : undefined}
                      onClick={() => setSel(i.id === sel ? null : i.id)}
                      style={{ cursor: "pointer", opacity: i.status === "closed" ? 0.6 : 1 }}
                    >
                      <td className="key" style={{ borderLeft: `3px solid ${sevColor(i.severity)}` }}>
                        {i.id}
                      </td>
                      <td>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
                          <Square status={i.status === "new" ? "critical" : i.status === "triaging" ? "warning" : "nominal"} live={i.status === "new"} />
                          {i.status}
                        </span>
                      </td>
                      <td style={{ color: sevColor(i.severity) }}>{i.severity}</td>
                      <td className="key">{i.hostLabel}</td>
                      <td className="key" title={i.title}>
                        {i.title}
                      </td>
                      <td className="num key" style={{ color: sevColor(i.severity) }}>
                        {i.peakRisk.toFixed(3)}
                      </td>
                      <td className="num">{i.alertCount.toLocaleString()}</td>
                      <td>{i.assignee ?? "—"}</td>
                      <td>{ago(i.opened)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} style={{ padding: 0 }}>
                      <Empty hint="Incidents open automatically when forecast risk crosses the operating threshold.">
                        {merged.length ? "No match for this filter" : "Queue clear"}
                      </Empty>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* ── Detail rail ──────────────────────────────────────────────── */}
      <div className="in-rail">
        <PanelHead
          title={selected ? selected.id : "Triage"}
          aside={selected ? <Chip level={selected.severity}>{selected.severity}</Chip> : undefined}
        />

        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {selected ? (
            <>
              <PanelBody>
                <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-1)" }}>
                  {selected.title}
                </div>
                <Data size="s" color="var(--paper-600)">
                  {[selected.technique, selected.tactic].filter(Boolean).join("  ·  ") || "unclassified"}
                </Data>

                <div style={{ marginTop: "var(--s-4)" }}>
                  <Micro style={{ marginBottom: "var(--s-2)" }}>Peak risk</Micro>
                  <Meter value={selected.peakRisk} level={selected.severity} height={6} />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: "var(--s-1)" }}>
                    <Data size="s" color="var(--paper-600)">
                      0.00
                    </Data>
                    <Data size="s" color={sevColor(selected.severity)}>
                      {selected.peakRisk.toFixed(3)}
                    </Data>
                  </div>
                </div>
              </PanelBody>

              {/* Actions — the workflow ladder */}
              <PanelHead title="Actions" />
              <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)" }}>
                <Btn
                  full
                  disabled={selected.status !== "new"}
                  sub={selected.assignee ? `assigned to ${selected.assignee}` : `assign to ${ANALYST}`}
                  onClick={() => act(selected, "triaging", `Acknowledged by ${ANALYST}.`)}
                >
                  Acknowledge
                </Btn>
                <Btn
                  full
                  disabled={selected.status === "contained" || selected.status === "closed"}
                  sub="Record the host as isolated"
                  onClick={() => act(selected, "contained", "Marked contained — host isolated.")}
                >
                  Mark contained
                </Btn>
                <Btn
                  full
                  danger
                  disabled={selected.status === "closed"}
                  sub="Resolve and remove from the open queue"
                  onClick={() => act(selected, "closed", "Closed.")}
                >
                  Close incident
                </Btn>
                <Btn full sub={selected.host} onClick={() => onOpenHost(selected.host)}>
                  Open host
                </Btn>
              </PanelBody>

              <PanelHead title="Detail" />
              <PanelBody>
                <Field label="Host" value={`${selected.hostLabel} · ${selected.host}`} />
                <Field label="Status" value={selected.status} />
                <Field label="Assignee" value={selected.assignee ?? "unassigned"} />
                <Field label="Opened" value={hhmm(selected.opened)} />
                <Field label="Last seen" value={hhmm(selected.lastSeen)} />
                <Field label="Duration" value={duration(selected.opened, selected.lastSeen)} />
                <Field label="Alerts" value={selected.alertCount.toLocaleString()} />
                {selected.leadTimeSeconds != null && (
                  <Field label="Lead time" value={`${selected.leadTimeSeconds.toFixed(1)}s`} color="var(--sev-nominal)" />
                )}
                {selected.campaignId != null && <Field label="Campaign" value={`#${selected.campaignId}`} />}
              </PanelBody>

              <PanelHead title="Activity" note={`${selected.notes.length}`} />
              <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)" }}>
                {selected.notes
                  .slice()
                  .reverse()
                  .map((n, i) => (
                    <div key={i} style={{ paddingBottom: "var(--s-2)", borderBottom: "var(--hair)" }}>
                      <Data size="s" color="var(--paper-600)" style={{ display: "block" }}>
                        {hhmm(n.at)}
                      </Data>
                      <div className="t-data-s" style={{ color: "var(--paper-400)", marginTop: 2, whiteSpace: "normal", lineHeight: 1.5 }}>
                        {n.text}
                      </div>
                    </div>
                  ))}
              </PanelBody>
            </>
          ) : (
            <>
              <PanelBody>
                <Readout
                  label="Open incidents"
                  value={String(open.length)}
                  scale="hero"
                  level={open.length ? "elevated" : "nominal"}
                  sub={open.length ? "select one to triage" : "queue clear"}
                />
              </PanelBody>

              <PanelHead title="By status" />
              <PanelBody>
                {INCIDENT_STATUSES.map((s) => (
                  <Field key={s} label={s} value={String(counts[s] ?? 0)} />
                ))}
              </PanelBody>

              <PanelHead title="Workflow" />
              <PanelBody>
                <div className="t-body" style={{ color: "var(--paper-400)" }}>
                  Incidents open automatically when forecast risk crosses the operating threshold, then
                  move new → triaging → contained → closed. Every transition is written to the incident&apos;s
                  activity trail.
                </div>
              </PanelBody>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
