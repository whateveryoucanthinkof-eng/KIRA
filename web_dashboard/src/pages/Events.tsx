import { useMemo, useState } from "react";
import type { AttackEvent, EventSeverity, NetworkEvent } from "../api/types";
import { Chip, Data, Empty, Micro, Panel, PanelHead, sevColor } from "../design/primitives";

interface EventsProps {
  events: NetworkEvent[];
  attackEvents: AttackEvent[];
  logLines: string[];
}

const SEVERITIES: EventSeverity[] = ["critical", "error", "warning", "info"];

function hhmmss(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return ts;
  }
}

function ymd(ts: string): string {
  try {
    return new Date(ts).toISOString().slice(0, 10);
  } catch {
    return "";
  }
}

/** Log severity read off the line itself — the backend streams raw text. */
function logColor(line: string): string {
  if (/CRITICAL|ALERT|FATAL/.test(line)) return "var(--sev-critical)";
  if (/ERROR|DENY|BLOCK|FAIL/.test(line)) return "var(--sev-elevated)";
  if (/WARN/.test(line)) return "var(--sev-warning)";
  if (/INFO/.test(line)) return "var(--paper-400)";
  return "var(--paper-600)";
}

export default function Events({ events, attackEvents, logLines }: EventsProps) {
  const [filter, setFilter] = useState<Set<EventSeverity>>(new Set());
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const counts = useMemo(() => {
    const c: Record<string, number> = { critical: 0, error: 0, warning: 0, info: 0 };
    for (const e of events) c[e.severity] = (c[e.severity] ?? 0) + 1;
    return c;
  }, [events]);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return events.filter((e) => {
      if (filter.size && !filter.has(e.severity)) return false;
      if (!q) return true;
      return [e.category, e.source, e.destination, e.message, e.raw].filter(Boolean).join(" ").toLowerCase().includes(q);
    });
  }, [events, filter, query]);

  const detail = rows.find((r) => r.id === selected) ?? null;

  function toggle(s: EventSeverity) {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  return (
    <div className="ev">
      {/* ── Table ────────────────────────────────────────────────────── */}
      <Panel clip className="ev-table">
        <PanelHead
          title="Events"
          note={`${rows.length} of ${events.length}`}
          aside={
            <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)" }}>
              {/* Severity filters — pressed state inverts */}
              {SEVERITIES.map((s) => {
                const on = filter.has(s);
                return (
                  <button
                    key={s}
                    onClick={() => toggle(s)}
                    aria-pressed={on}
                    className="t-micro"
                    style={{
                      padding: "3px var(--s-2)",
                      border: "var(--hair)",
                      borderLeft: `3px solid ${sevColor(s)}`,
                      background: on ? sevColor(s) : "transparent",
                      color: on ? "var(--ink-000)" : sevColor(s),
                      transition: "background 90ms steps(2,end), color 90ms steps(2,end)",
                    }}
                  >
                    {s} {counts[s] ?? 0}
                  </button>
                );
              })}

              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="filter…"
                className="t-data-s"
                style={{
                  width: 168,
                  padding: "4px var(--s-2)",
                  background: "var(--ink-000)",
                  border: "var(--hair)",
                  color: "var(--paper-000)",
                  outline: "none",
                }}
              />

              {attackEvents.length > 0 && <Chip level="critical">{attackEvents.length} active</Chip>}
            </div>
          }
        />

        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 92 }}>Date</th>
                <th style={{ width: 84 }}>Time</th>
                <th style={{ width: 96 }}>Severity</th>
                <th style={{ width: 116 }}>Category</th>
                <th style={{ width: 140 }}>Source</th>
                <th style={{ width: 140 }}>Destination</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {rows.length ? (
                rows.map((e) => (
                  <tr
                    key={e.id}
                    className={e.id === selected ? "is-selected" : undefined}
                    onClick={() => setSelected(e.id === selected ? null : e.id)}
                    style={{ cursor: "pointer" }}
                  >
                    <td style={{ borderLeft: `3px solid ${sevColor(e.severity)}` }}>{ymd(e.timestamp)}</td>
                    <td className="key">{hhmmss(e.timestamp)}</td>
                    <td style={{ color: sevColor(e.severity) }}>{e.severity}</td>
                    <td>{e.category}</td>
                    <td className="key">{e.source}</td>
                    <td>{e.destination ?? "—"}</td>
                    <td className="key" title={e.message}>
                      {e.message}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} style={{ padding: 0 }}>
                    <Empty
                      hint={
                        events.length
                          ? "No event matches the current filter."
                          : "Alerts, commands and attack stages are recorded here as the stream produces them."
                      }
                    >
                      {events.length ? "No matches" : "No events yet"}
                    </Empty>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Selected-row detail — the raw record, verbatim */}
        {detail && (
          <div style={{ borderTop: "var(--hard)", padding: "var(--s-3)", background: "var(--ink-000)", flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)", marginBottom: "var(--s-2)" }}>
              <Chip level={detail.severity}>{detail.severity}</Chip>
              <Micro>{detail.category}</Micro>
              <Data size="s" color="var(--paper-600)">
                {detail.timestamp}
              </Data>
            </div>
            <Data size="s" style={{ display: "block", whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
              {detail.raw ?? detail.message}
            </Data>
          </div>
        )}
      </Panel>

      {/* ── Console ──────────────────────────────────────────────────── */}
      <Panel clip className="ev-log">
        <PanelHead title="Console" note={logLines.length ? `${logLines.length} lines` : undefined} />
        <div className="log">
          {logLines.length ? (
            logLines.map((line, i) => (
              <div key={i} className="log-line" style={{ color: logColor(line) }}>
                {line}
              </div>
            ))
          ) : (
            <span className="log-line">— idle —</span>
          )}
        </div>
      </Panel>
    </div>
  );
}
