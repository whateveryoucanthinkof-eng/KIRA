import { useState, useRef, useEffect } from "react";
import type { NetworkEvent, EventSeverity } from "../api/types";
import { SeverityBadge } from "../components/shared/StatusBadge";

interface EventsProps {
  events: NetworkEvent[];
  logLines: string[];
}

const ALL_CATEGORIES = ["all", "intrusion", "anomaly", "traffic", "auth", "firewall", "system", "prediction"];

function fmtTs(ts: string): string {
  try {
    return new Date(ts).toLocaleString("en-US", {
      hour12: false,
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

const logLineColor = (line: string): string => {
  if (line.includes("CRITICAL") || line.includes("ALERT")) return "var(--color-status-red)";
  if (line.includes("WARN")) return "var(--color-status-amber)";
  if (line.includes("ERROR") || line.includes("DENY") || line.includes("BLOCK")) return "#e57373";
  if (line.includes("INFO")) return "var(--color-text-secondary)";
  return "var(--color-text-muted)";
};

export default function Events({ events, logLines }: EventsProps) {
  const [search, setSearch] = useState("");
  const [severity, setSeverity] = useState<EventSeverity | "all">("all");
  const [category, setCategory] = useState("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const filtered = events.filter((ev) => {
    if (severity !== "all" && ev.severity !== severity) return false;
    if (category !== "all" && ev.category !== category) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        ev.message.toLowerCase().includes(q) ||
        ev.source.toLowerCase().includes(q) ||
        (ev.destination?.toLowerCase().includes(q) ?? false) ||
        ev.category.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="events-page">
      {/* Events table */}
      <div className="panel panel-clipped events-table-panel">
        {/* Filters */}
        <div
          style={{
            padding: "10px 12px",
            borderBottom: "1px solid var(--color-border)",
            display: "flex",
            gap: 8,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <span className="panel-title">Events</span>
          <span
            style={{
              fontSize: 10,
              fontFamily: "var(--font-mono)",
              color: "var(--color-text-muted)",
              background: "var(--color-base)",
              padding: "2px 7px",
              borderRadius: 4,
            }}
          >
            {filtered.length} / {events.length}
          </span>

          <div style={{ flex: 1 }} />

          {/* Search */}
          <input
            type="text"
            placeholder="Search events…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              padding: "4px 10px",
              borderRadius: 4,
              border: "1px solid var(--color-border)",
              fontSize: 12,
              fontFamily: "var(--font-sans)",
              color: "var(--color-text-primary)",
              background: "var(--color-surface)",
              outline: "none",
              width: 200,
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--color-status-blue)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--color-border)")}
          />

          {/* Severity filter */}
          <div style={{ display: "flex", gap: 4 }}>
            {(["all", "critical", "error", "warning", "info"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSeverity(s)}
                style={{
                  padding: "3px 9px",
                  borderRadius: 4,
                  border: "1px solid var(--color-border)",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.04em",
                  cursor: "pointer",
                  background: severity === s ? "var(--color-text-primary)" : "transparent",
                  color: severity === s ? "white" : "var(--color-text-secondary)",
                  textTransform: "uppercase",
                  transition: "background 0.1s",
                }}
              >
                {s}
              </button>
            ))}
          </div>

          {/* Category filter */}
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--color-border)",
              fontSize: 11,
              background: "var(--color-surface)",
              color: "var(--color-text-secondary)",
              cursor: "pointer",
              outline: "none",
            }}
          >
            {ALL_CATEGORIES.map((c) => (
              <option key={c} value={c}>{c === "all" ? "All categories" : c}</option>
            ))}
          </select>
        </div>

        {/* Table */}
        <div style={{ overflow: "auto", flex: 1 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ background: "var(--color-base)", position: "sticky", top: 0, zIndex: 1 }}>
                {["Severity", "Time", "Category", "Source", "Destination", "Message"].map((h) => (
                  <th
                    key={h}
                    style={{
                      padding: "6px 12px",
                      textAlign: "left",
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.05em",
                      textTransform: "uppercase",
                      color: "var(--color-text-muted)",
                      borderBottom: "1px solid var(--color-border)",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ padding: "24px 12px", textAlign: "center", color: "var(--color-text-muted)", fontSize: 12 }}>
                    No events match the current filter.
                  </td>
                </tr>
              )}
              {filtered.map((ev) => (
                <>
                  <tr
                    key={ev.id}
                    style={{
                      borderBottom: expandedId === ev.id ? "none" : "1px solid var(--color-border)",
                      cursor: "pointer",
                      background: expandedId === ev.id ? "var(--color-base)" : "transparent",
                      transition: "background 0.1s",
                    }}
                    onClick={() => setExpandedId(expandedId === ev.id ? null : ev.id)}
                    onMouseEnter={(e) => {
                      if (expandedId !== ev.id)
                        (e.currentTarget as HTMLTableRowElement).style.background = "var(--color-base)";
                    }}
                    onMouseLeave={(e) => {
                      if (expandedId !== ev.id)
                        (e.currentTarget as HTMLTableRowElement).style.background = "transparent";
                    }}
                  >
                    <td style={{ padding: "6px 12px" }}>
                      <SeverityBadge severity={ev.severity} />
                    </td>
                    <td style={{ padding: "6px 12px", fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>
                      {fmtTs(ev.timestamp)}
                    </td>
                    <td style={{ padding: "6px 12px", fontSize: 11, color: "var(--color-text-secondary)", whiteSpace: "nowrap" }}>
                      {ev.category}
                    </td>
                    <td style={{ padding: "6px 12px", fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-secondary)", whiteSpace: "nowrap" }}>
                      {ev.source}
                    </td>
                    <td style={{ padding: "6px 12px", fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>
                      {ev.destination ?? "—"}
                    </td>
                    <td style={{ padding: "6px 12px", color: "var(--color-text-secondary)", maxWidth: 400 }}>
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>
                        {ev.message}
                      </span>
                    </td>
                  </tr>
                  {expandedId === ev.id && (
                    <tr key={`${ev.id}-exp`} style={{ borderBottom: "1px solid var(--color-border)", background: "var(--color-base)" }}>
                      <td colSpan={6} style={{ padding: "0 12px 10px 12px" }}>
                        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: ev.raw ? 8 : 0 }}>
                          {ev.protocol && <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>Protocol: <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{ev.protocol}</span></span>}
                          {ev.port && <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>Port: <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{ev.port}</span></span>}
                          <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>ID: <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{ev.id}</span></span>
                        </div>
                        {ev.raw && (
                          <pre style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-secondary)", background: "#f8f9fa", border: "1px solid var(--color-border)", borderRadius: 4, padding: "6px 10px", margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                            {ev.raw}
                          </pre>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Log console */}
      <div className="panel panel-clipped events-log-panel">
        <div className="panel-header">
          <span className="panel-title">System Log</span>
          <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
            live · {logLines.length} lines
          </span>
        </div>
        <div
          ref={logRef}
          style={{
            flex: 1,
            overflow: "auto",
            padding: "8px 12px",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            background: "#1a1e24",
            borderRadius: "0 0 8px 8px",
          }}
        >
          {logLines.map((line, i) => (
            <div key={i} style={{ color: logLineColor(line), lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
              {line}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
