import type { SystemStatus } from "../../api/types";
import { Micro, Square, Rule } from "../../design/primitives";

type Page =
  | "overview"
  | "stage"
  | "investigation"
  | "network"
  | "predictions"
  | "campaign"
  | "attck"
  | "replay"
  | "incidents"
  | "events"
  | "model"
  | "controls";

/**
 * Navigation carries no icons. An index and the label are enough, and the
 * active state is a full inversion — unmistakable without a glow or a rail.
 */
const NAV: { id: Page; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "stage", label: "Forecast Stage" },
  { id: "investigation", label: "Investigation" },
  { id: "network", label: "Network" },
  { id: "predictions", label: "Predictions" },
  { id: "campaign", label: "Campaign" },
  { id: "attck", label: "ATT&CK" },
  { id: "replay", label: "Replay" },
  { id: "incidents", label: "Incidents" },
  { id: "events", label: "Events" },
  { id: "model", label: "Model" },
  { id: "controls", label: "Controls" },
];

interface SidebarProps {
  activePage: Page;
  onNavigate: (page: Page) => void;
  status: SystemStatus | null;
}

function fmtUptime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds || 0));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return `${d}d ${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m`;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export default function Sidebar({ activePage, onNavigate, status }: SidebarProps) {
  const services: { label: string; val: string }[] = [
    { label: "Network", val: status?.networkStatus ?? "unknown" },
    { label: "Sensor", val: status?.telemetryStatus ?? "unknown" },
    { label: "Inference", val: status?.predictionStatus ?? "unknown" },
  ];

  return (
    <aside
      style={{
        width: "var(--rail-w)",
        flexShrink: 0,
        background: "var(--ink-050)",
        borderRight: "var(--hard)",
        display: "flex",
        flexDirection: "column",
        height: "100%",
      }}
    >
      {/* ── Masthead — mark and wordmark as one lockup ───────────── */}
      <div style={{ padding: "var(--s-3) var(--s-4)", borderBottom: "var(--hair)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <img src="/kira-mark.png" alt="K.I.R.A. mark" width={34} height={34} style={{ display: "block", flexShrink: 0 }} />
          <div
            style={{
              color: "var(--paper-000)",
              lineHeight: 1,
              fontFamily: "var(--face-display)",
              fontWeight: 700,
              fontSize: 20,
              letterSpacing: "0.02em",
              whiteSpace: "nowrap",
            }}
          >
            K.I.R.A.
          </div>
        </div>
        <Micro
          style={{ marginTop: 7, color: "var(--paper-600)", letterSpacing: "0.14em", fontSize: 8, lineHeight: 1.55 }}
          title="Kinetic Intrusion Risk Anticipator"
        >
          <span style={{ display: "block" }}>Kinetic Intrusion</span>
          <span style={{ display: "block" }}>Risk Anticipator</span>
        </Micro>
      </div>

      {/* ── Navigation ───────────────────────────────────────────── */}
      <nav style={{ flex: 1, padding: "var(--s-2) 0", minHeight: 0, overflowY: "auto" }}>
        {NAV.map((item, i) => {
          const active = item.id === activePage;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              aria-current={active ? "page" : undefined}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: "var(--s-3)",
                width: "100%",
                padding: "9px var(--s-4)",
                textAlign: "left",
                background: active ? "var(--paper-000)" : "transparent",
                color: active ? "var(--ink-000)" : "var(--paper-400)",
                transition: "background 90ms steps(2,end), color 90ms steps(2,end)",
              }}
              onMouseEnter={(e) => {
                if (active) return;
                e.currentTarget.style.background = "var(--ink-200)";
                e.currentTarget.style.color = "var(--paper-000)";
              }}
              onMouseLeave={(e) => {
                if (active) return;
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = "var(--paper-400)";
              }}
            >
              <span className="t-data-s" style={{ opacity: active ? 0.55 : 0.5, flexShrink: 0 }}>
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="t-label" style={{ letterSpacing: "0.02em" }}>
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>

      {/* ── Service health ───────────────────────────────────────── */}
      <div style={{ padding: "var(--s-3) var(--s-4) var(--s-4)", borderTop: "var(--hard)" }}>
        <Micro style={{ marginBottom: "var(--s-2)" }}>Services</Micro>

        {services.map(({ label, val }) => (
          <div
            key={label}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "var(--s-2)",
              padding: "5px 0",
              borderBottom: "var(--hair)",
            }}
          >
            <span className="t-label" style={{ color: "var(--paper-400)" }}>
              {label}
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "var(--s-2)" }}>
              <span className="t-data-s" style={{ color: "var(--paper-600)" }}>
                {val}
              </span>
              <Square status={val} live={val === "running"} />
            </span>
          </div>
        ))}

        <Rule style={{ margin: "var(--s-3) 0 var(--s-2)" }} />

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <Micro>Uptime</Micro>
          <span className="t-data-s" style={{ color: "var(--paper-400)", fontVariantNumeric: "tabular-nums" }}>
            {fmtUptime(status?.uptime ?? 0)}
          </span>
        </div>
      </div>
    </aside>
  );
}

export type { Page };
