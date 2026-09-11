import type { SystemStatus } from "../../api/types";
import { StatusDot } from "../shared/StatusBadge";

type Page = "overview" | "network" | "predictions" | "events" | "controls";

const navItems: { id: Page; label: string; icon: string }[] = [
  { id: "overview",    label: "Overview",    icon: "▦" },
  { id: "network",     label: "Network",     icon: "⬡" },
  { id: "predictions", label: "Predictions", icon: "◈" },
  { id: "events",      label: "Events",      icon: "≡" },
  { id: "controls",    label: "Controls",    icon: "⊙" },
];

interface SidebarProps {
  activePage: Page;
  onNavigate: (page: Page) => void;
  status: SystemStatus | null;
}

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${d}d ${h}h ${m}m`;
}

export default function Sidebar({ activePage, onNavigate, status }: SidebarProps) {
  return (
    <aside
      style={{
        width: 200,
        flexShrink: 0,
        background: "var(--color-surface)",
        borderRight: "1px solid var(--color-border)",
        display: "flex",
        flexDirection: "column",
        height: "100%",
      }}
    >
      {/* Brand */}
      <div
        style={{
          padding: "16px 16px 14px",
          borderBottom: "1px solid var(--color-border)",
        }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: "-0.01em",
            color: "var(--color-text-primary)",
          }}
        >
          Cyber Network
        </div>
        <div
          style={{
            fontSize: 11,
            color: "var(--color-text-muted)",
            marginTop: 1,
            fontWeight: 500,
          }}
        >
          Predictor
        </div>
      </div>

      {/* Nav */}
      <nav style={{ padding: "8px 8px", flex: 1 }}>
        {navItems.map((item) => {
          const isActive = item.id === activePage;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 9,
                width: "100%",
                padding: "7px 8px",
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: isActive ? 600 : 400,
                color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                background: isActive ? "var(--color-base)" : "transparent",
                textAlign: "left",
                transition: "background 0.1s, color 0.1s",
                marginBottom: 1,
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background = "var(--color-base)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                }
              }}
            >
              <span style={{ fontSize: 14, opacity: isActive ? 1 : 0.6 }}>{item.icon}</span>
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* System status footer */}
      {status && (
        <div
          style={{
            padding: "12px 14px",
            borderTop: "1px solid var(--color-border)",
            display: "flex",
            flexDirection: "column",
            gap: 5,
          }}
        >
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: 2 }}>
            Services
          </div>
          {[
            { label: "Network",    val: status.networkStatus },
            { label: "Telemetry",  val: status.telemetryStatus },
            { label: "Prediction", val: status.predictionStatus },
          ].map(({ label, val }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{label}</span>
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <StatusDot status={val} pulse />
                <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>{val}</span>
              </div>
            </div>
          ))}
          <div style={{ marginTop: 4, fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
            uptime {formatUptime(status.uptime)}
          </div>
        </div>
      )}
    </aside>
  );
}

export type { Page };
