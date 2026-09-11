import type { SystemStatus, SiteInfo } from "../../api/types";
import { ThreatBadge } from "../shared/StatusBadge";

interface HeaderProps {
  status: SystemStatus | null;
  site: SiteInfo | null;
  wsConnected: boolean;
  pageTitle: string;
  darkMode: boolean;
  onToggleDark: () => void;
}

function fmt(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

function DarkToggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      title={on ? "Switch to light mode" : "Switch to dark mode"}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        background: "none",
        border: "none",
        cursor: "pointer",
        padding: 0,
      }}
    >
      <span style={{ fontSize: 11, color: "var(--color-text-muted)", userSelect: "none" }}>
        {on ? "Dark" : "Light"}
      </span>
      {/* Track */}
      <span
        style={{
          position: "relative",
          display: "inline-block",
          width: 32,
          height: 18,
          borderRadius: 9,
          background: on ? "var(--color-status-blue)" : "var(--color-border-strong)",
          transition: "background 0.2s",
          flexShrink: 0,
        }}
      >
        {/* Thumb */}
        <span
          style={{
            position: "absolute",
            top: 2,
            left: on ? 16 : 2,
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: "white",
            boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
            transition: "left 0.2s",
          }}
        />
      </span>
    </button>
  );
}

export default function Header({ status, site, wsConnected, pageTitle, darkMode, onToggleDark }: HeaderProps) {
  return (
    <header
      style={{
        height: 48,
        background: "var(--color-surface)",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        alignItems: "center",
        padding: "0 20px",
        gap: 16,
        flexShrink: 0,
      }}
    >
      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>
        {pageTitle}
      </span>

      <div style={{ flex: 1 }} />

      {site && (
        <span style={{ fontSize: 11, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {site.name} · {site.location}
        </span>
      )}

      {status && <ThreatBadge level={status.threatLevel} />}

      {status && (
        <span style={{ fontSize: 11, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {fmt(status.lastUpdate)}
        </span>
      )}

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 5,
          padding: "3px 9px",
          borderRadius: 4,
          background: wsConnected ? "var(--color-status-green-bg)" : "var(--color-status-amber-bg)",
          border: `1px solid ${wsConnected ? "var(--color-status-green)" : "var(--color-status-amber)"}22`,
        }}
      >
        <span
          className={wsConnected ? "pulse-dot" : ""}
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            backgroundColor: wsConnected ? "var(--color-status-green)" : "var(--color-status-amber)",
            display: "inline-block",
          }}
        />
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.05em", color: wsConnected ? "var(--color-status-green)" : "var(--color-status-amber)" }}>
          {wsConnected ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      <DarkToggle on={darkMode} onToggle={onToggleDark} />
    </header>
  );
}
