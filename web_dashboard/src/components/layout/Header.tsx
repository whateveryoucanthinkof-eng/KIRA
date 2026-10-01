import type { SystemStatus, SiteInfo } from "../../api/types";
import { Chip, Square } from "../../design/primitives";

interface HeaderProps {
  status: SystemStatus | null;
  site: SiteInfo | null;
  wsConnected: boolean;
  pageTitle: string;
  theme: "ink" | "paper";
  onToggleTheme: () => void;
  /** Set when the console is showing a recorded or forecast moment, not the present. */
  timeline?: { label: string; onLive: () => void } | null;
}

/**
 * Two-state segmented control. A rectangle with one inverted half — no track,
 * no thumb, no transition easing. It reads as a switch because it is one.
 */
function ThemeSwitch({ theme, onToggle }: { theme: "ink" | "paper"; onToggle: () => void }) {
  return (
    <div style={{ display: "flex", border: "var(--hair)", flexShrink: 0 }}>
      {(["ink", "paper"] as const).map((t) => {
        const on = theme === t;
        return (
          <button
            key={t}
            onClick={() => !on && onToggle()}
            aria-pressed={on}
            className="t-micro"
            style={{
              padding: "5px var(--s-2)",
              background: on ? "var(--paper-000)" : "transparent",
              color: on ? "var(--ink-000)" : "var(--paper-600)",
              transition: "background 90ms steps(2,end), color 90ms steps(2,end)",
            }}
          >
            {t}
          </button>
        );
      })}
    </div>
  );
}

export default function Header({ status, site, wsConnected, pageTitle, theme, onToggleTheme, timeline }: HeaderProps) {
  return (
    <header
      style={{
        height: "var(--header-h)",
        flexShrink: 0,
        background: "var(--ink-050)",
        borderBottom: "var(--hard)",
        display: "flex",
        alignItems: "center",
        gap: "var(--s-4)",
        padding: "0 var(--s-4)",
      }}
    >
      <h1 className="t-display-m" style={{ margin: 0, color: "var(--paper-000)", whiteSpace: "nowrap" }}>
        {pageTitle}
      </h1>

      {site?.name && (
        <>
          <span style={{ width: 1, height: 20, background: "var(--rule-hair)", flexShrink: 0 }} />
          <span
            className="t-data-s"
            style={{ color: "var(--paper-400)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0 }}
          >
            {site.name}
          </span>
        </>
      )}

      <div style={{ flex: 1, minWidth: "var(--s-4)" }} />

      {timeline && (
        <button className="hdr-timeline" onClick={timeline.onLive} title="Return to live">
          <span>{timeline.label}</span>
          <b>Live</b>
        </button>
      )}

      {status && <Chip level={status.threatLevel}>{status.threatLevel}</Chip>}

      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "var(--s-2)",
          padding: "4px var(--s-2)",
          border: "var(--hair)",
          borderLeft: `3px solid ${wsConnected ? "var(--sev-nominal)" : "var(--sev-warning)"}`,
          flexShrink: 0,
        }}
      >
        <Square status={wsConnected ? "running" : "warning"} live={wsConnected} />
        <span className="t-micro" style={{ color: wsConnected ? "var(--sev-nominal)" : "var(--sev-warning)" }}>
          {wsConnected ? "Live" : "Offline"}
        </span>
      </span>

      <ThemeSwitch theme={theme} onToggle={onToggleTheme} />
    </header>
  );
}
