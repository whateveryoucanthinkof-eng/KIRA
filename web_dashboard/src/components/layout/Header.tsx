import type { SystemStatus, SiteInfo } from "../../api/types";
import { Chip, Micro, Square } from "../../design/primitives";

interface HeaderProps {
  status: SystemStatus | null;
  site: SiteInfo | null;
  wsConnected: boolean;
  pageTitle: string;
  theme: "ink" | "paper";
  onToggleTheme: () => void;
}

function clock(ts?: string): string {
  if (!ts) return "--:--:--";
  try {
    return new Date(ts).toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return "--:--:--";
  }
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

/** Model card straight off the wire — never hardcode the temporal contract. */
function ModelStrip({ status }: { status: SystemStatus | null }) {
  const meta = status?.model_meta as
    | { name?: string; version?: string; history_steps?: number; forecast_steps?: number; window_seconds?: number; threshold?: number }
    | null
    | undefined;

  if (!meta) {
    return (
      <span className="t-data-s" style={{ color: "var(--paper-600)" }}>
        no model loaded
      </span>
    );
  }

  const L = meta.history_steps;
  const K = meta.forecast_steps;
  const w = meta.window_seconds;
  const parts: string[] = [];
  if (meta.version) parts.push(`v${String(meta.version).replace(/^v/, "")}`);
  if (L != null && K != null) parts.push(`L=${L} K=${K}`);
  if (w != null) parts.push(`Δt=${w}s`);
  if (meta.threshold != null) parts.push(`θ=${meta.threshold.toFixed(2)}`);

  return (
    <span
      className="t-data-s"
      style={{ color: "var(--paper-600)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
      title={meta.name ?? undefined}
    >
      {parts.join("  ·  ")}
    </span>
  );
}

export default function Header({ status, site, wsConnected, pageTitle, theme, onToggleTheme }: HeaderProps) {
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

      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 1, minWidth: 0 }}>
        <Micro style={{ letterSpacing: "0.1em" }}>Model</Micro>
        <ModelStrip status={status} />
      </div>

      <span style={{ width: 1, height: 28, background: "var(--rule-hair)", flexShrink: 0 }} />

      {status && <Chip level={status.threatLevel}>{status.threatLevel}</Chip>}

      <span
        className="t-data-s"
        style={{ color: "var(--paper-400)", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}
        title="last backend update"
      >
        {clock(status?.lastUpdate)}
      </span>

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
