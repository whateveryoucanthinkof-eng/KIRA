import type { EventSeverity, ServiceStatus, ThreatLevel, AttackStatus } from "../../api/types";

interface StatusDotProps {
  status: ServiceStatus | AttackStatus | "online" | "offline" | "degraded" | "compromised" | "warning";
  pulse?: boolean;
}

const dotColor: Record<string, string> = {
  running: "var(--color-status-green)",
  active: "var(--color-status-green)",
  online: "var(--color-status-green)",
  stopped: "var(--color-text-muted)",
  inactive: "var(--color-text-muted)",
  offline: "var(--color-text-muted)",
  error: "var(--color-status-red)",
  compromised: "var(--color-status-red)",
  unknown: "var(--color-text-muted)",
  degraded: "var(--color-status-amber)",
  warning: "var(--color-status-amber)",
  none: "var(--color-text-muted)",
  mitigated: "var(--color-status-blue)",
  contained: "var(--color-status-blue)",
  saturated: "var(--color-status-red)",
  suspicious: "var(--color-status-amber)",
  down: "var(--color-status-red)",
};

export function StatusDot({ status, pulse }: StatusDotProps) {
  const color = dotColor[status] ?? "var(--color-text-muted)";
  return (
    <span
      className={pulse && (status === "running" || status === "active" || status === "online") ? "pulse-dot" : ""}
      style={{
        display: "inline-block",
        width: 7,
        height: 7,
        borderRadius: "50%",
        backgroundColor: color,
        flexShrink: 0,
      }}
    />
  );
}

interface SeverityBadgeProps {
  severity: EventSeverity;
  children?: React.ReactNode;
}

const severityStyles: Record<EventSeverity, { bg: string; color: string }> = {
  info:     { bg: "var(--color-status-blue-bg)",  color: "var(--color-status-blue)" },
  warning:  { bg: "var(--color-status-amber-bg)", color: "var(--color-status-amber)" },
  error:    { bg: "var(--color-status-red-bg)",   color: "var(--color-status-red)" },
  critical: { bg: "#fff0f0",                       color: "#b91c1c" },
};

export function SeverityBadge({ severity, children }: SeverityBadgeProps) {
  const norm = String(severity || "info").toLowerCase();
  const s = severityStyles[norm as EventSeverity] || severityStyles.info;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 7px",
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        backgroundColor: s.bg,
        color: s.color,
      }}
    >
      {children ?? severity}
    </span>
  );
}

interface ThreatBadgeProps {
  level?: string;
}

const threatStyles: Record<string, { bg: string; color: string }> = {
  low:      { bg: "var(--color-status-green-bg)",  color: "var(--color-status-green)" },
  nominal:  { bg: "var(--color-status-green-bg)",  color: "var(--color-status-green)" },
  medium:   { bg: "var(--color-status-amber-bg)",  color: "var(--color-status-amber)" },
  warning:  { bg: "var(--color-status-amber-bg)",  color: "var(--color-status-amber)" },
  high:     { bg: "var(--color-status-red-bg)",    color: "var(--color-status-red)" },
  elevated: { bg: "var(--color-status-red-bg)",    color: "var(--color-status-red)" },
  critical: { bg: "#fff0f0",                        color: "#b91c1c" },
};

export function ThreatBadge({ level }: ThreatBadgeProps) {
  const norm = String(level || "low").toLowerCase();
  const s = threatStyles[norm] || threatStyles.low;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        padding: "2px 9px",
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        backgroundColor: s.bg,
        color: s.color,
        border: `1px solid ${s.color}22`,
      }}
    >
      {level || "low"}
    </span>
  );
}
