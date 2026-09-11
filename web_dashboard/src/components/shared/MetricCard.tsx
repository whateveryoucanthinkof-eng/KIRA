interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  accent?: "green" | "amber" | "red" | "blue" | "default";
  mono?: boolean;
}

const accentColor: Record<string, string> = {
  green: "var(--color-status-green)",
  amber: "var(--color-status-amber)",
  red: "var(--color-status-red)",
  blue: "var(--color-status-blue)",
  default: "var(--color-text-primary)",
};

export function MetricCard({ label, value, unit, sub, accent = "default", mono }: MetricCardProps) {
  return (
    <div className="panel" style={{ padding: "11px 12px", minWidth: 0 }}>
      <div className="panel-title" style={{ marginBottom: 5, color: "var(--color-text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 3, minWidth: 0 }}>
        <span style={{
          fontSize: 20,
          fontWeight: 600,
          color: accentColor[accent],
          fontFamily: mono ? "var(--font-mono)" : undefined,
          letterSpacing: mono ? "-0.02em" : undefined,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          minWidth: 0,
        }}>
          {value}
        </span>
        {unit && (
          <span style={{ fontSize: 11, color: "var(--color-text-muted)", fontWeight: 500, flexShrink: 0 }}>
            {unit}
          </span>
        )}
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{sub}</div>
      )}
    </div>
  );
}
