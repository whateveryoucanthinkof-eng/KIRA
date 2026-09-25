/**
 * CYBERWORLD SOC — structural primitives.
 *
 * Every surface in the console is built from these. They encode the rules the
 * design system will not bend on: zero radius, zero shadow, hairline rules,
 * colour only where it carries meaning.
 *
 * See DESIGN_SYSTEM.md for the contract these implement.
 */

import type { CSSProperties, ReactNode } from "react";

/* ════════════════════════════════════════════════════════════════════════
   SEVERITY — the only sanctioned use of colour
   ════════════════════════════════════════════════════════════════════════ */

export type Sev = "nominal" | "warning" | "elevated" | "critical" | "neutral";

const SEV_VAR: Record<Sev, string> = {
  nominal: "var(--sev-nominal)",
  warning: "var(--sev-warning)",
  elevated: "var(--sev-elevated)",
  critical: "var(--sev-critical)",
  neutral: "var(--paper-400)",
};

/**
 * Normalize every severity vocabulary the backend speaks into one scale.
 *
 * Four sources converge here: PredictionData.alert_level (NOMINAL / WARNING /
 * ELEVATED / CRITICAL), SystemStatusEvent.threatLevel (low / medium / high /
 * critical), EventSeverity (info / warning / error / critical) and the
 * service/topology states (running, stopped, compromised, …).
 */
export function sev(raw: unknown): Sev {
  const s = String(raw ?? "").toLowerCase();
  switch (s) {
    case "critical":
    case "compromised":
    case "saturated":
    case "error":
    case "failed":
      return "critical";
    case "elevated":
    case "high":
      return "elevated";
    case "warning":
    case "medium":
    case "degraded":
    case "suspicious":
    case "starting":
    case "stopping":
      return "warning";
    case "nominal":
    case "low":
    case "running":
    case "active":
    case "online":
    case "live":
    case "ok":
      return "nominal";
    default:
      return "neutral";
  }
}

export function sevColor(raw: unknown): string {
  return SEV_VAR[sev(raw)];
}

/** Severity from a raw risk score against the model's own fitted threshold. */
export function sevFromRisk(risk: number, threshold = 0.65): Sev {
  if (!Number.isFinite(risk)) return "neutral";
  if (risk >= threshold + (1 - threshold) * 0.6) return "critical";
  if (risk >= threshold + (1 - threshold) * 0.25) return "elevated";
  if (risk >= threshold) return "warning";
  return "nominal";
}

/* ════════════════════════════════════════════════════════════════════════
   PANEL — a cell in the ruled sheet, not a floating card
   ════════════════════════════════════════════════════════════════════════ */

interface PanelProps {
  children: ReactNode;
  /** Drop the fill so the panel reads as bare ruled ground. */
  bare?: boolean;
  /** Hard-clip overflow. Needed for tables, logs and SVG canvases. */
  clip?: boolean;
  /** Suppress a shared edge so adjacent panels collapse into one rule. */
  collapse?: ("top" | "right" | "bottom" | "left")[];
  /**
   * Drop every border. For panels sitting inside a `.sheet` grid, where the
   * 1px gap over a rule-coloured ground already draws the hairlines — so
   * edges collapse perfectly and reflow without doubling up.
   */
  flush?: boolean;
  /** A severity spine down the left edge. */
  spine?: Sev;
  style?: CSSProperties;
  className?: string;
}

export function Panel({ children, bare, clip, collapse, flush, spine, style, className }: PanelProps) {
  const drop = new Set(collapse ?? []);
  const edge = (side: "top" | "right" | "bottom" | "left") => (flush || drop.has(side) ? "none" : "var(--hair)");
  return (
    <div
      className={className}
      style={{
        background: bare ? "transparent" : "var(--ink-050)",
        borderTop: edge("top"),
        borderRight: edge("right"),
        borderBottom: edge("bottom"),
        borderLeft: spine ? `3px solid ${SEV_VAR[spine]}` : edge("left"),
        minWidth: 0,
        minHeight: 0,
        overflow: clip ? "hidden" : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

interface PanelHeadProps {
  title: string;
  /** Right-aligned slot: chips, counts, legends, controls. */
  aside?: ReactNode;
  /** Small mono qualifier shown immediately after the title. */
  note?: string;
}

export function PanelHead({ title, aside, note }: PanelHeadProps) {
  return (
    <div
      style={{
        height: "var(--panel-head-h)",
        flexShrink: 0,
        padding: "0 var(--s-3)",
        borderBottom: "var(--hair)",
        display: "flex",
        alignItems: "center",
        gap: "var(--s-2)",
      }}
    >
      <span className="t-micro" style={{ color: "var(--paper-400)", whiteSpace: "nowrap" }}>
        {title}
      </span>
      {note && (
        <span className="t-data-s" style={{ color: "var(--paper-600)", whiteSpace: "nowrap" }}>
          {note}
        </span>
      )}
      {aside && <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "var(--s-2)", minWidth: 0 }}>{aside}</div>}
    </div>
  );
}

/** Panel body with the standard 16px gutter. */
export function PanelBody({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ padding: "var(--panel-pad)", minWidth: 0, ...style }}>{children}</div>;
}

/* ════════════════════════════════════════════════════════════════════════
   TYPE HELPERS
   ════════════════════════════════════════════════════════════════════════ */

export function Micro({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div className="t-micro" style={{ color: "var(--paper-600)", ...style }}>
      {children}
    </div>
  );
}

/** Monospace span. Everything a machine produced goes through here. */
export function Data({
  children,
  size = "m",
  color,
  style,
}: {
  children: ReactNode;
  size?: "s" | "m" | "l";
  color?: string;
  style?: CSSProperties;
}) {
  const cls = size === "s" ? "t-data-s" : size === "l" ? "t-data-l" : "t-data";
  return (
    <span className={cls} style={{ color: color ?? "var(--paper-000)", ...style }}>
      {children}
    </span>
  );
}

export function Rule({ vertical, hard, style }: { vertical?: boolean; hard?: boolean; style?: CSSProperties }) {
  const line = hard ? "var(--rule-hard)" : "var(--rule-hair)";
  return (
    <div
      style={
        vertical
          ? { width: 1, alignSelf: "stretch", background: line, flexShrink: 0, ...style }
          : { height: 1, width: "100%", background: line, flexShrink: 0, ...style }
      }
    />
  );
}

/* ════════════════════════════════════════════════════════════════════════
   SQUARE — the live indicator. A square, not a dot. It ticks, it never glows.
   ════════════════════════════════════════════════════════════════════════ */

export function Square({ status, live, size = 6 }: { status?: unknown; live?: boolean; size?: number }) {
  return (
    <span
      className={live ? "is-ticking" : undefined}
      style={{
        display: "inline-block",
        width: size,
        height: size,
        background: sevColor(status),
        flexShrink: 0,
      }}
    />
  );
}

/* ════════════════════════════════════════════════════════════════════════
   CHIP — severity as a left rule and text colour. No tinted pill.
   ════════════════════════════════════════════════════════════════════════ */

export function Chip({
  children,
  level,
  strong,
}: {
  children: ReactNode;
  level?: unknown;
  /** Invert to a solid severity block. Reserved for the single worst state on screen. */
  strong?: boolean;
}) {
  const s = sev(level);
  const c = SEV_VAR[s];
  return (
    <span
      className="t-micro"
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "3px var(--s-2)",
        // Border shorthand first, then the severity spine overrides the left edge.
        border: strong ? "1px solid transparent" : "var(--hair)",
        borderLeft: `3px solid ${strong ? "var(--ink-000)" : c}`,
        background: strong ? c : "var(--ink-200)",
        color: strong ? "var(--ink-000)" : c,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   READOUT — the primary number pattern
   ════════════════════════════════════════════════════════════════════════ */

interface ReadoutProps {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  level?: unknown;
  /** display: Instrument Serif hero numeral. data: mono KPI. */
  scale?: "hero" | "data";
}

export function Readout({ label, value, unit, sub, level, scale = "data" }: ReadoutProps) {
  const c = level === undefined ? "var(--paper-000)" : sevColor(level);
  return (
    <div style={{ minWidth: 0 }}>
      <Micro style={{ marginBottom: "var(--s-2)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {label}
      </Micro>
      <div style={{ display: "flex", alignItems: "baseline", gap: "var(--s-1)", minWidth: 0 }}>
        <span
          className={scale === "hero" ? "t-display-xl" : "t-data-l"}
          style={{ color: c, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        >
          {value}
        </span>
        {unit && (
          <span className="t-data-s" style={{ color: "var(--paper-600)", flexShrink: 0 }}>
            {unit}
          </span>
        )}
      </div>
      {sub && (
        <div
          className="t-data-s"
          style={{ color: "var(--paper-600)", marginTop: "var(--s-1)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   METER — square-ended bar on a hairline track
   ════════════════════════════════════════════════════════════════════════ */

export function Meter({
  value,
  level,
  height = 4,
  color,
}: {
  /** 0–1 */
  value: number;
  level?: unknown;
  height?: number;
  color?: string;
}) {
  const pct = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)) * 100;
  return (
    <div style={{ height, width: "100%", background: "var(--ink-200)", border: "var(--hair)", position: "relative" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          right: `${100 - pct}%`,
          background: color ?? (level === undefined ? "var(--paper-400)" : sevColor(level)),
          // A meter tracks a continuous quantity, so it eases. Discrete
          // indicators (Square, hover states) stay stepped — see tokens.css.
          transition: "right var(--dur-value) var(--ease), background var(--dur-value) var(--ease)",
        }}
      />
    </div>
  );
}

/** Labelled attribution bar: name on the left, bar and share on the right. */
export function BarRow({
  label,
  value,
  note,
  color,
  labelWidth = 128,
}: {
  label: string;
  /** 0–1 */
  value: number;
  note?: string;
  color?: string;
  labelWidth?: number;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)", minWidth: 0 }}>
      <span
        className="t-data-s"
        style={{
          width: labelWidth,
          flexShrink: 0,
          color: "var(--paper-400)",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
        title={label}
      >
        {label}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <Meter value={value} color={color ?? "var(--signal-observed)"} height={6} />
      </div>
      <span className="t-data-s" style={{ width: 44, textAlign: "right", flexShrink: 0, color: "var(--paper-000)" }}>
        {(value * 100).toFixed(1)}%
      </span>
      {note && (
        <span
          className="t-micro"
          style={{ width: 88, flexShrink: 0, color: "var(--paper-600)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {note}
        </span>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   FIELD — key/value row on a hairline
   ════════════════════════════════════════════════════════════════════════ */

export function Field({
  label,
  value,
  mono = true,
  color,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
  color?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        gap: "var(--s-3)",
        padding: "6px 0",
        borderBottom: "var(--hair)",
        minWidth: 0,
      }}
    >
      <span className="t-micro" style={{ color: "var(--paper-600)", flexShrink: 0 }}>
        {label}
      </span>
      <span
        className={mono ? "t-data" : "t-label"}
        style={{
          color: color ?? "var(--paper-000)",
          textAlign: "right",
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {value}
      </span>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   BUTTON — rectangle, hard rule, inverts on hover
   ════════════════════════════════════════════════════════════════════════ */

interface BtnProps {
  children: ReactNode;
  onClick?: () => void;
  sub?: string;
  disabled?: boolean;
  /** Adds a critical spine. For anything destructive or irreversible. */
  danger?: boolean;
  /** Solid bone fill at rest. One per group, maximum. */
  primary?: boolean;
  full?: boolean;
  title?: string;
}

export function Btn({ children, onClick, sub, disabled, danger, primary, full, title }: BtnProps) {
  const base: CSSProperties = {
    display: "block",
    width: full ? "100%" : undefined,
    textAlign: "left",
    padding: sub ? "var(--s-2) var(--s-3)" : "7px var(--s-3)",
    border: "var(--hard)",
    borderLeft: danger ? "3px solid var(--sev-critical)" : "var(--hard)",
    background: primary ? "var(--paper-000)" : "transparent",
    color: primary ? "var(--ink-000)" : disabled ? "var(--paper-600)" : "var(--paper-000)",
    opacity: disabled ? 0.45 : 1,
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "background 90ms steps(2,end), color 90ms steps(2,end)",
    minWidth: 0,
  };

  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      style={base}
      onMouseEnter={(e) => {
        if (disabled || primary) return;
        e.currentTarget.style.background = "var(--paper-000)";
        e.currentTarget.style.color = "var(--ink-000)";
      }}
      onMouseLeave={(e) => {
        if (disabled || primary) return;
        e.currentTarget.style.background = "transparent";
        e.currentTarget.style.color = "var(--paper-000)";
      }}
    >
      <span className="t-label" style={{ display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {children}
      </span>
      {sub && (
        <span
          className="t-data-s"
          style={{ display: "block", marginTop: 2, opacity: 0.62, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {sub}
        </span>
      )}
    </button>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   EMPTY — the standby state. Stated plainly, never apologetically.
   ════════════════════════════════════════════════════════════════════════ */

export function Empty({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div
      style={{
        height: "100%",
        minHeight: 96,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "var(--s-2)",
        padding: "var(--s-6)",
        textAlign: "center",
      }}
    >
      <span className="t-micro" style={{ color: "var(--paper-600)" }}>
        {children}
      </span>
      {hint && (
        <span className="t-data-s" style={{ color: "var(--paper-600)", opacity: 0.7, maxWidth: 280 }}>
          {hint}
        </span>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   SHEET — modal. A hard-edged panel on a flat scrim. No blur, no shadow.
   ════════════════════════════════════════════════════════════════════════ */

export function Sheet({ title, children, onDismiss }: { title: string; children: ReactNode; onDismiss: () => void }) {
  return (
    <div
      onClick={onDismiss}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,10,9,0.82)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
        padding: "var(--s-6)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ width: 460, maxWidth: "100%", background: "var(--ink-050)", border: "var(--hard)" }}
      >
        <PanelHead title={title} />
        <PanelBody>{children}</PanelBody>
      </div>
    </div>
  );
}
