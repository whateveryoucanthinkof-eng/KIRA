/**
 * CYBERWORLD SOC — chart system.
 *
 * One place that decides what every plot in the console looks like, so the
 * charts read as one instrument rather than five separate widgets.
 *
 * Conventions:
 *   · Horizontal gridlines only. Vertical grid is noise on a time axis.
 *   · No axis lines, no tick marks. The grid already implies the frame.
 *   · Observed series are SOLID and WARM. Forecast series are DASHED and COOL.
 *     Provenance survives greyscale — that is the test.
 *   · The model's own alert threshold is drawn on every risk plot.
 */

import { CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";
import type { ReactNode } from "react";

export const OBSERVED = "var(--signal-observed)";
export const FORECAST = "var(--signal-forecast)";

/** Shared axis/grid props. Spread these — do not restyle axes per chart. */
export const axisProps = {
  stroke: "var(--rule-hair)",
  tickLine: false,
  axisLine: false,
  tick: { fill: "var(--paper-600)", fontSize: 10, fontFamily: "var(--face-data)" },
} as const;

export function Grid() {
  return <CartesianGrid stroke="var(--rule-hair)" strokeDasharray="0" vertical={false} />;
}

export function TimeAxis({ dataKey = "t", interval }: { dataKey?: string; interval?: number | "preserveStartEnd" }) {
  return <XAxis dataKey={dataKey} {...axisProps} minTickGap={28} interval={interval ?? "preserveStartEnd"} />;
}

export function ValueAxis({
  domain,
  width = 34,
  unit,
  ticks,
}: {
  domain?: [number | string, number | string];
  width?: number;
  unit?: string;
  ticks?: number[];
}) {
  return <YAxis {...axisProps} width={width} domain={domain} ticks={ticks} unit={unit} />;
}

/**
 * The model's fitted alert threshold (PredictionData.threshold). Every risk
 * plot draws it — without it a risk curve is an unreadable number soup.
 */
export function ThresholdLine({ y, label = "THRESHOLD" }: { y: number; label?: string }) {
  if (!Number.isFinite(y)) return null;
  return (
    <ReferenceLine
      y={y}
      stroke="var(--rule-hard)"
      strokeDasharray="2 3"
      strokeWidth={1}
      label={{
        value: `${label} ${y.toFixed(2)}`,
        position: "insideTopLeft",
        fill: "var(--paper-600)",
        fontSize: 9,
        fontFamily: "var(--face-data)",
        letterSpacing: "0.08em",
        offset: 6,
      }}
    />
  );
}

/** Vertical rule marking "now" — the seam between observed and forecast. */
export function NowLine({ x, label = "NOW" }: { x: string | number; label?: string }) {
  return (
    <ReferenceLine
      x={x}
      stroke="var(--rule-hard)"
      strokeWidth={1}
      label={{
        value: label,
        position: "insideTop",
        fill: "var(--paper-600)",
        fontSize: 9,
        fontFamily: "var(--face-data)",
        letterSpacing: "0.1em",
        offset: 6,
      }}
    />
  );
}

/* ════════════════════════════════════════════════════════════════════════
   TOOLTIP — a hard-edged ink panel. No radius, no shadow, mono throughout.
   ════════════════════════════════════════════════════════════════════════ */

interface TipItem {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
}

export function Tip({
  active,
  payload,
  label,
  fmt,
  suffix,
}: {
  active?: boolean;
  payload?: unknown[];
  label?: string | number;
  /** Per-series value formatter. */
  fmt?: (v: number | string, name: string) => string;
  suffix?: string;
}) {
  if (!active || !payload?.length) return null;
  const items = (payload as TipItem[]).filter((i) => i.value !== undefined && i.value !== null);
  if (!items.length) return null;

  return (
    <div
      style={{
        background: "var(--ink-100)",
        border: "var(--hard)",
        padding: "var(--s-2) var(--s-3)",
        minWidth: 148,
      }}
    >
      {label !== undefined && (
        <div
          className="t-micro"
          style={{ color: "var(--paper-600)", paddingBottom: "var(--s-1)", marginBottom: "var(--s-1)", borderBottom: "var(--hair)" }}
        >
          {label}
        </div>
      )}
      {items.map((item, i) => {
        const raw = item.value as number | string;
        const text = fmt ? fmt(raw, String(item.name ?? "")) : typeof raw === "number" ? raw.toFixed(2) : String(raw);
        return (
          <div key={`${item.dataKey ?? i}`} style={{ display: "flex", alignItems: "center", gap: "var(--s-2)" }}>
            <span style={{ width: 6, height: 6, background: item.color, display: "inline-block", flexShrink: 0 }} />
            <span className="t-data-s" style={{ color: "var(--paper-400)" }}>
              {item.name}
            </span>
            <span className="t-data-s" style={{ color: "var(--paper-000)", marginLeft: "auto", paddingLeft: "var(--s-4)" }}>
              {text}
              {suffix}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   LEGEND — squares and mono labels. Dashed entries draw a dashed rule.
   ════════════════════════════════════════════════════════════════════════ */

export function Legend({ items }: { items: { color: string; label: string; dashed?: boolean; fill?: boolean }[] }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--s-3)", flexWrap: "wrap" }}>
      {items.map((it) => (
        <span key={it.label} style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
          {it.dashed ? (
            <span
              style={{
                width: 14,
                height: 0,
                borderTop: `1px dashed ${it.color}`,
                display: "inline-block",
                flexShrink: 0,
              }}
            />
          ) : (
            <span
              style={{
                width: it.fill ? 14 : 8,
                height: it.fill ? 8 : 8,
                background: it.color,
                opacity: it.fill ? 0.3 : 1,
                border: it.fill ? `1px solid ${it.color}` : undefined,
                display: "inline-block",
                flexShrink: 0,
              }}
            />
          )}
          <span className="t-micro" style={{ color: "var(--paper-600)" }}>
            {it.label}
          </span>
        </span>
      ))}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   SPARK — inline bar sparkline. No axes, no chrome, pure data-ink.
   ════════════════════════════════════════════════════════════════════════ */

export function Spark({
  values,
  height = 28,
  color = OBSERVED,
  max,
}: {
  values: number[];
  height?: number;
  color?: string;
  max?: number;
}) {
  if (!values.length) return <div style={{ height }} />;
  const peak = max ?? Math.max(...values, 1e-6);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height, width: "100%" }}>
      {values.map((v, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            minWidth: 1,
            height: `${Math.max(2, (Math.max(0, v) / peak) * 100)}%`,
            background: color,
            opacity: 0.25 + 0.75 * (i / Math.max(1, values.length - 1)),
          }}
        />
      ))}
    </div>
  );
}

/** Chart container that reserves the standard gutter. */
export function Plot({ height, children }: { height: number; children: ReactNode }) {
  return <div style={{ height, width: "100%", minWidth: 0, padding: "var(--s-3) var(--s-3) var(--s-2) 0" }}>{children}</div>;
}
