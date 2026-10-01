/**
 * Alternate futures.
 *
 * From the present, the forecast's three most likely continuations fan out
 * against a time axis: each curve ends where its first milestone is expected,
 * runs through the hosts it would touch, and is as thick as it is likely.
 * Escalation is critical, a pivot is warning, a back-off is muted — the
 * severity palette, nothing else. Every branch is forecast, so every branch is
 * dashed; the most likely one carries moving traffic.
 */

import { useEffect, useRef, useState } from "react";
import type { PredictionResult } from "../api/types";
import type { ForecastBranch } from "../types/forecast";
import { Empty, sevColor, sevFromRisk } from "../design/primitives";
import { OBSERVED } from "../design/charts";
import { HORIZON_SECONDS } from "../types/timeline";

const KIND_COLOUR: Record<ForecastBranch["kind"], string> = {
  escalation: "var(--sev-critical)",
  pivot: "var(--sev-warning)",
  backoff: "var(--paper-600)",
};

const KIND_LABEL: Record<ForecastBranch["kind"], string> = {
  escalation: "escalation",
  pivot: "pivot",
  backoff: "back-off",
};

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

function cubic(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export default function ForecastTree({
  branches,
  prediction,
  futureSeconds,
}: {
  branches: ForecastBranch[] | null;
  prediction: PredictionResult | null;
  futureSeconds: number | null;
}) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);

  if (!branches || !branches.length) {
    return (
      <div ref={ref} className="ft">
        <Empty hint="The backend forwards the rollout's risk and stage per step, not its competing continuations.">
          Branch rollout not streamed
        </Empty>
      </div>
    );
  }

  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);
  const technique = prediction?.mitre_technique ?? null;

  // Geometry, in CSS pixels of the measured box.
  const top = 36;
  const axisY = Math.max(top + 80, h - 30);
  const bottom = axisY - 22;
  const yc = (top + bottom) / 2;
  const padL = 20;
  const x0 = Math.max(170, w * 0.2);
  const xMax = Math.max(x0 + 120, w - 300);
  const xAt = (s: number) => x0 + (Math.min(HORIZON_SECONDS, Math.max(0, s)) / HORIZON_SECONDS) * (xMax - x0);
  const lanesY = [top + (bottom - top) * 0.1, yc, top + (bottom - top) * 0.9];

  const laid = [...branches]
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((b, k) => {
      const xe = xAt(b.horizon_seconds);
      const ye = lanesY[k] ?? yc;
      const c1x = x0 + (xe - x0) * 0.5;
      const c2x = x0 + (xe - x0) * 0.5;
      const d = `M ${x0} ${yc} C ${c1x} ${yc}, ${c2x} ${ye}, ${xe} ${ye}`;
      const hops = b.hops.map((hop, i) => {
        const t = (i + 1) / b.hops.length;
        return { ...hop, x: cubic(x0, c1x, c2x, xe, t), y: cubic(yc, yc, ye, ye, t) };
      });
      return { b, xe, ye, d, hops };
    });

  const lead = laid.reduce((a, c) => (c.b.probability > a.b.probability ? c : a), laid[0]);
  const hovered = laid.find((l) => l.b.id === hover) ?? null;

  return (
    <div ref={ref} className="ft" onMouseLeave={() => setHover(null)}>
      {w > 0 && (
        <svg width={w} height={h} className="ft-svg" aria-hidden>
          {/* Time axis: NOW to the end of the horizon. */}
          <line x1={x0} x2={xMax} y1={axisY} y2={axisY} stroke="var(--rule-hard)" strokeWidth={1} shapeRendering="crispEdges" />
          {[0, 4, 8, 12, 16].map((s) => (
            <g key={s}>
              <line x1={xAt(s)} x2={xAt(s)} y1={top - 8} y2={axisY + 4} stroke="var(--rule-hair)" strokeWidth={1} shapeRendering="crispEdges" />
              <text x={xAt(s)} y={axisY + 16} textAnchor="middle" fill="var(--paper-600)" fontSize={10} fontFamily="var(--face-data)">
                {s === 0 ? "NOW" : `+${s}s`}
              </text>
            </g>
          ))}

          {/* The cursor, when it is looking into the horizon. */}
          {futureSeconds != null && (
            <g>
              <line
                x1={xAt(futureSeconds)}
                x2={xAt(futureSeconds)}
                y1={top - 12}
                y2={axisY}
                stroke="var(--paper-000)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text x={xAt(futureSeconds) + 6} y={top - 2} fill="var(--paper-000)" fontSize={10} fontFamily="var(--face-data)">
                t +{futureSeconds}s
              </text>
            </g>
          )}

          {/* The path so far: observed, solid, warm. */}
          <line x1={padL} x2={x0} y1={yc} y2={yc} stroke={OBSERVED} strokeWidth={3} />
          <text x={padL} y={yc + 18} fill="var(--paper-600)" fontSize={9} fontFamily="var(--face-ui)" letterSpacing="1.6">
            OBSERVED
          </text>

          {/* Branches. */}
          {laid.map(({ b, d, xe, ye, hops }) => {
            const colour = KIND_COLOUR[b.kind];
            const dim = hover != null && hover !== b.id;
            const width = 1.4 + b.probability * 8;
            const reached = futureSeconds != null && b.horizon_seconds <= futureSeconds;
            return (
              <g key={b.id} style={{ opacity: dim ? 0.22 : 1, transition: "opacity var(--dur-quick) var(--ease)" }}>
                <path d={d} fill="none" stroke={colour} strokeWidth={width} strokeDasharray={b.kind === "backoff" ? "2 5" : "7 5"} strokeOpacity={b.kind === "backoff" ? 0.7 : 0.85} />
                {b === lead.b && b.kind !== "backoff" && (
                  <path d={d} fill="none" stroke="var(--paper-000)" strokeWidth={Math.max(1.5, width * 0.35)} strokeDasharray="6 26" className="is-traveling" strokeOpacity={0.75} />
                )}
                {hops.slice(0, -1).map((hp, i, list) => {
                  // Crowded neighbours alternate above and below the line.
                  const prev = list[i - 1];
                  const above = Boolean(prev) && Math.abs(hp.x - prev.x) < 64 && i % 2 === 1;
                  return (
                    <g key={`${hp.ip}-${i}`}>
                      <rect x={hp.x - 4} y={hp.y - 4} width={8} height={8} fill="var(--ink-000)" stroke={colour} strokeWidth={1.4} shapeRendering="crispEdges" />
                      <text x={hp.x} y={above ? hp.y - 10 : hp.y + 17} textAnchor="middle" fill="var(--paper-400)" fontSize={9.5} fontFamily="var(--face-data)">
                        {hp.name}
                      </text>
                    </g>
                  );
                })}
                {/* End of the branch: its first milestone. */}
                <rect
                  x={xe - 7}
                  y={ye - 7}
                  width={14}
                  height={14}
                  fill={reached ? colour : "var(--ink-000)"}
                  stroke={colour}
                  strokeWidth={2}
                  shapeRendering="crispEdges"
                />
                {/* Fat invisible hit target. */}
                <path
                  d={d}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={22}
                  style={{ cursor: "pointer", pointerEvents: "stroke" }}
                  onMouseEnter={() => setHover(b.id)}
                />
              </g>
            );
          })}

          {/* The present. */}
          <rect x={x0 - 6} y={yc - 6} width={12} height={12} fill="var(--paper-000)" shapeRendering="crispEdges" />
        </svg>
      )}

      {/* ── DOM labels over the drawing ─────────────────────────────── */}
      {w > 0 && (
        <>
          <div className="ft-now" style={{ left: x0, top: yc }}>
            <span>NOW</span>
            {technique && <b>{technique.split(" ")[0]}</b>}
            <em style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>risk {risk.toFixed(3)}</em>
          </div>

          {laid.map(({ b, xe, ye }) => (
            <div
              key={b.id}
              className={hover != null && hover !== b.id ? "ft-end is-dim" : "ft-end"}
              style={{ left: xe + 16, top: ye }}
              onMouseEnter={() => setHover(b.id)}
            >
              <div className="ft-end-head">
                <span className="ft-id" style={{ color: KIND_COLOUR[b.kind] }}>
                  {b.id}
                </span>
                <span className="ft-p">{(b.probability * 100).toFixed(0)}%</span>
                <span className="ft-bar">
                  <i style={{ width: `${b.probability * 100}%`, background: KIND_COLOUR[b.kind] }} />
                </span>
              </div>
              <div className="ft-label">{b.label}</div>
              <div className="ft-meta">
                {b.technique !== "—" ? `${b.technique} · ` : ""}+{b.horizon_seconds}s · → {b.hops[b.hops.length - 1]?.name ?? "—"}
              </div>
            </div>
          ))}

          {hovered && (
            <div
              className="ft-card"
              style={{
                left: Math.min(hovered.xe + 16, w - 316),
                top: hovered.ye > h / 2 ? hovered.ye - 18 : hovered.ye + 58,
                transform: hovered.ye > h / 2 ? "translateY(-100%)" : undefined,
                borderLeftColor: KIND_COLOUR[hovered.b.kind],
              }}
            >
              <div className="ft-card-head">
                <span>
                  Branch {hovered.b.id} · {KIND_LABEL[hovered.b.kind]}
                </span>
                <b style={{ color: KIND_COLOUR[hovered.b.kind] }}>{(hovered.b.probability * 100).toFixed(1)}%</b>
              </div>
              <dl>
                <dt>technique</dt>
                <dd>{hovered.b.technique}</dd>
                <dt>confidence</dt>
                <dd>{hovered.b.confidence.toFixed(3)}</dd>
                <dt>eta</dt>
                <dd>+{hovered.b.horizon_seconds}s</dd>
                <dt>path</dt>
                <dd>{hovered.b.hops.map((hp) => hp.name).join(" → ")}</dd>
                <dt>targets</dt>
                <dd>{hovered.b.hops.filter((hp) => hp.ip.startsWith("10.")).map((hp) => hp.ip).join(", ") || "—"}</dd>
                <dt>packets</dt>
                <dd>{hovered.b.packets.toLocaleString()}</dd>
                <dt>volume</dt>
                <dd>{fmtBytes(hovered.b.bytes)}</dd>
                <dt>peak risk</dt>
                <dd style={{ color: sevColor(sevFromRisk(hovered.b.peak_risk, threshold)) }}>{hovered.b.peak_risk.toFixed(3)}</dd>
              </dl>
            </div>
          )}
        </>
      )}
    </div>
  );
}
