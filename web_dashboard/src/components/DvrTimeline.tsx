/**
 * Replay timeline: the scrubber and the waveform on one time axis.
 *
 * Keyframe pins across the top, a severity strip carrying the playhead
 * handle, then the capture's scored risk (warm, filled) over its packet rate
 * (neutral line), and the capture clock beneath. Press or drag anywhere to
 * seek; the page reads the same position, so the stream and the inspector
 * follow. What has not been played yet is faded, not hidden: the analysis
 * already ran, the DVR only walks it.
 */

import { memo, useLayoutEffect, useRef, useState } from "react";
import type { ReplayRow } from "../types/replay";
import { sevColor, sevFromRisk } from "../design/primitives";
import { RISK_BAND_COLORS } from "../design/charts";

export interface Keyframe {
  /** Row index, not window number: the backend skips empty windows. */
  index: number;
  /** Short mark drawn on the pin: a technique id, ALERT, PEAK, CLEAR. */
  tag: string;
  /** What happened, for the pin's title and the transport readout. */
  label: string;
  color: string;
}

/* Gutters for the two scales, and the lanes top to bottom. */
const L = 40;
const R = 52;
const PINS = 30;
const TRACK = 12;
const GAP = 8;
const AXIS = 18;

/** Capture clock, m:ss.s */
export function fmtClock(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function tickLabel(sec: number): string {
  const m = Math.floor(sec / 60);
  return `${m}:${String(Math.round(sec - m * 60)).padStart(2, "0")}`;
}

function niceCeil(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

export function fmtRate(v: number): string {
  if (v >= 10_000) return `${Math.round(v / 1000)}k`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return String(Math.round(v));
}

interface Geometry {
  w: number;
  h: number;
}

/** Everything that does not move with the playhead, drawn once per capture and size. */
const Static = memo(function Static({
  rows,
  keyframes,
  w,
  h,
  threshold,
  windowSeconds,
}: Geometry & { rows: ReplayRow[]; keyframes: Keyframe[]; threshold: number; windowSeconds: number }) {
  const n = rows.length;
  const W = w - L - R;
  const cw = W / n;
  const waveTop = PINS + TRACK + GAP;
  const waveH = h - waveTop - AXIS;
  const cx = (i: number) => L + (i + 0.5) * cw;
  const yR = (v: number) => waveTop + (1 - v) * waveH;

  const rates = rows.map((r) => (r.packets != null ? r.packets / windowSeconds : null));
  const hasRate = rates.some((v) => v != null);
  const maxRate = niceCeil(Math.max(1, ...rates.map((v) => v ?? 0)));
  const yP = (v: number) => waveTop + (1 - v / maxRate) * waveH;

  // The risk trace wears the severity ladder: each segment takes the colour
  // of the band its windows sit in, meeting exactly at the crossings.
  const bands = rows.map((r) => {
    const s = sevFromRisk(r.risk, threshold);
    return s === "critical" ? 3 : s === "elevated" ? 2 : s === "warning" ? 1 : 0;
  });
  const bandTrace = (band: number): { line: string; area: string } => {
    const xy: [number, number][] = [];
    rows.forEach((r, i) => {
      // A window joins its own band; the window after a band also joins, so
      // adjacent segments share the crossing point and never leave a gap.
      if (bands[i] === band || (i > 0 && bands[i - 1] === band)) xy.push([cx(i), yR(r.risk)]);
    });
    if (!xy.length) return { line: "", area: "" };
    const line = xy.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
    const area = `${line}L${xy[xy.length - 1][0].toFixed(1)},${yR(0).toFixed(1)}L${xy[0][0].toFixed(1)},${yR(0).toFixed(1)}Z`;
    return { line, area };
  };
  const traces = ([0, 1, 2, 3] as const).map((b) => bandTrace(b));
  let rateLine = "";
  rates.forEach((v, i) => {
    if (v == null) return;
    rateLine += `${rateLine ? "L" : "M"}${cx(i).toFixed(1)},${yP(v).toFixed(1)}`;
  });

  // Clock ticks at least 64px apart.
  const total = n * windowSeconds;
  const step = [5, 10, 15, 30, 60, 120, 300, 600].find((s) => (s / total) * W >= 64) ?? 1200;
  const ticks: number[] = [];
  for (let t = 0; t <= total + 1e-6; t += step) ticks.push(t);

  // Pins: two label rows, each label taking the first row it fits.
  const ends = [-Infinity, -Infinity];
  const pins = [...keyframes]
    .sort((a, b) => a.index - b.index)
    .map((k) => {
      const x = cx(k.index);
      const tw = k.tag.length * 6.1 + 10;
      const flip = x + tw > L + W + R - 2;
      const x0 = flip ? x - tw : x;
      const row = ends[0] <= x0 - 3 ? 0 : ends[1] <= x0 - 3 ? 1 : ends[0] <= ends[1] ? 0 : 1;
      ends[row] = x0 + tw;
      return { ...k, x, x0, tw, row };
    });

  const thrY = yR(threshold);

  return (
    <g>
      {/* Severity strip, one cell per window. */}
      {rows.map((r, i) => (
        <rect
          key={i}
          x={L + i * cw}
          y={PINS}
          width={cw + 0.4}
          height={TRACK}
          fill={sevColor(sevFromRisk(r.risk, threshold))}
          opacity={0.18 + r.risk * 0.72}
        />
      ))}
      <rect x={L} y={PINS} width={W} height={TRACK} fill="none" stroke="var(--rule-hair)" />

      {/* Wave frame. */}
      <rect x={L} y={waveTop} width={W} height={waveH} fill="none" stroke="var(--rule-hair)" />
      <line x1={L} x2={L + W} y1={yR(0.5)} y2={yR(0.5)} stroke="var(--rule-hair)" />

      {/* Packet rate under the risk, so the risk reads first. */}
      {hasRate && <path d={rateLine} fill="none" stroke="var(--paper-600)" strokeWidth={1} strokeLinejoin="round" />}

      {traces.map((tr, i) =>
        tr.line ? (
          <g key={i}>
            <path d={tr.area} fill={RISK_BAND_COLORS[i]} fillOpacity={0.1} />
            <path d={tr.line} fill="none" stroke={RISK_BAND_COLORS[i]} strokeWidth={1.5} strokeLinejoin="round" />
          </g>
        ) : null,
      )}

      <line x1={L} x2={L + W} y1={thrY} y2={thrY} stroke="var(--sev-critical)" strokeDasharray="4 4" opacity={0.7} />

      {/* Scales. */}
      <g className="dv-scale">
        <text x={L - 6} y={yR(1) + 4} textAnchor="end">1.0</text>
        <text x={L - 6} y={thrY + 3.5} textAnchor="end" fill="var(--sev-critical)">
          {threshold.toFixed(2)}
        </text>
        <text x={L - 6} y={yR(0)} textAnchor="end">0</text>
        <text x={L - 6} y={yR(0) - 12} textAnchor="end" className="dv-unit">risk</text>
        {hasRate && (
          <>
            <text x={L + W + 6} y={yP(maxRate) + 4}>{fmtRate(maxRate)}</text>
            <text x={L + W + 6} y={yP(maxRate) + 16} className="dv-unit">pkt/s</text>
            <text x={L + W + 6} y={yR(0)}>0</text>
          </>
        )}
        {ticks.map((t) => {
          const x = L + (t / total) * W;
          return (
            <g key={t}>
              <line x1={x} x2={x} y1={waveTop + waveH} y2={waveTop + waveH + 4} stroke="var(--rule-hard)" />
              <text x={x} y={h - 3} textAnchor={t === 0 ? "start" : "middle"}>
                {tickLabel(t)}
              </text>
            </g>
          );
        })}
      </g>

      {/* Keyframe pins. */}
      {pins.map((p) => (
        <g key={`${p.tag}-${p.index}`} data-kf={p.index} className="dv-pin">
          <title>{`${tickLabel(rows[p.index].window * windowSeconds)} · ${p.label}`}</title>
          <line x1={p.x} x2={p.x} y1={p.row * 13 + 12} y2={PINS + TRACK} stroke={p.color} />
          <rect x={p.x0} y={p.row * 13} width={p.tw} height={12} fill="var(--ink-000)" stroke={p.color} />
          <text x={p.x0 + 5} y={p.row * 13 + 9} fill={p.color}>
            {p.tag}
          </text>
        </g>
      ))}
    </g>
  );
});

export default function DvrTimeline({
  rows,
  keyframes,
  pos,
  threshold,
  windowSeconds,
  onSeek,
  onScrub,
}: {
  rows: ReplayRow[];
  keyframes: Keyframe[];
  /** Playhead, in rows: floor is the current window, the fraction is time through it. */
  pos: number;
  threshold: number;
  windowSeconds: number;
  onSeek: (pos: number) => void;
  /** True while a drag holds the playhead. */
  onScrub: (active: boolean) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [size, setSize] = useState<Geometry>({ w: 0, h: 0 });
  const [hover, setHover] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.round(e.contentRect.width), h: Math.round(e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = rows.length;
  const { w, h } = size;
  const W = Math.max(1, w - L - R);
  const waveTop = PINS + TRACK + GAP;
  const waveH = Math.max(40, h - waveTop - AXIS);
  const at = (p: number) => L + (p / Math.max(1, n)) * W;

  const fromClientX = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return 0;
    return Math.max(0, Math.min(n - 1e-3, ((clientX - r.left - L) / W) * n));
  };

  const px = at(pos);
  const hv = hover != null ? rows[hover] : null;
  const hx = hover != null ? at(hover + 0.5) : 0;
  const rate = hv?.packets != null ? hv.packets / windowSeconds : null;

  return (
    <div
      ref={box}
      className="dv"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        const pin = (e.target as Element).closest?.("[data-kf]");
        e.currentTarget.setPointerCapture(e.pointerId);
        dragging.current = true;
        setHover(null);
        onScrub(true);
        // A pin lands at the end of its window, with the window's records shown.
        onSeek(pin ? Number(pin.getAttribute("data-kf")) + 0.999 : fromClientX(e.clientX));
      }}
      onPointerMove={(e) => {
        if (dragging.current) {
          onSeek(fromClientX(e.clientX));
          return;
        }
        setHover(Math.min(n - 1, Math.floor(fromClientX(e.clientX))));
      }}
      onPointerUp={(e) => {
        if (!dragging.current) return;
        dragging.current = false;
        e.currentTarget.releasePointerCapture(e.pointerId);
        onScrub(false);
      }}
      onPointerCancel={() => {
        if (!dragging.current) return;
        dragging.current = false;
        onScrub(false);
      }}
      onPointerLeave={() => setHover(null)}
    >
      {w > 0 && n > 0 && (
        <svg width={w} height={h} className="dv-svg">
          <Static rows={rows} keyframes={keyframes} w={w} h={h} threshold={threshold} windowSeconds={windowSeconds} />

          {/* Not yet played. */}
          <rect x={px} y={PINS} width={Math.max(0, L + W - px)} height={TRACK + GAP + waveH} className="dv-ahead" />

          {hv && (
            <g pointerEvents="none">
              <line x1={hx} x2={hx} y1={PINS} y2={waveTop + waveH} stroke="var(--paper-400)" strokeDasharray="2 2" />
              <rect
                x={hx - 3}
                y={waveTop + (1 - hv.risk) * waveH - 3}
                width={6}
                height={6}
                fill={sevColor(sevFromRisk(hv.risk, threshold))}
              />
            </g>
          )}

          <g pointerEvents="none">
            <line x1={px} x2={px} y1={PINS - 3} y2={waveTop + waveH} stroke="var(--paper-000)" strokeWidth={2} />
            <rect x={px - 5} y={PINS + 1} width={10} height={10} fill="var(--paper-000)" />
          </g>
        </svg>
      )}

      {hv && (
        <div className={`dv-read${hx > w - 250 ? " is-left" : ""}`} style={{ left: hx, top: waveTop + 6 }}>
          <b>W {String(hv.window).padStart(3, "0")}</b>
          <span>{tickLabel(hv.window * windowSeconds)}</span>
          <span style={{ color: sevColor(sevFromRisk(hv.risk, threshold)) }}>risk {hv.risk.toFixed(3)}</span>
          {rate != null && <span>{Math.round(rate).toLocaleString("en-US")} pkt/s</span>}
          {hv.mitre_technique && <span>{hv.mitre_technique.split(" ")[0]}</span>}
        </div>
      )}
    </div>
  );
}
