/**
 * Time-travel scrubber.
 *
 * One track, two scales: the recorded windows on the left, the forecast
 * horizon on the right of a hard NOW rule. The recorded part carries the risk
 * of every window as a heat strip and a trace; the horizon carries the
 * rollout's forecast, hatched and dashed like every forecast in the console.
 * Dragging moves the global cursor, so every view follows.
 */

import { useCallback, useRef } from "react";
import type { ForecastPoint } from "../api/types";
import { Square, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, OBSERVED } from "../design/charts";
import { HORIZON_SECONDS, WINDOW_SECONDS, type Cursor, type Frame } from "../types/timeline";

/** Share of the track given to the recorded windows; the rest is the horizon. */
const OBSERVED_SHARE = 0.8;
/** −10s, in windows. */
const BACK_STEP = 10 / WINDOW_SECONDS;

export default function ScrubDock({
  frames,
  viewIdx,
  cursor,
  playing,
  forecast,
  threshold,
  onCursor,
  onPlaying,
}: {
  frames: Frame[];
  viewIdx: number;
  cursor: Cursor;
  playing: boolean;
  /** The live window's rollout — what the horizon shows. */
  forecast: ForecastPoint[];
  threshold: number;
  onCursor: (c: Cursor) => void;
  onPlaying: (p: boolean) => void;
}) {
  const track = useRef<HTMLDivElement>(null);
  const n = frames.length;
  const liveIdx = n - 1;
  const span = Math.max(1, n - 1);

  // Position along the track, 0–1.
  const obsAt = (i: number) => (i / span) * OBSERVED_SHARE;
  const futAt = (s: number) => OBSERVED_SHARE + (s / HORIZON_SECONDS) * (1 - OBSERVED_SHARE);

  const head =
    cursor.kind === "future" ? futAt(cursor.seconds) : cursor.kind === "past" ? obsAt(viewIdx) : OBSERVED_SHARE;

  const fromX = useCallback(
    (clientX: number): Cursor => {
      const r = track.current?.getBoundingClientRect();
      if (!r || n === 0) return { kind: "live" };
      const u = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      if (u > OBSERVED_SHARE + 0.004) {
        const s = ((u - OBSERVED_SHARE) / (1 - OBSERVED_SHARE)) * HORIZON_SECONDS;
        const snapped = Math.max(WINDOW_SECONDS, Math.min(HORIZON_SECONDS, Math.round(s / WINDOW_SECONDS) * WINDOW_SECONDS));
        return { kind: "future", seconds: snapped };
      }
      const i = Math.round((u / OBSERVED_SHARE) * span);
      return i >= liveIdx ? { kind: "live" } : { kind: "past", seq: frames[i].seq };
    },
    [frames, n, span, liveIdx]
  );

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    onPlaying(false);
    onCursor(fromX(e.clientX));
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    onCursor(fromX(e.clientX));
  };

  const stepBy = (windows: number) => {
    onPlaying(false);
    if (cursor.kind === "future") {
      const s = cursor.seconds + windows * WINDOW_SECONDS;
      onCursor(s <= 0 ? { kind: "live" } : { kind: "future", seconds: Math.min(HORIZON_SECONDS, s) });
      return;
    }
    const from = cursor.kind === "past" ? viewIdx : liveIdx;
    const i = Math.max(0, Math.min(liveIdx, from + windows));
    onCursor(i >= liveIdx ? { kind: "live" } : { kind: "past", seq: frames[i].seq });
  };

  const peak = forecast.length ? forecast.reduce((a, f) => (f.predicted > a.predicted ? f : a), forecast[0]) : null;

  const togglePlay = () => {
    if (playing) {
      onPlaying(false);
      return;
    }
    if (cursor.kind === "live") {
      // Pause on the present: freeze this window while the stream continues.
      if (n) onCursor({ kind: "past", seq: frames[liveIdx].seq });
      return;
    }
    onPlaying(true);
  };

  // Readout for wherever the cursor is.
  let mode: "live" | "replay" | "forecast" | "paused" = "live";
  let when = "";
  let riskAt: number | null = null;
  let band: [number, number] | null = null;
  if (cursor.kind === "future") {
    mode = "forecast";
    const f = forecast.find((p) => p.horizonSeconds >= cursor.seconds) ?? forecast[forecast.length - 1];
    when = `t +${cursor.seconds}s`;
    if (f) {
      riskAt = f.predicted / 100;
      band = [f.lowerBound / 100, f.upperBound / 100];
    }
  } else {
    const f = frames[cursor.kind === "past" ? viewIdx : liveIdx];
    const back = (liveIdx - (cursor.kind === "past" ? viewIdx : liveIdx)) * WINDOW_SECONDS;
    mode = cursor.kind === "live" ? "live" : back === 0 ? "paused" : "replay";
    when = back === 0 ? f?.point.label ?? "" : `t −${back}s · ${f?.point.label ?? ""}`;
    riskAt = f ? f.point.risk : null;
  }

  const recordedSeconds = span * WINDOW_SECONDS;
  const ticks = [1, 2 / 3, 1 / 3].map((k) => Math.round((recordedSeconds * k) / 10) * 10).filter((v, i, a) => v > 0 && a.indexOf(v) === i);

  // Trace across the whole track: recorded risk, then the rollout.
  const trace = frames.map((f, i) => `${(obsAt(i) * 1000).toFixed(1)},${(40 - f.point.risk * 36).toFixed(1)}`).join(" ");
  const fcTrace = [
    ...(n ? [`${(OBSERVED_SHARE * 1000).toFixed(1)},${(40 - frames[liveIdx].point.risk * 36).toFixed(1)}`] : []),
    ...forecast.map((f) => `${(futAt(f.horizonSeconds) * 1000).toFixed(1)},${(40 - (f.predicted / 100) * 36).toFixed(1)}`),
  ].join(" ");

  return (
    <div className="sd">
      <div className="sd-controls">
        <button className="seg" onClick={() => stepBy(-BACK_STEP)} title="Back 10 seconds">
          −10s
        </button>
        <button className="seg" aria-pressed={playing} onClick={togglePlay} title={playing ? "Pause" : "Play"}>
          {playing ? "Pause" : cursor.kind === "live" ? "Pause" : "Play"}
        </button>
        <button className="seg" aria-pressed={cursor.kind === "live"} onClick={() => { onPlaying(false); onCursor({ kind: "live" }); }}>
          Live
        </button>
        <button
          className="seg"
          onClick={() => {
            onPlaying(false);
            if (peak) onCursor({ kind: "future", seconds: peak.horizonSeconds });
          }}
          disabled={!peak}
          title="Jump to the forecast peak"
        >
          Peak {peak ? `+${peak.horizonSeconds}s` : ""}
        </button>
      </div>

      <div
        ref={track}
        className="sd-track"
        role="slider"
        aria-label="Time cursor"
        aria-valuemin={-recordedSeconds}
        aria-valuemax={HORIZON_SECONDS}
        aria-valuenow={cursor.kind === "future" ? cursor.seconds : -(liveIdx - viewIdx) * WINDOW_SECONDS}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") stepBy(-1);
          else if (e.key === "ArrowRight") stepBy(1);
        }}
      >
        {/* Recorded windows: a heat strip, one cell per window. */}
        <div className="sd-observed" style={{ width: `${OBSERVED_SHARE * 100}%` }}>
          {frames.map((f, i) => (
            <i
              key={f.seq}
              style={{
                left: `${(i / span) * 100}%`,
                width: `${100 / span + 0.2}%`,
                background: sevColor(sevFromRisk(f.point.risk, threshold)),
                opacity: 0.18 + f.point.risk * 0.55,
              }}
            />
          ))}
        </div>
        {/* The horizon: forecast, hatched. */}
        <div className="sd-horizon" style={{ left: `${OBSERVED_SHARE * 100}%`, width: `${(1 - OBSERVED_SHARE) * 100}%` }} />

        <svg className="sd-trace" viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden>
          <line x1={0} x2={1000} y1={40 - threshold * 36} y2={40 - threshold * 36} stroke="var(--rule-hard)" strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />
          <polyline points={trace} fill="none" stroke={OBSERVED} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
          <polyline points={fcTrace} fill="none" stroke={FORECAST} strokeWidth={1.5} strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
        </svg>

        <div className="sd-now" style={{ left: `${OBSERVED_SHARE * 100}%` }} />

        {/* The playhead. */}
        <div className={`sd-head is-${mode}`} style={{ left: `${head * 100}%` }}>
          <span />
        </div>
      </div>

      <div className="sd-scale">
        {ticks.map((s) => (
          <span key={s} style={{ left: `${(1 - s / Math.max(1, recordedSeconds)) * OBSERVED_SHARE * 100}%` }}>
            −{s}s
          </span>
        ))}
        <span className="is-now" style={{ left: `${OBSERVED_SHARE * 100}%` }}>
          NOW
        </span>
        <span style={{ left: `${futAt(8) * 100}%` }}>+8s</span>
        <span style={{ left: `${futAt(16) * 100}%` }}>+16s</span>
      </div>

      <div className="sd-readout">
        <span className={`sd-mode is-${mode}`}>
          {mode === "live" && <Square live status="nominal" />}
          {mode}
        </span>
        <span className="sd-when">{when}</span>
        {riskAt != null && (
          <span className="sd-risk" style={{ color: sevColor(sevFromRisk(riskAt, threshold)) }}>
            {riskAt.toFixed(3)}
            {band && (
              <em>
                {" "}
                [{band[0].toFixed(2)}–{band[1].toFixed(2)}]
              </em>
            )}
          </span>
        )}
      </div>
    </div>
  );
}
