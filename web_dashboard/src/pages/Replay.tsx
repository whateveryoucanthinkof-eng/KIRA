import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { uploadReplay, analyseSample, fetchReplaySamples } from "../api/adapter";
import type { ReplaySample } from "../types/replay";
import type { ReplayReport, ReplayRow } from "../types/replay";
import { BarRow, Btn, Chip, Field, Micro, PanelHead, Readout, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Legend } from "../design/charts";
import DvrTimeline, { fmtClock, fmtRate, type Keyframe } from "../components/DvrTimeline";
import FlowStream from "../components/FlowStream";
import { WINDOW_SECONDS } from "../types/timeline";
import { ahead } from "../design/time";

/**
 * Offline capture analysis, played back as a DVR.
 *
 * PS 26153 asks for an interface that accepts a PCAP or CSV and runs fully
 * offline. `POST /api/replay` does exactly that — local parse, window-by-window
 * scoring, temp file deleted — and the backend forces the SOC rule layer off,
 * so every risk on this page is pure model output with no heuristic floor.
 *
 * The analysis runs once; the DVR then walks its windows at capture time or
 * faster, so the scored risk, the flow records and the model's reading of each
 * window can be watched in order, scrubbed, and jumped between keyframes.
 */

const ACCEPT = ".pcap,.pcapng,.csv,.binetflow";
/** Fallback alert cut, used only when a report carries no threshold. */
const THRESHOLD = 0.65;
const SPEEDS = [1, 2, 4, 8] as const;
type Speed = (typeof SPEEDS)[number];
/** Long enough to read the parse on camera; the real endpoint is usually slower. */
const MIN_PARSE_MS = 1700;
/** How long the finished parse stays up before the DVR takes over. */
const DONE_HOLD_MS = 650;
const CELLS = 48;

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

const pad3 = (n: number) => String(n).padStart(3, "0");
const techId = (t: string | null) => (t ? t.split(" ")[0] : null);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Keyframes: the first window of each technique, the first alert, the peak,
 * and the return below threshold. Derived from the report, never scripted.
 */
function keyframesOf(rows: ReplayRow[], threshold: number): Keyframe[] {
  const out: Keyframe[] = [];
  const seen = new Set<string>();
  let firstAlert = -1;
  let lastAlert = -1;
  let peak = 0;
  rows.forEach((r, i) => {
    if (r.risk > rows[peak].risk) peak = i;
    if (r.alert) {
      if (firstAlert < 0) firstAlert = i;
      lastAlert = i;
    }
    const lane = r.tactic_lane ?? r.stage;
    const key = r.mitre_technique ?? (lane !== "Benign" ? lane : null);
    if (key && !seen.has(key)) {
      seen.add(key);
      out.push({
        index: i,
        tag: techId(key) ?? key,
        label: `${lane} · ${key}`,
        color: sevColor(sevFromRisk(r.risk, threshold)),
      });
    }
  });
  if (firstAlert >= 0) {
    out.push({ index: firstAlert, tag: "ALERT", label: `first window over ${threshold.toFixed(2)} · ${rows[firstAlert].risk.toFixed(3)}`, color: "var(--sev-critical)" });
  }
  if (rows.length) {
    const p = rows[peak];
    out.push({ index: peak, tag: "PEAK", label: `highest score in the capture · ${p.risk.toFixed(3)}`, color: sevColor(sevFromRisk(p.risk, threshold)) });
  }
  if (lastAlert >= 0 && lastAlert < rows.length - 1) {
    out.push({ index: lastAlert + 1, tag: "CLEAR", label: "back under threshold for the rest of the capture", color: "var(--sev-nominal)" });
  }
  return out.sort((a, b) => a.index - b.index);
}

/* ── Transport glyphs: drawn, not an icon set ─────────────────────────── */

function Glyph({ kind }: { kind: "play" | "pause" | "rewind" | "reset" | "prev" | "next" }) {
  const d = {
    play: "M3 1.5L10.5 6L3 10.5Z",
    pause: "M2.5 1.5H5V10.5H2.5ZM7 1.5H9.5V10.5H7Z",
    rewind: "M6 1.5L0.5 6L6 10.5ZM11.5 1.5L6 6L11.5 10.5Z",
    reset: "M1.5 1.5H3V10.5H1.5ZM11 1.5L3.5 6L11 10.5Z",
    prev: "M5.5 1.5L0.5 6L5.5 10.5ZM9 3L12 6L9 9L6 6Z",
    next: "M6.5 1.5L11.5 6L6.5 10.5ZM3 3L6 6L3 9L0 6Z",
  }[kind];
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
      <path d={d} fill="currentColor" />
    </svg>
  );
}

/* ── Session survives leaving the page ────────────────────────────────── */

interface Session {
  report: ReplayReport;
  size: number;
  pos: number;
  speed: Speed;
}
let saved: Session | null = null;

interface Parsing {
  id: number;
  name: string;
  size: number;
  kind: "pcap" | "csv";
  /** Known for the built-in captures; estimated from size for an upload. */
  windows: number;
  estimated: boolean;
  t0: number;
  result: ReplayReport | null;
}

export default function Replay() {
  const [report, setReport] = useState<ReplayReport | null>(saved?.report ?? null);
  const [size, setSize] = useState(saved?.size ?? 0);
  const [pos, setPos] = useState(saved?.pos ?? 0);
  const [speed, setSpeed] = useState<Speed>(saved?.speed ?? 4);
  const [playing, setPlaying] = useState(false);
  const [dir, setDir] = useState<1 | -1>(1);
  const [parsing, setParsing] = useState<Parsing | null>(null);
  const [now, setNow] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const posRef = useRef(pos);
  const runId = useRef(0);
  const resume = useRef(false);

  const seek = useCallback((p: number) => {
    posRef.current = p;
    setPos(p);
  }, []);

  // Keep the session for the next visit.
  useEffect(() => {
    saved = report ? { report, size, pos: posRef.current, speed } : null;
  });
  useEffect(
    () => () => {
      runId.current++;
      if (saved) saved.pos = posRef.current;
    },
    []
  );

  /* ── Intake ──────────────────────────────────────────────────────────── */

  async function run(name: string, bytes: number, windows: number | null, work: () => Promise<ReplayReport>) {
    const id = ++runId.current;
    const suffix = name.toLowerCase().split(".").pop() ?? "";
    const est = windows ?? Math.max(12, Math.min(200, Math.round(bytes / 8192) || 48));
    const t0 = performance.now();
    setError(null);
    setPlaying(false);
    setParsing({
      id,
      name,
      size: bytes,
      kind: suffix === "csv" || suffix === "binetflow" ? "csv" : "pcap",
      windows: est,
      estimated: windows == null,
      t0,
      result: null,
    });
    try {
      const rep = await work();
      await sleep(Math.max(0, MIN_PARSE_MS - (performance.now() - t0)));
      if (runId.current !== id) return;
      setParsing((p) => (p && p.id === id ? { ...p, result: rep } : p));
      await sleep(DONE_HOLD_MS);
      if (runId.current !== id) return;
      setParsing(null);
      setReport(rep);
      setSize(bytes);
      setDir(1);
      seek(0);
      setPlaying(true);
    } catch (e) {
      if (runId.current !== id) return;
      setParsing(null);
      setError(e instanceof Error ? e.message : "analysis failed");
    }
  }

  // Drive the parse readout while it runs.
  useEffect(() => {
    if (!parsing) return;
    let raf = 0;
    const tick = () => {
      setNow(performance.now());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [parsing]);

  const analyse = (file: File) => void run(file.name, file.size, null, () => uploadReplay(file));
  const analyseBuiltIn = (s: ReplaySample) => void run(s.name, s.bytes, s.windows, () => analyseSample(s));

  const eject = () => {
    runId.current++;
    setPlaying(false);
    setReport(null);
    setParsing(null);
    seek(0);
  };

  /* ── Playback ────────────────────────────────────────────────────────── */

  const rows = useMemo(() => report?.results ?? [], [report]);
  const n = rows.length;

  useEffect(() => {
    if (!playing || !n) return;
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      // A frame's timestamp can precede the effect's clock read; never step backwards on it.
      const dt = Math.max(0, Math.min(0.1, (t - last) / 1000));
      last = Math.max(last, t);
      const p = posRef.current + (dir * dt * speed) / WINDOW_SECONDS;
      if (dir > 0 && p >= n - 1e-3) {
        seek(n - 1e-3);
        setPlaying(false);
        return;
      }
      if (dir < 0 && p <= 0) {
        seek(0);
        setPlaying(false);
        setDir(1);
        return;
      }
      seek(p);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, dir, speed, n, seek]);

  // The alert cut the replay was scored against: Branch A's fitted
  // operating point, not a constant.
  const threshold = report?.threshold ?? THRESHOLD;
  const keyframes = useMemo(() => keyframesOf(rows, threshold), [rows, threshold]);
  const cur = n ? Math.max(0, Math.min(n - 1, Math.floor(pos))) : 0;
  const atEnd = n > 0 && pos >= n - 2e-3;

  const togglePlay = () => {
    if (playing && dir > 0) {
      setPlaying(false);
      return;
    }
    if (atEnd) seek(0);
    setDir(1);
    setPlaying(true);
  };
  const rewind = () => {
    if (playing && dir < 0) {
      setPlaying(false);
      setDir(1);
      return;
    }
    if (pos <= 0) return;
    setDir(-1);
    setPlaying(true);
  };
  const reset = () => {
    setPlaying(false);
    setDir(1);
    seek(0);
  };
  /** Lands at the end of a window, so a paused inspection shows all its records. */
  const land = (i: number) => seek(Math.max(0, Math.min(n - 1e-3, i + 0.999)));
  const stepWindow = (d: number) => {
    setPlaying(false);
    land(cur + d);
  };
  const jumpKey = (d: 1 | -1) => {
    setPlaying(false);
    const k = d > 0 ? keyframes.find((x) => x.index > cur) : [...keyframes].reverse().find((x) => x.index < cur);
    if (k) land(k.index);
    else if (d < 0) seek(0);
    else land(n - 1);
  };

  // Space, arrows, Home/End. Registered once; always calls the latest handlers.
  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e: KeyboardEvent) => {
    if (!report || parsing) return;
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") {
      // A button reached by Tab keeps Space for itself.
      if (el?.tagName === "BUTTON") return;
      e.preventDefault();
      togglePlay();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      if (e.shiftKey) jumpKey(1);
      else stepWindow(1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (e.shiftKey) jumpKey(-1);
      else stepWindow(-1);
    } else if (e.key === "Home") {
      e.preventDefault();
      reset();
    } else if (e.key === "End") {
      e.preventDefault();
      setPlaying(false);
      land(n - 1);
    }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onScrub = useCallback(
    (active: boolean) => {
      if (active) {
        resume.current = playing;
        setPlaying(false);
      } else if (resume.current) {
        resume.current = false;
        setPlaying(true);
      }
    },
    [playing]
  );

  /* ── Render ──────────────────────────────────────────────────────────── */

  if (!report) {
    return (
      <Intake
        parsing={parsing}
        now={now}
        error={error}
        dragging={dragging}
        setDragging={setDragging}
        inputRef={inputRef}
        onFile={analyse}
        onSample={analyseBuiltIn}
      />
    );
  }

  const row = rows[cur];
  const level = sevFromRisk(row.risk, threshold);
  const total = n ? (rows[n - 1].window + 1) * WINDOW_SECONDS : 0;
  const tNow = row.window * WINDOW_SECONDS + (pos - cur) * WINDOW_SECONDS;
  const peakIdx = rows.reduce((b, r, i) => (r.risk > rows[b].risk ? i : b), 0);
  const peak = rows[peakIdx];
  const hasFlows = rows.some((r) => r.flows?.length);
  const records = rows.reduce((s, r) => s + (r.flows?.length ?? 0), 0);
  const packets = rows.reduce((s, r) => s + (r.packets ?? 0), 0);
  const nextKey = keyframes.find((k) => k.index > cur);

  const mode = playing ? (dir > 0 ? "play" : "rewind") : atEnd ? "end" : "paused";
  const modeLabel = { play: "Playing", rewind: "Rewind", end: "End", paused: "Paused" }[mode];

  const flows = row.flows ?? [];
  const onPath = flows.filter((f) => f.on_path).length;
  const rate = row.packets != null ? row.packets / WINDOW_SECONDS : null;
  const mbit = row.bytes != null ? (row.bytes * 8) / WINDOW_SECONDS / 1e6 : null;

  return (
    <div className="rp is-dvr">
      {/* ── Capture ───────────────────────────────────────────────────── */}
      <div className="rp-cap">
        <div className="rp-cap-file">
          <Micro>Capture</Micro>
          <b title={report.filename}>{report.filename}</b>
          <em>
            {report.kind.toUpperCase()} · {fmtBytes(size)} · {n} × {WINDOW_SECONDS.toFixed(1)} s windows · {fmtClock(total)}
          </em>
        </div>
        <div className="rp-cap-stat">
          <Micro>Flagged</Micro>
          <b style={{ color: report.flagged_windows ? "var(--sev-critical)" : "var(--sev-nominal)" }}>{report.flagged_windows}</b>
          <em>{n ? ((report.flagged_windows / n) * 100).toFixed(1) : "0"}% of windows</em>
        </div>
        <div className="rp-cap-stat">
          <Micro>Peak</Micro>
          <b style={{ color: sevColor(sevFromRisk(peak.risk, threshold)) }}>{peak.risk.toFixed(3)}</b>
          <em>
            W {pad3(peak.window)} · {fmtClock(peak.window * WINDOW_SECONDS)}
          </em>
        </div>
        {hasFlows && (
          <div className="rp-cap-stat">
            <Micro>Records</Micro>
            <b>{records.toLocaleString("en-US")}</b>
            <em>{(packets / 1e6).toFixed(2)} M packets</em>
          </div>
        )}
        <div className="rp-cap-side">
          <Chip level="nominal">rules disabled</Chip>
          <button className="rp-eject" onClick={eject} title="Close this capture and load another">
            Eject
          </button>
        </div>
      </div>

      <div className="rp-deck">
        {/* ── Transport ─────────────────────────────────────────────── */}
        {/* Clicks here don't take focus, so Space stays with the player. */}
        <div className="rp-transport" onMouseDown={(e) => e.preventDefault()}>
          <div className="rp-keys">
            <button className="rp-key" onClick={reset} title="Back to the start (Home)" aria-label="Reset">
              <Glyph kind="reset" />
            </button>
            <button className="rp-key" aria-pressed={playing && dir < 0} onClick={rewind} title="Rewind" aria-label="Rewind">
              <Glyph kind="rewind" />
            </button>
            <button
              className="rp-key is-main"
              aria-pressed={playing && dir > 0}
              onClick={togglePlay}
              title="Play / pause (Space)"
              aria-label={playing && dir > 0 ? "Pause" : "Play"}
            >
              <Glyph kind={playing && dir > 0 ? "pause" : "play"} />
            </button>
            <button className="rp-key" onClick={() => jumpKey(-1)} title="Previous keyframe (Shift ←)" aria-label="Previous keyframe">
              <Glyph kind="prev" />
            </button>
            <button className="rp-key" onClick={() => jumpKey(1)} title="Next keyframe (Shift →)" aria-label="Next keyframe">
              <Glyph kind="next" />
            </button>
          </div>

          <div className="seg-group" role="group" aria-label="Playback speed">
            {SPEEDS.map((s) => (
              <button key={s} className="seg" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                {s}×
              </button>
            ))}
          </div>

          <div className="rp-clock">
            <b>{fmtClock(tNow)}</b>
            <em>
              / {fmtClock(total)} · W {pad3(row.window)} / {n}
            </em>
          </div>

          <div className={`rp-mode is-${mode}`}>
            <i />
            {modeLabel}
          </div>

          {nextKey && (
            <div className="rp-next" title={nextKey.label}>
              <Micro>Next</Micro>
              <span style={{ color: nextKey.color }}>{nextKey.tag}</span>
              <em>in {(rows[nextKey.index].window * WINDOW_SECONDS - tNow).toFixed(1)} s</em>
            </div>
          )}
        </div>

        {/* ── Timeline ──────────────────────────────────────────────── */}
        <div className="rp-timeline">
          <PanelHead
            title="Capture timeline"
            note="press or drag to seek · pins jump"
            aside={
              <Legend
                items={[
                  { color: "var(--sev-nominal)", label: "risk", fill: true },
                  { color: "var(--sev-warning)", label: "≥ θ", fill: true },
                  { color: "var(--sev-critical)", label: "critical", fill: true },
                  ...(rows.some((r) => r.packets != null) ? [{ color: "var(--paper-600)", label: "packets / s" }] : []),
                  { color: "var(--sev-critical)", label: `threshold ${threshold.toFixed(2)}`, dashed: true },
                ]}
              />
            }
          />
          <DvrTimeline
            rows={rows}
            keyframes={keyframes}
            pos={pos}
            threshold={threshold}
            windowSeconds={WINDOW_SECONDS}
            onSeek={seek}
            onScrub={onScrub}
          />
        </div>

        {/* ── Flow records ──────────────────────────────────────────── */}
        <div className="rp-stream">
          <PanelHead
            title="Flow records"
            note={hasFlows ? `${flows.length} in window · ${onPath} on the attack path` : undefined}
            aside={
              hasFlows ? (
                <span className="rp-stream-key">
                  <i /> on attack path
                </span>
              ) : undefined
            }
          />
          <FlowStream rows={rows} pos={pos} threshold={threshold} windowSeconds={WINDOW_SECONDS} />
        </div>
      </div>

      {/* ── Window inspector ──────────────────────────────────────────── */}
      <aside className="rp-rail">
        <PanelHead
          title={`Window ${pad3(row.window)}`}
          note={`${fmtClock(row.window * WINDOW_SECONDS)} – ${fmtClock((row.window + 1) * WINDOW_SECONDS)}`}
          aside={row.alert ? <Chip level="critical">alert</Chip> : undefined}
        />
        <div className="rp-rail-body">
          <div className="rp-insp">
            <Readout
              label="Model risk · rules off"
              value={row.risk.toFixed(3)}
              scale="hero"
              level={level}
              sub={row.alert ? `over ${threshold.toFixed(2)} · alert raised` : `under ${threshold.toFixed(2)}`}
            />
          </div>
          <div className="rp-insp">
            <Field label="Target" value={row.target} />
            <Field label="Stage" value={row.tactic_lane ?? row.stage} />
            <Field
              label="Technique"
              value={<span title={row.mitre_technique ?? undefined}>{row.mitre_technique ?? "—"}</span>}
            />
            <Field label="Tactic" value={row.mitre_tactic ?? "—"} />
            {hasFlows && <Field label="Records" value={`${flows.length} · ${onPath} on path`} />}
            {rate != null && <Field label="Volume" value={`${fmtRate(rate)} pkt/s${mbit != null ? ` · ${mbit.toFixed(1)} Mb/s` : ""}`} />}
          </div>

          {row.forecast.length > 0 && (
            <>
              <PanelHead title="Rollout" note={`+${row.forecast[0].horizon_seconds} to +${row.forecast[row.forecast.length - 1].horizon_seconds} s`} />
              <Rollout points={row.forecast} threshold={threshold} />
            </>
          )}

          {row.top_features.length > 0 && (
            <>
              <PanelHead title="Attribution" note="input × gradient" />
              <div className="rp-insp rp-attr">
                {row.top_features.map((f) => (
                  <BarRow key={f.feature} label={f.feature} value={f.score} labelWidth={118} />
                ))}
              </div>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

/* ── The window's rollout: forecast, so cool and hatched ──────────────── */

function Rollout({ points, threshold }: { points: { horizon_seconds: number; risk: number }[]; threshold: number }) {
  const W = 240;
  const H = 56;
  const bw = W / points.length;
  const peak = points.reduce((b, p) => (p.risk > b.risk ? p : b), points[0]);
  return (
    <div className="rp-roll">
      <svg viewBox={`0 0 ${W} ${H + 12}`} width="100%" height={H + 12} preserveAspectRatio="none" aria-hidden>
        <defs>
          <pattern id="rp-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="5" height="5" fill="var(--fill-forecast)" />
            <line x1="0" y1="0" x2="0" y2="5" stroke={FORECAST} strokeWidth="1.4" opacity="0.55" />
          </pattern>
        </defs>
        {points.map((p, i) => {
          const h = Math.max(1, p.risk * H);
          return (
            <rect
              key={p.horizon_seconds}
              x={i * bw + 2}
              y={H - h}
              width={bw - 4}
              height={h}
              fill="url(#rp-hatch)"
              stroke={FORECAST}
              strokeDasharray="3 2"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
        <line x1={0} x2={W} y1={H - threshold * H} y2={H - threshold * H} stroke="var(--sev-critical)" strokeDasharray="4 4" opacity={0.7} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="rp-roll-foot">
        <span>{ahead(points[0].horizon_seconds)}</span>
        <span style={{ color: sevColor(sevFromRisk(peak.risk, threshold)) }}>
          peak {peak.risk.toFixed(3)} at {ahead(peak.horizon_seconds)}
        </span>
        <span>{ahead(points[points.length - 1].horizon_seconds)}</span>
      </div>
    </div>
  );
}

/* ── Intake: the scanner, the parse, the sample library ───────────────── */

function Intake({
  parsing,
  now,
  error,
  dragging,
  setDragging,
  inputRef,
  onFile,
  onSample,
}: {
  parsing: Parsing | null;
  now: number;
  error: string | null;
  dragging: boolean;
  setDragging: (v: boolean) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onFile: (f: File) => void;
  onSample: (s: ReplaySample) => void;
}) {
  // Live: the captures on this machine. Demo: the fixtures.
  const [samples, setSamples] = useState<ReplaySample[] | null>(null);
  useEffect(() => {
    let alive = true;
    fetchReplaySamples()
      .then((s) => alive && setSamples(s))
      .catch(() => alive && setSamples([]));
    return () => {
      alive = false;
    };
  }, []);
  return (
    <div className="rp is-intake">
      <div
        className={`rp-scan${dragging ? " is-over" : ""}${parsing ? " is-parsing" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!parsing) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f && !parsing) onFile(f);
        }}
      >
        <i className="rp-corner is-tl" />
        <i className="rp-corner is-tr" />
        <i className="rp-corner is-bl" />
        <i className="rp-corner is-br" />
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />

        {parsing ? (
          <ParseView p={parsing} now={now} />
        ) : (
          <div className="rp-scan-idle">
            <Micro>Offline analysis · rules disabled · no egress</Micro>
            <div className="t-display-m" style={{ color: "var(--paper-000)" }}>
              {dragging ? "Release to analyse" : "Drop a capture to analyse"}
            </div>
            <p>
              Parsed into {WINDOW_SECONDS.toFixed(1)} s windows and scored locally through TGNE → Branch A/B → DeepOP, then
              played back window by window. Nothing leaves this machine.
            </p>
            <div className="rp-scan-actions">
              <Btn primary onClick={() => inputRef.current?.click()}>
                Select file
              </Btn>
              <span>.pcap · .pcapng · .csv · .binetflow</span>
            </div>
          </div>
        )}
      </div>

      <div className="rp-lib">
        <div className="rp-lib-head">
          <Micro>Sample captures</Micro>
          <span>one click · the same analysis path as a dropped file</span>
        </div>
        <div className="rp-samples">
          {samples != null && !samples.length && (
            <span className="rp-sample-note">No captures in the repo's captures/ folder. Drop a .pcap or .csv to analyse it.</span>
          )}
          {(samples ?? []).map((s) => {
            const active = parsing?.name === s.name;
            return (
              <button
                key={s.id}
                disabled={!!parsing}
                onClick={() => onSample(s)}
                className={`rp-sample${active ? " is-active" : ""}${s.id === "benign" ? " is-benign" : ""}`}
              >
                <span className="rp-sample-head">
                  <b>{s.label}</b>
                  <em>{s.kind.toUpperCase()}</em>
                </span>
                <span className="rp-sample-file">{s.name}</span>
                <span className="rp-sample-meta">
                  {fmtBytes(s.bytes)}
                  {s.windows != null ? ` · ${s.windows} windows · ${fmtClock(s.windows * WINDOW_SECONDS)}` : ""}
                </span>
                <span className="rp-sample-note">{s.note}</span>
                <span className="rp-sample-src">{active ? (parsing?.result ? "Analysed" : "Parsing…") : s.source}</span>
              </button>
            );
          })}
        </div>
      </div>

      {error && <div className="rp-error">{error}</div>}
    </div>
  );
}

function ParseView({ p, now }: { p: Parsing; now: number }) {
  const expected = Math.max(MIN_PARSE_MS, 650 + Math.min(1500, p.windows * 7));
  const done = p.result;
  const prog = done ? 1 : Math.min(0.95, Math.max(0, (now - p.t0) / expected));
  const windows = done ? done.results.length : p.windows;
  const lit = Math.floor(prog * CELLS);

  const closed = Math.round(windows * Math.max(0, Math.min(1, (prog - 0.3) / 0.45)));
  const scored = Math.round(windows * Math.max(0, Math.min(1, (prog - 0.75) / 0.25)));
  const steps: { at: number; label: string; detail: string }[] = [
    { at: 0.08, label: "Capture read", detail: `${p.kind === "pcap" ? "packet capture" : "flow records"} · ${fmtBytes(p.size)}` },
    { at: 0.3, label: p.kind === "pcap" ? "Packets → flows" : "Records → flows", detail: p.kind === "pcap" ? "the sensor's own flow table" : "CIC-IDS / CTU-13 adapter" },
    { at: 0.75, label: `${WINDOW_SECONDS.toFixed(1)} s windows`, detail: `${pad3(closed)} / ${p.estimated && !done ? "~" : ""}${windows}` },
    { at: 1, label: "Scoring", detail: `TGNE → Branch A/B → DeepOP · ${pad3(scored)} / ${windows}` },
  ];

  return (
    <div className="rp-parse">
      <div className="rp-parse-head">
        <Micro>{done ? "Analysed" : "Parsing"}</Micro>
        <b>{p.name}</b>
      </div>
      <div className="rp-parse-line">
        <span>
          {done
            ? `${done.results.length} windows scored · ${done.flagged_windows} flagged`
            : `Parsing ${p.estimated ? "~" : ""}${windows} windowed frames…`}
        </span>
        <em>{Math.round(prog * 100)}%</em>
      </div>
      <div className="rp-cells" aria-hidden>
        {Array.from({ length: CELLS }, (_, i) => (
          <i key={i} className={i < lit ? "is-on" : undefined} />
        ))}
      </div>
      <ol className="rp-steps">
        {steps.map((s, i) => {
          const state = prog >= s.at ? "done" : i === 0 || prog >= steps[i - 1].at ? "run" : "wait";
          return (
            <li key={s.label} className={`is-${state}`}>
              <span>{state === "done" ? "✓" : state === "run" ? "▸" : "·"}</span>
              <b>{s.label}</b>
              <em>{s.detail}</em>
            </li>
          );
        })}
      </ol>
      {done && (
        <div className="rp-parse-done" style={{ color: done.flagged_windows ? "var(--sev-critical)" : "var(--sev-nominal)" }}>
          {done.flagged_windows
            ? `peak ${Math.max(...done.results.map((r) => r.risk)).toFixed(3)} · starting playback`
            : "no window over threshold · starting playback"}
        </div>
      )}
    </div>
  );
}
