/**
 * Forecast stage.
 *
 * One screen for the question the project exists to answer — where is this
 * intrusion going — laid out for a presenter: the attack storyline across the
 * top, a large stage that switches between the network, the competing futures
 * and the risk trajectory, the model's reasoning in the inspector, and a
 * time-travel scrubber along the bottom that moves every view at once.
 */

import { lazy, Suspense, useState } from "react";
import { Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip } from "recharts";
import type { ForecastPoint, PredictionResult, Topology } from "../api/types";
import type { Campaign } from "../types/campaign";
import type { FlowRecord, StateDim } from "../types/evidence";
import type { ForecastBranch } from "../types/forecast";
import type { AttentionMatrix as Attention } from "../types/attention";
import type { LivePoint, PredictionEnvelope } from "../types/live";
import type { Cursor, Frame } from "../types/timeline";
import { Chip, Data, Empty, Micro, Panel, PanelHead, Square, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Grid, NowLine, RISK_BAND_COLORS, splitRisk, ThresholdLine, Tip, TimeAxis, ValueAxis } from "../design/charts";
import { Num } from "../design/motion";
import Storyline from "../components/Storyline";
import ForecastTree from "../components/ForecastTree";
import AttentionMatrix from "../components/AttentionMatrix";
import ScrubDock from "../components/ScrubDock";
import StateInspector from "../components/StateInspector";
import Boundary, { hasWebGL } from "../components/Boundary";
import { ahead } from "../design/time";
import { IS_DEMO } from "../env";

const Network3D = lazy(() => import("../components/Network3D"));

type Mode = "topology" | "tree" | "trajectory" | "attention";

const MODES: { id: Mode; label: string }[] = [
  { id: "tree", label: "Forecast tree" },
  { id: "attention", label: "Attention" },
  { id: "trajectory", label: "Trajectory" },
  { id: "topology", label: "3D" },
];

export interface StageView {
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  forecast: ForecastPoint[];
  campaign: Campaign | null;
  topology: Topology | null;
  stateVector: StateDim[] | null;
  flows: FlowRecord[];
  flowsInWindow: number | null;
  history: LivePoint[];
  branches: ForecastBranch[] | null;
  attention: Attention | null;
}

/** Observed history into the rollout, with the forecast's band. */
function Trajectory({
  history,
  forecast,
  threshold,
  futureSeconds,
}: {
  history: LivePoint[];
  forecast: ForecastPoint[];
  threshold: number;
  futureSeconds: number | null;
}) {
  if (!history.length) return <Empty hint="Risk history accumulates from the prediction stream.">No inference history</Empty>;
  const last = history[history.length - 1];
  // The observed curve wears the severity ladder — green below θ, amber from
  // the alert threshold, red in the critical band.
  const data = splitRisk(
    [
      ...history.map((p, i) => ({
        t: p.label,
        observed: p.risk,
        // Join the forecast to the last observed point so the two read as one line.
        forecast: i === history.length - 1 ? p.risk : undefined,
      })),
      ...forecast.map((f) => ({
        t: `${ahead(f.horizonSeconds)}`,
        forecast: f.predicted / 100,
        band: [f.lowerBound / 100, f.upperBound / 100] as [number, number],
      })),
    ],
    threshold,
    (r) => (r as { observed?: number }).observed,
  );

  return (
    <div className="st-traj">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 22, right: 20, left: 0, bottom: 4 }}>
          <Grid />
          <TimeAxis count={data.length} />
          <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
          <Tooltip content={<Tip fmt={(v) => (Array.isArray(v) ? `${Number(v[0]).toFixed(2)}–${Number(v[1]).toFixed(2)}` : Number(v).toFixed(3))} />} cursor={{ stroke: "var(--rule-hard)" }} />
          <ThresholdLine y={threshold} />
          <Area type="monotone" dataKey="band" name="band" stroke="none" fill={FORECAST} fillOpacity={0.14} isAnimationActive={false} />
          {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
            <Area key={key} type="monotone" dataKey={key} name="observed" stroke="none" fill={RISK_BAND_COLORS[i]} fillOpacity={0.1} isAnimationActive={false} />
          ))}
          {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
            <Line key={key} type="monotone" dataKey={key} name="observed" stroke={RISK_BAND_COLORS[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
          ))}
          <Line
            type="monotone"
            dataKey="forecast"
            name="forecast"
            stroke={FORECAST}
            strokeWidth={2}
            strokeDasharray="4 3"
            dot={{ r: 2.5, fill: FORECAST, stroke: "none" }}
            isAnimationActive={false}
            connectNulls
          />
          <NowLine x={last.label} />
          {futureSeconds != null && (
            <ReferenceLine
              x={`${ahead(futureSeconds)}`}
              stroke="var(--paper-000)"
              strokeDasharray="3 3"
              label={{ value: `t ${ahead(futureSeconds)}`, position: "insideTopRight", fill: "var(--paper-000)", fontSize: 10, fontFamily: "var(--face-data)" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** The attack in progress, by name — the first thing anyone watching needs to know. */
function ActiveAttack({
  prediction,
  envelope,
  topology,
}: {
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  topology: Topology | null;
}) {
  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);
  const level = String(prediction?.alert_level ?? sevFromRisk(risk, threshold));
  const technique = prediction?.mitre_technique ?? null;
  const benign = !prediction || !technique || technique === "Benign" || prediction.predicted_stage === "Benign";
  const target = envelope?.target_ip ?? null;
  const targetName = target ? topology?.nodes.find((n) => n.ip === target)?.label : undefined;
  const colour = benign ? "var(--sev-nominal)" : sevColor(level);

  return (
    <div className="st-now" style={{ borderLeftColor: colour }}>
      <div className="st-now-label">
        <Square status={benign ? "nominal" : level} live={!benign} />
        <span>{benign ? "No active attack" : "Active attack"}</span>
        <b style={{ color: colour }}>
          {risk.toFixed(3)} {benign ? "" : level.toUpperCase()}
        </b>
      </div>
      <div className="st-now-name" title={technique ?? undefined} style={{ color: benign ? "var(--paper-400)" : "var(--paper-000)" }}>
        {benign ? "Baseline traffic" : technique}
      </div>
      <div className="st-now-meta" title={target ? `${targetName ?? ""} ${target}` : undefined}>
        {benign ? (
          <>below threshold {threshold.toFixed(2)}</>
        ) : (
          <>
            {prediction?.mitre_tactic ?? prediction?.predicted_stage} → {targetName ?? target ?? "—"}
          </>
        )}
      </div>
    </div>
  );
}

export default function Stage({
  view,
  frames,
  viewIdx,
  cursor,
  playing,
  onCursor,
  onPlaying,
  onEmulate,
}: {
  view: StageView;
  frames: Frame[];
  viewIdx: number;
  cursor: Cursor;
  playing: boolean;
  onCursor: (c: Cursor) => void;
  onPlaying: (p: boolean) => void;
  onEmulate: (kind: "c2" | "flood" | "contain") => void;
}) {
  const [mode, setModeState] = useState<Mode>(() => {
    try {
      const saved = localStorage.getItem("st_view");
      return saved === "topology" || saved === "trajectory" || saved === "attention" ? saved : "tree";
    } catch {
      return "tree";
    }
  });
  const setMode = (m: Mode) => {
    setModeState(m);
    try {
      localStorage.setItem("st_view", m);
    } catch {
      /* per-viewer convenience only */
    }
  };
  const [selected, setSelected] = useState<string | null>(null);

  // The inspector — verdict, rule override, the 27-D state — is open unless the viewer closes it.
  const [railOpen, setRailOpenState] = useState<boolean>(() => {
    try {
      return localStorage.getItem("st_inspector_v2") !== "closed";
    } catch {
      return true;
    }
  });
  const toggleRail = () => {
    setRailOpenState((open) => {
      try {
        localStorage.setItem("st_inspector_v2", open ? "closed" : "open");
      } catch {
        /* per-viewer convenience only */
      }
      return !open;
    });
  };

  const { prediction, envelope, forecast, campaign, branches } = view;
  const futureSeconds = cursor.kind === "future" ? cursor.seconds : null;
  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);
  const level = prediction?.alert_level ?? sevFromRisk(risk, threshold);
  const peak = Number(prediction?.max_future_risk ?? risk);
  const lead = branches?.find((b) => b.id === "A") ?? null;
  const liveIdx = frames.length - 1;
  const latestWindow = envelope?.state?.window_id ?? null;

  return (
    <div className="st">
      <div className="st-sheet sheet">
      {/* ── Storyline ─────────────────────────────────────────────────── */}
      <Panel flush className="st-story">
        <ActiveAttack prediction={prediction} envelope={envelope} topology={view.topology} />
        <Storyline prediction={prediction} forecast={forecast} campaign={campaign} branches={branches} futureSeconds={futureSeconds} />
      </Panel>

      <div className={railOpen ? "st-body has-rail" : "st-body"}>
        {/* ── Stage ───────────────────────────────────────────────────── */}
        <Panel flush clip className="st-main">
          <div className="st-head">
            <div className="st-tabs" role="tablist" aria-label="Stage view">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  role="tab"
                  className="tab"
                  aria-selected={mode === m.id}
                  onClick={() => setMode(m.id)}
                  disabled={m.id === "topology" && !hasWebGL()}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <div className="st-head-end">
              {/* Demo build: scenario buttons steer the sample stream. The real
                  console never launches traffic, so it keeps only Contain,
                  which records an isolation through /api/mitigate. */}
              <span className="emu" role="group" aria-label={IS_DEMO ? "Demo scenarios" : "Response"}>
                <span className="emu-label">{IS_DEMO ? "Emulate" : "Respond"}</span>
                {IS_DEMO && (
                  <>
                    <button className="emu-btn" onClick={() => onEmulate("c2")} title="Demo scenario: C2 beaconing">
                      C2 surge
                    </button>
                    <button className="emu-btn" onClick={() => onEmulate("flood")} title="Demo scenario: volumetric flood">
                      Flood
                    </button>
                  </>
                )}
                <button className="emu-btn is-contain" onClick={() => onEmulate("contain")} title="Record an isolation of the primary target">
                  Contain
                </button>
              </span>
              <button className="seg" aria-pressed={railOpen} onClick={toggleRail} title="Verdict, rule override and the 27-D state vector">
                Inspector
              </button>
            </div>
          </div>
          <div className="st-view">
            {mode === "tree" && <ForecastTree branches={branches} prediction={prediction} futureSeconds={futureSeconds} />}
            {mode === "attention" && <AttentionMatrix matrix={view.attention} focus={envelope?.focus_ips ?? []} />}
            {mode === "trajectory" && (
              <Trajectory history={view.history} forecast={forecast} threshold={threshold} futureSeconds={futureSeconds} />
            )}
            {mode === "topology" && (
              <Boundary fallback={<Empty hint="WebGL is unavailable; the Network view has a 2D graph.">3D view unavailable</Empty>}>
                <Suspense fallback={<Empty hint="Loading the WebGL renderer.">Preparing 3D view</Empty>}>
                  <Network3D
                    topology={view.topology}
                    envelope={envelope}
                    campaign={campaign}
                    flows={view.flows}
                    selected={selected}
                    onSelect={setSelected}
                    situation={prediction ? { level: prediction.alert_level, risk: prediction.risk, technique: prediction.mitre_technique } : null}
                  />
                </Suspense>
              </Boundary>
            )}

          </div>
        </Panel>

        {/* ── Inspector ───────────────────────────────────────────────── */}
        {railOpen && (
        <Panel flush clip className="st-rail">
          <PanelHead title="Verdict" aside={prediction ? <Chip level={level}>{String(level)}</Chip> : undefined} />
          <div className="st-verdict">
            <div className="st-risk" style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>
              {prediction ? <Num value={risk} digits={3} /> : "—"}
            </div>
            <div className="st-verdict-side">
              <Micro>Technique</Micro>
              <Data size="s" color="var(--paper-000)">
                {prediction?.mitre_technique ?? "—"}
              </Data>
              <Micro style={{ marginTop: 6 }}>Peak · lead branch</Micro>
              <Data size="s">
                {peak.toFixed(3)}
                {lead ? ` · ${lead.id} ${(lead.probability * 100).toFixed(0)}% ${lead.technique !== "—" ? lead.technique : "back-off"}` : ""}
              </Data>
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <StateInspector vector={view.stateVector} prediction={prediction} flows={view.flows} latestWindow={latestWindow} />
          </div>
        </Panel>
        )}
      </div>

      {/* ── Time travel ───────────────────────────────────────────────── */}
      <Panel flush className="st-dock">
        <ScrubDock
          frames={frames}
          viewIdx={viewIdx}
          cursor={cursor}
          playing={playing}
          forecast={frames[liveIdx]?.forecast ?? forecast}
          threshold={threshold}
          onCursor={onCursor}
          onPlaying={onPlaying}
        />
      </Panel>
      </div>
    </div>
  );
}
