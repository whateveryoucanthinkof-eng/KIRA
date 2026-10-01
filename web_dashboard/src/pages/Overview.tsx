import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip } from "recharts";
import type { SystemStatus, ForecastPoint, PredictionResult, Topology } from "../api/types";
import type { LivePoint, PredictionEnvelope } from "../types/live";
import type { Campaign } from "../types/campaign";
import type { ForecastBranch } from "../types/forecast";
import { applyEdit, rankIncidents, type Incident, type IncidentEdit } from "../types/incident";
import { Btn, Chip, Empty, Meter, Micro, PanelHead, Square, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Grid, Legend, NowLine, OBSERVED, RISK_BAND_COLORS, splitRisk, ThresholdLine, Tip, TimeAxis, ValueAxis } from "../design/charts";
import { Num } from "../design/motion";
import Storyline from "../components/Storyline";
import { ahead } from "../design/time";

/**
 * Mission control.
 *
 * The first screen answers three questions in order: what is happening, what
 * happens next, and what needs doing. The command band carries all three; the
 * storyline shows how far the intrusion has come; the risk chart runs from
 * the recorded windows straight into the forecast; the rail is the work queue.
 *
 * Panels that lived here before and now have better homes: the console
 * (Events, Controls), feature attribution (Forecast Stage inspector,
 * Investigation), the host graph (Network), the kill chain (the storyline).
 */

interface OverviewProps {
  status: SystemStatus | null;
  history: LivePoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  topology: Topology | null;
  campaign: Campaign | null;
  branches: ForecastBranch[] | null;
  incidents: Incident[];
  edits: Record<string, IncidentEdit>;
  onOpenIncident: (id: string) => void;
}

const n = (v: unknown, d = 0): number => (Number.isFinite(Number(v)) ? Number(v) : d);

function compact(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}

function ago(epochSeconds: number): string {
  const s = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
}

/* ── Command band ──────────────────────────────────────────────────────── */

function Verdict({
  prediction,
  envelope,
  topology,
}: {
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  topology: Topology | null;
}) {
  const threshold = n(prediction?.threshold, 0.65);
  const risk = n(prediction?.risk);
  const level = String(prediction?.alert_level ?? sevFromRisk(risk, threshold)).toLowerCase();
  const technique = prediction?.mitre_technique ?? null;
  const benign = !prediction || !technique || technique === "Benign" || prediction.predicted_stage === "Benign";
  const target = envelope?.target_ip ?? null;
  const targetName = target ? topology?.nodes.find((x) => x.ip === target)?.label : undefined;
  const colour = benign ? "var(--sev-nominal)" : sevColor(level);
  // A classified technique under the threshold is early activity, not yet an attack.
  const kicker = benign ? "No active attack" : risk < threshold ? "Under watch" : "Active attack";

  return (
    <div className="ov2-cell ov2-verdict" style={{ borderLeftColor: colour }}>
      <div className="ov2-kicker">
        <Square status={benign ? "nominal" : level} live={!benign} />
        <span>{kicker}</span>
        <span className="ov2-kicker-end">
          <Chip level={benign ? "nominal" : level}>{benign ? "nominal" : level}</Chip>
        </span>
      </div>
      <div className="ov2-verdict-main">
        <div className="ov2-risk" style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>
          {prediction ? <Num value={risk} digits={3} /> : "—"}
        </div>
        <div className="ov2-verdict-name">
          <b title={technique ?? undefined} style={{ color: benign ? "var(--paper-400)" : "var(--paper-000)" }}>
            {benign ? "Baseline traffic" : technique}
          </b>
          <span title={target ?? undefined}>
            {benign
              ? `below threshold ${threshold.toFixed(2)}`
              : `${prediction?.mitre_tactic ?? prediction?.predicted_stage ?? "—"} → ${targetName ? `${targetName} ` : ""}${target ?? "—"}`}
          </span>
        </div>
      </div>
      <div className="ov2-meter">
        <Meter value={risk} level={sevFromRisk(risk, threshold)} height={5} />
        <span className="ov2-meter-scale">
          <em style={{ left: 0 }}>0</em>
          <em style={{ left: `${threshold * 100}%`, transform: "translateX(-50%)" }}>θ {threshold.toFixed(2)}</em>
          <em style={{ right: 0 }}>1</em>
        </span>
      </div>
    </div>
  );
}

function Ahead({
  prediction,
  envelope,
  forecast,
  branches,
}: {
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  forecast: ForecastPoint[];
  branches: ForecastBranch[] | null;
}) {
  const ew = envelope?.early_warning;
  const lead = ew?.lead_time_seconds != null && Number.isFinite(Number(ew.lead_time_seconds)) ? Number(ew.lead_time_seconds) : null;
  const stage = String(prediction?.predicted_stage ?? envelope?.attack_phase ?? "next").toLowerCase();
  const peak = forecast.length ? forecast.reduce((a, f) => (f.predicted > a.predicted ? f : a), forecast[0]) : null;
  const threshold = n(prediction?.threshold, 0.65);
  const next = branches?.find((b) => b.id === "A") ?? null;

  return (
    <div className="ov2-cell ov2-ahead">
      <div className="ov2-kicker">
        <span>Early warning</span>
      </div>
      {lead != null ? (
        <div className="ov2-lead">
          <span className="ov2-lead-num">
            <Num value={lead} digits={1} />
            <small>s</small>
          </span>
          <span className="ov2-lead-text">
            ahead of the {stage} milestone
            <em>{ew?.target_milestone_desc ?? prediction?.mitre_technique ?? ""}</em>
          </span>
        </div>
      ) : (
        <div className="ov2-lead is-quiet">
          <span className="ov2-lead-text">
            No early warning this window
            <em>
              {peak
                ? `forecast peak ${(peak.predicted / 100).toFixed(3)} at ${ahead(peak.horizonSeconds)} · ${peak.predicted / 100 >= threshold ? "over" : "under"} θ`
                : "the rollout arrives with the next scored window"}
            </em>
          </span>
        </div>
      )}
      {next && (
        <div className="ov2-next" title={(next.hops ?? []).map((h) => h.name).join(" → ") || undefined}>
          <Micro>
            Next · branch {next.id} {(next.probability * 100).toFixed(0)}%
          </Micro>
          <b>{next.label}</b>
          <span>
            {next.technique !== "—" ? `${next.technique} · ` : ""}
            {next.stage} · {ahead(next.horizon_seconds)}
          </span>
        </div>
      )}
    </div>
  );
}

function Act({ top, open, onOpen }: { top: Incident | null; open: number; onOpen: (id: string) => void }) {
  if (!top) {
    return (
      <div className="ov2-cell ov2-act is-clear">
        <div className="ov2-kicker">
          <span>Needs you</span>
        </div>
        <b className="ov2-act-clear">No open incidents</b>
        <span className="ov2-act-meta">The queue is empty. New alerts correlate into incidents here.</span>
      </div>
    );
  }
  return (
    <div className="ov2-cell ov2-act" style={{ borderLeftColor: sevColor(top.severity) }}>
      <div className="ov2-kicker">
        <span>Needs you</span>
        <em>{open} open</em>
      </div>
      <div className="ov2-act-id">
        <b>{top.id}</b>
        <Chip level={top.severity}>{top.status}</Chip>
      </div>
      <span className="ov2-act-title" title={top.title}>
        {top.title}
      </span>
      <span className="ov2-act-meta">
        {top.hostLabel} {top.host} · {top.alertCount} alerts
        {top.leadTimeSeconds != null ? ` · lead ${top.leadTimeSeconds.toFixed(1)} s` : ""}
      </span>
      <Btn primary onClick={() => onOpen(top.id)}>
        Open {top.id} →
      </Btn>
    </div>
  );
}

/* ── Risk: observed into forecast ──────────────────────────────────────── */

function Trajectory({ history, forecast, threshold }: { history: LivePoint[]; forecast: ForecastPoint[]; threshold: number }) {
  if (!history.length) return <Empty hint="Risk history accumulates from the prediction stream.">No inference history</Empty>;
  const last = history[history.length - 1];
  // The observed curve wears the severity ladder: green below θ, amber from
  // the alert threshold, red in the critical band — the same thresholds that
  // colour every risk readout on the page.
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
    <div className="ov2-chart">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 16, right: 12, left: 0, bottom: 2 }}>
          <Grid />
          <TimeAxis count={data.length} />
          <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
          <Tooltip
            content={<Tip fmt={(v) => (Array.isArray(v) ? `${Number(v[0]).toFixed(2)}–${Number(v[1]).toFixed(2)}` : Number(v).toFixed(3))} />}
            cursor={{ stroke: "var(--rule-hard)" }}
          />
          <ThresholdLine y={threshold} />
          <Area type="monotone" dataKey="band" name="band" stroke="none" fill="var(--fill-forecast)" isAnimationActive={false} />
          {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
            <Area key={key} type="monotone" dataKey={key} name="observed" stroke="none" fill={RISK_BAND_COLORS[i]} fillOpacity={0.1} isAnimationActive={false} />
          ))}
          {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
            <Line key={key} type="monotone" dataKey={key} name="observed" stroke={RISK_BAND_COLORS[i]} strokeWidth={1.75} dot={false} isAnimationActive={false} />
          ))}
          <Line
            type="monotone"
            dataKey="forecast"
            name="forecast"
            stroke={FORECAST}
            strokeWidth={1.75}
            strokeDasharray="4 3"
            dot={{ r: 2, fill: FORECAST, stroke: "none" }}
            isAnimationActive={false}
            connectNulls
          />
          <NowLine x={last.label} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Work queue ────────────────────────────────────────────────────────── */

function Queue({ items, onOpen }: { items: Incident[]; onOpen: (id: string) => void }) {
  if (!items.length) {
    return <Empty hint="Alerts on one host correlate into an incident here.">No open incidents</Empty>;
  }
  return (
    <div className="ov2-queue">
      {items.map((i) => (
        <button
          key={i.id}
          className={`ov2-q${i.status === "contained" ? " is-contained" : ""}`}
          style={{ borderLeftColor: sevColor(i.status === "contained" ? "nominal" : i.severity) }}
          onClick={() => onOpen(i.id)}
          title={`Open ${i.id} in the incident war room`}
        >
          <span className="ov2-q-head">
            <b>{i.id}</b>
            <em style={{ color: sevColor(i.status === "contained" ? "nominal" : i.severity) }}>{i.status}</em>
            <i>{ago(i.lastSeen)}</i>
          </span>
          <span className="ov2-q-title">{i.title}</span>
          <span className="ov2-q-meta">
            {i.hostLabel} · {i.alertCount} alerts · peak {i.peakRisk.toFixed(2)}
            {i.leadTimeSeconds != null ? ` · lead ${i.leadTimeSeconds.toFixed(1)} s` : ""}
          </span>
        </button>
      ))}
    </div>
  );
}

/* ── Telemetry ticker ──────────────────────────────────────────────────── */

function Spark({ values, color = "var(--paper-400)" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => `${((i / (values.length - 1)) * 100).toFixed(1)},${(18 - ((v - lo) / span) * 16).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="ov2-spark" aria-hidden>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.25} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/* ════════════════════════════════════════════════════════════════════════ */

export default function Overview({
  status,
  history,
  forecast,
  prediction,
  envelope,
  topology,
  campaign,
  branches,
  incidents,
  edits,
  onOpenIncident,
}: OverviewProps) {
  if (!status) {
    return <Empty hint="Waiting for /api/status from the control backend.">Connecting</Empty>;
  }

  const threshold = n(prediction?.threshold, 0.65);
  const level = prediction?.alert_level ?? status.threatLevel;
  const recent = history.slice(-30);
  const last = history[history.length - 1];

  const queue = incidents
    .map((i) => applyEdit(i, edits[i.id]))
    .filter((i) => i.status !== "closed")
    .sort(rankIncidents);
  const open = queue.filter((i) => i.status === "new" || i.status === "triaging");

  const ticker: { label: string; value: string; unit?: string; spark?: number[]; color?: string; level?: unknown }[] = [
    {
      label: "Anomaly",
      value: n(status.anomalyScore).toFixed(1),
      unit: "%",
      spark: recent.map((p) => Math.max(p.risk, p.maxFutureRisk)),
      color: OBSERVED,
      level,
    },
    { label: "Throughput", value: n(status.throughput).toFixed(1), unit: "Mb/s", spark: recent.map((p) => p.throughput) },
    { label: "Flows", value: compact(n(status.activeConnections)), spark: recent.map((p) => p.flows) },
    { label: "Packets", value: compact(n(last?.packets)), spark: recent.map((p) => p.packets) },
    { label: "Pipeline", value: n(status.latency).toFixed(1), unit: "ms", spark: recent.map((p) => p.latencyMs) },
    { label: "Loss", value: n(status.packetLoss).toFixed(2), unit: "%" },
    { label: "Windows", value: String(history.length) },
  ];

  return (
    <div className="ov2">
      <div className="ov2-sheet sheet">
        {/* ── Command band: now · next · act ──────────────────────────── */}
        <div className="ov2-band">
          <Verdict prediction={prediction} envelope={envelope} topology={topology} />
          <Ahead prediction={prediction} envelope={envelope} forecast={forecast} branches={branches} />
          <Act top={open[0] ?? null} open={open.length} onOpen={onOpenIncident} />
        </div>

        {/* ── How far it has come ─────────────────────────────────────── */}
        <div className="ov2-story">
          <Storyline prediction={prediction} forecast={forecast} campaign={campaign} branches={branches} futureSeconds={null} />
        </div>

        {/* ── Risk and the queue ──────────────────────────────────────── */}
        <div className="ov2-main">
          <div className="ov2-panel">
            <PanelHead
              title="Risk · observed → forecast"
              note={`${history.length} windows · ${ahead(forecast.length ? forecast[forecast.length - 1].horizonSeconds : 0)} rollout`}
              aside={
                <Legend
                  items={[
                    { color: RISK_BAND_COLORS[0], label: "observed" },
                    { color: RISK_BAND_COLORS[1], label: "≥ threshold" },
                    { color: RISK_BAND_COLORS[3], label: "critical" },
                    { color: FORECAST, label: "forecast", dashed: true },
                    { color: FORECAST, label: "band", fill: true },
                  ]}
                />
              }
            />
            <Trajectory history={history} forecast={forecast} threshold={threshold} />
          </div>
          <div className="ov2-panel ov2-rail">
            <PanelHead title="Incidents" note={`${open.length} open · ${queue.length - open.length} contained`} />
            <Queue items={queue} onOpen={onOpenIncident} />
          </div>
        </div>

        {/* ── Telemetry ticker ────────────────────────────────────────── */}
        <div className="ov2-ticker">
          {ticker.map((k) => (
            <div key={k.label} className="ov2-tick">
              <span className="ov2-tick-label">{k.label}</span>
              <span className="ov2-tick-value">
                <b style={k.level !== undefined ? { color: sevColor(k.level) } : undefined}>{k.value}</b>
                {k.unit && <em>{k.unit}</em>}
              </span>
              {k.spark && <Spark values={k.spark} color={k.color} />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
