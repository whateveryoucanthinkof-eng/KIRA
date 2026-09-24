import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip } from "recharts";
import type { SystemStatus, ForecastPoint, PredictionResult, NetworkEvent, Topology } from "../api/types";
import type { LivePoint, PredictionEnvelope } from "../types/live";
import {
  BarRow,
  Chip,
  Data,
  Empty,
  Field,
  Meter,
  Micro,
  Panel,
  PanelBody,
  PanelHead,
  Readout,
  sevColor,
  sevFromRisk,
} from "../design/primitives";
import { FORECAST, Grid, Legend, OBSERVED, Plot, ThresholdLine, Tip, TimeAxis, ValueAxis } from "../design/charts";
import { Num } from "../design/motion";
import KillChain from "../components/KillChain";

interface OverviewProps {
  status: SystemStatus | null;
  history: LivePoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  topology: Topology | null;
  events: NetworkEvent[];
  logLines: string[];
}

/* ── formatting ────────────────────────────────────────────────────────── */

const n = (v: unknown, d = 0): number => (Number.isFinite(Number(v)) ? Number(v) : d);

function horizon(seconds: number, prefix = "+"): string {
  if (!Number.isFinite(seconds)) return "—";
  if (Math.abs(seconds) < 60) return `${prefix}${seconds % 1 === 0 ? seconds.toFixed(0) : seconds.toFixed(1)}s`;
  const m = seconds / 60;
  return `${prefix}${m % 1 === 0 ? m.toFixed(0) : m.toFixed(1)}m`;
}

function hhmmss(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return ts;
  }
}

function compact(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}

/* ════════════════════════════════════════════════════════════════════════
   VERDICT — the hero. One serif numeral, and the honest story behind it.
   ════════════════════════════════════════════════════════════════════════ */

function Verdict({ prediction, envelope }: { prediction: PredictionResult | null; envelope: PredictionEnvelope | null }) {
  if (!prediction) {
    return (
      <Panel flush>
        <PanelHead title="Verdict" />
        <Empty hint="No inference window has been scored yet. Start the sensor and inference from Controls.">
          Awaiting first prediction
        </Empty>
      </Panel>
    );
  }

  const risk = n(prediction.risk);
  const threshold = n(prediction.threshold, 0.65);
  const level = prediction.alert_level ?? sevFromRisk(risk, threshold);
  const color = sevColor(level);

  const ml = prediction.ml_risk;
  const mlTechnique = prediction.ml_technique ?? prediction.predicted_stage;
  const rule = prediction.rule_risk;
  const ruleTechnique = prediction.rule_technique;
  const bypass = prediction.mitigation_status === "traffic_persists" ? n(prediction.mitigation_bypass_flows) : 0;

  const lead = envelope?.early_warning?.lead_time_seconds;

  return (
    <Panel flush>
      <PanelHead
        title="Verdict"
        note={envelope?.timestamp ? hhmmss(envelope.timestamp) : undefined}
        aside={<Chip level={level} strong>{String(level)}</Chip>}
      />

      <PanelBody>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--s-6)", flexWrap: "wrap" }}>
          {/* The one hero numeral on the page */}
          <div style={{ minWidth: 168 }}>
            <Micro style={{ marginBottom: "var(--s-2)" }}>Observed Risk</Micro>
            <Num
              value={risk}
              digits={3}
              className="t-display-xl"
              style={{ color, display: "block", transition: "color var(--dur-value) var(--ease)" }}
            />
            <div style={{ marginTop: "var(--s-3)", width: 168 }}>
              <Meter value={risk} level={level} height={6} />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: "var(--s-1)" }}>
                <span className="t-data-s" style={{ color: "var(--paper-600)" }}>0.00</span>
                <span className="t-data-s" style={{ color: "var(--paper-600)" }}>θ {threshold.toFixed(2)}</span>
                <span className="t-data-s" style={{ color: "var(--paper-600)" }}>1.00</span>
              </div>
            </div>
          </div>

          {/* Technique attribution */}
          <div style={{ flex: 1, minWidth: 220 }}>
            <Micro style={{ marginBottom: "var(--s-2)" }}>Predicted Technique</Micro>
            <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-1)" }}>
              {prediction.mitre_technique ?? prediction.predicted_stage ?? "Unclassified"}
            </div>
            <Data size="s" color="var(--paper-600)">
              {[prediction.mitre_tactic_id, prediction.mitre_tactic].filter(Boolean).join("  ") || "no tactic mapping"}
            </Data>

            {prediction.mitre_description && (
              <div className="t-body" style={{ color: "var(--paper-400)", marginTop: "var(--s-3)", maxWidth: 380 }}>
                {prediction.mitre_description}
              </div>
            )}

            <div style={{ marginTop: "var(--s-4)", display: "grid", gap: 0 }}>
              <Field
                label="Max future risk"
                value={n(prediction.max_future_risk).toFixed(3)}
                color={sevColor(sevFromRisk(n(prediction.max_future_risk), threshold))}
              />
              <Field
                label="Technique conf."
                value={prediction.technique_confidence != null ? n(prediction.technique_confidence).toFixed(3) : "—"}
              />
              {lead != null && Number.isFinite(lead) && (
                // Stated fully in the early-warning band below; repeated here
                // so the verdict panel stands alone when read in isolation.
                <Field label="Early warning" value={`${n(lead).toFixed(1)}s lead`} color="var(--sev-nominal)" />
              )}
            </div>
          </div>
        </div>

        {/* ── Provenance ───────────────────────────────────────────────
            The verdict above is the model's output, unmodified. The SOC rule
            layer (off by default) is advisory: its opinion is shown here next
            to the model's and never replaces risk, technique or alert. */}
        <div style={{ marginTop: "var(--s-4)", paddingTop: "var(--s-3)", borderTop: "var(--hard)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)", marginBottom: "var(--s-2)" }}>
            <Micro>Provenance</Micro>
            <Chip level="nominal">Model output</Chip>
          </div>

          <div style={{ display: "flex", gap: "var(--s-6)", flexWrap: "wrap" }}>
            <Readout label="Model risk" value={ml != null ? n(ml).toFixed(3) : risk.toFixed(3)} level={level} sub="Branch A — the verdict" />
            <Readout label="Model technique" value={mlTechnique ?? "—"} sub="Branch A technique head" />
            <Readout
              label="Rule opinion"
              value={rule != null ? n(rule).toFixed(3) : "—"}
              sub={rule != null ? `advisory only · ${ruleTechnique ?? "unlabelled"}` : "off / no external traffic"}
            />
          </div>
        </div>

        {bypass > 0 && (
          // A recorded block is not enforcement. If traffic it should have
          // stopped is still on the wire, say so instead of showing "quiet".
          <div style={{ marginTop: "var(--s-3)", display: "flex", alignItems: "center", gap: "var(--s-2)" }}>
            <Chip level="critical" strong>Mitigation not effective</Chip>
            <Data size="s" color="var(--paper-400)">
              {bypass} flow{bypass === 1 ? "" : "s"} this window still match a recorded block or isolation
            </Data>
          </div>
        )}
      </PanelBody>
    </Panel>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   EARLY WARNING — the project's whole thesis, stated in one band.

   `early_warning.lead_time_seconds` is how far ahead of the observed
   milestone the model raised the alert. It is the number the system exists to
   produce, and it was previously a 12px table row.
   ════════════════════════════════════════════════════════════════════════ */

function EarlyWarning({ envelope, prediction }: { envelope: PredictionEnvelope | null; prediction: PredictionResult | null }) {
  const ew = envelope?.early_warning;
  const lead = ew?.lead_time_seconds;
  if (lead == null || !Number.isFinite(Number(lead))) return null;

  const stage = prediction?.predicted_stage ?? envelope?.attack_phase ?? "next";
  const technique = prediction?.mitre_technique ?? "";

  return (
    // Deliberately unkeyed. `alert_timestamp` changes every window, so keying
    // on it would remount the band each tick and replay the reveal as a
    // flicker. Mounting when the alert first fires is the only entrance.
    <div className="sheet">
      <Panel flush spine="nominal" className="is-revealing">
        <PanelBody style={{ display: "flex", alignItems: "center", gap: "var(--s-8)", flexWrap: "wrap" }}>
          <div>
            <Micro style={{ marginBottom: "var(--s-2)" }}>Early warning</Micro>
            <div style={{ display: "flex", alignItems: "baseline", gap: "var(--s-2)" }}>
              <Num value={Number(lead)} digits={1} className="t-display-xl" style={{ color: "var(--sev-nominal)" }} />
              <span className="t-data-l" style={{ color: "var(--sev-nominal)" }}>
                s
              </span>
            </div>
          </div>

          <div style={{ flex: 1, minWidth: 240 }}>
            <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-1)" }}>
              ahead of the {String(stage).toLowerCase()} milestone
            </div>
            <Data size="s" color="var(--paper-600)">
              {ew?.target_milestone_desc ?? technique}
            </Data>
          </div>

          <div style={{ minWidth: 200 }}>
            <Field label="Alerted" value={ew?.alert_timestamp ? hhmmss(ew.alert_timestamp) : "—"} />
            <Field
              label="Milestone"
              value={ew?.actual_milestone_timestamp ? hhmmss(ew.actual_milestone_timestamp) : "not yet observed"}
            />
          </div>
        </PanelBody>
      </Panel>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   PIPELINE — which model produced which part of the verdict.
   Straight from PredictionData.stage_provenance.
   ════════════════════════════════════════════════════════════════════════ */

function Pipeline({ prediction, envelope }: { prediction: PredictionResult | null; envelope: PredictionEnvelope | null }) {
  const prov = prediction?.stage_provenance ?? null;
  const lat = envelope?.latency;
  const total = n(lat?.total_ms);

  const stages = prov
    ? Object.entries(prov)
    : [
        ["TGNE", "encoder"],
        ["BRANCH_A", "nowcast"],
        ["BRANCH_B", "rollout"],
        ["DEEPOP", "sequence"],
      ];

  const segs = [
    { label: "telemetry", ms: n(lat?.telemetry_ms), color: "var(--paper-600)" },
    { label: "inference", ms: n(lat?.inference_ms), color: OBSERVED },
  ];
  const segTotal = segs.reduce((a, s) => a + s.ms, 0) || 1;

  return (
    <Panel flush style={{ display: "flex", flexDirection: "column" }}>
      <PanelHead title="Inference pipeline" note={prov ? "live provenance" : "idle"} />
      <PanelBody style={{ flex: 1, display: "flex", flexDirection: "column", gap: "var(--s-3)" }}>
        {stages.map(([stage, detail], i) => (
          <div
            key={stage}
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: "var(--s-2)",
              paddingBottom: "var(--s-2)",
              borderBottom: i < stages.length - 1 ? "var(--hair)" : undefined,
            }}
          >
            <span className="t-data-s" style={{ color: "var(--paper-600)", width: 16, flexShrink: 0 }}>
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="t-label" style={{ color: prov ? "var(--paper-000)" : "var(--paper-600)", width: 74, flexShrink: 0 }}>
              {stage}
            </span>
            <span
              className="t-data-s"
              style={{ color: "var(--paper-600)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
              title={String(detail)}
            >
              {String(detail)}
            </span>
          </div>
        ))}

        <div style={{ marginTop: "auto", paddingTop: "var(--s-3)", borderTop: "var(--hard)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "var(--s-2)" }}>
            <Micro>Latency budget</Micro>
            <Data size="s" color={total > 0 && total < 250 ? "var(--sev-nominal)" : "var(--paper-400)"}>
              {total > 0 ? `${total.toFixed(1)} ms` : "—"}
            </Data>
          </div>

          <div style={{ display: "flex", height: 6, border: "var(--hair)", background: "var(--ink-200)" }}>
            {segs.map((s) => (
              <div key={s.label} style={{ width: `${(s.ms / segTotal) * 100}%`, background: s.color }} title={`${s.label} ${s.ms.toFixed(1)}ms`} />
            ))}
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "var(--s-1)" }}>
            {segs.map((s) => (
              <span key={s.label} className="t-data-s" style={{ color: "var(--paper-600)" }}>
                {s.label} {s.ms.toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      </PanelBody>
    </Panel>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   HOST GRAPH — zone bands, square nodes. Ruled diagram, not a force layout.
   ════════════════════════════════════════════════════════════════════════ */

function HostGraph({ topology }: { topology: Topology }) {
  const nodes = topology.nodes ?? [];
  if (!nodes.length) return <Empty hint="Nodes appear as the sensor observes traffic.">No hosts discovered</Empty>;

  return (
    <svg width="100%" height="100%" viewBox="0 0 800 460" preserveAspectRatio="xMidYMid meet" style={{ display: "block" }}>
      {/* Zone rules */}
      {[
        { y: 60, label: "EXTERNAL" },
        { y: 190, label: "PERIMETER" },
        { y: 320, label: "ENTERPRISE" },
      ].map((band) => (
        <g key={band.label}>
          <line x1={0} y1={band.y - 34} x2={800} y2={band.y - 34} stroke="var(--rule-hair)" strokeWidth={1} shapeRendering="crispEdges" />
          <text x={8} y={band.y - 42} fill="var(--paper-600)" fontSize={9} fontFamily="var(--face-ui)" letterSpacing="2">
            {band.label}
          </text>
        </g>
      ))}

      {topology.edges?.map((e) => {
        const s = nodes.find((x) => x.id === e.source);
        const t = nodes.find((x) => x.id === e.target);
        if (!s || !t) return null;
        const crit = e.status === "saturated";
        return (
          <line
            key={e.id}
            x1={s.x}
            y1={s.y}
            x2={t.x}
            y2={t.y}
            stroke={crit ? "var(--sev-critical)" : e.status === "suspicious" ? "var(--sev-warning)" : "var(--rule-hard)"}
            strokeWidth={crit ? 1.6 : 1}
            strokeDasharray={e.status === "suspicious" ? "3 3" : undefined}
            shapeRendering={crit ? undefined : "crispEdges"}
          />
        );
      })}

      {nodes.map((node) => {
        const c = sevColor(node.status);
        const half = node.status === "compromised" ? 7 : 5;
        return (
          <g key={node.id}>
            <rect x={node.x - half} y={node.y - half} width={half * 2} height={half * 2} fill={c} shapeRendering="crispEdges" />
            <text
              x={node.x}
              y={node.y + half + 12}
              textAnchor="middle"
              fill="var(--paper-400)"
              fontSize={9}
              fontFamily="var(--face-data)"
            >
              {node.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ════════════════════════════════════════════════════════════════════════ */

export default function Overview({ status, history, forecast, prediction, envelope, topology, events, logLines }: OverviewProps) {
  if (!status) {
    return <Empty hint="Waiting for /api/status from the control backend.">Connecting</Empty>;
  }

  const threshold = n(prediction?.threshold, 0.65);
  const level = prediction?.alert_level ?? status.threatLevel;
  const last = history[history.length - 1];

  /* ── Risk trajectory: observed (solid, warm) vs forecast peak (dashed, cool) */
  const riskSeries = history.map((p) => ({
    t: p.label,
    observed: p.risk,
    forecast: p.maxFutureRisk,
  }));

  /* ── Measured telemetry, not derived from risk */
  const telemetrySeries = history.map((p) => ({
    t: p.label,
    throughput: Number(p.throughput.toFixed(2)),
    flows: p.flows,
  }));

  /* ── Forward rollout */
  const rollout = forecast.slice(0, 24).map((f) => ({
    t: horizon(f.horizonSeconds),
    risk: f.predicted / 100,
    band: [f.lowerBound / 100, f.upperBound / 100] as [number, number],
    stage: f.predictedStage ?? "—",
  }));

  const kpis: { label: string; value: string; unit?: string; sub?: string; level?: unknown }[] = [
    { label: "Anomaly", value: n(status.anomalyScore).toFixed(1), unit: "%", level, sub: "max(obs, forecast)" },
    { label: "Throughput", value: n(status.throughput).toFixed(1), unit: "Mb/s", sub: "measured" },
    { label: "Active flows", value: compact(n(status.activeConnections)), sub: "in window" },
    { label: "Packets", value: compact(n(last?.packets)), sub: "last window" },
    { label: "Pipeline", value: n(status.latency).toFixed(1), unit: "ms", sub: "telemetry + inference" },
    { label: "Packet loss", value: n(status.packetLoss).toFixed(2), unit: "%", sub: "unanswered flows" },
    // Sensor blind spots: windows scored on a partial capture (kernel drops) or
    // never scored (missed). Zero is the only healthy value, so any count is a warning.
    {
      label: "Sensor gaps",
      value: compact(n(status.incompleteWindows) + n(status.windowsMissed)),
      unit: "win",
      level: n(status.incompleteWindows) + n(status.windowsMissed) > 0 ? "warning" : undefined,
      sub: `${compact(n(status.sensorKernelDrops))} frames dropped · ${compact(n(status.windowsMissed))} missed`,
    },
    { label: "Windows", value: String(history.length), sub: `of ${90} retained` },
  ];

  const groups = prediction?.explainability?.groups ?? [];
  const feats = prediction?.explainability?.top_features ?? [];

  return (
    <div className="ov">
      {/* ── Verdict + pipeline ───────────────────────────────────────── */}
      <div className="sheet ov-verdict">
        <Verdict prediction={prediction} envelope={envelope} />
        <Pipeline prediction={prediction} envelope={envelope} />
      </div>

      {/* ── Kill chain: how far it got, and where the model says it goes ── */}
      <div className="sheet">
        <Panel flush>
          <PanelHead
            title="Kill chain"
            note={forecast.length ? `observed → +${forecast[forecast.length - 1].horizonSeconds.toFixed(0)}s forecast` : undefined}
            aside={
              prediction?.predicted_stage ? (
                <Chip level={level}>{prediction.predicted_stage}</Chip>
              ) : undefined
            }
          />
          <PanelBody>
            <KillChain
              stage={prediction?.predicted_stage}
              technique={prediction?.mitre_technique}
              forecast={forecast}
              threshold={threshold}
            />
          </PanelBody>
        </Panel>
      </div>

      {/* ── Early warning ────────────────────────────────────────────── */}
      <EarlyWarning envelope={envelope} prediction={prediction} />

      {/* ── KPI strip ────────────────────────────────────────────────── */}
      <div className="sheet ov-kpi">
        {kpis.map((k) => (
          <Panel key={k.label} flush>
            <PanelBody style={{ padding: "var(--s-3)" }}>
              <Readout label={k.label} value={k.value} unit={k.unit} sub={k.sub} level={k.level} />
            </PanelBody>
          </Panel>
        ))}
      </div>

      {/* ── Plots ────────────────────────────────────────────────────── */}
      <div className="sheet ov-plots">
        <Panel flush style={{ display: "flex", flexDirection: "column" }}>
          <PanelHead
            title="Risk trajectory"
            note={`${history.length} windows`}
            aside={
              <Legend
                items={[
                  { color: OBSERVED, label: "observed" },
                  { color: FORECAST, label: "forecast peak", dashed: true },
                ]}
              />
            }
          />
          {riskSeries.length ? (
            <Plot height={212}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={riskSeries} margin={{ top: 14, right: 8, left: 0, bottom: 0 }}>
                  <Grid />
                  <TimeAxis />
                  <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                  <Tooltip content={<Tip fmt={(v) => Number(v).toFixed(3)} />} cursor={{ stroke: "var(--rule-hard)" }} />
                  <ThresholdLine y={threshold} />
                  <Area type="monotone" dataKey="observed" name="observed" stroke="none" fill="var(--fill-observed)" isAnimationActive={false} />
                  <Line type="monotone" dataKey="observed" name="observed" stroke={OBSERVED} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="forecast peak"
                    stroke={FORECAST}
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Plot>
          ) : (
            <Empty hint="Risk history accumulates from the prediction stream.">No inference history</Empty>
          )}
        </Panel>

        <Panel flush style={{ display: "flex", flexDirection: "column" }}>
          <PanelHead
            title="Forward rollout"
            note={rollout.length ? `K=${rollout.length}` : undefined}
            aside={<Legend items={[{ color: FORECAST, label: "risk" }, { color: FORECAST, label: "band", fill: true }]} />}
          />
          {rollout.length ? (
            <Plot height={212}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={rollout} margin={{ top: 14, right: 8, left: 0, bottom: 0 }}>
                  <Grid />
                  <TimeAxis />
                  <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                  <Tooltip content={<Tip fmt={(v) => (Array.isArray(v) ? "" : Number(v).toFixed(3))} />} cursor={{ stroke: "var(--rule-hard)" }} />
                  <ThresholdLine y={threshold} />
                  <Area dataKey="band" name="band" stroke="none" fill="var(--fill-forecast)" isAnimationActive={false} />
                  {/* The rollout is a discrete 8-point shape that changes
                      meaningfully each window, so it earns a morph. The
                      rolling history charts stay un-animated — re-drawing a
                      90-point path every tick reads as lag, not motion. */}
                  <Line
                    type="monotone"
                    dataKey="risk"
                    name="risk"
                    stroke={FORECAST}
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={{ r: 1.5, fill: FORECAST, strokeWidth: 0 }}
                    isAnimationActive
                    animationDuration={200}
                    animationEasing="ease-out"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Plot>
          ) : (
            <Empty hint="The world model emits a K-step rollout with each scored window.">No forecast</Empty>
          )}
        </Panel>
      </div>

      {/* ── Telemetry + attribution + stream ─────────────────────────── */}
      <div className="sheet ov-lower">
        <Panel flush clip style={{ display: "flex", flexDirection: "column", minHeight: 260 }}>
          <PanelHead title="Host graph" note={topology ? `${topology.nodes.length} hosts · ${topology.edges.length} edges` : undefined} />
          <div style={{ flex: 1, minHeight: 0, padding: "var(--s-2)" }}>
            {topology ? <HostGraph topology={topology} /> : <Empty>No topology</Empty>}
          </div>
        </Panel>

        <Panel flush style={{ display: "flex", flexDirection: "column", minHeight: 260 }}>
          <PanelHead
            title="Feature attribution"
            note={prediction?.explainability?.method ?? undefined}
            aside={groups.length ? <Data size="s" color="var(--paper-600)">{groups.length} groups</Data> : undefined}
          />
          {feats.length ? (
            <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)", overflowY: "auto" }}>
              {feats.slice(0, 8).map((f) => (
                <BarRow key={f.feature} label={f.feature} value={n(f.score)} note={f.group} />
              ))}

              {groups.length > 0 && (
                <div style={{ marginTop: "var(--s-2)", paddingTop: "var(--s-3)", borderTop: "var(--hard)" }}>
                  <Micro style={{ marginBottom: "var(--s-2)" }}>By group</Micro>
                  <div style={{ display: "flex", height: 8, border: "var(--hair)" }}>
                    {groups.map((g, i) => (
                      <div
                        key={g.name}
                        title={`${g.name} ${n(g.percentage).toFixed(1)}%`}
                        style={{
                          width: `${n(g.percentage)}%`,
                          background: OBSERVED,
                          opacity: 1 - i * 0.13,
                          borderRight: i < groups.length - 1 ? "1px solid var(--ink-050)" : undefined,
                        }}
                      />
                    ))}
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s-2)", marginTop: "var(--s-2)" }}>
                    {groups.map((g) => (
                      <span key={g.name} className="t-data-s" style={{ color: "var(--paper-600)" }}>
                        {g.name} {n(g.percentage).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </PanelBody>
          ) : (
            <Empty hint="Input × Gradient attributions arrive with each scored window.">No attribution</Empty>
          )}
        </Panel>

        <Panel flush clip style={{ display: "flex", flexDirection: "column", minHeight: 260 }}>
          <PanelHead title="Event stream" note={events.length ? `${events.length}` : undefined} />
          {events.length ? (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
              {events.slice(0, 40).map((e) => (
                <div
                  key={e.id}
                  className="is-stamping"
                  style={{
                    padding: "var(--s-2) var(--s-3)",
                    borderBottom: "var(--hair)",
                    borderLeft: `3px solid ${sevColor(e.severity)}`,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: "var(--s-2)" }}>
                    <span className="t-micro" style={{ color: sevColor(e.severity) }}>
                      {e.category}
                    </span>
                    <Data size="s" color="var(--paper-600)">
                      {hhmmss(e.timestamp)}
                    </Data>
                  </div>
                  <div
                    className="t-data-s"
                    style={{ color: "var(--paper-400)", marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    title={e.message}
                  >
                    {e.message}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Empty hint="Alerts, commands and attack stages land here as they occur.">Stream quiet</Empty>
          )}
        </Panel>
      </div>

      {/* ── Telemetry + console ──────────────────────────────────────── */}
      <div className="sheet ov-plots">
        <Panel flush style={{ display: "flex", flexDirection: "column" }}>
          <PanelHead
            title="Measured telemetry"
            aside={<Legend items={[{ color: OBSERVED, label: "Mb/s" }, { color: "var(--paper-600)", label: "flows" }]} />}
          />
          {telemetrySeries.length ? (
            <Plot height={150}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={telemetrySeries} margin={{ top: 10, right: 8, left: 0, bottom: 0 }}>
                  <Grid />
                  <TimeAxis />
                  <ValueAxis width={40} />
                  <Tooltip content={<Tip />} cursor={{ stroke: "var(--rule-hard)" }} />
                  <Area
                    type="monotone"
                    dataKey="throughput"
                    name="Mb/s"
                    stroke={OBSERVED}
                    strokeWidth={1.5}
                    fill="var(--fill-observed)"
                    isAnimationActive={false}
                  />
                  <Line type="monotone" dataKey="flows" name="flows" stroke="var(--paper-600)" strokeWidth={1} dot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </Plot>
          ) : (
            <Empty hint="Throughput and flow counts come from the sensor's 2s windows.">No telemetry</Empty>
          )}
        </Panel>

        <Panel flush clip style={{ display: "flex", flexDirection: "column" }}>
          <PanelHead title="Console" note={logLines.length ? `${logLines.length} lines` : undefined} />
          <div className="log">
            {logLines.length ? (
              logLines.slice(-60).map((line, i) => (
                <div key={i} className="log-line">
                  {line}
                </div>
              ))
            ) : (
              <span className="log-line">— idle —</span>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
