import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip } from "recharts";
import type { ForecastPoint, PredictionResult } from "../api/types";
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
import { FORECAST, Grid, Legend, NowLine, OBSERVED, Plot, ThresholdLine, Tip, TimeAxis, ValueAxis } from "../design/charts";
import { Num } from "../design/motion";
import KillChain from "../components/KillChain";

interface PredictionsProps {
  history: LivePoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
}

const n = (v: unknown, d = 0): number => (Number.isFinite(Number(v)) ? Number(v) : d);

function horizon(seconds: number, prefix = "+"): string {
  if (!Number.isFinite(seconds)) return "—";
  if (Math.abs(seconds) < 60) return `${prefix}${seconds % 1 === 0 ? seconds.toFixed(0) : seconds.toFixed(1)}s`;
  const m = seconds / 60;
  return `${prefix}${m % 1 === 0 ? m.toFixed(0) : m.toFixed(1)}m`;
}

export default function Predictions({ history, forecast, prediction, envelope }: PredictionsProps) {
  const threshold = n(prediction?.threshold, 0.65);
  const risk = n(prediction?.risk);
  const maxFuture = n(prediction?.max_future_risk, risk);
  const level = prediction?.alert_level ?? sevFromRisk(risk, threshold);
  const span = forecast.length ? forecast[forecast.length - 1].horizonSeconds : null;

  /**
   * One continuous axis: measured history to the left of NOW, the model's
   * rollout to the right. Splitting them into two charts hides the only thing
   * that matters — whether the forecast continues the observed curve.
   */
  const joined = [
    ...history.slice(-40).map((p) => ({
      t: p.label,
      observed: p.risk,
      forecast: undefined as number | undefined,
      band: undefined as [number, number] | undefined,
    })),
    ...(prediction
      ? [{ t: "NOW", observed: risk, forecast: risk, band: [risk, risk] as [number, number] }]
      : []),
    ...forecast.map((f) => ({
      t: horizon(f.horizonSeconds),
      observed: undefined as number | undefined,
      forecast: f.predicted / 100,
      band: [f.lowerBound / 100, f.upperBound / 100] as [number, number],
    })),
  ];

  const stages = prediction?.stage_probabilities
    ? Object.entries(prediction.stage_probabilities).sort((a, b) => n(b[1]) - n(a[1]))
    : [];

  const feats = prediction?.explainability?.top_features ?? [];

  const head: { label: string; value: React.ReactNode; unit?: string; sub?: string; level?: unknown }[] = [
    { label: "Observed risk", value: <Num value={risk} digits={3} />, level, sub: "current window" },
    {
      label: "Max future risk",
      value: <Num value={maxFuture} digits={3} />,
      level: sevFromRisk(maxFuture, threshold),
      sub: span != null ? `within ${horizon(span, "")}` : "rollout peak",
    },
    {
      label: "Hazard",
      value: prediction?.hazard_score != null ? n(prediction.hazard_score).toFixed(3) : "—",
      sub: "onset within horizon",
    },
    {
      label: "Threshold",
      value: threshold.toFixed(3),
      sub: prediction?.alert ? "breached" : "not breached",
      level: prediction?.alert ? "critical" : "nominal",
    },
  ];

  return (
    <div className="pr">
      {/* ── Headline numbers ─────────────────────────────────────────── */}
      <div className="sheet pr-head">
        {head.map((k) => (
          <Panel key={k.label} flush>
            <PanelBody style={{ padding: "var(--s-3)" }}>
              <Readout label={k.label} value={k.value} unit={k.unit} sub={k.sub} level={k.level} />
            </PanelBody>
          </Panel>
        ))}
      </div>

      {/* ── Kill chain ───────────────────────────────────────────────── */}
      <div className="sheet">
        <Panel flush>
          <PanelHead
            title="Kill chain"
            note={span != null ? `observed → ${horizon(span)}` : undefined}
            aside={prediction?.predicted_stage ? <Chip level={level}>{prediction.predicted_stage}</Chip> : undefined}
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

      {/* ── Trajectory + rail ────────────────────────────────────────── */}
      <div className="sheet pr-body">
        <Panel flush style={{ display: "flex", flexDirection: "column" }}>
          <PanelHead
            title="Risk trajectory"
            note={span != null ? `history → ${horizon(span)}` : "history"}
            aside={
              <Legend
                items={[
                  { color: OBSERVED, label: "observed" },
                  { color: FORECAST, label: "forecast", dashed: true },
                  { color: FORECAST, label: "band", fill: true },
                ]}
              />
            }
          />

          {joined.length > 1 ? (
            <Plot height={288}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={joined} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
                  <Grid />
                  <TimeAxis />
                  <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                  <Tooltip
                    content={<Tip fmt={(v) => (Array.isArray(v) ? "" : Number(v).toFixed(3))} />}
                    cursor={{ stroke: "var(--rule-hard)" }}
                  />
                  <ThresholdLine y={threshold} />
                  {prediction && <NowLine x="NOW" />}

                  <Area dataKey="band" name="band" stroke="none" fill="var(--fill-forecast)" isAnimationActive={false} connectNulls />
                  <Area
                    type="monotone"
                    dataKey="observed"
                    name="observed"
                    stroke="none"
                    fill="var(--fill-observed)"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="observed"
                    name="observed"
                    stroke={OBSERVED}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="forecast"
                    stroke={FORECAST}
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={{ r: 1.5, fill: FORECAST, strokeWidth: 0 }}
                    isAnimationActive={false}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Plot>
          ) : (
            <Empty hint="History accumulates client-side from the prediction stream; the rollout arrives with each scored window.">
              No trajectory yet
            </Empty>
          )}

          {/* ── Rollout table ──────────────────────────────────────── */}
          <div style={{ borderTop: "var(--hard)", maxHeight: 260, overflowY: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 56 }}>Step</th>
                  <th style={{ width: 80 }}>Horizon</th>
                  <th>Predicted stage</th>
                  <th className="num" style={{ width: 72 }}>Risk</th>
                  <th className="num" style={{ width: 72 }}>Conf.</th>
                  <th style={{ width: 120 }}>Band</th>
                </tr>
              </thead>
              <tbody>
                {forecast.length ? (
                  forecast.map((f, i) => {
                    const r = f.predicted / 100;
                    const s = sevFromRisk(r, threshold);
                    return (
                      <tr key={`${f.horizonSeconds}-${i}`}>
                        <td className="key" style={{ borderLeft: `3px solid ${sevColor(s)}` }}>
                          {String(i + 1).padStart(2, "0")}
                        </td>
                        <td>{horizon(f.horizonSeconds)}</td>
                        <td className="key">{f.predictedStage ?? "—"}</td>
                        <td className="num key" style={{ color: sevColor(s) }}>
                          {r.toFixed(3)}
                        </td>
                        <td className="num">{n(f.confidence).toFixed(2)}</td>
                        <td>
                          <Meter value={r} level={s} height={4} />
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "var(--s-6)", color: "var(--paper-600)" }}>
                      no rollout
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* ── Rail ─────────────────────────────────────────────────── */}
        <div className="pr-rail" style={{ background: "var(--ink-050)" }}>
          <PanelHead title="Classification" aside={prediction ? <Chip level={level}>{String(level)}</Chip> : undefined} />
          <PanelBody>
            {prediction ? (
              <>
                <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-1)" }}>
                  {prediction.mitre_technique ?? prediction.predicted_stage ?? "Unclassified"}
                </div>
                <Data size="s" color="var(--paper-600)">
                  {[prediction.mitre_tactic_id, prediction.mitre_tactic].filter(Boolean).join("  ") || "no tactic mapping"}
                </Data>

                <div style={{ marginTop: "var(--s-4)" }}>
                  <Field label="Alert" value={prediction.alert ? "RAISED" : "clear"} color={prediction.alert ? "var(--sev-critical)" : undefined} />
                  <Field label="Stage" value={prediction.predicted_stage ?? "—"} />
                  <Field
                    label="Technique conf."
                    value={prediction.technique_confidence != null ? n(prediction.technique_confidence).toFixed(3) : "—"}
                  />
                  <Field
                    label="Malicious conf."
                    value={prediction.malicious_confidence != null ? n(prediction.malicious_confidence).toFixed(3) : "—"}
                  />
                  <Field
                    label="Precursor conf."
                    value={prediction.precursor_confidence != null ? n(prediction.precursor_confidence).toFixed(3) : "—"}
                  />
                  <Field label="Risk source" value={prediction.risk_source ?? "model"} />
                  <Field label="Model risk" value={prediction.ml_risk != null ? n(prediction.ml_risk).toFixed(3) : "—"} />
                  <Field label="Model technique" value={prediction.ml_technique ?? prediction.predicted_stage ?? "—"} />
                  <Field
                    label="Rule opinion (advisory)"
                    value={
                      prediction.rule_risk != null
                        ? `${n(prediction.rule_risk).toFixed(3)} ${prediction.rule_technique ?? ""}`.trim()
                        : "—"
                    }
                  />
                </div>
              </>
            ) : (
              <Empty hint="Start inference from Controls.">No classification</Empty>
            )}
          </PanelBody>

          {/* Stage distribution */}
          {stages.length > 0 && (
            <>
              <PanelHead title="Stage distribution" />
              <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)" }}>
                {stages.map(([name, p]) => (
                  <BarRow key={name} label={name} value={n(p)} labelWidth={110} />
                ))}
              </PanelBody>
            </>
          )}

          {/* Attribution */}
          <PanelHead title="Attribution" note={prediction?.explainability?.method ?? undefined} />
          <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)", flex: 1 }}>
            {feats.length ? (
              feats.slice(0, 8).map((f) => <BarRow key={f.feature} label={f.feature} value={n(f.score)} labelWidth={110} />)
            ) : (
              <Empty>No attribution</Empty>
            )}
          </PanelBody>

          {/* Early warning — the project's headline claim */}
          {envelope?.early_warning?.lead_time_seconds != null && (
            <>
              <PanelHead title="Early warning" />
              <PanelBody>
                <Readout
                  label="Lead time"
                  value={n(envelope.early_warning.lead_time_seconds).toFixed(1)}
                  unit="s"
                  level="nominal"
                  sub={envelope.early_warning.target_milestone_desc ?? "ahead of milestone"}
                />
              </PanelBody>
            </>
          )}

          {/* Pipeline state */}
          <PanelHead title="Pipeline" />
          <PanelBody>
            <Field label="Window" value={envelope?.state?.window_id != null ? `#${envelope.state.window_id}` : "—"} />
            <Field label="Sequence" value={envelope?.state?.sequence_ready ? "ready" : "filling"} />
            <Field label="Buffer" value={envelope?.state?.buffer_length != null ? String(envelope.state.buffer_length) : "—"} />
            <Field label="Packets" value={envelope?.state?.packet_count != null ? String(envelope.state.packet_count) : "—"} />
            <Field label="Flows" value={envelope?.state?.active_flows != null ? String(envelope.state.active_flows) : "—"} />
            <Field
              label="Total latency"
              value={envelope?.latency?.total_ms != null ? `${n(envelope.latency.total_ms).toFixed(1)} ms` : "—"}
            />
            <div style={{ marginTop: "var(--s-3)" }}>
              <Micro style={{ marginBottom: "var(--s-2)" }}>Mode</Micro>
              <Chip level={envelope?.mode === "LIVE" ? "nominal" : "neutral"}>{envelope?.mode ?? "STANDBY"}</Chip>
            </div>
          </PanelBody>
        </div>
      </div>
    </div>
  );
}
