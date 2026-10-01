import { useMemo, useState } from "react";
import { Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip } from "recharts";
import type { ForecastPoint, PredictionResult, Topology } from "../api/types";
import type { LivePoint, PredictionEnvelope } from "../types/live";
import type { Campaign } from "../types/campaign";
import type { FlowRecord, StateDim } from "../types/evidence";
import { Chip, Data, Empty, Micro, Panel, PanelBody, PanelHead, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Grid, Legend, NowLine, RISK_BAND_COLORS, splitRisk, ThresholdLine, Tip, TimeAxis, ValueAxis } from "../design/charts";
import { Num } from "../design/motion";
import KillChain from "../components/KillChain";
import AttackPath from "../components/AttackPath";
import StateInspector from "../components/StateInspector";
import FlowTable from "../components/FlowTable";
import { clockTime } from "../design/time";

/**
 * Investigation — prediction and campaign on one surface.
 *
 * The other views each answer one question. This one answers the analyst's
 * actual question in order: where is the intrusion now (kill chain), where is
 * it going and how sure is the model (H_hat trajectory with its conformal
 * band), which hosts does it run through (attack path), what drove the
 * verdict and was it overridden (inspector), and what is the evidence
 * (compacted alert nodes, raw flows).
 */

interface InvestigationProps {
  history: LivePoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  campaign: Campaign | null;
  topology: Topology | null;
  flows: FlowRecord[];
  flowsInWindow: number | null;
  stateVector: StateDim[] | null;
}

type Tab = "nodes" | "flows";

const n = (v: unknown, d = 0): number => (Number.isFinite(Number(v)) ? Number(v) : d);

function hhmmss(epochSeconds: number): string {
  try {
    return clockTime(epochSeconds * 1000);
  } catch {
    return "—";
  }
}

export default function Investigation({
  history,
  forecast,
  prediction,
  envelope,
  campaign,
  topology,
  flows,
  flowsInWindow,
  stateVector,
}: InvestigationProps) {
  const [tab, setTab] = useState<Tab>("nodes");

  const threshold = n(prediction?.threshold, 0.65);
  const risk = n(prediction?.risk);
  const level = prediction?.alert_level ?? sevFromRisk(risk, threshold);
  const latestWindow = envelope?.state?.window_id ?? null;
  const attacker = (envelope?.focus_ips ?? []).find((ip) => !ip.startsWith("10.")) ?? null;

  /* ── H_hat trajectory: 90 observed windows, NOW, then the K-step rollout ── */
  const series = useMemo(
    () =>
      // Observed windows wear the severity ladder; the rollout stays cool.
      splitRisk(
        [
          ...history.slice(-90).map((p) => ({
            t: p.label,
            observed: p.risk,
            forecast: undefined as number | undefined,
            band: undefined as [number, number] | undefined,
          })),
          ...(prediction ? [{ t: "NOW", observed: risk, forecast: risk, band: [risk, risk] as [number, number] }] : []),
          ...forecast.map((f) => ({
            t: `+${f.horizonSeconds}s`,
            observed: undefined as number | undefined,
            forecast: f.predicted / 100,
            band: [f.lowerBound / 100, f.upperBound / 100] as [number, number],
          })),
        ],
        threshold,
        (r) => r.observed,
      ),
    [history, forecast, prediction, risk, threshold],
  );

  const peak = forecast.length ? forecast.reduce((a, b) => (b.predicted > a.predicted ? b : a)) : null;
  const last = forecast.length ? forecast[forecast.length - 1] : null;
  const bandAtHorizon = last ? (last.upperBound - last.lowerBound) / 200 : null;

  // Hazard onset: the first rollout step to cross θ while observed risk is
  // still below it. That gap is the early warning.
  const onset = risk < threshold ? forecast.find((f) => f.predicted / 100 >= threshold) ?? null : null;

  const nodes = campaign ? [...campaign.nodes].sort((a, b) => a.node_id - b.node_id) : [];
  const roots = new Set(campaign?.root_cause_node_ids ?? []);
  const names = useMemo(() => {
    const m: Record<string, string> = {};
    for (const t of topology?.nodes ?? []) if (t.ip && t.label && t.label !== t.ip) m[t.ip] = t.label;
    return m;
  }, [topology]);

  return (
    <div className="iv">
      <div className="iv-grid">
        {/* ── Main column ─────────────────────────────────────────────── */}
        <div className="iv-main sheet">
          <Panel flush>
            <PanelHead
              title="Kill chain"
              note={last ? `observed → +${last.horizonSeconds}s forecast` : undefined}
              aside={prediction?.predicted_stage ? <Chip level={level}>{prediction.predicted_stage}</Chip> : undefined}
            />
            <PanelBody>
              <KillChain stage={prediction?.predicted_stage} technique={prediction?.mitre_technique} forecast={forecast} threshold={threshold} />
            </PanelBody>
          </Panel>

          <Panel flush>
            <PanelHead
              title="H_hat trajectory"
              note={`→ infiltration risk · ${history.length} observed · K=${forecast.length}`}
              aside={
                <Legend
                  items={[
                    { color: RISK_BAND_COLORS[0], label: "observed" },
                      { color: RISK_BAND_COLORS[1], label: "≥ threshold" },
                      { color: RISK_BAND_COLORS[3], label: "critical" },
                    { color: FORECAST, label: "forecast", dashed: true },
                    { color: FORECAST, label: "95% conformal", fill: true },
                  ]}
                />
              }
            />

            <div className="iv-stats">
              <div>
                <Micro>Observed</Micro>
                <Data size="l" color={sevColor(sevFromRisk(risk, threshold))}>
                  <Num value={risk} digits={3} />
                </Data>
              </div>
              <div>
                <Micro>Forecast peak</Micro>
                <Data size="l" color={peak ? sevColor(sevFromRisk(peak.predicted / 100, threshold)) : "var(--paper-600)"}>
                  {peak ? <Num value={peak.predicted / 100} digits={3} /> : "—"}
                </Data>
                {peak && (
                  <Data size="s" color="var(--paper-600)" style={{ display: "block" }}>
                    at +{peak.horizonSeconds}s
                  </Data>
                )}
              </div>
              <div>
                <Micro>Band at horizon</Micro>
                <Data size="l" color={FORECAST}>
                  {bandAtHorizon != null ? <Num value={bandAtHorizon} digits={3} prefix="±" /> : "—"}
                </Data>
                {last && (
                  <Data size="s" color="var(--paper-600)" style={{ display: "block" }}>
                    at +{last.horizonSeconds}s
                  </Data>
                )}
              </div>
              <div>
                <Micro>Hazard onset</Micro>
                <Data size="l" color={onset ? "var(--sev-warning)" : "var(--paper-600)"}>
                  {onset ? `+${onset.horizonSeconds}s` : risk >= threshold ? "active" : "none"}
                </Data>
                <Data size="s" color="var(--paper-600)" style={{ display: "block" }}>
                  {onset ? "forecast crosses θ" : risk >= threshold ? "already above θ" : "within horizon"}
                </Data>
              </div>
            </div>

            {series.length > 1 ? (
              <div style={{ height: 268, padding: "0 var(--s-3) var(--s-2) 0" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={series} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
                    <Grid />
                    <TimeAxis count={series.length} />
                    <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                    <Tooltip
                      content={<Tip fmt={(v) => (Array.isArray(v) ? `${Number(v[0]).toFixed(3)} – ${Number(v[1]).toFixed(3)}` : Number(v).toFixed(3))} />}
                      cursor={{ stroke: "var(--rule-hard)" }}
                    />
                    <ThresholdLine y={threshold} />
                    {prediction && <NowLine x="NOW" />}
                    {onset && (
                      <ReferenceLine
                        x={`+${onset.horizonSeconds}s`}
                        stroke="var(--sev-warning)"
                        strokeDasharray="2 3"
                        label={{
                          value: "ONSET",
                          position: "insideTopRight",
                          fill: "var(--sev-warning)",
                          fontSize: 9,
                          fontFamily: "var(--face-data)",
                          letterSpacing: "0.1em",
                          offset: 6,
                        }}
                      />
                    )}

                    {/* Conformal band — widens with horizon, as a finite-sample radius does. */}
                    <Area dataKey="band" name="95% conformal" stroke="none" fill="var(--fill-forecast)" isAnimationActive={false} connectNulls />
                    {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
                      <Area key={key} type="monotoneX" dataKey={key} name="observed" stroke="none" fill={RISK_BAND_COLORS[i]} fillOpacity={0.1} isAnimationActive={false} />
                    ))}
                    {(["observed-nominal", "observed-warning", "observed-elevated", "observed-critical"] as const).map((key, i) => (
                      <Line key={key} type="monotoneX" dataKey={key} name="observed" stroke={RISK_BAND_COLORS[i]} strokeWidth={1.6} dot={false} isAnimationActive={false} />
                    ))}
                    <Line
                      type="monotoneX"
                      dataKey="forecast"
                      name="forecast"
                      stroke={FORECAST}
                      strokeWidth={1.6}
                      strokeDasharray="4 3"
                      dot={{ r: 1.8, fill: FORECAST, strokeWidth: 0 }}
                      connectNulls
                      isAnimationActive
                      animationDuration={260}
                      animationEasing="ease-out"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <Empty hint="History accumulates from the prediction stream.">No trajectory yet</Empty>
            )}

            <div className="flow-foot">
              <Micro>Band</Micro>
              <Data size="s" color="var(--paper-600)">
                Branch B conformal radii · α = 0.05 · finite-sample
              </Data>
              <Data size="s" color="var(--paper-600)" style={{ marginLeft: "auto" }}>
                window Δt 2.0s · horizon {last ? `${last.horizonSeconds}s` : "—"}
              </Data>
            </div>
          </Panel>

          <Panel flush>
            <PanelHead
              title="Attack path"
              note={campaign ? `campaign #${campaign.campaign_id} · ${campaign.involved_hosts.length} hosts` : undefined}
              aside={
                campaign?.has_forecast_components ? (
                  <Data size="s" color={FORECAST}>
                    includes forecast hops
                  </Data>
                ) : undefined
              }
            />
            <PanelBody>
              <AttackPath
                campaign={campaign}
                topology={topology}
                flows={flows}
                attacker={attacker}
                latestWindow={latestWindow}
                threshold={threshold}
              />
            </PanelBody>
          </Panel>
        </div>

        {/* ── Inspector rail ──────────────────────────────────────────── */}
        <div className="iv-rail">
          <div className="iv-rail-inner">
            <StateInspector vector={stateVector} prediction={prediction} flows={flows} latestWindow={latestWindow} />
          </div>
        </div>
      </div>

      {/* ── Evidence ──────────────────────────────────────────────────── */}
      <Panel clip collapse={["top"]} className="iv-evidence">
        <div className="tabs">
          {(
            [
              ["nodes", "Compacted alert nodes", nodes.length],
              ["flows", "Flow evidence", flows.filter((f) => f.window === latestWindow).length],
            ] as const
          ).map(([id, label, count]) => (
            <button key={id} className="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
              <span>{label}</span>
              <span className="tab-count">{count}</span>
            </button>
          ))}
        </div>

        <div key={tab} className="is-entering" style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
          {tab === "nodes" ? (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
              {nodes.length ? (
                <table className="tbl">
                  <thead>
                    <tr>
                      <th style={{ width: 56 }}>Node</th>
                      <th style={{ width: 150 }}>Host</th>
                      <th style={{ width: 132 }}>Stage</th>
                      <th style={{ width: 92 }}>Technique</th>
                      <th style={{ width: 112 }}>Provenance</th>
                      <th className="num" style={{ width: 72 }}>Hits</th>
                      <th className="num" style={{ width: 80 }}>Max risk</th>
                      <th style={{ width: 110 }} />
                      <th className="num" style={{ width: 64 }}>Conf.</th>
                      <th style={{ width: 88 }}>First seen</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {nodes.map((nd) => {
                      const fc = nd.provenance === "FORECAST";
                      const s = sevFromRisk(nd.max_risk_score, threshold);
                      return (
                        <tr key={nd.node_id}>
                          <td className="key" style={{ borderLeft: `3px solid ${fc ? FORECAST : sevColor(s)}` }}>
                            {String(nd.node_id).padStart(2, "0")}
                          </td>
                          <td>
                            <span className="key">{names[nd.host_ip] ?? nd.host_ip}</span>
                            <span style={{ color: "var(--paper-600)" }}> {nd.host_ip}</span>
                          </td>
                          <td>{nd.coarse_category}</td>
                          <td className="key">{nd.technique_id}</td>
                          <td>
                            <span className={fc ? "prov is-fc" : "prov"}>{nd.provenance}</span>
                          </td>
                          <td className="num">{fc ? "—" : nd.hit_count.toLocaleString()}</td>
                          <td className="num key" style={{ color: fc ? FORECAST : sevColor(s) }}>
                            {nd.max_risk_score.toFixed(3)}
                          </td>
                          <td>
                            <span className="mini-meter">
                              <span style={{ width: `${nd.max_risk_score * 100}%`, background: fc ? FORECAST : sevColor(s) }} />
                            </span>
                          </td>
                          <td className="num">{nd.mean_confidence.toFixed(2)}</td>
                          <td>{fc ? "projected" : hhmmss(nd.start_time)}</td>
                          <td>{roots.has(nd.node_id) ? <Chip>root cause</Chip> : null}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <Empty hint="Nodes compact out of correlated alert trajectories once an intrusion is underway.">No compacted nodes</Empty>
              )}
            </div>
          ) : (
            <FlowTable flows={flows} flowsInWindow={flowsInWindow ?? undefined} latestWindow={latestWindow} names={names} />
          )}
        </div>
      </Panel>
    </div>
  );
}
