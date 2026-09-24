import { useMemo, useState } from "react";
import type { PredictionResult } from "../api/types";
import type { Campaign as CampaignData, CampaignNode } from "../types/campaign";
import {
  Chip,
  Data,
  Empty,
  Field,
  Micro,
  Panel,
  PanelBody,
  PanelHead,
  Readout,
  sevColor,
  sevFromRisk,
} from "../design/primitives";
import { FORECAST, Legend, OBSERVED } from "../design/charts";

interface CampaignProps {
  campaign: CampaignData | null;
  prediction: PredictionResult | null;
}

/* Layout constants for the causal graph canvas. */
const VB_W = 980;
const LANE_LABEL_H = 34;
const ROW_H = 96;
const GUTTER = 132;

function hhmmss(epochSeconds: number): string {
  try {
    return new Date(epochSeconds * 1000).toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return "—";
  }
}

function hostLabel(ip: string): string {
  const known: Record<string, string> = {
    "10.0.3.10": "dmz-web",
    "10.0.2.40": "srv-app",
    "10.0.2.30": "srv-file",
    "10.0.2.20": "srv-id",
    "10.0.2.10": "srv-dns",
    "10.0.2.50": "srv-db",
  };
  return known[ip] ?? ip;
}

export default function Campaign({ campaign, prediction }: CampaignProps) {
  const [sel, setSel] = useState<number | null>(null);

  const layout = useMemo(() => {
    if (!campaign) return null;

    const lanes = campaign.lanes;
    const hosts = [...new Set(campaign.nodes.map((n) => n.host_ip))];
    const laneW = (VB_W - GUTTER) / lanes.length;

    const pos = new Map<number, { x: number; y: number }>();
    for (const n of campaign.nodes) {
      const li = Math.max(0, lanes.indexOf(n.coarse_category));
      const hi = Math.max(0, hosts.indexOf(n.host_ip));
      pos.set(n.node_id, {
        x: GUTTER + laneW * li + laneW / 2,
        y: LANE_LABEL_H + ROW_H * hi + ROW_H / 2,
      });
    }

    return { lanes, hosts, laneW, pos, height: LANE_LABEL_H + ROW_H * hosts.length + 16 };
  }, [campaign]);

  if (!campaign || !layout) {
    return (
      <div className="cm">
        <Panel clip style={{ height: "100%" }}>
          <PanelHead title="Campaign" />
          {/* correlation/ is not wired into control_backend, so the live
              backend never emits a campaign. Saying "once inference is
              running" implied it would appear; it will not. */}
          <Empty hint="Campaign reconstruction is not connected to the live backend yet. Only demo fixtures populate this page.">
            No campaign reconstructed
          </Empty>
        </Panel>
      </div>
    );
  }

  const { lanes, hosts, laneW, pos, height } = layout;
  const byId = new Map(campaign.nodes.map((n) => [n.node_id, n]));
  const roots = new Set(campaign.root_cause_node_ids);
  const selected: CampaignNode | null = sel != null ? (byId.get(sel) ?? null) : null;
  const observed = campaign.nodes.filter((n) => n.provenance === "OBSERVED");
  const forecast = campaign.nodes.filter((n) => n.provenance === "FORECAST");
  const threshold = Number(prediction?.threshold ?? 0.65);

  return (
    <div className="cm">
      {/* ── Causal graph ─────────────────────────────────────────────── */}
      <div className="cm-main sheet">
        {/* flex 1 1 0 — without an explicit grow the graph sizes to content
            and collapses against the fixed-height table below it. */}
        <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <PanelHead
            title="Causal graph"
            note={`campaign #${campaign.campaign_id} · ${campaign.nodes.length} nodes · ${campaign.edges.length} edges`}
            aside={
              <Legend
                items={[
                  { color: OBSERVED, label: "observed" },
                  { color: FORECAST, label: "forecast", dashed: true },
                ]}
              />
            }
          />

          <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "var(--s-3)" }}>
            <svg
              width="100%"
              viewBox={`0 0 ${VB_W} ${height}`}
              preserveAspectRatio="xMidYMin meet"
              style={{ display: "block", minHeight: height }}
              onClick={() => setSel(null)}
            >
              <defs>
                <pattern id="cm-hatch" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">
                  <rect width="4" height="4" fill="var(--ink-050)" />
                  <line x1="0" y1="0" x2="0" y2="4" stroke={FORECAST} strokeWidth="1.6" />
                </pattern>
              </defs>

              {/* Kill-chain lanes */}
              {lanes.map((lane, i) => (
                <g key={lane}>
                  <line
                    x1={GUTTER + laneW * i}
                    y1={0}
                    x2={GUTTER + laneW * i}
                    y2={height}
                    stroke="var(--rule-hair)"
                    strokeWidth={1}
                    shapeRendering="crispEdges"
                  />
                  <text
                    x={GUTTER + laneW * i + 8}
                    y={14}
                    fill="var(--paper-600)"
                    fontSize={9}
                    fontFamily="var(--face-ui)"
                    letterSpacing="2"
                  >
                    {lane.toUpperCase()}
                  </text>
                </g>
              ))}

              {/* Host rows */}
              {hosts.map((h, i) => (
                <g key={h}>
                  <line
                    x1={0}
                    y1={LANE_LABEL_H + ROW_H * i}
                    x2={VB_W}
                    y2={LANE_LABEL_H + ROW_H * i}
                    stroke="var(--rule-hair)"
                    strokeWidth={1}
                    shapeRendering="crispEdges"
                  />
                  <text
                    x={8}
                    y={LANE_LABEL_H + ROW_H * i + ROW_H / 2 - 4}
                    fill="var(--paper-000)"
                    fontSize={11}
                    fontFamily="var(--face-data)"
                  >
                    {hostLabel(h)}
                  </text>
                  <text
                    x={8}
                    y={LANE_LABEL_H + ROW_H * i + ROW_H / 2 + 10}
                    fill="var(--paper-600)"
                    fontSize={9}
                    fontFamily="var(--face-data)"
                  >
                    {h}
                  </text>
                </g>
              ))}

              {/* Causal edges */}
              {campaign.edges.map((e) => {
                const a = pos.get(e.src);
                const b = pos.get(e.dst);
                const na = byId.get(e.src);
                const nb = byId.get(e.dst);
                if (!a || !b || !na || !nb) return null;
                const isForecast = nb.provenance === "FORECAST";
                const mx = (a.x + b.x) / 2;

                return (
                  <g key={`${e.src}-${e.dst}`}>
                    <path
                      d={`M ${a.x + 9} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x - 9} ${b.y}`}
                      fill="none"
                      stroke={isForecast ? FORECAST : "var(--rule-hard)"}
                      strokeWidth={1.2}
                      strokeDasharray={isForecast ? "3 3" : undefined}
                    />
                    <text
                      x={mx}
                      y={(a.y + b.y) / 2 - 5}
                      textAnchor="middle"
                      fill="var(--paper-600)"
                      fontSize={8.5}
                      fontFamily="var(--face-data)"
                    >
                      {e.causality_score.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* Alert nodes */}
              {campaign.nodes.map((n) => {
                const p = pos.get(n.node_id);
                if (!p) return null;
                const fc = n.provenance === "FORECAST";
                const on = n.node_id === sel;
                const root = roots.has(n.node_id);
                const c = sevColor(sevFromRisk(n.max_risk_score, threshold));

                return (
                  <g
                    key={n.node_id}
                    style={{ cursor: "pointer" }}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setSel(on ? null : n.node_id);
                    }}
                  >
                    {root && (
                      <rect
                        x={p.x - 17}
                        y={p.y - 17}
                        width={34}
                        height={34}
                        fill="none"
                        stroke="var(--paper-600)"
                        strokeWidth={1}
                        strokeDasharray="2 2"
                        shapeRendering="crispEdges"
                      />
                    )}
                    {on && (
                      <rect
                        x={p.x - 13}
                        y={p.y - 13}
                        width={26}
                        height={26}
                        fill="none"
                        stroke="var(--paper-000)"
                        strokeWidth={1}
                        shapeRendering="crispEdges"
                      />
                    )}
                    <rect
                      x={p.x - 9}
                      y={p.y - 9}
                      width={18}
                      height={18}
                      fill={fc ? "url(#cm-hatch)" : c}
                      stroke={fc ? FORECAST : "none"}
                      strokeWidth={fc ? 1 : 0}
                      shapeRendering="crispEdges"
                    />
                    <text x={p.x} y={p.y + 26} textAnchor="middle" fill="var(--paper-400)" fontSize={9} fontFamily="var(--face-data)">
                      {n.technique_id}
                    </text>
                    {root && (
                      <text x={p.x} y={p.y - 22} textAnchor="middle" fill="var(--paper-600)" fontSize={8} fontFamily="var(--face-ui)" letterSpacing="1.4">
                        ROOT
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </Panel>

        {/* ── Node table ─────────────────────────────────────────────── */}
        <Panel flush clip style={{ flex: "0 0 208px", display: "flex", flexDirection: "column" }}>
          <PanelHead title="Compacted alert nodes" note={`${observed.length} observed · ${forecast.length} forecast`} />
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 48 }}>Node</th>
                  <th style={{ width: 150 }}>Host</th>
                  <th style={{ width: 136 }}>Stage</th>
                  <th style={{ width: 88 }}>Technique</th>
                  <th style={{ width: 100 }}>Provenance</th>
                  <th className="num" style={{ width: 72 }}>Hits</th>
                  <th className="num" style={{ width: 80 }}>Max risk</th>
                  <th className="num" style={{ width: 80 }}>Conf.</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {campaign.nodes.map((n) => {
                  const s = sevFromRisk(n.max_risk_score, threshold);
                  return (
                    <tr
                      key={n.node_id}
                      className={n.node_id === sel ? "is-selected" : undefined}
                      onClick={() => setSel(n.node_id === sel ? null : n.node_id)}
                      style={{ cursor: "pointer", opacity: n.provenance === "FORECAST" ? 0.72 : 1 }}
                    >
                      <td className="key" style={{ borderLeft: `3px solid ${sevColor(s)}` }}>
                        {String(n.node_id).padStart(2, "0")}
                      </td>
                      <td className="key">{hostLabel(n.host_ip)}</td>
                      <td>{n.coarse_category}</td>
                      <td className="key">{n.technique_id}</td>
                      <td style={{ color: n.provenance === "FORECAST" ? FORECAST : "var(--paper-400)" }}>{n.provenance}</td>
                      <td className="num">{n.hit_count.toLocaleString()}</td>
                      <td className="num key" style={{ color: sevColor(s) }}>
                        {n.max_risk_score.toFixed(3)}
                      </td>
                      <td className="num">{n.mean_confidence.toFixed(2)}</td>
                      <td>{roots.has(n.node_id) ? <Chip>root cause</Chip> : null}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* ── Rail ─────────────────────────────────────────────────────── */}
      <div className="cm-rail">
        <PanelHead
          title={selected ? "Node" : "Campaign"}
          aside={<Chip level={sevFromRisk(campaign.max_risk_score, threshold)}>#{campaign.campaign_id}</Chip>}
        />

        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {selected ? (
            <PanelBody>
              <Readout
                label={`Node ${String(selected.node_id).padStart(2, "0")}`}
                value={selected.technique_id}
                level={sevFromRisk(selected.max_risk_score, threshold)}
                sub={selected.coarse_category}
              />
              <div style={{ marginTop: "var(--s-4)" }}>
                <Field label="Host" value={`${hostLabel(selected.host_ip)} · ${selected.host_ip}`} />
                <Field label="Stage" value={selected.coarse_category} />
                <Field
                  label="Provenance"
                  value={selected.provenance}
                  color={selected.provenance === "FORECAST" ? FORECAST : undefined}
                />
                <Field label="Hits" value={selected.hit_count.toLocaleString()} />
                <Field
                  label="Max risk"
                  value={selected.max_risk_score.toFixed(3)}
                  color={sevColor(sevFromRisk(selected.max_risk_score, threshold))}
                />
                <Field label="Mean conf." value={selected.mean_confidence.toFixed(3)} />
                <Field label="First seen" value={hhmmss(selected.start_time)} />
                <Field label="Last seen" value={hhmmss(selected.end_time)} />
                <Field label="Root cause" value={roots.has(selected.node_id) ? "yes" : "no"} />
              </div>

              <div style={{ marginTop: "var(--s-4)" }}>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Causal links</Micro>
                {campaign.edges
                  .filter((e) => e.src === selected.node_id || e.dst === selected.node_id)
                  .map((e) => {
                    const other = byId.get(e.src === selected.node_id ? e.dst : e.src);
                    return (
                      <Field
                        key={`${e.src}-${e.dst}`}
                        label={e.src === selected.node_id ? "→ downstream" : "← upstream"}
                        value={`${other?.technique_id ?? "?"}  ${e.causality_score.toFixed(2)}`}
                      />
                    );
                  })}
              </div>
            </PanelBody>
          ) : (
            <>
              <PanelBody>
                <Readout
                  label="Max risk in campaign"
                  value={campaign.max_risk_score.toFixed(3)}
                  level={sevFromRisk(campaign.max_risk_score, threshold)}
                  scale="hero"
                  sub={`${observed.length} observed stages`}
                />
              </PanelBody>

              <PanelHead title="Summary" />
              <PanelBody>
                <Field label="Campaign" value={`#${campaign.campaign_id}`} />
                <Field label="Hosts" value={String(campaign.involved_hosts.length)} />
                <Field label="Nodes" value={`${observed.length} obs · ${forecast.length} fc`} />
                <Field label="Edges" value={String(campaign.edges.length)} />
                <Field label="Duration" value={`${campaign.duration_sec.toFixed(0)}s`} />
                <Field label="First seen" value={hhmmss(campaign.start_time)} />
                <Field label="Last seen" value={hhmmss(campaign.end_time)} />
                <Field
                  label="Forecast components"
                  value={campaign.has_forecast_components ? "yes" : "no"}
                  color={campaign.has_forecast_components ? FORECAST : undefined}
                />
              </PanelBody>

              <PanelHead title="Involved hosts" />
              <PanelBody>
                {campaign.involved_hosts.map((h) => (
                  <Field key={h} label={hostLabel(h)} value={h} />
                ))}
              </PanelBody>

              <PanelHead title="Techniques observed" />
              <PanelBody>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s-1)" }}>
                  {campaign.attack_techniques.length ? (
                    campaign.attack_techniques.map((t) => <Chip key={t}>{t}</Chip>)
                  ) : (
                    <Data size="s" color="var(--paper-600)">
                      none yet
                    </Data>
                  )}
                </div>
              </PanelBody>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
