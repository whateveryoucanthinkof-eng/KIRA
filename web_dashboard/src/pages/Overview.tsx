import { useState } from "react";
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
} from "recharts";
import type {
  SystemStatus,
  TelemetryPoint,
  ForecastPoint,
  PredictionResult,
  AttackEvent,
  NetworkEvent,
  Topology,
} from "../api/types";
import { SeverityBadge, StatusDot } from "../components/shared/StatusBadge";
import { MetricCard } from "../components/shared/MetricCard";

interface OverviewProps {
  status: SystemStatus | null;
  telemetry: TelemetryPoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
  attackEvents: AttackEvent[];
  events: NetworkEvent[];
  topology: Topology | null;
  logLines: string[];
}

function fmtTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" });
  } catch { return ts; }
}

function fmtTs(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch { return ts; }
}

function fmtUptime(s: number): string {
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${d}d ${h}h ${m}m`;
}

function ChartTip({ active, payload, label }: { active?: boolean; payload?: unknown[]; label?: string }) {
  if (!active || !payload?.length) return null;
  const items = payload as Array<{ name: string; value: number; color: string }>;
  return (
    <div style={{
      background: "var(--color-surface)",
      border: "1px solid var(--color-border)",
      borderRadius: 6,
      padding: "8px 12px",
      fontSize: 11,
      fontFamily: "var(--font-mono)",
      boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      zIndex: 50,
    }}>
      <div style={{ color: "var(--color-text-muted)", marginBottom: 4 }}>{label}</div>
      {items.map((item) => (
        <div key={item.name} style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: item.color, display: "inline-block", flexShrink: 0 }} />
          <span style={{ color: "var(--color-text-secondary)" }}>{item.name}</span>
          <span style={{ color: "var(--color-text-primary)", fontWeight: 500, marginLeft: "auto", paddingLeft: 16 }}>{item.value}</span>
        </div>
      ))}
    </div>
  );
}

function TopologyMini({ topology }: { topology: Topology }) {
  const nodeColor: Record<string, string> = {
    online: "var(--color-status-green)",
    offline: "var(--color-text-muted)",
    degraded: "var(--color-status-amber)",
    compromised: "var(--color-status-red)",
    warning: "var(--color-status-amber)",
  };
  const edgeColor: Record<string, string> = {
    active: "#d1d5db",
    saturated: "var(--color-status-red)",
    down: "#e5e7eb",
    suspicious: "var(--color-status-amber)",
  };
  return (
    <svg width="100%" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet" style={{ display: "block" }}>
      {topology.edges.map((edge) => {
        const src = topology.nodes.find((n) => n.id === edge.source);
        const tgt = topology.nodes.find((n) => n.id === edge.target);
        if (!src || !tgt) return null;
        return (
          <line key={edge.id} x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
            stroke={edgeColor[edge.status] ?? "#d1d5db"}
            strokeWidth={edge.status === "saturated" ? 2.5 : 1.5}
            strokeDasharray={edge.status === "suspicious" ? "4 3" : undefined}
            opacity={0.8}
          />
        );
      })}
      {topology.nodes.map((node) => {
        const color = nodeColor[node.status] ?? "var(--color-text-muted)";
        const r = node.type === "router" || node.type === "firewall" ? 9 : node.type === "internet" ? 11 : node.type === "attacker" ? 8 : 7;
        return (
          <g key={node.id}>
            <circle cx={node.x} cy={node.y} r={r} fill={color} fillOpacity={0.15} stroke={color} strokeWidth={1.5} />
            <text x={node.x} y={node.y + r + 11} textAnchor="middle"
              style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-text-secondary)" }}>
              {node.label}
            </text>
            {node.ip && (
              <text x={node.x} y={node.y + r + 20} textAnchor="middle"
                style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)" }}>
                {node.ip}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}



function logLineColor(line: string): string {
  if (line.includes("CRITICAL") || line.includes("ALERT")) return "var(--color-status-red)";
  if (line.includes("WARN")) return "var(--color-status-amber)";
  if (line.includes("ERROR") || line.includes("DENY") || line.includes("BLOCK")) return "#e57373";
  if (line.includes("INFO")) return "#9ca3af";
  return "#6b7280";
}

export default function Overview({
  status, telemetry, forecast, prediction, attackEvents, events, topology, logLines,
}: OverviewProps) {
  if (!status) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-muted)", fontSize: 13 }}>
        Loading…
      </div>
    );
  }

  const chartData = telemetry.slice(-60).map((p) => ({
    t: fmtTime(p.timestamp),
    observed: p.observed,
    predicted: p.predicted,
    band: p.upperBound - p.lowerBound,
    lower: p.lowerBound,
    anomaly: Math.round(p.anomalyScore),
  }));

  const forecastChartData = forecast.slice(0, 20).map((p) => ({
    t: fmtTime(p.timestamp),
    predicted: p.predicted,
    band: p.upperBound - p.lowerBound,
    lower: p.lowerBound,
  }));

  const trafficMin = chartData.length ? Math.max(0, Math.min(...chartData.map(d => d.lower)) - 80) : 0;
  const forecastMin = forecastChartData.length ? Math.max(0, Math.min(...forecastChartData.map(d => d.lower)) - 60) : 0;

  const [explainExpanded, setExplainExpanded] = useState(true);
  const recentLogs = logLines.slice(-8);

  return (
    <div className="overview-scroll">

      {/* ── Row 1: Status bar ─────────────────────────────────── */}
      <div className="ov-status">
        <MetricCard label="Throughput" value={Math.round(status.throughput)} unit="Mbps" sub="10 GbE uplink" accent={status.throughput > 900 ? "red" : "default"} mono />
        <MetricCard label="Latency" value={status.latency.toFixed(0)} unit="ms" sub="edge→core" mono />
        <MetricCard label="Packet Loss" value={status.packetLoss.toFixed(1)} unit="%" sub="5m avg" accent={status.packetLoss > 1 ? "amber" : "default"} mono />
        <MetricCard label="Connections" value={status.activeConnections.toLocaleString()} sub="established TCP" mono />
        <MetricCard label="Anomaly Score" value={Math.round(status.anomalyScore)} unit="/ 100" accent={status.anomalyScore >= 70 ? "red" : status.anomalyScore >= 40 ? "amber" : "green"} mono />
        <MetricCard label="Threat Level" value={status.threatLevel.toUpperCase()} sub="current" accent={status.threatLevel === "high" || status.threatLevel === "critical" ? "red" : "amber"} />
        <MetricCard label="Uptime" value={fmtUptime(status.uptime)} sub="this session" />
      </div>

      {/* ── Rows 2-4: Main 2-col area ─────────────────────────── */}
      <div className="ov-main">

        {/* Left column */}
        <div className="ov-left">

          {/* Live Traffic */}
          <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="panel-header">
              <span className="panel-title">Live Traffic — Observed vs Predicted</span>
              <div style={{ display: "flex", gap: 14, alignItems: "center", flexShrink: 0 }}>
                <LegendItem color="var(--color-status-blue)" label="Observed" />
                <LegendItem color="var(--color-status-amber)" label="Predicted" dashed />
              </div>
            </div>
            <div style={{ padding: "12px 8px 4px", height: 200 }}>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="confBand" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"  stopColor="var(--color-status-amber)" stopOpacity={0.18} />
                      <stop offset="100%" stopColor="var(--color-status-amber)" stopOpacity={0.04} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" strokeOpacity={0.6} />
                  <XAxis dataKey="t" tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} interval={9} />
                  <YAxis domain={[trafficMin, "auto"]} tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} width={38} unit="M" />
                  <Tooltip content={<ChartTip />} />
                  <ReferenceLine y={900} stroke="var(--color-status-red)" strokeDasharray="4 3" strokeOpacity={0.5}
                    label={{ value: "cap", position: "right", fontSize: 9, fill: "var(--color-status-red)" }} />
                  {/* Stacked band: transparent base (lower) + visible band (upper-lower) */}
                  <Area type="monotone" dataKey="lower" stackId="cb" stroke="none" fill="transparent" legendType="none" />
                  <Area type="monotone" dataKey="band" stackId="cb" stroke="none" fill="url(#confBand)" legendType="none" />
                  <Line type="monotone" dataKey="observed" stroke="var(--color-status-blue)" strokeWidth={1.5} dot={false} name="Observed" />
                  <Line type="monotone" dataKey="predicted" stroke="var(--color-status-amber)" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="Predicted" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div style={{ borderTop: "1px solid var(--color-border)", padding: "8px 8px 4px" }}>
              <div className="panel-title" style={{ paddingLeft: 4, marginBottom: 6 }}>Anomaly Score (60m)</div>
              <div style={{ height: 40 }}>
                <ResponsiveContainer width="100%" height={40}>
                  <AreaChart data={chartData} margin={{ top: 0, right: 16, left: 38, bottom: 0 }}>
                    <defs>
                      <linearGradient id="anomGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="var(--color-status-red)" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="var(--color-status-red)" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="anomaly" stroke="var(--color-status-red)" strokeWidth={1} fill="url(#anomGrad)" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Middle row: Network Topology | 30-Min Forecast */}
          <div className="ov-mid">

            {/* Network Topology */}
            {topology ? (
              <div className="panel panel-clipped" style={{ display: "flex", flexDirection: "column" }}>
                <div className="panel-header">
                  <span className="panel-title">Network Topology</span>
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", flexShrink: 0 }}>
                    {topology.nodes.length}n · {topology.edges.length}l
                  </span>
                </div>
                <div style={{ padding: "8px 4px", flex: 1 }}>
                  <TopologyMini topology={topology} />
                </div>
              </div>
            ) : (
              <div className="panel" style={{ padding: 16, color: "var(--color-text-muted)", fontSize: 12 }}>No topology data</div>
            )}

            {/* 30-Min Forecast */}
            <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
              <div className="panel-header">
                <span className="panel-title">30-Min Forecast</span>
                <span style={{ fontSize: 9, color: "var(--color-text-muted)", flexShrink: 0 }}>horizon</span>
              </div>
              <div style={{ padding: "10px 8px 4px", height: 120 }}>
                <ResponsiveContainer width="100%" height={120}>
                  <ComposedChart data={forecastChartData} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="fcBand" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%"  stopColor="var(--color-status-blue)" stopOpacity={0.18} />
                        <stop offset="100%" stopColor="var(--color-status-blue)" stopOpacity={0.04} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" strokeOpacity={0.6} />
                    <XAxis dataKey="t" tick={{ fontSize: 8, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} interval={4} />
                    <YAxis domain={[forecastMin, "auto"]} tick={{ fontSize: 8, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} width={32} unit="M" />
                    <Tooltip content={<ChartTip />} />
                    <Area type="monotone" dataKey="lower" stackId="fc" stroke="none" fill="transparent" legendType="none" />
                    <Area type="monotone" dataKey="band" stackId="fc" stroke="none" fill="url(#fcBand)" legendType="none" />
                    <Line type="monotone" dataKey="predicted" stroke="var(--color-status-blue)" strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="Forecast" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div style={{ borderTop: "1px solid var(--color-border)", padding: "6px 12px 8px" }}>
                {forecast.slice(0, 4).map((f, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "3px 0", borderBottom: i < 3 ? "1px solid var(--color-border)" : "none" }}>
                    <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>+{(i + 1) * 5}m</span>
                    <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                      {f.predicted}<span style={{ color: "var(--color-text-muted)", fontSize: 9 }}> M</span>
                    </span>
                    <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                      ci {(f.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Model Inference Provenance & Active Threats — full left-column width */}
          <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="panel-header">
              <span className="panel-title">Model Inference Provenance & Active Threats</span>
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-status-red)", flexShrink: 0 }}>
                {attackEvents.filter((e) => !e.mitigated).length} active threats
              </span>
            </div>
            <div style={{ padding: "10px 12px" }}>
              {/* Dynamic Provenance Rendering */}
              {prediction?.stage_provenance ? (
                <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                  {Object.entries(prediction.stage_provenance).map(([key, val], idx, arr) => (
                    <div key={key} style={{ flex: 1, display: "flex", alignItems: "center" }}>
                      <div style={{ flex: 1, padding: "4px 8px", background: "var(--color-base)", borderRadius: 4, border: "1px solid var(--color-border)" }}>
                        <div style={{ fontSize: 9, color: "var(--color-text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: 2 }}>{key}</div>
                        <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-status-blue)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{val}</div>
                      </div>
                      {idx < arr.length - 1 && (
                        <div style={{ padding: "0 4px", color: "var(--color-text-muted)" }}>→</div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 11, color: "var(--color-text-muted)", marginBottom: 12 }}>Provenance data unavailable</div>
              )}
              {/* Attack event cards — horizontal when wide */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 6 }}>
                {attackEvents.map((atk) => (
                  <div key={atk.id} style={{
                    padding: "6px 8px",
                    borderRadius: 4,
                    background: atk.mitigated ? "var(--color-base)" : "var(--color-status-red-bg)",
                    border: `1px solid ${atk.mitigated ? "var(--color-border)" : "var(--color-status-red)"}22`,
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 2, gap: 6 }}>
                      <span style={{ fontSize: 11, fontWeight: 600, color: atk.mitigated ? "var(--color-text-muted)" : "var(--color-status-red)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {atk.type}
                      </span>
                      <span style={{ fontSize: 9, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
                        {atk.mitigated ? "mitigated" : atk.stage}
                      </span>
                    </div>
                    <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {atk.sourceIp} → {atk.targetIp}
                    </div>
                    <div style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginTop: 2 }}>
                      {atk.packets.toLocaleString()} pkts · {atk.bytes.toLocaleString()} B
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right column: Prediction Summary + Recent Events */}
        <div className="ov-right">

          {/* Prediction Summary */}
          {prediction && (
            <div className="panel" style={{ flexShrink: 0 }}>
              <div className="panel-header">
                <span className="panel-title">Prediction Summary</span>
                <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                  {prediction.horizon}m horizon
                </span>
              </div>
              <div style={{ padding: 12 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 6 }}>
                  <span style={{
                    fontSize: 36,
                    fontWeight: 700,
                    fontFamily: "var(--font-mono)",
                    lineHeight: 1,
                    color: prediction.value >= 0.7 ? "var(--color-status-red)" : prediction.value >= 0.4 ? "var(--color-status-amber)" : "var(--color-status-green)",
                  }}>
                    {(prediction.value * 100).toFixed(0)}%
                  </span>
                  <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>attack prob.</span>
                </div>
                <div style={{ display: "flex", gap: 12, marginBottom: 10 }}>
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                    confidence <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{(prediction.confidence * 100).toFixed(0)}%</span>
                  </span>
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {prediction.model}
                  </span>
                </div>
                <div className="panel-title" style={{ marginBottom: 8 }}>Contributing Signals</div>
                {prediction.signals.slice(0, 6).map((sig) => (
                  <div key={sig.name} style={{ marginBottom: 7 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, gap: 6 }}>
                      <span style={{ fontSize: 10, color: "var(--color-text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {sig.name}
                      </span>
                      <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", flexShrink: 0 }}>
                        {(sig.weight * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div style={{ height: 4, background: "var(--color-base)", borderRadius: 2 }}>
                      <div style={{
                        width: `${sig.weight * 100}%`, height: "100%", borderRadius: 2,
                        background: sig.direction === "positive" ? "var(--color-status-red)" : sig.direction === "negative" ? "var(--color-status-green)" : "var(--color-text-muted)",
                        transition: "width 0.3s ease",
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Events — fills remaining height */}
          <div className="panel panel-clipped ov-events">
            <div className="panel-header" style={{ flexShrink: 0 }}>
              <span className="panel-title">Recent Events</span>
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                {events.length} total
              </span>
            </div>
            <div style={{ overflowY: "auto", flex: 1, minHeight: 0 }}>
              {events.slice(0, 12).map((ev) => (
                <div key={ev.id} style={{
                  padding: "7px 12px",
                  borderBottom: "1px solid var(--color-border)",
                  display: "flex",
                  gap: 8,
                  alignItems: "flex-start",
                }}>
                  <div style={{ flexShrink: 0, paddingTop: 1 }}>
                    <SeverityBadge severity={ev.severity} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ev.message}
                    </div>
                    <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginTop: 1 }}>
                      {fmtTs(ev.timestamp)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Row 5: Bottom row — Anomaly | Explainability | Console ── */}
      <div className="ov-bottom">

        {/* Anomaly Score + service status */}
        <div className="panel" style={{ padding: "12px 14px" }}>
          <div className="panel-title" style={{ marginBottom: 10 }}>Live Anomaly Score</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ flex: 1, height: 6, background: "var(--color-base)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                width: `${status.anomalyScore}%`, height: "100%", borderRadius: 3,
                background: status.anomalyScore >= 80 ? "var(--color-status-red)" : status.anomalyScore >= 50 ? "var(--color-status-amber)" : "var(--color-status-green)",
                transition: "width 0.4s ease",
              }} />
            </div>
            <span style={{
              fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", minWidth: 32, textAlign: "right", flexShrink: 0,
              color: status.anomalyScore >= 80 ? "var(--color-status-red)" : status.anomalyScore >= 50 ? "var(--color-status-amber)" : "var(--color-status-green)",
            }}>
              {status.anomalyScore.toFixed(0)}
            </span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--color-text-muted)", marginBottom: 14 }}>
            <span>Normal</span><span>Suspicious</span><span>Critical</span>
          </div>
          <div className="panel-title" style={{ marginBottom: 8 }}>Service Health</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {([
              ["Network",    status.networkStatus],
              ["Telemetry",  status.telemetryStatus],
              ["Prediction", status.predictionStatus],
              ["Attack Det.", status.attackStatus === "none" ? "online" : status.attackStatus === "active" ? "compromised" : "warning"],
            ] as [string, string][]).map(([label, st]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <StatusDot status={st as "online" | "offline" | "degraded" | "compromised" | "warning"} pulse />
                <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{label}</span>
                <span style={{ marginLeft: "auto", fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>{st}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ML Explainability — SHAP-style feature impact */}
        <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-header">
            <span className="panel-title">ML Explainability — Feature Impact</span>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              {prediction && (
                <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                  {prediction.model.split(" ").slice(0, 2).join(" ")}
                </span>
              )}
              <button
                onClick={() => setExplainExpanded(v => !v)}
                style={{
                  fontSize: 10, fontWeight: 600, letterSpacing: "0.04em",
                  padding: "2px 8px", borderRadius: 4, cursor: "pointer",
                  border: "1px solid var(--color-border-strong)", background: "transparent",
                  color: "var(--color-status-blue)", whiteSpace: "nowrap",
                  fontFamily: "var(--font-sans)",
                }}
              >
                {explainExpanded ? "HIDE FEATURE DRIVERS" : "VIEW TOP FEATURE DRIVERS"}
              </button>
            </div>
          </div>
          {explainExpanded && (
            <div style={{ padding: "10px 14px 12px", flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
              {prediction ? prediction.signals.map((sig) => {
                const impact = sig.weight * 100;
                const isPos = sig.direction === "positive";
                const isNeg = sig.direction === "negative";
                const barColor = isPos ? "var(--color-status-red)" : isNeg ? "var(--color-status-green)" : "var(--color-text-muted)";
                const dirLabel = isPos ? "↑ risk" : isNeg ? "↓ risk" : "neutral";
                return (
                  <div key={sig.name} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8, alignItems: "center" }}>
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, alignItems: "baseline" }}>
                        <span style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{sig.name}</span>
                        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: barColor, marginLeft: 8, flexShrink: 0 }}>{dirLabel}</span>
                      </div>
                      <div style={{ height: 5, background: "var(--color-base)", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ width: `${impact}%`, height: "100%", borderRadius: 3, background: barColor }} />
                      </div>
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)", minWidth: 32, textAlign: "right" }}>
                      {impact.toFixed(0)}%
                    </span>
                  </div>
                );
              }) : (
                <div style={{ color: "var(--color-text-muted)", fontSize: 12, paddingTop: 8 }}>No prediction data</div>
              )}
            </div>
          )}
        </div>

        {/* System Console */}
        <div className="panel panel-clipped" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-header" style={{ flexShrink: 0 }}>
            <span className="panel-title">System Log</span>
            <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
              live · {logLines.length}
            </span>
          </div>
          <div style={{
            flex: 1,
            overflow: "auto",
            padding: "8px 10px",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            background: "#1a1e24",
            borderRadius: "0 0 8px 8px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
          }}>
            {recentLogs.map((line, i) => (
              <div key={i} style={{ color: logLineColor(line), lineHeight: 1.65, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {line}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

function LegendItem({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, whiteSpace: "nowrap" }}>
      <span style={{
        width: 20, height: 2, background: dashed ? "transparent" : color, display: "inline-block", borderRadius: 1,
        borderTop: dashed ? `2px dashed ${color}` : undefined,
      }} />
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
    </span>
  );
}
