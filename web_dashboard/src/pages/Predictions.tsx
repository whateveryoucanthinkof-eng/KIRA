import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import type { TelemetryPoint, PredictionResult, ForecastPoint } from "../api/types";

interface PredictionsProps {
  telemetry: TelemetryPoint[];
  forecast: ForecastPoint[];
  prediction: PredictionResult | null;
}

function fmtTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" });
  } catch { return ts; }
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: unknown[]; label?: string }) {
  if (!active || !payload?.length) return null;
  const items = payload as Array<{ name: string; value: number; color: string }>;
  return (
    <div style={{
      background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6,
      padding: "8px 12px", fontSize: 11, fontFamily: "var(--font-mono)",
      boxShadow: "0 2px 8px rgba(0,0,0,0.07)", zIndex: 50,
    }}>
      <div style={{ color: "var(--color-text-muted)", marginBottom: 5 }}>{label}</div>
      {items.map((item) => (
        <div key={item.name} style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 2 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: item.color, display: "inline-block", flexShrink: 0 }} />
          <span style={{ color: "var(--color-text-secondary)" }}>{item.name}</span>
          <span style={{ color: "var(--color-text-primary)", fontWeight: 500, marginLeft: "auto", paddingLeft: 12 }}>{item.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function Predictions({ telemetry, forecast, prediction }: PredictionsProps) {
  const combined = [
    ...telemetry.slice(-60).map((p) => ({
      t: fmtTime(p.timestamp),
      observed: p.observed,
      predicted: p.predicted,
      band: p.upperBound - p.lowerBound,
      lower: p.lowerBound,
      confidence: Math.round(p.confidence * 100),
    })),
    ...forecast.slice(0, 30).map((p) => ({
      t: fmtTime(p.timestamp),
      observed: undefined as number | undefined,
      predicted: p.predicted,
      band: p.upperBound - p.lowerBound,
      lower: p.lowerBound,
      confidence: Math.round(p.confidence * 100),
    })),
  ];

  const combinedMin = combined.length ? Math.max(0, Math.min(...combined.map(d => d.lower)) - 80) : 0;

  const signalData = prediction?.signals.map((s) => ({
    name: s.name,
    weight: Math.round(s.weight * 100),
    direction: s.direction,
    value: s.value,
  })) ?? [];

  const confidenceHistory = telemetry.slice(-60).map((p) => ({
    t: fmtTime(p.timestamp),
    confidence: Math.round(p.confidence * 100),
    anomaly: Math.round(p.anomalyScore),
  }));

  return (
    <div className="predictions-scroll">
      <div className="predictions-grid">

        {/* ── Left main column ── */}
        <div className="predictions-main">

          {/* Main time-series chart */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Observed vs Predicted — 60m History + 30m Forecast</span>
              <div style={{ display: "flex", gap: 12, fontSize: 10, flexShrink: 0 }}>
                {[
                  { color: "var(--color-status-blue)", label: "Observed" },
                  { color: "var(--color-status-amber)", label: "Predicted", dashed: true },
                  { color: "var(--color-status-amber)", label: "Band", band: true },
                ].map(({ color, label, dashed, band }) => (
                  <span key={label} style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>
                    {band
                      ? <span style={{ width: 14, height: 8, background: color, opacity: 0.12, display: "inline-block", borderRadius: 1 }} />
                      : <span style={{ width: 14, height: 2, background: color, display: "inline-block", borderRadius: 1, borderTop: dashed ? `2px dashed ${color}` : undefined }} />
                    }
                    {label}
                  </span>
                ))}
              </div>
            </div>

            <div style={{ padding: "12px 8px 4px", height: 300 }}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={combined} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="predBand" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="var(--color-status-amber)" stopOpacity={0.18} />
                      <stop offset="100%" stopColor="var(--color-status-amber)" stopOpacity={0.04} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" strokeOpacity={0.7} />
                  <XAxis dataKey="t" tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} interval={14} />
                  <YAxis domain={[combinedMin, "auto"]} tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} width={40} unit="M" />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine x={combined[59]?.t} stroke="var(--color-border-strong)" strokeDasharray="3 3"
                    label={{ value: "now", position: "top", fontSize: 9, fill: "var(--color-text-muted)" }} />
                  <Area type="monotone" dataKey="lower" stackId="pb" stroke="none" fill="transparent" legendType="none" />
                  <Area type="monotone" dataKey="band" stackId="pb" stroke="none" fill="url(#predBand)" legendType="none" />
                  <Line type="monotone" dataKey="observed" stroke="var(--color-status-blue)" strokeWidth={1.5} dot={false} connectNulls={false} name="Observed" />
                  <Line type="monotone" dataKey="predicted" stroke="var(--color-status-amber)" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="Predicted" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Confidence / anomaly overlay */}
            <div style={{ borderTop: "1px solid var(--color-border)", padding: "8px 8px 6px" }}>
              <div className="panel-title" style={{ paddingLeft: 4, marginBottom: 6 }}>Model Confidence &amp; Anomaly Score</div>
              <div style={{ height: 72 }}>
                <ResponsiveContainer width="100%" height={72}>
                  <LineChart data={confidenceHistory} margin={{ top: 2, right: 16, left: 40, bottom: 0 }}>
                    <XAxis dataKey="t" hide />
                    <YAxis domain={[0, 100]} hide />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine y={70} stroke="var(--color-status-amber)" strokeDasharray="3 3" strokeOpacity={0.4} />
                    <Line type="monotone" dataKey="confidence" stroke="var(--color-status-blue)" strokeWidth={1.5} dot={false} name="Confidence %" />
                    <Line type="monotone" dataKey="anomaly" stroke="var(--color-status-red)" strokeWidth={1.5} dot={false} name="Anomaly Score" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: "flex", gap: 14, paddingLeft: 44, marginTop: 4 }}>
                {[{ color: "var(--color-status-blue)", label: "Confidence %" }, { color: "var(--color-status-red)", label: "Anomaly Score" }].map(({ color, label }) => (
                  <span key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "var(--color-text-muted)" }}>
                    <span style={{ width: 14, height: 2, background: color, display: "inline-block", borderRadius: 1 }} />
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Forecast table */}
          <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="panel-header">
              <span className="panel-title">30-Minute Forecast Table</span>
            </div>
            <div style={{ overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "var(--color-base)" }}>
                    {["Time", "Horizon", "Predicted (Mbps)", "Lower", "Upper", "Confidence"].map((h) => (
                      <th key={h} style={{
                        padding: "6px 12px", textAlign: "left", fontSize: 10, fontWeight: 600,
                        letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--color-text-muted)",
                        borderBottom: "1px solid var(--color-border)", position: "sticky", top: 0,
                        background: "var(--color-base)", whiteSpace: "nowrap",
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {forecast.map((f, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid var(--color-border)", background: i % 2 === 0 ? "transparent" : "var(--color-base)" }}>
                      <td style={{ padding: "5px 12px", fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>{fmtTime(f.timestamp)}</td>
                      <td style={{ padding: "5px 12px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>+{i + 1}m</td>
                      <td style={{ padding: "5px 12px", fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--color-text-primary)" }}>{f.predicted}</td>
                      <td style={{ padding: "5px 12px", fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>{f.lowerBound}</td>
                      <td style={{ padding: "5px 12px", fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>{f.upperBound}</td>
                      <td style={{ padding: "5px 12px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 44, height: 3, background: "var(--color-border)", borderRadius: 2, flexShrink: 0 }}>
                            <div style={{
                              width: `${Math.round(f.confidence * 100)}%`, height: "100%", borderRadius: 2,
                              background: f.confidence >= 0.8 ? "var(--color-status-green)" : f.confidence >= 0.6 ? "var(--color-status-amber)" : "var(--color-status-red)",
                            }} />
                          </div>
                          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-text-muted)", flexShrink: 0 }}>{(f.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>{/* end predictions-main */}

        {/* ── Right sidebar ── */}
        <div className="predictions-sidebar">

          {/* Current prediction summary */}
          {prediction && (
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Current Prediction</span>
              </div>
              <div style={{ padding: 14 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                  {[
                    {
                      label: "Attack Probability",
                      value: `${(prediction.value * 100).toFixed(1)}%`,
                      accent: prediction.value >= 0.7 ? "var(--color-status-red)" : prediction.value >= 0.4 ? "var(--color-status-amber)" : "var(--color-status-green)",
                    },
                    { label: "Confidence",  value: `${(prediction.confidence * 100).toFixed(1)}%`, accent: "var(--color-text-primary)" },
                    { label: "Horizon",     value: `${prediction.horizon}m`,                        accent: "var(--color-text-primary)" },
                    { label: "Model",       value: prediction.model.split("/")[0].trim(),            accent: "var(--color-text-secondary)" },
                  ].map(({ label, value, accent }) => (
                    <div key={label} style={{ background: "var(--color-base)", borderRadius: 6, padding: "8px 10px", minWidth: 0 }}>
                      <div style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--color-text-muted)", marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "var(--font-mono)", color: accent, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{value}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {prediction.model}
                </div>
              </div>
            </div>
          )}

          {/* Contributing signals */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Contributing Signals</span>
            </div>
            <div style={{ padding: "10px 8px 4px", height: 200 }}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={signalData} layout="vertical" margin={{ top: 0, right: 16, left: 4, bottom: 0 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} unit="%" />
                  <YAxis type="category" dataKey="name" width={0} tick={false} axisLine={false} />
                  <Tooltip
                    cursor={{ fill: "var(--color-base)" }}
                    formatter={(v: unknown) => [`${v}%`, "Weight"]}
                    contentStyle={{ fontSize: 11, fontFamily: "var(--font-mono)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                  />
                  <Bar dataKey="weight" radius={[0, 3, 3, 0]}>
                    {signalData.map((entry, index) => (
                      <Cell
                        key={index}
                        fill={
                          entry.direction === "positive" ? "var(--color-status-red)" :
                          entry.direction === "negative" ? "var(--color-status-green)" :
                          "var(--color-status-blue)"
                        }
                        fillOpacity={0.75}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ padding: "4px 14px 10px" }}>
              {signalData.map((sig) => (
                <div key={sig.name} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 10, borderBottom: "1px solid var(--color-border)", gap: 8 }}>
                  <span style={{
                    color: sig.direction === "positive" ? "var(--color-status-red)" : sig.direction === "negative" ? "var(--color-status-green)" : "var(--color-text-secondary)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1,
                  }}>
                    {sig.name}
                  </span>
                  <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", flexShrink: 0 }}>{sig.value}</span>
                </div>
              ))}
            </div>
          </div>

        </div>{/* end predictions-sidebar */}

      </div>{/* end predictions-grid */}
    </div>
  );
}
