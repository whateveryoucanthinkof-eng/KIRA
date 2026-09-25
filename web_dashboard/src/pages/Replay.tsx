import { useRef, useState } from "react";
import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip } from "recharts";
import { uploadReplay, analyseSample } from "../api/adapter";
import { REPLAY_SAMPLES, type ReplaySample } from "../api/mock";
import type { ReplayReport, ReplayRow } from "../types/replay";
import {
  BarRow,
  Btn,
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

/**
 * Offline capture analysis.
 *
 * PS 26153 asks for an interface that accepts a PCAP or CSV and runs fully
 * offline. `POST /api/replay` does exactly that — local parse, window-by-window
 * scoring, temp file deleted — and the backend forces the SOC rule layer off,
 * so every risk on this page is pure model output with no heuristic floor.
 */

const ACCEPT = ".pcap,.pcapng,.csv,.binetflow";
const THRESHOLD = 0.65;

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

export default function Replay() {
  const [report, setReport] = useState<ReplayReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const [pending, setPending] = useState<{ name: string; size: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function run(name: string, size: number, work: () => Promise<ReplayReport>) {
    setBusy(true);
    setError(null);
    setReport(null);
    setSel(null);
    setPending({ name, size });
    try {
      setReport(await work());
    } catch (e) {
      setError(e instanceof Error ? e.message : "analysis failed");
    } finally {
      setBusy(false);
    }
  }

  const analyse = (file: File) => run(file.name, file.size, () => uploadReplay(file));
  const analyseBuiltIn = (s: ReplaySample) => run(s.name, s.bytes, () => analyseSample(s));

  const rows = report?.results ?? [];
  const peak = rows.length ? Math.max(...rows.map((r) => r.risk)) : 0;
  const selected: ReplayRow | null = sel != null ? (rows.find((r) => r.window === sel) ?? null) : null;

  const series = rows.map((r) => ({
    t: String(r.window),
    risk: r.risk,
    forecast: r.forecast.length ? Math.max(...r.forecast.map((f) => f.risk)) : undefined,
  }));

  return (
    <div className="rp">
      <div className="rp-main sheet">
        {/* ── Intake ───────────────────────────────────────────────── */}
        <Panel flush>
          <PanelHead
            title="Offline analysis"
            note="pcap · pcapng · csv · binetflow"
            aside={report ? <Chip level="nominal">rules disabled</Chip> : undefined}
          />
          <PanelBody>
            <div
              className="rp-drop"
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void analyse(f);
              }}
              style={{
                borderColor: dragging ? "var(--paper-000)" : "var(--rule-hard)",
                background: dragging ? "var(--ink-200)" : "transparent",
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void analyse(f);
                  e.target.value = "";
                }}
              />

              {busy ? (
                <div style={{ textAlign: "center" }}>
                  <Micro style={{ marginBottom: "var(--s-2)" }}>Analysing</Micro>
                  <div className="t-data" style={{ color: "var(--paper-000)", marginBottom: "var(--s-3)" }}>
                    {pending?.name}
                  </div>
                  <div style={{ width: 280, margin: "0 auto" }}>
                    <div className="rp-bar" />
                  </div>
                  <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: "var(--s-2)" }}>
                    parsing · windowing · scoring
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: "center" }}>
                  <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-2)" }}>
                    Drop a capture to analyse
                  </div>
                  <div className="t-body" style={{ color: "var(--paper-400)", marginBottom: "var(--s-4)", maxWidth: 460 }}>
                    Parsed, windowed and scored locally. Nothing leaves this machine, and the deterministic
                    rule layer is disabled so the scores are model output alone.
                  </div>
                  <Btn primary onClick={() => inputRef.current?.click()}>
                    Select file
                  </Btn>
                </div>
              )}
            </div>


            {/* ── Built-in captures ──────────────────────────────────
                Dragging a file on camera is awkward and a live audience has
                no capture to hand. These run the identical analysis path. */}
            <div style={{ marginTop: "var(--s-4)", paddingTop: "var(--s-3)", borderTop: "var(--hard)" }}>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "var(--s-2)" }}>
                <Micro>Sample captures</Micro>
                <Data size="s" color="var(--paper-600)">
                  {REPLAY_SAMPLES.length} available
                </Data>
              </div>

              <div className="rp-samples">
                {REPLAY_SAMPLES.map((s) => {
                  const active = pending?.name === s.name;
                  return (
                    <button
                      key={s.id}
                      disabled={busy}
                      onClick={() => void analyseBuiltIn(s)}
                      className="rp-sample"
                      style={{
                        borderLeft: `3px solid ${s.id === "benign" ? "var(--sev-nominal)" : OBSERVED}`,
                        background: active ? "var(--ink-200)" : "transparent",
                        opacity: busy && !active ? 0.45 : 1,
                        cursor: busy ? "not-allowed" : "pointer",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "var(--s-2)" }}>
                        <span className="t-label" style={{ color: "var(--paper-000)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {s.label}
                        </span>
                        <span className="t-data-s" style={{ color: "var(--paper-600)", flexShrink: 0 }}>
                          {s.kind.toUpperCase()}
                        </span>
                      </div>

                      <div
                        className="t-data-s"
                        style={{ color: "var(--paper-400)", marginTop: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                      >
                        {s.name}
                      </div>

                      <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: 2 }}>
                        {fmtBytes(s.bytes)} · {s.windows} windows
                      </div>

                      <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: "var(--s-2)", whiteSpace: "normal", lineHeight: 1.45 }}>
                        {s.note}
                      </div>

                      <div className="t-micro" style={{ color: "var(--paper-600)", marginTop: "var(--s-2)", opacity: 0.8 }}>
                        {s.source}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {error && (
              <div style={{ marginTop: "var(--s-3)", padding: "var(--s-2) var(--s-3)", border: "var(--hair)", borderLeft: "3px solid var(--sev-critical)" }}>
                <Data size="s" color="var(--sev-critical)">
                  {error}
                </Data>
              </div>
            )}
          </PanelBody>
        </Panel>

        {/* ── Result ───────────────────────────────────────────────── */}
        {report ? (
          <>
            <Panel flush>
              <PanelHead
                title="Risk across capture"
                note={`${report.filename} · ${report.windows_analyzed} windows`}
                aside={
                  <Legend
                    items={[
                      { color: OBSERVED, label: "scored risk" },
                      { color: FORECAST, label: "forecast peak", dashed: true },
                    ]}
                  />
                }
              />
              <Plot height={200}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={series} margin={{ top: 14, right: 12, left: 0, bottom: 0 }}>
                    <Grid />
                    <TimeAxis />
                    <ValueAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                    <Tooltip content={<Tip fmt={(v) => Number(v).toFixed(3)} />} cursor={{ stroke: "var(--rule-hard)" }} />
                    <ThresholdLine y={THRESHOLD} />
                    <Area type="monotone" dataKey="risk" name="scored risk" stroke="none" fill="var(--fill-observed)" isAnimationActive={false} />
                    <Line type="monotone" dataKey="risk" name="scored risk" stroke={OBSERVED} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name="forecast peak"
                      stroke={FORECAST}
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                      dot={false}
                      isAnimationActive={false}
                      connectNulls
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </Plot>
            </Panel>

            <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 220 }}>
              <PanelHead title="Scored windows" note={`${report.flagged_windows} flagged`} />
              <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
                <table className="tbl">
                  <thead>
                    <tr>
                      <th style={{ width: 70 }}>Window</th>
                      <th style={{ width: 130 }}>Target</th>
                      <th style={{ width: 130 }}>Stage</th>
                      <th>Technique</th>
                      <th className="num" style={{ width: 80 }}>Risk</th>
                      <th style={{ width: 120 }} />
                      <th style={{ width: 80 }}>Alert</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => {
                      const s = sevFromRisk(r.risk, THRESHOLD);
                      return (
                        <tr
                          key={r.window}
                          className={r.window === sel ? "is-selected" : undefined}
                          onClick={() => setSel(r.window === sel ? null : r.window)}
                          style={{ cursor: "pointer" }}
                        >
                          <td className="key" style={{ borderLeft: `3px solid ${sevColor(s)}` }}>
                            {String(r.window).padStart(3, "0")}
                          </td>
                          <td>{r.target}</td>
                          <td>{r.stage}</td>
                          <td className="key">{r.mitre_technique ?? "—"}</td>
                          <td className="num key" style={{ color: sevColor(s) }}>
                            {r.risk.toFixed(3)}
                          </td>
                          <td>
                            <Meter value={r.risk} level={s} height={4} />
                          </td>
                          <td style={{ color: r.alert ? "var(--sev-critical)" : "var(--paper-600)" }}>
                            {r.alert ? "RAISED" : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        ) : (
          !busy && (
            <Panel flush style={{ flex: "1 1 0" }}>
              <Empty hint="CIC-IDS2018 and CTU-13 both ship flow CSVs that drop straight in, as does any tcpdump or Wireshark capture.">
                No capture analysed
              </Empty>
            </Panel>
          )
        )}
      </div>

      {/* ── Rail ─────────────────────────────────────────────────────── */}
      <div className="rp-rail">
        <PanelHead title={selected ? `Window ${String(selected.window).padStart(3, "0")}` : "Report"} />

        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {selected ? (
            <>
              <PanelBody>
                <Readout
                  label="Scored risk"
                  value={selected.risk.toFixed(3)}
                  scale="hero"
                  level={sevFromRisk(selected.risk, THRESHOLD)}
                  sub={selected.alert ? "above threshold" : "below threshold"}
                />
              </PanelBody>

              <PanelHead title="Classification" />
              <PanelBody>
                <Field label="Target" value={selected.target} />
                <Field label="Stage" value={selected.stage} />
                <Field label="Technique" value={selected.mitre_technique ?? "—"} />
                <Field label="Tactic" value={selected.mitre_tactic ?? "—"} />
                <Field label="ML risk" value={selected.ml_risk != null ? selected.ml_risk.toFixed(3) : "—"} />
                <Field label="Alert" value={selected.alert ? "RAISED" : "clear"} color={selected.alert ? "var(--sev-critical)" : undefined} />
              </PanelBody>

              {selected.top_features.length > 0 && (
                <>
                  <PanelHead title="Attribution" />
                  <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)" }}>
                    {selected.top_features.map((f) => (
                      <BarRow key={f.feature} label={f.feature} value={f.score} labelWidth={110} />
                    ))}
                  </PanelBody>
                </>
              )}

              {selected.forecast.length > 0 && (
                <>
                  <PanelHead title="Rollout" />
                  <PanelBody>
                    {selected.forecast.map((f) => (
                      <Field
                        key={f.horizon_seconds}
                        label={`+${f.horizon_seconds}s`}
                        value={f.risk.toFixed(3)}
                        color={sevColor(sevFromRisk(f.risk, THRESHOLD))}
                      />
                    ))}
                  </PanelBody>
                </>
              )}
            </>
          ) : report ? (
            <>
              <PanelBody>
                <Readout
                  label="Peak risk in capture"
                  value={peak.toFixed(3)}
                  scale="hero"
                  level={sevFromRisk(peak, THRESHOLD)}
                  sub={`${report.flagged_windows} of ${report.windows_analyzed} windows flagged`}
                />
              </PanelBody>

              <PanelHead title="File" />
              <PanelBody>
                <Field label="Name" value={report.filename} />
                <Field label="Kind" value={report.kind.toUpperCase()} />
                <Field label="Size" value={pending ? fmtBytes(pending.size) : "—"} />
                <Field label="Windows" value={String(report.windows_analyzed)} />
                <Field label="Flagged" value={String(report.flagged_windows)} color={report.flagged_windows ? "var(--sev-critical)" : undefined} />
                <Field
                  label="Flag rate"
                  value={
                    report.windows_analyzed
                      ? `${((report.flagged_windows / report.windows_analyzed) * 100).toFixed(1)}%`
                      : "—"
                  }
                />
              </PanelBody>

              <PanelHead title="Evaluation mode" />
              <PanelBody>
                <Field label="Rule layer" value="disabled" color="var(--sev-nominal)" />
                <Field label="Network egress" value="none" color="var(--sev-nominal)" />
                <Field label="Threshold" value={THRESHOLD.toFixed(2)} />
                <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: "var(--s-3)" }}>
                  The deterministic SOC rules are switched off for offline analysis. A heuristic that floors
                  risk at 0.40 on any external flow would contaminate an evaluation, so these scores are the
                  model alone.
                </div>
              </PanelBody>

              <PanelBody>
                <Btn full onClick={() => inputRef.current?.click()}>
                  Analyse another file
                </Btn>
              </PanelBody>
            </>
          ) : (
            <PanelBody>
              <Micro style={{ marginBottom: "var(--s-3)" }}>Accepted input</Micro>
              <Field label="Packet capture" value=".pcap · .pcapng" />
              <Field label="Flow records" value=".csv · .binetflow" />
              <div className="t-body" style={{ color: "var(--paper-400)", marginTop: "var(--s-4)" }}>
                The capture is written to a temp file, parsed into 2-second flow windows, scored through the
                same TGNE → Branch A/B → DeepOP path as the live feed, and deleted. No cloud dependency, no
                egress.
              </div>
            </PanelBody>
          )}
        </div>
      </div>
    </div>
  );
}
