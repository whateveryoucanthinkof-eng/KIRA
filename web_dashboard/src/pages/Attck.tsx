import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ForecastPoint, PredictionResult, Topology } from "../api/types";
import type { Campaign } from "../types/campaign";
import type { FlowRecord } from "../types/evidence";
import type { ForecastBranch } from "../types/forecast";
import type { PredictionEnvelope } from "../types/live";
import { Chip, Micro, Panel, PanelHead, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Legend, OBSERVED } from "../design/charts";
import { clockTime } from "../design/time";
import { HORIZON_SECONDS, WINDOW_SECONDS } from "../types/timeline";

/**
 * ATT&CK coverage matrix.
 *
 * Tactic columns, technique cells. Every entry is a technique the deployed
 * heads can actually emit — the union of TECHNIQUE_VOCAB in
 * `branch_a_gnn_lstm/sequence_dataset.py` (14 classes) and the DeepOP token
 * set, annotated from `control_backend/model_adapter.py:TECHNIQUE_TO_MITRE`.
 * Nothing here is aspirational ATT&CK surface: if a cell exists, a head can
 * predict it.
 *
 * Cell state is live. Observed cells fill warm from OBSERVED campaign nodes;
 * the technique in progress carries the verdict's severity and a ticking
 * frame; forecast cells hatch cool from the rollout and its branches. Each
 * card carries its activation probability across the +16s horizon, the
 * vector path threads the observed techniques in the order they happened,
 * and a card opens an inspector with the flows that are its evidence.
 */

interface Tech {
  id: string;
  name: string;
  /** Which head can emit it. */
  heads: string;
}

const MATRIX: { tactic: string; id: string; techniques: Tech[] }[] = [
  {
    tactic: "Reconnaissance",
    id: "TA0043",
    techniques: [
      { id: "T1595", name: "Active Scanning", heads: "Branch A · DeepOP" },
      { id: "T1046", name: "Network Service Discovery", heads: "Branch A" },
    ],
  },
  {
    tactic: "Initial Access",
    id: "TA0001",
    techniques: [
      { id: "T1190", name: "Exploit Public-Facing Application", heads: "Branch A · DeepOP" },
      { id: "T1189", name: "Drive-by Compromise", heads: "Branch A" },
    ],
  },
  {
    tactic: "Execution",
    id: "TA0002",
    techniques: [{ id: "T1204", name: "User Execution", heads: "Branch A" }],
  },
  {
    tactic: "Credential Access",
    id: "TA0006",
    techniques: [{ id: "T1110", name: "Brute Force", heads: "Branch A · DeepOP" }],
  },
  {
    tactic: "Lateral Movement",
    id: "TA0008",
    techniques: [{ id: "T1021", name: "Remote Services", heads: "correlation" }],
  },
  {
    tactic: "Command and Control",
    id: "TA0011",
    techniques: [
      { id: "T1071", name: "Application Layer Protocol", heads: "Branch A · DeepOP" },
      { id: "T1071.001", name: "Web Protocols", heads: "Branch A" },
      { id: "T1568.001", name: "Fast Flux DNS", heads: "Branch A" },
    ],
  },
  {
    tactic: "Collection",
    id: "TA0009",
    techniques: [{ id: "T1005", name: "Data from Local System", heads: "Branch A · DeepOP" }],
  },
  {
    tactic: "Exfiltration",
    id: "TA0010",
    techniques: [{ id: "T1020", name: "Automated Exfiltration", heads: "Branch A" }],
  },
  {
    tactic: "Impact",
    id: "TA0040",
    techniques: [
      { id: "T1498", name: "Network Denial of Service", heads: "Branch A · DeepOP" },
      { id: "T1498.001", name: "Direct Network Flood", heads: "Branch A" },
    ],
  },
];

const ALL = MATRIX.flatMap((c) => c.techniques.map((t) => ({ ...t, tactic: c.tactic, tacticId: c.id })));
const VOCAB = new Set(ALL.map((t) => t.id));

/** Stage to the techniques a rollout step implies, for forecast shading. */
const STAGE_TECHNIQUES: Record<string, string[]> = {
  Recon: ["T1595", "T1046"],
  InitialAccess: ["T1190", "T1189", "T1110"],
  Execution: ["T1204"],
  LateralMovement: ["T1021"],
  C2: ["T1071", "T1071.001", "T1568.001"],
  Exfiltration: ["T1005", "T1020"],
  Impact: ["T1498", "T1498.001"],
};

type CellState = "active" | "observed" | "forecast" | "idle";

interface AttckProps {
  prediction: PredictionResult | null;
  forecast: ForecastPoint[];
  campaign: Campaign | null;
  branches: ForecastBranch[] | null;
  flows: FlowRecord[];
  envelope: PredictionEnvelope | null;
  topology: Topology | null;
}

/** First token of "T1498 Network Denial of Service" is the technique id. */
function techId(label: string | null | undefined): string | null {
  if (!label) return null;
  const first = String(label).split(/\s+/)[0];
  return /^T\d/.test(first) ? first : null;
}

/** A branch's technique as a vocabulary cell: exact id, else its stage's lead technique. */
function branchCell(b: ForecastBranch): string | null {
  if (VOCAB.has(b.technique)) return b.technique;
  return STAGE_TECHNIQUES[b.stage]?.[0] ?? null;
}

/** Rises through 0.5 a second before the expected milestone. */
function rise(t: number, at: number): number {
  return 1 / (1 + Math.exp(-(t - at + 1) / 1.2));
}

const SAMPLES = Array.from({ length: HORIZON_SECONDS / WINDOW_SECONDS + 1 }, (_, i) => i * WINDOW_SECONDS);

function fmtBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

function Spark({ values, colour }: { values: number[]; colour: string }) {
  const peak = values.length ? Math.max(...values) : 0;
  if (peak < 0.03) return <span className="at-spark is-flat" />;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${(16 - v * 14).toFixed(2)}`);
  return (
    <svg className="at-spark" viewBox="0 0 100 16" preserveAspectRatio="none" aria-hidden>
      <polygon points={`0,16 ${pts.join(" ")} 100,16`} fill={colour} fillOpacity={0.16} />
      <polyline points={pts.join(" ")} fill="none" stroke={colour} strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export default function Attck({ prediction, forecast, campaign, branches, flows, envelope, topology }: AttckProps) {
  const [pinned, setPinned] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [showPath, setShowPath] = useState(true);

  const threshold = Number(prediction?.threshold ?? 0.65);
  const now = techId(prediction?.mitre_technique);
  const nowRisk = Number(prediction?.risk ?? 0);

  /* ── Cell state and risk ─────────────────────────────────────────── */
  const { states, riskOf, counts } = useMemo(() => {
    const states = new Map<string, CellState>();
    const riskOf = new Map<string, number>();

    for (const node of campaign?.nodes ?? []) {
      if (node.provenance !== "OBSERVED") continue;
      states.set(node.technique_id, "observed");
      riskOf.set(node.technique_id, Math.max(riskOf.get(node.technique_id) ?? 0, node.max_risk_score));
    }
    if (now) {
      states.set(now, "active");
      riskOf.set(now, Math.max(riskOf.get(now) ?? 0, nowRisk));
    }
    const project = (id: string, r: number) => {
      const st = states.get(id);
      if (st === "observed" || st === "active") return;
      states.set(id, "forecast");
      riskOf.set(id, Math.max(riskOf.get(id) ?? 0, r));
    };
    for (const node of campaign?.nodes ?? []) if (node.provenance === "FORECAST") project(node.technique_id, node.max_risk_score);
    for (const f of forecast) for (const t of STAGE_TECHNIQUES[f.predictedStage ?? ""] ?? []) project(t, f.predicted / 100);
    for (const b of branches ?? []) {
      const c = b.kind === "backoff" ? null : branchCell(b);
      if (c) project(c, b.peak_risk);
    }

    let obs = 0;
    let fc = 0;
    for (const v of states.values()) {
      if (v === "forecast") fc += 1;
      else obs += 1;
    }
    return { states, riskOf, counts: { observed: obs, forecast: fc, total: ALL.length } };
  }, [forecast, campaign, branches, now, nowRisk]);

  /* ── Activation over the horizon, per cell ───────────────────────── */
  const activation = useMemo(() => {
    const m = new Map<string, number[]>();
    const backoff = (branches ?? []).find((b) => b.kind === "backoff");
    for (const t of ALL) {
      const series = SAMPLES.map((s) => {
        if (t.id === now) {
          const base = Number(prediction?.technique_confidence ?? 0.6);
          return base * (1 - (backoff ? backoff.probability * rise(s, backoff.horizon_seconds) : 0));
        }
        let p = 0;
        for (const b of branches ?? []) {
          if (b.kind !== "backoff" && branchCell(b) === t.id) p += b.probability * rise(s, b.horizon_seconds);
        }
        for (const f of forecast) {
          if ((STAGE_TECHNIQUES[f.predictedStage ?? ""] ?? []).includes(t.id)) p = Math.max(p, f.confidence * 0.55 * rise(s, f.horizonSeconds));
        }
        return Math.min(1, p);
      });
      m.set(t.id, series);
    }
    return m;
  }, [branches, forecast, now, prediction]);

  /* ── The vector path: observed techniques in the order they happened ─ */
  const path = useMemo(() => {
    const seen = new Map<string, number>();
    for (const n of campaign?.nodes ?? []) {
      if (n.provenance !== "OBSERVED" || !VOCAB.has(n.technique_id)) continue;
      seen.set(n.technique_id, Math.min(seen.get(n.technique_id) ?? Infinity, n.start_time));
    }
    const observed = [...seen.entries()].sort((a, b) => a[1] - b[1]).map(([id]) => id);
    if (now && VOCAB.has(now) && !observed.includes(now)) observed.push(now);
    const ahead = (branches ?? [])
      .filter((b) => b.kind !== "backoff")
      .sort((a, b) => a.horizon_seconds - b.horizon_seconds)
      .map(branchCell)
      .filter((id): id is string => id != null && !observed.includes(id))
      .slice(0, 2);
    return { observed, ahead };
  }, [campaign, branches, now]);

  /* ── Geometry for the overlay and the inspector ─────────────────── */
  const canvas = useRef<HTMLDivElement>(null);
  const cells = useRef(new Map<string, HTMLButtonElement>());
  const [boxes, setBoxes] = useState<Map<string, DOMRect>>(new Map());
  const [size, setSize] = useState({ w: 0, h: 0 });

  const measure = () => {
    const root = canvas.current;
    if (!root) return;
    const r = root.getBoundingClientRect();
    const next = new Map<string, DOMRect>();
    for (const [id, el] of cells.current) {
      const b = el.getBoundingClientRect();
      next.set(id, new DOMRect(b.left - r.left, b.top - r.top, b.width, b.height));
    }
    setBoxes(next);
    setSize({ w: root.scrollWidth, h: root.scrollHeight });
  };

  const pathKey = `${path.observed.join(">")}|${path.ahead.join(">")}`;
  useLayoutEffect(measure, [pathKey, showPath]);
  useEffect(() => {
    const root = canvas.current;
    if (!root || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    return () => ro.disconnect();
  }, []);

  // Esc closes a pinned inspector.
  useEffect(() => {
    if (!pinned) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPinned(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pinned]);

  const centre = (id: string) => {
    const b = boxes.get(id);
    return b ? { x: b.x + b.width / 2, y: b.y + b.height / 2 } : null;
  };
  const curve = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = b.x - a.x;
    const lift = Math.min(70, 18 + Math.abs(dx) * 0.12);
    return `M ${a.x} ${a.y} C ${a.x + dx * 0.35} ${a.y - lift}, ${b.x - dx * 0.35} ${b.y - lift}, ${b.x} ${b.y}`;
  };

  /* ── Inspector content ───────────────────────────────────────────── */
  const open = pinned ?? hover;
  const tech = open ? (ALL.find((t) => t.id === open) ?? null) : null;
  const openState: CellState = open ? (states.get(open) ?? "idle") : "idle";
  const nameOf = (ip: string) => topology?.nodes.find((n) => n.ip === ip)?.label ?? (ip.startsWith("10.") ? ip : "external");

  const inspector = useMemo(() => {
    if (!tech) return null;
    const nodes = (campaign?.nodes ?? []).filter((n) => n.technique_id === tech.id);
    const branch = (branches ?? []).find((b) => b.kind !== "backoff" && branchCell(b) === tech.id) ?? null;
    const confidence =
      tech.id === now
        ? Number(prediction?.technique_confidence ?? 0)
        : nodes.length
          ? Math.max(...nodes.map((n) => n.mean_confidence))
          : branch
            ? branch.confidence
            : null;

    const hosts = new Set<string>(nodes.map((n) => n.host_ip));
    if (tech.id === now) for (const ip of envelope?.focus_ips ?? []) hosts.add(ip);
    if (branch) for (const hp of branch.hops) hosts.add(hp.ip);

    const latestWindow = envelope?.state?.window_id ?? null;
    const evidence =
      tech.id === now
        ? flows
            .filter((f) => f.on_path && f.window === latestWindow)
            .sort((a, b) => b.fwd_packets + b.bwd_packets - (a.fwd_packets + a.bwd_packets))
        : [];
    const packets = evidence.reduce((s, f) => s + f.fwd_packets + f.bwd_packets, 0);
    const first = nodes.length ? nodes.reduce((a, n) => (n.start_time < a.start_time ? n : a), nodes[0]) : null;
    const since = first && campaign ? Math.round(first.start_time - campaign.end_time) : null;

    return { nodes, branch, confidence, hosts: [...hosts], evidence, packets, latestWindow, first, since };
  }, [tech, campaign, branches, now, prediction, envelope, flows]);

  // The inspector opens in the free space under the matrix, beneath its card,
  // so it never covers another technique; a rule ties it to the card.
  const box = open ? boxes.get(open) : undefined;
  const gridBottom = Math.max(0, ...[...boxes.values()].map((b) => b.y + b.height));
  const cardW = Math.min(780, Math.max(300, size.w - 8));
  const wideCard = cardW >= 620;
  const place = box
    ? {
        left: Math.max(4, Math.min(box.x + box.width / 2 - cardW / 2, size.w - cardW - 4)),
        top: gridBottom + 18,
        anchorX: box.x + box.width / 2,
        anchorTop: box.y + box.height,
      }
    : null;

  return (
    <div className="at">
      <div className="at-main sheet">
        <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <PanelHead
            title="ATT&CK coverage"
            note={`${counts.observed} observed · ${counts.forecast} forecast · ${counts.total} in model vocabulary`}
            aside={
              <>
                <Legend
                  items={[
                    { color: OBSERVED, label: "observed" },
                    { color: sevColor(sevFromRisk(nowRisk, threshold)), label: "in progress" },
                    { color: FORECAST, label: "forecast", dashed: true },
                  ]}
                />
                <button className="seg" aria-pressed={showPath} onClick={() => setShowPath((v) => !v)} title="Thread the observed techniques in order">
                  Vector path
                </button>
              </>
            }
          />

          <div className="at-scroll" onClick={() => setPinned(null)}>
            <div className="at-canvas" ref={canvas}>
              <div className="at-grid">
                {MATRIX.map((col) => (
                  <div key={col.id} className="at-col">
                    <div className="at-tactic">
                      <span title={col.tactic}>{col.tactic}</span>
                      <em>{col.id}</em>
                    </div>

                    <div className="at-cells">
                      {col.techniques.map((t) => {
                        const state = states.get(t.id) ?? "idle";
                        const risk = riskOf.get(t.id);
                        const step = path.observed.indexOf(t.id);
                        const colour =
                          state === "active"
                            ? sevColor(sevFromRisk(nowRisk, threshold))
                            : state === "observed"
                              ? OBSERVED
                              : state === "forecast"
                                ? FORECAST
                                : "var(--rule-hard)";
                        const cls = ["at-cell", `is-${state}`, t.id === pinned && "is-pinned", t.id === hover && "is-hover"].filter(Boolean).join(" ");
                        return (
                          <button
                            key={t.id}
                            ref={(el) => {
                              if (el) cells.current.set(t.id, el);
                              else cells.current.delete(t.id);
                            }}
                            className={cls}
                            style={{ borderLeftColor: colour, ...(state === "active" ? { borderColor: colour } : {}) }}
                            onMouseEnter={() => setHover(t.id)}
                            onMouseLeave={() => setHover((h) => (h === t.id ? null : h))}
                            onClick={(e) => {
                              e.stopPropagation();
                              setPinned((p) => (p === t.id ? null : t.id));
                            }}
                            aria-label={`${t.id} ${t.name}, ${state}`}
                          >
                            {showPath && step >= 0 && <span className="at-step">{step + 1}</span>}
                            <span className="at-cell-head">
                              <b style={{ color: state === "idle" ? "var(--paper-600)" : state === "forecast" ? FORECAST : "var(--paper-000)" }}>{t.id}</b>
                              {state === "active" && <em style={{ color: colour }}>now</em>}
                              {risk != null && state === "observed" && (
                                <i style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>{risk.toFixed(2)}</i>
                              )}
                            </span>
                            <span className="at-cell-name">{t.name}</span>
                            <Spark values={activation.get(t.id) ?? []} colour={state === "active" ? colour : FORECAST} />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {/* ── Vector path ─────────────────────────────────────────── */}
              {showPath && size.w > 0 && (
                <svg className="at-path" width={size.w} height={size.h} aria-hidden>
                  {path.observed.slice(1).map((id, i) => {
                    const a = centre(path.observed[i]);
                    const b = centre(id);
                    if (!a || !b) return null;
                    const d = curve(a, b);
                    return (
                      <g key={`o${i}`}>
                        <path d={d} fill="none" stroke="var(--sev-critical)" strokeWidth={2.4} strokeOpacity={0.8} />
                        <path d={d} fill="none" stroke="var(--paper-000)" strokeWidth={1.4} strokeDasharray="6 26" className="is-traveling" strokeOpacity={0.8} />
                      </g>
                    );
                  })}
                  {path.ahead.map((id, i) => {
                    const from = i === 0 ? path.observed[path.observed.length - 1] : path.ahead[i - 1];
                    const a = from ? centre(from) : null;
                    const b = centre(id);
                    if (!a || !b) return null;
                    return <path key={`f${i}`} d={curve(a, b)} fill="none" stroke={FORECAST} strokeWidth={1.8} strokeDasharray="5 4" />;
                  })}
                </svg>
              )}

              {/* ── Inspector ──────────────────────────────────────────── */}
              {tech && inspector && place && (
                <span
                  className="at-tether"
                  style={{ left: place.anchorX, top: place.anchorTop + 4, height: Math.max(0, place.top - place.anchorTop - 4) }}
                />
              )}
              {tech && inspector && place && (
                <div
                  className={pinned ? "at-inspect is-pinned" : "at-inspect"}
                  style={{ left: place.left, top: place.top, width: cardW }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className={wideCard ? "at-inspect-cols" : "at-inspect-cols is-stacked"}>
                  <div>
                  <div className="at-inspect-head">
                    <div>
                      <b>{tech.id}</b>
                      <span>{tech.name}</span>
                    </div>
                    {inspector.confidence != null && (
                      <em>
                        {(inspector.confidence * 100).toFixed(1)}%<small>confidence</small>
                      </em>
                    )}
                  </div>
                  <div className="at-inspect-sub">
                    <span>
                      {tech.tactic} · {tech.tacticId}
                    </span>
                    <Chip level={openState === "active" ? sevFromRisk(nowRisk, threshold) : openState === "observed" ? "nominal" : "unknown"}>
                      {openState === "active" ? "in progress" : openState === "idle" ? "not seen" : openState}
                    </Chip>
                  </div>

                  <Micro style={{ margin: "var(--s-2) 0 4px" }}>Affected assets</Micro>
                  <div className="at-hosts">
                    {inspector.hosts.length ? (
                      inspector.hosts.map((ip) => (
                        <span key={ip} className={ip.startsWith("10.") ? "at-host" : "at-host is-external"}>
                          {nameOf(ip)} <em>{ip}</em>
                        </span>
                      ))
                    ) : (
                      <span className="at-none">none attributed</span>
                    )}
                  </div>

                  </div>
                  <div>
                  <Micro style={{ margin: wideCard ? "0 0 4px" : "var(--s-2) 0 4px" }}>Telemetry evidence</Micro>
                  {openState === "active" && inspector.evidence.length ? (
                    <>
                      <pre className="at-evidence">
                        {inspector.evidence
                          .slice(0, 4)
                          .map((f) => {
                            const t = new Date(f.ts_us / 1000);
                            const ms = String(t.getMilliseconds()).padStart(3, "0");
                            const fl = f.flags;
                            return `${clockTime(t)}.${ms}  ${f.src_ip}:${f.src_port} → ${f.dst_ip}:${f.dst_port} ${f.protocol}\n  fwd ${f.fwd_packets} pkt / ${fmtBytes(f.fwd_bytes)}  bwd ${f.bwd_packets} pkt / ${fmtBytes(f.bwd_bytes)}  ${(f.duration_ms / 1000).toFixed(1)}s\n  SYN ${fl.syn} ACK ${fl.ack} PSH ${fl.psh} RST ${fl.rst} FIN ${fl.fin}\n`;
                          })
                          .join("")}
                      </pre>
                      <div className="at-evidence-foot">
                        {inspector.evidence.length} on-path flows in window #{inspector.latestWindow} ·{" "}
                        {Math.round(inspector.packets / WINDOW_SECONDS).toLocaleString()} pkt/s · flow_table.py
                      </div>
                    </>
                  ) : openState === "observed" && inspector.first ? (
                    <div className="at-note">
                      {inspector.first.hit_count.toLocaleString()} hits on {nameOf(inspector.first.host_ip)} · first seen t {inspector.since}s · peak risk{" "}
                      {inspector.first.max_risk_score.toFixed(2)}. Its flows have left the live window — scrub the Stage back to t {inspector.since}s to replay
                      them.
                    </div>
                  ) : openState === "forecast" ? (
                    <div className="at-note">
                      Not yet observed.{" "}
                      {inspector.branch
                        ? `Branch ${inspector.branch.id} puts it at +${inspector.branch.horizon_seconds}s with ${(inspector.branch.probability * 100).toFixed(0)}% of the forecast mass — ${inspector.branch.packets.toLocaleString()} packets, ${fmtBytes(inspector.branch.bytes)} if it plays out.`
                        : "The rollout reaches this stage inside the horizon."}
                    </div>
                  ) : (
                    <div className="at-note">No flows attributed in this window.</div>
                  )}
                  </div>
                  </div>

                  <div className="at-inspect-foot">
                    <span>emitted by {tech.heads}</span>
                    <span>{pinned ? "pinned · esc" : "click to pin"}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="at-caption">
            Only techniques the deployed heads can emit are listed — the 14-class Branch A vocabulary and the DeepOP token set, not ATT&CK as a
            whole. Sparklines: activation probability across the +16s horizon.
          </div>
        </Panel>
      </div>
    </div>
  );
}
