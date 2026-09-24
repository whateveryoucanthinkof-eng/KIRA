import { useMemo, useState } from "react";
import type { Topology, TopologyEdge, TopologyNode } from "../api/types";
import type { PredictionEnvelope } from "../types/live";
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
  Square,
  sevColor,
} from "../design/primitives";
import { Legend } from "../design/charts";

interface NetworkProps {
  topology: Topology | null;
  envelope: PredictionEnvelope | null;
}

const ZONES = [
  { key: "external", label: "External", top: 8, bottom: 128 },
  { key: "perimeter", label: "Perimeter", top: 128, bottom: 248 },
  { key: "enterprise", label: "Enterprise", top: 248, bottom: 452 },
];

function edgeStroke(e: TopologyEdge): string {
  if (e.status === "saturated") return "var(--sev-critical)";
  if (e.status === "suspicious") return "var(--sev-warning)";
  if (e.status === "down") return "var(--rule-hair)";
  return "var(--rule-hard)";
}

export default function Network({ topology, envelope }: NetworkProps) {
  const [selNode, setSelNode] = useState<string | null>(null);
  const [selEdge, setSelEdge] = useState<string | null>(null);

  const focus = useMemo(() => new Set(envelope?.focus_ips ?? []), [envelope]);

  const nodes = topology?.nodes ?? [];
  const edges = topology?.edges ?? [];

  const node = nodes.find((x) => x.id === selNode) ?? null;
  const edge = edges.find((x) => x.id === selEdge) ?? null;

  const counts = useMemo(() => {
    const c: Record<string, number> = { online: 0, warning: 0, degraded: 0, compromised: 0, offline: 0 };
    for (const x of nodes) c[x.status] = (c[x.status] ?? 0) + 1;
    return c;
  }, [nodes]);

  function pick(n: TopologyNode) {
    setSelNode(n.id === selNode ? null : n.id);
    setSelEdge(null);
  }

  return (
    <div className="nw">
      {/* ── Canvas ───────────────────────────────────────────────────── */}
      <div className="nw-canvas sheet" style={{ gap: 1 }}>
        {/* flex 1 1 0 — without an explicit grow the canvas sizes to content
            and collapses against the fixed-height inventory below it. */}
        <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <PanelHead
            title="Host graph"
            note={topology ? `${nodes.length} hosts · ${edges.length} edges` : undefined}
            aside={
              <Legend
                items={[
                  { color: "var(--sev-nominal)", label: "online" },
                  { color: "var(--sev-warning)", label: "at risk" },
                  { color: "var(--sev-critical)", label: "compromised" },
                  { color: "var(--paper-600)", label: "stale" },
                ]}
              />
            }
          />

          <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
            {nodes.length ? (
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 800 460"
                preserveAspectRatio="xMidYMid meet"
                style={{ display: "block" }}
                onClick={() => {
                  setSelNode(null);
                  setSelEdge(null);
                }}
              >
                {/* Zone bands — ruled, labelled, no fills */}
                {ZONES.map((z) => (
                  <g key={z.key}>
                    <line
                      x1={0}
                      y1={z.top}
                      x2={800}
                      y2={z.top}
                      stroke="var(--rule-hair)"
                      strokeWidth={1}
                      shapeRendering="crispEdges"
                    />
                    <text x={8} y={z.top + 13} fill="var(--paper-600)" fontSize={9} fontFamily="var(--face-ui)" letterSpacing="2.2">
                      {z.label.toUpperCase()}
                    </text>
                  </g>
                ))}

                {/* Edges */}
                {edges.map((e) => {
                  const s = nodes.find((x) => x.id === e.source);
                  const t = nodes.find((x) => x.id === e.target);
                  if (!s || !t) return null;
                  const on = e.id === selEdge;
                  const hot = e.status === "saturated";

                  // An attack path is a link the model is currently attending
                  // to at both ends (PredictionEvent.focus_ips).
                  const isPath =
                    focus.size > 0 && !!s.ip && !!t.ip && focus.has(s.ip) && focus.has(t.ip);

                  // The dash accelerates with link utilisation, so escalation
                  // is legible as speed before you read a single number.
                  const travelSec = Math.max(0.35, 2.4 - (e.utilization ?? 0) / 45);

                  return (
                    <g key={e.id}>
                      {/* Fat invisible hit target — 1px lines are unclickable */}
                      <line
                        x1={s.x}
                        y1={s.y}
                        x2={t.x}
                        y2={t.y}
                        stroke="transparent"
                        strokeWidth={10}
                        style={{ cursor: "pointer" }}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          setSelEdge(e.id === selEdge ? null : e.id);
                          setSelNode(null);
                        }}
                      />
                      <line
                        x1={s.x}
                        y1={s.y}
                        x2={t.x}
                        y2={t.y}
                        stroke={on ? "var(--paper-000)" : edgeStroke(e)}
                        strokeWidth={on ? 2 : hot ? 1.6 : 1}
                        strokeDasharray={e.status === "suspicious" ? "3 3" : undefined}
                        shapeRendering={hot || on ? undefined : "crispEdges"}
                        pointerEvents="none"
                      />
                      {isPath && (
                        <line
                          className="is-traveling"
                          x1={s.x}
                          y1={s.y}
                          x2={t.x}
                          y2={t.y}
                          stroke={sevColor(hot ? "critical" : "elevated")}
                          strokeWidth={2}
                          strokeDasharray="6 26"
                          strokeLinecap="butt"
                          pointerEvents="none"
                          style={{ animationDuration: `${travelSec}s` }}
                        />
                      )}
                    </g>
                  );
                })}

                {/* Nodes */}
                {nodes.map((x) => {
                  const on = x.id === selNode;
                  const c = sevColor(x.status);
                  const half = x.status === "compromised" ? 8 : 6;
                  const isFocus = x.ip ? focus.has(x.ip) : false;
                  const isTarget = !!x.ip && x.ip === envelope?.target_ip;
                  return (
                    <g
                      key={x.id}
                      style={{ cursor: "pointer" }}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        pick(x);
                      }}
                    >
                      {/* Focus bracket — hosts the model is currently attending to */}
                      {isFocus && (
                        <rect
                          x={x.x - half - 5}
                          y={x.y - half - 5}
                          width={(half + 5) * 2}
                          height={(half + 5) * 2}
                          fill="none"
                          stroke="var(--sev-warning)"
                          strokeWidth={1}
                          strokeDasharray="2 2"
                          shapeRendering="crispEdges"
                        />
                      )}

                      {/* Primary target — corner ticks plus a standing label */}
                      {isTarget && (
                        <g pointerEvents="none">
                          {[
                            [-1, -1],
                            [1, -1],
                            [-1, 1],
                            [1, 1],
                          ].map(([sx, sy]) => {
                            const d = half + 10;
                            return (
                              <path
                                key={`${sx}${sy}`}
                                d={`M ${x.x + sx * d} ${x.y + sy * d - sy * 6} L ${x.x + sx * d} ${x.y + sy * d} L ${x.x + sx * d - sx * 6} ${x.y + sy * d}`}
                                fill="none"
                                stroke={sevColor(x.status)}
                                strokeWidth={1.4}
                                shapeRendering="crispEdges"
                              />
                            );
                          })}
                          <text
                            x={x.x}
                            y={x.y - half - 16}
                            textAnchor="middle"
                            fill="var(--paper-600)"
                            fontSize={8}
                            fontFamily="var(--face-ui)"
                            letterSpacing="1.6"
                          >
                            TARGET
                          </text>
                        </g>
                      )}
                      {on && (
                        <rect
                          x={x.x - half - 4}
                          y={x.y - half - 4}
                          width={(half + 4) * 2}
                          height={(half + 4) * 2}
                          fill="none"
                          stroke="var(--paper-000)"
                          strokeWidth={1}
                          shapeRendering="crispEdges"
                        />
                      )}
                      <rect
                        x={x.x - half}
                        y={x.y - half}
                        width={half * 2}
                        height={half * 2}
                        fill={c}
                        shapeRendering="crispEdges"
                      />
                      <text
                        x={x.x}
                        y={x.y + half + 12}
                        textAnchor="middle"
                        fill={on ? "var(--paper-000)" : "var(--paper-400)"}
                        fontSize={9}
                        fontFamily="var(--face-data)"
                      >
                        {x.label}
                      </text>
                      {x.ip && x.ip !== x.label && (
                        <text
                          x={x.x}
                          y={x.y + half + 22}
                          textAnchor="middle"
                          fill="var(--paper-600)"
                          fontSize={8}
                          fontFamily="var(--face-data)"
                        >
                          {x.ip}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            ) : (
              <Empty hint="Nodes are discovered from observed SPAN traffic and expire after their TTL.">
                No hosts discovered
              </Empty>
            )}
          </div>
        </Panel>

        {/* ── Host table ───────────────────────────────────────────── */}
        <Panel flush clip style={{ flex: "0 0 200px", display: "flex", flexDirection: "column" }}>
          <PanelHead title="Host inventory" note={`${nodes.length}`} />
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 150 }}>Host</th>
                  <th style={{ width: 130 }}>Address</th>
                  <th style={{ width: 96 }}>Role</th>
                  <th style={{ width: 110 }}>Status</th>
                  <th className="num" style={{ width: 90 }}>Bytes in</th>
                  <th className="num" style={{ width: 90 }}>Bytes out</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {nodes.map((x) => {
                  const raw = x as TopologyNode & { bytes_in?: number; bytes_out?: number; role?: string; zone?: string };
                  return (
                    <tr
                      key={x.id}
                      className={x.id === selNode ? "is-selected" : undefined}
                      onClick={() => pick(x)}
                      style={{ cursor: "pointer" }}
                    >
                      <td className="key" style={{ borderLeft: `3px solid ${sevColor(x.status)}` }}>
                        {x.label}
                      </td>
                      <td>{x.ip ?? "—"}</td>
                      <td>{raw.role ?? x.type}</td>
                      <td style={{ color: sevColor(x.status) }}>{x.status}</td>
                      <td className="num">{raw.bytes_in != null ? raw.bytes_in.toLocaleString() : "—"}</td>
                      <td className="num">{raw.bytes_out != null ? raw.bytes_out.toLocaleString() : "—"}</td>
                      <td>{x.ip && focus.has(x.ip) ? <Chip level="warning">focus</Chip> : null}</td>
                    </tr>
                  );
                })}
                {!nodes.length && (
                  <tr>
                    <td colSpan={7} style={{ textAlign: "center", padding: "var(--s-6)", color: "var(--paper-600)" }}>
                      no hosts
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* ── Inspector rail ───────────────────────────────────────────── */}
      <div className="nw-rail" style={{ background: "var(--ink-050)", borderTop: "var(--hair)", borderRight: "var(--hair)", borderBottom: "var(--hair)" }}>
        <PanelHead title={edge ? "Link" : node ? "Host" : "Inspector"} />

        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {node && (
            <PanelBody>
              <Readout label="Host" value={node.label} level={node.status} sub={node.ip ?? undefined} />
              <div style={{ marginTop: "var(--s-4)" }}>
                <Field label="Address" value={node.ip ?? "—"} />
                <Field label="Type" value={node.type} />
                <Field label="Status" value={node.status} color={sevColor(node.status)} />
                {(() => {
                  const raw = node as TopologyNode & {
                    role?: string;
                    zone?: string;
                    risk?: number;
                    bytes_in?: number;
                    bytes_out?: number;
                    last_seen?: number;
                    stale?: boolean;
                  };
                  return (
                    <>
                      {raw.role && <Field label="Role" value={raw.role} />}
                      {raw.zone && <Field label="Zone" value={raw.zone} />}
                      {raw.risk != null && (
                        <Field label="Risk" value={Number(raw.risk).toFixed(3)} color={sevColor(node.status)} />
                      )}
                      {raw.bytes_in != null && <Field label="Bytes in" value={raw.bytes_in.toLocaleString()} />}
                      {raw.bytes_out != null && <Field label="Bytes out" value={raw.bytes_out.toLocaleString()} />}
                      {raw.stale != null && <Field label="Stale" value={raw.stale ? "yes" : "no"} />}
                    </>
                  );
                })()}
              </div>

              {node.ip && focus.has(node.ip) && (
                <div style={{ marginTop: "var(--s-4)" }}>
                  <Micro style={{ marginBottom: "var(--s-2)" }}>Model focus</Micro>
                  <Chip level="warning">Attended this window</Chip>
                </div>
              )}

              {node.services?.length ? (
                <div style={{ marginTop: "var(--s-4)" }}>
                  <Micro style={{ marginBottom: "var(--s-2)" }}>Services</Micro>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s-1)" }}>
                    {node.services.map((s) => (
                      <Chip key={s}>{s}</Chip>
                    ))}
                  </div>
                </div>
              ) : null}
            </PanelBody>
          )}

          {edge && (
            <PanelBody>
              <Readout label="Link" value={`${edge.source} → ${edge.target}`} level={edge.status} />
              <div style={{ marginTop: "var(--s-4)" }}>
                <Field label="Status" value={edge.status} color={sevColor(edge.status)} />
                <Field label="Protocol" value={edge.protocol ?? "—"} />
                <Field label="Rate" value={edge.bandwidth != null ? `${edge.bandwidth} Mb/s` : "—"} />
                <Field label="Utilisation" value={edge.utilization != null ? `${edge.utilization}%` : "—"} />
              </div>
            </PanelBody>
          )}

          {!node && !edge && (
            <>
              <PanelBody>
                <Micro style={{ marginBottom: "var(--s-3)" }}>Fleet</Micro>
                {Object.entries(counts).map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "6px 0",
                      borderBottom: "var(--hair)",
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: "var(--s-2)" }}>
                      <Square status={k} />
                      <span className="t-label" style={{ color: "var(--paper-400)" }}>
                        {k}
                      </span>
                    </span>
                    <Data size="s">{v}</Data>
                  </div>
                ))}
              </PanelBody>

              <PanelHead title="Discovery" />
              <PanelBody>
                <Field label="Nodes" value={String(nodes.length)} />
                <Field label="Edges" value={String(edges.length)} />
                <Field label="Updated" value={topology ? new Date(topology.lastUpdated).toLocaleTimeString("en-GB", { hour12: false }) : "—"} />
                {envelope?.target_ip && <Field label="Primary target" value={envelope.target_ip} />}
              </PanelBody>

              <PanelBody>
                <Empty hint="Select a host or a link on the graph to inspect it.">Nothing selected</Empty>
              </PanelBody>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
