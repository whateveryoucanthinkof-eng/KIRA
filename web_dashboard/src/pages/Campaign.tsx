import { useEffect, useMemo, useRef, useState } from "react";
import type { PredictionResult, Topology } from "../api/types";
import type { Campaign as CampaignData, CampaignNode } from "../types/campaign";
import type { ForecastBranch } from "../types/forecast";
import type { PredictionEnvelope } from "../types/live";
import type { Incident, IncidentEdit } from "../types/incident";
import { applyEdit } from "../types/incident";
import type { Toast } from "../components/Feedback";
import { Btn, Chip, Empty, Field, Meter, Micro, Panel, PanelHead, Sheet, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, Legend, OBSERVED } from "../design/charts";
import { Num } from "../design/motion";
import { clockTime } from "../design/time";
import { sendMitigate } from "../api/adapter";
import { isInternal, assetName } from "../design/site";
import { LANES, TECHNIQUE_NAMES } from "../design/lanes";

/**
 * Campaign dossier.
 *
 * correlation/campaign_merge.py assembles correlated alert trajectories into a
 * campaign: compacted nodes on kill-chain lanes, joined by scored causal
 * edges, with a root cause. This view is that object read as an intelligence
 * product — who (as far as flow telemetry can say), when it started, how fast
 * it is moving, which hosts it has reached, which incidents it explains —
 * and the one action that ends all of it at once.
 *
 * Attribution is deliberately not attempted: flow records carry no actor
 * indicators, so the actor is an internal cluster designation, the way
 * analysts track activity that has not been attributed.
 */

interface CampaignProps {
  campaign: CampaignData | null;
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  branches: ForecastBranch[] | null;
  incidents: Incident[];
  edits: Record<string, IncidentEdit>;
  onUpdateIncident: (id: string, patch: IncidentEdit) => void;
  onToast: (t: Toast) => void;
  topology: Topology | null;
}

/* Layout for the evolution graph — drawn at the size of its panel, 1:1. */
const LANE_LABEL_H = 28;
const GUTTER = 112;
const CARD_H = 42;

/** Short lane names, so every lane fits side by side on a small screen. */
const LANE_SHORT: Record<string, string> = Object.fromEntries(LANES.map((l) => [l.key, l.key === "CredentialAccess" ? "Credentials" : l.label]));


function hhmmss(epochSeconds: number): string {
  try {
    return clockTime(epochSeconds * 1000);
  } catch {
    return "—";
  }
}

type NodeStatus = "completed" | "active" | "forecast" | "contained";
type BottomTab = "assets" | "nodes" | "detail";

export default function Campaign({
  campaign,
  prediction,
  envelope,
  branches,
  incidents,
  edits,
  onUpdateIncident,
  onToast,
  topology,
}: CampaignProps) {
  const [sel, setSel] = useState<number | null>(null);
  const [hoverHost, setHoverHost] = useState<string | null>(null);
  const [tab, setTab] = useState<BottomTab>("assets");
  const [confirm, setConfirm] = useState(false);
  // Neutralisation: how many nodes the containment pulse has reached, and the
  // observed-node count when it ran — a campaign that restarts drops below it.
  const [pulse, setPulse] = useState<number | null>(null);
  const [neutralizedAt, setNeutralizedAt] = useState<number | null>(null);
  const timers = useRef<number[]>([]);
  useEffect(() => () => timers.current.forEach((t) => window.clearTimeout(t)), []);

  // The graph is laid out for the space it has, rather than scaled into it.
  const canvasRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setBox({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
    // Re-attach when the canvas first mounts (the campaign can arrive after the page).
  }, [campaign != null]);

  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);
  const nameOf = (ip: string) => topology?.nodes.find((n) => n.ip === ip)?.label ?? (isInternal(ip) ? (assetName(ip) ?? ip) : "external");

  const layout = useMemo(() => {
    if (!campaign) return null;
    const lanes = campaign.lanes;
    const hosts = [...new Set(campaign.nodes.map((n) => n.host_ip))];
    const width = Math.max(760, box.w);
    const rowH = Math.max(52, Math.min(110, (box.h - LANE_LABEL_H - 8) / Math.max(1, hosts.length)));
    const laneW = (width - GUTTER) / lanes.length;
    const pos = new Map<number, { x: number; y: number }>();
    for (const n of campaign.nodes) {
      const li = Math.max(0, lanes.indexOf(n.coarse_category));
      const hi = Math.max(0, hosts.indexOf(n.host_ip));
      pos.set(n.node_id, { x: GUTTER + laneW * li + laneW / 2, y: LANE_LABEL_H + rowH * hi + rowH / 2 });
    }
    return { lanes, hosts, laneW, rowH, width, pos, height: LANE_LABEL_H + rowH * hosts.length + 6 };
  }, [campaign, box.w, box.h]);

  const observedCount = campaign?.nodes.filter((n) => n.provenance === "OBSERVED").length ?? 0;
  // A campaign that has started over is a new breach: the old neutralisation no longer applies.
  useEffect(() => {
    if (neutralizedAt != null && observedCount < neutralizedAt) {
      setNeutralizedAt(null);
      setPulse(null);
    }
  }, [observedCount, neutralizedAt]);

  if (!campaign || !layout) {
    return (
      <div className="cm">
        <Panel clip style={{ height: "100%" }}>
          <PanelHead title="Campaign" />
          <Empty hint="Campaigns are assembled from correlated alert trajectories once inference is running.">No campaign reconstructed</Empty>
        </Panel>
      </div>
    );
  }

  const { lanes, hosts, laneW, rowH: ROW_H, width: VB_W, pos, height } = layout;
  const byId = new Map(campaign.nodes.map((n) => [n.node_id, n]));
  const roots = new Set(campaign.root_cause_node_ids);
  const selected: CampaignNode | null = sel != null ? (byId.get(sel) ?? null) : null;
  const observed = campaign.nodes.filter((n) => n.provenance === "OBSERVED");
  const forecastNodes = campaign.nodes.filter((n) => n.provenance === "FORECAST");

  /* ── Who, when, how fast ─────────────────────────────────────────── */
  // A reference, not a codename: nothing in the pipeline attributes a
  // campaign to an actor, so it is not given an actor-style name.
  const codename = `campaign ${String(campaign.campaign_id).padStart(4, "0")}`;
  const attacker = (envelope?.focus_ips ?? []).find((ip) => !isInternal(ip)) ?? (branches ?? []).flatMap((b) => b.hops ?? []).find((h) => !isInternal(h.ip))?.ip ?? null;
  const firstSeen = observed.length ? Math.min(...observed.map((n) => n.start_time)) : campaign.start_time;
  const since = Math.round(firstSeen - campaign.end_time);
  const minutes = Math.max(1 / 60, (campaign.end_time - firstSeen) / 60);
  const reachedHosts = [...new Set(observed.map((n) => n.host_ip))];
  const stageVelocity = observed.length / minutes;
  const hostVelocity = (reachedHosts.length + (attacker ? 1 : 0) - 1) / minutes;
  const ttp = [...observed].sort((a, b) => a.start_time - b.start_time).map((n) => n.technique_id);

  const merged = incidents.map((i) => applyEdit(i, edits[i.id]));
  const bound = merged.filter((i) => i.campaignId === campaign.campaign_id || (campaign.involved_hosts.includes(i.host) && i.status !== "closed"));
  const boundOn = (ip: string) => bound.filter((i) => i.host === ip);

  const benign = prediction?.predicted_stage === "Benign";
  const liveContained = bound.some((i) => i.campaignId === campaign.campaign_id && (i.status === "contained" || i.status === "closed"));
  const neutralized = neutralizedAt != null;
  const contained = neutralized || liveContained || (benign && observed.length > 0);
  const status = contained
    ? { label: "Contained", level: "nominal" }
    : observed.length >= 3
      ? { label: "Active multi-stage breach", level: "critical" }
      : observed.length
        ? { label: "Active", level: "elevated" }
        : { label: "Forming", level: "warning" };

  /* ── Per-node status ─────────────────────────────────────────────── */
  const nowTech = prediction?.mitre_technique ? prediction.mitre_technique.split(" ")[0] : null;
  const order = [...campaign.nodes].sort((a, b) => lanes.indexOf(a.coarse_category) - lanes.indexOf(b.coarse_category) || a.node_id - b.node_id);
  const pulseRank = new Map(order.map((n, i) => [n.node_id, i]));
  const probabilityOf = (n: CampaignNode): number => {
    const b = (branches ?? []).find((x) => x.kind !== "backoff" && (x.technique === n.technique_id || (x.stage === n.coarse_category && (x.hops ?? []).some((h) => h.ip === n.host_ip))));
    return b ? b.probability : n.mean_confidence;
  };
  const statusOf = (n: CampaignNode): NodeStatus => {
    if (pulse != null && (pulseRank.get(n.node_id) ?? 0) < pulse) return "contained";
    if (contained && pulse == null && n.provenance === "OBSERVED") return "contained";
    if (n.provenance === "FORECAST") return "forecast";
    return n.technique_id === nowTech && !benign ? "active" : "completed";
  };

  /* ── Assets: every host the campaign has reached or puts at risk ─── */
  const assets = (() => {
    const out: { ip: string; name: string; state: "compromised" | "reached" | "at-risk" | "contained"; value: number; label: string; nodes: CampaignNode[] }[] = [];
    for (const ip of campaign.involved_hosts) {
      if (!isInternal(ip)) continue;
      const nodes = campaign.nodes.filter((n) => n.host_ip === ip);
      const seen = nodes.filter((n) => n.provenance === "OBSERVED");
      if (seen.length) {
        const peak = Math.max(...seen.map((n) => n.max_risk_score));
        const isTarget = ip === envelope?.target_ip;
        out.push({
          ip,
          name: nameOf(ip),
          state: contained ? "contained" : isTarget ? "compromised" : "reached",
          value: peak,
          label: contained ? "contained" : `${Math.round(peak * 100)}% ${isTarget ? "compromised" : "reached"}`,
          nodes,
        });
      } else {
        const p = Math.max(...nodes.map(probabilityOf));
        out.push({ ip, name: nameOf(ip), state: contained ? "contained" : "at-risk", value: p, label: contained ? "averted" : `${Math.round(p * 100)}% at risk`, nodes });
      }
    }
    for (const b of (branches ?? []).filter((x) => x.kind !== "backoff")) {
      for (const h of b.hops ?? []) {
        if (!isInternal(h.ip) || out.some((a) => a.ip === h.ip)) continue;
        out.push({ ip: h.ip, name: h.name, state: contained ? "contained" : "at-risk", value: b.probability, label: contained ? "averted" : `${Math.round(b.probability * 100)}% at risk`, nodes: [] });
      }
    }
    const rank = { compromised: 0, reached: 1, "at-risk": 2, contained: 3 };
    return out.sort((a, b) => rank[a.state] - rank[b.state] || b.value - a.value);
  })();

  /* ── Neutralisation ──────────────────────────────────────────────── */
  const isolateHosts = reachedHosts.filter((ip) => isInternal(ip));
  const campaignId = campaign.campaign_id;

  function neutralize() {
    setConfirm(false);
    setTab("assets");
    setPulse(0);
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
    const step = 240;
    order.forEach((n, i) => {
      timers.current.push(
        window.setTimeout(() => {
          setPulse(i + 1);
          // Each host is isolated as the pulse first reaches it.
          if (n.provenance === "OBSERVED" && order.findIndex((m) => m.host_ip === n.host_ip) === i && isInternal(n.host_ip)) {
            void sendMitigate({ action: "ISOLATE_HOST", target: n.host_ip } as never);
          }
        }, step * (i + 1))
      );
    });
    timers.current.push(
      window.setTimeout(() => {
        if (attacker) void sendMitigate({ action: "BLOCK_IP", target: attacker } as never);
        setNeutralizedAt(observedCount);
        const at = Date.now() / 1000;
        for (const inc of bound.filter((i) => i.status !== "closed" && i.status !== "contained")) {
          onUpdateIncident(inc.id, {
            status: "contained",
            notes: [...(edits[inc.id]?.notes ?? []), { at, text: `Campaign #${campaignId} neutralised: ${isolateHosts.length} hosts isolated${attacker ? `, ${attacker} blocked` : ""}.` }],
          });
        }
        onToast({
          id: `neu${Date.now()}`,
          level: "nominal",
          title: `${codename[0].toUpperCase()}${codename.slice(1)} neutralised`,
          body: `${isolateHosts.length} hosts isolated${attacker ? ` · ${attacker}/32 blocked` : ""} · ${bound.length} incidents contained`,
        });
      }, step * (order.length + 1))
    );
  }

  const nodeColour = (s: NodeStatus, n: CampaignNode) =>
    s === "contained" ? "var(--sev-nominal)" : s === "forecast" ? FORECAST : s === "active" ? sevColor(sevFromRisk(risk, threshold)) : sevColor(sevFromRisk(n.max_risk_score, threshold));

  return (
    <div className="cm">
      {/* ── Dossier ──────────────────────────────────────────────────── */}
      <div className="cm-dossier" style={{ borderLeftColor: sevColor(status.level) }}>
        <div className="cm-dossier-main">
          <div className="cm-title">
            <span className="cm-id">CAMPAIGN-{String(campaign.campaign_id).padStart(2, "0")}</span>
            <h2>{codename[0].toUpperCase() + codename.slice(1)}</h2>
            <span className={`cm-status is-${status.level}`}>{status.label}</span>
          </div>
          <div className="cm-facts">
            <div title="Flow records carry no actor indicators; activity is clustered by TTP sequence and source infrastructure.">
              <Micro>Actor</Micro>
              <b>UNC-CW-{String(campaign.campaign_id).padStart(2, "0")}</b>
              <em>internal cluster · unattributed</em>
            </div>
            <div>
              <Micro>First observed</Micro>
              <b>t {since}s</b>
              <em>{hhmmss(firstSeen)}</em>
            </div>
            <div>
              <Micro>Incidents bound</Micro>
              <b>{bound.length}</b>
              <em>{bound.map((i) => i.id).join(" · ") || "none"}</em>
            </div>
            <div>
              <Micro>Velocity</Micro>
              <b>{stageVelocity.toFixed(1)} stages/min</b>
              <em>{hostVelocity.toFixed(1)} hosts/min</em>
            </div>
            <div>
              <Micro>Source</Micro>
              <b>{attacker ?? "aged out"}</b>
              <em>{campaign.involved_hosts.length} hosts involved</em>
            </div>
          </div>
          <div className="cm-ttp">
            <Micro>TTP sequence</Micro>
            {ttp.length ? (
              ttp.map((t, i) => (
                <span key={`${t}-${i}`}>
                  {i > 0 && <i>→</i>}
                  <Chip>{t}</Chip>
                </span>
              ))
            ) : (
              <em>no stages observed yet</em>
            )}
            {forecastNodes.length > 0 && !contained && (
              <span className="cm-ttp-fc">
                <i>→</i> {forecastNodes.map((n) => n.technique_id).join(" · ")} <em>forecast</em>
              </span>
            )}
          </div>
        </div>

        <div className="cm-dossier-side">
          <div className="cm-risk">
            <Micro>Campaign risk</Micro>
            <span style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>
              <Num value={risk} digits={3} />
            </span>
            <em>peak {campaign.max_risk_score.toFixed(3)}</em>
          </div>
          <button className={contained ? "cm-nuke is-done" : "cm-nuke"} disabled={contained || pulse != null || !isolateHosts.length} onClick={() => setConfirm(true)}>
            {contained ? "✓ Neutralised" : pulse != null ? "Neutralising…" : "Neutralize campaign"}
            <small>
              {contained ? "all reached hosts isolated" : `isolate ${isolateHosts.length} hosts · block source`}
            </small>
          </button>
        </div>
      </div>

      {/* ── Evolution graph ─────────────────────────────────────────── */}
      <Panel flush clip className="cm-graph">
        <PanelHead
          title="Attack evolution"
          note={`${observed.length} observed · ${forecastNodes.length} forecast · ${campaign.edges.length} causal links`}
          aside={
            <Legend
              items={[
                { color: OBSERVED, label: "completed" },
                { color: sevColor(sevFromRisk(risk, threshold)), label: "active" },
                { color: FORECAST, label: "forecast", dashed: true },
                { color: "var(--sev-nominal)", label: "contained" },
              ]}
            />
          }
        />
        <div className="cm-canvas" ref={canvasRef} onClick={() => setSel(null)}>
          <svg width={VB_W} height={height} style={{ display: "block" }}>
            {/* Lanes */}
            {lanes.map((lane, i) => (
              <g key={lane}>
                <line x1={GUTTER + laneW * i} y1={0} x2={GUTTER + laneW * i} y2={height} stroke="var(--rule-hair)" strokeWidth={1} shapeRendering="crispEdges" />
                <text x={GUTTER + laneW * i + 8} y={16} fill="var(--paper-600)" fontSize={9.5} fontFamily="var(--face-ui)" letterSpacing="1.8">
                  {(LANE_SHORT[lane] ?? lane).toUpperCase()}
                </text>
              </g>
            ))}

            {/* Host rows */}
            {hosts.map((h, i) => {
              const dim = hoverHost != null && hoverHost !== h;
              return (
                <g key={h} style={{ opacity: dim ? 0.3 : 1 }} onMouseEnter={() => setHoverHost(h)} onMouseLeave={() => setHoverHost(null)}>
                  <line x1={0} y1={LANE_LABEL_H + ROW_H * i} x2={VB_W} y2={LANE_LABEL_H + ROW_H * i} stroke="var(--rule-hair)" strokeWidth={1} shapeRendering="crispEdges" />
                  <text x={8} y={LANE_LABEL_H + ROW_H * i + ROW_H / 2 - 4} fill={hoverHost === h ? "var(--paper-000)" : "var(--paper-400)"} fontSize={12} fontFamily="var(--face-data)">
                    {nameOf(h)}
                  </text>
                  <text x={8} y={LANE_LABEL_H + ROW_H * i + ROW_H / 2 + 11} fill="var(--paper-600)" fontSize={9.5} fontFamily="var(--face-data)">
                    {h}
                  </text>
                </g>
              );
            })}

            {/* Causal edges */}
            {campaign.edges.map((e) => {
              const a = pos.get(e.src);
              const b = pos.get(e.dst);
              const na = byId.get(e.src);
              const nb = byId.get(e.dst);
              if (!a || !b || !na || !nb) return null;
              const sb = statusOf(nb);
              const sa = statusOf(na);
              const hw = (laneW - 14) / 2;
              const x1 = a.x + (b.x >= a.x ? hw : -hw);
              const x2 = b.x - (b.x >= a.x ? hw : -hw);
              const mx = (x1 + x2) / 2;
              const d = `M ${x1} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${x2} ${b.y}`;
              const dim = hoverHost != null && hoverHost !== na.host_ip && hoverHost !== nb.host_ip;
              const live = !contained && sa !== "forecast" && sb !== "forecast" && sb !== "contained";
              return (
                <g key={`${e.src}-${e.dst}`} style={{ opacity: dim ? 0.2 : 1 }}>
                  <path
                    d={d}
                    fill="none"
                    stroke={sb === "contained" ? "var(--sev-nominal)" : sb === "forecast" ? FORECAST : "var(--sev-critical)"}
                    strokeWidth={sb === "forecast" ? 1.4 : 2}
                    strokeOpacity={sb === "forecast" ? 0.9 : 0.7}
                    strokeDasharray={sb === "forecast" ? "5 4" : undefined}
                  />
                  {live && <path d={d} fill="none" stroke="var(--paper-000)" strokeWidth={1.3} strokeDasharray="6 26" className="is-traveling" strokeOpacity={0.8} />}
                  <text x={mx} y={(a.y + b.y) / 2 - 6} textAnchor="middle" fill="var(--paper-600)" fontSize={9} fontFamily="var(--face-data)">
                    {e.causality_score.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Stage cards */}
            {campaign.nodes.map((n) => {
              const p = pos.get(n.node_id);
              if (!p) return null;
              const s = statusOf(n);
              const c = nodeColour(s, n);
              const w = laneW - 14;
              const x = p.x - w / 2;
              const y = p.y - CARD_H / 2;
              const on = n.node_id === sel;
              const dim = hoverHost != null && hoverHost !== n.host_ip;
              const inc = boundOn(n.host_ip)[0];
              const statusText =
                s === "contained"
                  ? "CONTAINED"
                  : s === "forecast"
                    ? `FORECAST ${Math.round(probabilityOf(n) * 100)}%`
                    : s === "active"
                      ? `ACTIVE${inc ? ` · ${inc.id}` : ""}`
                      : `DONE${inc ? ` · ${inc.id}` : ""}`;
              return (
                <g
                  key={n.node_id}
                  className={s === "active" ? "cm-node is-active" : "cm-node"}
                  style={{ cursor: "pointer", opacity: dim ? 0.25 : 1 }}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    setSel(on ? null : n.node_id);
                    setTab(on ? "assets" : "detail");
                  }}
                >
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={CARD_H}
                    fill={s === "forecast" ? "var(--ink-050)" : "var(--ink-100)"}
                    stroke={on ? "var(--paper-000)" : c}
                    strokeWidth={on ? 1.6 : 1}
                    strokeDasharray={s === "forecast" ? "4 3" : undefined}
                    shapeRendering="crispEdges"
                  />
                  <rect x={x} y={y} width={4} height={CARD_H} fill={c} shapeRendering="crispEdges" />
                  {roots.has(n.node_id) && (
                    <text x={x + w - 6} y={y + 11} textAnchor="end" fill="var(--paper-600)" fontSize={8} fontFamily="var(--face-ui)" letterSpacing="1.2">
                      ROOT
                    </text>
                  )}
                  <text x={x + 10} y={y + 13} fill={s === "forecast" ? FORECAST : "var(--paper-000)"} fontSize={11} fontFamily="var(--face-data)" fontWeight={500}>
                    {n.technique_id}
                  </text>
                  <text x={x + 10} y={y + 25} fill="var(--paper-400)" fontSize={9.5} fontFamily="var(--face-data)">
                    {TECHNIQUE_NAMES[n.technique_id] ?? n.coarse_category}
                  </text>
                  <text x={x + 10} y={y + 36} fill={c} fontSize={8.5} fontFamily="var(--face-ui)" fontWeight={600} letterSpacing="1">
                    {statusText}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </Panel>

      {/* ── Assets · nodes · detail ─────────────────────────────────── */}
      <Panel flush clip className="cm-bottom">
        <div className="tabs" role="tablist">
          <button className="tab" role="tab" aria-selected={tab === "assets"} onClick={() => setTab("assets")}>
            Correlated assets <span className="tab-count">{assets.length}</span>
          </button>
          <button className="tab" role="tab" aria-selected={tab === "nodes"} onClick={() => setTab("nodes")}>
            Alert nodes <span className="tab-count">{campaign.nodes.length}</span>
          </button>
          {selected && (
            <button className="tab" role="tab" aria-selected={tab === "detail"} onClick={() => setTab("detail")}>
              Node {String(selected.node_id).padStart(2, "0")} · {selected.technique_id}
            </button>
          )}
        </div>

        {tab === "assets" && (
          <div className="cm-assets">
            {assets.map((a) => {
              const incs = boundOn(a.ip);
              const hits = a.nodes.filter((n) => n.provenance === "OBSERVED").reduce((s, n) => s + n.hit_count, 0);
              const level = a.state === "contained" ? "nominal" : a.state === "at-risk" ? "warning" : "critical";
              return (
                <div
                  key={a.ip}
                  className={`cm-asset is-${a.state}${hoverHost === a.ip ? " is-hover" : ""}`}
                  onMouseEnter={() => setHoverHost(a.ip)}
                  onMouseLeave={() => setHoverHost(null)}
                >
                  <div className="cm-asset-head">
                    <b>{a.name}</b>
                    <em>{a.ip}</em>
                  </div>
                  <div className="cm-asset-state" style={{ color: sevColor(level) }}>
                    {a.label}
                  </div>
                  <Meter value={a.state === "contained" ? 0 : a.value} level={level} height={5} />
                  <div className="cm-asset-facts">
                    {a.nodes.filter((n) => n.provenance === "OBSERVED").length} stages · {hits.toLocaleString()} hits
                    {incs.length ? ` · ${incs.map((i) => i.id).join(" ")}` : ""}
                  </div>
                </div>
              );
            })}
            {!assets.length && <Empty>No hosts correlated yet</Empty>}
          </div>
        )}

        {tab === "nodes" && (
          <div className="cm-table">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 48 }}>Node</th>
                  <th style={{ width: 130 }}>Host</th>
                  <th style={{ width: 120 }}>Stage</th>
                  <th style={{ width: 84 }}>Technique</th>
                  <th style={{ width: 96 }}>Provenance</th>
                  <th className="num" style={{ width: 70 }}>Hits</th>
                  <th className="num" style={{ width: 76 }}>Max risk</th>
                  <th className="num" style={{ width: 64 }}>Conf.</th>
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
                      onClick={() => {
                        setSel(n.node_id);
                        setTab("detail");
                      }}
                      onMouseEnter={() => setHoverHost(n.host_ip)}
                      onMouseLeave={() => setHoverHost(null)}
                      style={{ cursor: "pointer", opacity: n.provenance === "FORECAST" ? 0.72 : 1 }}
                    >
                      <td className="key" style={{ borderLeft: `3px solid ${sevColor(s)}` }}>
                        {String(n.node_id).padStart(2, "0")}
                      </td>
                      <td className="key">{nameOf(n.host_ip)}</td>
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
        )}

        {tab === "detail" && selected && (
          <div className="cm-detail">
            <div>
              <Field label="Technique" value={`${selected.technique_id} · ${TECHNIQUE_NAMES[selected.technique_id] ?? "—"}`} />
              <Field label="Stage" value={selected.coarse_category} />
              <Field label="Host" value={`${nameOf(selected.host_ip)} · ${selected.host_ip}`} />
            </div>
            <div>
              <Field label="Provenance" value={selected.provenance} color={selected.provenance === "FORECAST" ? FORECAST : undefined} />
              <Field label="Hits" value={selected.hit_count.toLocaleString()} />
              <Field label="Max risk" value={selected.max_risk_score.toFixed(3)} color={sevColor(sevFromRisk(selected.max_risk_score, threshold))} />
            </div>
            <div>
              <Field label="Mean conf." value={selected.mean_confidence.toFixed(3)} />
              <Field label="First seen" value={hhmmss(selected.start_time)} />
              <Field label="Root cause" value={roots.has(selected.node_id) ? "yes" : "no"} />
            </div>
            <div>
              <Micro style={{ marginBottom: 4 }}>Causal links</Micro>
              {campaign.edges
                .filter((e) => e.src === selected.node_id || e.dst === selected.node_id)
                .map((e) => {
                  const other = byId.get(e.src === selected.node_id ? e.dst : e.src);
                  return (
                    <Field key={`${e.src}-${e.dst}`} label={e.src === selected.node_id ? "→ downstream" : "← upstream"} value={`${other?.technique_id ?? "?"}  ${e.causality_score.toFixed(2)}`} />
                  );
                })}
            </div>
          </div>
        )}
      </Panel>

      {/* ── Confirmation ─────────────────────────────────────────────── */}
      {confirm && (
        <Sheet title={`Neutralize ${codename}`} onDismiss={() => setConfirm(false)}>
          <div className="cm-confirm">
            <p>
              Isolate every host this campaign has reached and block its source. Each step goes through <code>/api/mitigate</code>, the same
              path as a manual isolation.
            </p>
            <ol>
              {isolateHosts.map((ip) => (
                <li key={ip}>
                  <span>isolate</span> {nameOf(ip)} <em>{ip}</em>
                  {boundOn(ip).length ? <i>{boundOn(ip).map((i) => i.id).join(" ")}</i> : null}
                </li>
              ))}
              {attacker && (
                <li>
                  <span>block</span> {attacker}/32 <em>at the edge</em>
                </li>
              )}
            </ol>
            <div className="cm-confirm-actions">
              <Btn onClick={() => setConfirm(false)}>Cancel</Btn>
              <Btn danger onClick={neutralize} sub={`${isolateHosts.length + (attacker ? 1 : 0)} actions`}>
                Confirm neutralization
              </Btn>
            </div>
          </div>
        </Sheet>
      )}
    </div>
  );
}
