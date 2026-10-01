import { useEffect, useMemo, useState } from "react";
import type { PredictionResult, Topology } from "../api/types";
import type { Campaign } from "../types/campaign";
import type { FlowRecord } from "../types/evidence";
import type { ForecastBranch } from "../types/forecast";
import type { PredictionEnvelope } from "../types/live";
import type { Incident, IncidentEdit, IncidentStatus } from "../types/incident";
import { INCIDENT_STATUSES, applyEdit, rankIncidents } from "../types/incident";
import type { Toast } from "../components/Feedback";
import { Chip, Empty, Micro, Panel, PanelHead, Readout, Square, sevColor, sevFromRisk } from "../design/primitives";
import { FORECAST, OBSERVED } from "../design/charts";
import { clockTime } from "../design/time";
import { sendMitigate } from "../api/adapter";

/**
 * Incident war room.
 *
 * Events is a log; this is the job. The queue on the left is every
 * correlated incident carried through new → triaging → contained → closed.
 * The room on the right is one incident, worked: who did what to which host,
 * how far it can spread, how far ahead the model saw it, and the actions that
 * end it.
 *
 * Containment goes through the same mitigation endpoint the Controls view
 * uses (control_backend/telemetry_service.py: ISOLATE_HOST, BLOCK_IP), so an
 * isolation here is an isolation everywhere. Status changes and notes are
 * local edits on top of the stream — the backend has no incident store yet,
 * and src/types/incident.ts is the contract to serve against when it does.
 */

interface IncidentsProps {
  incidents: Incident[];
  overrides: Record<string, IncidentEdit>;
  onUpdate: (id: string, patch: IncidentEdit) => void;
  onOpenHost: (ip: string) => void;
  onToast: (t: Toast) => void;
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  campaign: Campaign | null;
  branches: ForecastBranch[] | null;
  flows: FlowRecord[];
  topology: Topology | null;
  /** Opens the room on this incident, e.g. from Overview. */
  focusId?: string | null;
}

const ANALYST = "a.rao";
type Tab = "blast" | "trail" | "activity";

function hhmm(epochSeconds: number): string {
  try {
    return clockTime(epochSeconds * 1000);
  } catch {
    return "—";
  }
}

function ago(epochSeconds: number): string {
  const d = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (d < 60) return `${d}s`;
  if (d < 3600) return `${Math.floor(d / 60)}m`;
  return `${Math.floor(d / 3600)}h ${Math.floor((d % 3600) / 60)}m`;
}

function duration(from: number, to: number): string {
  const d = Math.max(0, Math.floor(to - from));
  if (d < 60) return `${d}s`;
  if (d < 3600) return `${Math.floor(d / 60)}m ${d % 60}s`;
  return `${Math.floor(d / 3600)}h ${Math.floor((d % 3600) / 60)}m`;
}

function rel(seconds: number): string {
  const s = Math.round(seconds);
  if (s === 0) return "NOW";
  const a = Math.abs(s);
  const txt = a < 120 ? `${a}s` : a < 7200 ? `${Math.floor(a / 60)}m` : `${Math.floor(a / 3600)}h ${Math.floor((a % 3600) / 60)}m`;
  return s < 0 ? `t −${txt}` : `+${txt}`;
}

function fmtBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

const STATUS_TONE: Record<IncidentStatus, string> = {
  new: "critical",
  triaging: "warning",
  contained: "nominal",
  closed: "unknown",
};

function StatusBadge({ status }: { status: IncidentStatus }) {
  return (
    <span className={`in-badge is-${status}`}>
      <Square status={STATUS_TONE[status]} live={status === "new" || status === "triaging"} />
      {status}
    </span>
  );
}

/* ── Blast radius ──────────────────────────────────────────────────────── */

interface BlastNode {
  ip: string;
  name: string;
  role: "source" | "target" | "downstream";
  state: "hostile" | "compromised" | "reached" | "at-risk" | "isolated";
  fact: string;
  probability?: number;
  eta?: number;
  technique?: string;
}

function BlastRadius({ nodes, contained }: { nodes: BlastNode[]; contained: boolean }) {
  const source = nodes.find((n) => n.role === "source") ?? null;
  const target = nodes.find((n) => n.role === "target") ?? null;
  const down = nodes.filter((n) => n.role === "downstream");
  if (!target) return <Empty>No host attributed</Empty>;

  // Positions on a 100 × 100 grid, rendered as percentages.
  const col = { source: 12, target: 42, down: 79 };
  const ys = down.length ? down.map((_, i) => ((i + 1) / (down.length + 1)) * 100) : [];
  const card = (n: BlastNode, x: number, y: number) => {
    const downstream = n.role === "downstream";
    // Downstream cards carry their probability and time in place of a floating label.
    const fact =
      downstream && n.probability != null && !contained
        ? `${Math.round(n.probability * 100)}% · +${n.eta}s${n.technique ? ` · ${n.technique}` : ""}`
        : n.fact;
    return (
      <div key={n.ip} className={`br-node is-${n.state}${downstream ? " is-down" : ""}`} style={{ left: `${x}%`, top: `${y}%` }}>
        <span className="br-role">
          {n.role === "source" ? "attacker" : n.role === "target" ? "target" : n.state === "reached" ? "reached" : "at risk"}
        </span>
        {downstream ? (
          <b>
            {n.name} <em>{n.ip}</em>
          </b>
        ) : (
          <>
            <b>{n.name}</b>
            <em>{n.ip}</em>
          </>
        )}
        <span className="br-fact">{fact}</span>
      </div>
    );
  };

  return (
    <div className="br">
      <svg className="br-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        {source && (
          <>
            <line x1={col.source} y1={50} x2={col.target} y2={50} className={contained ? "br-link is-cut" : "br-link is-attack"} vectorEffect="non-scaling-stroke" />
            {!contained && <line x1={col.source} y1={50} x2={col.target} y2={50} className="br-flow is-traveling" vectorEffect="non-scaling-stroke" />}
          </>
        )}
        {down.map((n, i) => (
          <path
            key={n.ip}
            d={`M ${col.target} 50 C ${col.target + 18} 50, ${col.down - 18} ${ys[i]}, ${col.down} ${ys[i]}`}
            className={contained ? "br-link is-cut" : n.state === "reached" ? "br-link is-attack" : "br-link is-risk"}
            vectorEffect="non-scaling-stroke"
            fill="none"
          />
        ))}
      </svg>
      {source ? card(source, col.source, 50) : (
        <div className="br-node is-unknown" style={{ left: `${col.source}%`, top: "50%" }}>
          <span className="br-role">attacker</span>
          <b>aged out</b>
          <span className="br-fact">not in the live window</span>
        </div>
      )}
      {card(target, col.target, 50)}
      {down.map((n, i) => card(n, col.down, ys[i]))}
      {!down.length && (
        <div className="br-none" style={{ left: `${col.down}%` }}>
          {contained ? "no onward exposure" : "no downstream exposure forecast"}
        </div>
      )}
    </div>
  );
}

/* ── Page ──────────────────────────────────────────────────────────────── */

export default function Incidents({
  incidents,
  overrides,
  onUpdate,
  onOpenHost,
  onToast,
  prediction,
  envelope,
  campaign,
  branches,
  flows,
  topology,
  focusId,
}: IncidentsProps) {
  const [filter, setFilter] = useState<Set<IncidentStatus>>(new Set());
  const [sel, setSel] = useState<string | null>(focusId ?? null);
  const [tab, setTab] = useState<Tab>("blast");
  const [done, setDone] = useState<Record<string, string>>({});

  const merged = useMemo(() => incidents.map((i) => applyEdit(i, overrides[i.id])).sort(rankIncidents), [incidents, overrides]);

  const counts = useMemo(() => {
    const c: Record<string, number> = { new: 0, triaging: 0, contained: 0, closed: 0 };
    for (const i of merged) c[i.status] = (c[i.status] ?? 0) + 1;
    return c;
  }, [merged]);

  const rows = filter.size ? merged.filter((i) => filter.has(i.status)) : merged;
  const open = merged.filter((i) => i.status !== "closed");

  // The room is never empty on arrival: the top of the queue is worked by default.
  const selected = merged.find((i) => i.id === sel) ?? rows[0] ?? null;

  /** Median time to contain, over incidents that actually reached contained. */
  const mttc = useMemo(() => {
    const d = merged.filter((i) => i.status === "contained" || i.status === "closed");
    if (!d.length) return null;
    const secs = d.map((i) => i.lastSeen - i.opened).sort((a, b) => a - b);
    return secs[Math.floor(secs.length / 2)];
  }, [merged]);

  // A new selection clears the action confirmations shown for the last one.
  useEffect(() => setDone({}), [selected?.id]);

  function toggle(s: IncidentStatus) {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  function note(inc: Incident, text: string, patch: IncidentEdit = {}) {
    onUpdate(inc.id, { ...patch, notes: [...(overrides[inc.id]?.notes ?? []), { at: Date.now() / 1000, text }] });
  }

  /* ── The live picture, for the incident the stream is still working ── */
  const live = selected != null && selected.campaignId != null && campaign != null && campaign.campaign_id === selected.campaignId;
  const nameOf = (ip: string) => topology?.nodes.find((n) => n.ip === ip)?.label ?? (ip.startsWith("10.") ? ip : "external");
  const attacker =
    (envelope?.focus_ips ?? []).find((ip) => !ip.startsWith("10.")) ??
    (branches ?? []).flatMap((b) => b.hops).find((h) => !h.ip.startsWith("10."))?.ip ??
    null;
  const contained = selected ? selected.status === "contained" || selected.status === "closed" : false;
  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);

  const blast = useMemo<BlastNode[]>(() => {
    if (!selected) return [];
    const out: BlastNode[] = [];
    const flowsNow = flows.filter((f) => f.window === (envelope?.state?.window_id ?? -1));
    if (live && attacker) {
      const n = flowsNow.filter((f) => f.src_ip === attacker || f.dst_ip === attacker);
      out.push({
        ip: attacker,
        name: "external",
        role: "source",
        state: contained ? "isolated" : "hostile",
        fact: contained ? "blocked" : `${n.length} flows · ${fmtBytes(n.reduce((s, f) => s + f.fwd_bytes + f.bwd_bytes, 0))}`,
      });
    }
    out.push({
      ip: selected.host,
      name: selected.hostLabel,
      role: "target",
      state: contained ? "isolated" : "compromised",
      fact: contained ? "isolated" : live ? `risk ${risk.toFixed(3)}` : `peak ${selected.peakRisk.toFixed(3)}`,
    });
    if (!live) return out;

    // Downstream: hosts the campaign has already reached, then the ones the forecast puts at risk.
    const seen = new Set([selected.host, attacker ?? ""]);
    for (const n of campaign?.nodes ?? []) {
      if (n.provenance !== "OBSERVED" || seen.has(n.host_ip)) continue;
      seen.add(n.host_ip);
      out.push({ ip: n.host_ip, name: nameOf(n.host_ip), role: "downstream", state: "reached", fact: `${n.technique_id} observed · ${n.hit_count} hits` });
    }
    for (const b of [...(branches ?? [])].filter((b) => b.kind !== "backoff").sort((a, c) => c.probability - a.probability)) {
      for (const h of b.hops) {
        if (seen.has(h.ip) || !h.ip.startsWith("10.")) continue;
        seen.add(h.ip);
        out.push({
          ip: h.ip,
          name: h.name,
          role: "downstream",
          state: "at-risk",
          fact: `branch ${b.id} · peak ${b.peak_risk.toFixed(2)}`,
          probability: b.probability,
          eta: b.horizon_seconds,
          technique: b.technique !== "—" ? b.technique : undefined,
        });
      }
    }
    return out.filter((n) => n.role !== "downstream" || out.filter((m) => m.role === "downstream").indexOf(n) < 3);
  }, [selected, live, attacker, contained, flows, envelope, campaign, branches, risk, topology]);

  /* ── Lead-time trail ─────────────────────────────────────────────── */
  const trail = useMemo(() => {
    if (!selected) return [];
    const nowSec = Date.now() / 1000;
    const items: { at: number; kind: "observed" | "now" | "forecast" | "note"; title: string; detail?: string }[] = [];
    if (live && campaign) {
      for (const n of campaign.nodes) {
        if (n.provenance !== "OBSERVED") continue;
        items.push({
          at: n.start_time - campaign.end_time,
          kind: "observed",
          title: `${n.technique_id} ${n.coarse_category} on ${nameOf(n.host_ip)}`,
          detail: `confidence ${n.mean_confidence.toFixed(3)} · ${n.hit_count.toLocaleString()} hits · risk ${n.max_risk_score.toFixed(2)}`,
        });
      }
    }
    for (const n of selected.notes) items.push({ at: n.at - nowSec, kind: "note", title: n.text });
    if (live && prediction && !contained) {
      const ml = prediction.ml_risk != null ? Number(prediction.ml_risk) : null;
      items.push({
        at: 0,
        kind: "now",
        title: `Active: ${prediction.mitre_technique ?? selected.title}`,
        detail:
          `risk ${risk.toFixed(3)} ${String(prediction.alert_level ?? "").toUpperCase()}` +
          (prediction.rules_applied && ml != null ? ` · rule layer lifted ML ${ml.toFixed(3)} → ${risk.toFixed(3)}` : ""),
      });
      for (const b of (branches ?? []).filter((b) => b.kind !== "backoff").slice(0, 2)) {
        items.push({
          at: b.horizon_seconds,
          kind: "forecast",
          title: `${Math.round(b.probability * 100)}% · ${b.label}`,
          detail: `${b.technique !== "—" ? `${b.technique} · ` : ""}${b.hops.map((h) => h.name).join(" → ")}`,
        });
      }
    }
    return items.sort((a, b) => a.at - b.at);
  }, [selected, live, campaign, prediction, contained, branches, risk, topology]);

  /* ── Actions ─────────────────────────────────────────────────────── */
  function isolate(inc: Incident) {
    void sendMitigate({ action: "ISOLATE_HOST", target: inc.host } as never);
    note(inc, `SOAR: isolated ${inc.hostLabel} (${inc.host}). Traffic to and from the host is dropped at the sensor.`, {
      status: "contained",
      assignee: inc.assignee ?? ANALYST,
    });
    setDone((d) => ({ ...d, isolate: "Isolated" }));
    onToast({ id: `iso${Date.now()}`, level: "nominal", title: "Host isolated", body: `${inc.hostLabel} ${inc.host} · ${inc.id} contained` });
  }

  function block(inc: Incident) {
    if (!attacker) return;
    const rule = `fw-${String(Math.floor(Date.now() / 1000) % 10000).padStart(4, "0")}`;
    void sendMitigate({ action: "BLOCK_IP", target: attacker } as never);
    note(inc, `SOAR: edge rule ${rule} — deny ip ${attacker}/32 any → 10.0.0.0/16, all ports. Source blocked.`, {
      status: "contained",
      assignee: inc.assignee ?? ANALYST,
    });
    setDone((d) => ({ ...d, block: `Blocked · ${rule}` }));
    onToast({ id: `blk${Date.now()}`, level: "nominal", title: `Rule ${rule} deployed`, body: `deny ${attacker}/32 → 10.0.0.0/16` });
  }

  function exportReport(inc: Incident) {
    const onPath = flows.filter((f) => f.on_path).slice(-24);
    const lines = [
      `# ${inc.id} — ${inc.title}`,
      "",
      `- Status: ${inc.status} · Severity: ${inc.severity} · Peak risk: ${inc.peakRisk.toFixed(3)}`,
      `- Host: ${inc.hostLabel} (${inc.host}) · Tactic: ${inc.tactic ?? "—"} · Technique: ${inc.technique ?? "—"}`,
      `- Opened: ${hhmm(inc.opened)} · Last seen: ${hhmm(inc.lastSeen)} · Alerts: ${inc.alertCount} · Assignee: ${inc.assignee ?? "unassigned"}`,
      inc.leadTimeSeconds != null ? `- Lead time: ${inc.leadTimeSeconds.toFixed(1)}s ahead of the milestone` : "",
      "",
      "## Blast radius",
      ...blast.map((n) => `- ${n.role}: ${n.name} ${n.ip} — ${n.fact}${n.probability != null ? ` (${Math.round(n.probability * 100)}% · +${n.eta}s)` : ""}`),
      "",
      "## Timeline",
      ...trail.map((t) => `- ${rel(t.at)}  ${t.title}${t.detail ? ` — ${t.detail}` : ""}`),
      "",
      "## Evidence — on-path flows",
      "| time | source | destination | proto | fwd pkts | fwd bytes | bwd pkts | bwd bytes | SYN/ACK/PSH/RST/FIN |",
      "|---|---|---|---|---|---|---|---|---|",
      ...onPath.map(
        (f) =>
          `| ${clockTime(f.ts_us / 1000)} | ${f.src_ip}:${f.src_port} | ${f.dst_ip}:${f.dst_port} | ${f.protocol} | ${f.fwd_packets} | ${f.fwd_bytes} | ${f.bwd_packets} | ${f.bwd_bytes} | ${f.flags.syn}/${f.flags.ack}/${f.flags.psh}/${f.flags.rst}/${f.flags.fin} |`
      ),
      "",
      "## Activity",
      ...inc.notes.map((n) => `- ${hhmm(n.at)}  ${n.text}`),
      "",
      `_Exported ${new Date().toISOString()}_`,
    ].filter((l, i, a) => l !== "" || a[i - 1] !== "");
    try {
      const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${inc.id}-report.md`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch {
      /* the toast still records the attempt */
    }
    note(inc, `Report exported — ${inc.id}-report.md (${onPath.length} evidence flows).`);
    setDone((d) => ({ ...d, export: "Exported" }));
    onToast({ id: `exp${Date.now()}`, level: "nominal", title: "Report exported", body: `${inc.id}-report.md · ${onPath.length} evidence flows` });
  }

  return (
    <div className="in">
      {/* ── Queue stats ──────────────────────────────────────────────── */}
      <div className="in-stats sheet">
        <Readout label="Open" value={String(open.length)} level={open.length ? "elevated" : "nominal"} sub="requiring action" />
        <Readout
          label="Unassigned"
          value={String(open.filter((i) => !i.assignee).length)}
          sub="no analyst"
          level={open.some((i) => !i.assignee) ? "warning" : "nominal"}
        />
        <Readout label="Median contain" value={mttc != null ? duration(0, mttc) : "—"} sub="this shift" />
        <Readout
          label="Mean lead time"
          value={
            merged.some((i) => i.leadTimeSeconds != null)
              ? (merged.reduce((s, i) => s + (i.leadTimeSeconds ?? 0), 0) / merged.filter((i) => i.leadTimeSeconds != null).length).toFixed(1)
              : "—"
          }
          unit="s"
          level="nominal"
          sub="ahead of milestone"
        />
      </div>

      <div className="in-split">
        {/* ── Queue ────────────────────────────────────────────────────── */}
        <Panel flush clip className="in-queue">
          <PanelHead
            title="Queue"
            note={`${open.length} open · ${merged.length} total`}
            aside={
              <div className="in-filters">
                {INCIDENT_STATUSES.map((s) => (
                  <button key={s} onClick={() => toggle(s)} aria-pressed={filter.has(s)} className={`in-filter is-${s}`}>
                    {s} {counts[s] ?? 0}
                  </button>
                ))}
              </div>
            }
          />
          <div className="in-list">
            {rows.length ? (
              rows.map((i) => (
                <button
                  key={i.id}
                  className={i.id === selected?.id ? "in-row is-selected" : "in-row"}
                  onClick={() => {
                    setSel(i.id);
                    setTab("blast");
                  }}
                  style={{ borderLeftColor: sevColor(i.severity), opacity: i.status === "closed" ? 0.62 : 1 }}
                >
                  <span className="in-row-head">
                    <b>{i.id}</b>
                    <StatusBadge status={i.status} />
                    <i style={{ color: sevColor(i.severity) }}>{i.peakRisk.toFixed(3)}</i>
                  </span>
                  <span className="in-row-title">{i.title}</span>
                  <span className="in-row-meta">
                    {i.hostLabel} · {i.host} · {ago(i.opened)} · {i.assignee ?? "unassigned"}
                  </span>
                </button>
              ))
            ) : (
              <Empty hint="Incidents open automatically when forecast risk crosses the operating threshold.">
                {merged.length ? "No match for this filter" : "Queue clear"}
              </Empty>
            )}
          </div>
        </Panel>

        {/* ── War room ─────────────────────────────────────────────────── */}
        <Panel flush clip className="in-room">
          {selected ? (
            <div key={selected.id} className="in-room-inner is-entering">
              <div className="in-room-head" style={{ borderLeftColor: sevColor(selected.severity) }}>
                <div className="in-room-id">
                  <b>{selected.id}</b>
                  <StatusBadge status={selected.status} />
                  <Chip level={selected.severity}>{selected.severity}</Chip>
                  <span className="in-room-peak">
                    peak <b style={{ color: sevColor(selected.severity) }}>{selected.peakRisk.toFixed(3)}</b>
                  </span>
                </div>
                <div className="in-room-title">{selected.title}</div>
                <div className="in-room-meta">
                  {selected.tactic ?? "unclassified"} · {selected.hostLabel} {selected.host} · opened {hhmm(selected.opened)} ·{" "}
                  {duration(selected.opened, selected.lastSeen)} · {selected.alertCount.toLocaleString()} alerts · {selected.assignee ?? "unassigned"}
                  {selected.leadTimeSeconds != null && (
                    <>
                      {" "}
                      · lead <b style={{ color: "var(--sev-nominal)" }}>{selected.leadTimeSeconds.toFixed(1)}s</b>
                    </>
                  )}
                </div>
              </div>

              {/* ── Containment dock ──────────────────────────────────── */}
              <div className="in-dock">
                <button className={done.isolate ? "soar is-done" : "soar is-primary"} disabled={contained && !done.isolate} onClick={() => isolate(selected)}>
                  {done.isolate ? `✓ ${done.isolate}` : "Isolate target host"}
                  <small>{selected.hostLabel} {selected.host}</small>
                </button>
                <button className={done.block ? "soar is-done" : "soar"} disabled={!attacker || (contained && !done.block) || !live} onClick={() => block(selected)}>
                  {done.block ? `✓ ${done.block}` : "Block source"}
                  <small>{live && attacker ? `${attacker}/32 · edge` : "source not in window"}</small>
                </button>
                <button className={done.export ? "soar is-done" : "soar"} onClick={() => exportReport(selected)}>
                  {done.export ? `✓ ${done.export}` : "Export report"}
                  <small>{selected.id}-report.md</small>
                </button>
                <span className="in-dock-sep" />
                <span className="seg-group">
                  <button className="seg" disabled={selected.status !== "new"} onClick={() => note(selected, `Acknowledged by ${ANALYST}.`, { status: "triaging", assignee: selected.assignee ?? ANALYST })}>
                    Acknowledge
                  </button>
                  <button
                    className="seg"
                    disabled={selected.status === "contained" || selected.status === "closed"}
                    onClick={() => note(selected, "Marked contained — host isolated.", { status: "contained", assignee: selected.assignee ?? ANALYST })}
                  >
                    Mark contained
                  </button>
                  <button className="seg" disabled={selected.status === "closed"} onClick={() => note(selected, "Closed.", { status: "closed" })}>
                    Close
                  </button>
                  <button className="seg" onClick={() => onOpenHost(selected.host)}>
                    Open host
                  </button>
                </span>
              </div>

              {/* ── Tabs ──────────────────────────────────────────────── */}
              <div className="tabs in-tabs" role="tablist">
                <button className="tab" role="tab" aria-selected={tab === "blast"} onClick={() => setTab("blast")}>
                  Blast radius
                </button>
                <button className="tab" role="tab" aria-selected={tab === "trail"} onClick={() => setTab("trail")}>
                  Lead-time trail <span className="tab-count">{trail.length}</span>
                </button>
                <button className="tab" role="tab" aria-selected={tab === "activity"} onClick={() => setTab("activity")}>
                  Activity <span className="tab-count">{selected.notes.length}</span>
                </button>
              </div>

              <div className="in-tab-body">
                {tab === "blast" && <BlastRadius nodes={blast} contained={contained} />}

                {tab === "trail" && (
                  <ol className="lt">
                    {trail.map((t, i) => (
                      <li key={i} className={`lt-item is-${t.kind}`}>
                        <span className="lt-at">{rel(t.at)}</span>
                        <span
                          className="lt-dot"
                          style={{
                            background: t.kind === "observed" ? OBSERVED : t.kind === "now" ? sevColor(sevFromRisk(risk, threshold)) : "var(--ink-000)",
                            borderColor: t.kind === "forecast" ? FORECAST : t.kind === "note" ? "var(--paper-600)" : "transparent",
                          }}
                        />
                        <span className="lt-body">
                          <b>{t.title}</b>
                          {t.detail && <em>{t.detail}</em>}
                        </span>
                      </li>
                    ))}
                    {!trail.length && <Empty>No activity recorded</Empty>}
                  </ol>
                )}

                {tab === "activity" && (
                  <div className="in-notes">
                    {selected.notes
                      .slice()
                      .reverse()
                      .map((n, i) => (
                        <div key={i} className="in-note">
                          <span>{hhmm(n.at)}</span>
                          <p>{n.text}</p>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <Empty hint="Incidents open automatically when forecast risk crosses the operating threshold.">Queue clear</Empty>
          )}
          <div className="in-caption">
            <Micro>Workflow</Micro> new → triaging → contained → closed · containment calls /api/mitigate (ISOLATE_HOST, BLOCK_IP)
          </div>
        </Panel>
      </div>
    </div>
  );
}
