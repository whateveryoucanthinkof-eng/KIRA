/**
 * Demo fixtures — active only when `VITE_DEMO_MODE === 'true'` (`.env.demo`,
 * loaded by `npm run demo`). Dispatched from `adapter.ts`; never reached in a
 * live build.
 *
 * These emit the REAL wire shapes from `control_backend/schema.py`
 * (PredictionEvent, TopologyEvent, SystemStatusEvent) so the console renders
 * offline exactly as it does against the backend. Vocabularies are the
 * project's own:
 *
 *   techniques      branch_a_gnn_lstm/sequence_dataset.py:TECHNIQUE_VOCAB
 *   MITRE mapping   control_backend/model_adapter.py:TECHNIQUE_TO_MITRE
 *   host attributes data_unification/host_attributes.py
 *   feature groups  control_backend/model_adapter.py (group map)
 *   kill-chain      correlation/causal_edge_scorer.py:TACTIC_ORDER
 *   campaign shape  correlation/campaign_merge.py:AttackCampaign
 *   alert nodes     correlation/graph_compaction.py:CompactedAlertNode
 *   zones + assets  config/sites/containerlab-enterprise.yaml
 *
 * Nothing here uses Math.random: every value is a deterministic function of
 * the window index, so the scenario replays identically on every recording.
 */

import type { SystemStatus, Topology, SiteInfo, MitigationPayload, WSHandlers } from "./types";
import type { LivePoint } from "../types/live";
import type { Campaign, CampaignNode } from "../types/campaign";
import type { ReplayReport, ReplayRow } from "../types/replay";
import type { Incident } from "../types/incident";

/* ── Site (config/sites/containerlab-enterprise.yaml) ──────────────────── */

const ASSETS = [
  { ip: "10.0.3.10", name: "dmz-web", zone: "dmz" },
  { ip: "10.0.2.40", name: "srv-app", zone: "servers" },
  { ip: "10.0.2.30", name: "srv-file", zone: "servers" },
  { ip: "10.0.2.20", name: "srv-id", zone: "servers" },
  { ip: "10.0.2.10", name: "srv-dns", zone: "servers" },
  { ip: "10.0.2.50", name: "srv-db", zone: "servers" },
  { ip: "10.0.1.21", name: "ws-office", zone: "users" },
  { ip: "10.0.1.22", name: "ws-web", zone: "users" },
];

const ATTACKER = "192.168.100.7";
const TARGET = "10.0.3.10";

/** 830ms × 72 windows ≈ 58s for a full Recon → Impact → recovery arc. */
export const TICK_MS = 830;
const CYCLE = 72;
const RECOVERY_STEP = 62;
const THRESHOLD = 0.65;

/**
 * A scripted intrusion, one phase per band of the cycle. Risk climbs through
 * the kill chain rather than flickering, so the observed curve and the
 * forecast have a relationship worth watching.
 */
const PHASES = [
  { until: 14, stage: "Recon", technique: "T1046", risk: [0.04, 0.22], label: "T1046 Network Service Discovery", tactic: "Reconnaissance", tacticId: "TA0043", desc: "Sweeping the DMZ for reachable services." },
  { until: 26, stage: "InitialAccess", technique: "T1110", risk: [0.24, 0.52], label: "T1110 Brute Force", tactic: "Credential Access", tacticId: "TA0006", desc: "Repeated authentication attempts against the exposed web host." },
  { until: 38, stage: "InitialAccess", technique: "T1190", risk: [0.55, 0.78], label: "T1190 Exploit Public-Facing Application", tactic: "Initial Access", tacticId: "TA0001", desc: "Exploit attempts against the public-facing application." },
  { until: 50, stage: "C2", technique: "T1071", risk: [0.72, 0.91], label: "T1071 Application Layer Protocol", tactic: "Command and Control", tacticId: "TA0011", desc: "Beaconing consistent with application-layer command and control." },
  { until: RECOVERY_STEP, stage: "Impact", technique: "T1498", risk: [0.88, 0.96], label: "T1498 Network Denial of Service", tactic: "Impact", tacticId: "TA0040", desc: "Volumetric flood against the target host." },
  { until: CYCLE, stage: "Benign", technique: "Benign", risk: [0.1, 0.03], label: "Benign", tactic: "—", tacticId: "—", desc: "Traffic returned to baseline after mitigation." },
];

/** correlation/causal_edge_scorer.py:TACTIC_ORDER */
const LANES = ["Recon", "InitialAccess", "Execution", "C2", "LateralMovement", "Exfiltration", "Impact"];

/**
 * Campaign skeleton. A node is OBSERVED once the scenario reaches its `at`
 * step and FORECAST before that, so the graph grows as the intrusion advances.
 * Timestamps are derived per-window in `buildCampaign`.
 */
const CAMPAIGN_NODES: (Omit<CampaignNode, "provenance" | "start_time" | "end_time"> & { at: number })[] = [
  { node_id: 0, host_ip: TARGET, coarse_category: "Recon", technique_id: "T1046", hit_count: 41, max_risk_score: 0.22, mean_confidence: 0.74, at: 4 },
  { node_id: 1, host_ip: TARGET, coarse_category: "InitialAccess", technique_id: "T1110", hit_count: 188, max_risk_score: 0.52, mean_confidence: 0.81, at: 16 },
  { node_id: 2, host_ip: TARGET, coarse_category: "InitialAccess", technique_id: "T1190", hit_count: 23, max_risk_score: 0.78, mean_confidence: 0.86, at: 28 },
  { node_id: 3, host_ip: TARGET, coarse_category: "C2", technique_id: "T1071", hit_count: 64, max_risk_score: 0.91, mean_confidence: 0.9, at: 40 },
  { node_id: 4, host_ip: "10.0.2.40", coarse_category: "LateralMovement", technique_id: "T1021", hit_count: 12, max_risk_score: 0.83, mean_confidence: 0.77, at: 46 },
  { node_id: 5, host_ip: "10.0.2.50", coarse_category: "Exfiltration", technique_id: "T1005", hit_count: 7, max_risk_score: 0.88, mean_confidence: 0.71, at: 52 },
  { node_id: 6, host_ip: TARGET, coarse_category: "Impact", technique_id: "T1498", hit_count: 2104, max_risk_score: 0.96, mean_confidence: 0.94, at: 54 },
];

const CAMPAIGN_EDGES = [
  { src: 0, dst: 1, causality_score: 0.91 },
  { src: 1, dst: 2, causality_score: 0.88 },
  { src: 2, dst: 3, causality_score: 0.84 },
  { src: 3, dst: 4, causality_score: 0.79 },
  { src: 4, dst: 5, causality_score: 0.73 },
  { src: 3, dst: 6, causality_score: 0.81 },
];

/** 15 host attributes (data_unification/host_attributes.py) + 12 TGNE dims = 27-D. */
const HOST_ATTRS = [
  "flow_count", "fwd_bytes", "bwd_bytes", "total_bytes", "fwd_packets",
  "bwd_packets", "total_packets", "unique_peers", "unique_dst_ports",
  "tcp_ratio", "udp_ratio", "avg_duration", "byte_rate", "packet_rate", "peer_density",
];

const GROUP_OF: Record<string, string> = {
  flow_count: "Connectivity", unique_peers: "Connectivity", unique_dst_ports: "Connectivity", peer_density: "Connectivity",
  fwd_bytes: "Volume", bwd_bytes: "Volume", total_bytes: "Volume", fwd_packets: "Volume", bwd_packets: "Volume", total_packets: "Volume",
  tcp_ratio: "Protocol", udp_ratio: "Protocol",
  avg_duration: "Timing",
  byte_rate: "Rate", packet_rate: "Rate",
};

const MODEL_META = {
  name: "cyberworld dual-branch + DeepOP",
  version: "1.1.0-live-retrained",
  feature_count: 27,
  // cyberworld_v4/config.py: 15 x 2s history, 5 x 30s forecast.
  history_steps: 15,
  window_seconds: 2.0,
  forecast_steps: 5,
  forecast_step_seconds: 30.0,
  checkpoint: "saved_models/branch_a/branch_a_lstm.pt",
  threshold: THRESHOLD,
  rules_enabled: false,
};

/* ── deterministic helpers ─────────────────────────────────────────────── */

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

/** Stable pseudo-noise. Same step always yields the same value. */
function wobble(step: number, seed: number): number {
  return (Math.sin(step * 0.7 + seed * 2.3) + Math.sin(step * 1.9 + seed * 5.1)) * 0.5;
}

function phaseAt(step: number) {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  let prev = 0;
  for (let i = 0; i < PHASES.length; i++) {
    const p = PHASES[i];
    if (s < p.until) return { p, i, t: (s - prev) / (p.until - prev) };
    prev = p.until;
  }
  return { p: PHASES[PHASES.length - 1], i: PHASES.length - 1, t: 1 };
}

function alertLevelFor(risk: number): string {
  if (risk >= THRESHOLD + (1 - THRESHOLD) * 0.6) return "CRITICAL";
  if (risk >= THRESHOLD + (1 - THRESHOLD) * 0.25) return "ELEVATED";
  if (risk >= THRESHOLD) return "WARNING";
  return "NOMINAL";
}

/**
 * The full analytic state for one window. Shared by the pre-seed backfill and
 * the live emitter so a seeded point is indistinguishable from a streamed one.
 */
function windowState(step: number) {
  const { p, i, t } = phaseAt(step);
  const hot = p.technique !== "Benign";
  const base = lerp(p.risk[0], p.risk[1], t);
  const risk = Math.max(0, Math.min(0.99, base + wobble(step, 1) * 0.02));

  // Mirrors the live path: the displayed risk IS the model output. The
  // advisory rule layer is off by default and never replaces it.
  const mlRisk = risk;
  const rulesApplied = false;
  const ruleRisk: number | null = null;
  const shown = mlRisk;

  const flows = Math.round(hot ? 260 + shown * 1700 : 240 + wobble(step, 13) * 40);
  const packets = Math.round(hot ? 900 + shown * 14_000 : 700 + wobble(step, 17) * 180);
  const throughput = Number((hot ? 38 + shown * 690 : 36 + wobble(step, 19) * 7).toFixed(2));
  const telemetryMs = Number((3.1 + Math.abs(wobble(step, 23)) * 1.6).toFixed(2));
  const inferenceMs = Number((7.4 + Math.abs(wobble(step, 29)) * 3.2).toFixed(2));

  // K=5 rollout, 30s per step, confidence decaying with horizon.
  const forecast = Array.from({ length: 5 }, (_, k) => {
    const drift = hot ? 1 + (k + 1) * 0.035 : 1 - (k + 1) * 0.06;
    // Walk the forecast forward through the kill chain rather than repeating
    // the current stage — this is what the timeline's forecast cells render.
    const ahead = PHASES[Math.min(PHASES.length - 2, i + (k < 3 ? 0 : k < 6 ? 1 : 2))];
    const risk = Math.max(0, Math.min(0.99, shown * drift + wobble(step + k, 7) * 0.015));
    // Demo stand-in for the backend's per-step conformal band: wider further out.
    const halfWidth = 0.04 + k * 0.02;
    return {
      horizon_seconds: (k + 1) * 30,
      risk,
      confidence: Math.max(0.35, 0.92 - k * 0.07),
      predicted_stage: hot ? ahead.stage : "Benign",
      risk_lower: Math.max(0, risk - halfWidth),
      risk_upper: Math.min(1, risk + halfWidth),
    };
  });

  return {
    p, i, t, hot, risk, mlRisk, ruleRisk, rulesApplied, shown,
    flows, packets, throughput, telemetryMs, inferenceMs, forecast,
    maxFuture: Math.max(...forecast.map((f) => f.risk)),
    alertLevel: alertLevelFor(shown),
  };
}

/** Input × Gradient attributions: latent dims lead when quiet, rate/volume when flooding. */
function attribution(step: number, hot: boolean) {
  const feats = [
    ...HOST_ATTRS.map((f, i) => ({
      feature: f,
      raw: Math.abs(wobble(step, i + 11)) * (hot ? (f === "byte_rate" || f === "packet_rate" || f === "unique_dst_ports" ? 3.4 : 1.1) : 0.6),
      group: GROUP_OF[f] ?? "General",
    })),
    ...Array.from({ length: 12 }, (_, i) => ({
      feature: `H_emb_${i}`,
      raw: Math.abs(wobble(step, i + 41)) * (hot ? 0.9 : 1.8),
      group: "TGNE Latent",
    })),
  ].sort((a, b) => b.raw - a.raw);

  const total = feats.reduce((s, f) => s + f.raw, 0) || 1;
  const byGroup = new Map<string, number>();
  for (const f of feats) byGroup.set(f.group, (byGroup.get(f.group) ?? 0) + f.raw);

  return {
    available: true,
    method: "Input x Gradient Saliency",
    groups: [...byGroup.entries()]
      .map(([name, v]) => ({ name, percentage: (v / total) * 100 }))
      .sort((a, b) => b.percentage - a.percentage),
    top_features: feats.slice(0, 8).map((f) => ({ feature: f.feature, score: f.raw / total, group: f.group })),
  };
}

function buildCampaign(step: number, now: number): Campaign {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  const nodes: CampaignNode[] = CAMPAIGN_NODES.map((n) => ({
    node_id: n.node_id,
    host_ip: n.host_ip,
    coarse_category: n.coarse_category,
    technique_id: n.technique_id,
    hit_count: n.hit_count,
    max_risk_score: n.max_risk_score,
    mean_confidence: n.mean_confidence,
    provenance: s >= n.at ? "OBSERVED" : "FORECAST",
    start_time: now / 1000 - (s - n.at) * 2,
    end_time: now / 1000 - Math.max(0, s - n.at - 4) * 2,
  }));

  const observed = nodes.filter((n) => n.provenance === "OBSERVED");

  return {
    campaign_id: 1,
    involved_hosts: [...new Set(nodes.map((n) => n.host_ip))],
    root_cause_node_ids: [0],
    nodes,
    edges: CAMPAIGN_EDGES,
    max_risk_score: observed.length ? Math.max(...observed.map((n) => n.max_risk_score)) : 0,
    start_time: now / 1000 - s * 2,
    end_time: now / 1000,
    duration_sec: s * 2,
    attack_techniques: [...new Set(observed.map((n) => n.technique_id))],
    has_forecast_components: nodes.some((n) => n.provenance === "FORECAST"),
    lanes: LANES,
  };
}

/* ── REST fixtures ─────────────────────────────────────────────────────── */

const BASE_STATUS = {
  // A healthy sensor: no frames dropped, every window delivered.
  sensorKernelDrops: 0,
  incompleteWindows: 0,
  windowsMissed: 0,
  mode: "LIVE",
  site_id: "hq-core",
  lab_mode: true,
  model_loaded: true,
  model_meta: MODEL_META,
  sensor_interface: "span0",
  nodes_running: 15,
  total_nodes: 15,
  topology_nodes: ASSETS.length + 1,
  topology_edges: 9,
  workloads_active: true,
  ml_active: true,
};

export const mockFetchStatus = async (): Promise<SystemStatus> =>
  ({
    networkStatus: "running",
    telemetryStatus: "running",
    predictionStatus: "running",
    attackStatus: "none",
    uptime: 14_820,
    lastUpdate: new Date().toISOString(),
    anomalyScore: 4.1,
    threatLevel: "low",
    packetLoss: 0.3,
    latency: 11.2,
    throughput: 38,
    activeConnections: 246,
    ...BASE_STATUS,
  }) as unknown as SystemStatus;

export const mockFetchSite = async (): Promise<SiteInfo> => ({
  name: "HQ-CORE / Enterprise Segment",
  location: "Local",
  timezone: "UTC",
  subnet: "10.0.0.0/16",
  externalIp: ATTACKER,
  description: "Primary enterprise segment — SPAN mirror on the core uplink.",
});

/* ── Pre-seed ──────────────────────────────────────────────────────────── */

/**
 * `n` windows of calm baseline ending now, so the console opens with full
 * charts at rest and the intrusion begins on camera rather than off it.
 *
 * Seeded points use the same `windowState()` the live emitter does, indexed
 * into the benign tail of the cycle — so the seam where the stream takes over
 * is invisible.
 */
export function seedHistory(n = 90): LivePoint[] {
  const now = Date.now();
  const out: LivePoint[] = [];
  for (let k = n; k > 0; k--) {
    const step = RECOVERY_STEP + 4 + (n - k) * 0.35; // drift inside the benign band
    const w = windowState(Math.floor(step));
    const t = now - k * TICK_MS;
    out.push({
      t,
      label: new Date(t).toLocaleTimeString("en-GB", { hour12: false }),
      windowId: -k,
      risk: w.shown,
      maxFutureRisk: w.maxFuture,
      mlRisk: w.mlRisk,
      ruleRisk: w.ruleRisk,
      throughput: w.throughput,
      packets: w.packets,
      flows: w.flows,
      latencyMs: Number((w.telemetryMs + w.inferenceMs).toFixed(2)),
    });
  }
  return out;
}

/**
 * Event backfill — a prior incident that has already closed out.
 *
 * Alerts only enter the stream once live risk crosses the threshold, roughly
 * 20s into the scenario. Without this the Events page reads empty for the
 * first third of a recording.
 */
export function seedEvents(): {
  severity: "info" | "warning" | "error" | "critical";
  category: string;
  source: string;
  destination?: string;
  message: string;
  raw?: string;
  ageMs: number;
}[] {
  return [
    { ageMs: 41_000, severity: "info", category: "SYSTEM", source: "bus", message: "Connected to real-time event bus" },
    { ageMs: 38_000, severity: "info", category: "COMMAND", source: "start_telemetry", message: "Command completed: start_telemetry" },
    { ageMs: 36_000, severity: "info", category: "COMMAND", source: "start_ml", message: "Command completed: start_ml" },
    { ageMs: 33_000, severity: "info", category: "NOMINAL", source: "10.0.2.10", message: "Baseline established — 14 hosts, 31 edges", raw: "hosts=14 edges=31 window=2.0s" },
    {
      ageMs: 28_000, severity: "warning", category: "WARNING", source: "192.168.100.41", destination: "10.0.3.10",
      message: "T1046 Network Service Discovery — Reconnaissance",
      raw: "risk=0.6712 max_future=0.7003 ml=0.6712 rule=— source=model",
    },
    {
      ageMs: 24_000, severity: "warning", category: "ELEVATED", source: "192.168.100.41", destination: "10.0.3.10",
      message: "T1110 Brute Force — Credential Access",
      raw: "risk=0.7844 max_future=0.8319 ml=0.7844 rule=— source=model",
    },
    {
      ageMs: 19_000, severity: "critical", category: "CRITICAL", source: "192.168.100.41", destination: "10.0.3.10",
      message: "T1190 Exploit Public-Facing Application — Initial Access",
      raw: "risk=0.9021 max_future=0.9337 ml=0.9021 rule=— source=model",
    },
    { ageMs: 15_000, severity: "info", category: "COMMAND", source: "mitigate", message: "Mitigation ISOLATE_HOST → 10.0.3.10 recorded" },
    { ageMs: 12_000, severity: "info", category: "NOMINAL", source: "10.0.3.10", message: "Risk returned below threshold after isolation", raw: "risk=0.0912 threshold=0.65" },
    { ageMs: 6_000, severity: "info", category: "SYSTEM", source: "ml", message: "Inference state reset" },
  ];
}

/** Console backfill so the log panes are never blank on the first frame. */
export function seedLog(n = 24): string[] {
  const now = Date.now();
  const lines: string[] = [];
  for (let k = n; k > 0; k--) {
    const ts = new Date(now - k * TICK_MS * 2).toISOString();
    const w = windowState(RECOVERY_STEP + 4 + (n - k));
    lines.push(`[${ts}] sensor INFO: window ${1000 - k} closed — ${w.packets} packets, ${w.flows} flows, ${w.telemetryMs.toFixed(1)}ms`);
    if (k % 4 === 0) lines.push(`[${ts}] ml   INFO: nowcast risk=${w.shown.toFixed(3)} level=${w.alertLevel} θ=${THRESHOLD.toFixed(2)}`);
  }
  return lines;
}

/* ── Topology ──────────────────────────────────────────────────────────── */

function buildTopology(step: number): Topology {
  const w = windowState(step);
  const risk = w.shown;
  const hot = w.hot;

  const nodes: Topology["nodes"] = [
    { id: ATTACKER, label: "external", ip: ATTACKER, type: "internet", status: hot ? "compromised" : "online", x: 400, y: 62 },
    ...ASSETS.map((a, i) => {
      const isTarget = a.ip === TARGET;
      const lateral = hot && risk > 0.8 && (a.ip === "10.0.2.40" || a.ip === "10.0.2.50");
      const nodeRisk = isTarget ? risk : lateral ? risk * 0.7 : risk * 0.2;
      const band = a.zone === "dmz" ? 196 : a.zone === "servers" ? 318 : 404;
      const inBand = ASSETS.filter((x) => x.zone === a.zone);
      const idx = inBand.findIndex((x) => x.ip === a.ip);
      const spread = 640 / (inBand.length + 1);
      return {
        id: a.ip,
        label: a.name,
        ip: a.ip,
        type: "server" as const,
        status: nodeRisk >= 0.8 ? ("compromised" as const) : nodeRisk >= 0.5 ? ("warning" as const) : ("online" as const),
        x: 80 + spread * (idx + 1),
        y: band + (i % 2) * 14,
      };
    }),
  ];

  const edges: Topology["edges"] = [
    {
      id: `${ATTACKER}-${TARGET}`,
      source: ATTACKER,
      target: TARGET,
      status: hot ? (risk > 0.85 ? "saturated" : "suspicious") : "active",
      protocol: "TCP",
      bandwidth: Math.round(hot ? risk * 780 : 6),
      utilization: Math.round(hot ? risk * 84 : 1),
    },
    ...["10.0.2.40", "10.0.2.50", "10.0.2.30", "10.0.2.20"].map((dst) => {
      const lateral = hot && risk > 0.8 && (dst === "10.0.2.40" || dst === "10.0.2.50");
      return {
        id: `${TARGET}-${dst}`,
        source: TARGET,
        target: dst,
        status: lateral ? ("suspicious" as const) : ("active" as const),
        protocol: "TCP",
        bandwidth: 8 + Math.round(Math.abs(wobble(step, dst.length)) * 12),
        utilization: 2,
      };
    }),
    ...["10.0.1.21", "10.0.1.22"].map((src) => ({
      id: `${src}-10.0.2.10`,
      source: src,
      target: "10.0.2.10",
      status: "active" as const,
      protocol: "UDP",
      bandwidth: 3,
      utilization: 1,
    })),
  ];

  return { nodes, edges, lastUpdated: new Date().toISOString() };
}

export const mockFetchTopology = async (): Promise<Topology> => buildTopology(RECOVERY_STEP + 4);

/* ── Incident queue ────────────────────────────────────────────────────── */

const ANALYSTS = ["a.rao", "m.iyer", "s.khan", null];

/**
 * Closed and contained incidents from earlier in the shift.
 *
 * A queue that only ever holds the one live incident reads as a toy. These
 * give the triage view a history to sit against, and their resolution times
 * are what the queue's median-time-to-contain is computed from.
 */
function historicIncidents(now: number): Incident[] {
  const h = (sec: number) => now - sec;
  return [
    {
      id: "INC-0138",
      host: "10.0.1.22",
      hostLabel: "ws-web",
      title: "T1071.001 Web Protocols",
      technique: "T1071.001",
      tactic: "Command and Control",
      severity: "elevated",
      status: "closed",
      opened: h(9_840),
      lastSeen: h(9_120),
      peakRisk: 0.781,
      alertCount: 46,
      assignee: "m.iyer",
      leadTimeSeconds: 8.4,
      campaignId: null,
      notes: [
        { at: h(9_780), text: "Beaconing to an unregistered domain. Escalated." },
        { at: h(9_180), text: "Confirmed false positive — internal telemetry agent update." },
        { at: h(9_120), text: "Closed, no action." },
      ],
    },
    {
      id: "INC-0140",
      host: "10.0.2.30",
      hostLabel: "srv-file",
      title: "T1110 Brute Force",
      technique: "T1110",
      tactic: "Credential Access",
      severity: "warning",
      status: "closed",
      opened: h(6_600),
      lastSeen: h(6_180),
      peakRisk: 0.694,
      alertCount: 212,
      assignee: "a.rao",
      leadTimeSeconds: 5.1,
      campaignId: null,
      notes: [
        { at: h(6_540), text: "SMB auth failures from a single source. Source blocked at the edge." },
        { at: h(6_180), text: "No further attempts. Closed." },
      ],
    },
    {
      id: "INC-0141",
      host: "10.0.2.50",
      hostLabel: "srv-db",
      title: "T1046 Network Service Discovery",
      technique: "T1046",
      tactic: "Reconnaissance",
      severity: "warning",
      status: "contained",
      opened: h(2_460),
      lastSeen: h(1_980),
      peakRisk: 0.712,
      alertCount: 38,
      assignee: "s.khan",
      leadTimeSeconds: 11.7,
      campaignId: null,
      notes: [
        { at: h(2_400), text: "Port sweep against the database segment." },
        { at: h(1_980), text: "Source isolated. Monitoring for recurrence." },
      ],
    },
  ];
}

/**
 * The live incident, opened once the scenario crosses into ELEVATED and
 * escalating with it. Returns an empty list while traffic is nominal, so the
 * queue genuinely fills on camera rather than starting full.
 */
function liveIncident(step: number, now: number): Incident | null {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  const w = windowState(step);

  // Opens when the chain reaches the exploit phase.
  if (s < 26 || s >= CYCLE - 4) return null;

  const openedAt = now - (s - 26) * (TICK_MS / 1000) * 2;
  const contained = s >= RECOVERY_STEP;
  const sev: Incident["severity"] = contained
    ? "warning"
    : w.alertLevel === "CRITICAL"
      ? "critical"
      : w.alertLevel === "ELEVATED"
        ? "elevated"
        : "warning";

  const notes: { at: number; text: string }[] = [
    { at: openedAt, text: "Opened automatically — forecast risk crossed the operating threshold." },
  ];
  if (s >= 38) notes.push({ at: now - (s - 38) * 2, text: "Branch B rollout projects C2 within 8s. Escalated to ELEVATED." });
  if (s >= 50) notes.push({ at: now - (s - 50) * 2, text: "DeepOP decoded an Impact token sequence. Escalated to CRITICAL." });
  if (contained) notes.push({ at: now - (s - RECOVERY_STEP) * 2, text: "Host isolated. Risk returned below threshold." });

  return {
    id: "INC-0142",
    host: TARGET,
    hostLabel: "dmz-web",
    title: w.p.label,
    technique: w.p.technique,
    tactic: w.p.tactic,
    severity: sev,
    status: contained ? "contained" : s >= 38 ? "triaging" : "new",
    opened: openedAt,
    lastSeen: now,
    peakRisk: Math.min(0.96, 0.52 + (s - 26) * 0.012),
    alertCount: Math.max(1, (s - 26) * 7),
    assignee: s >= 38 ? "a.rao" : null,
    leadTimeSeconds: 11.4,
    campaignId: 1,
    notes,
  };
}

function buildIncidents(step: number, nowMs: number): Incident[] {
  const now = nowMs / 1000;
  const live = liveIncident(step, now);
  return [...(live ? [live] : []), ...historicIncidents(now)];
}

/** Seed for first paint, before the stream has produced a window. */
export function seedIncidents(): Incident[] {
  return buildIncidents(RECOVERY_STEP + 4, Date.now());
}

/* ── Offline replay ────────────────────────────────────────────────────── */

/**
 * Built-in captures.
 *
 * Every profile below is a genuinely different shape, not the same curve under
 * a different filename — an escalating intrusion, steady C2 beaconing, bursty
 * credential stuffing, a volumetric flood, and a clean baseline. The clean one
 * matters most: a detector that flags everything is not a detector, and
 * `benign-baseline` scoring zero flagged windows is the fastest way to show
 * the threshold is doing real work.
 */
export interface ReplaySample {
  id: string;
  /** Filename as it would arrive on disk. */
  name: string;
  label: string;
  kind: "pcap" | "csv";
  bytes: number;
  windows: number;
  source: string;
  note: string;
}

export const REPLAY_SAMPLES: ReplaySample[] = [
  {
    id: "infiltration",
    name: "dmz-infiltration-chain.pcap",
    label: "DMZ infiltration chain",
    kind: "pcap",
    bytes: 1_284_096,
    windows: 156,
    source: "CIC-IDS2018 · Infiltration",
    note: "Scan → brute force → exploit → C2 → flood against a single DMZ host.",
  },
  {
    id: "botnet",
    name: "ctu13-neris-capture.binetflow",
    label: "Neris botnet C2",
    kind: "csv",
    bytes: 742_400,
    windows: 92,
    source: "CTU-13 · Scenario 1",
    note: "Sustained IRC beaconing with periodic check-in spikes. No volumetric phase.",
  },
  {
    id: "bruteforce",
    name: "ssh-credential-stuffing.pcap",
    label: "SSH credential stuffing",
    kind: "pcap",
    bytes: 512_000,
    windows: 64,
    source: "CIC-IDS2018 · SSH-Patator",
    note: "Four attempt bursts separated by back-off. Risk is bursty, not monotonic.",
  },
  {
    id: "ddos",
    name: "hoic-volumetric-flood.pcap",
    label: "HOIC volumetric flood",
    kind: "pcap",
    bytes: 2_097_152,
    windows: 128,
    source: "CIC-IDS2018 · DDOS-HOIC",
    note: "Quiet for a third of the capture, then a step change into sustained flood.",
  },
  {
    id: "benign",
    name: "hq-core-baseline.pcap",
    label: "Clean enterprise baseline",
    kind: "pcap",
    bytes: 968_704,
    windows: 112,
    source: "HQ-CORE · business hours",
    note: "Normal office traffic. Expected result: zero windows flagged.",
  },
];

interface Beat {
  risk: number;
  stage: string;
  technique: string | null;
  tactic: string | null;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(0.99, v));
}

/**
 * Risk and technique shape per profile, as a function of position through the
 * capture (0–1). Deterministic, so a sample replays identically every time.
 */
function beatFor(profile: string, u: number, i: number): Beat {
  const jitter = wobble(i, 5) * 0.012;

  switch (profile) {
    case "botnet": {
      // Steady beaconing with a check-in spike every seventh window.
      const beacon = i % 7 === 0 ? 0.17 : 0;
      return {
        risk: clamp01(0.52 + Math.sin(u * Math.PI * 1.4) * 0.09 + beacon + jitter),
        stage: "C2",
        technique: "T1071.001 Web Protocols",
        tactic: "Command and Control",
      };
    }
    case "bruteforce": {
      // Four attempt bursts with back-off between them.
      const phase = (u * 4) % 1;
      const inBurst = phase < 0.45;
      return {
        risk: clamp01((inBurst ? 0.58 + phase * 0.5 : 0.16 + phase * 0.1) + jitter),
        stage: inBurst ? "InitialAccess" : "Recon",
        technique: inBurst ? "T1110 Brute Force" : "T1046 Network Service Discovery",
        tactic: inBurst ? "Credential Access" : "Reconnaissance",
      };
    }
    case "ddos": {
      // Step change into a sustained flood a third of the way through.
      const hot = u > 0.34;
      return {
        risk: clamp01((hot ? 0.9 + Math.sin(u * 30) * 0.03 : 0.05) + jitter),
        stage: hot ? "Impact" : "Benign",
        technique: hot ? "T1498 Network Denial of Service" : null,
        tactic: hot ? "Impact" : null,
      };
    }
    case "benign": {
      // Business-hours traffic. Nothing here should cross the threshold.
      return {
        risk: clamp01(0.045 + Math.abs(wobble(i, 9)) * 0.05),
        stage: "Benign",
        technique: null,
        tactic: null,
      };
    }
    default: {
      // Escalating intrusion — the same phase script the live path walks.
      //
      // The phase risk is the (scripted) model output, lightly damped.
      const w = windowState(Math.floor(u * CYCLE));
      return {
        risk: clamp01(w.risk * 0.96 + jitter),
        stage: w.p.stage,
        technique: w.hot ? w.p.label : null,
        tactic: w.hot ? w.p.tactic : null,
      };
    }
  }
}

/** Builds the report the real `/api/replay` would return for a given profile. */
function scoreCapture(
  filename: string,
  kind: ReplayReport["kind"],
  windows: number,
  profile: string
): ReplayReport {
  const results: ReplayRow[] = [];
  const span = Math.max(1, windows - 1);

  for (let i = 0; i < windows; i++) {
    const beat = beatFor(profile, i / span, i);
    const prev = i > 0 ? beatFor(profile, (i - 1) / span, i - 1).risk : beat.risk;
    const rising = beat.risk >= prev;
    const ex = attribution(i, beat.risk > 0.3);

    const forecast = Array.from({ length: 5 }, (_, k) => ({
      horizon_seconds: (k + 1) * 30,
      risk: clamp01(beat.risk * (rising ? 1 + (k + 1) * 0.03 : 1 - (k + 1) * 0.045)),
    }));

    results.push({
      window: i,
      target: profile === "botnet" ? "10.0.1.21" : profile === "benign" ? "10.0.2.10" : TARGET,
      risk: beat.risk,
      ml_risk: beat.risk,
      alert: beat.risk >= THRESHOLD,
      stage: beat.stage,
      mitre_tactic: beat.tactic,
      mitre_technique: beat.technique,
      forecast,
      top_features: ex.top_features.slice(0, 5).map((f) => ({ feature: f.feature, score: f.score, group: f.group })),
    });
  }

  return {
    filename,
    kind,
    windows_analyzed: results.length,
    flagged_windows: results.filter((r) => r.alert).length,
    rules_disabled: true,
    results,
  };
}

/** Parsing a capture is not instant; a result in 0ms reads as fake. */
function parseDelay(windows: number): Promise<void> {
  return new Promise((r) => setTimeout(r, 650 + Math.min(1500, windows * 7)));
}

/** Runs one of the built-in captures. */
export async function mockReplaySample(id: string): Promise<ReplayReport> {
  const sample = REPLAY_SAMPLES.find((s) => s.id === id) ?? REPLAY_SAMPLES[0];
  await parseDelay(sample.windows);
  return scoreCapture(sample.name, sample.kind, sample.windows, sample.id);
}

/**
 * Scores an uploaded capture window by window, mirroring `POST /api/replay`.
 *
 * The real endpoint forces the SOC rule layer off for offline analysis, so
 * `risk` here is `ml_risk` — no rule floor, no technique overwrite. Window
 * count scales with file size the way a real parse would.
 */
export async function mockReplay(file: File, maxWindows = 200): Promise<ReplayReport> {
  const suffix = file.name.toLowerCase().split(".").pop() ?? "";
  const kind: ReplayReport["kind"] = suffix === "csv" || suffix === "binetflow" ? "csv" : "pcap";

  // ~1 window per 8KB, bounded by the caller's limit. A 1.2MB pcap lands
  // around 150 windows, the order the real parser produces.
  const windows = Math.max(12, Math.min(maxWindows, Math.round(file.size / 8192) || 48));

  // A dropped file whose name matches a known corpus gets that profile, so
  // dragging in a real CTU-13 export behaves the way its sample does.
  const n = file.name.toLowerCase();
  const profile =
    n.includes("neris") || n.includes("binetflow")
      ? "botnet"
      : n.includes("patator") || n.includes("brute") || n.includes("ssh")
        ? "bruteforce"
        : n.includes("hoic") || n.includes("loic") || n.includes("ddos") || n.includes("dos")
          ? "ddos"
          : n.includes("benign") || n.includes("baseline") || n.includes("normal")
            ? "benign"
            : "infiltration";

  await parseDelay(windows);
  return scoreCapture(file.name, kind, windows, profile);
}

/* ── Scenario control ──────────────────────────────────────────────────── */

/** Live instance, so command/mitigation fixtures can steer the scenario. */
let active: MockWebSocket | null = null;

export const mockSendCommand = async (command: string): Promise<void> => {
  active?.note(`command ${command} accepted`);
  if (command === "reset_environment" || command === "stop_attack") active?.jumpTo(RECOVERY_STEP);
  if (command === "start_attack") active?.jumpTo(0);
};

/**
 * Mitigation collapses the scenario into its recovery phase. On camera this
 * reads as cause and effect: the operator isolates the host and risk drops —
 * worth more in a demo than any static panel.
 */
export const mockSendMitigate = async (payload: MitigationPayload): Promise<void> => {
  active?.note(`mitigation ${payload.action}${payload.target ? ` → ${payload.target}` : ""} applied`);
  if (!String(payload.action).startsWith("CLEAR")) active?.jumpTo(RECOVERY_STEP);
};

/* ── Live stream ───────────────────────────────────────────────────────── */

export class MockWebSocket {
  private handlers: WSHandlers;
  private timer: ReturnType<typeof setInterval> | undefined;
  private opening: ReturnType<typeof setTimeout> | undefined;
  private closed = false;
  private step = 0;

  constructor(handlers: WSHandlers) {
    this.handlers = handlers;
    active = this;
    // Guarded against close-during-connect: React StrictMode mounts effects
    // twice in dev, so the first socket is closed before this fires. Without
    // the flag its interval would still start and the scenario would run at
    // double speed.
    this.opening = setTimeout(() => {
      if (this.closed) return;
      this.handlers.onOpen?.();
      this.emit();
      this.timer = setInterval(() => this.emit(), TICK_MS);
    }, 200);
  }

  /** Jump the scenario to a specific window — used by the command fixtures. */
  jumpTo(step: number) {
    this.step = step;
  }

  /** Push a synthetic console line (command + mitigation acknowledgements). */
  note(text: string) {
    this.handlers.onCommandOutput?.({
      command: "ui",
      line: `[${new Date().toISOString()}] ctrl INFO: ${text}`,
    });
  }

  private emit() {
    const step = this.step++;
    const now = new Date().toISOString();
    const w = windowState(step);
    const explainability = attribution(step, w.hot);

    this.handlers.onPrediction?.({
      type: "prediction",
      mode: "LIVE",
      timestamp: now,
      wall_clock: now,
      model: MODEL_META,
      state: {
        window_id: step,
        sequence_ready: true,
        packet_count: w.packets,
        active_flows: w.flows,
        pipeline_latency_ms: w.telemetryMs,
        buffer_length: 5,
      },
      prediction: {
        risk: w.shown,
        max_future_risk: w.maxFuture,
        hazard_score: Math.min(0.99, w.maxFuture * 0.92),
        malicious_confidence: Math.min(0.99, 0.42 + w.shown * 0.55),
        precursor_confidence: Math.min(0.99, 0.3 + w.shown * 0.4),
        alert: w.shown >= THRESHOLD || w.maxFuture >= THRESHOLD,
        alert_level: w.alertLevel,
        threshold: THRESHOLD,
        predicted_stage: w.p.stage,
        mitre_tactic: w.p.tactic,
        mitre_technique: w.p.label,
        mitre_tactic_id: w.p.tacticId,
        mitre_description: w.p.desc,
        technique_confidence: Math.min(0.99, 0.5 + w.shown * 0.45),
        stage_probabilities: Object.fromEntries(
          ["Benign", "Recon", "InitialAccess", "C2", "Impact"].map((s) => [
            s,
            s === w.p.stage
              ? Math.min(0.95, 0.5 + w.shown * 0.4)
              : Math.max(0.01, (1 - w.shown) * 0.22 + Math.abs(wobble(step, s.length)) * 0.05),
          ])
        ),
        stage_provenance: {
          TGNE: "h_emb(12)",
          BRANCH_A: "risk+technique",
          BRANCH_B: "H_hat trajectory",
          DEEPOP: "technique sequence",
        },
        ml_risk: w.mlRisk,
        ml_technique: w.hot ? w.p.label : "Benign",
        rule_risk: w.ruleRisk,
        rule_technique: null,
        rules_applied: w.rulesApplied,
        risk_source: "model",
        // Fields adapter.ts normally attaches on the real path
        value: w.shown,
        confidence: Math.min(0.99, 0.42 + w.shown * 0.55),
        horizon: 16,
        model: "cyberworld Ensemble",
        branch_a_risk: w.shown,
        branch_b_risk: w.maxFuture,
        explainability,
        signals: explainability.top_features.map((f) => ({
          name: f.feature,
          weight: f.score,
          direction: "neutral",
          value: f.group,
        })),
      },
      forecast: w.forecast,
      explainability,
      latency: {
        telemetry_ms: w.telemetryMs,
        inference_ms: w.inferenceMs,
        total_ms: Number((w.telemetryMs + w.inferenceMs).toFixed(2)),
      },
      early_warning:
        w.hot && w.shown > 0.5
          ? {
              is_alert: true,
              alert_timestamp: now,
              actual_milestone_timestamp: null,
              lead_time_seconds: Number((6 + Math.abs(wobble(step, 31)) * 7).toFixed(1)),
              target_milestone_desc: `first observed ${w.p.stage} milestone`,
            }
          : null,
      attack_active: w.hot,
      attack_phase: w.p.stage,
      focus_ips: w.hot ? [ATTACKER, TARGET] : [],
      focus_edges: w.hot ? [{ src: ATTACKER, dst: TARGET }] : [],
      target_ip: TARGET,
      throughput: w.throughput,
      // Demo-only: correlation/ is not wired into control_backend yet.
      campaign: buildCampaign(step, Date.now()),
      incidents: buildIncidents(step, Date.now()),
    } as never);

    this.handlers.onTopologyUpdate?.(buildTopology(step));

    this.handlers.onSystemStatus?.({
      networkStatus: "running",
      telemetryStatus: "running",
      predictionStatus: "running",
      attackStatus: w.hot ? "active" : "none",
      uptime: 14_820 + step * 2,
      lastUpdate: now,
      anomalyScore: Math.round(Math.max(w.shown, w.maxFuture) * 1000) / 10,
      threatLevel: w.alertLevel.toLowerCase(),
      packetLoss: Number((w.hot ? w.shown * 4.1 : 0.3).toFixed(2)),
      latency: Number((w.telemetryMs + w.inferenceMs).toFixed(2)),
      throughput: w.throughput,
      activeConnections: w.flows,
      ...BASE_STATUS,
    } as unknown as SystemStatus);

    /* Console traffic so the log panes stay alive under the charts. */
    if (step % 3 === 0) {
      this.handlers.onCommandOutput?.({
        command: "sensor",
        line: `[${now}] sensor INFO: window ${step} closed — ${w.packets} packets, ${w.flows} flows, ${w.telemetryMs.toFixed(1)}ms`,
      });
    }
    if (w.hot && step % 4 === 0) {
      this.handlers.onCommandOutput?.({
        command: "ml",
        line: `[${now}] ml   ${w.alertLevel === "CRITICAL" ? "CRITICAL" : "WARN"}: ${w.p.label} risk=${w.shown.toFixed(3)} target=${TARGET}`,
      });
    }
  }

  close() {
    this.closed = true;
    clearTimeout(this.opening);
    clearInterval(this.timer);
    if (active === this) active = null;
    // Deliberately silent. A client-initiated close is a teardown, not a
    // dropped link — firing onClose here would push a red "connection lost"
    // line into the console on every StrictMode remount, which is the first
    // thing a viewer would see.
  }
}
