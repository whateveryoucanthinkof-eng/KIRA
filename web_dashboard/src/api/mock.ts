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

import type { SystemStatus, Topology, TopologyNode, SiteInfo, MitigationPayload, WSHandlers } from "./types";
import type { Campaign, CampaignNode } from "../types/campaign";
import type { ReplayReport, ReplayRow } from "../types/replay";
import type { Incident } from "../types/incident";
import type { FlowFlags, FlowProtocol, FlowRecord, StateDim } from "../types/evidence";
import type { ForecastBranch } from "../types/forecast";
import type { AttentionMatrix, AttentionRole, SaliencyTerm } from "../types/attention";

/* ── Site (config/sites/containerlab-enterprise.yaml) ──────────────────── */

interface Asset {
  ip: string;
  name: string;
  zone: "dmz" | "servers" | "users";
  /** Core hosts carry the scenario; background hosts are the segment's population. */
  tier: "core" | "background";
}

const CORE_ASSETS: Asset[] = [
  { ip: "10.0.3.10", name: "dmz-web", zone: "dmz", tier: "core" },
  { ip: "10.0.2.40", name: "srv-app", zone: "servers", tier: "core" },
  { ip: "10.0.2.30", name: "srv-file", zone: "servers", tier: "core" },
  { ip: "10.0.2.20", name: "srv-id", zone: "servers", tier: "core" },
  { ip: "10.0.2.10", name: "srv-dns", zone: "servers", tier: "core" },
  { ip: "10.0.2.50", name: "srv-db", zone: "servers", tier: "core" },
  { ip: "10.0.1.11", name: "ws-office", zone: "users", tier: "core" },
  { ip: "10.0.1.12", name: "ws-web", zone: "users", tier: "core" },
  { ip: "10.0.1.13", name: "ws-file", zone: "users", tier: "core" },
  { ip: "10.0.1.14", name: "ws-app", zone: "users", tier: "core" },
];

/** The rest of the segment: DMZ services, infrastructure, and the desk floor. */
const BACKGROUND_ASSETS: Asset[] = [
  { ip: "10.0.3.20", name: "dmz-mail", zone: "dmz", tier: "background" },
  { ip: "10.0.3.30", name: "dmz-vpn", zone: "dmz", tier: "background" },
  { ip: "10.0.2.60", name: "srv-backup", zone: "servers", tier: "background" },
  { ip: "10.0.2.70", name: "srv-mon", zone: "servers", tier: "background" },
  { ip: "10.0.2.80", name: "srv-print", zone: "servers", tier: "background" },
  ...Array.from({ length: 20 }, (_, i): Asset => ({
    ip: `10.0.1.${15 + i}`,
    name: `hq-ws-${String(15 + i).padStart(3, "0")}`,
    zone: "users",
    tier: "background",
  })),
];

const ASSETS: Asset[] = [...CORE_ASSETS, ...BACKGROUND_ASSETS];

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
  { until: 14, stage: "Recon", technique: "T1046", risk: [0.45, 0.75], label: "T1046 Network Service Discovery", tactic: "Reconnaissance", tacticId: "TA0043", desc: "Sweeping the DMZ for reachable services." },
  { until: 26, stage: "InitialAccess", technique: "T1110", risk: [0.55, 0.80], label: "T1110 Brute Force", tactic: "Credential Access", tacticId: "TA0006", desc: "Repeated authentication attempts against the exposed web host." },
  { until: 38, stage: "InitialAccess", technique: "T1190", risk: [0.78, 0.92], label: "T1190 Exploit Public-Facing Application", tactic: "Initial Access", tacticId: "TA0001", desc: "Exploit attempts against the public-facing application." },
  { until: 50, stage: "C2", technique: "T1071", risk: [0.86, 0.95], label: "T1071 Application Layer Protocol", tactic: "Command and Control", tacticId: "TA0011", desc: "Beaconing consistent with application-layer command and control." },
  { until: RECOVERY_STEP, stage: "Impact", technique: "T1498", risk: [0.92, 0.98], label: "T1498 Network Denial of Service", tactic: "Impact", tacticId: "TA0040", desc: "Volumetric flood against the target host." },
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

/**
 * Alternate futures per phase: the three continuations the forecast weighs
 * from the present, with the path each would take through the lab. Keyed by
 * PHASES index; the C2 phase splits once the lateral hop is observed.
 */
type BranchTemplate = Omit<ForecastBranch, "id" | "probability" | "peak_risk" | "hops"> & { p: number; hops: string[] };

const BRANCHES: BranchTemplate[][] = [
  [
    { kind: "escalation", label: "Brute-force the exposed web login", stage: "InitialAccess", technique: "T1110", p: 0.62, confidence: 0.71, horizon_seconds: 6, hops: [ATTACKER, TARGET], packets: 9_800, bytes: 1_900_000 },
    { kind: "pivot", label: "Probe the VPN gateway", stage: "InitialAccess", technique: "T1595", p: 0.26, confidence: 0.58, horizon_seconds: 10, hops: [ATTACKER, "10.0.3.30"], packets: 3_100, bytes: 610_000 },
    { kind: "backoff", label: "Scan ends without follow-up", stage: "Benign", technique: "—", p: 0.12, confidence: 0.66, horizon_seconds: 4, hops: [ATTACKER], packets: 420, bytes: 61_000 },
  ],
  [
    { kind: "escalation", label: "Exploit the public-facing application", stage: "InitialAccess", technique: "T1190", p: 0.66, confidence: 0.76, horizon_seconds: 6, hops: [ATTACKER, TARGET], packets: 12_600, bytes: 3_400_000 },
    { kind: "pivot", label: "Reuse harvested credentials on the VPN", stage: "InitialAccess", technique: "T1110", p: 0.23, confidence: 0.61, horizon_seconds: 12, hops: [ATTACKER, "10.0.3.30", "10.0.2.20"], packets: 2_400, bytes: 520_000 },
    { kind: "backoff", label: "Account lockout ends the attempt", stage: "Benign", technique: "—", p: 0.11, confidence: 0.69, horizon_seconds: 4, hops: [ATTACKER], packets: 380, bytes: 52_000 },
  ],
  [
    { kind: "escalation", label: "Establish C2 from the web host", stage: "C2", technique: "T1071", p: 0.7, confidence: 0.82, horizon_seconds: 6, hops: [TARGET, ATTACKER], packets: 6_200, bytes: 940_000 },
    { kind: "pivot", label: "Pivot to the application tier", stage: "LateralMovement", technique: "T1021", p: 0.21, confidence: 0.64, horizon_seconds: 12, hops: [TARGET, "10.0.2.40"], packets: 4_100, bytes: 1_300_000 },
    { kind: "backoff", label: "Exploit fails, session drops", stage: "Benign", technique: "—", p: 0.09, confidence: 0.7, horizon_seconds: 4, hops: [TARGET, ATTACKER], packets: 310, bytes: 44_000 },
  ],
  [
    { kind: "escalation", label: "Exfiltrate via HTTP POST over the C2 channel", stage: "Exfiltration", technique: "T1020", p: 0.74, confidence: 0.84, horizon_seconds: 8, hops: ["10.0.2.50", TARGET, ATTACKER], packets: 18_400, bytes: 22_600_000 },
    { kind: "pivot", label: "Lateral movement to the identity server", stage: "LateralMovement", technique: "T1021", p: 0.19, confidence: 0.69, horizon_seconds: 12, hops: [TARGET, "10.0.2.40", "10.0.2.20"], packets: 5_300, bytes: 2_100_000 },
    { kind: "backoff", label: "Beacon backs off", stage: "Benign", technique: "—", p: 0.07, confidence: 0.72, horizon_seconds: 4, hops: [TARGET, ATTACKER], packets: 300, bytes: 45_000 },
  ],
  [
    { kind: "escalation", label: "Sustain the flood against the web tier", stage: "Impact", technique: "T1498", p: 0.64, confidence: 0.9, horizon_seconds: 4, hops: [ATTACKER, TARGET], packets: 142_000, bytes: 88_000_000 },
    { kind: "pivot", label: "Exfiltrate under cover of the flood", stage: "Exfiltration", technique: "T1020", p: 0.24, confidence: 0.66, horizon_seconds: 10, hops: ["10.0.2.50", TARGET, ATTACKER], packets: 16_900, bytes: 19_800_000 },
    { kind: "backoff", label: "Flood subsides", stage: "Benign", technique: "—", p: 0.12, confidence: 0.63, horizon_seconds: 6, hops: [ATTACKER, TARGET], packets: 2_200, bytes: 310_000 },
  ],
  [
    { kind: "backoff", label: "Contained — no onward activity", stage: "Benign", technique: "—", p: 0.86, confidence: 0.81, horizon_seconds: 4, hops: [TARGET], packets: 0, bytes: 0 },
    { kind: "escalation", label: "Re-entry through the web tier", stage: "InitialAccess", technique: "T1190", p: 0.1, confidence: 0.44, horizon_seconds: 14, hops: [ATTACKER, TARGET], packets: 7_400, bytes: 1_800_000 },
    { kind: "pivot", label: "Dormant foothold on the app tier", stage: "LateralMovement", technique: "T1021", p: 0.04, confidence: 0.38, horizon_seconds: 16, hops: ["10.0.2.40", "10.0.2.20"], packets: 900, bytes: 210_000 },
  ],
];

/** Once the pivot to srv-app is observed, the identity-server branch firms up. */
const LATERAL_OBSERVED = 46;

const HOST_NAMES: Record<string, string> = { [ATTACKER]: "external" };

function branchesAt(step: number, w: ReturnType<typeof windowState>): ForecastBranch[] {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  let templates = BRANCHES[w.i];
  if (w.i === 3 && s >= LATERAL_OBSERVED) {
    templates = templates.map((t) =>
      t.kind === "escalation" ? { ...t, p: 0.58 } : t.kind === "pivot" ? { ...t, p: 0.34, horizon_seconds: 8 } : { ...t, p: 0.08 }
    );
  }
  const raw = templates.map((t, k) => Math.max(0.01, t.p + wobble(step, 40 + k) * 0.02));
  const total = raw.reduce((a, b) => a + b, 0);
  const name = (ip: string) => HOST_NAMES[ip] ?? ASSETS.find((a) => a.ip === ip)?.name ?? ip;

  return templates
    .map((t, k) => {
      const peak =
        t.kind === "backoff"
          ? w.shown * 0.35
          : t.kind === "escalation"
            ? Math.min(0.99, Math.max(w.maxFuture, 0.3) + 0.04)
            : Math.min(0.97, Math.max(w.maxFuture, 0.25) * 0.9);
      const scale = 1 + wobble(step, 50 + k) * 0.04;
      return {
        kind: t.kind,
        label: t.label,
        stage: t.stage,
        technique: t.technique,
        probability: raw[k] / total,
        confidence: Math.min(0.99, t.confidence + wobble(step, 60 + k) * 0.015),
        horizon_seconds: t.horizon_seconds,
        hops: t.hops.map((ip) => ({ ip, name: name(ip) })),
        packets: Math.round(t.packets * scale),
        bytes: Math.round(t.bytes * scale),
        peak_risk: Number(peak.toFixed(3)),
      };
    })
    .sort((a, b) => b.probability - a.probability)
    .map((b, k) => ({ ...b, id: (["A", "B", "C"] as const)[k] }));
}

/* ── TGNE temporal attention ──────────────────────────────────────────── */

/** The hosts the attention view follows: the attacker, the DMZ web tier, servers, three desks. */
const ATTN_HOSTS: { ip: string; role: AttentionRole }[] = [
  { ip: ATTACKER, role: "external" },
  { ip: TARGET, role: "dmz" },
  { ip: "10.0.2.40", role: "server" },
  { ip: "10.0.2.50", role: "server" },
  { ip: "10.0.2.20", role: "server" },
  { ip: "10.0.2.30", role: "server" },
  { ip: "10.0.2.10", role: "server" },
  { ip: "10.0.1.11", role: "workstation" },
  { ip: "10.0.1.12", role: "workstation" },
  { ip: "10.0.1.14", role: "workstation" },
];

/** data_unification/tgne_features.py:EDGE_FEATURE_NAMES, then the time encoding and neighbour memory. */
const KEY_INPUTS: { feature: string; group: SaliencyTerm["group"] }[] = [
  ...[
    "log1p_fwd_bytes",
    "log1p_bwd_bytes",
    "log1p_fwd_packets",
    "log1p_bwd_packets",
    "duration_norm_300s",
    "log1p_byte_rate_norm",
    "log1p_packet_rate_norm",
    "is_tcp",
    "is_udp",
    "is_icmp",
    "dst_port_log_norm",
    "directional_flow_asymmetry",
  ].map((feature) => ({ feature, group: "edge" as const })),
  { feature: "Δt time encoding", group: "time" },
  { feature: "neighbour memory h(t)", group: "memory" },
];

/** Which inputs drive the logit for each kind of edge, as signed shares. */
const SALIENCY_TEMPLATES: Record<string, Record<string, number>> = {
  benign: { "neighbour memory h(t)": 0.38, "Δt time encoding": 0.18, log1p_bwd_bytes: 0.1, log1p_fwd_bytes: 0.08, duration_norm_300s: 0.06, dst_port_log_norm: -0.06, log1p_packet_rate_norm: 0.04, is_tcp: 0.04 },
  scan: { dst_port_log_norm: 0.32, directional_flow_asymmetry: 0.22, log1p_packet_rate_norm: 0.12, is_tcp: 0.1, log1p_bwd_packets: -0.14, duration_norm_300s: -0.12, "neighbour memory h(t)": 0.08, "Δt time encoding": 0.06 },
  brute: { log1p_fwd_packets: 0.28, log1p_packet_rate_norm: 0.24, log1p_bwd_bytes: 0.1, dst_port_log_norm: 0.06, duration_norm_300s: -0.08, "neighbour memory h(t)": 0.14, "Δt time encoding": 0.1 },
  exploit: { log1p_fwd_bytes: 0.3, log1p_byte_rate_norm: 0.22, directional_flow_asymmetry: 0.16, dst_port_log_norm: 0.05, duration_norm_300s: -0.05, "neighbour memory h(t)": 0.14, "Δt time encoding": 0.08 },
  c2: { duration_norm_300s: 0.26, "Δt time encoding": 0.24, log1p_bwd_bytes: 0.1, log1p_fwd_packets: 0.08, is_tcp: 0.06, log1p_byte_rate_norm: -0.12, "neighbour memory h(t)": 0.18 },
  lateral: { "neighbour memory h(t)": 0.26, dst_port_log_norm: 0.22, log1p_fwd_bytes: 0.16, "Δt time encoding": 0.1, is_tcp: 0.08, directional_flow_asymmetry: 0.06, log1p_packet_rate_norm: -0.06 },
  exfil: { log1p_fwd_bytes: 0.34, directional_flow_asymmetry: 0.22, log1p_byte_rate_norm: 0.2, duration_norm_300s: 0.08, "neighbour memory h(t)": 0.1, log1p_bwd_bytes: -0.08 },
  flood: { log1p_packet_rate_norm: 0.34, log1p_fwd_packets: 0.26, directional_flow_asymmetry: 0.18, is_udp: 0.08, "Δt time encoding": 0.1, "neighbour memory h(t)": 0.12, duration_norm_300s: -0.1, log1p_bwd_packets: -0.06 },
};

/** Ordinary traffic: how strongly each role attends to each other role. */
function baseAffinity(a: { ip: string; role: AttentionRole }, b: { ip: string; role: AttentionRole }): number {
  const svc: Record<string, number> = { "10.0.2.10": 1.3, "10.0.2.30": 1.0, "10.0.2.40": 0.9, "10.0.2.20": 0.6, "10.0.2.50": -1.2 };
  if (a.role === "external") return b.ip === TARGET ? 1.0 : -1.2;
  if (b.role === "external") return a.ip === TARGET ? -0.3 : -1.6;
  if (a.role === "workstation") return b.role === "workstation" ? -0.8 : b.role === "dmz" ? -0.2 : (svc[b.ip] ?? 0);
  if (a.role === "dmz") return b.ip === "10.0.2.40" ? 1.2 : b.ip === "10.0.2.10" ? 0.5 : -0.8;
  // Servers.
  const pair = `${a.ip}>${b.ip}`;
  const known: Record<string, number> = {
    "10.0.2.40>10.0.2.50": 1.3, "10.0.2.50>10.0.2.40": 1.5, "10.0.2.40>10.0.2.20": 0.9, "10.0.2.40>10.0.3.10": 1.0,
    "10.0.2.20>10.0.2.40": 0.8, "10.0.2.30>10.0.2.10": 0.4, "10.0.2.10>10.0.2.40": 0.5,
  };
  if (known[pair] != null) return known[pair];
  if (b.role === "workstation") return a.ip === "10.0.2.30" ? 1.1 : a.ip === "10.0.2.20" ? 0.8 : a.ip === "10.0.2.10" ? 0.7 : 0.2;
  return b.ip === "10.0.2.10" ? 0.6 : -0.5;
}

/** Attack edges active in a window: [from, to, template, logit boost]. */
function attackEdges(step: number, w: ReturnType<typeof windowState>): [string, string, string, number][] {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  const k = 0.55 + 0.45 * w.t;
  const APP = "10.0.2.40", DB = "10.0.2.50", ID = "10.0.2.20";
  switch (w.i) {
    case 0: return [[ATTACKER, TARGET, "scan", 1.6 * k], [TARGET, ATTACKER, "scan", 1.2 * k]];
    case 1: return [[ATTACKER, TARGET, "brute", 2.0 * k], [TARGET, ATTACKER, "brute", 2.0 * k]];
    case 2: return [[ATTACKER, TARGET, "exploit", 2.3 * k], [TARGET, ATTACKER, "exploit", 2.4 * k]];
    case 3: {
      const e: [string, string, string, number][] = [[TARGET, ATTACKER, "c2", 3.6 * k], [ATTACKER, TARGET, "c2", 2.2 * k]];
      if (s >= LATERAL_OBSERVED) e.push([TARGET, APP, "lateral", 1.6], [APP, TARGET, "lateral", 2.4], [APP, ID, "lateral", 1.5]);
      return e;
    }
    case 4: {
      const e: [string, string, string, number][] = [[ATTACKER, TARGET, "flood", 2.6], [TARGET, ATTACKER, "flood", 4.3], [APP, TARGET, "flood", 1.2]];
      if (s >= 52) e.push([DB, TARGET, "exfil", 3.6], [TARGET, DB, "exfil", 1.0]);
      return e;
    }
    default: return [];
  }
}

/** Softmax over a host's temporal neighbours; `skip` holds itself and hosts absent this window. */
function softmaxRow(logits: number[], skip: Set<number>): number[] {
  const live = logits.filter((_, j) => !skip.has(j));
  if (!live.length) return logits.map(() => 0);
  const m = Math.max(...live);
  const ex = logits.map((v, j) => (skip.has(j) ? 0 : Math.exp(v - m)));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / z);
}

const ATTN_BASELINE = -0.4;

function attentionAt(step: number, w: ReturnType<typeof windowState>): AttentionMatrix {
  const n = ATTN_HOSTS.length;
  const name = (ip: string) => HOST_NAMES[ip] ?? ASSETS.find((a) => a.ip === ip)?.name ?? ip;
  const boosts = new Map(attackEdges(step, w).map(([a, b, t, v]) => [`${a}>${b}`, { t, v }]));
  // The attacker is only a temporal neighbour while it is present and fresh.
  const atk = attackerPresence(step);
  const absent = new Set<number>(atk.present && !atk.stale ? [] : [0]);

  const logits = ATTN_HOSTS.map((a, i) =>
    ATTN_HOSTS.map((b, j) => {
      if (i === j) return 0;
      const boost = boosts.get(`${a.ip}>${b.ip}`)?.v ?? 0;
      return baseAffinity(a, b) + boost + wobble(step, 90 + i * 11 + j) * 0.12;
    })
  );
  const heads = [0.22, -0.22].map((sign) =>
    logits.map((row, i) =>
      absent.has(i) ? row.map(() => 0) : softmaxRow(row.map((v, j) => v + sign * wobble(step + 3, 70 + i * 7 + j)), new Set([i, ...absent]))
    )
  );
  const alpha = heads[0].map((row, i) => row.map((v, j) => (i === j ? 0 : (v + heads[1][i][j]) / 2)));

  const saliency = ATTN_HOSTS.map((a, i) =>
    ATTN_HOSTS.map((b, j): SaliencyTerm[] => {
      if (i === j || absent.has(i) || absent.has(j)) return [];
      const kind = boosts.get(`${a.ip}>${b.ip}`)?.t ?? "benign";
      const tpl = SALIENCY_TEMPLATES[kind];
      const raw = KEY_INPUTS.map((inp, q) => (tpl[inp.feature] ?? 0) + wobble(step, 200 + i * 31 + j * 7 + q) * 0.012);
      const total = raw.reduce((x, y) => x + y, 0) || 1;
      const span = logits[i][j] - ATTN_BASELINE;
      return KEY_INPUTS.map((inp, q) => {
        const share = raw[q];
        const value = Math.max(0.01, Math.min(0.99, share > 0 ? 0.18 + share * 2.1 : 0.12 + Math.abs(wobble(step, 300 + q)) * 0.1));
        return { feature: inp.feature, group: inp.group, value: Number(value.toFixed(3)), contribution: Number(((share / total) * span).toFixed(4)) };
      }).sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution));
    })
  );

  const r = (m: number[][]) => m.map((row) => row.map((v) => Number(v.toFixed(4))));
  return {
    layer: "TGNE · TemporalAttentionLayer",
    heads: 2,
    neighbors: 20,
    nodes: ATTN_HOSTS.map((h) => ({ ip: h.ip, name: name(h.ip), role: h.role })),
    alpha: r(alpha),
    alpha_heads: heads.map(r),
    logits: r(logits),
    baseline_logit: ATTN_BASELINE,
    saliency,
  };
}

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
  name: "K.I.R.A. dual-branch + DeepOP",
  version: "1.1.0-live-retrained",
  feature_count: 27,
  history_steps: 5,
  window_seconds: 2.0,
  forecast_steps: 8,
  checkpoint: "saved_models/branch_a/branch_a_lstm.pt",
  threshold: THRESHOLD,
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

  // The SOC rule layer engages once external traffic is present — mirrors
  // model_adapter.py:585, so the provenance strip has something honest to show.
  // ml_risk stays the discounted model output (lower), while the rule layer
  // asserts roughly the phase risk, so the shown risk tracks the stage's real
  // severity band — the colour ladder (amber → red) matches the kill chain
  // rather than being pinned just under the critical line by the discount.
  const mlRisk = Math.max(0, Math.min(0.99, risk * (hot ? 0.72 : 1) + wobble(step, 3) * 0.03));
  const rulesApplied = hot && risk > 0.35;
  const ruleRisk = rulesApplied ? Math.min(0.98, Math.max(risk + 0.02, mlRisk)) : null;
  const shown = rulesApplied && ruleRisk != null ? Math.max(mlRisk, ruleRisk) : mlRisk;

  const flows = Math.round(hot ? 260 + shown * 1700 : 240 + wobble(step, 13) * 40);
  const packets = Math.round(hot ? 900 + shown * 14_000 : 700 + wobble(step, 17) * 180);
  const throughput = Number((hot ? 38 + shown * 690 : 36 + wobble(step, 19) * 7).toFixed(2));
  const telemetryMs = Number((3.1 + Math.abs(wobble(step, 23)) * 1.6).toFixed(2));
  const inferenceMs = Number((7.4 + Math.abs(wobble(step, 29)) * 3.2).toFixed(2));

  // K=8 rollout, 2s per step, confidence decaying with horizon.
  const forecast = Array.from({ length: 8 }, (_, k) => {
    const drift = hot ? 1 + (k + 1) * 0.035 : 1 - (k + 1) * 0.06;
    // Walk the forecast forward through the kill chain rather than repeating
    // the current stage — this is what the timeline's forecast cells render.
    const ahead = PHASES[Math.min(PHASES.length - 2, i + (k < 3 ? 0 : k < 6 ? 1 : 2))];
    return {
      horizon_seconds: (k + 1) * 2,
      risk: Math.max(0, Math.min(0.99, shown * drift + wobble(step + k, 7) * 0.015)),
      confidence: Math.max(0.35, 0.92 - k * 0.07),
      predicted_stage: hot ? ahead.stage : "Benign",
    };
  });

  return {
    p, i, t, hot, risk, mlRisk, ruleRisk, rulesApplied, shown,
    flows, packets, throughput, telemetryMs, inferenceMs, forecast,
    maxFuture: Math.max(...forecast.map((f) => f.risk)),
    alertLevel: alertLevelFor(shown),
  };
}

/* ── State vector + attribution ───────────────────────────────────────── */

/**
 * Host-attribute values per kill-chain stage at full intensity. Baseline is
 * ordinary business traffic; each stage pulls the attributes that stage
 * actually moves — a scan inflates unique_dst_ports and starves bwd_packets,
 * a flood saturates packet_rate and flow_count.
 */
const ATTR_BASELINE: Record<string, number> = {
  flow_count: 0.22, fwd_bytes: 0.18, bwd_bytes: 0.2, total_bytes: 0.19, fwd_packets: 0.17,
  bwd_packets: 0.19, total_packets: 0.18, unique_peers: 0.15, unique_dst_ports: 0.08,
  tcp_ratio: 0.82, udp_ratio: 0.18, avg_duration: 0.46, byte_rate: 0.16, packet_rate: 0.15, peer_density: 0.12,
};

const ATTR_BY_STAGE: Record<string, Partial<Record<string, number>>> = {
  Recon: { unique_dst_ports: 0.91, flow_count: 0.58, bwd_packets: 0.07, avg_duration: 0.04, packet_rate: 0.41, tcp_ratio: 0.97, udp_ratio: 0.03 },
  InitialAccess: { flow_count: 0.52, fwd_bytes: 0.44, bwd_bytes: 0.38, total_bytes: 0.41, avg_duration: 0.12, packet_rate: 0.38, byte_rate: 0.34 },
  Execution: { fwd_bytes: 0.36, avg_duration: 0.28, byte_rate: 0.29 },
  C2: { avg_duration: 0.86, fwd_bytes: 0.29, bwd_bytes: 0.33, flow_count: 0.31, unique_peers: 0.08, peer_density: 0.06 },
  LateralMovement: { unique_peers: 0.42, peer_density: 0.47, flow_count: 0.36 },
  Exfiltration: { fwd_bytes: 0.88, total_bytes: 0.81, byte_rate: 0.72 },
  Impact: {
    flow_count: 0.97, fwd_packets: 0.96, total_packets: 0.95, packet_rate: 0.98, byte_rate: 0.83,
    unique_peers: 0.88, avg_duration: 0.02, bwd_packets: 0.05, peer_density: 0.61,
  },
};

/** Stable pseudo-random in [0,1) — same inputs, same output, every run. */
function hash01(a: number, b: number, c: number): number {
  const x = Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453;
  return x - Math.floor(x);
}

function stageSeed(stage: string): number {
  let h = 0;
  for (let i = 0; i < stage.length; i++) h = (h * 31 + stage.charCodeAt(i)) % 9973;
  return h;
}

/**
 * How far a stage's behavioural signature has developed.
 *
 * Deliberately not the risk score. A port scan looks like a port scan from its
 * first packets — unique_dst_ports spikes immediately — even while the model
 * still rates it low risk, because a scan on its own is low risk. Tying the
 * attributes to risk hid the signature exactly when it is most legible.
 */
function signatureOf(stage: string, progress: number): number {
  return stage === "Benign" ? 0 : 0.6 + 0.4 * Math.max(0, Math.min(1, progress));
}

/**
 * The 27-D state the models read: 12 TGNE latent dims then the 15 host
 * attributes, in the index order data_unification/host_attributes.py uses.
 *
 * Attribution is Input x Gradient in spirit — distance from the benign
 * baseline times a per-dim sensitivity — so it rises exactly where the
 * attack moves the input, rather than being noise.
 */
function stateVector(step: number, stage: string, intensity: number): StateDim[] {
  const seed = stageSeed(stage);
  const target = ATTR_BY_STAGE[stage] ?? {};

  const latent = Array.from({ length: 12 }, (_, i) => {
    const base = wobble(0, i + 41) * 0.6;
    const pull = stage === "Benign" ? 0 : Math.sin(seed * (i + 1) * 0.37) * 1.0;
    const value = base + pull * intensity + wobble(step, i + 41) * 0.08;
    return { feature: `H_emb_${i}`, group: "TGNE Latent", value, base, sens: 0.35 };
  });

  const attrs = HOST_ATTRS.map((f, i) => {
    const base = ATTR_BASELINE[f] ?? 0.2;
    const goal = target[f] ?? base;
    const value = Math.max(0, Math.min(1, lerp(base, goal, intensity) + wobble(step, i + 11) * 0.02));
    return { feature: f, group: GROUP_OF[f] ?? "General", value, base, sens: 1.2 };
  });

  const dims = [...latent, ...attrs];
  // Floor keeps every dim visible; nothing in a real saliency map is exactly zero.
  const raw = dims.map((d) => Math.abs(d.value - d.base) * d.sens + 0.012);
  const total = raw.reduce((s, r) => s + r, 0) || 1;

  return dims.map((d, index) => ({
    index,
    feature: d.feature,
    group: d.group,
    value: Number(d.value.toFixed(4)),
    attribution: raw[index] / total,
  }));
}

/** Top-8 + group shares, derived from the same vector the inspector shows. */
function attribution(step: number, stage: string, intensity: number) {
  const vec = stateVector(step, stage, intensity);
  const byGroup = new Map<string, number>();
  for (const d of vec) byGroup.set(d.group, (byGroup.get(d.group) ?? 0) + d.attribution);

  return {
    available: true,
    method: "Input x Gradient Saliency",
    groups: [...byGroup.entries()]
      .map(([name, v]) => ({ name, percentage: v * 100 }))
      .sort((a, b) => b.percentage - a.percentage),
    top_features: [...vec]
      .sort((a, b) => b.attribution - a.attribution)
      .slice(0, 8)
      .map((d) => ({ feature: d.feature, score: d.attribution, group: d.group })),
  };
}

/* ── Flow evidence ─────────────────────────────────────────────────────── */

/** workloads/attacker_scenario.py:PORTS_TO_SCAN, verbatim. */
const SCAN_PORTS = [
  21, 22, 23, 25, 53, 80, 81, 88, 110, 111, 135, 139, 143, 389, 443, 445, 465, 587, 636, 993, 995,
  1433, 1521, 2049, 2375, 3000, 3306, 3389, 5000, 5432, 5601, 5900, 6379, 6443, 8000, 8080, 8081, 8443, 9000,
];
/** dmz-web serves HTTP on :80 only (nodes/services/web_dmz.py). */
const OPEN_PORTS = new Set([80]);

/** Internet clients of the DMZ services. They come and go on their own schedules. */
const BENIGN_EXTERNAL = Array.from({ length: 22 }, (_, i) => `192.168.100.${21 + i * 2}`);

interface FlowSpec {
  src: string;
  sport?: number;
  dst: string;
  dport: number;
  proto: FlowProtocol;
  fb: number;
  bb: number;
  fp: number;
  bp: number;
  dur: number;
  flags: FlowFlags;
  path: boolean;
}

const NOFLAGS: FlowFlags = { syn: 0, ack: 0, psh: 0, rst: 0, fin: 0, urg: 0 };

function isInternal(ip: string): boolean {
  return ip.startsWith("10.");
}

/** A normal, completed TCP session: handshake, data, graceful close. */
function session(fp: number, bp: number): FlowFlags {
  return { syn: 2, ack: fp + bp - 2, psh: Math.max(1, Math.round((fp + bp) * 0.35)), rst: 0, fin: 2, urg: 0 };
}

/**
 * The flows behind one window — a sample, as the sensor itself caps its
 * snapshot (state_builder.py writes at most 256). Benign background is always
 * present; attack flows follow the same phase script as the verdict, so the
 * evidence and the classification tell one story.
 */
function buildFlows(step: number, nowMs: number): FlowRecord[] {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  const { p } = phaseAt(step);
  const r = (i: number, k: number) => hash01(step, i, k);
  const eph = (i: number) => 49152 + Math.floor(r(i, 3) * 16000);
  const specs: FlowSpec[] = [];

  /* Background: DNS, directory, app-to-db, web from users and the internet. */
  for (const ws of ["10.0.1.11", "10.0.1.12", "10.0.1.13", "10.0.1.14"]) {
    specs.push({ src: ws, dst: "10.0.2.10", dport: 53, proto: "UDP", fb: 74, bb: 138, fp: 1, bp: 1, dur: 1.8, flags: NOFLAGS, path: false });
  }
  // srv-id is an HTTP identity service on :8080 (identity_server.py).
  specs.push({ src: "10.0.1.11", dst: "10.0.2.20", dport: 8080, proto: "TCP", fb: 1210, bb: 3480, fp: 9, bp: 11, dur: 41, flags: session(9, 11), path: false });
  specs.push({ src: "10.0.1.14", dst: "10.0.2.40", dport: 8000, proto: "TCP", fb: 3120, bb: 21_400, fp: 16, bp: 24, dur: 164, flags: session(16, 24), path: false });
  specs.push({ src: "10.0.2.40", dst: "10.0.2.50", dport: 5432, proto: "TCP", fb: 6240, bb: 48_120, fp: 22, bp: 41, dur: 118, flags: session(22, 41), path: false });
  // srv-file serves HTTP on :8080 (file_server.py), not SMB.
  specs.push({ src: "10.0.1.13", dst: "10.0.2.30", dport: 8080, proto: "TCP", fb: 18_300, bb: 212_400, fp: 64, bp: 158, dur: 640, flags: session(64, 158), path: false });
  specs.push({ src: "10.0.1.12", dst: TARGET, dport: 80, proto: "TCP", fb: 2140, bb: 38_900, fp: 14, bp: 31, dur: 212, flags: session(14, 31), path: false });
  BENIGN_EXTERNAL.forEach((ext, ci) => {
    const c = clientPresence(ci, step);
    if (!c.present || c.stale) return;
    // Most visitors browse the web tier; some relay mail or hold a VPN session.
    const kind = ci % 7 === 3 ? "mail" : ci % 5 === 4 ? "vpn" : "web";
    if (kind === "mail") {
      specs.push({ src: ext, dst: "10.0.3.20", dport: 25, proto: "TCP", fb: 14_200, bb: 1_120, fp: 18, bp: 9, dur: 410, flags: session(18, 9), path: false });
    } else if (kind === "vpn") {
      specs.push({ src: ext, dst: "10.0.3.30", dport: 443, proto: "TCP", fb: 38_400, bb: 112_600, fp: 96, bp: 140, dur: 1900, flags: session(96, 140), path: false });
    } else {
      specs.push({ src: ext, dst: TARGET, dport: 80, proto: "TCP", fb: 1880, bb: 24_600, fp: 12, bp: 22, dur: 186, flags: session(12, 22), path: false });
    }
  });

  /* The desk floor: name resolution, file and app use, the occasional print job. */
  BACKGROUND_ASSETS.filter((a) => a.zone === "users").forEach((ws, wi) => {
    const roll = (k: number) => hash01(step, wi + 40, k);
    if (roll(1) < 0.7) specs.push({ src: ws.ip, dst: "10.0.2.10", dport: 53, proto: "UDP", fb: 74, bb: 138, fp: 1, bp: 1, dur: 1.6, flags: NOFLAGS, path: false });
    if (roll(2) < 0.35) specs.push({ src: ws.ip, dst: "10.0.2.30", dport: 8080, proto: "TCP", fb: 4_100, bb: 96_000, fp: 22, bp: 70, dur: 420, flags: session(22, 70), path: false });
    if (roll(3) < 0.3) specs.push({ src: ws.ip, dst: "10.0.2.40", dport: 8000, proto: "TCP", fb: 2_600, bb: 18_300, fp: 14, bp: 20, dur: 150, flags: session(14, 20), path: false });
    if (roll(4) < 0.2) specs.push({ src: ws.ip, dst: "10.0.2.20", dport: 8080, proto: "TCP", fb: 1_210, bb: 3_480, fp: 9, bp: 11, dur: 40, flags: session(9, 11), path: false });
    if (roll(5) < 0.08) specs.push({ src: ws.ip, dst: "10.0.2.80", dport: 9100, proto: "TCP", fb: 410_000, bb: 900, fp: 290, bp: 40, dur: 2_400, flags: session(290, 40), path: false });
  });

  /* Infrastructure: monitoring polls, the nightly-style backup pull. */
  for (const target of ["10.0.2.10", "10.0.2.20", "10.0.2.30", "10.0.2.40", "10.0.2.50", "10.0.3.10"]) {
    if (hash01(step, target.length, 61) < 0.5) {
      specs.push({ src: "10.0.2.70", dst: target, dport: 9100, proto: "TCP", fb: 310, bb: 7_900, fp: 5, bp: 8, dur: 12, flags: session(5, 8), path: false });
    }
  }
  if (hash01(step, 3, 67) < 0.4) {
    specs.push({ src: "10.0.2.60", dst: "10.0.2.50", dport: 5432, proto: "TCP", fb: 2_200, bb: 1_480_000, fp: 30, bp: 1_040, dur: 1_800, flags: session(30, 1040), path: false });
  }

  /* Attack traffic, by phase. Stops at recovery: the host is isolated. */
  if (s < RECOVERY_STEP) {
    if (p.stage === "Recon") {
      // Connect scan: one SYN per port, RST from closed ports, SYN-ACK from open.
      for (let i = 0; i < 12; i++) {
        const port = SCAN_PORTS[(s * 5 + i) % SCAN_PORTS.length];
        const open = OPEN_PORTS.has(port);
        specs.push({
          src: ATTACKER, dst: TARGET, dport: port, proto: "TCP",
          fb: 60, bb: open ? 58 : 54, fp: 1, bp: 1, dur: 0.3 + r(i, 7) * 0.4,
          flags: open ? { ...NOFLAGS, syn: 2, ack: 1, rst: 1 } : { ...NOFLAGS, syn: 1, rst: 1, ack: 1 },
          path: true,
        });
      }
    } else if (p.technique === "T1110") {
      // Credential stuffing against the web login: short sessions, 401 each time.
      for (let i = 0; i < 10; i++) {
        specs.push({ src: ATTACKER, dst: TARGET, dport: 80, proto: "TCP", fb: 1180 + Math.floor(r(i, 9) * 240), bb: 812, fp: 7, bp: 5, dur: 38 + r(i, 8) * 30, flags: session(7, 5), path: true });
      }
    } else if (p.technique === "T1190") {
      // Exploit payloads: large request bodies, heavy PSH.
      for (let i = 0; i < 4; i++) {
        specs.push({
          src: ATTACKER, dst: TARGET, dport: 80, proto: "TCP",
          fb: 17_400 + Math.floor(r(i, 9) * 4800), bb: 3900, fp: 21, bp: 8, dur: 310 + r(i, 8) * 140,
          flags: { ...session(21, 8), psh: 18 }, path: true,
        });
      }
    } else if (p.stage === "C2") {
      // Outbound beacon from the compromised host, small and regular.
      specs.push({ src: TARGET, dst: ATTACKER, dport: 443, proto: "TCP", fb: 418, bb: 1124, fp: 5, bp: 6, dur: 64, flags: session(5, 6), path: true });
      if (s >= 46) {
        // Lateral: the app server's HTTP API on :8000 (app_server.py), the
        // first target attacker_scenario.py's LATERAL_PIVOT_STORM probes.
        specs.push({ src: TARGET, dst: "10.0.2.40", dport: 8000, proto: "TCP", fb: 8820, bb: 3140, fp: 26, bp: 19, dur: 910, flags: session(26, 19), path: true });
      }
    } else if (p.stage === "Impact") {
      // SYN flood: randomised source ports, one or two SYNs each, nothing back.
      for (let i = 0; i < 20; i++) {
        const syn = 1 + Math.floor(r(i, 11) * 2);
        specs.push({ src: ATTACKER, dst: TARGET, dport: 80, proto: "TCP", fb: 60 * syn, bb: 0, fp: syn, bp: 0, dur: 0.05, flags: { ...NOFLAGS, syn }, path: true });
      }
      if (s >= 52) {
        // Staging for exfiltration: database pulled through the app server.
        specs.push({ src: "10.0.2.40", dst: "10.0.2.50", dport: 5432, proto: "TCP", fb: 9600, bb: 2_840_000, fp: 212, bp: 1980, dur: 1840, flags: session(1980, 212), path: true });
        specs.push({ src: TARGET, dst: ATTACKER, dport: 443, proto: "TCP", fb: 2_612_000, bb: 14_200, fp: 1822, bp: 340, dur: 1910, flags: session(1822, 340), path: true });
      }
    }
  }

  // Spread first-packet times across the window, newest last.
  const span = TICK_MS * 1000;
  const base = nowMs * 1000 - span;
  return specs
    .map((f, i) => toFlow(f, `w${step}-${i}`, step, Math.floor(base + r(i, 13) * span), f.sport ?? eph(i)))
    .sort((a, b) => a.ts_us - b.ts_us);
}

function toFlow(f: FlowSpec, id: string, window: number, ts_us: number, src_port: number): FlowRecord {
  return {
    id,
    window,
    ts_us,
    src_ip: f.src,
    src_port,
    dst_ip: f.dst,
    dst_port: f.dport,
    protocol: f.proto,
    fwd_bytes: f.fb,
    bwd_bytes: f.bb,
    fwd_packets: f.fp,
    bwd_packets: f.bp,
    duration_ms: Number(f.dur.toFixed(3)),
    flags: f.proto === "TCP" ? f.flags : NOFLAGS,
    direction: isInternal(f.src) && isInternal(f.dst) ? "internal" : isInternal(f.dst) ? "inbound" : "outbound",
    on_path: f.path,
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
  mode: "LIVE",
  site_id: "hq-core",
  lab_mode: true,
  model_loaded: true,
  model_meta: MODEL_META,
  sensor_interface: "span0",
  nodes_running: 15,
  total_nodes: 15,
  topology_nodes: ASSETS.length + 14,
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
 * Backfill for the console's history and its time-travel scrubber: the
 * windows before the stream starts, as full payloads, so scrubbing back past
 * the moment the page opened shows real state rather than a gap.
 *
 * Every backfilled window sits at the calm tail of the cycle — the aftermath
 * of a previous, contained incident — and each uses a distinct step, so the
 * noise varies window to window. The last one hands over to the stream's
 * first window, which opens the next cycle, so the seam is invisible.
 */
export function seedFrames(n = 90): { payload: unknown; topology: Topology }[] {
  const now = Date.now();
  const out: { payload: unknown; topology: Topology }[] = [];
  for (let k = n; k > 0; k--) {
    const step = CYCLE - 1 + CYCLE * (n - k + 1);
    out.push({ payload: payloadAt(step, now - k * TICK_MS, -k), topology: buildTopology(step) });
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
      raw: "risk=0.6712 max_future=0.7003 ml=0.4881 rule=0.6712 rules_applied=true",
    },
    {
      ageMs: 24_000, severity: "warning", category: "ELEVATED", source: "192.168.100.41", destination: "10.0.3.10",
      message: "T1110 Brute Force — Credential Access",
      raw: "risk=0.7844 max_future=0.8319 ml=0.5602 rule=0.7844 rules_applied=true",
    },
    {
      ageMs: 19_000, severity: "critical", category: "CRITICAL", source: "192.168.100.41", destination: "10.0.3.10",
      message: "T1190 Exploit Public-Facing Application — Initial Access",
      raw: "risk=0.9021 max_future=0.9337 ml=0.6498 rule=0.9021 rules_applied=true",
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

/**
 * Discovery, as control_backend/topology_service.py does it: hosts exist
 * because traffic was seen, edges exist because a flow crossed them, and a
 * host that goes quiet turns stale and is then evicted on its TTL.
 *
 * Nothing here is a fixed drawing. The attacker materialises when the scan
 * starts, goes stale once the host is isolated and disappears a few windows
 * later; benign external clients come and go on their own schedules.
 */

type DiscoveredNode = TopologyNode & {
  role: "internal" | "external";
  zone: "external" | "dmz" | "servers" | "users";
  risk: number;
  bytes_in: number;
  bytes_out: number;
  stale: boolean;
  /** Core hosts carry the scenario and are always labelled; background hosts are population. */
  tier: "core" | "background";
};

/** Windows a quiet host stays visible (stale) before eviction. */
const STALE_WINDOWS = 6;

/** The attacker exists from the first probe until its TTL runs out. */
function attackerPresence(step: number): { present: boolean; stale: boolean } {
  const s = ((step % CYCLE) + CYCLE) % CYCLE;
  if (s >= RECOVERY_STEP + STALE_WINDOWS) return { present: false, stale: false };
  return { present: true, stale: s >= RECOVERY_STEP };
}

/**
 * Benign clients drift in and out. Each has its own period and duty cycle, so
 * the external layer is never the same twice — sessions open, go quiet, age
 * out, and new visitors arrive.
 */
function clientPresence(i: number, step: number): { present: boolean; stale: boolean } {
  const period = 22 + (i % 5) * 4;
  const active = Math.round(period * (0.45 + (i % 3) * 0.12));
  const phase = (((step + i * 7) % period) + period) % period;
  if (phase >= active) return { present: false, stale: false };
  return { present: true, stale: phase >= active - 3 };
}

/** Row position for a host: core hosts evenly spaced, background in the gaps. */
function slotOf(band: Asset[], ip: string): number {
  const named: Asset[] = band.filter((b: Asset) => b.tier === "core");
  const bg: Asset[] = band.filter((b: Asset) => b.tier !== "core");
  const n = band.length;
  const slots: (Asset | null)[] = new Array(n).fill(null);
  named.forEach((c: Asset, k: number) => {
    let at = Math.min(n - 1, Math.floor(((k + 0.5) * n) / named.length));
    while (slots[at]) at = (at + 1) % n;
    slots[at] = c;
  });
  let j = 0;
  for (let i = 0; i < n; i++) if (!slots[i]) slots[i] = bg[j++];
  return Math.max(0, slots.findIndex((x) => x?.ip === ip));
}

function buildTopology(step: number): Topology {
  const w = windowState(step);
  const risk = w.shown;
  const hot = w.hot;
  const flows = buildFlows(step, Date.now());

  // Per-host bytes from this window's flows.
  const bin = new Map<string, number>();
  const bout = new Map<string, number>();
  for (const f of flows) {
    bout.set(f.src_ip, (bout.get(f.src_ip) ?? 0) + f.fwd_bytes);
    bin.set(f.src_ip, (bin.get(f.src_ip) ?? 0) + f.bwd_bytes);
    bin.set(f.dst_ip, (bin.get(f.dst_ip) ?? 0) + f.fwd_bytes);
    bout.set(f.dst_ip, (bout.get(f.dst_ip) ?? 0) + f.bwd_bytes);
  }

  const statusFor = (r: number, stale: boolean): TopologyNode["status"] =>
    stale ? "offline" : r >= 0.8 ? "compromised" : r >= 0.5 ? "warning" : "online";

  /* External band: the attacker first, then whichever clients are present. */
  const externals: { ip: string; label: string; r: number; stale: boolean; tier: "core" | "background" }[] = [];
  const atk = attackerPresence(step);
  if (atk.present) externals.push({ ip: ATTACKER, label: "external", r: atk.stale ? 0.1 : hot ? Math.max(risk, 0.82) : 0.1, stale: atk.stale, tier: "core" });
  BENIGN_EXTERNAL.forEach((ip, i) => {
    const c = clientPresence(i, step);
    if (c.present) externals.push({ ip, label: `client-${ip.split(".")[3]}`, r: 0.04, stale: c.stale, tier: "background" });
  });

  const extSpread = 640 / (externals.length + 1);
  const nodes: DiscoveredNode[] = externals.map((e, i) => ({
    id: e.ip,
    label: e.label,
    ip: e.ip,
    type: "internet",
    status: statusFor(e.r, e.stale),
    x: 80 + extSpread * (i + 1),
    y: 62,
    role: "external",
    zone: "external",
    risk: e.r,
    bytes_in: bin.get(e.ip) ?? 0,
    bytes_out: bout.get(e.ip) ?? 0,
    stale: e.stale,
    tier: e.tier,
  }));

  /* Internal hosts are always talking, so they never age out. */
  ASSETS.forEach((a, i) => {
    const isTarget = a.ip === TARGET;
    const lateral = hot && risk > 0.8 && (a.ip === "10.0.2.40" || a.ip === "10.0.2.50");
    const nodeRisk = isTarget ? risk : lateral ? risk * 0.7 : risk * 0.2;
    const band = a.zone === "dmz" ? 196 : a.zone === "servers" ? 318 : 404;
    // Named hosts are spaced evenly through the row, population fills the
    // gaps, so labelled hosts never sit shoulder to shoulder in the 2D graph.
    const inBand = ASSETS.filter((x) => x.zone === a.zone);
    const idx = slotOf(inBand, a.ip);
    const spread = 640 / (inBand.length + 1);
    nodes.push({
      id: a.ip,
      label: a.name,
      ip: a.ip,
      type: a.zone === "users" ? "host" : "server",
      status: statusFor(nodeRisk, false),
      x: 80 + spread * (idx + 1),
      y: band + (i % 2) * 14,
      role: "internal",
      zone: a.zone as DiscoveredNode["zone"],
      risk: Number(nodeRisk.toFixed(4)),
      bytes_in: bin.get(a.ip) ?? 0,
      bytes_out: bout.get(a.ip) ?? 0,
      stale: false,
      tier: a.tier,
    });
  });

  /* Edges are observed flows, collapsed to host pairs. */
  const present = new Set(nodes.map((n) => n.ip));
  const pairs = new Map<string, { src: string; dst: string; bytes: number; path: boolean; proto: string }>();
  for (const f of flows) {
    if (!present.has(f.src_ip) || !present.has(f.dst_ip)) continue;
    const key = `${f.src_ip}-${f.dst_ip}`;
    const e = pairs.get(key) ?? { src: f.src_ip, dst: f.dst_ip, bytes: 0, path: false, proto: f.protocol };
    e.bytes += f.fwd_bytes + f.bwd_bytes;
    e.path = e.path || f.on_path;
    pairs.set(key, e);
  }

  const windowSec = 2.0;
  const edges: Topology["edges"] = [...pairs.values()].map((e) => {
    const mbps = (e.bytes * 8) / 1e6 / windowSec;
    return {
      id: `${e.src}-${e.dst}`,
      source: e.src,
      target: e.dst,
      status: e.path ? (risk > 0.85 ? "saturated" : "suspicious") : "active",
      protocol: e.proto,
      bandwidth: Number(mbps.toFixed(2)),
      utilization: Math.min(100, Math.round(mbps / 10)),
    };
  });

  return { nodes, edges, lastUpdated: new Date().toISOString() };
}

export const mockFetchTopology = async (): Promise<Topology> => buildTopology(BASELINE_BAND.start);

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
      host: "10.0.1.12",
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

  // A contained incident keeps the attack that opened it, not the calm after it.
  const cause = contained ? PHASES[PHASES.length - 2] : w.p;
  return {
    id: "INC-0142",
    host: TARGET,
    hostLabel: "dmz-web",
    title: cause.label,
    technique: cause.technique,
    tactic: cause.tactic,
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
        technique: "T1071 Application Layer Protocol",
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
      // Deliberately NOT windowState().mlRisk: that applies the live path's
      // pre-rule discount, which capped this curve at 0.68 and left the
      // chain peaking lower than the standalone flood sample even though it
      // ends in the same T1498 flood. Rules are off here, so the phase risk
      // is the model output, lightly damped.
      const w = windowState(Math.min(CYCLE - 1, Math.floor(u * CYCLE)));
      return {
        risk: clamp01(w.risk * 0.96 + jitter),
        stage: w.p.stage,
        technique: w.hot ? w.p.label : null,
        tactic: w.hot ? w.p.tactic : null,
      };
    }
  }
}

/** Replay windows are the model's real 2.0s windows, not the live demo's compressed tick. */
const REPLAY_WINDOW_US = 2_000_000;

/** The IRC server the Neris sample beacons to, on the lab's internet segment. */
const NERIS_C2 = "192.168.100.66";
/** HOIC is a many-source flood; these are the sample's attacking hosts. */
const HOIC_SOURCES = Array.from({ length: 28 }, (_, i) => `192.168.100.${101 + i * 3}`);

/** When each sample was captured, so flow timestamps are stable across runs. */
const CAPTURED_AT: Record<string, number> = {
  infiltration: Date.UTC(2026, 2, 14, 10, 22, 5),
  botnet: Date.UTC(2026, 2, 11, 21, 4, 40),
  bruteforce: Date.UTC(2026, 2, 12, 3, 17, 12),
  ddos: Date.UTC(2026, 2, 15, 14, 2, 51),
  benign: Date.UTC(2026, 2, 10, 11, 0, 0),
};

/** Attack traffic per profile, shaped to match `beatFor` window for window. */
function replaySpecs(profile: string, i: number, u: number): FlowSpec[] {
  const r = (k: number, c: number) => hash01(i + 7919, k, c);
  const specs: FlowSpec[] = [];

  if (profile === "botnet") {
    // IRC keep-alive on the open channel, every window.
    specs.push({ src: "10.0.1.11", dst: NERIS_C2, dport: 6667, proto: "TCP", fb: 142 + Math.floor(r(0, 5) * 40), bb: 96, fp: 2, bp: 2, dur: 180 + r(0, 6) * 120, flags: { ...NOFLAGS, ack: 4, psh: 2 }, path: true });
    if (i % 7 === 0) {
      // Check-in: resolve the server again, then a burst of command traffic.
      specs.push({ src: "10.0.1.11", dst: "10.0.2.10", dport: 53, proto: "UDP", fb: 81, bb: 97, fp: 1, bp: 1, dur: 2.1, flags: NOFLAGS, path: true });
      for (let k = 1; k <= 5; k++) {
        specs.push({ src: "10.0.1.11", dst: NERIS_C2, dport: 6667, proto: "TCP", fb: 620 + Math.floor(r(k, 5) * 400), bb: 1_840 + Math.floor(r(k, 7) * 900), fp: 6, bp: 5, dur: 240 + r(k, 6) * 300, flags: session(6, 5), path: true });
      }
    }
  } else if (profile === "bruteforce") {
    // SSH-Patator: short authenticate-and-fail sessions in bursts, a lone retry in the back-off.
    const phase = (u * 4) % 1;
    const n = phase < 0.45 ? 8 + Math.floor(phase * 12) : 1;
    for (let k = 0; k < n; k++) {
      const fp = 12 + Math.floor(r(k, 5) * 5);
      const bp = 10 + Math.floor(r(k, 6) * 4);
      specs.push({ src: ATTACKER, dst: TARGET, dport: 22, proto: "TCP", fb: 2_300 + Math.floor(r(k, 7) * 600), bb: 3_000 + Math.floor(r(k, 8) * 500), fp, bp, dur: 380 + r(k, 9) * 520, flags: session(fp, bp), path: true });
    }
  } else if (profile === "ddos" && u > 0.34) {
    // HOIC: every source holds a keep-alive HTTP connection and floods GETs down it.
    HOIC_SOURCES.forEach((src, k) => {
      const fp = 520 + Math.floor(r(k, 5) * 560);
      const bp = Math.floor(fp * (0.42 + r(k, 6) * 0.1));
      specs.push({ src, dst: TARGET, dport: 80, proto: "TCP", fb: fp * 312, bb: bp * 1_180, fp, bp, dur: 1_940 + r(k, 7) * 50, flags: { syn: 1, ack: fp + bp - 1, psh: fp, rst: 0, fin: 0, urg: 0 }, path: true });
    });
  }
  return specs;
}

/**
 * The flows behind one replayed window. Background is the lab's own traffic
 * lifted from a quiet window of the live scenario and re-timed onto this
 * window; the attack traffic follows the profile.
 */
function replayFlows(profile: string, i: number, u: number, startUs: number): FlowRecord[] {
  const r = (k: number, c: number) => hash01(i + 7919, k, c);
  const eph = (k: number) => 49152 + Math.floor(r(k, 3) * 16000);
  const start = startUs + i * REPLAY_WINDOW_US;
  const liveSpan = TICK_MS * 1000;

  // A step in the scenario's benign tail, different for every window.
  const quiet = CYCLE * (i + 2) + RECOVERY_STEP + 2 + (i % 8);
  const lifted = buildFlows(quiet, 0).filter((f) => !f.on_path);
  if (profile === "infiltration") {
    lifted.push(...buildFlows(Math.min(CYCLE - 1, Math.floor(u * CYCLE)), 0).filter((f) => f.on_path));
  }

  const out: FlowRecord[] = lifted.map((f, k) => ({
    ...f,
    id: `r${i}-${k}`,
    window: i,
    ts_us: start + Math.floor(((f.ts_us + liveSpan) / liveSpan) * REPLAY_WINDOW_US * 0.98),
    src_port: f.src_port >= 49152 ? eph(k) : f.src_port,
  }));
  replaySpecs(profile, i, u).forEach((f, k) => {
    out.push(toFlow(f, `r${i}-a${k}`, i, start + Math.floor(r(k, 13) * REPLAY_WINDOW_US * 0.98), f.sport ?? eph(k + 100)));
  });
  return out.sort((a, b) => a.ts_us - b.ts_us);
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
  const startUs = (CAPTURED_AT[profile] ?? CAPTURED_AT.infiltration) * 1000;

  for (let i = 0; i < windows; i++) {
    const beat = beatFor(profile, i / span, i);
    const prev = i > 0 ? beatFor(profile, (i - 1) / span, i - 1).risk : beat.risk;
    const rising = beat.risk >= prev;
    const ex = attribution(i, beat.stage, signatureOf(beat.stage, beat.risk));

    const forecast = Array.from({ length: 8 }, (_, k) => ({
      horizon_seconds: (k + 1) * 2,
      risk: clamp01(beat.risk * (rising ? 1 + (k + 1) * 0.03 : 1 - (k + 1) * 0.045)),
    }));

    const flows = replayFlows(profile, i, i / span, startUs);
    // A SYN flood opens a flow per spoofed source port, far more than the
    // sensor's 256-record sample holds, so the window's count comes from the
    // capture rather than from the records.
    const unsampled = profile === "infiltration" && beat.stage === "Impact" ? Math.round(24_000 + wobble(i, 23) * 3_000) : 0;

    results.push({
      window: i,
      target: profile === "botnet" ? "10.0.1.11" : profile === "benign" ? "10.0.2.10" : TARGET,
      risk: beat.risk,
      ml_risk: beat.risk,
      alert: beat.risk >= THRESHOLD,
      stage: beat.stage,
      mitre_tactic: beat.tactic,
      mitre_technique: beat.technique,
      forecast,
      top_features: ex.top_features.slice(0, 5).map((f) => ({ feature: f.feature, score: f.score, group: f.group })),
      packets: unsampled + flows.reduce((n, f) => n + f.fwd_packets + f.bwd_packets, 0),
      bytes: unsampled * 60 + flows.reduce((n, f) => n + f.fwd_bytes + f.bwd_bytes, 0),
      flows,
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

/* ── Window payloads ───────────────────────────────────────────────────── */

/**
 * The full /ws prediction payload for one window. Pure — the same step and
 * clock give the same payload — so the live stream and the time-travel
 * backfill are built by one function.
 */
function payloadAt(step: number, nowMs: number, windowId: number = step, lead?: { stage: string; secondsRemaining: number }) {
  const now = new Date(nowMs).toISOString();
  const w = windowState(step);
  const intensity = signatureOf(w.p.stage, w.t);
  const explainability = attribution(step, w.p.stage, intensity);

  // Prediction-leads-observation. While a lead is active (nmap and hping3 fire
  // this), the observed risk stays w.shown (calm baseline) but the forecast and
  // the whole alarm state already call the incoming stage — the model flagging
  // the attack seconds before the traffic actually changes. The predicted stage
  // is sampled at the top of its own band, so its risk, technique and severity
  // match where it is headed (recon → T1046 warning, ddos → T1498 critical).
  // When the lead elapses the stream promotes into the stage and observed risk
  // catches up.
  const wf = lead ? windowState(STAGE_BANDS[lead.stage].end - 1) : null;
  const leadTarget = wf ? wf.shown : 0;
  const leadForecast = wf
    ? Array.from({ length: 8 }, (_, k) => ({
        horizon_seconds: (k + 1) * 2,
        // Climbs from the calm present up to the predicted stage's risk.
        risk: Math.min(0.98, 0.12 + ((k + 1) / 8) * Math.max(0, leadTarget - 0.12)),
        confidence: Math.max(0.45, 0.92 - k * 0.05),
        predicted_stage: wf.p.stage,
      }))
    : null;
  const forecast = leadForecast ?? w.forecast;
  const maxFuture = wf ? leadTarget : w.maxFuture;
  const predStage = wf ? wf.p.stage : w.p.stage;
  const predTactic = wf ? wf.p.tactic : w.p.tactic;
  const predTechnique = wf ? wf.p.label : w.p.label;
  const predTacticId = wf ? wf.p.tacticId : w.p.tacticId;
  const predDesc = wf ? wf.p.desc : w.p.desc;
  // During a lead the ALARM state is already raised at the predicted stage's
  // severity — alert, attack active, early warning — so the console reads
  // "attack detected, needs attention" the instant the model predicts it. Only
  // the observed risk line (prediction.risk = w.shown) stays calm until the
  // lead elapses.
  const alertLevel = wf ? wf.alertLevel : w.alertLevel;
  const alert = lead ? true : w.shown >= THRESHOLD || w.maxFuture >= THRESHOLD;
  // The stage distribution and precursor confidence follow the forecast in a lead.
  const probStage = lead ? predStage : w.p.stage;
  const probBase = lead ? maxFuture : w.shown;

  return {
    type: "prediction",
    mode: "LIVE",
    timestamp: now,
    wall_clock: now,
    model: MODEL_META,
    state: {
      window_id: windowId,
      sequence_ready: true,
      packet_count: w.packets,
      active_flows: w.flows,
      pipeline_latency_ms: w.telemetryMs,
      buffer_length: 5,
    },
    prediction: {
      risk: w.shown,
      max_future_risk: maxFuture,
      hazard_score: Math.min(0.99, maxFuture * 0.92),
      malicious_confidence: Math.min(0.99, 0.42 + w.shown * 0.55),
      precursor_confidence: Math.min(0.99, 0.3 + probBase * 0.55),
      alert,
      alert_level: alertLevel,
      threshold: THRESHOLD,
      predicted_stage: predStage,
      mitre_tactic: predTactic,
      mitre_technique: predTechnique,
      mitre_tactic_id: predTacticId,
      mitre_description: predDesc,
      technique_confidence: Math.min(0.99, 0.5 + probBase * 0.45),
      stage_probabilities: Object.fromEntries(
        ["Benign", "Recon", "InitialAccess", "C2", "Impact"].map((s) => [
          s,
          s === probStage
            ? Math.min(0.95, 0.5 + probBase * 0.4)
            : Math.max(0.01, (1 - probBase) * 0.22 + Math.abs(wobble(step, s.length)) * 0.05),
        ])
      ),
      stage_provenance: {
        TGNE: "h_emb(12)",
        BRANCH_A: "risk+technique",
        BRANCH_B: "H_hat trajectory",
        DEEPOP: "technique sequence",
      },
      ml_risk: w.mlRisk,
      rule_risk: w.ruleRisk,
      rules_applied: w.rulesApplied,
      // Fields adapter.ts normally attaches on the real path
      value: w.shown,
      confidence: Math.min(0.99, 0.42 + w.shown * 0.55),
      horizon: 16,
      model: "K.I.R.A. Ensemble",
      branch_a_risk: w.shown,
      branch_b_risk: maxFuture,
      explainability,
      signals: explainability.top_features.map((f) => ({
        name: f.feature,
        weight: f.score,
        direction: "neutral",
        value: f.group,
      })),
    },
    forecast,
    explainability,
    latency: {
      telemetry_ms: w.telemetryMs,
      inference_ms: w.inferenceMs,
      total_ms: Number((w.telemetryMs + w.inferenceMs).toFixed(2)),
    },
    early_warning: lead
      ? {
          is_alert: true,
          alert_timestamp: now,
          actual_milestone_timestamp: null,
          lead_time_seconds: Number(lead.secondsRemaining.toFixed(1)),
          target_milestone_desc: "predicted T1498 volumetric flood",
        }
      : w.hot && w.shown > 0.5
        ? {
            is_alert: true,
            alert_timestamp: now,
            actual_milestone_timestamp: null,
            lead_time_seconds: Number((6 + Math.abs(wobble(step, 31)) * 7).toFixed(1)),
            target_milestone_desc: `first observed ${w.p.stage} milestone`,
          }
        : null,
    attack_active: lead ? true : w.hot,
    attack_phase: predStage,
    focus_ips: lead || w.hot ? [ATTACKER, TARGET] : [],
    focus_edges: lead || w.hot ? [{ src: ATTACKER, dst: TARGET }] : [],
    target_ip: TARGET,
    throughput: w.throughput,
    // Demo-only: correlation/ is not wired into control_backend yet.
    campaign: buildCampaign(step, nowMs),
    incidents: buildIncidents(step, nowMs),
    // Evidence. Present in the pipeline but not forwarded over /ws yet —
    // see src/types/evidence.ts. Absent against a live backend.
    flows: buildFlows(step, nowMs),
    flows_in_window: w.flows,
    state_vector: stateVector(step, w.p.stage, intensity),
    // Demo-only: the rollout's competing continuations (types/forecast.ts).
    branches: branchesAt(step, w),
    // Demo-only: TGNE temporal attention (types/attention.ts).
    attention: attentionAt(step, w),
  };
}

function statusAt(step: number, now: string, lead?: { stage: string; secondsRemaining: number }): SystemStatus {
  const w = windowState(step);
  // During a lead the forecast (not the observed traffic) drives the top-line
  // threat: the anomaly score and threat level jump immediately to the
  // predicted stage's severity while observed risk is still calm.
  const wf = lead ? windowState(STAGE_BANDS[lead.stage].end - 1) : null;
  const maxFuture = wf ? wf.shown : w.maxFuture;
  return {
    networkStatus: "running",
    telemetryStatus: "running",
    predictionStatus: "running",
    attackStatus: lead || w.hot ? "active" : "none",
    uptime: 14_820 + step * 2,
    lastUpdate: now,
    anomalyScore: Math.round(Math.max(w.shown, maxFuture) * 1000) / 10,
    // Alarm reflects the predicted stage during a lead, even though observed risk is calm.
    threatLevel: wf ? wf.alertLevel.toLowerCase() : w.alertLevel.toLowerCase(),
    packetLoss: Number((w.hot ? w.shown * 4.1 : 0.3).toFixed(2)),
    latency: Number((w.telemetryMs + w.inferenceMs).toFixed(2)),
    throughput: w.throughput,
    activeConnections: w.flows,
    ...BASE_STATUS,
  } as unknown as SystemStatus;
}

/* ── Scenario control ──────────────────────────────────────────────────── */

/**
 * Kill-chain stage → the window band [start, end) it occupies within one CYCLE
 * (see PHASES). The live stream walks into a band when the stage is triggered
 * and then holds near its top, so a triggered stage stays on screen with its
 * forecast projecting ahead — rather than marching on into recovery.
 *
 * Stage names match both the UI's ScenarioId (recon/probe/exploit/c2/lateral)
 * and the terminal demo bridge (which adds `ddos`).
 */
const STAGE_BANDS: Record<string, { start: number; end: number }> = {
  recon: { start: 0, end: 14 },
  probe: { start: 14, end: 26 },
  exploit: { start: 26, end: 38 },
  c2: { start: 38, end: 50 },
  lateral: { start: LATERAL_OBSERVED, end: 50 },
  ddos: { start: 50, end: RECOVERY_STEP },
};

/** Calm business-as-usual band: past the attacker's TTL, so it has fully aged out. */
const BASELINE_BAND = { start: RECOVERY_STEP + STALE_WINDOWS, end: CYCLE };

/** The "contained, risk dropping" band mitigation collapses into. */
const CONTAIN_BAND = { start: RECOVERY_STEP, end: RECOVERY_STEP + 4 };

/** Windows at the top of a band the stream oscillates over while held in a stage. */
const HOLD_WINDOWS = 4;

/**
 * Lead time before the observed risk actually spikes: the forecast + alarm call
 * the incoming stage this long ahead — the "we predicted it early" beat of the
 * demo. hping3/DDoS leads by ~5s, nmap/recon by ~2.5s.
 */
const DDOS_LEAD_MS = 5000;
const RECON_LEAD_MS = 2500;

/** Live instance, so command/mitigation fixtures can steer the scenario. */
let active: MockWebSocket | null = null;

export const mockSendCommand = async (command: string): Promise<void> => {
  active?.note(`command ${command} accepted`);
  switch (command) {
    case "reset_environment":
    case "stop_attack":
      active?.toBaseline();
      break;
    // Adversary emulation: hold workloads/attacker_scenario.py at a stage.
    case "start_attack":
    case "emulate_recon":
      // Predict first, spike the observed risk ~2.5s later (see enterStage lead).
      active?.enterStage("recon", RECON_LEAD_MS);
      break;
    case "emulate_probe":
      active?.enterStage("probe");
      break;
    case "emulate_exploit":
      active?.enterStage("exploit");
      break;
    case "emulate_c2":
      active?.enterStage("c2");
      break;
    case "emulate_lateral":
      active?.enterStage("lateral");
      break;
    case "emulate_ddos":
      // Predict first, spike the observed risk 5s later (see enterStage lead).
      active?.enterStage("ddos", DDOS_LEAD_MS);
      break;
  }
};

/**
 * Mitigation collapses the scenario into its recovery phase and holds it there.
 * On camera this reads as cause and effect: the operator isolates the host and
 * risk drops — worth more in a demo than any static panel.
 */
export const mockSendMitigate = async (payload: MitigationPayload): Promise<void> => {
  active?.note(`mitigation ${payload.action}${payload.target ? ` → ${payload.target}` : ""} applied`);
  if (!String(payload.action).startsWith("CLEAR")) active?.contain();
};

/**
 * Apply a terminal-driven trigger (from the Vite demo bridge, `/__demo/*`).
 * A bare stage name arrives over SSE; route it through the same fixtures the
 * UI uses so the scenario, console and events all move together.
 */
export function applyDemoTrigger(stage: string): void {
  const s = stage.toLowerCase().trim();
  if (s === "reset") return void mockSendCommand("reset_environment");
  if (s === "contain") return void mockSendMitigate({ action: "ISOLATE_HOST", target: TARGET } as unknown as MitigationPayload);
  if (STAGE_BANDS[s]) return void mockSendCommand(`emulate_${s}`);
}

/* ── Live stream ───────────────────────────────────────────────────────── */

export class MockWebSocket {
  private handlers: WSHandlers;
  private timer: ReturnType<typeof setInterval> | undefined;
  private opening: ReturnType<typeof setTimeout> | undefined;
  private closed = false;
  /** Scenario position within one CYCLE — drives both the phase and the noise. */
  private pos = BASELINE_BAND.start;
  /** The band pos is confined to, or null for the calm baseline. */
  private band: { start: number; end: number } | null = null;
  /** Monotonic window id for display, independent of the (looping) position. */
  private seq = 0;
  /** Stage the forecast is calling ahead of observation, or null. */
  private leadStage: string | null = null;
  /** Wall-clock time (ms) when the lead ends and observed risk spikes. */
  private leadUntil = 0;

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

  /**
   * Enter a kill-chain stage from its start, then hold near its top.
   *
   * With `leadMs`, the forecast calls the stage now while the observed risk
   * stays at a calm baseline; after the lead elapses the stream promotes into
   * the stage and the observed risk spikes — the "predicted it early" beat.
   */
  enterStage(stage: string, leadMs = 0) {
    const band = STAGE_BANDS[stage];
    if (!band) return;
    if (leadMs > 0) {
      this.leadStage = stage;
      this.leadUntil = Date.now() + leadMs;
      this.band = null;
      this.pos = BASELINE_BAND.start;
    } else {
      this.leadStage = null;
      this.leadUntil = 0;
      this.band = band;
      this.pos = band.start;
    }
  }

  /** Return to a calm baseline (no attack) and hold there. */
  toBaseline() {
    this.leadStage = null;
    this.leadUntil = 0;
    this.band = null;
    this.pos = BASELINE_BAND.start;
  }

  /** Collapse into the contained-and-recovering window and hold there. */
  contain() {
    this.leadStage = null;
    this.leadUntil = 0;
    this.band = CONTAIN_BAND;
    this.pos = CONTAIN_BAND.start;
  }

  /** Push a synthetic console line (command + mitigation acknowledgements). */
  note(text: string) {
    this.handlers.onCommandOutput?.({
      command: "ui",
      line: `[${new Date().toISOString()}] ctrl INFO: ${text}`,
    });
  }

  /** Step the position forward, looping across the top of the active band. */
  private advance() {
    const band = this.band ?? BASELINE_BAND;
    this.pos++;
    if (this.pos >= band.end) {
      // Hold: oscillate across the top few windows so the charts keep breathing
      // while the scenario stays in its stage rather than advancing.
      this.pos = Math.max(band.start, band.end - HOLD_WINDOWS);
    }
  }

  private emit() {
    const nowMs = Date.now();
    // If a prediction lead has elapsed, the forecast attack now actually lands:
    // promote into the stage so the observed risk spikes to meet the forecast.
    if (this.leadStage && nowMs >= this.leadUntil) {
      const band = STAGE_BANDS[this.leadStage];
      this.leadStage = null;
      this.leadUntil = 0;
      if (band) {
        this.band = band;
        // Jump straight to the hold zone so the observed risk spikes to the
        // severity that was predicted, rather than ramping up from the band's
        // calm start (which would briefly dip the alarm back down).
        this.pos = Math.max(band.start, band.end - HOLD_WINDOWS);
      }
    }
    const lead = this.leadStage
      ? { stage: this.leadStage, secondsRemaining: Math.max(0, (this.leadUntil - nowMs) / 1000) }
      : undefined;

    const step = this.pos;
    const wid = this.seq++;
    const now = new Date(nowMs).toISOString();
    const w = windowState(step);

    this.handlers.onPrediction?.(payloadAt(step, nowMs, wid, lead) as never);
    this.handlers.onTopologyUpdate?.(buildTopology(step));
    this.handlers.onSystemStatus?.(statusAt(step, now, lead));

    /* Console traffic so the log panes stay alive under the charts. */
    if (wid % 3 === 0) {
      this.handlers.onCommandOutput?.({
        command: "sensor",
        line: `[${now}] sensor INFO: window ${wid} closed — ${w.packets} packets, ${w.flows} flows, ${w.telemetryMs.toFixed(1)}ms`,
      });
    }
    if (lead) {
      // The forecast is calling the stage while the wire is still quiet.
      if (wid % 2 === 0) {
        const wf = windowState(STAGE_BANDS[lead.stage].end - 1);
        const word = wf.alertLevel === "CRITICAL" ? "CRITICAL" : "WARN";
        this.handlers.onCommandOutput?.({
          command: "ml",
          line: `[${now}] ml   ${word}: ${wf.p.label} predicted — NEEDS ATTENTION, onset in ~${lead.secondsRemaining.toFixed(1)}s target=${TARGET}`,
        });
      }
    } else if (w.hot && wid % 4 === 0) {
      this.handlers.onCommandOutput?.({
        command: "ml",
        line: `[${now}] ml   ${w.alertLevel === "CRITICAL" ? "CRITICAL" : "WARN"}: ${w.p.label} risk=${w.shown.toFixed(3)} target=${TARGET}`,
      });
    }

    this.advance();
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
