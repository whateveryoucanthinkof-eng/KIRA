/**
 * API / WebSocket adapter layer.
 * Connected to the real backend.
 */

import type {
  SystemStatus,
  ThreatLevel,
  Topology,
  SiteInfo,
  MitigationPayload,
  WSHandlers,
  ContributingSignal,
  ExplainabilityPayload,
} from "./types";
import type { ReplayReport } from "../types/replay";
import type { ReplaySample } from "./mock";
import {
  mockFetchStatus,
  mockFetchSite,
  mockFetchTopology,
  mockSendCommand,
  mockSendMitigate,
  mockReplay,
  mockReplaySample,
  MockWebSocket
} from "./mock";

export const BASE_URL = "/api";
export const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

function normalizeThreatLevel(raw: any): ThreatLevel {
  const s = String(raw || "low").toLowerCase();
  if (s === "critical") return "critical";
  if (s === "high" || s === "elevated") return "high";
  if (s === "medium" || s === "warning") return "medium";
  return "low";
}

// Map real backend status to the Figma UI status format
function mapSystemStatus(backendStatus: any): SystemStatus {
  return {
    networkStatus: backendStatus.network || "stopped",
    telemetryStatus: backendStatus.sensor || "stopped",
    predictionStatus: backendStatus.ml || "stopped",
    attackStatus: backendStatus.attack === "running" ? "active" : "none",
    uptime: backendStatus.uptime || 0,
    lastUpdate: backendStatus.timestamp || new Date().toISOString(),
    anomalyScore: backendStatus.anomalyScore || 0,
    threatLevel: normalizeThreatLevel(backendStatus.threatLevel),
    packetLoss: backendStatus.packetLoss || 0,
    latency: backendStatus.latency || 0,
    throughput: backendStatus.throughput || 0,
    activeConnections: backendStatus.activeConnections || 0,
    // Keep raw backend fields for conditional logic
    ...backendStatus,
  };
}

// ─── Topology Layout Engine ──────────────────────────────────────────────────
const nodeLayoutCache = new Map<string, { x: number; y: number }>();
const edgeStatsCache = new Map<string, { lastBytes: number; lastTime: number; currentMbps: number }>();

function mapTopology(backendTopo: any): Topology {
  const nodes = backendTopo.nodes || [];

  const extNodes = nodes.filter((n: any) => n.role === "external" || n.zone === "external");
  const dmzNodes = nodes.filter((n: any) => n.role === "sensor" || n.zone === "dmz");
  const intNodes = nodes.filter((n: any) =>
    !extNodes.includes(n) && !dmzNodes.includes(n)
  );

  const width = 800;
  const padding = 150;

  function assignPositions(group: any[], baseY: number, ySpread: number = 40) {
    const nodesPerRow = 5;
    group.forEach((n, i) => {
      if (nodeLayoutCache.has(n.id)) {
        const pos = nodeLayoutCache.get(n.id)!;
        n.x = pos.x;
        n.y = pos.y;
      } else {
        const row = Math.floor(i / nodesPerRow);
        const col = i % nodesPerRow;
        const rowCount = Math.ceil(group.length / nodesPerRow);
        const currentGroupSize = row === rowCount - 1 && group.length % nodesPerRow !== 0
          ? group.length % nodesPerRow
          : Math.min(group.length, nodesPerRow);

        const spacing = currentGroupSize > 1 ? (width - 2 * padding) / (currentGroupSize - 1) : 0;
        const x = currentGroupSize === 1 ? width / 2 : padding + (col * spacing);

        // Stagger rows
        const y = baseY + (row * ySpread) + (Math.random() * 20 - 10);
        n.x = x;
        n.y = y;
        nodeLayoutCache.set(n.id, { x, y });
      }

      if (n.role === "external") n.type = "internet";
      else if (n.role === "sensor") n.type = "firewall";
      else n.type = "host";

      if (n.risk >= 0.8) n.status = "compromised";
      else if (n.risk >= 0.5) n.status = "warning";
      else if (n.stale) n.status = "offline";
      else n.status = "online";

      if (!n.label) n.label = n.ip || n.id;
    });
  }

  // Shift Y positions to give Internal more space
  assignPositions(extNodes, 50, 40);
  assignPositions(dmzNodes, 160, 40);
  assignPositions(intNodes, 280, 70);

  const mappedNodes = [...extNodes, ...dmzNodes, ...intNodes];

  const now = Date.now();
  const LINK_CAPACITY_MBPS = 1000; // 1 Gbps theoretical link

  const mappedEdges = (backendTopo.edges || []).map((e: any) => {
    const id = `${e.src}-${e.dst}-${e.protocol}-${e.dst_port || 0}`;
    const currentBytes = e.bytes || 0;

    let mbps = 0;
    if (edgeStatsCache.has(id)) {
      const prev = edgeStatsCache.get(id)!;
      const dtSec = (now - prev.lastTime) / 1000;
      if (dtSec > 0.1) {
        // Delta bytes -> bits -> megabits
        const deltaBytes = Math.max(0, currentBytes - prev.lastBytes);
        mbps = (deltaBytes * 8) / 1_000_000 / dtSec;
      } else {
        mbps = prev.currentMbps; // Keep previous if time delta is too small (e.g., immediate re-renders)
      }

      // Cache the new values
      edgeStatsCache.set(id, { lastBytes: currentBytes, lastTime: now, currentMbps: mbps });
    } else {
      // First time seeing this edge, start tracking
      edgeStatsCache.set(id, { lastBytes: currentBytes, lastTime: now, currentMbps: 0 });
    }

    const utilPct = Math.min(100, Math.max(0, Math.round((mbps / LINK_CAPACITY_MBPS) * 100)));

    return {
      id,
      source: e.src || e.source,
      target: e.dst || e.target,
      status: utilPct >= 90 ? "saturated" : "active",
      protocol: e.protocol === 6 ? "TCP" : e.protocol === 17 ? "UDP" : "IP",
      bandwidth: Math.round(mbps), // Store live Mbps
      utilization: utilPct
    };
  });

  return {
    nodes: mappedNodes,
    edges: mappedEdges,
    lastUpdated: new Date().toISOString()
  };
}

// ─── REST helpers ────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function fetchStatus(): Promise<SystemStatus> {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return mockFetchStatus();
  }
  const raw = await apiFetch<any>("/status");
  return mapSystemStatus(raw);
}

export async function fetchSite(): Promise<SiteInfo> {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return mockFetchSite();
  }
  return apiFetch<SiteInfo>("/site");
}

export async function fetchTopology(): Promise<Topology> {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return mockFetchTopology();
  }
  const raw = await apiFetch<any>("/topology");
  return mapTopology(raw);
}

export async function sendCommand(command: string): Promise<void> {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return mockSendCommand(command);
  }
  await apiFetch<void>(`/command/${command}`, { method: "POST" });
}

/**
 * Offline analysis of an uploaded capture or flow CSV (`POST /api/replay`,
 * control_backend/main.py:237). Runs fully local — no egress — and the backend
 * forces the SOC rule layer off for this path, so `risk` here is pure model
 * output. Multipart, so no JSON Content-Type header.
 */
export async function uploadReplay(file: File, maxWindows = 200): Promise<ReplayReport> {
  if (import.meta.env.VITE_DEMO_MODE === "true") {
    return mockReplay(file, maxWindows);
  }
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${BASE_URL}/replay?max_windows=${maxWindows}`, { method: "POST", body });
  if (!res.ok) throw new Error(`API /replay → ${res.status} ${res.statusText}`);
  return res.json() as Promise<ReplayReport>;
}

/**
 * Runs one of the built-in captures listed in `REPLAY_SAMPLES`.
 *
 * Against a live backend the sample ships as a static asset, so it is fetched
 * and posted through the same multipart endpoint a dropped file uses — the
 * analysis path is identical either way.
 */
export async function analyseSample(sample: ReplaySample): Promise<ReplayReport> {
  if (import.meta.env.VITE_DEMO_MODE === "true") {
    return mockReplaySample(sample.id);
  }
  const res = await fetch(`/samples/${sample.name}`);
  if (!res.ok) throw new Error(`sample ${sample.name} not available (${res.status})`);
  const blob = await res.blob();
  return uploadReplay(new File([blob], sample.name, { type: blob.type }));
}

export async function sendMitigate(payload: MitigationPayload): Promise<void> {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return mockSendMitigate(payload);
  }
  await apiFetch<void>("/mitigate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ─── Explainability / horizon mapping ────────────────────────────────────────

/**
 * Real horizon of the forecast in seconds: forecast_steps x window_seconds
 * (e.g. 8 x 2.0 = 16s). Falls back to the last forecast point's horizon_seconds.
 */
function horizonSeconds(p: any): number {
  const steps = Number(p?.model?.forecast_steps);
  const win = Number(p?.model?.window_seconds);
  if (Number.isFinite(steps) && Number.isFinite(win) && steps > 0 && win > 0) {
    return steps * win;
  }
  const fc: any[] = Array.isArray(p?.forecast) ? p.forecast : [];
  const last = fc.length ? Number(fc[fc.length - 1]?.horizon_seconds) : NaN;
  return Number.isFinite(last) ? last : 0;
}

/** Normalize schema.py:ExplainabilityPayload off the wire. */
function normalizeExplainability(raw: any): ExplainabilityPayload | null {
  if (!raw || typeof raw !== "object") return null;
  const groups = Array.isArray(raw.groups) ? raw.groups : [];
  const feats = Array.isArray(raw.top_features) ? raw.top_features : [];
  return {
    available: Boolean(raw.available),
    method: raw.method ?? null,
    groups: groups
      .filter((g: any) => g && g.name != null)
      .map((g: any) => ({ name: String(g.name), percentage: Number(g.percentage) || 0 })),
    top_features: feats
      .filter((f: any) => f && f.feature != null)
      .map((f: any) => ({
        feature: String(f.feature),
        score: Number(f.score) || 0,
        group: String(f.group ?? "General"),
      })),
  };
}

/**
 * The model's real Input x Gradient attributions, in the shape the existing
 * signal components render. `weight` is the feature's share of total
 * attribution (0–1); the attribution is a magnitude, so it has no sign —
 * hence direction "neutral". `value` carries the feature group label.
 */
function explainabilitySignals(ex: ExplainabilityPayload | null): ContributingSignal[] {
  if (!ex || !ex.available) return [];
  return ex.top_features.map((f) => ({
    name: f.feature,
    weight: f.score,
    direction: "neutral" as const,
    value: f.group,
  }));
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

export function connectWebSocket(handlers: WSHandlers): any {
  if (import.meta.env.VITE_DEMO_MODE === 'true') {
    return new MockWebSocket(handlers);
  }
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => handlers.onOpen?.();
  ws.onerror = (e) => handlers.onError?.(e);
  ws.onclose = () => handlers.onClose?.();

  ws.onmessage = (event) => {
    let parsed: any;
    try {
      parsed = JSON.parse(event.data as string);
    } catch {
      return;
    }

    // The backend sends events flattened, not inside a 'payload' wrapper
    const p = parsed;
    switch (parsed.type) {
      case "prediction": {
        const explain = normalizeExplainability(p.explainability);
        handlers.onPrediction?.({
          ...p,
          prediction: p.prediction ? {
            // Spread the wire payload first so every PredictionData field
            // survives: alert_level, threshold, predicted_stage, mitre_tactic,
            // mitre_technique, stage_probabilities, stage_provenance, …
            ...p.prediction,
            horizon: horizonSeconds(p),
            value: p.prediction.risk || 0,
            confidence: p.prediction.malicious_confidence || 0,
            branch_a_risk: p.prediction.risk ?? 0,
            branch_b_risk: p.prediction.max_future_risk ?? p.prediction.risk ?? 0,
            model: "cyberworld Ensemble",
            explainability: explain,
            // Real per-feature attributions. The old hardcoded trio (hazard
            // score / forecast error / max future risk) were risk scores, not
            // features — forecast_error is never set by the backend at all.
            signals: explainabilitySignals(explain),
          } : null
        });
        break;
      }
      case "topology_update":
        handlers.onTopologyUpdate?.(mapTopology(p));
        break;
      case "ml_reset":
        handlers.onMLReset?.(p);
        break;
      case "system_status":
        handlers.onSystemStatus?.(mapSystemStatus(p));
        break;
      case "command_started":
        handlers.onCommandStarted?.(p);
        break;
      case "command_completed":
        handlers.onCommandCompleted?.(p);
        break;
      case "command_output":
        handlers.onCommandOutput?.(p);
        break;
      case "attack_event":
        handlers.onAttackEvent?.(p);
        break;
    }
  };

  return ws;
}
