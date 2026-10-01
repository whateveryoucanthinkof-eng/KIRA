/**
 * API / WebSocket adapter layer — the single point of contact with
 * control_backend/. In a demo build (IS_DEMO, `.env.demo`) every call is
 * routed to src/api/mock.ts instead; a live build never reaches the mock.
 */

import type {
  SystemStatus,
  ThreatLevel,
  Topology,
  SiteInfo,
  MitigationPayload,
  WSHandlers,
} from "./types";
import type { ReplayReport } from "../types/replay";
import type { ReplaySample } from "../types/replay";
import {
  mockFetchStatus,
  mockFetchSite,
  mockFetchTopology,
  mockSendCommand,
  mockSendMitigate,
  mockReplay,
  mockReplaySample,
  MockWebSocket,
  REPLAY_SAMPLES,
} from "./mock";
import { IS_DEMO } from "../env";
import { normalizePrediction } from "./normalize";

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
  if (IS_DEMO) {
    return mockFetchStatus();
  }
  const raw = await apiFetch<any>("/status");
  return mapSystemStatus(raw);
}

export async function fetchSite(): Promise<SiteInfo> {
  if (IS_DEMO) {
    return mockFetchSite();
  }
  const raw = await apiFetch<any>("/site");
  // /api/site carries display_name; the header renders `name`.
  return { ...raw, name: raw.display_name ?? raw.site_id ?? "" } as SiteInfo;
}

export async function fetchTopology(): Promise<Topology> {
  if (IS_DEMO) {
    return mockFetchTopology();
  }
  const raw = await apiFetch<any>("/topology");
  return mapTopology(raw);
}

export async function sendCommand(command: string): Promise<void> {
  if (IS_DEMO) {
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
  if (IS_DEMO) {
    return mockReplay(file, maxWindows);
  }
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${BASE_URL}/replay?max_windows=${maxWindows}`, { method: "POST", body });
  if (!res.ok) throw new Error(`API /replay → ${res.status} ${res.statusText}`);
  return res.json() as Promise<ReplayReport>;
}

/**
 * The built-in captures the Replay page offers. Live: whatever is in the
 * repo's captures/ folder (GET /api/replay/samples). Demo: the fixtures.
 */
export async function fetchReplaySamples(): Promise<ReplaySample[]> {
  if (IS_DEMO) return REPLAY_SAMPLES;
  const raw = await apiFetch<any[]>("/replay/samples");
  return raw.map((r) => ({
    id: String(r.id),
    name: String(r.name),
    label: String(r.label ?? r.name),
    kind: r.kind === "csv" ? "csv" : "pcap",
    bytes: Number(r.bytes) || 0,
    // The window count is only known once the capture is parsed.
    windows: null,
    source: String(r.source ?? ""),
    note: "A capture on this machine, scored by the loaded models.",
  }));
}

/**
 * Runs one of the built-in captures. Live, the backend analyses the file in
 * place (POST /api/replay/samples/{name}) through the same path an upload
 * takes.
 */
export async function analyseSample(sample: ReplaySample): Promise<ReplayReport> {
  if (IS_DEMO) {
    return mockReplaySample(sample.id);
  }
  return apiFetch<ReplayReport>(`/replay/samples/${encodeURIComponent(sample.name)}`, { method: "POST" });
}

export async function sendMitigate(payload: MitigationPayload): Promise<void> {
  if (IS_DEMO) {
    return mockSendMitigate(payload);
  }
  await apiFetch<void>("/mitigate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

export function connectWebSocket(handlers: WSHandlers): any {
  if (IS_DEMO) {
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
        handlers.onPrediction?.(normalizePrediction(p));
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
