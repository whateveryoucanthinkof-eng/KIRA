import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Sidebar, { type Page } from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import Overview from "./pages/Overview";
import Network from "./pages/Network";
import Predictions from "./pages/Predictions";
import Events from "./pages/Events";
import Controls, { type ScenarioId } from "./pages/Controls";
import Campaign from "./pages/Campaign";
import Attck from "./pages/Attck";
import Replay from "./pages/Replay";
import Model from "./pages/Model";
import Incidents from "./pages/Incidents";
import Investigation from "./pages/Investigation";
import Stage from "./pages/Stage";
import Palette, { type PaletteItem } from "./components/Palette";
import { Toasts, Shortcuts, useHotkeys, useAlertToasts, type Toast } from "./components/Feedback";
import { fetchStatus, fetchSite, fetchTopology, connectWebSocket, sendCommand, sendMitigate } from "./api/adapter";
import { seedFrames, seedLog, seedEvents, seedIncidents } from "./api/mock";
import { normalizePrediction } from "./api/normalize";
import { IS_DEMO } from "./env";
import { Enter } from "./design/motion";
import type { Campaign as CampaignData } from "./types/campaign";
import type { Incident, IncidentEdit } from "./types/incident";
import type { FlowRecord, StateDim } from "./types/evidence";
import type { ForecastBranch } from "./types/forecast";
import type { AttentionMatrix } from "./types/attention";
import { FORECAST_STEP_SECONDS, HORIZON_SECONDS, WINDOW_SECONDS, setContract, type Cursor, type Frame } from "./types/timeline";
import { setSite as setSiteGeometry } from "./design/site";
import type {
  SystemStatus,
  ForecastPoint,
  PredictionResult,
  Topology,
  NetworkEvent,
  AttackEvent,
  SiteInfo,
} from "./api/types";
import type { LivePoint, PredictionEnvelope, Theme } from "./types/live";
import { ahead, clockTime } from "./design/time";
import { chime } from "./design/sound";

const PAGE_TITLES: Record<Page, string> = {
  overview: "Overview",
  stage: "Forecast Stage",
  investigation: "Investigation",
  network: "Network",
  predictions: "Predictions",
  campaign: "Campaign",
  attck: "ATT&CK",
  replay: "Replay",
  incidents: "Incidents",
  events: "Events",
  model: "Model",
  controls: "Controls",
};

/** Rolling window of inference history kept client-side. */
const HISTORY = 90;
const LOG_LINES = 400;
const EVENT_ROWS = 300;
/** Flow rows retained across windows for the evidence tables. */
const FLOW_ROWS = 240;

function tickLabel(t: number): string {
  return clockTime(t);
}

function num(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function nullableNum(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/**
 * One window of the prediction stream, parsed once. The live stream and the
 * time-travel backfill go through the same function, so a backfilled frame is
 * indistinguishable from a streamed one.
 */
function parseWindow(p: Record<string, any>) {
  const pred = p.prediction as Record<string, any> | null | undefined;
  const base = Date.parse(p.timestamp ?? "") || Date.now();

  // Operational context the pages render: pipeline timing, early warning,
  // focus hosts. Kept separate from PredictionData.
  const envelope: PredictionEnvelope = {
    mode: p.mode,
    timestamp: p.timestamp,
    wall_clock: p.wall_clock,
    state: p.state,
    latency: p.latency,
    early_warning: p.early_warning ?? null,
    attack_active: Boolean(p.attack_active),
    attack_phase: p.attack_phase ?? null,
    focus_ips: Array.isArray(p.focus_ips) ? p.focus_ips : [],
    target_ip: p.target_ip ?? null,
    throughput: num(p.throughput),
  };

  const forecast: ForecastPoint[] | null = Array.isArray(p.forecast)
    ? p.forecast.map((f: Record<string, any>) => {
        const risk = num(f.risk) * 100;
        const conf = num(f.confidence, 0.5);
        // The band is the backend's split-conformal interval on risk
        // (control_backend/forecast_band.py), fitted on validation. When the
        // checkpoint carries none, there is no band: the bounds collapse onto
        // the forecast rather than being invented from the decoder's
        // confidence, which is a token probability, not a risk interval.
        const banded = f.risk_lower != null && f.risk_upper != null;
        return {
          timestamp: new Date(base + num(f.horizon_seconds) * 1000).toISOString(),
          horizonSeconds: num(f.horizon_seconds),
          // Lane for the kill chain; the DeepOP token itself as technique.
          predictedStage: f.tactic_lane ?? f.predicted_stage ?? null,
          technique: f.predicted_stage ?? null,
          predicted: risk,
          lowerBound: banded ? num(f.risk_lower) * 100 : risk,
          upperBound: banded ? num(f.risk_upper) * 100 : risk,
          banded,
          confidence: conf,
        };
      })
    : null;

  let point: LivePoint | null = null;
  if (pred) {
    const risk = num(pred.risk);
    // Everything charted is measured: the backend's own telemetry fields.
    point = {
      t: base,
      label: tickLabel(base),
      windowId: p.state?.window_id ?? null,
      risk,
      maxFutureRisk: num(pred.max_future_risk, risk),
      mlRisk: nullableNum(pred.ml_risk),
      ruleRisk: nullableNum(pred.rule_risk),
      throughput: num(p.throughput),
      packets: num(p.state?.packet_count),
      flows: num(p.state?.active_flows),
      latencyMs: num(p.latency?.total_ms, num(p.state?.pipeline_latency_ms)),
    };
  }

  return {
    envelope,
    prediction: pred ? (pred as PredictionResult) : null,
    forecast,
    // correlation/ is not exposed over the wire yet; present in demo fixtures,
    // absent against a live backend.
    campaign: p.campaign ? (p.campaign as CampaignData) : null,
    incidents: Array.isArray(p.incidents) ? (p.incidents as Incident[]) : null,
    flows: Array.isArray(p.flows) ? (p.flows as FlowRecord[]) : null,
    flowsInWindow: p.flows_in_window != null ? num(p.flows_in_window) : null,
    stateVector: Array.isArray(p.state_vector) ? (p.state_vector as StateDim[]) : null,
    branches: Array.isArray(p.branches) ? (p.branches as ForecastBranch[]) : null,
    attention: p.attention && Array.isArray(p.attention.alpha) ? (p.attention as AttentionMatrix) : null,
    point,
  };
}

type ParsedWindow = ReturnType<typeof parseWindow>;

function toFrame(seq: number, w: ParsedWindow, topology: Topology | null): Frame | null {
  if (!w.prediction || !w.point) return null;
  return {
    seq,
    point: w.point,
    prediction: w.prediction,
    envelope: w.envelope,
    forecast: w.forecast ?? [],
    campaign: w.campaign,
    topology,
    stateVector: w.stateVector,
    flows: w.flows ?? [],
    flowsInWindow: w.flowsInWindow,
    branches: w.branches,
    attention: w.attention,
  };
}

/** Demo scenario names for the console log (pages/Controls.tsx:SCENARIOS). */
const SCENARIO_LOG: Record<ScenarioId, string> = {
  recon: "RECON_BURST",
  probe: "CREDENTIAL_STUFFING",
  exploit: "EXPLOIT_ATTEMPT",
  c2: "C2_BEACONING",
  flood: "VOLUMETRIC_FLOOD",
};

/** Replay runs at twice the stream's rate, so it catches up with live. */
const REPLAY_STEP_MS = 415;

/** ISO-8601 with a trailing Z — matches control_backend/schema.py:utc_now_iso. */
function nowIso(): string {
  return new Date().toISOString();
}

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  // Every window, kept whole for time travel; the charts' history is derived
  // from it. Backfilled in demo mode so the console opens with full charts at
  // rest and the intrusion begins on camera. Empty against a live backend,
  // which has no history endpoint to backfill from.
  const frameSeq = useRef(0);
  const [frames, setFrames] = useState<Frame[]>(() =>
    IS_DEMO
      ? (seedFrames(HISTORY)
          .map(({ payload, topology }) => toFrame(++frameSeq.current, parseWindow(normalizePrediction(payload)), topology))
          .filter(Boolean) as Frame[])
      : []
  );
  const history = useMemo(() => frames.map((f) => f.point), [frames]);
  const [cursor, setCursor] = useState<Cursor>({ kind: "live" });
  const [playing, setPlaying] = useState(false);
  const [campaign, setCampaign] = useState<CampaignData | null>(null);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [envelope, setEnvelope] = useState<PredictionEnvelope | null>(null);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [events, setEvents] = useState<NetworkEvent[]>(() =>
    IS_DEMO
      ? seedEvents()
          .map((e, i): NetworkEvent => ({
            id: `seed${i}`,
            timestamp: new Date(Date.now() - e.ageMs).toISOString(),
            severity: e.severity,
            category: e.category,
            source: e.source,
            destination: e.destination,
            message: e.message,
            raw: e.raw,
          }))
          .reverse() // newest first, matching pushEvent
      : []
  );
  const [attackEvents, setAttackEvents] = useState<AttackEvent[]>([]);
  const [logLines, setLogLines] = useState<string[]>(() => (IS_DEMO ? seedLog(24) : []));
  const [site, setSite] = useState<SiteInfo | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>(() => (IS_DEMO ? seedIncidents() : []));
  // Analyst actions are held separately so they survive the incident
  // stream replacing the queue on every window.
  // Evidence: a rolling buffer of recent flows, and the full 27-D state of
  // the latest window. Both are demo-only until the backend forwards them.
  const [flows, setFlows] = useState<FlowRecord[]>([]);
  const [flowsInWindow, setFlowsInWindow] = useState<number | null>(null);
  const [stateVector, setStateVector] = useState<StateDim[] | null>(null);
  const [incidentEdits, setIncidentEdits] = useState<Record<string, IncidentEdit>>({});
  // An incident another page asked to open; cleared once the viewer moves on.
  const [focusIncident, setFocusIncident] = useState<string | null>(null);

  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("soc_theme");
    return saved === "paper" ? "paper" : "ink";
  });

  const wsRef = useRef<WebSocket | null>(null);
  const eventSeq = useRef(0);

  /* ── Theme ─────────────────────────────────────────────────────────── */

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "ink" ? "paper" : "ink";
      localStorage.setItem("soc_theme", next);
      return next;
    });
  }, []);

  /* ── Event log ─────────────────────────────────────────────────────── */

  const pushEvent = useCallback((ev: Omit<NetworkEvent, "id" | "timestamp"> & { timestamp?: string }) => {
    eventSeq.current += 1;
    const row: NetworkEvent = {
      id: `e${eventSeq.current}`,
      timestamp: ev.timestamp ?? nowIso(),
      severity: ev.severity,
      category: ev.category,
      source: ev.source,
      destination: ev.destination,
      message: ev.message,
      protocol: ev.protocol,
      port: ev.port,
      raw: ev.raw,
    };
    setEvents((prev) => [row, ...prev].slice(0, EVENT_ROWS));
  }, []);

  const pushLog = useCallback((line: string) => {
    setLogLines((prev) => [...prev, line].slice(-LOG_LINES));
  }, []);

  /* ── Uptime ticker ─────────────────────────────────────────────────── */

  useEffect(() => {
    const timer = setInterval(() => {
      setStatus((prev) => (prev ? { ...prev, uptime: prev.uptime + 1 } : prev));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  /* ── Backend ───────────────────────────────────────────────────────── */

  /** A status from REST or the bus. Bus pushes carry only the live counters,
   * so they are merged over the last full status rather than replacing it
   * (which would drop model_loaded, model_meta and the contract). */
  const applyStatus = useCallback((s: SystemStatus) => {
    if (s.contract) setContract(s.contract);
    setStatus((prev) => (prev ? { ...prev, ...s } : s));
  }, []);

  useEffect(() => {
    fetchStatus().then(applyStatus).catch(console.warn);
    fetchSite()
      .then((s) => {
        setSiteGeometry(s);
        setSite(s);
      })
      .catch(console.warn);
    fetchTopology().then(setTopology).catch(console.warn);

    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    function connect() {
      const ws = connectWebSocket({
        onOpen: () => {
          setWsConnected(true);
          pushLog(`[${nowIso()}] bus INFO: connected to real-time event bus`);
          pushEvent({ severity: "info", category: "SYSTEM", source: "bus", message: "Connected to real-time event bus" });
        },

        onClose: () => {
          setWsConnected(false);
          pushLog(`[${nowIso()}] bus ERROR: connection lost — retrying in 3s`);
          pushEvent({ severity: "error", category: "SYSTEM", source: "bus", message: "Backend connection lost, reconnecting" });
          reconnectTimer = setTimeout(connect, 3000);
        },

        onError: () => {
          /* onClose drives reconnection */
        },

        onSystemStatus: applyStatus,

        onTopologyUpdate: (t: Topology) => {
          setTopology(t);
          // The window's topology arrives just after its prediction.
          setFrames((prev) => (prev.length ? [...prev.slice(0, -1), { ...prev[prev.length - 1], topology: t }] : prev));
        },

        onPrediction: (data: unknown) => {
          const p = data as Record<string, any>;
          const pred = p.prediction as Record<string, any> | null | undefined;
          const w = parseWindow(p);

          setEnvelope(w.envelope);
          if (w.campaign) setCampaign(w.campaign);
          if (w.incidents) setIncidents(w.incidents);
          if (w.flows) {
            const incoming = w.flows;
            setFlows((prev) => [...prev, ...incoming].slice(-FLOW_ROWS));
          }
          if (w.flowsInWindow != null) setFlowsInWindow(w.flowsInWindow);
          if (w.stateVector) setStateVector(w.stateVector);
          if (w.forecast) setForecast(w.forecast);

          if (!pred || !w.prediction || !w.point) return;
          setPrediction(w.prediction);

          const risk = w.point.risk;
          const maxFuture = w.point.maxFutureRisk;
          const frame = toFrame(++frameSeq.current, w, null);
          if (frame) {
            setFrames((prev) => {
              const last = prev[prev.length - 1];
              // Carry the topology forward until this window's update lands.
              return [...prev, { ...frame, topology: last?.topology ?? null }].slice(-HISTORY);
            });
          }

          setStatus((prev) =>
            prev
              ? {
                  ...prev,
                  anomalyScore: Math.round(Math.max(risk, maxFuture) * 1000) / 10,
                  // alert_level is NOMINAL|WARNING|ELEVATED|CRITICAL — all
                  // members of ThreatLevel once lowercased.
                  threatLevel: String(pred.alert_level ?? prev.threatLevel).toLowerCase() as SystemStatus["threatLevel"],
                  throughput: num(p.throughput, prev.throughput),
                  activeConnections: num(p.state?.active_flows, prev.activeConnections),
                  latency: num(p.latency?.total_ms, prev.latency),
                }
              : prev
          );

          // An alert is an event. This is what fills the Events page.
          if (pred.alert) {
            const lvl = String(pred.alert_level ?? "ALERT").toUpperCase();
            pushEvent({
              timestamp: p.timestamp,
              severity: lvl === "CRITICAL" ? "critical" : "warning",
              // Raised by the forecast peak while the observed level is still nominal.
              category: lvl === "NOMINAL" ? "FORECAST" : lvl,
              // Attacker → target, as the analyst reads it.
              source: ((p.focus_ips ?? []) as string[]).find((ip) => ip !== p.target_ip) ?? p.target_ip ?? "—",
              destination: p.target_ip ?? undefined,
              message: pred.mitre_technique
                ? `${pred.mitre_technique} — ${pred.mitre_tactic ?? "unclassified tactic"}`
                : `Risk ${risk.toFixed(3)} crossed threshold ${num(pred.threshold, 0.65).toFixed(2)}`,
              raw: `risk=${risk.toFixed(4)} max_future=${maxFuture.toFixed(4)} ml=${
                pred.ml_risk != null ? num(pred.ml_risk).toFixed(4) : "—"
              } rule=${pred.rule_risk != null ? num(pred.rule_risk).toFixed(4) : "—"} rules_applied=${Boolean(
                pred.rules_applied
              )}`,
            });
          }
        },

        onMLReset: () => {
          setPrediction(null);
          setEnvelope(null);
          setForecast([]);
          setFrames([]);
          setCursor({ kind: "live" });
          setFlows([]);
          setStateVector(null);
          pushEvent({ severity: "info", category: "SYSTEM", source: "ml", message: "Inference state reset" });
        },

        onCommandStarted: (d) => {
          pushLog(`[${nowIso()}] cmd  INFO: started ${d.command}`);
          pushEvent({ severity: "info", category: "COMMAND", source: d.command, message: `Command started: ${d.command}` });
        },

        onCommandCompleted: (d) => {
          pushLog(`[${nowIso()}] cmd  INFO: completed ${d.command}`);
          pushEvent({ severity: "info", category: "COMMAND", source: d.command, message: `Command completed: ${d.command}` });

          fetchStatus().then(applyStatus).catch(() => {});

          if (d.command === "reset_environment" || d.command === "stop_network" || d.command === "stop_attack") {
            setAttackEvents([]);
          }
          if (d.command === "reset_environment" || d.command === "stop_telemetry") {
            fetchTopology().then(setTopology).catch(() => {});
          }
        },

        onCommandOutput: (d) => pushLog(d.line),

        onAttackEvent: (d) => {
          const stage = String(d.stage ?? "");
          if (stage === "STOPPED" || stage.includes("TERMINATED")) {
            setAttackEvents([]);
            return;
          }
          setAttackEvents((prev) => {
            const i = prev.findIndex((e) => e.id === d.id);
            if (i >= 0) {
              const next = [...prev];
              next[i] = { ...next[i], ...d };
              return next;
            }
            return [...prev, d];
          });
          pushEvent({
            severity: d.severity ?? "warning",
            category: "ATTACK",
            source: d.sourceIp ?? "—",
            destination: d.targetIp,
            message: `${d.type ?? "Attack"} — stage ${stage}`,
          });
        },
      });

      wsRef.current = ws;
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [pushEvent, pushLog, applyStatus]);

  /* ── Time travel ───────────────────────────────────────────────────── */

  const framesRef = useRef(frames);
  framesRef.current = frames;
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;

  // Playback: step forward through recorded frames until live, or through
  // the forecast horizon until its end.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const fs = framesRef.current;
      const c = cursorRef.current;
      if (c.kind === "past") {
        const i = fs.findIndex((f) => f.seq === c.seq);
        if (i < 0 || i >= fs.length - 2) {
          setCursor({ kind: "live" });
          setPlaying(false);
        } else {
          setCursor({ kind: "past", seq: fs[i + 1].seq });
        }
      } else if (c.kind === "future") {
        if (c.seconds >= HORIZON_SECONDS) setPlaying(false);
        else setCursor({ kind: "future", seconds: Math.min(HORIZON_SECONDS, c.seconds + FORECAST_STEP_SECONDS) });
      } else {
        setPlaying(false);
      }
    }, REPLAY_STEP_MS);
    return () => clearInterval(id);
  }, [playing]);

  const liveIdx = frames.length - 1;
  let viewIdx = liveIdx;
  if (cursor.kind === "past") {
    const i = frames.findIndex((f) => f.seq === cursor.seq);
    viewIdx = i >= 0 ? i : 0;
  }
  const replaying = cursor.kind === "past" && viewIdx < liveIdx;
  const vf = replaying ? frames[viewIdx] : null;

  const view = useMemo(
    () => ({
      prediction: vf ? vf.prediction : prediction,
      envelope: vf ? vf.envelope : envelope,
      forecast: vf ? vf.forecast : forecast,
      campaign: vf ? vf.campaign : campaign,
      topology: vf ? (vf.topology ?? topology) : topology,
      stateVector: vf ? vf.stateVector : stateVector,
      flows: vf ? frames.slice(Math.max(0, viewIdx - 3), viewIdx + 1).flatMap((f) => f.flows) : flows,
      flowsInWindow: vf ? vf.flowsInWindow : flowsInWindow,
      history: vf ? history.slice(0, viewIdx + 1) : history,
      branches: vf ? vf.branches : (frames[liveIdx]?.branches ?? null),
      attention: vf ? vf.attention : (frames[liveIdx]?.attention ?? null),
    }),
    [vf, viewIdx, liveIdx, frames, history, prediction, envelope, forecast, campaign, topology, stateVector, flows, flowsInWindow]
  );

  const goLive = useCallback(() => {
    setCursor({ kind: "live" });
    setPlaying(false);
  }, []);

  // The header says when the console is not showing the present.
  const timeline =
    replaying
      ? { label: `Replay · t −${(liveIdx - viewIdx) * WINDOW_SECONDS}s`, onLive: goLive }
      : cursor.kind === "future"
        ? { label: `Forecast · t ${ahead(cursor.seconds)}`, onLive: goLive }
        : null;
  const viewStatus =
    replaying && status && view.prediction?.alert_level
      ? { ...status, threatLevel: String(view.prediction.alert_level).toLowerCase() as SystemStatus["threatLevel"] }
      : status;

  /* ── Adversary emulation ───────────────────────────────────────────── */

  const emulate = useCallback(
    (kind: ScenarioId | "contain") => {
      goLive();
      if (kind === "contain") {
        const target = envelope?.target_ip;
        if (target) void sendMitigate({ action: "ISOLATE_HOST", target } as never);
        pushLog(`[${nowIso()}] ctrl INFO: containment — isolate ${target ?? "primary target"}`);
        return;
      }
      // Attack scenarios exist only in the demo build: the real console never
      // launches traffic. Lab attacks run outside the console, against the
      // Containerlab range, and the console watches for them.
      if (!IS_DEMO) return;
      void sendCommand(`emulate_${kind}`);
      pushLog(`[${nowIso()}] emu  INFO: demo scenario → ${SCENARIO_LOG[kind]}`);
    },
    [envelope, goLive, pushLog]
  );

  /* ── Feedback + keyboard ───────────────────────────────────────────── */

  const pushToast = useCallback((t: Toast) => {
    setToasts((prev) => [...prev.slice(-2), t]);
    chime(); // silent unless the viewer has turned sound on
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const incidentsRef = useRef(incidents);
  incidentsRef.current = incidents;
  const updateIncident = useCallback((id: string, patch: IncidentEdit) => {
    const count = incidentsRef.current.find((i) => i.id === id)?.alertCount ?? 0;
    setIncidentEdits((prev) => {
      const cur = prev[id];
      // Edits from an earlier life of this incident are dropped, not merged.
      const base = cur && (cur.since == null || count >= cur.since) ? cur : {};
      return { ...prev, [id]: { ...base, ...patch, since: base.since ?? count } };
    });
  }, []);

  const openIncident = useCallback((id: string) => {
    setFocusIncident(id);
    setPage("incidents");
  }, []);

  useEffect(() => {
    if (page !== "incidents") setFocusIncident(null);
  }, [page]);

  const openHost = useCallback((ip: string) => {
    setPage("network");
    pushLog(`[${nowIso()}] ui   INFO: opened host ${ip}`);
  }, [pushLog]);

  useAlertToasts(
    prediction?.alert_level,
    { technique: prediction?.mitre_technique, host: envelope?.target_ip },
    pushToast
  );

  const pageIds = Object.keys(PAGE_TITLES) as Page[];

  useHotkeys({
    onIndex: useCallback((i: number) => {
      if (pageIds[i]) setPage(pageIds[i]);
    }, [pageIds]),
    onStep: useCallback((d: number) => {
      setPage((cur) => {
        const i = pageIds.indexOf(cur);
        return pageIds[(i + d + pageIds.length) % pageIds.length];
      });
    }, [pageIds]),
    onTheme: toggleTheme,
    onShortcuts: useCallback(() => setShortcutsOpen((o) => !o), []),
    onEscape: useCallback(() => {
      setShortcutsOpen(false);
      setPaletteOpen(false);
    }, []),
  });

  /* ── Command palette ───────────────────────────────────────────────── */

  const paletteItems = useMemo<PaletteItem[]>(() => {
    const nav = (Object.keys(PAGE_TITLES) as Page[]).map((id, i) => ({
      id: `nav:${id}`,
      group: "Go to",
      label: PAGE_TITLES[id],
      hint: String(i + 1).padStart(2, "0"),
      run: () => setPage(id),
    }));

    return [
      ...nav,
      {
        id: "shortcuts",
        group: "View",
        label: "Keyboard reference",
        hint: "?",
        run: () => setShortcutsOpen(true),
      },
      {
        id: "theme",
        group: "View",
        label: theme === "ink" ? "Switch to paper theme" : "Switch to ink theme",
        hint: theme,
        run: toggleTheme,
      },
      {
        id: "isolate",
        group: "Respond",
        label: "Isolate primary target",
        hint: envelope?.target_ip ?? "no target",
        danger: true,
        run: () => {
          setPage("controls");
          if (envelope?.target_ip) {
            void sendMitigate({ action: "ISOLATE_HOST", target: envelope.target_ip } as never);
            pushLog(`[${nowIso()}] ctrl INFO: isolate ${envelope.target_ip} requested from palette`);
          }
        },
      },
      {
        id: "clear",
        group: "Respond",
        label: "Clear all defenses",
        run: () => {
          void sendMitigate({ action: "CLEAR", target: null } as never);
          pushLog(`[${nowIso()}] ctrl INFO: clear defenses requested from palette`);
        },
      },
    ];
  }, [theme, toggleTheme, envelope, pushLog]);

  return (
    <div className="app" data-page={page}>
      <Sidebar activePage={page} onNavigate={setPage} status={status} />

      <div className="app-main">
        <Header
          status={viewStatus}
          timeline={timeline}
          site={site}
          wsConnected={wsConnected}
          pageTitle={PAGE_TITLES[page]}
          theme={theme}
          onToggleTheme={toggleTheme}
        />

        <main className="page">
          <Enter k={page}>
            {page === "overview" && (
              <Overview
                status={viewStatus}
                history={view.history}
                forecast={view.forecast}
                prediction={view.prediction}
                envelope={view.envelope}
                topology={view.topology}
                campaign={view.campaign}
                branches={view.branches}
                incidents={incidents}
                edits={incidentEdits}
                onOpenIncident={openIncident}
              />
            )}
            {page === "investigation" && (
              <Investigation
                history={view.history}
                forecast={view.forecast}
                prediction={view.prediction}
                envelope={view.envelope}
                campaign={view.campaign}
                topology={view.topology}
                flows={view.flows}
                flowsInWindow={view.flowsInWindow}
                stateVector={view.stateVector}
              />
            )}
            {page === "stage" && (
              <Stage
                view={view}
                frames={frames}
                viewIdx={viewIdx}
                cursor={cursor}
                playing={playing}
                onCursor={setCursor}
                onPlaying={setPlaying}
                onEmulate={emulate}
              />
            )}
            {page === "network" && (
              <Network
                topology={view.topology}
                envelope={view.envelope}
                flows={view.flows}
                flowsInWindow={view.flowsInWindow}
                campaign={view.campaign}
                prediction={view.prediction}
              />
            )}
            {page === "predictions" && (
              <Predictions history={view.history} forecast={view.forecast} prediction={view.prediction} envelope={view.envelope} />
            )}
            {page === "campaign" && (
              <Campaign
                campaign={view.campaign}
                prediction={view.prediction}
                envelope={view.envelope}
                branches={view.branches}
                incidents={incidents}
                edits={incidentEdits}
                onUpdateIncident={updateIncident}
                onToast={pushToast}
                topology={view.topology}
              />
            )}
            {page === "attck" && (
              <Attck
                prediction={view.prediction}
                forecast={view.forecast}
                campaign={view.campaign}
                branches={view.branches}
                flows={view.flows}
                envelope={view.envelope}
                topology={view.topology}
              />
            )}
            {page === "replay" && <Replay />}
            {page === "incidents" && (
              <Incidents
                incidents={incidents}
                overrides={incidentEdits}
                onUpdate={updateIncident}
                onOpenHost={openHost}
                onToast={pushToast}
                prediction={prediction}
                envelope={envelope}
                campaign={campaign}
                branches={frames[liveIdx]?.branches ?? null}
                flows={flows}
                topology={topology}
                focusId={focusIncident}
              />
            )}
            {page === "model" && <Model status={viewStatus} prediction={view.prediction} envelope={view.envelope} />}
            {page === "events" && <Events events={events} attackEvents={attackEvents} logLines={logLines} onNavigate={setPage} />}
            {page === "controls" && (
              <Controls
                status={status}
                logLines={logLines}
                onLog={pushLog}
                prediction={prediction}
                envelope={envelope}
                onEmulate={emulate}
                onNavigate={setPage}
              />
            )}
          </Enter>
        </main>
      </div>

      <Palette items={paletteItems} open={paletteOpen} onOpenChange={setPaletteOpen} />
      <Shortcuts open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
