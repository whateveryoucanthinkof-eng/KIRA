import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Sidebar, { type Page } from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import Overview from "./pages/Overview";
import Network from "./pages/Network";
import Predictions from "./pages/Predictions";
import Events from "./pages/Events";
import Controls from "./pages/Controls";
import Campaign from "./pages/Campaign";
import Attck from "./pages/Attck";
import Replay from "./pages/Replay";
import Model from "./pages/Model";
import Incidents from "./pages/Incidents";
import Palette, { type PaletteItem } from "./components/Palette";
import { Toasts, Shortcuts, useHotkeys, useAlertToasts, type Toast } from "./components/Feedback";
import { fetchStatus, fetchSite, fetchTopology, connectWebSocket, sendMitigate } from "./api/adapter";
import { seedHistory, seedLog, seedEvents, seedIncidents } from "./api/mock";
import { IS_DEMO } from "./env";
import { Enter } from "./design/motion";
import type { Campaign as CampaignData } from "./types/campaign";
import type { Incident } from "./types/incident";
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

const PAGE_TITLES: Record<Page, string> = {
  overview: "Overview",
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

function tickLabel(t: number): string {
  return new Date(t).toLocaleTimeString("en-GB", { hour12: false });
}

function num(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function nullableNum(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** ISO-8601 with a trailing Z — matches control_backend/schema.py:utc_now_iso. */
function nowIso(): string {
  return new Date().toISOString();
}

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  // Seeded in demo mode so the console opens with full charts at rest and the
  // intrusion begins on camera rather than off it. Empty against a live
  // backend, which has no history endpoint to backfill from.
  const [history, setHistory] = useState<LivePoint[]>(() => (IS_DEMO ? seedHistory(90) : []));
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
  const [incidentEdits, setIncidentEdits] = useState<Record<string, Partial<Incident>>>({});

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

  useEffect(() => {
    fetchStatus().then(setStatus).catch(console.warn);
    fetchSite().then(setSite).catch(console.warn);
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

        onSystemStatus: setStatus,

        onTopologyUpdate: setTopology,

        onPrediction: (data: unknown) => {
          const p = data as Record<string, any>;
          const pred = p.prediction as Record<string, any> | null | undefined;

          // Operational context the pages render: pipeline timing, early
          // warning, focus hosts. Kept separate from PredictionData.
          setEnvelope({
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
          });

          // correlation/ is not exposed over the wire yet; the field is
          // present in demo fixtures and simply absent against a live backend.
          if (p.campaign) setCampaign(p.campaign as CampaignData);
          if (Array.isArray(p.incidents)) setIncidents(p.incidents as Incident[]);

          if (Array.isArray(p.forecast)) {
            setForecast(
              p.forecast.map((f: Record<string, any>) => {
                const risk = num(f.risk) * 100;
                const conf = num(f.confidence, 0.5);
                // The band is the backend's split-conformal interval for this
                // step (control_backend/forecast_band.py). It used to be
                // invented here as risk +/- 20 * DeepOP confidence, which
                // measured nothing. No fitted band -> no band.
                const banded = f.risk_lower != null && f.risk_upper != null;
                return {
                  timestamp: new Date(Date.now() + num(f.horizon_seconds) * 1000).toISOString(),
                  horizonSeconds: num(f.horizon_seconds),
                  predictedStage: f.predicted_stage ?? null,
                  predicted: risk,
                  lowerBound: banded ? num(f.risk_lower) * 100 : risk,
                  upperBound: banded ? num(f.risk_upper) * 100 : risk,
                  confidence: conf,
                };
              })
            );
          }

          if (!pred) return;
          setPrediction(pred as PredictionResult);

          const risk = num(pred.risk);
          const maxFuture = num(pred.max_future_risk, risk);
          const t = Date.parse(p.timestamp ?? "") || Date.now();

          // Everything charted below is measured. The previous build scaled
          // the traffic series off the risk score with hardcoded lead/lag
          // offsets; these are the backend's own telemetry fields instead.
          const point: LivePoint = {
            t,
            label: tickLabel(t),
            windowId: p.state?.window_id ?? null,
            risk,
            maxFutureRisk: maxFuture,
            mlRisk: nullableNum(pred.ml_risk),
            ruleRisk: nullableNum(pred.rule_risk),
            throughput: num(p.throughput),
            packets: num(p.state?.packet_count),
            flows: num(p.state?.active_flows),
            latencyMs: num(p.latency?.total_ms, num(p.state?.pipeline_latency_ms)),
          };

          setHistory((prev) => {
            if (prev.length && prev[prev.length - 1].t === point.t && prev[prev.length - 1].windowId === point.windowId) {
              return prev;
            }
            return [...prev, point].slice(-HISTORY);
          });

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
            pushEvent({
              timestamp: p.timestamp,
              severity: String(pred.alert_level).toUpperCase() === "CRITICAL" ? "critical" : "warning",
              category: String(pred.alert_level ?? "ALERT").toUpperCase(),
              source: p.target_ip ?? "—",
              destination: (p.focus_ips ?? [])[0],
              message: pred.mitre_technique
                ? `${pred.mitre_technique} — ${pred.mitre_tactic ?? "unclassified tactic"}`
                : `Risk ${risk.toFixed(3)} crossed threshold ${num(pred.threshold, 0.65).toFixed(2)}`,
              raw: `risk=${risk.toFixed(4)} max_future=${maxFuture.toFixed(4)} ml=${
                pred.ml_risk != null ? num(pred.ml_risk).toFixed(4) : "—"
              } rule=${pred.rule_risk != null ? num(pred.rule_risk).toFixed(4) : "—"} source=${
                pred.risk_source ?? "model"
              }`,
            });
          }
        },

        onMLReset: () => {
          setPrediction(null);
          setEnvelope(null);
          setForecast([]);
          setHistory([]);
          pushEvent({ severity: "info", category: "SYSTEM", source: "ml", message: "Inference state reset" });
        },

        onCommandStarted: (d) => {
          pushLog(`[${nowIso()}] cmd  INFO: started ${d.command}`);
          pushEvent({ severity: "info", category: "COMMAND", source: d.command, message: `Command started: ${d.command}` });
        },

        onCommandCompleted: (d) => {
          pushLog(`[${nowIso()}] cmd  INFO: completed ${d.command}`);
          pushEvent({ severity: "info", category: "COMMAND", source: d.command, message: `Command completed: ${d.command}` });

          fetchStatus().then(setStatus).catch(() => {});

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
  }, [pushEvent, pushLog]);

  /* ── Feedback + keyboard ───────────────────────────────────────────── */

  const pushToast = useCallback((t: Toast) => {
    setToasts((prev) => [...prev.slice(-2), t]);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const updateIncident = useCallback((id: string, patch: Partial<Incident>) => {
    setIncidentEdits((prev) => ({ ...prev, [id]: { ...(prev[id] ?? {}), ...patch } }));
  }, []);

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
    <div className="app">
      <Sidebar activePage={page} onNavigate={setPage} status={status} />

      <div className="app-main">
        {IS_DEMO && (
          // `npm run demo` serves scripted fixtures from src/api/mock.ts. It
          // must never be mistakable for the live system, on any page.
          <div
            role="status"
            style={{
              background: "var(--sev-warning)",
              color: "#000",
              padding: "6px 16px",
              font: "600 12px/1.4 system-ui, sans-serif",
              letterSpacing: "0.04em",
              textAlign: "center",
            }}
          >
            DEMO DATA — scripted fixtures from src/api/mock.ts. No backend, no sensor, no model is running.
          </div>
        )}
        <Header
          status={status}
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
                status={status}
                history={history}
                forecast={forecast}
                prediction={prediction}
                envelope={envelope}
                topology={topology}
                events={events}
                logLines={logLines}
              />
            )}
            {page === "network" && <Network topology={topology} envelope={envelope} />}
            {page === "predictions" && (
              <Predictions history={history} forecast={forecast} prediction={prediction} envelope={envelope} />
            )}
            {page === "campaign" && <Campaign campaign={campaign} prediction={prediction} />}
            {page === "attck" && <Attck prediction={prediction} forecast={forecast} campaign={campaign} />}
            {page === "replay" && <Replay />}
            {page === "incidents" && (
              <Incidents
                incidents={incidents}
                overrides={incidentEdits}
                onUpdate={updateIncident}
                onOpenHost={openHost}
              />
            )}
            {page === "model" && <Model status={status} prediction={prediction} envelope={envelope} />}
            {page === "events" && <Events events={events} attackEvents={attackEvents} logLines={logLines} />}
            {page === "controls" && <Controls status={status} logLines={logLines} onLog={pushLog} />}
          </Enter>
        </main>
      </div>

      <Palette items={paletteItems} open={paletteOpen} onOpenChange={setPaletteOpen} />
      <Shortcuts open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
