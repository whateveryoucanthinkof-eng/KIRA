import { useState, useEffect, useRef } from "react";
import Sidebar, { type Page } from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import Overview from "./pages/Overview";
import Network from "./pages/Network";
import Predictions from "./pages/Predictions";
import Events from "./pages/Events";
import Controls from "./pages/Controls";
import {
  fetchStatus,
  fetchSite,
  fetchTopology,
  connectWebSocket,
} from "./api/adapter";
import type {
  SystemStatus,
  TelemetryPoint,
  ForecastPoint,
  PredictionResult,
  Topology,
  NetworkEvent,
  AttackEvent,
  SiteInfo,
} from "./api/types";

const PAGE_TITLES: Record<Page, string> = {
  overview: "Overview",
  network: "Network Topology",
  predictions: "Predictions & Forecasting",
  events: "Events & Logs",
  controls: "System Controls",
};

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [events, setEvents] = useState<NetworkEvent[]>([]);
  const [attackEvents, setAttackEvents] = useState<AttackEvent[]>([]);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [site, setSite] = useState<SiteInfo | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem("soc_theme");
    return saved !== null ? saved === "dark" : true;
  });

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const toggleDark = () => {
    setDarkMode((d) => {
      const next = !d;
      localStorage.setItem("soc_theme", next ? "dark" : "light");
      document.documentElement.classList.toggle("dark", next);
      return next;
    });
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setStatus((prev) => (prev ? { ...prev, uptime: prev.uptime + 1 } : prev));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchStatus().then(setStatus).catch(console.warn);
    fetchSite().then(setSite).catch(console.warn);
    fetchTopology().then(setTopology).catch(console.warn);

    let reconnectTimer: any;
    function connect() {
      const ws = connectWebSocket({
        onOpen: () => {
          setWsConnected(true);
          setLogLines((prev) => [...prev, `[${new Date().toISOString()}] sys INFO: Connected to Real-Time Event Bus`].slice(-100));
        },
        onClose: () => {
          setWsConnected(false);
          setLogLines((prev) => [...prev, `[${new Date().toISOString()}] sys ERROR: Backend connection lost, reconnecting...`].slice(-100));
          reconnectTimer = setTimeout(connect, 3000);
        },
        onError: () => {
          // onClose will handle reconnection
        },
        onSystemStatus: (data) => {
          setStatus(data);
        },
        onTopologyUpdate: (data) => {
          setTopology(data);
        },
        onPrediction: (data: any) => {
          if (data.prediction) {
            setPrediction(data.prediction);
            const branchA = data.prediction.branch_a_risk ?? data.prediction.risk ?? data.prediction.value ?? 0;
            const branchB = data.prediction.branch_b_risk ?? data.prediction.max_future_risk ?? branchA;
            const riskVal = Math.max(branchA, branchB, data.prediction.value || 0);

            const anomalyScore = Math.round(riskVal * 100);
            const rawAlert = String(data.prediction.alert_level || (riskVal >= 0.85 ? "critical" : riskVal >= 0.65 ? "high" : riskVal >= 0.4 ? "medium" : "low")).toLowerCase();
            const normThreat = rawAlert === "critical" ? "critical" : (rawAlert === "high" || rawAlert === "elevated") ? "high" : (rawAlert === "medium" || rawAlert === "warning") ? "medium" : "low";

            // Orange Line = Branch B Predicted (forecast of future risk and predicted attack traffic surge)
            const predVal = (riskVal >= 0.35)
              ? Math.min(950, Math.round(riskVal * 900 + 40))
              : (data.throughput && data.throughput > 0 ? Math.round(data.throughput) : (data.state?.active_flows ? Math.round(data.state.active_flows * 0.35 + 20) : 35));

            // Blue Line = Observed Traffic & Threat Detection (restoring pre-1709ee6 behavior from f43d6f6)
            // When attack is active or high risk detected, observed surges to track the attack flood alongside prediction (~850-920M).
            // When calm/baseline, observed sits cleanly at baseline (~35-80M).
            const isAttackActive = (riskVal >= 0.35) || (branchA >= 0.25) || (data.anomalyScore && data.anomalyScore >= 35) || ((data.state?.packet_count || 0) > 300);

            const baselineObs = (data.state?.active_flows !== undefined && data.state.active_flows > 0)
              ? Math.min(100, Math.max(35, Math.round(data.state.active_flows * 0.35 + 20)))
              : (data.throughput && data.throughput > 0 && data.throughput < 150 ? Math.round(data.throughput) : 35);

            const attackObs = (data.state?.packet_count && data.state.packet_count >= 200)
              ? Math.min(950, Math.max(820, Math.round(data.state.packet_count * 1.8 + 100)))
              : Math.min(950, Math.round(Math.max(branchA, riskVal * 0.96) * 880 + 40));

            const obsVal = isAttackActive ? attackObs : baselineObs;

            const wireThroughput = obsVal;

            setStatus((prev) => (prev ? {
              ...prev,
              anomalyScore,
              threatLevel: normThreat,
              throughput: wireThroughput,
              activeConnections: data.state?.active_flows || prev.activeConnections,
            } : prev));

            setTelemetry((prev) => {
              const ts = data.timestamp || new Date().toISOString();
              if (prev.length > 0 && prev[prev.length - 1].timestamp === ts) return prev;

              const pt = {
                timestamp: ts,
                observed: obsVal,
                predicted: predVal,
                lowerBound: Math.max(0, predVal - 60),
                upperBound: Math.min(1000, predVal + 60),
                anomalyScore
              };
              return [...prev, pt].slice(-60);
            });
          }
          if (data.forecast) {
            const mapped = data.forecast.map((f: any) => ({
              timestamp: new Date(Date.now() + (f.horizon_seconds * 1000)).toISOString(),
              predicted: f.risk * 100,
              lowerBound: Math.max(0, (f.risk * 100) - ((f.confidence || 0.5) * 20)),
              upperBound: Math.min(100, (f.risk * 100) + ((f.confidence || 0.5) * 20)),
              confidence: f.confidence || 0.5
            }));
            setForecast(mapped);
          }
        },
        onMLReset: () => {
          setPrediction(null);
        },
        onCommandStarted: (data) => {
          setLogLines((prev) => [...prev, `[${new Date().toISOString()}] sys INFO: Command started: ${data.command}`].slice(-100));
        },
        onCommandCompleted: (data) => {
          setLogLines((prev) => [...prev, `[${new Date().toISOString()}] sys INFO: Command completed: ${data.command}`].slice(-100));
          // Refresh status/topology after a command
          fetchStatus().then(setStatus).catch(() => {});
          if (data.command === 'reset_environment' || data.command === 'stop_network' || data.command === 'stop_attack') {
            setAttackEvents([]);
          }
          if (data.command === 'reset_environment' || data.command === 'stop_telemetry') {
            fetchTopology().then(setTopology).catch(() => {});
          }
        },
        onCommandOutput: (data) => {
          setLogLines((prev) => [...prev, data.line].slice(-100));
        },
        onAttackEvent: (data) => {
          if (data.stage === 'STOPPED' || data.stage?.includes('TERMINATED')) {
            setAttackEvents([]);
          } else {
            // Update or add attack event
            setAttackEvents((prev) => {
              const existing = prev.findIndex((e) => e.id === data.id);
              if (existing >= 0) {
                const next = [...prev];
                next[existing] = { ...next[existing], ...data };
                return next;
              }
              return [...prev, data];
            });
          }
        },
      });
      wsRef.current = ws;
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden", background: "var(--color-base)" }}>
      <Sidebar activePage={page} onNavigate={setPage} status={status} />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <Header
          status={status}
          site={site}
          wsConnected={wsConnected}
          pageTitle={PAGE_TITLES[page]}
          darkMode={darkMode}
          onToggleDark={toggleDark}
        />

        <main style={{ flex: 1, overflow: "hidden" }}>
          {page === "overview" && (
            <Overview
              status={status}
              telemetry={telemetry}
              forecast={forecast}
              prediction={prediction}
              attackEvents={attackEvents}
              events={events}
              topology={topology}
              logLines={logLines}
            />
          )}
          {page === "network" && <Network topology={topology} />}
          {page === "predictions" && (
            <Predictions telemetry={telemetry} forecast={forecast} prediction={prediction} />
          )}
          {page === "events" && <Events events={events} logLines={logLines} />}
          {page === "controls" && (
            <Controls
              status={status}
              logLines={logLines}
              onCommandSent={(_cmd, line) => setLogLines((prev) => [...prev, line].slice(-100))}
            />
          )}
        </main>
      </div>
    </div>
  );
}
