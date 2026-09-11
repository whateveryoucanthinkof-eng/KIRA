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
  const [darkMode, setDarkMode] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  const toggleDark = () => {
    setDarkMode((d) => {
      document.documentElement.classList.toggle("dark", !d);
      return !d;
    });
  };

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
          setTelemetry((prev) => {
            const ts = data.lastUpdate || new Date().toISOString();
            if (prev.length > 0 && prev[prev.length - 1].timestamp === ts) return prev;
            
            const pt = {
              timestamp: ts,
              observed: data.throughput || 0,
              predicted: data.throughput || 0,
              lowerBound: Math.max(0, (data.throughput || 0) - 100),
              upperBound: (data.throughput || 0) + 100,
              anomalyScore: data.anomalyScore || 0
            };
            return [...prev, pt].slice(-60);
          });
        },
        onTopologyUpdate: (data) => {
          setTopology(data);
        },
        onPrediction: (data: any) => {
          if (data.prediction) {
            setPrediction(data.prediction);
            setTelemetry((prev) => {
              const ts = data.timestamp || new Date().toISOString();
              // Prevent exact duplicates if multiple events fire
              if (prev.length > 0 && prev[prev.length - 1].timestamp === ts) return prev;
              
              const pt = {
                timestamp: ts,
                observed: data.state?.active_flows || data.state?.packet_count || 0,
                predicted: data.prediction.value * 100, // value is mapped to risk in adapter
                lowerBound: Math.max(0, (data.prediction.value * 100) - 20),
                upperBound: Math.min(100, (data.prediction.value * 100) + 20),
                anomalyScore: data.prediction.value * 100
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
