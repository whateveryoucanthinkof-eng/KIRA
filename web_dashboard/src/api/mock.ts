import {
  SystemStatus,
  Topology,
  SiteInfo,
  MitigationPayload,
  WSHandlers,
} from "./types";

export const mockFetchStatus = async (): Promise<SystemStatus> => {
  return {
    networkStatus: "running",
    telemetryStatus: "running",
    predictionStatus: "running",
    attackStatus: "none",
    uptime: 3600,
    lastUpdate: new Date().toISOString(),
    anomalyScore: 12,
    threatLevel: "low",
    packetLoss: 0.1,
    latency: 12,
    throughput: 450,
    activeConnections: 1240,
  };
};

export const mockFetchSite = async (): Promise<SiteInfo> => {
  return {
    name: "Demo Site (Frontend Mock)",
    location: "Local Browser",
    timezone: "UTC",
    subnet: "10.0.0.0/16",
    externalIp: "192.168.1.100",
    description: "This is a frontend-only mock environment.",
  };
};

export const mockFetchTopology = async (): Promise<Topology> => {
  return {
    nodes: [
      { id: "ext1", label: "Internet", type: "internet", status: "online", x: 400, y: 50 },
      { id: "fw1", label: "Firewall", type: "firewall", status: "online", x: 400, y: 150 },
      { id: "web1", label: "Web Server 1", type: "server", status: "online", x: 200, y: 300 },
      { id: "web2", label: "Web Server 2", type: "server", status: "online", x: 600, y: 300 },
      { id: "db1", label: "Database", type: "server", status: "online", x: 400, y: 450 },
    ],
    edges: [
      { id: "ext1-fw1-tcp-0", source: "ext1", target: "fw1", status: "active", bandwidth: 120, utilization: 12, protocol: "TCP" },
      { id: "fw1-web1-tcp-80", source: "fw1", target: "web1", status: "active", bandwidth: 60, utilization: 6, protocol: "TCP" },
      { id: "fw1-web2-tcp-80", source: "fw1", target: "web2", status: "active", bandwidth: 50, utilization: 5, protocol: "TCP" },
      { id: "web1-db1-tcp-5432", source: "web1", target: "db1", status: "active", bandwidth: 10, utilization: 1, protocol: "TCP" },
      { id: "web2-db1-tcp-5432", source: "web2", target: "db1", status: "active", bandwidth: 8, utilization: 1, protocol: "TCP" },
    ],
    lastUpdated: new Date().toISOString(),
  };
};

export const mockSendCommand = async (command: string): Promise<void> => {
  console.log(`[Mock] Command sent: ${command}`);
  return Promise.resolve();
};

export const mockSendMitigate = async (payload: MitigationPayload): Promise<void> => {
  console.log(`[Mock] Mitigation sent:`, payload);
  return Promise.resolve();
};

export class MockWebSocket {
  private handlers: WSHandlers;
  private interval: any;

  constructor(handlers: WSHandlers) {
    this.handlers = handlers;
    
    // Simulate connection delay
    setTimeout(() => {
      this.handlers.onOpen?.();
      this.startEmitting();
    }, 500);
  }

  startEmitting() {
    this.interval = setInterval(() => {
      const now = new Date().toISOString();
      const riskValue = Math.random() * 0.4;
      
      this.handlers.onPrediction?.({
        timestamp: now,
        value: riskValue,
        confidence: 0.8,
        horizon: 5,
        model: "Mock Model",
        signals: [
          { name: "Hazard Score", weight: 0.6, direction: "positive", value: (riskValue * 100).toFixed(1) }
        ]
      });
      
      this.handlers.onSystemStatus?.({
        networkStatus: "running",
        telemetryStatus: "running",
        predictionStatus: "running",
        attackStatus: "none",
        uptime: 3600,
        lastUpdate: now,
        anomalyScore: riskValue * 100,
        threatLevel: riskValue > 0.3 ? "medium" : "low",
        packetLoss: 0,
        latency: 10 + Math.random() * 5,
        throughput: 400 + Math.random() * 100,
        activeConnections: 1200 + Math.floor(Math.random() * 100),
      });

    }, 2000);
  }

  close() {
    clearInterval(this.interval);
    this.handlers.onClose?.();
  }
}
