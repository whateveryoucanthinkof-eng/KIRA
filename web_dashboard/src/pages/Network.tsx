import { useState } from "react";
import type { Topology, TopologyNode, TopologyEdge } from "../api/types";
import { StatusDot } from "../components/shared/StatusBadge";

interface NetworkProps {
  topology: Topology | null;
}

const nodeTypeIcon: Record<string, string> = {
  internet: "☁",
  firewall: "⛨",
  router: "⊕",
  switch: "⊞",
  server: "▣",
  host: "□",
  attacker: "⚠",
};

const nodeStatusColor: Record<string, string> = {
  online: "var(--color-status-green)",
  offline: "var(--color-text-muted)",
  degraded: "var(--color-status-amber)",
  compromised: "var(--color-status-red)",
  warning: "var(--color-status-amber)",
};

const edgeStatusColor: Record<string, string> = {
  active: "#9ca3af",
  saturated: "var(--color-status-red)",
  down: "#e5e7eb",
  suspicious: "var(--color-status-amber)",
};

function UtilBar({ pct }: { pct: number }) {
  const color =
    pct >= 90 ? "var(--color-status-red)" :
    pct >= 70 ? "var(--color-status-amber)" :
    "var(--color-status-green)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 4, background: "var(--color-base)", borderRadius: 2 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", minWidth: 26, textAlign: "right" }}>
        {pct}%
      </span>
    </div>
  );
}

export default function Network({ topology }: NetworkProps) {
  const [selectedNode, setSelectedNode] = useState<TopologyNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<TopologyEdge | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  if (!topology) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-muted)", fontSize: 13 }}>
        Loading topology…
      </div>
    );
  }

  const adjacentEdges = hoveredNode
    ? new Set(topology.edges.filter((e) => e.source === hoveredNode || e.target === hoveredNode).map((e) => e.id))
    : null;

  const selectedNodeEdges = selectedNode
    ? topology.edges.filter((e) => e.source === selectedNode.id || e.target === selectedNode.id)
    : [];

  function renderEdge(edge: TopologyEdge) {
    const src = topology!.nodes.find((n) => n.id === edge.source);
    const tgt = topology!.nodes.find((n) => n.id === edge.target);
    if (!src || !tgt) return null;
    const isHighlighted = adjacentEdges?.has(edge.id) || selectedNode?.id === src.id || selectedNode?.id === tgt.id;
    const isSelected = selectedEdge?.id === edge.id;
    const color = edgeStatusColor[edge.status] ?? "#9ca3af";
    const opacity = hoveredNode || selectedNode ? (isHighlighted ? 1 : 0.15) : 0.7;
    return (
      <g key={edge.id} onClick={() => { setSelectedEdge(edge); setSelectedNode(null); }} style={{ cursor: "pointer" }}>
        <line
          x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
          stroke="transparent"
          strokeWidth={12}
        />
        <line
          x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
          stroke={isSelected ? "var(--color-status-blue)" : color}
          strokeWidth={isSelected ? 2.5 : edge.status === "saturated" ? 2.5 : 1.5}
          strokeDasharray={edge.status === "suspicious" ? "5 3" : undefined}
          opacity={opacity}
          strokeLinejoin="round"
        />
        {/* utilization label at midpoint */}
        {edge.utilization !== undefined && isHighlighted && (
          <text
            x={(src.x + tgt.x) / 2}
            y={(src.y + tgt.y) / 2 - 6}
            textAnchor="middle"
            style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: color }}
          >
            {edge.utilization}%
          </text>
        )}
      </g>
    );
  }

  function renderNode(node: TopologyNode) {
    const color = nodeStatusColor[node.status] ?? "var(--color-text-muted)";
    const isSelected = selectedNode?.id === node.id;
    const isHovered = hoveredNode === node.id;
    const r = node.type === "internet" ? 14 : node.type === "firewall" || node.type === "router" ? 11 : node.type === "attacker" ? 10 : 9;
    const opacity = hoveredNode && hoveredNode !== node.id && !adjacentEdges?.has(node.id) ? 0.3 : 1;
    return (
      <g
        key={node.id}
        style={{ cursor: "pointer" }}
        opacity={opacity}
        onClick={() => { setSelectedNode(isSelected ? null : node); setSelectedEdge(null); }}
        onMouseEnter={() => setHoveredNode(node.id)}
        onMouseLeave={() => setHoveredNode(null)}
      >
        {/* selection ring */}
        {(isSelected || isHovered) && (
          <circle cx={node.x} cy={node.y} r={r + 5} fill="none" stroke={isSelected ? "var(--color-status-blue)" : color} strokeWidth={1.5} strokeDasharray={isSelected ? undefined : "3 2"} opacity={0.5} />
        )}
        <circle
          cx={node.x} cy={node.y} r={r}
          fill={isSelected ? "var(--color-status-blue-bg)" : color + "18"}
          stroke={isSelected ? "var(--color-status-blue)" : color}
          strokeWidth={isSelected ? 2 : 1.5}
        />
        <text
          x={node.x} y={node.y + 4}
          textAnchor="middle"
          style={{ fontSize: node.type === "internet" ? 12 : 10, fill: isSelected ? "var(--color-status-blue)" : color, fontFamily: "var(--font-mono)", userSelect: "none" }}
        >
          {nodeTypeIcon[node.type] ?? "○"}
        </text>
        <text
          x={node.x} y={node.y + r + 13}
          textAnchor="middle"
          style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-text-secondary)", fontWeight: isSelected ? "600" : "400", userSelect: "none" }}
        >
          {node.label}
        </text>
        {node.ip && (
          <text
            x={node.x} y={node.y + r + 22}
            textAnchor="middle"
            style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)", userSelect: "none" }}
          >
            {node.ip}
          </text>
        )}
      </g>
    );
  }

  const counts = {
    total: topology.nodes.length,
    online: topology.nodes.filter((n) => n.status === "online").length,
    compromised: topology.nodes.filter((n) => n.status === "compromised").length,
    degraded: topology.nodes.filter((n) => n.status === "degraded" || n.status === "warning").length,
  };

  return (
    <div className="network-page">
      {/* Main SVG canvas */}
      <div className="network-canvas-area">
        {/* Toolbar */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
            {counts.total} nodes · {topology.edges.length} links ·{" "}
            <span style={{ color: "var(--color-status-green)" }}>{counts.online} online</span>
            {counts.compromised > 0 && <span style={{ color: "var(--color-status-red)" }}> · {counts.compromised} compromised</span>}
            {counts.degraded > 0 && <span style={{ color: "var(--color-status-amber)" }}> · {counts.degraded} degraded</span>}
          </span>
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", gap: 12, fontSize: 10, color: "var(--color-text-muted)" }}>
            {[
              { color: edgeStatusColor.active, label: "active" },
              { color: edgeStatusColor.suspicious, label: "suspicious", dash: true },
              { color: edgeStatusColor.saturated, label: "saturated" },
            ].map(({ color, label, dash }) => (
              <span key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <svg width={18} height={6}><line x1={0} y1={3} x2={18} y2={3} stroke={color} strokeWidth={1.5} strokeDasharray={dash ? "4 2" : undefined} /></svg>
                {label}
              </span>
            ))}
          </div>
        </div>

        {/* SVG topology */}
        <div
          className="panel panel-clipped"
          style={{ flex: 1, overflow: "hidden", position: "relative" }}
          onClick={(e) => {
            if ((e.target as SVGElement).tagName === "svg" || (e.target as SVGRectElement).tagName === "rect") {
              setSelectedNode(null);
              setSelectedEdge(null);
            }
          }}
        >
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 800 540"
            style={{ display: "block" }}
          >
            <rect width="800" height="540" fill="transparent" />
            {/* Zone labels */}
            <text x={20} y={20} style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: 1 }}>EXTERNAL</text>
            <text x={20} y={130} style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: 1 }}>PERIMETER</text>
            <text x={20} y={240} style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: 1 }}>INTERNAL</text>

            {/* Zone dividers */}
            <line x1={10} y1={95} x2={790} y2={95} stroke="var(--color-border)" strokeWidth={1} strokeDasharray="4 4" />
            <line x1={10} y1={210} x2={790} y2={210} stroke="var(--color-border)" strokeWidth={1} strokeDasharray="4 4" />

            {/* DMZ label */}
            <text x={20} y={190} style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "#cbd5e1", textTransform: "uppercase", letterSpacing: 1 }}>DMZ</text>

            {/* Edges first */}
            {topology.edges.map(renderEdge)}
            {/* Nodes on top */}
            {topology.nodes.map(renderNode)}
          </svg>
        </div>
      </div>

      {/* Detail panel */}
      <div className="network-detail-panel">
        <div className="panel-header" style={{ borderBottom: "1px solid var(--color-border)" }}>
          <span className="panel-title">
            {selectedNode ? "Node Detail" : selectedEdge ? "Link Detail" : "Selection"}
          </span>
        </div>

        {!selectedNode && !selectedEdge && (
          <div style={{ padding: 16, color: "var(--color-text-muted)", fontSize: 12 }}>
            Click a node or link to inspect it.
          </div>
        )}

        {selectedNode && (
          <div style={{ padding: 14, overflow: "auto", flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
              <StatusDot status={selectedNode.status} pulse={selectedNode.status === "online"} />
              <span style={{ fontSize: 14, fontWeight: 600, color: "var(--color-text-primary)" }}>
                {selectedNode.label}
              </span>
            </div>

            {[
              { label: "Type", value: selectedNode.type },
              { label: "Status", value: selectedNode.status },
              { label: "IP Address", value: selectedNode.ip ?? "—", mono: true },
              { label: "OS", value: selectedNode.os ?? "—" },
            ].map(({ label, value, mono }) => (
              <div key={label} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>{label}</div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", fontFamily: mono ? "var(--font-mono)" : undefined }}>{value}</div>
              </div>
            ))}

            {selectedNode.services && selectedNode.services.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>Services</div>
                {selectedNode.services.map((svc) => (
                  <div key={svc} style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)", padding: "2px 0" }}>
                    {svc}
                  </div>
                ))}
              </div>
            )}

            {selectedNodeEdges.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
                  Connections ({selectedNodeEdges.length})
                </div>
                {selectedNodeEdges.map((edge) => {
                  const other = topology.nodes.find(
                    (n) => n.id === (edge.source === selectedNode.id ? edge.target : edge.source)
                  );
                  return (
                    <div
                      key={edge.id}
                      style={{
                        padding: "6px 8px",
                        borderRadius: 4,
                        background: "var(--color-base)",
                        marginBottom: 4,
                        cursor: "pointer",
                      }}
                      onClick={() => { setSelectedEdge(edge); }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                          {other?.label ?? "unknown"}
                        </span>
                        <span
                          style={{
                            fontSize: 9,
                            color: edgeStatusColor[edge.status] ?? "var(--color-text-muted)",
                            fontWeight: 600,
                          }}
                        >
                          {edge.status}
                        </span>
                      </div>
                      {edge.utilization !== undefined && (
                        <div style={{ marginTop: 4 }}>
                          <UtilBar pct={edge.utilization} />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {selectedEdge && !selectedNode && (
          <div style={{ padding: 14, overflow: "auto", flex: 1 }}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginBottom: 4 }}>link</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                {topology.nodes.find((n) => n.id === selectedEdge.source)?.label} →{" "}
                {topology.nodes.find((n) => n.id === selectedEdge.target)?.label}
              </div>
            </div>
            {[
              { label: "Status", value: selectedEdge.status },
              { label: "Protocol", value: selectedEdge.protocol ?? "Ethernet" },
              { label: "Bandwidth", value: selectedEdge.bandwidth ? `${selectedEdge.bandwidth >= 1000 ? `${selectedEdge.bandwidth / 1000} Gbps` : `${selectedEdge.bandwidth} Mbps`}` : "—" },
            ].map(({ label, value }) => (
              <div key={label} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>{label}</div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{value}</div>
              </div>
            ))}
            {selectedEdge.utilization !== undefined && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Utilization</div>
                <UtilBar pct={selectedEdge.utilization} />
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        <div style={{ padding: "10px 14px", borderTop: "1px solid var(--color-border)" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 8 }}>Node Status</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {[
              { status: "online" as const, label: "Online" },
              { status: "warning" as const, label: "Warning" },
              { status: "degraded" as const, label: "Degraded" },
              { status: "compromised" as const, label: "Compromised" },
            ].map(({ status, label }) => (
              <div key={status} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <StatusDot status={status} />
                <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
