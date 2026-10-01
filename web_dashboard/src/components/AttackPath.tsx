import type { Campaign, CampaignNode } from "../types/campaign";
import type { FlowRecord } from "../types/evidence";
import type { Topology } from "../api/types";
import { Data, Meter, Micro, Square, sevColor, sevFromRisk } from "../design/primitives";

/**
 * Attack path — the campaign read as a chain of hosts rather than a graph.
 *
 * Each card is one host the intrusion has reached (or is forecast to reach),
 * in the order the causal edges run. The link between two cards carries the
 * evidence for the hop: the service port, the bytes that crossed it in the
 * latest window, and the causality score correlation/causal_edge_scorer.py
 * assigned. A host reached only in the forecast is drawn hatched, the same
 * encoding the kill chain and the campaign graph use.
 */

interface Hop {
  ip: string;
  label: string;
  zone: string;
  nodes: CampaignNode[];
  observed: boolean;
  risk: number;
  status: string;
}

interface Link {
  port: number | null;
  bytes: number;
  causality: number | null;
  forecast: boolean;
}

const ZONE_OF: Record<string, string> = {
  "10.0.3.": "DMZ",
  "10.0.2.": "Servers",
  "10.0.1.": "Users",
};

function zoneOf(ip: string): string {
  for (const [prefix, zone] of Object.entries(ZONE_OF)) if (ip.startsWith(prefix)) return zone;
  return "External";
}

function fmtBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export default function AttackPath({
  campaign,
  topology,
  flows,
  attacker,
  latestWindow,
  threshold = 0.65,
}: {
  campaign: Campaign | null;
  topology: Topology | null;
  flows: FlowRecord[];
  /** External source, from PredictionEvent.focus_ips. */
  attacker: string | null;
  latestWindow: number | null;
  threshold?: number;
}) {
  if (!campaign || !campaign.nodes.length) {
    return (
      <div style={{ padding: "var(--s-6)", textAlign: "center" }}>
        <Micro>No attack path</Micro>
      </div>
    );
  }

  const nameOf = (ip: string) => topology?.nodes.find((n) => n.ip === ip)?.label ?? ip;
  const statusOf = (ip: string) => topology?.nodes.find((n) => n.ip === ip)?.status ?? "online";

  // Hosts in first-reached order, following the campaign's node sequence.
  const order: string[] = [];
  for (const n of [...campaign.nodes].sort((a, b) => a.node_id - b.node_id)) {
    if (!order.includes(n.host_ip)) order.push(n.host_ip);
  }

  const hops: Hop[] = order.map((ip) => {
    const nodes = campaign.nodes.filter((n) => n.host_ip === ip);
    const observed = nodes.some((n) => n.provenance === "OBSERVED");
    const risk = Math.max(...nodes.map((n) => n.max_risk_score));
    return { ip, label: nameOf(ip), zone: zoneOf(ip), nodes, observed, risk, status: statusOf(ip) };
  });

  // The external origin leads the chain when the model has attributed one.
  const chain: Hop[] = attacker
    ? [
        {
          ip: attacker,
          label: "external source",
          zone: "External",
          nodes: [],
          observed: true,
          risk: hops[0]?.risk ?? 0,
          status: statusOf(attacker),
        },
        ...hops,
      ]
    : hops;

  const window = flows.filter((f) => f.window === latestWindow);

  const linkBetween = (a: Hop, b: Hop): Link => {
    const hop = window.filter(
      (f) => (f.src_ip === a.ip && f.dst_ip === b.ip) || (f.src_ip === b.ip && f.dst_ip === a.ip)
    );
    const bytes = hop.reduce((s, f) => s + f.fwd_bytes + f.bwd_bytes, 0);
    const port = hop.length ? hop[0].dst_port : null;

    // Causality: the strongest scored edge from any node on a to any on b.
    const aIds = new Set(a.nodes.map((n) => n.node_id));
    const bIds = new Set(b.nodes.map((n) => n.node_id));
    const scores = campaign.edges.filter((e) => aIds.has(e.src) && bIds.has(e.dst)).map((e) => e.causality_score);

    return { port, bytes, causality: scores.length ? Math.max(...scores) : null, forecast: !b.observed };
  };

  return (
    <div className="path">
      {chain.map((hop, i) => {
        const lvl = sevFromRisk(hop.risk, threshold);
        const link = i > 0 ? linkBetween(chain[i - 1], hop) : null;
        const techniques = [...new Set(hop.nodes.map((n) => n.technique_id))];

        return (
          <div key={hop.ip} className="path-step">
            {link && (
              <div className={link.forecast ? "path-link is-forecast" : link.bytes > 0 ? "path-link is-live" : "path-link"}>
                <span className="path-line" />
                <span className="path-meta">
                  <Data size="s" color="var(--paper-400)">
                    {link.port != null ? `:${link.port}` : link.forecast ? "forecast" : "—"}
                  </Data>
                  {link.bytes > 0 && (
                    <Data size="s" color="var(--paper-600)">
                      {fmtBytes(link.bytes)}
                    </Data>
                  )}
                  {link.causality != null && (
                    <Data size="s" color="var(--paper-600)">
                      c={link.causality.toFixed(2)}
                    </Data>
                  )}
                </span>
                <span className="path-arrow">▸</span>
              </div>
            )}

            <div
              className={hop.observed ? "path-card" : "path-card is-forecast"}
              style={{ borderLeft: `3px solid ${hop.observed ? sevColor(lvl) : "var(--signal-forecast)"}` }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)", marginBottom: 6 }}>
                <Square status={hop.observed ? hop.status : "neutral"} live={hop.observed && hop.status === "compromised"} />
                <span className="t-micro" style={{ color: "var(--paper-600)" }}>
                  {hop.zone}
                </span>
                <span className="t-micro" style={{ marginLeft: "auto", color: hop.observed ? "var(--paper-600)" : "var(--signal-forecast)" }}>
                  {hop.observed ? "observed" : "forecast"}
                </span>
              </div>

              <div className="t-label" style={{ color: "var(--paper-000)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {hop.label}
              </div>
              <Data size="s" color="var(--paper-400)">
                {hop.ip}
              </Data>

              <div style={{ margin: "var(--s-2) 0 var(--s-1)" }}>
                <Meter value={hop.risk} level={hop.observed ? lvl : undefined} color={hop.observed ? undefined : "var(--signal-forecast)"} height={4} />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "var(--s-2)" }}>
                <Data size="s" color={hop.observed ? sevColor(lvl) : "var(--signal-forecast)"}>
                  {hop.risk.toFixed(3)}
                </Data>
                <Data size="s" color="var(--paper-600)" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {techniques.length ? techniques.join(" · ") : "origin"}
                </Data>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
