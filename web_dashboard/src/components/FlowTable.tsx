import { useMemo, useState } from "react";
import type { FlowRecord, FlowFlags } from "../types/evidence";
import { Data, Empty, Micro, sevColor } from "../design/primitives";
import { clockTime } from "../design/time";

/**
 * Flow evidence — the raw 5-tuples behind a verdict.
 *
 * This is what an analyst drops into when a risk score is not enough: which
 * addresses, which ports, how many bytes, which TCP flags. Flag counts are the
 * ones telemetry/flow/flow_table.py aggregates from the per-packet flags
 * parsed in telemetry/capture/sniffer.py.
 *
 * Pause freezes the table on a snapshot so a row can be read while the
 * stream keeps running underneath.
 */

const FLAG_KEYS: { key: keyof FlowFlags; ch: string; name: string }[] = [
  { key: "syn", ch: "S", name: "SYN" },
  { key: "ack", ch: "A", name: "ACK" },
  { key: "psh", ch: "P", name: "PSH" },
  { key: "rst", ch: "R", name: "RST" },
  { key: "fin", ch: "F", name: "FIN" },
  { key: "urg", ch: "U", name: "URG" },
];

/** HH:MM:SS.µµµµµµ from integer epoch microseconds. */
function stamp(tsUs: number): string {
  const d = new Date(Math.floor(tsUs / 1000));
  const hms = clockTime(d);
  return `${hms}.${String(tsUs % 1_000_000).padStart(6, "0")}`;
}

function bytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

/** Six-cell flag strip. A lit cell means the flag appeared at least once. */
function Flags({ flags, tcp }: { flags: FlowFlags; tcp: boolean }) {
  if (!tcp) {
    return (
      <span className="t-data-s" style={{ color: "var(--paper-600)" }}>
        —
      </span>
    );
  }
  const title = FLAG_KEYS.map((f) => `${f.name} ${flags[f.key]}`).join("  ·  ");
  return (
    <span className="flag-strip" title={title}>
      {FLAG_KEYS.map((f) => {
        const n = flags[f.key];
        return (
          <span key={f.key} className={n > 0 ? "flag is-on" : "flag"}>
            {f.ch}
          </span>
        );
      })}
    </span>
  );
}

interface FlowTableProps {
  flows: FlowRecord[];
  /** Active flows in the latest window, before the sensor's sample cap. */
  flowsInWindow?: number;
  latestWindow?: number | null;
  /** Hosts to label in place of raw addresses. */
  names?: Record<string, string>;
  maxRows?: number;
}

export default function FlowTable({ flows, flowsInWindow, latestWindow, names = {}, maxRows = 160 }: FlowTableProps) {
  const [pathOnly, setPathOnly] = useState(false);
  const [proto, setProto] = useState<"ALL" | "TCP" | "UDP">("ALL");
  const [query, setQuery] = useState("");
  const [frozen, setFrozen] = useState<FlowRecord[] | null>(null);

  // Newest first. When paused, read from the snapshot instead of the stream.
  const source = frozen ?? flows;

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return [...source]
      .reverse()
      .filter((f) => (pathOnly ? f.on_path : true))
      .filter((f) => (proto === "ALL" ? true : f.protocol === proto))
      .filter((f) => {
        if (!q) return true;
        const hay = `${f.src_ip}:${f.src_port} ${f.dst_ip}:${f.dst_port} ${names[f.src_ip] ?? ""} ${names[f.dst_ip] ?? ""} ${f.protocol}`;
        return hay.toLowerCase().includes(q);
      })
      .slice(0, maxRows);
  }, [source, pathOnly, proto, query, names, maxRows]);

  const latest = source.filter((f) => f.window === latestWindow);
  const pathCount = latest.filter((f) => f.on_path).length;

  const endpoint = (ip: string, port: number) => (
    <>
      <span style={{ color: "var(--paper-000)" }}>{names[ip] ?? ip}</span>
      <span style={{ color: "var(--paper-600)" }}>:{port}</span>
    </>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
      {/* ── Toolbar ─────────────────────────────────────────────────── */}
      <div className="flow-bar">
        <button
          className="seg"
          aria-pressed={!frozen}
          onClick={() => setFrozen(frozen ? null : [...flows])}
          title={frozen ? "Resume the live stream" : "Freeze the table to read a row"}
        >
          {frozen ? "Paused" : "Live"}
        </button>

        <span className="seg-group">
          {(["ALL", "TCP", "UDP"] as const).map((p) => (
            <button key={p} className="seg" aria-pressed={proto === p} onClick={() => setProto(p)}>
              {p}
            </button>
          ))}
        </span>

        <button className="seg" aria-pressed={pathOnly} onClick={() => setPathOnly((v) => !v)}>
          Attack path only
        </button>

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="ip, port, host…"
          className="t-data-s flow-search"
        />

        <span style={{ marginLeft: "auto", display: "flex", gap: "var(--s-3)", alignItems: "baseline" }}>
          {latestWindow != null && (
            <Data size="s" color="var(--paper-600)">
              window #{latestWindow}
            </Data>
          )}
          <Data size="s" color={pathCount ? sevColor("elevated") : "var(--paper-600)"}>
            {pathCount} on path
          </Data>
          {flowsInWindow != null && (
            <Data size="s" color="var(--paper-600)">
              sample {latest.length} of {flowsInWindow.toLocaleString()}
            </Data>
          )}
        </span>
      </div>

      {/* ── Table ───────────────────────────────────────────────────── */}
      <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        {rows.length ? (
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 142 }}>First packet</th>
                <th style={{ width: 50 }}>Proto</th>
                <th style={{ width: 178 }}>Source</th>
                <th style={{ width: 20 }} />
                <th style={{ width: 178 }}>Destination</th>
                <th className="num" style={{ width: 82 }}>Fwd</th>
                <th className="num" style={{ width: 82 }}>Bwd</th>
                <th className="num" style={{ width: 76 }}>Pkts f/b</th>
                <th style={{ width: 104 }}>Flags</th>
                <th className="num">Duration</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((f) => (
                <tr key={f.id} className={f.window === latestWindow && !frozen ? "is-stamping" : undefined}>
                  <td style={{ borderLeft: `3px solid ${f.on_path ? sevColor("elevated") : "transparent"}` }}>{stamp(f.ts_us)}</td>
                  <td>{f.protocol}</td>
                  <td>{endpoint(f.src_ip, f.src_port)}</td>
                  <td style={{ color: "var(--paper-600)", textAlign: "center" }}>
                    {f.direction === "outbound" ? "↗" : f.direction === "inbound" ? "↘" : "→"}
                  </td>
                  <td>{endpoint(f.dst_ip, f.dst_port)}</td>
                  <td className="num key">{bytes(f.fwd_bytes)}</td>
                  <td className="num">{bytes(f.bwd_bytes)}</td>
                  <td className="num">
                    {f.fwd_packets.toLocaleString()}/{f.bwd_packets.toLocaleString()}
                  </td>
                  <td>
                    <Flags flags={f.flags} tcp={f.protocol === "TCP"} />
                  </td>
                  <td className="num">{f.duration_ms.toFixed(3)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Empty
            hint={
              flows.length
                ? "No flow matches the current filter."
                : "Flow records are written by the sensor per window but not yet forwarded over the event bus."
            }
          >
            {flows.length ? "No matches" : "No flow evidence"}
          </Empty>
        )}
      </div>

      <div className="flow-foot">
        <Micro>Flags</Micro>
        {FLAG_KEYS.map((f) => (
          <span key={f.key} className="t-data-s" style={{ color: "var(--paper-600)" }}>
            <span style={{ color: "var(--paper-000)" }}>{f.ch}</span> {f.name}
          </span>
        ))}
        <span className="t-data-s" style={{ color: "var(--paper-600)", marginLeft: "auto" }}>
          ↘ inbound · ↗ outbound · → internal
        </span>
      </div>
    </div>
  );
}
