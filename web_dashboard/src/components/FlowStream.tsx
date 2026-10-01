/**
 * Flow record stream for the replay DVR.
 *
 * The records behind each window appear as the playhead crosses their first
 * packet, under a divider carrying the window's score. These are flow
 * records, which is what the sensor keeps: endpoints, protocol, the TCP flags
 * seen, packets, bytes and duration. There are no payloads to show, so none
 * are shown. Records on the attributed attack path are marked; the rest of a
 * window is summarised to its largest few so the attack stays readable.
 */

import { useLayoutEffect, useMemo, useRef, type ReactElement } from "react";
import type { ReplayRow } from "../types/replay";
import type { FlowRecord } from "../types/evidence";
import { sevColor, sevFromRisk } from "../design/primitives";
import { fmtClock, fmtRate } from "./DvrTimeline";

/** Windows of history kept above the current one. */
const BACK = 6;
const MAX_PATH = 10;
const MAX_BACKGROUND = 3;

const FLAG_LETTERS: [keyof FlowRecord["flags"], string][] = [
  ["syn", "S"],
  ["ack", "A"],
  ["psh", "P"],
  ["rst", "R"],
  ["fin", "F"],
  ["urg", "U"],
];

// Pads, never truncates: a clipped address or port would misreport the record.
const pad = (s: string, n: number) => (s.length >= n ? s : s + " ".repeat(n - s.length));
const lpad = (s: string, n: number) => (s.length >= n ? s : " ".repeat(n - s.length) + s);

/** One row of the stream, header or record, in fixed columns. */
function cols(c: [string, string, string, string, string, string, string, string, string], arrow: string): string {
  const [t, no, src, dst, proto, flags, pkts, bytes, dur] = c;
  return `${lpad(t, 8)}  ${lpad(no, 6)}  ${pad(src, 21)} ${arrow} ${pad(dst, 21)}  ${pad(proto, 4)} ${pad(flags, 5)} ${lpad(pkts, 9)}  ${lpad(bytes, 13)}  ${lpad(dur, 7)}`;
}

function size(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function flagsOf(f: FlowRecord): string {
  if (f.protocol !== "TCP") return "·";
  return FLAG_LETTERS.filter(([k]) => f.flags[k] > 0)
    .map(([, l]) => l)
    .join("");
}

function line(f: FlowRecord, rel: number, no: number): string {
  const d = f.duration_ms;
  return cols(
    [
      rel.toFixed(3),
      `#${no}`,
      `${f.src_ip}:${f.src_port}`,
      `${f.dst_ip}:${f.dst_port}`,
      f.protocol,
      flagsOf(f),
      `${f.fwd_packets}/${f.bwd_packets}`,
      `${size(f.fwd_bytes)}/${size(f.bwd_bytes)}`,
      `${d < 10 ? d.toFixed(1) : Math.round(d)}ms`,
    ],
    "→"
  );
}

const HEAD = cols(["t (s)", "flow", "source", "destination", "prot", "flags", "pkts ↑/↓", "bytes ↑/↓", "dur"], " ");

interface Shown {
  rel: number;
  text: string;
  path: boolean;
  id: string;
}

interface Win {
  total: number;
  onPath: number;
  shown: Shown[];
}

export default function FlowStream({
  rows,
  pos,
  threshold,
  windowSeconds,
}: {
  rows: ReplayRow[];
  pos: number;
  threshold: number;
  windowSeconds: number;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const hasFlows = rows.some((r) => r.flows && r.flows.length);

  // Every window's lines, numbered across the capture, built once.
  const wins = useMemo<Win[]>(() => {
    const origin = rows[0]?.flows?.[0]?.ts_us ?? 0;
    let seen = 0;
    return rows.map((r) => {
      const flows = r.flows ?? [];
      const numbered = flows.map((f, k) => ({ f, no: seen + k + 1, rel: (f.ts_us - origin) / 1e6 }));
      seen += flows.length;
      const path = numbered.filter((x) => x.f.on_path).slice(0, MAX_PATH);
      const background = numbered
        .filter((x) => !x.f.on_path)
        .sort((a, b) => b.f.fwd_bytes + b.f.bwd_bytes - (a.f.fwd_bytes + a.f.bwd_bytes))
        .slice(0, MAX_BACKGROUND);
      const shown = [...path, ...background]
        .sort((a, b) => a.rel - b.rel)
        .map((x) => ({ rel: x.rel, text: line(x.f, x.rel, x.no), path: x.f.on_path, id: x.f.id }));
      return { total: flows.length, onPath: numbered.filter((x) => x.f.on_path).length, shown };
    });
  }, [rows]);

  const n = rows.length;
  const cur = Math.max(0, Math.min(n - 1, Math.floor(pos)));
  const tNow = (rows[cur]?.window ?? 0) * windowSeconds + (pos - cur) * windowSeconds;
  const live = wins[cur]?.shown ?? [];
  let revealed = 0;
  while (revealed < live.length && live[revealed].rel <= tNow) revealed++;

  const body = useMemo(() => {
    const out: ReactElement[] = [];
    for (let i = Math.max(0, cur - BACK); i <= cur; i++) {
      const r = rows[i];
      const win = wins[i];
      const level = sevFromRisk(r.risk, threshold);
      const rate = r.packets != null ? r.packets / windowSeconds : null;
      out.push(
        <div key={`w${i}`} className={`fs-win${r.alert ? " is-alert" : ""}`} style={{ borderLeftColor: sevColor(level) }}>
          <b>W {String(r.window).padStart(3, "0")}</b>
          <span>{fmtClock(r.window * windowSeconds)}</span>
          <span>
            {win.total} records{win.total > win.shown.length ? ` · ${win.shown.length} shown` : ""}
          </span>
          {rate != null && <span>{fmtRate(rate)} pkt/s</span>}
          <span className="fs-win-risk" style={{ color: sevColor(level) }}>
            risk {r.risk.toFixed(3)}
            {r.alert ? " · ALERT" : ""}
          </span>
          {r.mitre_technique && <em>{r.mitre_technique}</em>}
        </div>
      );
      const lines = i === cur ? win.shown.slice(0, revealed) : win.shown;
      for (const l of lines) {
        out.push(
          <div key={l.id} className={`fs-line${l.path ? " is-path" : ""}`}>
            {l.text}
          </div>
        );
      }
    }
    return out;
  }, [rows, wins, cur, revealed, threshold, windowSeconds]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [body]);

  if (!hasFlows) {
    return (
      <div className="fs-empty">
        This backend's <code>/api/replay</code> returns window scores only. The flow records behind each window
        appear here once the endpoint forwards them.
      </div>
    );
  }

  return (
    <div className="fs">
      <div className="fs-head">{HEAD}</div>
      <div
        ref={scroller}
        className="fs-body"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {body}
      </div>
    </div>
  );
}
