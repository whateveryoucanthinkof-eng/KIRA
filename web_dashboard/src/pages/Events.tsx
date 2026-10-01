import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { AttackEvent, EventSeverity, NetworkEvent } from "../api/types";
import type { Page } from "../components/layout/Sidebar";
import { Chip, Empty, PanelHead, sevColor } from "../design/primitives";
import { clockTime } from "../design/time";

/**
 * Events: the log, made searchable.
 *
 * A volume histogram across the top that brushes a time range; facets down
 * the left with live counts; a query bar that takes free text and
 * `host:` `sev:` `tech:` `cat:` tokens; rows that open in place to the raw
 * record with its key=value pairs laid out. The console stays underneath.
 */

interface EventsProps {
  events: NetworkEvent[];
  attackEvents: AttackEvent[];
  logLines: string[];
  onNavigate: (page: Page) => void;
}

type Dim = "sev" | "cat" | "host" | "tech";

const SEVERITIES: EventSeverity[] = ["critical", "error", "warning", "info"];
const DIMS: { dim: Dim; label: string }[] = [
  { dim: "sev", label: "Severity" },
  { dim: "cat", label: "Category" },
  { dim: "host", label: "Host" },
  { dim: "tech", label: "Technique" },
];
const TECH = /\bT\d{4}(?:\.\d{3})?\b/;
const IP = /^\d{1,3}(?:\.\d{1,3}){3}$/;

function hhmmss(ts: string): string {
  try {
    return clockTime(ts);
  } catch {
    return ts;
  }
}

/** Log severity read off the line itself — the backend streams raw text. */
function logColor(line: string): string {
  if (/CRITICAL|ALERT|FATAL/.test(line)) return "var(--sev-critical)";
  if (/ERROR|DENY|BLOCK|FAIL/.test(line)) return "var(--sev-elevated)";
  if (/WARN/.test(line)) return "var(--sev-warning)";
  if (/INFO/.test(line)) return "var(--paper-400)";
  return "var(--paper-600)";
}

interface Row {
  e: NetworkEvent;
  t: number;
  tech: string | null;
  hosts: string[];
}

function valuesOf(r: Row, dim: Dim): string[] {
  switch (dim) {
    case "sev":
      return [r.e.severity];
    case "cat":
      return [r.e.category];
    case "host":
      return r.hosts;
    case "tech":
      return r.tech ? [r.tech] : [];
  }
}

/** `host:10.0.3.10 sev:critical tech:T1190 cat:COMMAND` plus free text. */
function parseQuery(q: string): { tokens: { dim: Dim; value: string }[]; text: string } {
  const tokens: { dim: Dim; value: string }[] = [];
  const rest: string[] = [];
  for (const part of q.trim().split(/\s+/).filter(Boolean)) {
    const m = /^(host|sev|tech|cat):(.+)$/i.exec(part);
    if (m) tokens.push({ dim: m[1].toLowerCase() as Dim, value: m[2] });
    else rest.push(part);
  }
  return { tokens, text: rest.join(" ").toLowerCase() };
}

function sameValue(dim: Dim, a: string, b: string): boolean {
  return dim === "host" ? a === b : a.toLowerCase() === b.toLowerCase();
}

/** Free text wrapped in <mark> where it matches. */
function highlight(s: string, text: string): ReactNode {
  if (!text) return s;
  const i = s.toLowerCase().indexOf(text);
  if (i < 0) return s;
  return (
    <>
      {s.slice(0, i)}
      <mark>{s.slice(i, i + text.length)}</mark>
      {s.slice(i + text.length)}
    </>
  );
}

/* ── Volume histogram with a brush ─────────────────────────────────────── */

function Histogram({
  rows,
  range,
  onRange,
}: {
  rows: Row[];
  range: [number, number] | null;
  onRange: (r: [number, number] | null) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x0: number; moved: boolean } | null>(null);
  const [live, setLive] = useState<[number, number] | null>(null);

  const now = Date.now();
  const t0 = rows.length ? Math.min(...rows.map((r) => r.t)) : now - 60_000;
  const t1 = Math.max(now, t0 + 60_000);
  const span = t1 - t0;
  // A fixed number of buckets, so bars stay thin however little is in view.
  const n = 60;
  const size = span / n;
  const buckets = Array.from({ length: n }, () => ({ critical: 0, error: 0, warning: 0, info: 0 }));
  for (const r of rows) {
    const i = Math.min(n - 1, Math.floor((r.t - t0) / size));
    buckets[i][r.e.severity] += 1;
  }
  const peak = Math.max(1, ...buckets.map((b) => b.critical + b.error + b.warning + b.info));

  const toT = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    return t0 + Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * span;
  };
  const shown = live ?? range;
  const pct = (t: number) => ((t - t0) / span) * 100;

  return (
    <div className="ev2-hist">
      <div
        ref={box}
        className="ev2-hist-plot"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { x0: toT(e.clientX), moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const t = toT(e.clientX);
          if (Math.abs(pct(t) - pct(d.x0)) > 0.6) d.moved = true;
          if (d.moved) setLive([Math.min(d.x0, t), Math.max(d.x0, t)]);
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          if (d.moved) onRange([Math.min(d.x0, toT(e.clientX)), Math.max(d.x0, toT(e.clientX))]);
          else onRange(null);
          setLive(null);
        }}
      >
        {buckets.map((b, i) => {
          let y = 0;
          return (
            <div key={i} className="ev2-bar" style={{ left: `${(i / n) * 100}%`, width: `${100 / n}%` }}>
              {SEVERITIES.map((s) => {
                const h = (b[s] / peak) * 100;
                if (!h) return null;
                const el = <i key={s} style={{ bottom: `${y}%`, height: `${h}%`, background: sevColor(s === "info" ? "neutral" : s) }} />;
                y += h;
                return el;
              })}
            </div>
          );
        })}
        {shown && (
          <>
            <div className="ev2-dim" style={{ left: 0, width: `${pct(shown[0])}%` }} />
            <div className="ev2-brush" style={{ left: `${pct(shown[0])}%`, width: `${pct(shown[1]) - pct(shown[0])}%` }} />
            <div className="ev2-dim" style={{ left: `${pct(shown[1])}%`, right: 0 }} />
          </>
        )}
      </div>
      <div className="ev2-hist-axis">
        <span>{hhmmss(new Date(t0).toISOString())}</span>
        <span>
          {shown
            ? `${hhmmss(new Date(shown[0]).toISOString())} – ${hhmmss(new Date(shown[1]).toISOString())} · click to clear`
            : `${size >= 10_000 ? Math.round(size / 1000) : (size / 1000).toFixed(1)}s buckets · drag to select a range`}
        </span>
        <span>now</span>
      </div>
    </div>
  );
}

/* ── Raw record, laid out ──────────────────────────────────────────────── */

function Detail({ r, onNavigate, onFilter }: { r: Row; onNavigate: (p: Page) => void; onFilter: (dim: Dim, v: string) => void }) {
  const raw = r.e.raw ?? "";
  const kv = [...raw.matchAll(/(\w+)=(\S+)/g)].map((m) => [m[1], m[2]] as const);
  return (
    <div className="ev2-detail">
      {kv.length > 0 && (
        <div className="ev2-kv">
          {kv.map(([k, v]) => (
            <span key={k}>
              <em>{k}</em>
              <b>{v}</b>
            </span>
          ))}
        </div>
      )}
      <code>{raw || r.e.message}</code>
      <div className="ev2-detail-actions">
        <span className="ev2-stamp">{r.e.timestamp}</span>
        {r.hosts.map((h) => (
          <button key={h} className="ev2-link" onClick={() => onFilter("host", h)}>
            Only {h}
          </button>
        ))}
        {r.hosts.length > 0 && (
          <button className="ev2-link" onClick={() => onNavigate("investigation")}>
            Investigate →
          </button>
        )}
        {r.tech && (
          <button className="ev2-link" onClick={() => onNavigate("attck")}>
            {r.tech} in ATT&CK →
          </button>
        )}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════ */

export default function Events({ events, attackEvents, logLines, onNavigate }: EventsProps) {
  const [picked, setPicked] = useState<Record<Dim, string[]>>({ sev: [], cat: [], host: [], tech: [] });
  const [query, setQuery] = useState("");
  const [range, setRange] = useState<[number, number] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [frozen, setFrozen] = useState<NetworkEvent[] | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(true);
  const tableRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines, consoleOpen]);

  // While paused, the table holds still; new events wait above it.
  const source = follow ? events : (frozen ?? events);
  // Counted by id: the buffer is capped, so lengths stop growing once it is full.
  const newest = frozen?.[0]?.id;
  const waiting = follow || !frozen ? 0 : newest ? Math.max(0, events.findIndex((e) => e.id === newest)) : events.length;

  const all = useMemo<Row[]>(
    () =>
      source.map((e) => {
        const m = TECH.exec(e.message);
        return {
          e,
          t: Date.parse(e.timestamp) || 0,
          tech: m ? m[0] : null,
          hosts: [e.source, e.destination].filter((h): h is string => !!h && IP.test(h)),
        };
      }),
    [source]
  );

  const { tokens, text } = parseQuery(query);

  /**
   * Every filter except, optionally, one dimension — so a facet counts what
   * choosing it would add, and the histogram shows volume outside the brush.
   */
  const pass = (r: Row, except?: Dim | "time") => {
    if (except !== "time" && range && (r.t < range[0] || r.t > range[1])) return false;
    for (const { dim } of DIMS) {
      if (dim === except) continue;
      const want = [...picked[dim], ...tokens.filter((t) => t.dim === dim).map((t) => t.value)];
      if (want.length && !valuesOf(r, dim).some((v) => want.some((w) => sameValue(dim, v, w)))) return false;
    }
    if (text) {
      const hay = [r.e.category, r.e.source, r.e.destination, r.e.message, r.e.raw].filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(text)) return false;
    }
    return true;
  };

  const rows = all.filter((r) => pass(r));
  const volume = all.filter((r) => pass(r, "time"));

  const facets = DIMS.map(({ dim, label }) => {
    const counts = new Map<string, number>();
    for (const r of all) {
      if (!pass(r, dim)) continue;
      for (const v of valuesOf(r, dim)) counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    for (const v of picked[dim]) if (!counts.has(v)) counts.set(v, 0);
    const list = [...counts.entries()].sort((a, b) =>
      dim === "sev" ? SEVERITIES.indexOf(a[0] as EventSeverity) - SEVERITIES.indexOf(b[0] as EventSeverity) : b[1] - a[1]
    );
    return { dim, label, list };
  });

  const toggle = (dim: Dim, v: string) =>
    setPicked((p) => ({ ...p, [dim]: p[dim].includes(v) ? p[dim].filter((x) => x !== v) : [...p[dim], v] }));
  const add = (dim: Dim, v: string) => setPicked((p) => (p[dim].includes(v) ? p : { ...p, [dim]: [...p[dim], v] }));

  // Enter turns typed tokens into chips; free text stays in the box.
  const commit = () => {
    if (!tokens.length) return;
    setPicked((p) => {
      const next = { ...p };
      for (const t of tokens) {
        const v = t.dim === "sev" || t.dim === "cat" ? (t.dim === "cat" ? t.value.toUpperCase() : t.value.toLowerCase()) : t.value;
        if (!next[t.dim].includes(v)) next[t.dim] = [...next[t.dim], v];
      }
      return next;
    });
    setQuery(text);
  };

  const chips = DIMS.flatMap(({ dim }) => picked[dim].map((v) => ({ dim, v })));
  const anyFilter = chips.length > 0 || !!query || !!range;

  const setFollowing = (on: boolean) => {
    setFollow(on);
    setFrozen(on ? null : events);
    if (on && tableRef.current) tableRef.current.scrollTop = 0;
  };

  const fresh = Date.now() - 1500;

  return (
    <div className="ev2">
      <div className={`ev2-sheet sheet${consoleOpen ? "" : " is-console-closed"}`}>
        {/* ── Volume ───────────────────────────────────────────────────── */}
        <Histogram rows={volume} range={range} onRange={setRange} />

        <div className="ev2-body">
          {/* ── Facets ─────────────────────────────────────────────────── */}
          <div className="ev2-facets">
            {facets.map(({ dim, label, list }) => (
              <div key={dim} className="ev2-facet">
                <div className="ev2-facet-head">{label}</div>
                {list.length ? (
                  list.slice(0, 8).map(([v, c]) => {
                    const on = picked[dim].includes(v);
                    return (
                      <button
                        key={v}
                        className={`ev2-fv${on ? " is-on" : ""}`}
                        aria-pressed={on}
                        onClick={() => toggle(dim, v)}
                        style={dim === "sev" ? { borderLeftColor: sevColor(v === "info" ? "neutral" : v) } : undefined}
                      >
                        <i>{on ? "■" : "□"}</i>
                        <span>{v}</span>
                        <em>{c}</em>
                      </button>
                    );
                  })
                ) : (
                  <span className="ev2-fv-none">none in view</span>
                )}
              </div>
            ))}
          </div>

          {/* ── Query + table ──────────────────────────────────────────── */}
          <div className="ev2-main">
            <div className="ev2-query">
              {chips.map(({ dim, v }) => (
                <button key={`${dim}:${v}`} className="ev2-chip" onClick={() => toggle(dim, v)} title="Remove">
                  <em>{dim}</em>
                  {v}
                  <i>×</i>
                </button>
              ))}
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commit();
                  if (e.key === "Escape") setQuery("");
                }}
                placeholder={chips.length ? "add text or host: sev: tech: cat:" : "search · host:10.0.3.10 sev:critical tech:T1190 cat:COMMAND"}
                spellCheck={false}
              />
              {anyFilter && (
                <button
                  className="ev2-clear"
                  onClick={() => {
                    setPicked({ sev: [], cat: [], host: [], tech: [] });
                    setQuery("");
                    setRange(null);
                  }}
                >
                  Clear
                </button>
              )}
              <span className="ev2-count">
                {rows.length} of {events.length}
              </span>
              {attackEvents.length > 0 && <Chip level="critical">{attackEvents.length} attack stages</Chip>}
              <button className="seg" aria-pressed={follow} onClick={() => setFollowing(!follow)} title="Keep the newest events in view">
                follow
              </button>
            </div>

            {waiting > 0 && (
              <button className="ev2-waiting" onClick={() => setFollowing(true)}>
                {waiting} new event{waiting === 1 ? "" : "s"} · show latest ↑
              </button>
            )}

            <div
              ref={tableRef}
              className="ev2-table"
              onScroll={(e) => {
                // Scrolling away from the newest rows pauses the feed so they stop moving.
                if (follow && e.currentTarget.scrollTop > 24) setFollowing(false);
              }}
            >
              {rows.length ? (
                rows.map((r) => {
                  const isOpen = open === r.e.id;
                  return (
                    <div key={r.e.id} className={`ev2-row${isOpen ? " is-open" : ""}`}>
                      <div
                        className={`ev2-line${follow && r.t > fresh ? " is-stamping" : ""}`}
                        style={{ borderLeftColor: sevColor(r.e.severity === "info" ? "neutral" : r.e.severity) }}
                        onClick={() => setOpen(isOpen ? null : r.e.id)}
                      >
                        <span className="ev2-time">{hhmmss(r.e.timestamp)}</span>
                        <span className="ev2-sev" style={{ color: sevColor(r.e.severity === "info" ? "neutral" : r.e.severity) }}>
                          {r.e.severity}
                        </span>
                        <span className="ev2-cat">{r.e.category}</span>
                        <span className="ev2-hosts">
                          {r.e.source}
                          {r.e.destination ? ` → ${r.e.destination}` : ""}
                        </span>
                        <span className="ev2-msg" title={r.e.message}>
                          {highlight(r.e.message, text)}
                        </span>
                        {r.tech && (
                          <button
                            className="ev2-tech"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              add("tech", r.tech!);
                            }}
                            title={`Only ${r.tech}`}
                          >
                            {r.tech}
                          </button>
                        )}
                      </div>
                      {isOpen && <Detail r={r} onNavigate={onNavigate} onFilter={add} />}
                    </div>
                  );
                })
              ) : (
                <Empty
                  hint={
                    events.length
                      ? "Nothing matches the filters. Clear them, or widen the time range."
                      : "Alerts, commands and attack stages are recorded here as the stream produces them."
                  }
                >
                  {events.length ? "No matches" : "No events yet"}
                </Empty>
              )}
            </div>
          </div>
        </div>

        {/* ── Console ──────────────────────────────────────────────────── */}
        <div className="ev2-console">
          <PanelHead
            title="Console"
            note={logLines.length ? `${logLines.length} lines` : undefined}
            aside={
              <button className="seg" aria-pressed={consoleOpen} onClick={() => setConsoleOpen((o) => !o)}>
                {consoleOpen ? "hide" : "show"}
              </button>
            }
          />
          {consoleOpen && (
            <div className="log" ref={logRef}>
              {logLines.length ? (
                logLines.slice(-120).map((line, i) => (
                  <div key={i} className="log-line" style={{ color: logColor(line) }}>
                    {line}
                  </div>
                ))
              ) : (
                <span className="log-line">— idle —</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
