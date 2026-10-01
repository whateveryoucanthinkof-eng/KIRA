/**
 * TGNE attention matrix.
 *
 * Where the graph encoder is looking. Each row is a host building its
 * embedding; each cell is how much of that host's attention lands on one of
 * its temporal neighbours, so a row sums to 1. Ordinary traffic spreads thin
 * and warm; attention that concentrates on a hostile edge turns critical.
 *
 * Hover previews a cell, click locks it; the saliency panel then walks the
 * attention logit from the baseline through each input's Input × Gradient
 * contribution — the waterfall is the arithmetic, not an illustration.
 */

import { useEffect, useRef, useState } from "react";
import type { AttentionMatrix as Matrix, SaliencyTerm } from "../types/attention";
import { Chip, Empty, Micro } from "../design/primitives";
import { OBSERVED } from "../design/charts";
import { blip, useSound } from "../design/sound";
import UnderDev from "./UnderDev";

type Cell = [number, number];

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

const TOP_TERMS = 6;

function fmt(v: number, digits = 2): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
}

/** Baseline → each contribution → logit, as spans on one axis. */
function Waterfall({ terms, baseline, logit, hostile }: { terms: SaliencyTerm[]; baseline: number; logit: number; hostile: boolean }) {
  const top = terms.slice(0, TOP_TERMS);
  const rest = terms.slice(TOP_TERMS).reduce((a, t) => a + t.contribution, 0);
  const rows: { label: string; value: number; from: number; to: number; kind: "base" | "term" | "rest" | "total" }[] = [];
  let cum = baseline;
  rows.push({ label: "baseline", value: baseline, from: 0, to: baseline, kind: "base" });
  for (const t of top) {
    rows.push({ label: t.feature, value: t.contribution, from: cum, to: cum + t.contribution, kind: "term" });
    cum += t.contribution;
  }
  if (terms.length > TOP_TERMS) {
    rows.push({ label: `${terms.length - TOP_TERMS} other inputs`, value: rest, from: cum, to: cum + rest, kind: "rest" });
    cum += rest;
  }
  rows.push({ label: "attention logit", value: logit, from: 0, to: logit, kind: "total" });

  const lo = Math.min(0, ...rows.map((r) => Math.min(r.from, r.to)));
  const hi = Math.max(0.5, ...rows.map((r) => Math.max(r.from, r.to)));
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;

  return (
    <div className="wf">
      {rows.map((r, i) => {
        const left = x(Math.min(r.from, r.to));
        const width = Math.max(0.6, Math.abs(x(r.to) - x(r.from)));
        const colour =
          r.kind === "base" || r.kind === "total"
            ? "var(--paper-400)"
            : r.value < 0
              ? "var(--sev-nominal)"
              : hostile
                ? "var(--sev-critical)"
                : OBSERVED;
        return (
          <div key={i} className={`wf-row is-${r.kind}`}>
            <span className="wf-label" title={r.label}>
              {r.label}
            </span>
            <span className="wf-value" style={r.kind === "term" || r.kind === "rest" ? { color: colour } : undefined}>
              {r.kind === "term" || r.kind === "rest" ? fmt(r.value) : r.value.toFixed(2)}
            </span>
            <span className="wf-track">
              <span className="wf-zero" style={{ left: `${x(0)}%` }} />
              <i style={{ left: `${left}%`, width: `${width}%`, background: colour }} />
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function AttentionMatrix({ matrix, focus }: { matrix: Matrix | null; focus: string[] }) {
  const [gridRef, { w, h }] = useSize<HTMLDivElement>();
  const [rootRef, { w: rootW }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<Cell | null>(null);
  const [locked, setLocked] = useState<Cell | null>(null);
  const [sound, setSound] = useSound();

  const n = matrix?.nodes.length ?? 0;

  // Esc releases a locked cell; arrows move it.
  useEffect(() => {
    if (!locked) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const [i, j] = locked;
      const move: Record<string, Cell> = {
        ArrowUp: [Math.max(0, i - 1), j],
        ArrowDown: [Math.min(n - 1, i + 1), j],
        ArrowLeft: [i, Math.max(0, j - 1)],
        ArrowRight: [i, Math.min(n - 1, j + 1)],
      };
      if (e.key === "Escape") setLocked(null);
      else if (move[e.key]) {
        e.preventDefault();
        setLocked(move[e.key]);
        blip(1100, 30);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [locked, n]);

  if (!matrix) {
    return (
      <div className="am" ref={rootRef}>
        <div ref={gridRef} className="am-empty">
          <UnderDev block title="docs/DASHBOARD_INTEGRATION.md — TGNE attention extraction">
            Under development — the TGNE attention layer's weights are computed on every window but not yet read out
            to the console
          </UnderDev>
        </div>
      </div>
    );
  }

  const { nodes, alpha, alpha_heads, logits, saliency, baseline_logit } = matrix;
  const hostileIps = new Set(focus);
  const hostileEdge = (i: number, j: number) => hostileIps.has(nodes[i].ip) || hostileIps.has(nodes[j].ip);
  const absent = (i: number) => alpha[i].every((v) => v === 0);

  // With nothing chosen, show the strongest attention on a hostile edge — or the strongest anywhere.
  let best: Cell = [0, 1];
  let bestScore = -1;
  alpha.forEach((row, i) =>
    row.forEach((v, j) => {
      if (i === j) return;
      const score = v + (hostileEdge(i, j) ? 1 : 0);
      if (score > bestScore) {
        bestScore = score;
        best = [i, j];
      }
    })
  );
  const active: Cell = locked ?? hover ?? best;
  const [ai, aj] = active;
  const a = alpha[ai]?.[aj] ?? 0;
  const terms = saliency[ai]?.[aj] ?? [];
  const hostile = hostileEdge(ai, aj) && a >= 0.3;
  const rank = [...(alpha[ai] ?? [])]
    .map((v, j) => ({ v, j }))
    .filter((c) => c.j !== ai && c.v > 0)
    .sort((p, q) => q.v - p.v)
    .findIndex((c) => c.j === aj);
  const neighbours = (alpha[ai] ?? []).filter((v, j) => j !== ai && v > 0).length;

  // Square cells sized to the space: row labels on the left, column labels on top.
  const rowHead = 104;
  const colHead = 58;
  const cell = Math.max(26, Math.min(58, Math.floor(Math.min((w - rowHead) / n, (h - colHead) / n)) - 2));

  const fill = (i: number, j: number, v: number) => {
    const pct = Math.round(Math.min(1, v / 0.95) * 88);
    const base = hostileEdge(i, j) && v >= 0.3 ? "var(--sev-critical)" : "var(--signal-observed)";
    return `color-mix(in srgb, ${base} ${pct}%, var(--ink-100))`;
  };

  const choose = (c: Cell, lock: boolean) => {
    if (lock) {
      const same = locked && locked[0] === c[0] && locked[1] === c[1];
      setLocked(same ? null : c);
      blip(same ? 700 : 1200, 50);
    } else {
      setHover(c);
      blip(320 + (alpha[c[0]]?.[c[1]] ?? 0) * 900, 28, 0.025);
    }
  };

  // Where there is no room for a saliency column, the breakdown is a popover
  // beside the cell being inspected, on the side away from it.
  const compact = rootW > 0 && rootW < 820;
  const showInspect = !compact || hover != null || locked != null;

  return (
    <div className={compact ? "am is-compact" : "am"} ref={rootRef}>
      {/* ── Matrix ──────────────────────────────────────────────────────── */}
      <div className="am-matrix">
        <div ref={gridRef} className="am-body" onMouseLeave={() => setHover(null)}>
          {w > 0 && (
            <div
              className="am-grid"
              style={{ gridTemplateColumns: `${rowHead}px repeat(${n}, ${cell}px)`, gridTemplateRows: `${colHead}px repeat(${n}, ${cell}px)` }}
            >
              <div className="am-corner">
                <span>query ↓</span>
                <span>neighbour →</span>
              </div>
              {nodes.map((c, j) => (
                <div key={`c${c.ip}`} className={j === aj ? "am-col is-on" : absent(j) ? "am-col is-absent" : "am-col"} title={c.ip}>
                  <span>{c.name}</span>
                </div>
              ))}
              {nodes.map((r, i) => (
                <div key={`r${r.ip}`} className="am-row-wrap">
                  <div className={i === ai ? "am-row is-on" : absent(i) ? "am-row is-absent" : "am-row"}>
                    <b title={r.ip}>{r.name}</b>
                    {cell >= 34 && <span>{r.ip}</span>}
                  </div>
                  {nodes.map((_, j) => {
                    const v = alpha[i][j];
                    const self = i === j;
                    const gone = !self && (absent(i) || absent(j));
                    const on = i === ai && j === aj;
                    const lockedHere = locked && locked[0] === i && locked[1] === j;
                    const cls = ["am-cell", self && "is-self", gone && "is-absent", on && "is-on", lockedHere && "is-locked", (i === ai || j === aj) && !on && "is-cross"]
                      .filter(Boolean)
                      .join(" ");
                    return (
                      <button
                        key={j}
                        className={cls}
                        style={self || gone ? undefined : { background: fill(i, j, v), color: v > 0.45 ? "var(--ink-000)" : "var(--paper-400)" }}
                        onMouseEnter={() => !self && !gone && choose([i, j], false)}
                        onClick={() => !self && !gone && choose([i, j], true)}
                        disabled={self || gone}
                        aria-label={self ? `${r.name} (self)` : `${r.name} attends to ${nodes[j].name}: ${v.toFixed(3)}`}
                      >
                        {!self && !gone && cell >= 28 && v >= 0.01 ? v.toFixed(2).replace(/^0/, "") : ""}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="am-caption">
          <span className="am-legend">
            <span>α 0</span>
            {[0.1, 0.3, 0.5, 0.7, 0.9].map((v) => (
              <i
                key={v}
                className="am-step"
                style={{ background: `color-mix(in srgb, var(--signal-observed) ${Math.round((v / 0.95) * 88)}%, var(--ink-100))` }}
              />
            ))}
            <span>1</span>
            <i className="am-swatch" style={{ background: "var(--sev-critical)" }} />
            <span>hostile</span>
          </span>
          <span className="am-meta" title="Hatched: a host is not its own neighbour. Dashed: not a neighbour this window.">
            {matrix.layer} · {matrix.heads} heads · {matrix.neighbors} neighbours · rows sum to 1
          </span>
          <button className="seg am-sound" aria-pressed={sound} onClick={() => setSound(!sound)} title="Synthesized interface sound">
            Sound {sound ? "on" : "off"}
          </button>
        </div>
      </div>

      {/* ── Saliency ────────────────────────────────────────────────────── */}
      {showInspect && (
      <div
        className={compact ? "am-inspect is-floating" : "am-inspect"}
        style={compact ? (aj >= n / 2 ? { left: 8 } : { right: 8 }) : undefined}
      >
        <div className="am-inspect-head">
          <Micro>Feature saliency</Micro>
          {locked ? <Chip level="nominal">locked · esc</Chip> : <span className="am-hint">{hover ? "hover" : "strongest edge"} · click to lock</span>}
        </div>

        <div className="am-pair">
          <div>
            <Micro>Query</Micro>
            <b>{nodes[ai]?.name}</b>
            <span>{nodes[ai]?.ip}</span>
          </div>
          <span className="am-arrow">→</span>
          <div>
            <Micro>Neighbour</Micro>
            <b>{nodes[aj]?.name}</b>
            <span>{nodes[aj]?.ip}</span>
          </div>
        </div>

        <div className="am-alpha">
          <span className="am-alpha-sym">α</span>
          <span className="am-alpha-val" style={{ color: hostile ? "var(--sev-critical)" : "var(--paper-000)" }}>
            {a.toFixed(3)}
          </span>
          <div className="am-alpha-facts">
            <span>logit {logits[ai]?.[aj]?.toFixed(2)}</span>
            <span>
              rank {rank + 1} of {neighbours}
            </span>
            <span>
              h1 {alpha_heads[0]?.[ai]?.[aj]?.toFixed(3)} · h2 {alpha_heads[1]?.[ai]?.[aj]?.toFixed(3)}
            </span>
          </div>
        </div>

        <div className="am-section">
          <Micro>Input × Gradient → attention logit</Micro>
        </div>
        {terms.length ? (
          <Waterfall terms={terms} baseline={baseline_logit} logit={logits[ai][aj]} hostile={hostile} />
        ) : (
          <Empty>No neighbour on this edge</Empty>
        )}

        <div className="am-foot" title="bita/model/temporal_attention.py">
          key = [ neighbour memory ; 12 edge features ; Δt ]
        </div>
      </div>
      )}
    </div>
  );
}
