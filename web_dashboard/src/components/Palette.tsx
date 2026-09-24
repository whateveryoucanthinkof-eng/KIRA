import { useEffect, useMemo, useRef, useState } from "react";
import { Micro } from "../design/primitives";

/**
 * Command palette (Ctrl/Cmd+K).
 *
 * Navigation and actions on one keystroke. Operators do not hunt through a
 * sidebar during an incident, and a console that can only be driven by mouse
 * reads as a dashboard rather than a tool.
 *
 * Styled as a hard-edged sheet on a flat scrim — no blur, no radius, no
 * shadow — so it belongs to the same document as everything behind it.
 */

export interface PaletteItem {
  id: string;
  label: string;
  group: string;
  /** Right-aligned mono hint: a shortcut, a target, a state. */
  hint?: string;
  run: () => void;
  danger?: boolean;
}

export default function Palette({ items, open, onOpenChange }: {
  items: PaletteItem[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  /* Global hotkey. Capture phase so it beats the browser's find-in-page. */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenChange(!open);
      } else if (e.key === "Escape" && open) {
        onOpenChange(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    // Focus after paint, or the input is not in the DOM yet.
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  /** Subsequence match — "ovw" finds "Overview", the way editors do it. */
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((it) => {
      const hay = `${it.group} ${it.label} ${it.hint ?? ""}`.toLowerCase();
      let i = 0;
      for (const ch of q) {
        i = hay.indexOf(ch, i);
        if (i === -1) return false;
        i += 1;
      }
      return true;
    });
  }, [items, query]);

  useEffect(() => {
    if (cursor >= matches.length) setCursor(Math.max(0, matches.length - 1));
  }, [matches.length, cursor]);

  // Keep the highlighted row in view as the cursor moves.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[data-active="true"]')?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  if (!open) return null;

  function choose(item: PaletteItem | undefined) {
    if (!item) return;
    onOpenChange(false);
    item.run();
  }

  return (
    <div
      onClick={() => onOpenChange(false)}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,10,9,0.82)",
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "center",
        paddingTop: "12vh",
        zIndex: 200,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="is-entering"
        style={{ width: 620, maxWidth: "92vw", background: "var(--ink-050)", border: "var(--hard)" }}
      >
        <input
          ref={inputRef}
          value={query}
          placeholder="Search views and actions…"
          onChange={(e) => {
            setQuery(e.target.value);
            setCursor(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setCursor((c) => Math.min(matches.length - 1, c + 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setCursor((c) => Math.max(0, c - 1));
            } else if (e.key === "Enter") {
              e.preventDefault();
              choose(matches[cursor]);
            }
          }}
          className="t-data"
          style={{
            width: "100%",
            padding: "var(--s-3) var(--s-4)",
            background: "var(--ink-000)",
            border: "none",
            borderBottom: "var(--hard)",
            color: "var(--paper-000)",
            outline: "none",
          }}
        />

        <div ref={listRef} style={{ maxHeight: "48vh", overflowY: "auto" }}>
          {matches.length ? (
            matches.map((it, i) => {
              const on = i === cursor;
              return (
                <button
                  key={it.id}
                  data-active={on}
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => choose(it)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "var(--s-3)",
                    width: "100%",
                    padding: "9px var(--s-4)",
                    textAlign: "left",
                    background: on ? "var(--paper-000)" : "transparent",
                    color: on ? "var(--ink-000)" : "var(--paper-400)",
                    borderLeft: `3px solid ${it.danger ? "var(--sev-critical)" : on ? "var(--ink-000)" : "transparent"}`,
                  }}
                >
                  <span className="t-micro" style={{ width: 88, flexShrink: 0, opacity: 0.6 }}>
                    {it.group}
                  </span>
                  <span className="t-label" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {it.label}
                  </span>
                  {it.hint && (
                    <span className="t-data-s" style={{ marginLeft: "auto", opacity: 0.6, flexShrink: 0 }}>
                      {it.hint}
                    </span>
                  )}
                </button>
              );
            })
          ) : (
            <div style={{ padding: "var(--s-6)", textAlign: "center" }}>
              <Micro>No match</Micro>
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            gap: "var(--s-4)",
            padding: "var(--s-2) var(--s-4)",
            borderTop: "var(--hair)",
            background: "var(--ink-100)",
          }}
        >
          {[
            ["↑↓", "navigate"],
            ["↵", "run"],
            ["esc", "close"],
          ].map(([k, v]) => (
            <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-1)" }}>
              <span className="t-data-s" style={{ color: "var(--paper-000)", border: "var(--hair)", padding: "1px 5px" }}>
                {k}
              </span>
              <Micro>{v}</Micro>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
