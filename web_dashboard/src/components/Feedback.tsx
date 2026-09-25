import { useEffect, useState } from "react";
import { Micro, Square, sevColor } from "../design/primitives";

/**
 * Transient feedback: alert toasts and the keyboard reference.
 *
 * Both are hard-edged panels on the same ruled language as everything else —
 * no rounded cards, no drop shadows, no slide-and-fade. A toast arrives with
 * one short wipe and leaves.
 */

/* ── Toasts ────────────────────────────────────────────────────────────── */

export interface Toast {
  id: string;
  level: string;
  title: string;
  body: string;
  /** Optional jump target, e.g. the incident queue. */
  action?: { label: string; run: () => void };
}

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  // Each toast retires itself after 7s — long enough to read on video.
  useEffect(() => {
    const timers = toasts.map((t) => setTimeout(() => onDismiss(t.id), 7000));
    return () => timers.forEach(clearTimeout);
  }, [toasts, onDismiss]);

  if (!toasts.length) return null;

  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="toast is-revealing"
          style={{ borderLeft: `3px solid ${sevColor(t.level)}` }}
          role="status"
        >
          <div style={{ display: "flex", alignItems: "center", gap: "var(--s-2)", marginBottom: 4 }}>
            <Square status={t.level} live />
            <span className="t-micro" style={{ color: sevColor(t.level) }}>
              {t.title}
            </span>
            <button
              onClick={() => onDismiss(t.id)}
              className="t-data-s"
              style={{ marginLeft: "auto", color: "var(--paper-600)", padding: "0 2px" }}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>

          <div className="t-data-s" style={{ color: "var(--paper-400)", whiteSpace: "normal", lineHeight: 1.5 }}>
            {t.body}
          </div>

          {t.action && (
            <button
              onClick={() => {
                t.action?.run();
                onDismiss(t.id);
              }}
              className="t-micro"
              style={{
                marginTop: "var(--s-2)",
                padding: "3px var(--s-2)",
                border: "var(--hard)",
                color: "var(--paper-000)",
              }}
            >
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Keyboard reference ────────────────────────────────────────────────── */

const KEYS: { group: string; rows: [string, string][] }[] = [
  {
    group: "Navigation",
    rows: [
      ["1 – 9, 0", "Jump to a view by its sidebar index"],
      ["Ctrl / Cmd + K", "Command palette"],
      ["[  ]", "Previous / next view"],
    ],
  },
  {
    group: "View",
    rows: [
      ["T", "Toggle ink / paper theme"],
      ["?", "This reference"],
      ["Esc", "Close any overlay"],
    ],
  },
];

export function Shortcuts({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,10,9,0.82)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 210,
        padding: "var(--s-6)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="is-entering"
        style={{ width: 520, maxWidth: "92vw", background: "var(--ink-050)", border: "var(--hard)" }}
      >
        <div
          style={{
            height: "var(--panel-head-h)",
            padding: "0 var(--s-4)",
            borderBottom: "var(--hair)",
            display: "flex",
            alignItems: "center",
          }}
        >
          <span className="t-micro" style={{ color: "var(--paper-400)" }}>
            Keyboard
          </span>
        </div>

        <div style={{ padding: "var(--s-4)" }}>
          {KEYS.map((g) => (
            <div key={g.group} style={{ marginBottom: "var(--s-4)" }}>
              <Micro style={{ marginBottom: "var(--s-2)" }}>{g.group}</Micro>
              {g.rows.map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    justifyContent: "space-between",
                    gap: "var(--s-4)",
                    padding: "6px 0",
                    borderBottom: "var(--hair)",
                  }}
                >
                  <span className="t-data-s" style={{ color: "var(--paper-000)", border: "var(--hair)", padding: "1px 6px" }}>
                    {k}
                  </span>
                  <span className="t-label" style={{ color: "var(--paper-400)", textAlign: "right" }}>
                    {v}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Global keyboard handling for navigation and overlays.
 *
 * Ignores keystrokes originating in a text field, so typing an IP into the
 * mitigation prompt does not teleport the operator to another view.
 */
export function useHotkeys(handlers: {
  onIndex: (i: number) => void;
  onStep: (delta: number) => void;
  onTheme: () => void;
  onShortcuts: () => void;
  onEscape: () => void;
}) {
  const { onIndex, onStep, onTheme, onShortcuts, onEscape } = handlers;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key >= "1" && e.key <= "9") {
        onIndex(Number(e.key) - 1);
      } else if (e.key === "0") {
        // The sidebar numbers ten views; 0 addresses the tenth.
        onIndex(9);
      } else if (e.key === "[") {
        onStep(-1);
      } else if (e.key === "]") {
        onStep(1);
      } else if (e.key.toLowerCase() === "t") {
        onTheme();
      } else if (e.key === "?") {
        onShortcuts();
      } else if (e.key === "Escape") {
        onEscape();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onIndex, onStep, onTheme, onShortcuts, onEscape]);
}

/** Tracks alert transitions and emits a toast when severity escalates. */
export function useAlertToasts(
  level: string | null | undefined,
  detail: { technique?: string | null; host?: string | null },
  onToast: (t: Toast) => void
) {
  const [last, setLast] = useState<string | null>(null);

  useEffect(() => {
    const now = String(level ?? "").toUpperCase();
    if (!now || now === last) return;

    const rank = (s: string) => ["NOMINAL", "WARNING", "ELEVATED", "CRITICAL"].indexOf(s);
    // Only escalation is worth interrupting for; de-escalation is good news.
    if (last != null && rank(now) > rank(last) && rank(now) >= 2) {
      onToast({
        id: `t${Date.now()}`,
        level: now.toLowerCase(),
        title: `${now} alert`,
        body: `${detail.technique ?? "Unclassified activity"}${detail.host ? ` on ${detail.host}` : ""}`,
      });
    }
    setLast(now);
  }, [level, last, detail.technique, detail.host, onToast]);
}
