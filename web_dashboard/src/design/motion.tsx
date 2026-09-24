/**
 * Eased value motion.
 *
 * The console's identity is stepped — indicators tick, controls snap. This
 * module holds the one exception: continuous quantities. A risk score moving
 * 0.871 → 0.884 is an interpolation of a real number, so interpolating it on
 * screen is honest as well as smoother to watch.
 *
 * Everything here degrades to an instant jump under `prefers-reduced-motion`.
 */

import { useEffect, useRef, useState } from "react";

const REDUCED = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** ease-out, no overshoot — matches --ease in tokens.css */
function ease(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Animate a number toward `value` over `duration`, driven by rAF.
 *
 * Re-targets mid-flight rather than restarting, so a stream arriving faster
 * than the animation never stutters — it just bends toward the new target.
 */
export function useEasedNumber(value: number, duration = 280): number {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const start = useRef(0);
  const raf = useRef(0);
  const current = useRef(value);

  useEffect(() => {
    if (!Number.isFinite(value)) return;

    if (REDUCED()) {
      current.current = value;
      setShown(value);
      return;
    }

    from.current = current.current;
    start.current = performance.now();

    const tick = (now: number) => {
      const t = Math.min(1, (now - start.current) / duration);
      const v = from.current + (value - from.current) * ease(t);
      current.current = v;
      setShown(v);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };

    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [value, duration]);

  return shown;
}

/**
 * A numeral that rolls to its new value.
 *
 * `tabular-nums` keeps the glyph box fixed so digits change in place instead
 * of the whole number reflowing on every frame — the difference between a
 * readout and a slot machine.
 */
export function Num({
  value,
  digits = 3,
  duration = 320,
  prefix,
  suffix,
  className,
  style,
}: {
  value: number;
  digits?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  style?: React.CSSProperties;
}) {
  const shown = useEasedNumber(value, duration);
  return (
    <span className={className} style={{ fontVariantNumeric: "tabular-nums", ...style }}>
      {prefix}
      {Number.isFinite(shown) ? shown.toFixed(digits) : "—"}
      {suffix}
    </span>
  );
}

/**
 * Remount-keyed wrapper that replays the entrance wipe. Used on page change:
 * pass the page id as `k` and the whole surface wipes in once.
 */
export function Enter({ k, children }: { k: string | number; children: React.ReactNode }) {
  return (
    <div key={k} className="is-entering" style={{ height: "100%", minHeight: 0 }}>
      {children}
    </div>
  );
}
