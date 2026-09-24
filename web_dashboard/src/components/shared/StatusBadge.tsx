/**
 * Compatibility shim.
 *
 * The visual vocabulary now lives in `src/design/primitives.tsx`:
 *   StatusDot     → Square  (a square that ticks, not a dot that pulses)
 *   SeverityBadge → Chip
 *   ThreatBadge   → Chip
 *
 * Prefer importing from `design/primitives` directly in new code.
 */

import { Chip, Square } from "../../design/primitives";

export function StatusDot({ status, pulse }: { status?: unknown; pulse?: boolean }) {
  return <Square status={status} live={pulse} />;
}

export function SeverityBadge({ severity, children }: { severity?: unknown; children?: React.ReactNode }) {
  return <Chip level={severity}>{children ?? String(severity ?? "info")}</Chip>;
}

export function ThreatBadge({ level }: { level?: string }) {
  return <Chip level={level}>{level ?? "low"}</Chip>;
}

export { Chip, Square };
