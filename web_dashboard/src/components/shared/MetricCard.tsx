/**
 * Compatibility shim — superseded by `Readout` in `src/design/primitives.tsx`.
 * Prefer `Readout` (inside a `Panel`) in new code.
 */

import { Panel, PanelBody, Readout } from "../../design/primitives";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  accent?: "green" | "amber" | "red" | "blue" | "default";
  mono?: boolean;
}

const ACCENT_TO_LEVEL: Record<string, string | undefined> = {
  green: "nominal",
  amber: "warning",
  red: "critical",
  blue: undefined,
  default: undefined,
};

export function MetricCard({ label, value, unit, sub, accent = "default" }: MetricCardProps) {
  return (
    <Panel flush>
      <PanelBody style={{ padding: "var(--s-3)" }}>
        <Readout label={label} value={value} unit={unit} sub={sub} level={ACCENT_TO_LEVEL[accent]} />
      </PanelBody>
    </Panel>
  );
}

export { Readout };
