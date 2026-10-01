/**
 * Clock time for the console: HH:MM:SS, 24-hour.
 *
 * One formatter, built once. Date#toLocaleTimeString constructs a new Intl
 * formatter on every call, and the live views format hundreds of timestamps a
 * tick — enough to show up as dropped frames.
 */
const HMS = new Intl.DateTimeFormat("en-GB", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

export function clockTime(value: number | string | Date): string {
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? "Invalid Date" : HMS.format(d);
}

/** A horizon ahead of now: "+30s", "+2.5m". */
export function ahead(seconds: number): string {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 120) return `+${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)}s`;
  const m = seconds / 60;
  return `+${Number.isInteger(m) ? m : m.toFixed(1)}m`;
}
