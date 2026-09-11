import { useState, useRef, useEffect } from "react";
import type { SystemStatus } from "../api/types";
import { sendCommand, sendMitigate } from "../api/adapter";

// ── Types ──────────────────────────────────────────────────────
type LogFilter = "ALL" | "CONTROL" | "MODEL" | "ATTACK" | "TELEMETRY";
type Outcome = "success" | "failed";

interface LogEntry {
  id: number;
  ts: string;
  text: string;
  cat: "control" | "model" | "attack" | "telemetry" | "system";
}

interface ControlsProps {
  status: SystemStatus | null;
  logLines: string[];
  onCommandSent?: (cmd: string, line: string) => void;
}

// ── Log helpers ────────────────────────────────────────────────
let _lid = 0;

function makeEntry(text: string, cat: LogEntry["cat"]): LogEntry {
  return {
    id: _lid++,
    ts: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    text,
    cat,
  };
}

function parseCat(text: string): LogEntry["cat"] {
  if (/ml\[|\[ML\]|predict/.test(text)) return "model";
  if (/ALERT|ATTACK|\[ATK\]|attack_event|CRITICAL/.test(text)) return "attack";
  if (/telemetry|sensor|SPAN|\[TEL\]/.test(text)) return "telemetry";
  if (/\[CTL\]|cmd:|mitigate/.test(text)) return "control";
  return "system";
}

function matchFilter(e: LogEntry, f: LogFilter): boolean {
  if (f === "ALL") return true;
  const map: Record<LogFilter, LogEntry["cat"]> = {
    ALL: "system", CONTROL: "control", MODEL: "model", ATTACK: "attack", TELEMETRY: "telemetry",
  };
  return e.cat === map[f];
}

function entryColor(e: LogEntry): string {
  if (e.cat === "attack" || e.text.includes("FAIL")) return "var(--color-status-red)";
  if (e.cat === "control") return "var(--color-status-blue)";
  if (e.cat === "model") return "var(--color-status-amber)";
  if (e.cat === "telemetry") return "var(--color-status-green)";
  if (e.text.includes("WARN")) return "var(--color-status-amber)";
  return "var(--color-text-muted)";
}

// ── Sub-components ─────────────────────────────────────────────

function ControlBtn({
  label, sublabel, onClick, variant = "default",
  disabled = false, disabledReason, executing = false, outcome, fullWidth = false,
}: {
  label: string; sublabel?: string; onClick: () => void;
  variant?: "primary" | "default" | "stop" | "destructive" | "warn";
  disabled?: boolean; disabledReason?: string;
  executing?: boolean; outcome?: Outcome; fullWidth?: boolean;
}) {
  const off = disabled || executing;
  const colorMap: Record<string, string> = {
    primary: "var(--color-status-blue)",
    stop: "var(--color-text-secondary)",
    destructive: "var(--color-status-red)",
    warn: "var(--color-status-amber)",
    default: "var(--color-text-primary)",
  };
  const bgMap: Record<string, string> = {
    primary: "var(--color-status-blue)",
    stop: "transparent",
    destructive: "var(--color-status-red-bg)",
    warn: "var(--color-status-amber-bg)",
    default: "transparent",
  };
  const borderMap: Record<string, string> = {
    primary: "var(--color-status-blue)",
    stop: "var(--color-border-strong)",
    destructive: "color-mix(in srgb, var(--color-status-red) 30%, transparent)",
    warn: "color-mix(in srgb, var(--color-status-amber) 30%, transparent)",
    default: "var(--color-border-strong)",
  };

  const indicatorColor =
    executing ? "var(--color-status-amber)" :
    outcome === "success" ? "var(--color-status-green)" :
    outcome === "failed" ? "var(--color-status-red)" : undefined;

  const displayLabel = executing ? "EXECUTING…" : label;
  const tooltipText = disabled && disabledReason ? disabledReason : undefined;

  return (
    <div title={tooltipText} style={{ width: fullWidth ? "100%" : undefined, minWidth: 0, maxWidth: "100%" }}>
      <button
        onClick={off ? undefined : onClick}
        style={{
          display: "flex", flexDirection: "column", alignItems: "flex-start",
          padding: "7px 11px", borderRadius: 6, fontSize: 11, fontWeight: 600,
          letterSpacing: "0.04em", cursor: off ? "not-allowed" : "pointer",
          border: `1px solid ${borderMap[variant]}`,
          background: variant === "primary" ? (off ? "var(--color-border-strong)" : bgMap[variant]) : bgMap[variant],
          color: variant === "primary" ? "white" : colorMap[variant],
          opacity: off ? 0.48 : 1, transition: "all 0.1s",
          width: fullWidth ? "100%" : undefined, minHeight: 32, maxWidth: "100%",
          fontFamily: "var(--font-sans)", textAlign: "left",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          {indicatorColor && (
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: indicatorColor, flexShrink: 0, display: "inline-block", animation: executing ? "pulse-dot 1.2s ease-in-out infinite" : undefined }} />
          )}
          <span style={{ whiteSpace: "nowrap" }}>{displayLabel}</span>
        </div>
        {sublabel && (
          <span style={{ fontSize: 9, fontWeight: 400, color: variant === "primary" ? "rgba(255,255,255,0.7)" : "var(--color-text-muted)", marginTop: 2, whiteSpace: "normal", lineHeight: 1.4, maxWidth: 260 }}>
            {disabled && disabledReason ? disabledReason : sublabel}
          </span>
        )}
      </button>
    </div>
  );
}

function StatusPill({ label, active, color = "var(--color-status-green)" }: {
  label: string; active: boolean; color?: string;
}) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, letterSpacing: "0.06em",
      background: active ? `color-mix(in srgb, ${color} 12%, transparent)` : "var(--color-base)",
      color: active ? color : "var(--color-text-muted)",
      border: `1px solid ${active ? `color-mix(in srgb, ${color} 30%, transparent)` : "var(--color-border)"}`,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: active ? color : "var(--color-border-strong)", display: "inline-block" }} />
      {label}
    </span>
  );
}

function Group({ title, badge, children }: {
  title: string; badge?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">{title}</span>
        {badge}
      </div>
      <div style={{ padding: "10px 12px", display: "flex", flexWrap: "wrap", gap: 8, alignItems: "flex-start" }}>
        {children}
      </div>
    </div>
  );
}

function ConfirmModal({ label, description, onConfirm, onCancel }: {
  label: string; description?: string; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onCancel}>
      <div className="panel" style={{ width: 420, padding: 24 }} onClick={e => e.stopPropagation()}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 8 }}>{label}</div>
        {description && (
          <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 20, lineHeight: 1.65 }}>{description}</div>
        )}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={onCancel} style={{ padding: "6px 14px", borderRadius: 5, fontSize: 12, fontWeight: 600, border: "1px solid var(--color-border-strong)", background: "transparent", color: "var(--color-text-secondary)", cursor: "pointer" }}>Cancel</button>
          <button onClick={onConfirm} style={{ padding: "6px 14px", borderRadius: 5, fontSize: 12, fontWeight: 600, border: "none", background: "var(--color-status-red)", color: "white", cursor: "pointer" }}>Confirm</button>
        </div>
      </div>
    </div>
  );
}

function InputModal({ title, inputLabel, placeholder, defaultValue = "", onConfirm, onCancel }: {
  title: string; inputLabel: string; placeholder?: string; defaultValue?: string;
  onConfirm: (val: string) => void; onCancel: () => void;
}) {
  const [val, setVal] = useState(defaultValue);
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onCancel}>
      <div className="panel" style={{ width: 420, padding: 24 }} onClick={e => e.stopPropagation()}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 16 }}>{title}</div>
        <label style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 }}>{inputLabel}</label>
        <input
          autoFocus type="text" value={val} onChange={e => setVal(e.target.value)}
          placeholder={placeholder}
          onKeyDown={e => { if (e.key === "Enter" && val.trim()) onConfirm(val.trim()); if (e.key === "Escape") onCancel(); }}
          style={{ width: "100%", padding: "7px 10px", borderRadius: 5, border: "1px solid var(--color-border-strong)", fontSize: 13, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)", background: "var(--color-base)", outline: "none", marginBottom: 16, boxSizing: "border-box" }}
        />
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={onCancel} style={{ padding: "6px 14px", borderRadius: 5, fontSize: 12, fontWeight: 600, border: "1px solid var(--color-border-strong)", background: "transparent", color: "var(--color-text-secondary)", cursor: "pointer" }}>Cancel</button>
          <button onClick={() => val.trim() && onConfirm(val.trim())} style={{ padding: "6px 14px", borderRadius: 5, fontSize: 12, fontWeight: 600, border: "none", background: val.trim() ? "var(--color-status-red)" : "var(--color-border-strong)", color: "white", cursor: val.trim() ? "pointer" : "not-allowed" }}>Apply</button>
        </div>
      </div>
    </div>
  );
}

// ── Tag row for active defenses ────────────────────────────────
function TagRow({ items, onRemove, color = "var(--color-status-red)" }: {
  items: string[]; onRemove?: (v: string) => void; color?: string;
}) {
  if (!items.length) return <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>None</span>;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {items.map(v => (
        <span key={v} style={{
          display: "inline-flex", alignItems: "center", gap: 4,
          fontSize: 10, fontFamily: "var(--font-mono)",
          padding: "2px 7px", borderRadius: 4,
          background: `color-mix(in srgb, ${color} 10%, transparent)`,
          color, border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
        }}>
          {v}
          {onRemove && (
            <button onClick={() => onRemove(v)} style={{ background: "none", border: "none", cursor: "pointer", color, padding: 0, lineHeight: 1, fontSize: 12 }}>×</button>
          )}
        </span>
      ))}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────
export default function Controls({ status, logLines, onCommandSent }: ControlsProps) {
  // Operational state (derived from real backend status)
  const networkOnline = Boolean(status?.network_online || status?.networkStatus === "running");
  const telemetryRunning = Boolean(status?.sensor_active || status?.telemetryStatus === "running");
  const mlRunning = Boolean(status?.ml_active || status?.predictionStatus === "running");
  const workloadsRunning = Boolean(status?.workloads_active);

  // Command execution
  const [runningCmd, setRunningCmd] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({});

  // SOAR defenses
  const [isolatedHosts, setIsolatedHosts] = useState<string[]>([]);
  const [blockedIPs, setBlockedIPs] = useState<string[]>([]);
  const [blockedPorts, setBlockedPorts] = useState<string[]>(["80"]);
  const [credentialsRevoked, setCredentialsRevoked] = useState(false);

  // Console log
  const [log, setLog] = useState<LogEntry[]>(() =>
    logLines.slice(-80).map(t => makeEntry(t, parseCat(t)))
  );
  const [logFilter, setLogFilter] = useState<LogFilter>("ALL");
  const [autoScroll, setAutoScroll] = useState(true);

  // Modals
  const [confirmState, setConfirmState] = useState<{ label: string; description?: string; onConfirm: () => void } | null>(null);
  const [inputState, setInputState] = useState<{ title: string; inputLabel: string; placeholder?: string; defaultValue?: string; onConfirm: (v: string) => void } | null>(null);

  const consoleRef = useRef<HTMLDivElement>(null);

  // Auto-scroll console
  useEffect(() => {
    if (autoScroll && consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [log, autoScroll]);

  function appendLog(text: string, cat: LogEntry["cat"]) {
    const entry = makeEntry(text, cat);
    setLog(prev => [...prev, entry].slice(-500));
    onCommandSent?.(cat, `[${entry.ts}] ${text}`);
  }

  async function executeCommand(cmdName: string, label: string) {
    if (runningCmd) return;
    setRunningCmd(cmdName);
    setOutcomes(prev => { const r = { ...prev }; delete r[cmdName]; return r; });
    appendLog(`>> ${label}`, "control");

    try {
      await sendCommand(cmdName);
      setOutcomes(prev => ({ ...prev, [cmdName]: "success" }));
      appendLog(`${label}: SUCCESS`, "control");
    } catch (e: any) {
      setOutcomes(prev => ({ ...prev, [cmdName]: "failed" }));
      appendLog(`${label}: FAILED (${e.message || "Network error"})`, "control");
    } finally {
      setRunningCmd(null);
    }
  }


  async function executeMitigate(action: string, target: string | null, label: string) {
    appendLog(`mitigate: ${action}${target ? " → " + target : ""}`, "control");
    try {
      await sendMitigate({ action, target } as Parameters<typeof sendMitigate>[0]);
      appendLog(`${label}: recorded`, "control");
    } catch {
      appendLog(`${label}: FAILED (Network error)`, "control");
    }
  }

  function confirm(label: string, description: string, onConfirm: () => void) {
    setConfirmState({ label, description, onConfirm });
  }

  function inputPrompt(title: string, inputLabel: string, options: { placeholder?: string; defaultValue?: string }, onConfirm: (v: string) => void) {
    setInputState({ title, inputLabel, ...options, onConfirm });
  }

  // Console export
  function exportLog() {
    const text = log.map(e => `[${e.ts}] [${e.cat.toUpperCase()}] ${e.text}`).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `cnp-log-${Date.now()}.txt`; a.click();
    URL.revokeObjectURL(url);
  }

  const busy = runningCmd !== null;
  const filtered = log.filter(e => matchFilter(e, logFilter));

  return (
    <div className="controls-page">

      {/* ── Body: left groups + right SOAR ── */}
      <div className="controls-body">

        {/* Left: command groups */}
        <div className="controls-left">

          {/* NETWORK */}
          <Group title="Network" badge={
            <StatusPill label={networkOnline ? "ONLINE" : "OFFLINE"} active={networkOnline} />
          }>
            <ControlBtn
              label="Start Network"
              sublabel="Deploy Containerlab enterprise network"
              variant="primary"
              executing={runningCmd === "start_network"}
              outcome={outcomes["start_network"]}
              disabled={networkOnline || busy}
              disabledReason={networkOnline ? "Network already online" : busy ? "Command in progress" : undefined}
              onClick={() => executeCommand("start_network", "Start Network")}
            />
            <ControlBtn
              label="Health Check"
              sublabel="Run connectivity & security policy check"
              variant="default"
              executing={runningCmd === "healthcheck"}
              outcome={outcomes["healthcheck"]}
              disabled={!networkOnline || busy}
              disabledReason={!networkOnline ? "Requires network online" : undefined}
              onClick={() => executeCommand("healthcheck", "Health Check")}
            />
            <ControlBtn
              label="Stop Network"
              sublabel="Destroy Containerlab — requires confirmation"
              variant="destructive"
              executing={runningCmd === "stop_network"}
              outcome={outcomes["stop_network"]}
              disabled={!networkOnline || busy}
              disabledReason={!networkOnline ? "Network is already offline" : undefined}
              onClick={() => confirm(
                "Stop Network",
                "This destroys the Containerlab network and stops all telemetry, workloads and ML. This action cannot be undone without re-deploying.",
                () => { setConfirmState(null); executeCommand("stop_network", "Stop Network"); }
              )}
            />
            <ControlBtn
              label="Reset Environment"
              sublabel="Stop ML/workloads, clear defenses & model history"
              variant="destructive"
              executing={runningCmd === "reset_environment"}
              outcome={outcomes["reset_environment"]}
              disabled={busy}
              onClick={() => confirm(
                "Reset Environment",
                "Stops ML inference and workloads, disarms external monitoring, clears isolated hosts, blocked IPs/ports, revoked credentials and resets ML history. Does NOT destroy the Containerlab network.",
                () => { setConfirmState(null); executeCommand("reset_environment", "Reset Environment"); }
              )}
            />
          </Group>

          {/* TELEMETRY */}
          <Group title="Telemetry" badge={
            <StatusPill label={telemetryRunning ? "SENSOR RUNNING" : "SENSOR STOPPED"} active={telemetryRunning} />
          }>
            <ControlBtn
              label="Start Sensor"
              sublabel="Begin passive packet capture and flow telemetry"
              variant="primary"
              executing={runningCmd === "start_telemetry"}
              outcome={outcomes["start_telemetry"]}
              disabled={telemetryRunning || !networkOnline || busy}
              disabledReason={!networkOnline ? "Requires network online" : telemetryRunning ? "Sensor already running" : undefined}
              onClick={() => executeCommand("start_telemetry", "Start Sensor")}
            />
            <ControlBtn
              label="Stop Sensor"
              sublabel="Halts packet capture — also stops ML"
              variant="stop"
              executing={runningCmd === "stop_telemetry"}
              outcome={outcomes["stop_telemetry"]}
              disabled={!telemetryRunning || busy}
              disabledReason={!telemetryRunning ? "Sensor is not running" : undefined}
              onClick={() => executeCommand("stop_telemetry", "Stop Sensor")}
            />
            <ControlBtn
              label="Verify Telemetry"
              sublabel="Lab only — run verify_telemetry.sh diagnostic"
              variant="default"
              executing={runningCmd === "verify_telemetry"}
              outcome={outcomes["verify_telemetry"]}
              disabled={!networkOnline || busy}
              disabledReason={!networkOnline ? "Requires lab network online" : undefined}
              onClick={() => executeCommand("verify_telemetry", "Verify Telemetry")}
            />
          </Group>

          {/* ML WORLD MODEL */}
          <Group title="ML World Model" badge={
            <StatusPill
              label={mlRunning ? "ML: LIVE" : "ML: STANDBY"}
              active={mlRunning}
              color="var(--color-status-blue)"
            />
          }>
            <ControlBtn
              label="Start ML"
              sublabel="Activate live inference on incoming telemetry windows"
              variant="primary"
              executing={runningCmd === "start_ml"}
              outcome={outcomes["start_ml"]}
              disabled={mlRunning || !telemetryRunning || busy}
              disabledReason={!telemetryRunning ? "Requires telemetry sensor running" : mlRunning ? "ML already running" : undefined}
              onClick={() => executeCommand("start_ml", "Start ML")}
            />
            <ControlBtn
              label="Stop ML"
              sublabel="Stop inference — sensor remains active"
              variant="stop"
              executing={runningCmd === "stop_ml"}
              outcome={outcomes["stop_ml"]}
              disabled={!mlRunning || busy}
              disabledReason={!mlRunning ? "ML is not running" : undefined}
              onClick={() => executeCommand("stop_ml", "Stop ML")}
            />
          </Group>

          {/* WORKLOADS */}
          <Group title="Normal Workloads" badge={
            <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>Lab only</span>
          }>
            <ControlBtn
              label="Start Workloads"
              sublabel="Enterprise background traffic: browsing / DNS / file / app"
              variant="default"
              executing={runningCmd === "start_normal_traffic"}
              outcome={outcomes["start_normal_traffic"]}
              disabled={workloadsRunning || !networkOnline || busy}
              disabledReason={!networkOnline ? "Requires lab network online" : workloadsRunning ? "Workloads already running" : undefined}
              onClick={() => executeCommand("start_normal_traffic", "Start Workloads")}
            />
            <ControlBtn
              label="Stop Workloads"
              sublabel="Stop background workload processes"
              variant="stop"
              executing={runningCmd === "stop_normal_traffic"}
              outcome={outcomes["stop_normal_traffic"]}
              disabled={!workloadsRunning || busy}
              disabledReason={!workloadsRunning ? "Workloads are not running" : undefined}
              onClick={() => executeCommand("stop_normal_traffic", "Stop Workloads")}
            />
          </Group>

          {/* EXTERNAL MONITORING */}
          <Group title="External Monitoring" badge={
            <StatusPill label={status?.attackStatus === "active" ? "ARMED" : "DISARMED"} active={status?.attackStatus === "active"} color="var(--color-status-amber)" />
          }>
            <ControlBtn
              label="Arm External"
              sublabel="Arm dashboard for external attack"
              variant="default"
              executing={runningCmd === "start_attack"}
              outcome={outcomes["start_attack"]}
              disabled={status?.attackStatus === "active" || busy}
              disabledReason={status?.attackStatus === "active" ? "Already armed" : undefined}
              onClick={() => executeCommand("start_attack", "Arm External")}
            />
            <ControlBtn
              label="Disarm"
              sublabel="Disarm external-attack monitoring"
              variant="stop"
              executing={runningCmd === "stop_attack"}
              outcome={outcomes["stop_attack"]}
              disabled={status?.attackStatus !== "active" || busy}
              disabledReason={status?.attackStatus !== "active" ? "Not armed" : undefined}
              onClick={() => executeCommand("stop_attack", "Disarm")}
            />
          </Group>


        </div>

        {/* Right: SOAR + Active Defenses + Status */}
        <div className="controls-right">

          {/* Defensive SOAR */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Defensive SOAR</span>
              <span style={{ fontSize: 9, color: "var(--color-text-muted)" }}>Records state — no direct infra enforcement</span>
            </div>
            <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
              <ControlBtn
                label="Isolate Host"
                sublabel="Filter flows involving this host from telemetry"
                variant="destructive"
                fullWidth
                disabled={busy}
                onClick={() => inputPrompt(
                  "Isolate Host",
                  "Host IP address",
                  { placeholder: "192.168.x.x" },
                  (ip) => {
                    setInputState(null);
                    setIsolatedHosts(prev => prev.includes(ip) ? prev : [...prev, ip]);
                    executeMitigate("ISOLATE_HOST", ip, `Isolate host ${ip}`);
                  }
                )}
              />
              <ControlBtn
                label="Block IP"
                sublabel="Exclude matching flows from telemetry pipeline"
                variant="destructive"
                fullWidth
                disabled={busy}
                onClick={() => inputPrompt(
                  "Block IP Address",
                  "IP address to block",
                  { placeholder: "x.x.x.x" },
                  (ip) => {
                    setInputState(null);
                    setBlockedIPs(prev => prev.includes(ip) ? prev : [...prev, ip]);
                    executeMitigate("BLOCK_IP", ip, `Block IP ${ip}`);
                  }
                )}
              />
              <ControlBtn
                label="Block Port 80 (explicit)"
                sublabel="Exclude destination-port flows from telemetry"
                variant="destructive"
                fullWidth
                disabled={busy}
                onClick={() => inputPrompt(
                  "Block Port",
                  "Destination port number (e.g. 80, 443, 8080)",
                  { placeholder: "80", defaultValue: "80" },
                  (port) => {
                    setInputState(null);
                    setBlockedPorts(prev => prev.includes(port) ? prev : [...prev, port]);
                    executeMitigate("BLOCK_PORT", port, `Block port ${port}`);
                  }
                )}
              />
              <ControlBtn
                label={credentialsRevoked ? "Credentials Revoked" : "Revoke Credentials"}
                sublabel="Account: compromised_admin — records revocation state only"
                variant="destructive"
                fullWidth
                disabled={credentialsRevoked || busy}
                disabledReason={credentialsRevoked ? "Already revoked" : undefined}
                onClick={() => confirm(
                  "Revoke Credentials — compromised_admin",
                  "Records credential revocation state for account 'compromised_admin'. This does NOT contact an identity provider or actually revoke access to any system.",
                  () => {
                    setConfirmState(null);
                    setCredentialsRevoked(true);
                    executeMitigate("REVOKE_CREDENTIALS", "compromised_admin", "Revoke compromised_admin credentials");
                  }
                )}
              />
              <div style={{ borderTop: "1px solid var(--color-border)", marginTop: 2, paddingTop: 8 }}>
                <ControlBtn
                  label="Clear Defenses"
                  sublabel="Remove all isolation, blocks, and credential revocations"
                  variant="destructive"
                  fullWidth
                  disabled={busy}
                  onClick={() => confirm(
                    "Clear All Defenses",
                    "Clears isolated hosts, blocked IPs, blocked ports and revoked credentials, and resets ML model history. This is not the same as Reset Environment.",
                    () => {
                      setConfirmState(null);
                      setIsolatedHosts([]); setBlockedIPs([]); setBlockedPorts([]); setCredentialsRevoked(false);
                      executeMitigate("CLEAR_DEFENSES", null, "Clear all defenses");
                    }
                  )}
                />
              </div>
            </div>
          </div>

          {/* Active Defenses */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Active Defenses</span>
            </div>
            <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginBottom: 5 }}>ISOLATED HOSTS</div>
                <TagRow items={isolatedHosts} onRemove={ip => setIsolatedHosts(prev => prev.filter(x => x !== ip))} />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginBottom: 5 }}>BLOCKED IPs</div>
                <TagRow items={blockedIPs} onRemove={ip => setBlockedIPs(prev => prev.filter(x => x !== ip))} />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginBottom: 5 }}>BLOCKED PORTS</div>
                <TagRow items={blockedPorts} color="var(--color-status-amber)" onRemove={p => setBlockedPorts(prev => prev.filter(x => x !== p))} />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginBottom: 5 }}>CREDENTIALS</div>
                {credentialsRevoked
                  ? <TagRow items={["compromised_admin"]} color="var(--color-status-amber)" />
                  : <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>None revoked</span>}
              </div>
            </div>
          </div>

          {/* System Status */}
          {status && (
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">System Status</span>
              </div>
              <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 7 }}>
                {([
                  ["Network svc", status.networkStatus],
                  ["Telemetry svc", status.telemetryStatus],
                  ["Prediction svc", status.predictionStatus],
                ] as [string, string][]).map(([label, st]) => (
                  <div key={label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{label}</span>
                    <StatusPill
                      label={st}
                      active={st === "online"}
                      color={st === "online" ? "var(--color-status-green)" : st === "degraded" ? "var(--color-status-amber)" : "var(--color-status-red)"}
                    />
                  </div>
                ))}
                <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: 7, display: "flex", flexDirection: "column", gap: 5 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                    <span style={{ color: "var(--color-text-muted)" }}>Anomaly score</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: status.anomalyScore >= 70 ? "var(--color-status-red)" : "var(--color-text-secondary)" }}>
                      {status.anomalyScore.toFixed(0)} / 100
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                    <span style={{ color: "var(--color-text-muted)" }}>Throughput</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{Math.round(status.throughput)} Mbps</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                    <span style={{ color: "var(--color-text-muted)" }}>Connections</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{status.activeConnections.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* ── Pinned console ── */}
      <div className="panel panel-clipped controls-console">
        <div className="panel-header" style={{ flexShrink: 0 }}>
          <span className="panel-title">System Console</span>
          {/* Filter tabs */}
          <div style={{ display: "flex", gap: 3 }}>
            {(["ALL", "CONTROL", "MODEL", "ATTACK", "TELEMETRY"] as LogFilter[]).map(f => (
              <button key={f} onClick={() => setLogFilter(f)} style={{
                padding: "2px 8px", borderRadius: 3, fontSize: 9, fontWeight: 600, letterSpacing: "0.05em",
                cursor: "pointer", border: "1px solid transparent",
                background: logFilter === f ? "var(--color-text-primary)" : "transparent",
                color: logFilter === f ? "var(--color-surface)" : "var(--color-text-muted)",
                transition: "all 0.1s",
              }}>
                {f}
              </button>
            ))}
          </div>
          <div style={{ flex: 1 }} />
          {/* Controls */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <button
              onClick={() => setAutoScroll(v => !v)}
              style={{
                padding: "2px 8px", borderRadius: 3, fontSize: 9, fontWeight: 600, letterSpacing: "0.05em",
                cursor: "pointer", border: "1px solid var(--color-border-strong)",
                background: autoScroll ? "var(--color-status-green-bg)" : "transparent",
                color: autoScroll ? "var(--color-status-green)" : "var(--color-text-muted)",
              }}
            >
              {autoScroll ? "AUTO-SCROLL ON" : "AUTO-SCROLL PAUSED"}
            </button>
            <button onClick={exportLog} style={{
              padding: "2px 8px", borderRadius: 3, fontSize: 9, fontWeight: 600,
              cursor: "pointer", border: "1px solid var(--color-border-strong)",
              background: "transparent", color: "var(--color-text-muted)", letterSpacing: "0.04em",
            }}>EXPORT</button>
            <button onClick={() => setLog([])} style={{
              padding: "2px 8px", borderRadius: 3, fontSize: 9, fontWeight: 600,
              cursor: "pointer", border: "1px solid var(--color-border-strong)",
              background: "transparent", color: "var(--color-text-muted)", letterSpacing: "0.04em",
            }}>CLEAR</button>
          </div>
          <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginLeft: 4 }}>
            {filtered.length} lines
          </span>
        </div>
        <div
          ref={consoleRef}
          style={{
            flex: 1, overflow: "auto", padding: "8px 12px",
            fontFamily: "var(--font-mono)", fontSize: 10.5,
            background: "#181c22", lineHeight: 1.65,
          }}
        >
          {filtered.length === 0 ? (
            <span style={{ color: "#4a5568" }}>No log entries match the current filter.</span>
          ) : filtered.map(e => (
            <div key={e.id} style={{ color: entryColor(e), whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
              <span style={{ color: "#4a5568", userSelect: "none" }}>{e.ts} </span>
              {e.text}
            </div>
          ))}
        </div>
      </div>

      {/* Modals */}
      {confirmState && (
        <ConfirmModal
          label={confirmState.label}
          description={confirmState.description}
          onConfirm={confirmState.onConfirm}
          onCancel={() => setConfirmState(null)}
        />
      )}
      {inputState && (
        <InputModal
          title={inputState.title}
          inputLabel={inputState.inputLabel}
          placeholder={inputState.placeholder}
          defaultValue={inputState.defaultValue}
          onConfirm={inputState.onConfirm}
          onCancel={() => setInputState(null)}
        />
      )}
    </div>
  );
}
