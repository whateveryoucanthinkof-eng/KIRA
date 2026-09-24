import { useEffect, useRef, useState } from "react";
import type { SystemStatus } from "../api/types";
import { sendCommand, sendMitigate } from "../api/adapter";
import {
  Btn,
  Chip,
  Field,
  Micro,
  Panel,
  PanelBody,
  PanelHead,
  Sheet,
  Square,
  sevColor,
} from "../design/primitives";

interface ControlsProps {
  status: SystemStatus | null;
  logLines: string[];
  onLog: (line: string) => void;
}

type Outcome = "success" | "failed";

/**
 * Lab-only commands. The backend rejects these outright when the site is not
 * in lab mode (control_backend/commands.py:90) — better to disable them here
 * than to let an operator fire a command that cannot succeed.
 */
const LAB_ONLY = new Set([
  "build_environment",
  "start_network",
  "stop_network",
  "healthcheck",
  "verify_telemetry",
  "start_normal_traffic",
  "stop_normal_traffic",
]);

function logColor(line: string): string {
  if (/CRITICAL|ALERT|FATAL/.test(line)) return "var(--sev-critical)";
  if (/ERROR|DENY|BLOCK|FAIL/.test(line)) return "var(--sev-elevated)";
  if (/WARN/.test(line)) return "var(--sev-warning)";
  if (/SUCCESS|INFO/.test(line)) return "var(--paper-400)";
  return "var(--paper-600)";
}

function Group({ title, state, children }: { title: string; state?: { label: string; level: unknown }; children: React.ReactNode }) {
  return (
    <>
      <PanelHead
        title={title}
        aside={
          state ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--s-2)" }}>
              <Square status={state.level} live={state.level === "nominal"} />
              <span className="t-micro" style={{ color: sevColor(state.level) }}>
                {state.label}
              </span>
            </span>
          ) : undefined
        }
      />
      <PanelBody>
        <div className="ct-grid">{children}</div>
      </PanelBody>
    </>
  );
}

export default function Controls({ status, logLines, onLog }: ControlsProps) {
  const [running, setRunning] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({});
  const [defenses, setDefenses] = useState<string[]>([]);
  const [confirmState, setConfirm] = useState<{ label: string; body: string; run: () => void } | null>(null);
  const [promptState, setPrompt] = useState<{
    title: string;
    label: string;
    placeholder?: string;
    run: (v: string) => void;
  } | null>(null);
  const [promptValue, setPromptValue] = useState("");

  const consoleRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  useEffect(() => {
    if (follow && consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [logLines, follow]);

  const lab = Boolean(status?.lab_mode);
  const busy = running !== null;

  const networkOn = status?.networkStatus === "running";
  const sensorOn = status?.telemetryStatus === "running";
  const mlOn = status?.predictionStatus === "running";
  const workloadsOn = Boolean(status?.workloads_active);
  const armed = status?.attackStatus === "active";

  function stamp(text: string) {
    onLog(`[${new Date().toISOString()}] ui   ${text}`);
  }

  async function run(cmd: string, label: string) {
    if (busy) return;
    setRunning(cmd);
    setOutcomes((p) => {
      const next = { ...p };
      delete next[cmd];
      return next;
    });
    stamp(`INFO: >> ${label}`);
    try {
      await sendCommand(cmd);
      setOutcomes((p) => ({ ...p, [cmd]: "success" }));
      stamp(`INFO: ${label}: SUCCESS`);
    } catch (e) {
      setOutcomes((p) => ({ ...p, [cmd]: "failed" }));
      stamp(`ERROR: ${label}: FAILED (${e instanceof Error ? e.message : "network error"})`);
    } finally {
      setRunning(null);
    }
  }

  async function mitigate(action: string, target: string | null, label: string) {
    stamp(`INFO: mitigate ${action}${target ? ` → ${target}` : ""}`);
    try {
      await sendMitigate({ action, target } as unknown as Parameters<typeof sendMitigate>[0]);
      stamp(`INFO: ${label}: recorded`);
      setDefenses((p) => (action.startsWith("CLEAR") ? [] : [...new Set([...p, label])]));
    } catch (e) {
      stamp(`ERROR: ${label}: FAILED (${e instanceof Error ? e.message : "network error"})`);
    }
  }

  /** Command button with its last outcome shown as a spine. */
  function Cmd({
    cmd,
    label,
    sub,
    disabled,
    danger,
    confirm,
  }: {
    cmd: string;
    label: string;
    sub: string;
    disabled?: boolean;
    danger?: boolean;
    confirm?: string;
  }) {
    const gated = LAB_ONLY.has(cmd) && !lab;
    const outcome = outcomes[cmd];
    const isRunning = running === cmd;

    return (
      <div style={{ position: "relative", minWidth: 0 }}>
        <Btn
          full
          danger={danger}
          disabled={busy || disabled || gated}
          title={gated ? "Lab mode only — the backend rejects this command for a live site." : sub}
          sub={isRunning ? "running…" : sub}
          onClick={() => {
            if (confirm) setConfirm({ label, body: confirm, run: () => void run(cmd, label) });
            else void run(cmd, label);
          }}
        >
          {label}
        </Btn>
        {(outcome || gated) && (
          <span
            className="t-micro"
            style={{
              position: "absolute",
              top: 6,
              right: 8,
              color: gated ? "var(--paper-600)" : outcome === "success" ? "var(--sev-nominal)" : "var(--sev-critical)",
              pointerEvents: "none",
            }}
          >
            {gated ? "lab only" : outcome === "success" ? "ok" : "fail"}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="ct">
      <div className="ct-body">
        {/* ── Command groups ───────────────────────────────────────── */}
        <div className="ct-left">
          <Group title="Network" state={{ label: networkOn ? "Online" : "Offline", level: networkOn ? "nominal" : "neutral" }}>
            <Cmd cmd="start_network" label="Start Network" sub="Bring up the enterprise network fabric" disabled={networkOn} />
            <Cmd cmd="healthcheck" label="Health Check" sub="Connectivity and policy diagnostic" />
            <Cmd
              cmd="stop_network"
              label="Stop Network"
              sub="Tear down the network fabric"
              danger
              confirm="This tears down the running network fabric. Every node and its state is lost."
            />
            <Cmd
              cmd="reset_environment"
              label="Reset Environment"
              sub="Stop inference and workloads, clear defenses"
              danger
              confirm="Stops inference and workloads, clears recorded defenses and discards model history."
            />
          </Group>

          <Group title="Telemetry" state={{ label: sensorOn ? "Capturing" : "Stopped", level: sensorOn ? "nominal" : "neutral" }}>
            <Cmd cmd="start_telemetry" label="Start Sensor" sub="Passive capture, 2s flow windows" disabled={sensorOn} />
            <Cmd cmd="stop_telemetry" label="Stop Sensor" sub="Halts capture — also stops inference" disabled={!sensorOn} danger />
            <Cmd cmd="verify_telemetry" label="Verify Telemetry" sub="Run the sensor diagnostic script" />
          </Group>

          <Group title="Inference" state={{ label: mlOn ? "Live" : "Standby", level: mlOn ? "nominal" : "neutral" }}>
            <Cmd cmd="start_ml" label="Start Inference" sub="Score incoming telemetry windows" disabled={mlOn || !sensorOn} />
            <Cmd cmd="stop_ml" label="Stop Inference" sub="Sensor keeps capturing" disabled={!mlOn} />
          </Group>

          <Group title="Workloads" state={{ label: workloadsOn ? "Generating" : "Idle", level: workloadsOn ? "nominal" : "neutral" }}>
            <Cmd
              cmd="start_normal_traffic"
              label="Start Workloads"
              sub="Background browsing, DNS, file and app traffic"
              disabled={workloadsOn}
            />
            <Cmd cmd="stop_normal_traffic" label="Stop Workloads" sub="Halt background traffic generators" disabled={!workloadsOn} />
          </Group>

          <Group title="External monitoring" state={{ label: armed ? "Armed" : "Disarmed", level: armed ? "warning" : "neutral" }}>
            <Cmd cmd="start_attack" label="Arm External" sub="Watch for externally-originated attack traffic" disabled={armed} />
            <Cmd cmd="stop_attack" label="Disarm" sub="Stop external-attack monitoring" disabled={!armed} />
          </Group>
        </div>

        {/* ── Response rail ────────────────────────────────────────── */}
        <div className="ct-right">
          <PanelHead title="Response" aside={<Chip level={defenses.length ? "warning" : "nominal"}>{defenses.length} active</Chip>} />
          <PanelBody style={{ display: "flex", flexDirection: "column", gap: "var(--s-2)" }}>
            {/* These are recorded intents, not enforcement — say so plainly. */}
            <div
              className="t-data-s"
              style={{ color: "var(--paper-600)", paddingBottom: "var(--s-2)", borderBottom: "var(--hair)", marginBottom: "var(--s-1)" }}
            >
              Recorded as operator intent and applied as a telemetry filter. Not wired to firewall or policy enforcement.
            </div>

            <Btn
              full
              sub="Filter flows involving a host"
              onClick={() =>
                setPrompt({
                  title: "Isolate host",
                  label: "Host address",
                  placeholder: "10.0.2.40",
                  run: (v) => void mitigate("ISOLATE_HOST", v, `Isolate ${v}`),
                })
              }
            >
              Isolate Host
            </Btn>

            <Btn
              full
              sub="Exclude matching flows from the pipeline"
              onClick={() =>
                setPrompt({
                  title: "Block address",
                  label: "Source address",
                  placeholder: "192.168.100.7",
                  run: (v) => void mitigate("BLOCK_ATTACKER_IP", v, `Block ${v}`),
                })
              }
            >
              Block Address
            </Btn>

            <Btn
              full
              sub="Exclude a destination port"
              onClick={() =>
                setPrompt({
                  title: "Block port",
                  label: "Destination port",
                  placeholder: "80",
                  run: (v) => void mitigate("BLOCK_PORT", v, `Block port ${v}`),
                })
              }
            >
              Block Port
            </Btn>

            <Btn full sub="Record a credential revocation" onClick={() => void mitigate("REVOKE_CREDENTIALS", null, "Revoke credentials")}>
              Revoke Credentials
            </Btn>

            <Btn
              full
              danger
              sub="Remove every isolation, block and revocation"
              disabled={!defenses.length}
              onClick={() =>
                setConfirm({
                  label: "Clear defenses",
                  body: "Removes every recorded isolation, block and credential revocation.",
                  run: () => void mitigate("CLEAR", null, "Clear defenses"),
                })
              }
            >
              Clear Defenses
            </Btn>

            {defenses.length > 0 && (
              <div style={{ marginTop: "var(--s-2)", paddingTop: "var(--s-3)", borderTop: "var(--hard)" }}>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Active</Micro>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s-1)" }}>
                  {defenses.map((d) => (
                    <Chip key={d} level="warning">
                      {d}
                    </Chip>
                  ))}
                </div>
              </div>
            )}
          </PanelBody>

          <PanelHead title="Runtime" />
          <PanelBody>
            <Field label="Mode" value={status?.mode ?? "—"} />
            <Field label="Site" value={status?.site_id ?? "—"} />
            <Field label="Lab mode" value={lab ? "yes" : "no"} />
            <Field label="Interface" value={status?.sensor_interface ?? "—"} />
            <Field label="Nodes" value={status ? `${status.nodes_running ?? 0} / ${status.total_nodes ?? 0}` : "—"} />
            <Field label="Topology" value={status ? `${status.topology_nodes ?? 0}n · ${status.topology_edges ?? 0}e` : "—"} />
            <Field label="Model" value={status?.model_loaded ? "loaded" : "not loaded"} color={status?.model_loaded ? "var(--sev-nominal)" : "var(--sev-warning)"} />
            <Field label="Active command" value={status?.active_command ?? running ?? "idle"} />
          </PanelBody>
        </div>
      </div>

      {/* ── Console ──────────────────────────────────────────────────── */}
      <Panel clip className="ct-console">
        <PanelHead
          title="Console"
          note={`${logLines.length} lines`}
          aside={
            <button
              onClick={() => setFollow((f) => !f)}
              aria-pressed={follow}
              className="t-micro"
              style={{
                padding: "3px var(--s-2)",
                border: "var(--hair)",
                background: follow ? "var(--paper-000)" : "transparent",
                color: follow ? "var(--ink-000)" : "var(--paper-600)",
              }}
            >
              follow
            </button>
          }
        />
        <div className="log" ref={consoleRef}>
          {logLines.length ? (
            logLines.map((line, i) => (
              <div key={i} className="log-line" style={{ color: logColor(line) }}>
                {line}
              </div>
            ))
          ) : (
            <span className="log-line">— idle —</span>
          )}
        </div>
      </Panel>

      {/* ── Confirm ──────────────────────────────────────────────────── */}
      {confirmState && (
        <Sheet title={confirmState.label} onDismiss={() => setConfirm(null)}>
          <div className="t-body" style={{ color: "var(--paper-400)", marginBottom: "var(--s-6)" }}>
            {confirmState.body}
          </div>
          <div style={{ display: "flex", gap: "var(--s-2)", justifyContent: "flex-end" }}>
            <Btn onClick={() => setConfirm(null)}>Cancel</Btn>
            <Btn
              danger
              primary
              onClick={() => {
                confirmState.run();
                setConfirm(null);
              }}
            >
              {confirmState.label}
            </Btn>
          </div>
        </Sheet>
      )}

      {/* ── Prompt ───────────────────────────────────────────────────── */}
      {promptState && (
        <Sheet
          title={promptState.title}
          onDismiss={() => {
            setPrompt(null);
            setPromptValue("");
          }}
        >
          <Micro style={{ marginBottom: "var(--s-2)" }}>{promptState.label}</Micro>
          <input
            autoFocus
            value={promptValue}
            placeholder={promptState.placeholder}
            onChange={(e) => setPromptValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || !promptValue.trim()) return;
              promptState.run(promptValue.trim());
              setPrompt(null);
              setPromptValue("");
            }}
            className="t-data"
            style={{
              width: "100%",
              padding: "var(--s-2)",
              background: "var(--ink-000)",
              border: "var(--hard)",
              color: "var(--paper-000)",
              outline: "none",
              marginBottom: "var(--s-6)",
            }}
          />
          <div style={{ display: "flex", gap: "var(--s-2)", justifyContent: "flex-end" }}>
            <Btn
              onClick={() => {
                setPrompt(null);
                setPromptValue("");
              }}
            >
              Cancel
            </Btn>
            <Btn
              primary
              disabled={!promptValue.trim()}
              onClick={() => {
                promptState.run(promptValue.trim());
                setPrompt(null);
                setPromptValue("");
              }}
            >
              Apply
            </Btn>
          </div>
        </Sheet>
      )}

      {/* Screen-reader status for the running command */}
      <div aria-live="polite" style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
        {running ? `Running ${running}` : ""}
      </div>
    </div>
  );
}
