import { useEffect, useRef, useState } from "react";
import type { PredictionResult, SystemStatus } from "../api/types";
import type { PredictionEnvelope } from "../types/live";
import type { Page } from "../components/layout/Sidebar";
import { sendCommand, sendMitigate } from "../api/adapter";
import { Btn, Chip, Micro, PanelHead, Sheet, Square, sevColor, sevFromRisk } from "../design/primitives";
import { IS_DEMO } from "../env";

/**
 * Controls: run the lab, watch it, defend it.
 *
 * The service strip keeps every lab command the backend accepts
 * (control_backend/commands.py). Response playbooks are sequences of the real
 * /api/mitigate actions, each step logged as it runs; the containment playbook
 * finishes by waiting for the model's risk to fall back under threshold.
 *
 * The left column differs by build. The real console never launches an
 * attack: attacks run against the Containerlab range from outside it, and the
 * column arms external-traffic monitoring and tracks what the model sees. The
 * demo build (IS_DEMO) has scenario buttons that steer the sample stream
 * instead; they are compiled only into that build.
 */

/** Demo scenarios — each one a technique our models can emit. */
export type ScenarioId = "recon" | "probe" | "exploit" | "c2" | "flood";

/** Beginner mode hides lab plumbing (service strip, raw console) behind a toggle. */
const BEGINNER_KEY = "kira-beginner";

interface ControlsProps {
  status: SystemStatus | null;
  logLines: string[];
  onLog: (line: string) => void;
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
  onEmulate: (scenario: ScenarioId) => void;
  onNavigate: (page: Page) => void;
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

/** Demo scenarios, stage by stage — only techniques our models can emit. */
export const SCENARIOS: { id: ScenarioId; stage: string; name: string; chain: string[]; what: string; plain: string }[] = [
  { id: "recon", stage: "1", name: "RECON_BURST", chain: ["T1046"], what: "Port sweep of the DMZ", plain: "A scanner maps which doors (ports) are open on your servers." },
  { id: "probe", stage: "2", name: "CREDENTIAL_STUFFING", chain: ["T1110"], what: "Login brute force against dmz-web", plain: "Someone tries many username/password guesses on the web login." },
  { id: "exploit", stage: "3", name: "EXPLOIT_ATTEMPT", chain: ["T1190"], what: "Exploit payloads against the web app", plain: "An attacker tries a known software flaw to break in." },
  { id: "c2", stage: "4", name: "C2_BEACONING", chain: ["T1071"], what: "Periodic beaconing to an outside server", plain: "A break-in quietly phones home for instructions." },
  { id: "flood", stage: "5", name: "VOLUMETRIC_FLOOD", chain: ["T1498"], what: "SYN flood against the web tier", plain: "The attacker floods the network to knock a service over." },
];

function logColor(line: string): string {
  if (/CRITICAL|ALERT|FATAL/.test(line)) return "var(--sev-critical)";
  if (/ERROR|DENY|BLOCK|FAIL/.test(line)) return "var(--sev-elevated)";
  if (/WARN/.test(line)) return "var(--sev-warning)";
  if (/SUCCESS|INFO/.test(line)) return "var(--paper-400)";
  return "var(--paper-600)";
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ── Playbooks ─────────────────────────────────────────────────────────── */

type PlaybookId = "contain" | "block" | "port" | "creds" | "clear";
type StepState = "wait" | "run" | "done" | "fail";

interface Step {
  label: string;
  detail: string;
  state: StepState;
}

interface Run {
  id: PlaybookId;
  steps: Step[];
  finished: boolean;
}

const PLAYBOOKS: { id: PlaybookId; name: string; what: string; plain: string; danger?: boolean }[] = [
  { id: "contain", name: "Contain host", what: "Isolate the target, block the source, confirm risk falls", plain: "The full emergency response — one click, then watch the risk drop." },
  { id: "block", name: "Block source", what: "Drop the attacker's address from the pipeline", plain: "Ban the attacker's IP so their traffic is ignored." },
  { id: "port", name: "Port lockdown", what: "Exclude the attacked service port", plain: "Close the door they are using to get in." },
  { id: "creds", name: "Credential reset", what: "Record a revocation for the targeted accounts", plain: "Lock out stolen usernames and passwords." },
  { id: "clear", name: "Clear all", what: "Remove every isolation, block and revocation", plain: "Undo every defense and start fresh.", danger: true },
];

export default function Controls({ status, logLines, onLog, prediction, envelope, onEmulate, onNavigate }: ControlsProps) {
  const [running, setRunning] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({});
  const [defenses, setDefenses] = useState<string[]>([]);
  const [confirmState, setConfirm] = useState<{ label: string; body: string; run: () => void } | null>(null);
  const [edit, setEdit] = useState<{ field: "target" | "source" | "port"; value: string } | null>(null);
  const [override, setOverride] = useState<{ target?: string; source?: string; port?: string }>({});
  const [launch, setLaunch] = useState<{ id: ScenarioId; at: number; alertAt: number | null } | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const consoleRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const [beginner, setBeginner] = useState(() => localStorage.getItem(BEGINNER_KEY) !== "off");

  // The containment check reads the latest verdict while the playbook runs.
  const live = useRef({ prediction, envelope });
  live.current = { prediction, envelope };

  useEffect(() => {
    if (follow && consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [logLines, follow]);

  // Clock for the run tracker.
  useEffect(() => {
    if (!launch) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [launch]);

  // First alert after a launch: when the model caught it.
  useEffect(() => {
    if (launch && launch.alertAt == null && prediction?.alert) setLaunch({ ...launch, alertAt: Date.now() });
  }, [launch, prediction?.alert]);

  const lab = Boolean(status?.lab_mode);
  // Without loadable checkpoints the backend refuses Start Inference and says why.
  const modelsLoaded = IS_DEMO || status?.model_loaded !== false;
  const modelError = status?.model_error ? String(status.model_error).split("\n")[0] : null;
  const busy = running !== null;
  const threshold = Number(prediction?.threshold ?? 0.65);
  const risk = Number(prediction?.risk ?? 0);

  // The last target and source the stream named. A contained window names
  // neither, but the playbooks still need to know who they acted on.
  const seen = useRef({ target: "", source: "" });
  const liveTarget = envelope?.target_ip ?? "";
  const liveSource = envelope?.focus_ips?.find((ip) => ip !== envelope?.target_ip) ?? "";
  if (liveTarget) seen.current.target = liveTarget;
  if (liveSource) seen.current.source = liveSource;
  const target = override.target ?? (liveTarget || seen.current.target);
  const source = override.source ?? (liveSource || seen.current.source);
  const port = override.port ?? "80";

  function stamp(text: string) {
    onLog(`[${new Date().toISOString()}] ui   ${text}`);
  }

  /* ── Lab commands ──────────────────────────────────────────────────── */

  async function command(cmd: string, label: string) {
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

  function Cmd({ cmd, label, title, disabled, danger, confirm }: { cmd: string; label: string; title: string; disabled?: boolean; danger?: boolean; confirm?: string }) {
    const gated = LAB_ONLY.has(cmd) && !lab;
    const outcome = outcomes[cmd];
    return (
      <button
        className={`ct2-cmd${danger ? " is-danger" : ""}${outcome ? ` is-${outcome}` : ""}`}
        disabled={busy || disabled || gated}
        title={gated ? "Lab mode only — the backend rejects this command for a live site." : title}
        onClick={() => {
          if (confirm) setConfirm({ label: title, body: confirm, run: () => void command(cmd, title) });
          else void command(cmd, title);
        }}
      >
        {running === cmd ? "…" : label}
        {outcome === "success" && <i>✓</i>}
        {outcome === "failed" && <i>✗</i>}
      </button>
    );
  }

  const services: { name: string; on: boolean; state: [string, string]; level?: unknown; cmds: React.ReactNode }[] = [
    {
      name: "Network",
      on: status?.networkStatus === "running",
      state: ["online", "offline"],
      cmds: (
        <>
          <Cmd cmd="start_network" label="Start" title="Start Network" disabled={status?.networkStatus === "running"} />
          <Cmd cmd="healthcheck" label="Health" title="Health Check" />
          <Cmd cmd="stop_network" label="Stop" title="Stop Network" danger confirm="This tears down the running network fabric. Every node and its state is lost." />
          <Cmd
            cmd="reset_environment"
            label="Reset"
            title="Reset Environment"
            danger
            confirm="Stops inference and workloads, clears recorded defenses and discards model history."
          />
        </>
      ),
    },
    {
      name: "Sensor",
      on: status?.telemetryStatus === "running",
      state: ["capturing", "stopped"],
      cmds: (
        <>
          <Cmd cmd="start_telemetry" label="Start" title="Start Sensor" disabled={status?.telemetryStatus === "running"} />
          <Cmd cmd="verify_telemetry" label="Verify" title="Verify Telemetry" />
          <Cmd cmd="stop_telemetry" label="Stop" title="Stop Sensor — also stops inference" danger disabled={status?.telemetryStatus !== "running"} />
        </>
      ),
    },
    {
      name: "Inference",
      on: status?.predictionStatus === "running",
      state: ["live", "standby"],
      cmds: (
        <>
          <Cmd
            cmd="start_ml"
            label="Start"
            title={modelsLoaded ? "Start Inference" : `Models not loaded — ${modelError ?? "no checkpoints"}`}
            disabled={!modelsLoaded || status?.predictionStatus === "running" || status?.telemetryStatus !== "running"}
          />
          <Cmd cmd="stop_ml" label="Stop" title="Stop Inference — the sensor keeps capturing" disabled={status?.predictionStatus !== "running"} />
        </>
      ),
    },
    {
      name: "Workloads",
      on: Boolean(status?.workloads_active),
      state: ["generating", "idle"],
      cmds: (
        <>
          <Cmd cmd="start_normal_traffic" label="Start" title="Start Workloads" disabled={Boolean(status?.workloads_active)} />
          <Cmd cmd="stop_normal_traffic" label="Stop" title="Stop Workloads" disabled={!status?.workloads_active} />
        </>
      ),
    },
    {
      name: "External",
      on: Boolean(status?.attack_armed),
      state: ["armed", "disarmed"],
      level: status?.attack_armed ? "warning" : undefined,
      cmds: (
        <>
          <Cmd cmd="start_attack" label="Arm" title="Arm external-attack monitoring" disabled={Boolean(status?.attack_armed)} />
          <Cmd cmd="stop_attack" label="Disarm" title="Disarm external-attack monitoring" disabled={!status?.attack_armed} />
        </>
      ),
    },
  ];

  /* ── Emulation ─────────────────────────────────────────────────────── */

  function emulate(id: ScenarioId) {
    onEmulate(id);
    setLaunch({ id, at: Date.now(), alertAt: prediction?.alert ? Date.now() : null });
    setNow(Date.now());
  }

  // Real console: the run tracker follows the armed monitoring window.
  const armed = Boolean(status?.attack_armed ?? status?.attackStatus === "active");
  const armedAt = useRef<number | null>(null);
  if (!IS_DEMO) {
    if (armed && armedAt.current == null) armedAt.current = Date.now();
    if (!armed) armedAt.current = null;
  }
  useEffect(() => {
    if (IS_DEMO) return;
    if (armed) setLaunch((l) => l ?? { id: "recon", at: armedAt.current ?? Date.now(), alertAt: null });
    else setLaunch(null);
  }, [armed]);

  const launched = launch ? (IS_DEMO ? SCENARIOS.find((s) => s.id === launch.id) : { name: "EXTERNAL TRAFFIC" }) : null;
  const lead = envelope?.early_warning?.lead_time_seconds;

  /* ── Playbooks ─────────────────────────────────────────────────────── */

  function stepsFor(id: PlaybookId): Step[] {
    switch (id) {
      case "contain":
        return [
          { label: "Isolate host", detail: `ISOLATE_HOST → ${target || "no target"}`, state: "wait" },
          { label: "Block source", detail: `BLOCK_ATTACKER_IP → ${source || "no source"}`, state: "wait" },
          { label: "Confirm", detail: `model risk under ${threshold.toFixed(2)}`, state: "wait" },
        ];
      case "block":
        return [{ label: "Block source", detail: `BLOCK_ATTACKER_IP → ${source || "no source"}`, state: "wait" }];
      case "port":
        return [{ label: "Block port", detail: `BLOCK_PORT → ${port}`, state: "wait" }];
      case "creds":
        return [{ label: "Revoke credentials", detail: "REVOKE_CREDENTIALS", state: "wait" }];
      case "clear":
        return [{ label: "Clear defences", detail: "CLEAR", state: "wait" }];
    }
  }

  async function mitigate(action: string, tgt: string | null, label: string): Promise<boolean> {
    stamp(`INFO: mitigate ${action}${tgt ? ` → ${tgt}` : ""}`);
    try {
      await sendMitigate({ action, target: tgt } as unknown as Parameters<typeof sendMitigate>[0]);
      stamp(`INFO: ${label}: recorded`);
      setDefenses((p) => (action === "CLEAR" ? [] : [...new Set([...p, label])]));
      return true;
    } catch (e) {
      stamp(`ERROR: ${label}: FAILED (${e instanceof Error ? e.message : "network error"})`);
      return false;
    }
  }

  async function play(id: PlaybookId) {
    const steps = stepsFor(id);
    const mark = (i: number, state: StepState, detail?: string) => {
      steps[i] = { ...steps[i], state, detail: detail ?? steps[i].detail };
      setRun({ id, steps: [...steps], finished: false });
    };
    setRun({ id, steps: [...steps], finished: false });
    stamp(`INFO: playbook ${PLAYBOOKS.find((p) => p.id === id)?.name}`);

    const actions: Record<Exclude<PlaybookId, "contain">, [string, string | null, string]> = {
      block: ["BLOCK_ATTACKER_IP", source || null, `Block ${source}`],
      port: ["BLOCK_PORT", port, `Block port ${port}`],
      creds: ["REVOKE_CREDENTIALS", null, "Revoke credentials"],
      clear: ["CLEAR", null, "Clear defences"],
    };

    if (id === "contain") {
      mark(0, "run");
      const a = target ? await mitigate("ISOLATE_HOST", target, `Isolate ${target}`) : false;
      mark(0, a ? "done" : "fail", a ? undefined : target ? undefined : "no target in the live window");
      mark(1, "run");
      const b = source ? await mitigate("BLOCK_ATTACKER_IP", source, `Block ${source}`) : false;
      mark(1, b ? "done" : "fail", b ? undefined : source ? undefined : "no source in the live window");
      mark(2, "run", "waiting for the next scored windows…");
      let ok = false;
      let last = 0;
      for (let t = 0; t < 30 && !ok; t++) {
        await sleep(500);
        const p = live.current.prediction;
        last = Number(p?.risk ?? 0);
        ok = p != null && last < Number(p.threshold ?? 0.65);
      }
      mark(2, ok ? "done" : "fail", ok ? `risk ${last.toFixed(3)} under ${threshold.toFixed(2)}` : `risk still ${last.toFixed(3)} after 15 s`);
      stamp(ok ? `INFO: containment confirmed, risk ${last.toFixed(3)}` : `WARN: containment not yet confirmed, risk ${last.toFixed(3)}`);
    } else {
      const [action, tgt, label] = actions[id];
      mark(0, "run");
      const ok = await mitigate(action, tgt, label);
      mark(0, ok ? "done" : "fail");
    }
    setRun((r) => (r ? { ...r, finished: true } : r));
  }

  const playing = run != null && !run.finished;

  /* ── Render ────────────────────────────────────────────────────────── */

  const networkOn = status?.networkStatus === "running";
  const sensorOn = status?.telemetryStatus === "running";
  const mlOn = status?.predictionStatus === "running";
  const step1Done = networkOn && sensorOn && mlOn;
  const friendlyStatus = !modelsLoaded
    ? { tone: "warning" as const, text: `Models not loaded — inference is unavailable. ${modelError ?? ""}`.trim() }
    : !step1Done
    ? { tone: "warning" as const, text: "Setup incomplete — start the network, sensor and inference below." }
    : defenses.length > 0
      ? { tone: "warning" as const, text: `Defended — ${defenses.length} defense${defenses.length === 1 ? "" : "s"} active.` }
      : risk >= threshold
        ? { tone: "critical" as const, text: "Under attack — risk is above the alert threshold. Run “Contain host” on the right to stop it." }
        : {
            tone: "nominal" as const,
            text: IS_DEMO
              ? "All clear. The demo path: launch a scenario on the left, watch it get caught, then run “Contain host” on the right."
              : "All clear. Arm external monitoring on the left, run your attack against the lab from outside the console, and watch the model catch it.",
          };

  return (
    <div className={`ct2${beginner ? " is-beginner" : ""}`}>
      <div className="ct2-sheet sheet">
        {/* ── Beginner banner: plain-language state + safe path ───────── */}
        <div className={`ct2-guide is-${friendlyStatus.tone}`}>
          <div className="ct2-guide-main">
            <b>{friendlyStatus.text}</b>
            <span>
              {IS_DEMO
                ? "Demo dashboard — sample data, nothing here touches a network. "
                : "Nothing here touches a real firewall — defenses are recorded, not enforced. "}
              Toggle
              <b> Show advanced</b> for the full service controls and raw console.
            </span>
          </div>
          <button
            className="ct2-guide-toggle"
            aria-pressed={!beginner}
            onClick={() => {
              localStorage.setItem(BEGINNER_KEY, beginner ? "off" : "on");
              setBeginner(!beginner);
            }}
            title={beginner ? "Show service controls and the raw console" : "Back to the simple view"}
          >
            {beginner ? "Show advanced" : "Simple view"}
          </button>
        </div>

        {/* ── Services (advanced only) ─────────────────────────────────── */}
        {!beginner && (
          <div className="ct2-services">
            {services.map((s) => {
              const level = s.level ?? (s.on ? "nominal" : "neutral");
              return (
                <div key={s.name} className="ct2-svc">
                  <div className="ct2-svc-head">
                    <Square status={level} live={s.on} />
                    <b>{s.name}</b>
                    <em style={{ color: sevColor(level) }}>{s.on ? s.state[0] : s.state[1]}</em>
                  </div>
                  <div className="ct2-svc-cmds">{s.cmds}</div>
                </div>
              );
            })}
          </div>
        )}

        {beginner && !step1Done && (
          <div className="ct2-setup">
            <b>One-time setup</b>
            <span>The pipeline is not running yet. Start Network → Sensor → Inference, in that order.</span>
            <button className="ct2-guide-toggle" onClick={() => { localStorage.setItem(BEGINNER_KEY, "off"); setBeginner(false); }}>
              Open service controls
            </button>
          </div>
        )}

        <div className="ct2-body">
          {/* ── Adversary emulation (demo) / external monitor (real) ───── */}
          <div className="ct2-col">
            {IS_DEMO ? (
              <>
                <PanelHead title={beginner ? "1 · Launch a scenario (demo data)" : "Demo scenarios"} note={beginner ? "watch the model catch it" : "sample stream · demo build only"} />
                <div className="ct2-list">
                  {SCENARIOS.map((s) => (
                    <div key={s.id} className={`ct2-row${launch?.id === s.id ? " is-active" : ""}`}>
                      <span className="ct2-row-no">{s.stage}</span>
                      <span className="ct2-row-main">
                        <b>{s.name}</b>
                        <em>{beginner ? s.plain : s.what}</em>
                      </span>
                      <span className="ct2-chain">
                        {s.chain.map((t) => (
                          <i key={t}>{t}</i>
                        ))}
                      </span>
                      <button className="ct2-go" onClick={() => emulate(s.id)} title={`Run ${s.name} — ${beginner ? s.plain : s.what}`}>
                        Launch
                      </button>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <>
                <PanelHead
                  title={beginner ? "1 · Watch for an attack" : "External traffic"}
                  note={beginner ? "attacks run outside the console" : "start_attack · stop_attack"}
                  aside={<Chip level={armed ? "warning" : undefined}>{armed ? "armed" : "disarmed"}</Chip>}
                />
                <div className="ct2-list">
                  <div className={`ct2-row${armed ? " is-active" : ""}`}>
                    <span className="ct2-row-no">1</span>
                    <span className="ct2-row-main">
                      <b>{armed ? "MONITORING ARMED" : "ARM MONITORING"}</b>
                      <em>
                        {beginner
                          ? "Tell the console an outside attack is expected. It never changes the risk score."
                          : "Marks the window an external attack is expected; display only, never a model input"}
                      </em>
                    </span>
                    <button
                      className="ct2-go"
                      disabled={busy}
                      onClick={() => void command(armed ? "stop_attack" : "start_attack", armed ? "Disarm external monitoring" : "Arm external monitoring")}
                    >
                      {armed ? "Disarm" : "Arm"}
                    </button>
                  </div>
                  <div className="ct2-row">
                    <span className="ct2-row-no">2</span>
                    <span className="ct2-row-main">
                      <b>RUN THE ATTACK</b>
                      <em>
                        {lab
                          ? "From the range's external attacker node, against the enterprise segment the sensor mirrors."
                          : "From outside the monitored network, against a host on the SPAN the sensor sees."}
                      </em>
                    </span>
                  </div>
                </div>
              </>
            )}

            <div className={`ct2-track${launched ? "" : " is-idle"}`}>
              {launched && launch ? (
                <>
                  <div className="ct2-track-head">
                    <Micro>Run</Micro>
                    <b>{launched.name}</b>
                    <em>{((now - launch.at) / 1000).toFixed(0)} s ago</em>
                  </div>
                  <div className="ct2-track-grid">
                    <span>
                      <Micro>Model sees</Micro>
                      <b style={{ color: sevColor(sevFromRisk(risk, threshold)) }}>{prediction?.mitre_technique ?? "—"}</b>
                    </span>
                    <span>
                      <Micro>First alert</Micro>
                      <b>{launch.alertAt != null ? `+${((launch.alertAt - launch.at) / 1000).toFixed(1)} s` : "watching…"}</b>
                    </span>
                    <span>
                      <Micro>Lead time</Micro>
                      <b style={{ color: lead != null ? "var(--sev-nominal)" : undefined }}>{lead != null ? `${Number(lead).toFixed(1)} s` : "—"}</b>
                    </span>
                  </div>
                  <button className="ct2-link" onClick={() => onNavigate("stage")}>
                    Watch it on the Forecast Stage →
                  </button>
                </>
              ) : (
                <span>
                  {IS_DEMO
                    ? beginner
                      ? "Pick a scenario above to launch it. K.I.R.A. will detect it, raise an alert, and show how much warning time you got — sample data only."
                      : "Launch a scenario to watch the model catch it: the technique it reads, when it first alerts, and how far ahead."
                    : "Arm monitoring, then run the attack: this tracks the technique the model reads, when it first alerts, and how far ahead."}
                </span>
              )}
            </div>
          </div>

          {/* ── Response playbooks ──────────────────────────────────────── */}
          <div className="ct2-col">
            <PanelHead
              title={beginner ? "2 · Defend (stop the attack)" : "Response playbooks"}
              note={beginner ? "one-click responses" : "/api/mitigate"}
              aside={<Chip level={defenses.length ? "warning" : "nominal"}>{defenses.length} active</Chip>}
            />
            <div className="ct2-targets">
              {(
                [
                  ["target", "Target", target],
                  ["source", "Source", source],
                  ["port", "Port", port],
                ] as const
              ).map(([field, label, value]) => (
                <button key={field} className="ct2-target" onClick={() => setEdit({ field, value })} title={`Change the ${label.toLowerCase()}`}>
                  <Micro>{label}</Micro>
                  <b>{value || "—"}</b>
                </button>
              ))}
            </div>
            <div className="ct2-list">
              {PLAYBOOKS.map((pb) => {
                const mine = run?.id === pb.id ? run : null;
                return (
                  <div key={pb.id} className={`ct2-pb${mine ? " is-active" : ""}`}>
                    <div className="ct2-pb-head">
                      <span className="ct2-row-main">
                        <b>{pb.name}</b>
                        <em>{beginner ? pb.plain : pb.what}</em>
                      </span>
                      <button
                        className={`ct2-go${pb.danger ? " is-danger" : ""}`}
                        disabled={playing || (pb.id === "clear" && !defenses.length)}
                        onClick={() => {
                          if (pb.danger) setConfirm({ label: pb.name, body: pb.what + ".", run: () => void play(pb.id) });
                          else void play(pb.id);
                        }}
                      >
                        Run
                      </button>
                    </div>
                    {mine && (
                      <ol className="ct2-steps">
                        {mine.steps.map((s) => (
                          <li key={s.label} className={`is-${s.state}`}>
                            <span>{s.state === "done" ? "✓" : s.state === "fail" ? "✗" : s.state === "run" ? "▸" : "·"}</span>
                            <b>{s.label}</b>
                            <em>{s.detail}</em>
                          </li>
                        ))}
                      </ol>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="ct2-note">
              {defenses.length > 0 && (
                <div className="ct2-defs">
                  {defenses.map((d) => (
                    <Chip key={d} level="warning">
                      {d}
                    </Chip>
                  ))}
                </div>
              )}
              Recorded as operator intent, not enforced: the model keeps scoring the traffic it sees, and reports it if blocked traffic persists.
            </div>
          </div>
        </div>

        {/* ── Console ──────────────────────────────────────────────────── */}
        <div className="ct2-console">
          <PanelHead
            title={beginner ? "Activity — what the system just did" : "Console"}
            note={beginner ? "plain events" : `${logLines.length} lines`}
            aside={
              <button className="seg" onClick={() => setFollow((f) => !f)} aria-pressed={follow}>
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
        </div>
      </div>

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

      {/* ── Target override ──────────────────────────────────────────── */}
      {edit && (
        <Sheet title={`Playbook ${edit.field}`} onDismiss={() => setEdit(null)}>
          <Micro style={{ marginBottom: "var(--s-2)" }}>
            {edit.field === "port" ? "Destination port" : edit.field === "source" ? "Source address" : "Host address"} · empty follows the live window
          </Micro>
          <input
            autoFocus
            value={edit.value}
            onChange={(e) => setEdit({ ...edit, value: e.target.value })}
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              setOverride((o) => ({ ...o, [edit.field]: edit.value.trim() || undefined }));
              setEdit(null);
            }}
            className="t-data ct2-input"
          />
          <div style={{ display: "flex", gap: "var(--s-2)", justifyContent: "flex-end" }}>
            <Btn onClick={() => setEdit(null)}>Cancel</Btn>
            <Btn
              primary
              onClick={() => {
                setOverride((o) => ({ ...o, [edit.field]: edit.value.trim() || undefined }));
                setEdit(null);
              }}
            >
              Apply
            </Btn>
          </div>
        </Sheet>
      )}

      {/* Screen-reader status for the running command */}
      <div aria-live="polite" style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
        {running ? `Running ${running}` : playing ? `Running playbook ${run?.id}` : ""}
      </div>
    </div>
  );
}
