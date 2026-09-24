import { useMemo, useState } from "react";
import type { ForecastPoint, PredictionResult } from "../api/types";
import type { Campaign } from "../types/campaign";
import {
  Chip,
  Data,
  Field,
  Micro,
  Panel,
  PanelBody,
  PanelHead,
  Readout,
  sevColor,
  sevFromRisk,
} from "../design/primitives";
import { FORECAST, Legend, OBSERVED } from "../design/charts";

/**
 * ATT&CK coverage matrix.
 *
 * Tactic columns, technique cells. Every entry is a technique the deployed
 * heads can actually emit — the union of TECHNIQUE_VOCAB in
 * `branch_a_gnn_lstm/sequence_dataset.py` (14 classes) and the DeepOP token
 * set, annotated from `control_backend/model_adapter.py:TECHNIQUE_TO_MITRE`.
 * Nothing here is aspirational ATT&CK surface: if a cell exists, a head can
 * predict it.
 *
 * Cell state is live. Observed cells fill warm from OBSERVED campaign nodes
 * and the current verdict; forecast cells hatch cool from the rollout.
 */

interface Tech {
  id: string;
  name: string;
  /** Which head can emit it. */
  heads: string;
}

const MATRIX: { tactic: string; id: string; techniques: Tech[] }[] = [
  {
    tactic: "Reconnaissance",
    id: "TA0043",
    techniques: [
      { id: "T1595", name: "Active Scanning", heads: "Branch A · DeepOP" },
      { id: "T1046", name: "Network Service Discovery", heads: "Branch A" },
    ],
  },
  {
    tactic: "Initial Access",
    id: "TA0001",
    techniques: [
      { id: "T1190", name: "Exploit Public-Facing Application", heads: "Branch A · DeepOP" },
      { id: "T1189", name: "Drive-by Compromise", heads: "Branch A" },
    ],
  },
  {
    tactic: "Execution",
    id: "TA0002",
    techniques: [{ id: "T1204", name: "User Execution", heads: "Branch A" }],
  },
  {
    tactic: "Credential Access",
    id: "TA0006",
    techniques: [{ id: "T1110", name: "Brute Force", heads: "Branch A · DeepOP" }],
  },
  {
    tactic: "Lateral Movement",
    id: "TA0008",
    techniques: [{ id: "T1021", name: "Remote Services", heads: "correlation" }],
  },
  {
    tactic: "Command and Control",
    id: "TA0011",
    techniques: [
      { id: "T1071", name: "Application Layer Protocol", heads: "Branch A · DeepOP" },
      { id: "T1071.001", name: "Web Protocols", heads: "Branch A" },
      { id: "T1568.001", name: "Fast Flux DNS", heads: "Branch A" },
    ],
  },
  {
    tactic: "Collection",
    id: "TA0009",
    techniques: [{ id: "T1005", name: "Data from Local System", heads: "Branch A · DeepOP" }],
  },
  {
    tactic: "Exfiltration",
    id: "TA0010",
    techniques: [{ id: "T1020", name: "Automated Exfiltration", heads: "Branch A" }],
  },
  {
    tactic: "Impact",
    id: "TA0040",
    techniques: [
      { id: "T1498", name: "Network Denial of Service", heads: "Branch A · DeepOP" },
      { id: "T1498.001", name: "Direct Network Flood", heads: "Branch A" },
    ],
  },
];

/** Stage to the techniques a rollout step implies, for forecast shading. */
const STAGE_TECHNIQUES: Record<string, string[]> = {
  Recon: ["T1595", "T1046"],
  InitialAccess: ["T1190", "T1189", "T1110"],
  Execution: ["T1204"],
  LateralMovement: ["T1021"],
  C2: ["T1071", "T1071.001", "T1568.001"],
  Exfiltration: ["T1005", "T1020"],
  Impact: ["T1498", "T1498.001"],
};

type CellState = "observed" | "forecast" | "idle";

interface AttckProps {
  prediction: PredictionResult | null;
  forecast: ForecastPoint[];
  campaign: Campaign | null;
}

/** First token of "T1498 Network Denial of Service" is the technique id. */
function techId(label: string | null | undefined): string | null {
  if (!label) return null;
  const first = String(label).split(/\s+/)[0];
  return /^T\d/.test(first) ? first : null;
}

export default function Attck({ prediction, forecast, campaign }: AttckProps) {
  const [sel, setSel] = useState<string | null>(null);

  const { states, riskOf, counts } = useMemo(() => {
    const states = new Map<string, CellState>();
    const riskOf = new Map<string, number>();

    // Observed: campaign nodes already seen, plus the current verdict.
    for (const node of campaign?.nodes ?? []) {
      if (node.provenance !== "OBSERVED") continue;
      states.set(node.technique_id, "observed");
      riskOf.set(node.technique_id, Math.max(riskOf.get(node.technique_id) ?? 0, node.max_risk_score));
    }
    const now = techId(prediction?.mitre_technique);
    if (now) {
      states.set(now, "observed");
      riskOf.set(now, Math.max(riskOf.get(now) ?? 0, Number(prediction?.risk ?? 0)));
    }

    // Forecast: not-yet-observed campaign nodes and rollout stages.
    for (const node of campaign?.nodes ?? []) {
      if (node.provenance !== "FORECAST" || states.get(node.technique_id) === "observed") continue;
      states.set(node.technique_id, "forecast");
      riskOf.set(node.technique_id, Math.max(riskOf.get(node.technique_id) ?? 0, node.max_risk_score));
    }
    for (const f of forecast) {
      for (const t of STAGE_TECHNIQUES[f.predictedStage ?? ""] ?? []) {
        if (states.get(t) === "observed") continue;
        states.set(t, "forecast");
        riskOf.set(t, Math.max(riskOf.get(t) ?? 0, f.predicted / 100));
      }
    }

    let obs = 0;
    let fc = 0;
    for (const v of states.values()) {
      if (v === "observed") obs += 1;
      else fc += 1;
    }
    const total = MATRIX.reduce((s, c) => s + c.techniques.length, 0);
    return { states, riskOf, counts: { observed: obs, forecast: fc, total } };
  }, [prediction, forecast, campaign]);

  const threshold = Number(prediction?.threshold ?? 0.65);
  const selTech = sel ? (MATRIX.flatMap((c) => c.techniques).find((t) => t.id === sel) ?? null) : null;
  const selTactic = sel ? (MATRIX.find((c) => c.techniques.some((t) => t.id === sel)) ?? null) : null;
  const selRisk = sel ? riskOf.get(sel) : undefined;

  return (
    <div className="at">
      <div className="at-main sheet">
        <Panel flush clip style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <PanelHead
            title="ATT&CK coverage"
            note={`${counts.observed} observed · ${counts.forecast} forecast · ${counts.total} in model vocabulary`}
            aside={
              <Legend
                items={[
                  { color: OBSERVED, label: "observed" },
                  { color: FORECAST, label: "forecast", dashed: true },
                ]}
              />
            }
          />

          <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "var(--s-3)" }}>
            <div className="at-grid">
              {MATRIX.map((col) => (
                <div key={col.id} style={{ minWidth: 0 }}>
                  <div style={{ paddingBottom: "var(--s-2)", borderBottom: "var(--hard)", marginBottom: "var(--s-2)" }}>
                    <div
                      className="t-micro"
                      style={{ color: "var(--paper-000)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                      title={col.tactic}
                    >
                      {col.tactic}
                    </div>
                    <Data size="s" color="var(--paper-600)">
                      {col.id}
                    </Data>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "var(--s-1)" }}>
                    {col.techniques.map((t) => {
                      const state = states.get(t.id) ?? "idle";
                      const on = t.id === sel;
                      const risk = riskOf.get(t.id);
                      const obs = state === "observed";
                      const fc = state === "forecast";
                      const spine = obs
                        ? risk != null
                          ? sevColor(sevFromRisk(risk, threshold))
                          : OBSERVED
                        : fc
                          ? FORECAST
                          : "var(--rule-hair)";

                      return (
                        <button
                          key={t.id}
                          onClick={() => setSel(on ? null : t.id)}
                          className={fc ? "at-cell at-cell-fc" : "at-cell"}
                          style={{
                            borderLeft: `3px solid ${spine}`,
                            background: on || obs ? "var(--ink-200)" : "transparent",
                            outline: on ? "1px solid var(--paper-000)" : undefined,
                            opacity: state === "idle" ? 0.45 : 1,
                          }}
                        >
                          <span
                            className="t-data-s"
                            style={{ color: obs ? "var(--paper-000)" : fc ? FORECAST : "var(--paper-600)", position: "relative" }}
                          >
                            {t.id}
                          </span>
                          <span
                            className="t-data-s"
                            style={{
                              color: "var(--paper-600)",
                              display: "block",
                              position: "relative",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {t.name}
                          </span>
                          {risk != null && (
                            <span
                              className="t-data-s"
                              style={{
                                position: "absolute",
                                top: 5,
                                right: 6,
                                color: obs ? sevColor(sevFromRisk(risk, threshold)) : FORECAST,
                              }}
                            >
                              {risk.toFixed(2)}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* ── Rail ─────────────────────────────────────────────────────── */}
      <div className="at-rail">
        <PanelHead title={selTech ? "Technique" : "Coverage"} />

        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {selTech && selTactic ? (
            <PanelBody>
              <Readout
                label={selTactic.tactic}
                value={selTech.id}
                level={selRisk != null && states.get(selTech.id) === "observed" ? sevFromRisk(selRisk, threshold) : undefined}
                sub={selTech.name}
              />
              <div style={{ marginTop: "var(--s-4)" }}>
                <Field label="Tactic" value={selTactic.tactic} mono={false} />
                <Field label="Tactic ID" value={selTactic.id} />
                <Field label="Technique" value={selTech.id} />
                <Field label="Emitted by" value={selTech.heads} mono={false} />
                <Field
                  label="State"
                  value={states.get(selTech.id) ?? "not seen"}
                  color={
                    states.get(selTech.id) === "observed"
                      ? "var(--paper-000)"
                      : states.get(selTech.id) === "forecast"
                        ? FORECAST
                        : undefined
                  }
                />
                {selRisk != null && (
                  <Field label="Peak risk" value={selRisk.toFixed(3)} color={sevColor(sevFromRisk(selRisk, threshold))} />
                )}
              </div>

              <div style={{ marginTop: "var(--s-4)" }}>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Correlated nodes</Micro>
                {(campaign?.nodes ?? []).some((n) => n.technique_id === selTech.id) ? (
                  (campaign?.nodes ?? [])
                    .filter((n) => n.technique_id === selTech.id)
                    .map((n) => (
                      <Field key={n.node_id} label={n.host_ip} value={`${n.provenance} · ${n.hit_count} hits`} />
                    ))
                ) : (
                  <Data size="s" color="var(--paper-600)">
                    no correlated nodes
                  </Data>
                )}
              </div>
            </PanelBody>
          ) : (
            <>
              <PanelBody>
                <Readout
                  label="Techniques observed"
                  value={String(counts.observed)}
                  scale="hero"
                  level={counts.observed > 0 ? "elevated" : "nominal"}
                  sub={`of ${counts.total} in the model vocabulary`}
                />
              </PanelBody>

              <PanelHead title="Breakdown" />
              <PanelBody>
                <Field label="Observed" value={String(counts.observed)} />
                <Field label="Forecast" value={String(counts.forecast)} color={FORECAST} />
                <Field label="Not seen" value={String(Math.max(0, counts.total - counts.observed - counts.forecast))} />
                <Field label="Vocabulary" value={`${counts.total} techniques`} />
                <Field label="Tactics" value={`${MATRIX.length} covered`} />
              </PanelBody>

              <PanelHead title="Current verdict" />
              <PanelBody>
                {prediction?.mitre_technique ? (
                  <>
                    <div className="t-display-m" style={{ color: "var(--paper-000)", marginBottom: "var(--s-1)" }}>
                      {prediction.mitre_technique}
                    </div>
                    <Data size="s" color="var(--paper-600)">
                      {[prediction.mitre_tactic_id, prediction.mitre_tactic].filter(Boolean).join("  ")}
                    </Data>
                    <div style={{ marginTop: "var(--s-3)" }}>
                      <Chip level={prediction.alert_level}>{String(prediction.alert_level)}</Chip>
                    </div>
                  </>
                ) : (
                  <Data size="s" color="var(--paper-600)">
                    no active classification
                  </Data>
                )}
              </PanelBody>

              <PanelBody>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Scope</Micro>
                <div className="t-body" style={{ color: "var(--paper-400)" }}>
                  Only techniques the deployed heads can emit are listed. Coverage is bounded by the
                  14-class Branch A vocabulary and the DeepOP token set, not by ATT&CK as a whole.
                </div>
              </PanelBody>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
