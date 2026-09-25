import { useState } from "react";
import type { SystemStatus, PredictionResult } from "../api/types";
import type { PredictionEnvelope } from "../types/live";
import { Chip, Data, Field, Micro, Panel, PanelBody, PanelHead, Readout, Meter } from "../design/primitives";
import { FORECAST, OBSERVED } from "../design/charts";

/**
 * Model architecture and serving contract.
 *
 * Every figure here is read from the shipped manifests in `saved_models/`
 * (`branch_a.manifest.json`, `branch_b.manifest.json`, `deepop.manifest.json`)
 * and the TGNE config — not invented for display. The temporal contract row
 * is read live off `model_meta`, which the adapter adopts from the loaded
 * checkpoint (`model_adapter.py:_adopt_contract`), so it always reflects the
 * weights actually serving rather than a hardcoded L/K.
 */

interface Stage {
  key: string;
  ord: string;
  name: string;
  type: string;
  role: string;
  checkpoint: string;
  version: string;
  params: [string, string][];
  outputs: string[];
}

const STAGES: Stage[] = [
  {
    key: "tgne",
    ord: "01",
    name: "TGNE",
    type: "BiTA · BiGRU + Transformer",
    role: "Temporal graph encoder. Consumes the flow graph for a 2s window and emits a 12-dimensional latent per host.",
    checkpoint: "bita/saved_models/bita_bigru_transformer-warden_alerts.pth",
    version: "unified_final",
    params: [
      ["aggregator", "bigru_transformer"],
      ["embedding module", "graph_attention"],
      ["memory updater", "gru"],
      ["node feature dim", "12"],
      ["edge feature dim", "12"],
      ["attention heads", "2"],
      ["layers", "1"],
      ["dropout", "0.1"],
    ],
    outputs: ["h_emb — 12-D host latent"],
  },
  {
    key: "branch_a",
    ord: "02",
    name: "Branch A",
    type: "MultiTaskLSTM",
    role: "Nowcast. Scores the current window for risk, ATT&CK technique and kill-chain stage from the 27-D history sequence.",
    checkpoint: "saved_models/branch_a/branch_a_lstm.pt",
    version: "1.1.0-live-retrained",
    params: [
      ["input dim", "27"],
      ["hidden dim", "64"],
      ["layers", "2"],
      ["risk head", "Linear(1) · Sigmoid"],
      ["technique head", "Linear(14) · logits"],
      ["stage head", "Linear(4) · logits"],
      ["optimizer", "AdamW · lr 1e-3"],
      ["train hosts", "4,007"],
    ],
    outputs: ["risk_score", "technique_logits (14)", "gradation_logits (4)", "attention_weights"],
  },
  {
    key: "branch_b",
    ord: "03",
    name: "Branch B",
    type: "HostWorldDynamicsTransformer",
    role: "World model. Rolls the host latent forward K steps and scores infiltration risk along the predicted trajectory.",
    checkpoint: "saved_models/branch_b/host_wdt.pt",
    version: "1.0.0",
    params: [
      ["d_latent", "12"],
      ["d_model", "64"],
      ["attention heads", "4"],
      ["encoder layers", "3"],
      ["feedforward", "128"],
      ["risk head", "InfiltrationRiskHead(32)"],
      ["aggregation", "peak_severity_max"],
      ["optimizer", "AdamW · lr 5e-4"],
    ],
    outputs: ["H_hat — K-step latent rollout", "step_risks [B,K]", "cumulative_risk [B]"],
  },
  {
    key: "deepop",
    ord: "04",
    name: "DeepOP",
    type: "Causal Window Attention decoder",
    role: "Sequence decoder. Turns the predicted trajectory into a future (category, technique) token sequence.",
    checkpoint: "saved_models/deepop/cwa_forecast_decoder.pt",
    version: "1.0.0",
    params: [
      ["d_latent", "12"],
      ["d_model", "72"],
      ["vocab size", "10"],
      ["decoder layers", "2"],
      ["attention heads", "6"],
      ["window sizes", "2 · 4 · 8"],
      ["attention", "CausalWindowAttention_ONW"],
      ["conditioning", "WDT predicted + oracle"],
    ],
    outputs: ["technique token sequence", "per-step token probabilities"],
  },
];

/** data_unification/host_attributes.py — the 15 host temporal attributes. */
const HOST_ATTRS: [string, string][] = [
  ["flow_count", "Connectivity"],
  ["unique_peers", "Connectivity"],
  ["unique_dst_ports", "Connectivity"],
  ["peer_density", "Connectivity"],
  ["fwd_bytes", "Volume"],
  ["bwd_bytes", "Volume"],
  ["total_bytes", "Volume"],
  ["fwd_packets", "Volume"],
  ["bwd_packets", "Volume"],
  ["total_packets", "Volume"],
  ["tcp_ratio", "Protocol"],
  ["udp_ratio", "Protocol"],
  ["avg_duration", "Timing"],
  ["byte_rate", "Rate"],
  ["packet_rate", "Rate"],
];

interface ModelProps {
  status: SystemStatus | null;
  prediction: PredictionResult | null;
  envelope: PredictionEnvelope | null;
}

export default function Model({ status, prediction, envelope }: ModelProps) {
  const [sel, setSel] = useState<string>("branch_a");
  const stage = STAGES.find((s) => s.key === sel) ?? STAGES[0];

  const meta = status?.model_meta as
    | { name?: string; version?: string; history_steps?: number; forecast_steps?: number; window_seconds?: number; feature_count?: number; threshold?: number; checkpoint?: string }
    | null
    | undefined;

  const L = meta?.history_steps;
  const K = meta?.forecast_steps;
  const dt = meta?.window_seconds;
  const lat = envelope?.latency;

  return (
    <div className="md">
      <div className="md-main sheet">
        {/* ── Pipeline ─────────────────────────────────────────────── */}
        <Panel flush>
          <PanelHead
            title="Inference pipeline"
            note="flows → TGNE → Branch A ∥ Branch B → DeepOP"
            aside={<Chip level={status?.model_loaded ? "nominal" : "warning"}>{status?.model_loaded ? "loaded" : "not loaded"}</Chip>}
          />
          <PanelBody>
            <div className="md-pipe">
              {STAGES.map((s, i) => {
                const on = s.key === sel;
                return (
                  <button
                    key={s.key}
                    onClick={() => setSel(s.key)}
                    className="md-stage"
                    style={{
                      background: on ? "var(--paper-000)" : "var(--ink-200)",
                      color: on ? "var(--ink-000)" : "var(--paper-000)",
                      borderLeft: `3px solid ${on ? "var(--ink-000)" : i < 2 ? OBSERVED : FORECAST}`,
                    }}
                  >
                    <span className="t-data-s" style={{ opacity: 0.6, display: "block" }}>
                      {s.ord}
                    </span>
                    <span className="t-label" style={{ display: "block", marginTop: 2 }}>
                      {s.name}
                    </span>
                    <span
                      className="t-data-s"
                      style={{ display: "block", opacity: 0.66, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                    >
                      {s.type}
                    </span>
                  </button>
                );
              })}
            </div>

            <div
              className="t-data-s"
              style={{ color: "var(--paper-600)", marginTop: "var(--s-3)", paddingTop: "var(--s-2)", borderTop: "var(--hair)" }}
            >
              Branch A nowcasts the present window. Branch B rolls the same latent forward and hands the
              predicted trajectory to DeepOP, which decodes it into a technique sequence. Both branches
              read the identical TGNE embedding — nothing downstream sees raw packets.
            </div>
          </PanelBody>
        </Panel>

        {/* ── Selected stage ───────────────────────────────────────── */}
        <Panel flush>
          <PanelHead title={`${stage.ord} · ${stage.name}`} note={stage.type} aside={<Data size="s" color="var(--paper-600)">v{stage.version}</Data>} />
          <PanelBody>
            <div className="t-body" style={{ color: "var(--paper-400)", maxWidth: 780, marginBottom: "var(--s-4)" }}>
              {stage.role}
            </div>

            <div className="md-cols">
              <div>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Architecture</Micro>
                {stage.params.map(([k, v]) => (
                  <Field key={k} label={k} value={v} />
                ))}
              </div>

              <div>
                <Micro style={{ marginBottom: "var(--s-2)" }}>Outputs</Micro>
                {stage.outputs.map((o) => (
                  <div
                    key={o}
                    className="t-data-s"
                    style={{ padding: "6px 0", borderBottom: "var(--hair)", color: "var(--paper-000)" }}
                  >
                    {o}
                  </div>
                ))}

                <Micro style={{ margin: "var(--s-4) 0 var(--s-2)" }}>Checkpoint</Micro>
                <Data size="s" color="var(--paper-600)" style={{ wordBreak: "break-all" }}>
                  {stage.checkpoint}
                </Data>
              </div>
            </div>
          </PanelBody>
        </Panel>

        {/* ── Feature contract ─────────────────────────────────────── */}
        <Panel flush>
          <PanelHead title="Feature contract" note="27-D state vector · 12 TGNE latent + 15 host attributes" />
          <PanelBody>
            <div style={{ display: "flex", gap: 2, marginBottom: "var(--s-3)" }}>
              <div style={{ flex: 12, height: 10, background: OBSERVED }} title="12 TGNE latent dimensions" />
              <div style={{ flex: 15, height: 10, background: "var(--ink-300)", border: "var(--hair)" }} title="15 host temporal attributes" />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "var(--s-4)" }}>
              <Data size="s" color="var(--paper-600)">
                H_emb_0 … H_emb_11 — learned graph latent
              </Data>
              <Data size="s" color="var(--paper-600)">
                15 host temporal attributes
              </Data>
            </div>

            <div className="md-attrs">
              {HOST_ATTRS.map(([name, group]) => (
                <div key={name} style={{ padding: "5px var(--s-2)", border: "var(--hair)", minWidth: 0 }}>
                  <div
                    className="t-data-s"
                    style={{ color: "var(--paper-000)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                  >
                    {name}
                  </div>
                  <div className="t-micro" style={{ color: "var(--paper-600)" }}>
                    {group}
                  </div>
                </div>
              ))}
            </div>
          </PanelBody>
        </Panel>
      </div>

      {/* ── Rail ─────────────────────────────────────────────────────── */}
      <div className="md-rail">
        <PanelHead title="Serving contract" />
        <PanelBody>
          <Readout
            label="State vector"
            value={String(meta?.feature_count ?? 27)}
            unit="dims"
            scale="hero"
            sub="12 latent + 15 attributes"
          />
        </PanelBody>

        <PanelHead title="Temporal contract" note="read from the loaded checkpoint" />
        <PanelBody>
          <Field label="Window Δt" value={dt != null ? `${dt}s` : "—"} />
          <Field label="History L" value={L != null ? `${L} windows` : "—"} />
          <Field label="Forecast K" value={K != null ? `${K} windows` : "—"} />
          <Field label="History span" value={L != null && dt != null ? `${(L * dt).toFixed(0)}s` : "—"} />
          <Field label="Forecast span" value={K != null && dt != null ? `${(K * dt).toFixed(0)}s` : "—"} />
          <Field
            label="Alert threshold"
            value={meta?.threshold != null ? meta.threshold.toFixed(3) : prediction?.threshold?.toFixed(3) ?? "—"}
          />
          <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: "var(--s-3)" }}>
            The adapter adopts this from the checkpoint and refuses to serve on mismatch, so it is never
            hardcoded in the console.
          </div>
        </PanelBody>

        <PanelHead title="Pipeline latency" />
        <PanelBody>
          <Field label="Telemetry" value={lat?.telemetry_ms != null ? `${Number(lat.telemetry_ms).toFixed(2)} ms` : "—"} />
          <Field label="Inference" value={lat?.inference_ms != null ? `${Number(lat.inference_ms).toFixed(2)} ms` : "—"} />
          <Field
            label="Total"
            value={lat?.total_ms != null ? `${Number(lat.total_ms).toFixed(2)} ms` : "—"}
            color={lat?.total_ms != null && Number(lat.total_ms) < 250 ? "var(--sev-nominal)" : undefined}
          />
          <div style={{ marginTop: "var(--s-3)" }}>
            <Micro style={{ marginBottom: "var(--s-2)" }}>Budget against Δt</Micro>
            <Meter
              value={lat?.total_ms != null && dt ? Number(lat.total_ms) / (dt * 1000) : 0}
              level="nominal"
              height={6}
            />
            <div className="t-data-s" style={{ color: "var(--paper-600)", marginTop: "var(--s-1)" }}>
              {lat?.total_ms != null && dt
                ? `${((Number(lat.total_ms) / (dt * 1000)) * 100).toFixed(1)}% of the ${dt}s window`
                : "—"}
            </div>
          </div>
        </PanelBody>

        <PanelHead title="Training provenance" />
        <PanelBody>
          <Field label="Corpus" value="CIC-IDS2018 + CTU-13" mono={false} />
          <Field label="Split policy" value="file-disjoint 80/20" mono={false} />
          <Field label="Scenario split" value="disjoint scenarios" mono={false} />
          <Field label="Seed" value="42" />
          <Field label="Train samples" value="35,993" />
          <Field label="Validation" value="3,400" />
          <Field label="Label leakage" value="zero (observable-only)" mono={false} />
        </PanelBody>
      </div>
    </div>
  );
}
