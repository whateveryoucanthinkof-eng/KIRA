import { useState } from "react";
import type { PredictionResult } from "../api/types";
import type { FlowRecord, StateDim } from "../types/evidence";
import { Chip, Data, Empty, Micro, PanelBody, PanelHead, sevColor, sevFromRisk } from "../design/primitives";
import { OBSERVED } from "../design/charts";
import { Num } from "../design/motion";

/**
 * Model output inspector.
 *
 * Two questions an analyst asks of any verdict, answered from the raw numbers:
 *
 *   1. Did the model produce this, or did a rule?
 *      control_backend/model_adapter.py runs a deterministic SOC rule layer
 *      after Branch A. When it fires it can lift the displayed risk well above
 *      what the model said. The override strip places ML, rule and displayed
 *      risk on one axis against the threshold, so the size of the override is
 *      visible rather than implied.
 *
 *   2. What in the input drove it?
 *      All 27 dims of the state vector — 12 TGNE latent, 15 host attributes —
 *      with their share of Input x Gradient attribution. The backend computes
 *      all 27 and forwards the top 8; this shows the full vector.
 */

const ATTR_START = 12;

function fmtValue(d: StateDim): string {
  // Latent dims are signed and unbounded; attributes are clipped to [0, 1].
  if (d.index < ATTR_START) return `${d.value >= 0 ? "+" : "−"}${Math.abs(d.value).toFixed(3)}`;
  return d.value.toFixed(3);
}

/** ML, rule and displayed risk on one 0–1 axis, with the threshold marked. */
function OverrideStrip({ ml, rule, shown, threshold }: { ml: number | null; rule: number | null; shown: number; threshold: number }) {
  const pct = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  const lo = ml != null ? Math.min(ml, shown) : shown;
  const hi = ml != null ? Math.max(ml, shown) : shown;

  return (
    <div className="ovr">
      <div className="ovr-track">
        {/* The span the rule layer added on top of the model. */}
        {ml != null && hi - lo > 0.005 && <div className="ovr-span" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />}
        <div className="ovr-threshold" style={{ left: pct(threshold) }} />
        {ml != null && <div className="ovr-mark is-ml" style={{ left: pct(ml) }} title={`ML ${ml.toFixed(3)}`} />}
        {rule != null && <div className="ovr-mark is-rule" style={{ left: pct(rule) }} title={`Rule ${rule.toFixed(3)}`} />}
        <div className="ovr-mark is-shown" style={{ left: pct(shown), background: sevColor(sevFromRisk(shown, threshold)) }} title={`Displayed ${shown.toFixed(3)}`} />
      </div>
      <div className="ovr-scale">
        <Data size="s" color="var(--paper-600)">
          0.00
        </Data>
        <Data size="s" color="var(--paper-600)" style={{ position: "absolute", left: pct(threshold), transform: "translateX(-50%)" }}>
          θ {threshold.toFixed(2)}
        </Data>
        <Data size="s" color="var(--paper-600)">
          1.00
        </Data>
      </div>
    </div>
  );
}

export default function StateInspector({
  vector,
  prediction,
  flows,
  latestWindow,
}: {
  vector: StateDim[] | null;
  prediction: PredictionResult | null;
  flows: FlowRecord[];
  latestWindow: number | null;
}) {
  const [sortBy, setSortBy] = useState<"index" | "weight">("index");

  const threshold = Number(prediction?.threshold ?? 0.65);
  const shown = Number(prediction?.risk ?? 0);
  const ml = prediction?.ml_risk != null ? Number(prediction.ml_risk) : null;
  const rule = prediction?.rule_risk != null ? Number(prediction.rule_risk) : null;
  const applied = Boolean(prediction?.rules_applied);
  const delta = ml != null ? shown - ml : null;

  const external = flows.filter((f) => f.window === latestWindow && f.direction !== "internal").length;

  const dims = vector ? (sortBy === "weight" ? [...vector].sort((a, b) => b.attribution - a.attribution) : vector) : [];
  const maxW = dims.length ? Math.max(...dims.map((d) => d.attribution)) : 1;
  const topSet = new Set(vector ? [...vector].sort((a, b) => b.attribution - a.attribution).slice(0, 3).map((d) => d.index) : []);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
      {/* ── Rule override ───────────────────────────────────────────── */}
      <PanelHead
        title="Rule override"
        aside={
          prediction ? (
            <Chip level={applied ? "warning" : "nominal"}>{applied ? "Rule applied" : "Model output"}</Chip>
          ) : undefined
        }
      />
      <PanelBody>
        {prediction ? (
          <>
            <div className="ovr-readouts">
              <div>
                <Micro>ML · Branch A</Micro>
                <Data size="l" color="var(--paper-000)">
                  {ml != null ? <Num value={ml} digits={3} /> : "—"}
                </Data>
              </div>
              <div>
                <Micro>Rule layer</Micro>
                <Data size="l" color={applied ? "var(--sev-warning)" : "var(--paper-600)"}>
                  {rule != null ? <Num value={rule} digits={3} /> : "—"}
                </Data>
              </div>
              <div>
                <Micro>Displayed</Micro>
                <Data size="l" color={sevColor(sevFromRisk(shown, threshold))}>
                  <Num value={shown} digits={3} />
                </Data>
              </div>
            </div>

            <OverrideStrip ml={ml} rule={rule} shown={shown} threshold={threshold} />

            <div className="ovr-legend">
              <span>
                <i className="ovr-key is-ml" /> ML
              </span>
              <span>
                <i className="ovr-key is-rule" /> rule
              </span>
              <span>
                <i className="ovr-key is-shown" /> displayed
              </span>
              {delta != null && (
                <Data size="s" color={delta > 0.005 ? "var(--sev-warning)" : "var(--paper-600)"} style={{ marginLeft: "auto" }}>
                  override {delta >= 0 ? "+" : "−"}
                  {Math.abs(delta).toFixed(3)}
                </Data>
              )}
            </div>

            <div className="ovr-facts">
              <div>
                <Micro>External flows</Micro>
                <Data size="s">{latestWindow != null ? `${external} in window #${latestWindow}` : "—"}</Data>
              </div>
              <div>
                <Micro>Technique source</Micro>
                <Data size="s">Branch A · technique head</Data>
              </div>
            </div>
          </>
        ) : (
          <Empty>No verdict yet</Empty>
        )}
      </PanelBody>

      {/* ── State vector ────────────────────────────────────────────── */}
      <PanelHead
        title="State vector"
        note="27-D"
        aside={
          <span className="seg-group">
            {(["index", "weight"] as const).map((k) => (
              <button key={k} className="seg" aria-pressed={sortBy === k} onClick={() => setSortBy(k)}>
                {k}
              </button>
            ))}
          </span>
        }
      />

      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {dims.length ? (
          <div className="vec">
            {dims.map((d, i) => {
              const split = sortBy === "index" && d.index === ATTR_START;
              const lead = topSet.has(d.index);
              return (
                <div key={d.feature}>
                  {sortBy === "index" && i === 0 && <div className="vec-group">TGNE latent · h_emb 0–11</div>}
                  {split && <div className="vec-group">Host attributes · clipped [0, 1]</div>}
                  <div className={lead ? "vec-row is-lead" : "vec-row"}>
                    <span className="vec-idx">{String(d.index).padStart(2, "0")}</span>
                    <span className="vec-name" title={`${d.feature} · ${d.group}`}>
                      {d.feature}
                    </span>
                    <span className="vec-val">{fmtValue(d)}</span>
                    <span className="vec-bar">
                      <span style={{ width: `${(d.attribution / maxW) * 100}%`, background: lead ? "var(--paper-000)" : OBSERVED }} />
                    </span>
                    <span className="vec-pct">{(d.attribution * 100).toFixed(1)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <Empty hint="All 27 dimensions and their attributions arrive with each scored window once inference is live.">No state vector yet</Empty>
        )}
      </div>

      <div className="flow-foot">
        <Micro>Input × Gradient · last timestep</Micro>
        {vector && (
          <Data size="s" color="var(--paper-600)" style={{ marginLeft: "auto" }}>
            Σ {vector.reduce((s, d) => s + d.attribution, 0).toFixed(3)}
          </Data>
        )}
      </div>
    </div>
  );
}
