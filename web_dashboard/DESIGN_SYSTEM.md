# Field Manual — CyberWorld SOC Design System

The console is a **ruled technical document**, not a deck of floating cards.
Depth comes from surface value and rule weight. Never from shadow.

Everything here is implemented in three files:

| File | Holds |
|---|---|
| [src/design/tokens.css](src/design/tokens.css) | fonts, both themes, reset, type scale, motion |
| [src/design/primitives.tsx](src/design/primitives.tsx) | `Panel` `Chip` `Square` `Readout` `Meter` `Btn` `Field` `Sheet` … |
| [src/design/charts.tsx](src/design/charts.tsx) | axis/grid/tooltip/legend config for every plot |
| [src/design/motion.tsx](src/design/motion.tsx) | `Num` (eased numeral), `useEasedNumber`, `Enter` |
| [src/components/KillChain.tsx](src/components/KillChain.tsx) | the ATT&CK stage timeline |

---

## 0. The do-not list

These are enforced, not suggested. Violating one breaks the system.

- **`border-radius` is `0`.** Everywhere, without exception. The reset sets it globally.
- **No `box-shadow`, no `backdrop-filter`, no blur, no glow.** There is no glassmorphism here.
- **No gradient as decoration.** Two sanctioned uses, both data-ink: a flat-to-transparent chart
  area fill at ≤14% opacity, and the 45° forecast hatch (`repeating-linear-gradient` in
  `.at-cell-fc`, `<pattern>` in `KillChain`/`Campaign`). The hatch *encodes provenance* — it is
  how forecast reads as forecast in greyscale. Nothing else may interpolate colour.
- **No icon library**, no glyph sets, no decorative SVG. Navigation is numbered; status is a square.
- **No accent hue.** There is no brand cyan, no violet, no neon. Colour that is not encoding risk is a bug.
- **Nothing is centred** except empty states. The sheet is left-aligned and rule-anchored.

Check them mechanically:

```bash
grep -rn  "border-radius\|borderRadius"                       src/   # only 0, in the reset
grep -rniE "box-shadow|backdrop-filter|blur\("                 src/   # no hits
grep -rniE "\bInter\b|Roboto|SF Pro"                           src/   # no hits
grep -rn  "radial-gradient"                                  src/   # no hits
grep -rn  "linear-gradient"                                  src/   # only the forecast hatch
grep -rniE "00F0FF|A855F7|EF4444|10B981|purple|violet|cyan"    src/   # no hits
```

---

## 1. Colour

Colour is **data-ink**. If it does not encode risk, it is monochrome.

### Surfaces — INK (default) / PAPER

The base is a *warm* near-black, not blue-black. Blue-black plus a bright accent is
the generic dashboard look this system exists to avoid.

| Token | INK | PAPER | Use |
|---|---|---|---|
| `--ink-000` | `#0A0A09` | `#F4F1E8` | page canvas, log ground, input fills |
| `--ink-050` | `#121211` | `#FBF9F4` | panel fill |
| `--ink-100` | `#191918` | `#FFFFFF` | sticky table header, tooltip |
| `--ink-200` | `#232322` | `#EAE6DB` | row hover, chip ground |
| `--rule-hair` | `#2C2C2A` | `#DDD8CA` | the default divider |
| `--rule-hard` | `#46453F` | `#A8A192` | structural edge, focus, threshold |

### Text

| Token | INK | PAPER | Use |
|---|---|---|---|
| `--paper-000` | `#F2EFE6` | `#17161A` | primary, hero numerals |
| `--paper-400` | `#A8A49A` | `#57544C` | secondary, table cells |
| `--paper-600` | `#6E6B63` | `#86827A` | micro-labels, axis ticks, units |

### Severity — the only sanctioned colour

Four steps, so `ELEVATED` is visually distinct from `WARNING`. Desaturated to sit on
a warm ground; `#EF4444` / `#10B981` read as neon here and are deliberately not used.

| Token | INK | Maps to |
|---|---|---|
| `--sev-nominal` | `#4F9D69` | `alert_level: NOMINAL`, `threatLevel: low`, service running |
| `--sev-warning` | `#C89331` | `WARNING` / `medium` / degraded |
| `--sev-elevated` | `#C4643A` | `ELEVATED` / `high` |
| `--sev-critical` | `#C4362F` | `CRITICAL` / compromised / saturated |

`sev()` in `primitives.tsx` normalises all four backend vocabularies
(`alert_level`, `threatLevel`, `EventSeverity`, service/topology states) onto this scale.
`sevFromRisk(risk, threshold)` mirrors the backend's own banding
(`control_backend/model_adapter.py:465`).

### Observed vs forecast — temperature, not hue

The spec this system replaced wanted violet for "predictive". Instead:

```
--signal-observed  #D8CFAE   warm   — measured, solid stroke
--signal-forecast  #7D8CA1   cool   — predicted, dashed stroke
```

**Observed data is warm and solid. Forecast data is cool and dashed.** The temperature
split plus the dash pattern carries provenance, so no third accent hue is needed — and
the distinction survives greyscale, which a hue pair would not.

---

## 2. Typography

Three voices, each with one job.

| Face | Role | Rule |
|---|---|---|
| **Instrument Serif** | display | Page titles and **one** hero numeral per view. Nothing else is allowed to be large. |
| **Space Grotesk** | UI | Labels, buttons, nav, prose — anything a human wrote. |
| **JetBrains Mono** | data | IPs, ports, risk values, feature names, technique IDs, timestamps, log lines — anything a machine produced. |

> **The rule that keeps it coherent:** machine output is mono, human language is Space Grotesk,
> and only Instrument Serif gets to be big.

### Scale — eight steps. If a size is not here, it does not exist.

| Class | Face / size / line / tracking |
|---|---|
| `.t-display-xl` | Instrument Serif 56 / 1.0 / −0.02em — hero numeral |
| `.t-display-l` | Instrument Serif 32 / 1.1 / −0.01em |
| `.t-display-m` | Instrument Serif 22 / 1.15 / −0.01em — page title, technique name |
| `.t-micro` | Space Grotesk 500 · 10 / 1.2 / 0.14em · UPPERCASE — panel titles, field labels |
| `.t-label` | Space Grotesk 500 · 12 / 1.3 — buttons, nav, chips |
| `.t-body` | Space Grotesk 400 · 13 / 1.45 — prose |
| `.t-data-l` | JetBrains Mono 500 · 18 / 1.2 — KPI values |
| `.t-data` | JetBrains Mono 400 · 12 / 1.4 — table cells, inspector values |
| `.t-data-s` | JetBrains Mono 400 · 10.5 / 1.35 — dense tables, logs, axis ticks |

---

## 3. Structure

```
--radius   0                      everything is a rectangle
--s-1..12  4 8 12 16 24 32 48     the only spacing values
--panel-pad     16px
--panel-head-h  32px
--rail-w        216px   --header-h  56px
--hair   1px solid var(--rule-hair)
--hard   1px solid var(--rule-hard)
```

### The sheet

Panels are cells in a ruled document. A `.sheet` is a grid with a **1px gap over a
rule-coloured ground** — the gap *is* the hairline:

```css
.sheet { display: grid; gap: 1px; background: var(--rule-hair); border: 1px solid var(--rule-hair); }
```

Panels inside a sheet take `flush` (no borders of their own). Edges collapse perfectly,
never double up, and survive responsive reflow — which per-panel borders cannot do.
Stacked sheets drop the seam between them (`.ov > .sheet + .sheet { border-top: none }`).

---

## 4. Components

| Component | Rules |
|---|---|
| **`Panel`** | `--ink-050` fill, hairline border, zero radius. `flush` for sheet cells, `clip` for tables/logs/SVG, `spine` for a severity edge. |
| **`PanelHead`** | 32px bar, `.t-micro` title, optional mono `note`, right-aligned `aside` slot. |
| **`Chip`** | Zero radius, hairline box, **severity as text colour + 3px left rule**, `--ink-200` ground. `strong` inverts to a solid severity block — one per screen, for the worst state only. |
| **`Square`** | 6px filled **square**, not a circle. `live` steps `opacity 1 → 0.22` over 2s on `steps(1, end)` so it **ticks**. It never breathes and never glows. |
| **`Readout`** | `.t-micro` label → value (`data` or `hero`) → unit in `--paper-600` → sub-line. The primary number pattern. |
| **`Meter`** | Square-ended bar on a hairline track. Fill transitions on `steps(12, end)` — stepped, not smooth. |
| **`BarRow`** | Attribution row: mono label, meter, right-aligned percentage, group note. |
| **`Field`** | Key/value on a hairline. Micro label left, mono value right. |
| **`Btn`** | Rectangle, `--hard` border, transparent fill. **Hover inverts** to bone fill / ink text. `danger` adds a critical left spine; `primary` is bone-filled at rest (one per group). |
| **`Sheet`** | Modal: hard-edged panel on a flat `rgba(10,10,9,0.82)` scrim. No blur, no shadow. |
| **`Empty`** | Standby stated plainly — a micro line plus an optional hint. Never apologetic, never an illustration. |
| **Nav item** | `01 Overview`. Active state is a **full inversion** — unmistakable without an icon or a glow. |
| **Table** (`.tbl`) | Sticky `--ink-100` header on a hard rule, hairline rows, `--ink-200` hover, 10.5px mono cells. Severity rides a **3px left rule on the row**, never a tinted background — tints destroy legibility at that size. |

### Focus

One treatment everywhere: `outline: 2px solid var(--paper-000); outline-offset: 2px`.
No glow, no ring colour per state.

### Motion — two registers

The split is the point. **Discrete machine state is stepped; continuous quantities are
eased.** A service is running or it is not, so its indicator ticks. A risk score moving
0.871 → 0.884 is an interpolation of a real number, so interpolating it on screen is
honest as well as smoother.

| Register | What | Timing |
|---|---|---|
| **Stepped** | live square, hover/active inversion, log line arrival | `steps()` — 2s/1 step live, 90ms/2 steps hover |
| **Eased** | hero numerals (`<Num>`), meters, kill-chain fill, page entrance, forecast rollout | `--ease` `cubic-bezier(.22,.61,.36,1)` at `--dur-value` 280ms / `--dur-fill` 400ms |

Nothing breathes, pulses or glows in either register. `<Num>` (`design/motion.tsx`)
re-targets mid-flight rather than restarting, so a stream arriving faster than the
animation bends toward the new value instead of stuttering.

**Rolling history charts are deliberately not animated.** Re-drawing a 90-point path
every 830ms tick reads as lag, not motion. Only the discrete 8-point forecast rollout —
which changes shape meaningfully each window — gets a 200ms morph.

Everything is disabled under `prefers-reduced-motion`, which also clamps all transitions
to 1ms.

---

## 5. Charts

Configured once in `charts.tsx`. Do not restyle axes per chart.

- Horizontal gridlines only, `--rule-hair`. Vertical grid is noise on a time axis.
- No axis lines, no tick marks. The grid already implies the frame.
- Ticks: `.t-data-s` in `--paper-600`.
- **Observed**: solid 1.5px `--signal-observed`, area fill `--fill-observed` (10%).
- **Forecast**: 1.5px `--signal-forecast`, `strokeDasharray="3 3"`, band `--fill-forecast` (12%), no stroke.
- **`<ThresholdLine>` on every risk plot** — the model's own fitted `prediction.threshold`,
  dashed `--rule-hard`, labelled `THRESHOLD 0.65` in mono. A risk curve without it is unreadable.
- **`<NowLine>`** marks the seam between measured history and the rollout.
- **Tooltip**: hard-edged `--ink-100` panel, `--hard` border, mono, zero radius, zero shadow.
- `isAnimationActive={false}` on every series — a live console must not re-animate each tick.

---

## 6. What the console surfaces

The design exists to make these legible. Each is already on the wire from
`control_backend/schema.py` and was previously unrendered.

| Surfaced | Wire field | Why it matters |
|---|---|---|
| **Risk provenance** | `ml_risk` / `rule_risk` / `rules_applied` | A deterministic SOC rule layer can overwrite the model's risk and technique (`model_adapter.py:585`). Showing the blended number alone misrepresents what produced it. |
| **Stage provenance** | `stage_provenance` | Which of TGNE / Branch A / Branch B / DeepOP produced which part of the verdict. |
| **Early warning** | `early_warning.lead_time_seconds` | The project's headline claim — seconds of lead before the milestone. |
| **Latency budget** | `latency.{telemetry_ms, inference_ms, total_ms}` | Proves the pipeline is real-time. |
| **Model card** | `model_meta` | `L`, `K`, `Δt`, `θ` read from the **loaded checkpoint** — never hardcoded, because the adapter adopts the contract off the weights. |
| **Measured telemetry** | `throughput`, `state.packet_count`, `state.active_flows` | Charted directly. Nothing on a plot is derived from the risk score. |
| **Kill chain** | `predicted_stage` + `ForecastPoint.predicted_stage` | How far the intrusion actually got, and where the model says it goes next. Lanes are `correlation/causal_edge_scorer.py:TACTIC_ORDER`. |
| **Campaign graph** | `correlation/` dataclasses | Compacted alert nodes on kill-chain lanes, scored causal edges, root cause. Rendered from demo fixtures today — `correlation/` is not yet wired into `control_backend/`, and [src/types/campaign.ts](src/types/campaign.ts) is the contract to serve against when it is. |

### Demo mode

`.env.demo` sets `VITE_DEMO_MODE`, which routes `adapter.ts` to `src/api/mock.ts` — a
scripted Recon → Impact → recovery scenario on the real wire shapes, pre-seeded so the
console opens with full charts. Nothing in it uses `Math.random`: every value is a
deterministic function of the window index, so the scenario replays identically on every
recording. The header carries a `SIMULATED FEED` chip whenever the flag is set.

---

## 7. Adding a surface

1. Put it in a `.sheet` grid; give each `Panel` `flush`.
2. Title it with `PanelHead`. Machine qualifiers go in `note`, controls in `aside`.
3. Numbers go through `Readout` (KPI) or `Data` (inline). Never style a number ad hoc.
4. Severity goes through `sev()` / `sevColor()` — never a literal hex.
5. Charts import from `charts.tsx`. If a risk value is plotted, draw `<ThresholdLine>`.
6. Empty is a state, not an accident: render `<Empty>` with a hint naming what would fill it.
7. Run the greps in §0 before you call it done.
