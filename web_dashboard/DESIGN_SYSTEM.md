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
| [src/components/Network3D.tsx](src/components/Network3D.tsx) | the 3D host graph (lazy chunk) |
| [src/pages/Stage.tsx](src/pages/Stage.tsx) | the Forecast Stage: storyline, forecast tree, trajectory, time travel |
| [src/types/timeline.ts](src/types/timeline.ts) | frames and the global time cursor |

---

## 0. The do-not list

These are enforced, not suggested. Violating one breaks the system.

- **`border-radius` is `0`.** Everywhere, without exception. The reset sets it globally.
- **No `box-shadow`, no `backdrop-filter`, no blur, no glow.** There is no glassmorphism here.
  One scoped exception: the 3D viewport (§7) is a separate instrument — a void with its own
  palette, bloom and glass — and none of it leaks into the DOM around it.
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

The greps cover the DOM. The viewport's palette lives in one object (`VIZ` in
`Network3D.tsx`) and is reviewed there, not grepped.

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
| **Stage provenance** | `stage_provenance` | Which of TGNE / Branch A / Branch B / DeepOP produced which part of the verdict. The Model view lays out the same four stages. |
| **Early warning** | `early_warning.lead_time_seconds` | The project's headline claim — seconds of lead before the milestone. |
| **Latency budget** | `latency.{telemetry_ms, inference_ms, total_ms}` | Proves the pipeline is real-time. Overview KPI strip and the Model view. |
| **Model card** | `model_meta` | `L`, `K`, `Δt`, `θ` read from the **loaded checkpoint** — never hardcoded, because the adapter adopts the contract off the weights. |
| **Measured telemetry** | `throughput`, `state.packet_count`, `state.active_flows` | Charted directly. Nothing on a plot is derived from the risk score. |
| **Kill chain** | `predicted_stage` + `ForecastPoint.predicted_stage` | How far the intrusion actually got, and where the model says it goes next. Lanes are `correlation/causal_edge_scorer.py:TACTIC_ORDER`. |
| **Forecast branches** | [src/types/forecast.ts](src/types/forecast.ts) | The rollout's three most likely continuations, with the path, volume and confidence of each. Demo fixture today: the backend forwards the rollout's risk and stage per step, not its competing continuations. |
| **TGNE attention** | [src/types/attention.ts](src/types/attention.ts) | Host × host weights from `bita/model/temporal_attention.py` (2 heads, 20 temporal neighbours), with Input × Gradient over the key's inputs by their schema names. Demo fixture today: the adapter extracts the embedding but not the layer's weights. |
| **Campaign graph** | `correlation/` dataclasses | Compacted alert nodes on kill-chain lanes, scored causal edges, root cause. Rendered from demo fixtures today — `correlation/` is not yet wired into `control_backend/`, and [src/types/campaign.ts](src/types/campaign.ts) is the contract to serve against when it is. |

### Demo mode

`.env.demo` sets `VITE_DEMO_MODE`, which routes `adapter.ts` to `src/api/mock.ts` — a
scripted Recon → Impact → recovery scenario on the real wire shapes, pre-seeded so the
console opens with full charts. Nothing in it uses `Math.random`: every value is a
deterministic function of the window index, so the scenario replays identically on every
recording. The flag is not surfaced in the UI; it only selects the data source.

**Record from a production build.** `npm run demo:prod` builds the demo into `dist-demo/` and
serves it on `:8444`. The dev server runs React in development mode and is several times slower
per tick — measured on an Intel Iris Plus laptop, Overview ran at ~3 fps under `npm run demo` and
~55 fps from the production build.

---

## 7. The 3D host graph

[src/components/Network3D.tsx](src/components/Network3D.tsx) — the Network view's default mode,
with a 2D toggle.

**The one place the do-not list is suspended.** The viewport is a void (`#030712`) with its own
palette (`VIZ`), glass materials, bloom and a vignette. It is a deliberate, *bounded* exception:
the scene is where the console shows the network as a living system, and it is judged as a
visual. Everything drawn over it — zone names, labels, hover card, HUD — is still DOM in the
console's own type, on dark tokens scoped to `.n3`, so it reads the same in INK and PAPER. The
glow never leaves the canvas.

| Element | Rule |
|---|---|
| Zones | Four stacked holographic planes — External, DMZ, Servers, Users — each a grid with a lit rim, tied by a vertical spine |
| Hosts | Glass (`MeshPhysicalMaterial`, transmission). Octahedron = server, sphere = workstation or client, icosahedron + counter-rotating wire shell = attacker. Emissive colour carries status: emerald nominal, amber at risk, crimson hostile |
| Links | Bezier arcs. Baseline traffic is a faint hairline; the attack path is heavy HDR crimson; a hop the campaign **forecasts** but has not observed is **dashed** blue |
| Packets | Instanced comets along each link, in the direction the flow was initiated, at a density taken from the window's bytes |
| Light | Hostile hosts light the glass around them from a fixed pool of three point lights |
| Labels | One DOM overlay projected each frame. Only core hosts carry labels — the background population stays quiet |
| Camera | Fitted to the viewport: the stack's extremes are projected and the closest distance that keeps them in frame is found, again on resize, full screen and Reset. Slow auto-orbit for recording; pauses on hover; drag, zoom |
| Full screen | `F` or the HUD button puts the viewport itself in full screen, with a short camera dolly. The inspector rail is off screen there, so a selected host gets a pinned card, and a live strip carries level, risk and technique |

**Driven by discovery, not a layout file.** Hosts appear when traffic is first seen, ease into
place, turn translucent when stale and are evicted on TTL; edges are observed flows.

**Performance rules — each one was a measured stall.**

- **Compile before the first frame.** The canvas mounts with its loop stopped; `compileAsync`
  builds the scene's programs in parallel, then the loop starts, and the boot overlay lifts after
  three quick frames. Compiling inside the first frame blocked the page for ~4s.
- **Keep program keys stable.** Light count and `transmission > 0` are part of every lit shader's
  key: lights come from a fixed pool and transmission never changes. Mounting a light per hostile
  host recompiled the scene mid-attack.
- **Flat transparent planes are `forceSinglePass`.** three.js otherwise splits a transparent
  `DoubleSide` material into back and front passes and re-selects its program for each, every
  frame — the layers alone were ~95% of program-selection time.
- **Rewrite link buffers in place.** drei's `setPoints` allocates a new GPU buffer per call; the
  links write into the one it already made. The reallocation caused recurring GC pauses.
- **The environment is memoised.** drei re-captures it whenever its children change identity.

**Operational notes.** three.js is its own lazily-loaded chunk (~1.1MB, ~300KB gzip), so no
other view pays for it. WebGL support is probed **once** and cached — browsers cap live contexts
at about 16 and evict the oldest. No WebGL, or a lost context, falls back to the 2D graph through
an error boundary. Reduced motion stops the orbit and the packets.

---

## 8. The Forecast Stage and time travel

[src/pages/Stage.tsx](src/pages/Stage.tsx) is the presenter's screen, second in the navigation.

| Part | Rule |
|---|---|
| Active attack | The attack in progress, by name, at the head of the storyline: technique, tactic, target host, live risk and level, its left rule in the verdict's severity. "No active attack — baseline traffic" when there is none. On this page it replaces the alert toasts, which would only cover the inspector |
| Attack storyline | Seven kill-chain stages on one rail, each pinned to a moment: first observed (`t −82s`, from the campaign's OBSERVED nodes), `NOW` (predicted_stage, in the verdict's severity, framed by a stepped tick), or expected (`+8s`, from the leading branch and the rollout). Observed rail is solid and warm, forecast dashed and cool |
| Forecast tree | Three branches fan from NOW against a 0–16s axis. Thickness is probability; colour is severity by kind — escalation critical, pivot warning, back-off muted. All dashed: all forecast. The most likely branch carries the travelling dash the attack path uses elsewhere. Hover gives the numbers: technique, confidence, ETA, path, targets, packets, volume, peak risk |
| Risk trajectory | The recorded history into the rollout, with the band, threshold and NOW rule |
| Attention | The TGNE attention matrix: rows are the host attending, columns its temporal neighbours, each row summing to 1. Warm bone scales with α; attention on a hostile edge turns critical. The legend is stepped, not a gradient. Hover previews, click locks (arrows move, Esc releases), and the side panel walks the logit from baseline through each input's Input × Gradient contribution — a waterfall that is the arithmetic |
| Inspector | Verdict, rule override, the 27-D state vector with Input × Gradient — the Investigation inspector, reused. Open by default; the Inspector control hides it. On short screens it tightens rather than dropping anything but the secondary facts line |
| Emulation | Three controls in the stage head: C2 surge, lateral spread, contain. In demo mode they steer the scenario to those stages of `workloads/attacker_scenario.py`; against a live lab the first starts the attacker and containment isolates the target |

**Sound** is off unless the viewer turns it on (the Attention view's toggle): synthesized blips on cell hover and a two-tone chime on an escalating alert, no audio files ([src/design/sound.ts](src/design/sound.ts)).

**Time travel.** Every window is kept as a *frame* — verdict, rollout, branches, campaign,
topology, evidence — and the charts' history is derived from the frames, so there is one source.
A single cursor selects what **every view** renders: live, a recorded frame, or a point in the
horizon. The dock at the bottom of the Stage drives it (drag, ←/→, −10s, Play/Pause, Live, Peak);
the header shows a `Replay · t −24s | Live` control on every page while the console is not showing
the present. Playback runs at twice the stream rate so it catches up with live. Frames carry a
client sequence number because steering the scenario repeats window ids. In demo mode the 90
windows before the stream starts are backfilled as full frames from the same payload builder the
stream uses, so scrubbing back past page load shows real state.

---

## 9. ATT&CK and the incident war room

**ATT&CK** (`07`). Cards are the model's vocabulary only. Observed cards are solid and warm with
their risk; the technique in progress carries the verdict's severity, a ticking frame and *NOW*;
forecast cards are dashed and hatched. Each card carries its activation probability across the
+16s horizon as a sparkline, from the forecast branches and the rollout. The vector path threads
the observed techniques in the order they happened — numbered, drawn beneath the cards so it
never crosses their text — and continues dashed into the forecast. A card opens an inspector in
the free space under the matrix, tethered to it: confidence, affected hosts, and the on-path flow
records as evidence. The sensor records flows, not payloads, so no payload is ever shown. Forecast
branches name only techniques in this vocabulary, so every forecast has a card.

**Incidents** (`09`). The queue on the left, one incident worked on the right: header, a
containment dock, and tabs for the blast radius, the lead-time trail and the activity log.
Isolate and Block call `/api/mitigate` with `ISOLATE_HOST` and `BLOCK_IP` — the actions
`control_backend/telemetry_service.py` accepts — so containment here is containment everywhere.
Export writes a real Markdown report with the flow evidence. Analyst edits (status, assignee,
notes) sit on top of the stream and expire when an incident restarts — detected by its alert count
falling — so an earlier "contained" never sticks to a new attack on the same host.

**Campaign** (`06`). The correlation campaign read as a dossier: codename, status, the actor as
an internal cluster (`UNC-CW-NN`, unattributed — flow records carry no actor indicators, so no
named group is ever claimed), first-observed time, the incidents it explains (bound by campaign
id, or by host while open), velocity in stages and hosts per minute, source, and the TTP sequence.
The evolution graph is laid out at its panel's size — kill-chain lanes by host rows, stage cards
marked done, active, forecast with probability, or contained. Asset cards carry a compromise or
risk gauge and highlight their host's stages on hover. Neutralise isolates every reached host and
blocks the source through `/api/mitigate`, one host at a time as a containment pulse runs across
the graph, then marks the bound incidents contained.

**Replay** (`08`). Offline analysis played back as a DVR. The intake is a scanner bed with a
stepped scan line and registration marks; a sample card or a dropped file runs the parse, shown
as a segmented bar over the four real steps (read, flows, 2.0 s windows, TGNE → Branch A/B →
DeepOP). The DVR is one time axis: keyframe pins derived from the report (first window of each
technique, first alert, peak, clear), a severity strip carrying the playhead, then model risk
(warm, filled) over packet rate (neutral line), with what has not been played faded rather than
hidden. Transport keys are drawn glyphs, not an icon set. The flow stream shows each window's
records as the playhead crosses their first packet: endpoints, flags, packets, bytes, duration.
Records carry no payload, so none is shown. The rail reads the current window: risk, technique,
volume, the rollout (cool, hatched) and attribution. Space, arrows and Shift-arrows drive it.

**Overview** (`01`). Mission control. A command band answers three questions left to right:
what is happening (the verdict, the attack by name, target, risk against θ), what happens next
(the lead time, and the likeliest branch hatched as forecast) and what needs doing (the top of
the incident queue, one button into the war room). Below it the storyline, one risk chart that
runs from the recorded windows through NOW into the rollout and its band, the incident queue,
and a ticker of seven measurements with their last 30 windows. A classified technique under
θ reads "Under watch", not "Active attack". The console, attribution and host graph moved to
Events and Controls, the Stage inspector and Investigation, and Network.

**Controls** (`12`). A service strip keeps every lab command, gated and confirmed as before.
Adversary emulation lists the real stages of `workloads/attacker_scenario.py`; launching one
starts a run tracker that shows what the model reads, when it first alerts and the lead time.
Response playbooks are sequences of the backend's real `/api/mitigate` actions with a step log;
the containment playbook ends by waiting for the model's risk to fall under θ. Targets follow
the live window and can be overridden. The console stays underneath.

**Events** (`10`). The log made searchable: a brushable volume histogram, facets with counts
of what choosing them would add, a query bar taking free text and `host:` `sev:` `tech:` `cat:`
tokens that become removable chips, rows that open in place to the raw record with its
`key=value` pairs laid out, and a follow mode that holds the rows still while scrolled. Alerts
raised by the forecast while the observed level is nominal are filed as `FORECAST`.

---

## 10. Adding a surface

1. Put it in a `.sheet` grid; give each `Panel` `flush`.
2. Title it with `PanelHead`. Machine qualifiers go in `note`, controls in `aside`.
3. Numbers go through `Readout` (KPI) or `Data` (inline). Never style a number ad hoc.
4. Severity goes through `sev()` / `sevColor()` — never a literal hex.
5. Charts import from `charts.tsx`. If a risk value is plotted, draw `<ThresholdLine>`.
6. Empty is a state, not an accident: render `<Empty>` with a hint naming what would fill it.
7. Run the greps in §0 before you call it done.
