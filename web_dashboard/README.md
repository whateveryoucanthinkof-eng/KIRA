# K.I.R.A. dashboard

The SOC console frontend: React 19 + Vite + Tailwind CSS v4 + three.js, TypeScript throughout. The UI
is the Cyber World (K.I.R.A.) console, wired to this repo's backend in `control_backend/`.

## Running

From the repo root:

```bash
python run_dashboard.py          # the real console (backend + models + UI) on :8000 — use this
python run_dashboard.py --demo   # the demo dashboard: the UI on sample data, no backend, on :8443
```

Both build this folder on demand (`dist/`, `dist-demo/`). To work on the frontend directly:

```bash
npm ci
npm run dev          # HMR dev server; expects the backend on :8000
npm run demo         # HMR dev server on sample data (VITE_DEMO_MODE=true)
npm run build        # the console bundle  -> dist/       (what the backend serves)
npm run build:demo   # the demo bundle     -> dist-demo/  (what --demo serves)
```

`npm` is the toolchain — `package-lock.json` is the lockfile. Node 20+.

## Real console vs demo

- **Real**: every number comes from the backend. Fields no model produces yet render as
  *Under development* (`src/components/UnderDev.tsx`). There are no attack buttons; Controls arms
  external-traffic monitoring and tracks what the model catches.
- **Demo** (`IS_DEMO`, `src/env.ts`): `src/api/adapter.ts` routes every call to `src/api/mock.ts`, which
  emits the backend's exact wire format. Attack-scenario buttons exist only here. Live builds resolve
  `mock.ts` to the inert `mock.live.ts`, so no sample data ships in the real bundle.

## Pages

| Page | Shows |
|------|-------|
| Overview | Verdict, early warning, kill chain, observed vs forecast risk, incidents, sensor ticker |
| Forecast Stage | Storyline, DeepOP's alternate futures, attention (under development), trajectory, 3D, 27-D inspector, time travel |
| Investigation | One host's trajectory, forecast onset, state vector and flow evidence |
| Network | Discovery-driven host graph (3D/2D), inventory, flows |
| Predictions | Forecast table, classification, stage distribution, attribution |
| Campaign | Correlated campaign graph (heuristic `correlation/`), assets, neutralisation |
| ATT&CK | Coverage matrix of the techniques our heads can emit |
| Replay | Offline analysis of a capture or flow CSV, with a DVR |
| Incidents | Alert-grouped incidents with an analyst workflow |
| Events | Command, model, attack and telemetry log stream |
| Model | Model card and contract |
| Controls | Containerlab lifecycle, sensor, inference, external monitoring, response playbooks |

## Layout

- `src/api/` — wire contract (`types.ts`), REST + WebSocket client (`adapter.ts`), the one prediction
  normaliser both streams use (`normalize.ts`), demo fixtures (`mock.ts`) and their live stub
- `src/types/` — view models; `timeline.ts` holds the temporal contract taken from `/api/status`
- `src/design/` — tokens, primitives, charts, `lanes.ts` (kill-chain lanes), `site.ts` (CIDRs/zones from `/api/site`)
- `src/pages/`, `src/components/` — one component per page, shared pieces

See `docs/DASHBOARD_INTEGRATION.md` for where every panel's data comes from.
