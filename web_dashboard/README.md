# cyberworld dashboard

The SOC console frontend: React 19 + Vite + Tailwind CSS v4, TypeScript throughout.

It renders the live host graph and ATT&CK-aware risk forecasts produced by the backend. It holds no
state of its own — everything arrives over `/api` and the `/ws` WebSocket from `control_backend/`.

## Running

From the repo root, the normal path builds and serves this automatically:

```bash
python run_dashboard.py --site local-default
```

To work on the frontend directly:

```bash
npm install
npm run dev      # HMR dev server; expects the backend on :8000
npm run build    # production bundle → dist/ (what the backend serves)
npm run demo     # mock data, no backend required (VITE_DEMO_MODE=true)
```

`npm` is the toolchain — `package-lock.json` is the lockfile.

## Pages

| Page | Shows |
|------|-------|
| Overview | Risk posture, key metrics, observed vs. predicted risk |
| Network | Discovery-driven host graph — nodes appear only when SPAN observes them |
| Predictions | Branch B forecast trajectory and ATT&CK technique attribution |
| Events | Command, model, attack, and telemetry log stream |
| Controls | Sensor/ML lifecycle, Lab Mode orchestration, mitigation actions |

## Layout

- `src/api/` — backend contract (`types.ts`), REST + WebSocket client (`adapter.ts`), demo fixtures (`mock.ts`)
- `src/pages/` — one component per page above
- `src/components/` — `layout/` shell pieces, `shared/` reusable UI
- `src/index.css` — global styles and the Tailwind v4 import

See `AGENTS.md` for the working notes, including the Figma Make plugins still present in
`vite.config.ts`.
