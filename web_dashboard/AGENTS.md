# cyberworld-dashboard

React + Vite + Tailwind CSS v4 frontend for the cyberworld SOC console. It is served by the FastAPI
backend in `control_backend/`, not deployed independently.

## Running it

There is **no dev server already running** — start one yourself.

```bash
npm install           # first time
npm run dev           # dev server with HMR
npm run build         # production bundle → dist/, which the backend serves
npm run demo          # dev server in demo mode (VITE_DEMO_MODE=true → mock data)
```

Normally you do not run any of these by hand: `python run_dashboard.py` from the repo root builds
`dist/` when it is missing and then serves it. `scripts/run_control_panel.sh` does the same for the
lab workflow. The toolchain is **npm** — the lockfile is `package-lock.json`.

## Backend contract

The dashboard talks to the backend and has no data of its own:

- `src/api/types.ts` — the shared event/DTO contract. Mirrors `control_backend/schema.py`; change both together.
- `src/api/adapter.ts` — REST calls against `/api` plus the `/ws` WebSocket. Single point of contact with the backend.
- `src/api/mock.ts` — fixtures used only when `VITE_DEMO_MODE === 'true'`, dispatched from `adapter.ts`.

Timestamps arrive as ISO-8601 with a `Z` suffix (`control_backend/schema.py:utc_now_iso`).

## Structure

- `src/main.tsx` — React entrypoint; imports `src/index.css`, mounts `src/App.tsx` into `#root`
- `src/App.tsx` — shell: routing between pages, WebSocket lifecycle, shared state
- `src/pages/` — `Overview`, `Network`, `Predictions`, `Events`, `Controls` (one per sidebar entry)
- `src/components/layout/` — `Header`, `Sidebar`
- `src/components/shared/` — `MetricCard`, `StatusBadge`
- `src/index.css` — global CSS and the Tailwind v4 import
- `index.html` — Vite shell. The `<title>` and head/body slots are filled by the
  `figmaSiteConfiguration` plugin in `vite.config.ts`, so set the title there, not in the HTML.

## Notes

- Tailwind v4 via `@tailwindcss/vite`; no `tailwind.config` or PostCSS config is needed. Theme
  customization belongs in `src/index.css`.
- `@` is aliased to `src/`.
- `vite.config.ts` still carries the Figma Make scaffolding plugins this project was generated from.
  `figmaSiteConfiguration` runs on build and owns the document head; the others are `apply: 'serve'`
  (dev-only). Leave them unless you are deliberately untangling the build.
- The app is dark-mode only — `<html class="dark">`.
- Export components as default exports.
