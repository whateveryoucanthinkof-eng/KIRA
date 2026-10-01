# cyberworld-dashboard (K.I.R.A.)

React + Vite + Tailwind CSS v4 frontend for the K.I.R.A. SOC console — the Cyber World UI, wired to this
repo's backend. It is served by the FastAPI backend in `control_backend/`, not deployed independently.

Rules for changes:

- Keep the Cyber World look. Small additions are fine when the backend sends something the UI had no
  place for; style them with the existing tokens.
- Never show a value no model produced. A field without a source renders `<UnderDev />`; the demo
  (`src/api/mock.ts`) may fill it. Panel-by-panel sources: `docs/DASHBOARD_INTEGRATION.md`.
- No attack-launching UI outside `IS_DEMO`.
- Never hardcode the temporal contract (`src/types/timeline.ts`) or the site's address space
  (`src/design/site.ts`).

## Running it

There is **no dev server already running** — start one yourself.

```bash
npm install           # first time
npm run dev           # dev server with HMR
npm run build         # production bundle → dist/, which the backend serves
npm run demo          # dev server in demo mode (VITE_DEMO_MODE=true → mock data)
npm run build:demo    # demo bundle → dist-demo/, which `run_dashboard.py --demo` serves
```

Normally you do not run any of these by hand: `python run_dashboard.py` (real) and
`python run_dashboard.py --demo` from the repo root build when the sources are newer than the build, then
serve it. The toolchain is **npm** — the lockfile is `package-lock.json`. In this workspace npm lives in
the `claude-dev` toolbox.

## Backend contract

The dashboard talks to the backend and has no data of its own:

- `src/api/types.ts` — the shared event/DTO contract. Mirrors `control_backend/schema.py`; change both together.
- `src/api/adapter.ts` — REST calls against `/api` plus the `/ws` WebSocket. Single point of contact with the backend.
- `src/api/normalize.ts` — the one place a /ws prediction is mapped for the UI; the demo stream uses it too.
- `src/api/mock.ts` — fixtures used only when `VITE_DEMO_MODE === 'true'`, dispatched from `adapter.ts`.
  Live builds resolve it to `mock.live.ts` (`vite.config.ts:liveBuildDropsDemoFixtures`).

Timestamps arrive as ISO-8601 with a `Z` suffix (`control_backend/schema.py:utc_now_iso`).

## Structure

- `src/main.tsx` — React entrypoint; imports `src/index.css`, mounts `src/App.tsx` into `#root`
- `src/App.tsx` — shell: routing between pages, WebSocket lifecycle, shared state
- `src/pages/` — one per sidebar entry (12: Overview … Controls)
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
