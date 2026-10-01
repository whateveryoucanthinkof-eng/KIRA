# 33 — Porting the K.I.R.A. (Cyber World) frontend onto our real backend

Branch: `feat/kira-dashboard`. Started 2026-10-01. Progress log: newest entries at the bottom.
The panel-by-panel data map lives in `docs/DASHBOARD_INTEGRATION.md`.

## What was asked

- Use the Cyber World repo's `web_dashboard/` as the UI, **exactly** (K.I.R.A. branding, pages,
  3D network, timeline, sounds). Small additions are allowed for data our backend sends that Cyber
  World's UI has no place for (forecast band, sensor drops, mitigation status), styled the same way.
- Do **not** bring over Cyber World's backend, adapter or ML. Our `control_backend/` is the source
  of truth. Where our models can produce something the backend doesn't send yet, extend the backend.
  Where the models can't produce it, the UI says **"Under development"**.
- **Real dashboard** (`python run_dashboard.py`): Containerlab controls, real data, no attack
  buttons. **Demo dashboard** (`python run_dashboard.py --demo`): sample data plus in-app attack
  scenario buttons, only so people can see the UI. No fake shell commands (Cyber World's
  `scripts/demo_attack.sh` shadowed `nmap`/`hping3`; not ported).

## What the comparison found (before any change)

1. Both repos share the same API routes (`/api/status|site|topology|replay|command|mitigate`, `/ws`).
   The Cyber World frontend is a later evolution of ours (~26k vs ~12.7k lines).
2. Cyber World's backend is **older** than ours: its adapter turns the hand-written port-count rule
   layer on by default and lets it overwrite the model's risk/technique. Ours has the rules off and
   advisory only, plus a conformal forecast band, sensor-drop accounting and mitigation tracking.
3. Several Cyber World panels are fed only by mock data (their own comments say "Not yet on /ws"):
   raw flows, the 27-D state vector, alternate futures (A/B/C), the TGNE attention matrix, campaign,
   incidents, replay packet/byte volume.
4. Cyber World's UI also **invents** its forecast band on the client: `risk ± (1 - confidence) × 40`.
   Our backend sends a real one (`risk_lower`/`risk_upper`).
5. Cyber World's timeline hardcodes a 16 s horizon (8 × 2 s). Our contract is 5 × 30 s = 150 s
   (`cyberworld_v4/config.py`), and the served value comes from the checkpoints.
6. **Our backend cannot start today.** `telemetry_service.py` imports the model singleton at module
   load. With no Branch A/B/DeepOP checkpoints, and an encoder checkpoint on feature schema 1.0.0
   while the tree is 2.0.0, that import raises. `/api/status` also hardcodes `model_loaded=True`.

## Plan

1. Copy Cyber World's `web_dashboard/` over ours; build it in the `claude-dev` toolbox.
2. Backend: start without models (report `model_loaded=false` with the reason; telemetry, topology
   and lab controls still work; `start_ml` refuses with the reason).
3. Backend: send flows, the full 27-D state vector, DeepOP's top-3 continuations, TGNE attention,
   campaign (`correlation/`) and incidents over `/ws`; add volume and flows to `/api/replay`.
4. Frontend: real band, horizon from the model contract, "Under development" where the model can't
   produce a field, no attack buttons outside demo.
5. Demo: mock data reshaped to our contract; scenario buttons compiled into the demo build only.
6. Verify: backend tests, build both modes, screenshot both UIs against Cyber World's.

## Log

### 2026-10-01 — first pass complete

Commits on `feat/kira-dashboard`, in order:

1. **Baseline**: Cyber World's `web_dashboard/` copied verbatim, so every adaptation is a diff.
2. **Backend starts without models**: `get_model_adapter()` loads once under a lock, remembers why it
   failed, and retries on Start Inference. `/api/status` reports `model_loaded`, `model_error`, `contract`.
   `/api/replay` answers 503 with the reason.
3. **Backend sends the evidence**: `tactic_lane`, DeepOP top-3 `branches` (branch A == the served
   forecast, tested), the full 27-D `state_vector`, `flows` (with TCP flags), `campaign` and `incidents`
   (`correlation/` wired into serving for the first time), replay volume, real replay samples.
4. **Frontend wired**: contract-driven timeline, real conformal band, shared lanes (Credential Access
   added; Execution/Lateral marked under development), site-aware internal/zones instead of 18
   hardcoded `10.` checks, `UnderDev` marks, real Controls (no launch buttons; external monitor), demo
   scenarios compiled into the demo build only, live build stubs the fixtures, codenames dropped,
   demo mock reshaped to our contract/vocabulary with no rule lift and no scripted "predicted early".
5. **Launcher, topology fix, docs**: `run_dashboard.py --demo`; on-demand UI builds (npm or toolbox);
   topology ages on the sensor clock (replayed captures were evicted on arrival).

Found on the way and fixed:

- `/ws` status pushes reused `attack` for "the model is alerting" while REST used it for "armed":
  added `attack_armed`.
- Cyber World's demo displayed rule-lifted risk (`max(ml, rule)`); its UI invented the forecast band;
  its inner launcher bound `0.0.0.0` with no token (replaced by a delegate to the root launcher).
- Replay used a hardcoded 0.65 threshold; now the model's fitted one.

Verified:

- Backend tests: the four modules that failed collection now run; 29 new tests in
  `tests/test_dashboard_evidence.py`.
- Both UIs typecheck and build; demo strings are absent from the live bundle.
- Screenshots of all 12 pages: Cyber World reference vs our demo (same look, differences only where
  intended), and the real console running without models on a replayed capture.

Deferred, with a design in `docs/DASHBOARD_INTEGRATION.md`: TGNE attention read-out (cannot be
checked against real weights until the encoder is retrained).

Model gaps recorded for the ML review: `do_this_in_next_session_ml_review.md` §8.
