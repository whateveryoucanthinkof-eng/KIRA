/**
 * Build-time environment flags.
 *
 * Its own module rather than a const on `App` so layout components can read it
 * without importing the shell back (App → Header → App is a cycle, and an ESM
 * cycle on a module-init const hits the temporal dead zone).
 */

/** `.env.demo`, loaded by `npm run demo`. Routes `adapter.ts` to `mock.ts`. */
export const IS_DEMO = import.meta.env.VITE_DEMO_MODE === "true";
