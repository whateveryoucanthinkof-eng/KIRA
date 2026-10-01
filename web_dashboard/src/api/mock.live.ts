/**
 * What `src/api/mock.ts` resolves to in a live (non-demo) build.
 *
 * vite.config.ts swaps the demo fixtures for this file unless the build mode
 * is `demo`, so the real console's bundle carries no sample data at all —
 * not even as unreachable code. Every export mirrors mock.ts's signature and
 * is inert: the adapter only calls these behind IS_DEMO, which is false here.
 */

import type { SystemStatus, Topology, SiteInfo, MitigationPayload, WSHandlers } from "./types";
import type { ReplayReport, ReplaySample } from "../types/replay";
import type { Incident } from "../types/incident";

export type { ReplaySample } from "../types/replay";

const unavailable = (what: string) => new Error(`${what}: demo fixtures are not part of the live build`);

export const TICK_MS = 0;
export const REPLAY_SAMPLES: ReplaySample[] = [];
export const mockFetchStatus = async (): Promise<SystemStatus> => Promise.reject(unavailable("status"));
export const mockFetchSite = async (): Promise<SiteInfo> => Promise.reject(unavailable("site"));
export const mockFetchTopology = async (): Promise<Topology> => Promise.reject(unavailable("topology"));
export const mockSendCommand = async (_command: string): Promise<void> => Promise.reject(unavailable("command"));
export const mockSendMitigate = async (_payload: MitigationPayload): Promise<void> => Promise.reject(unavailable("mitigate"));
export const mockReplay = async (_file: File, _maxWindows?: number): Promise<ReplayReport> => Promise.reject(unavailable("replay"));
export const mockReplaySample = async (_id: string): Promise<ReplayReport> => Promise.reject(unavailable("replay"));
export const seedFrames = (_n?: number): { payload: unknown; topology: Topology }[] => [];
export const seedEvents = (): { severity: "info" | "warning" | "error" | "critical"; category: string; source: string; destination?: string; message: string; raw?: string; ageMs: number }[] => [];
export const seedLog = (_n?: number): string[] => [];
export const seedIncidents = (): Incident[] => [];

export class MockWebSocket {
  constructor(_handlers: WSHandlers) {
    throw unavailable("stream");
  }
  close(): void {}
}
