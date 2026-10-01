/**
 * Interface sound.
 *
 * Off unless the viewer turns it on, and synthesized, so no audio ships: a
 * blip is one sine partial with a 4ms attack and a short exponential decay —
 * felt more than heard. The setting is a per-viewer convenience and lives in
 * browser storage.
 */

import { useSyncExternalStore } from "react";

const KEY = "ui_sound";
let enabled = (() => {
  try {
    return localStorage.getItem(KEY) === "on";
  } catch {
    return false;
  }
})();
const listeners = new Set<() => void>();
let ctx: AudioContext | null = null;

export function setSoundEnabled(on: boolean) {
  enabled = on;
  try {
    localStorage.setItem(KEY, on ? "on" : "off");
  } catch {
    /* per-viewer convenience only */
  }
  listeners.forEach((l) => l());
  if (on) blip(1320, 45);
}

export function useSound(): [boolean, (on: boolean) => void] {
  const on = useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => enabled
  );
  return [on, setSoundEnabled];
}

/** A single short tone. Silent unless sound is on; never throws. */
export function blip(freq = 880, ms = 40, gain = 0.035) {
  if (!enabled || typeof window === "undefined") return;
  try {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return;
    ctx ??= new AC();
    if (ctx.state === "suspended") void ctx.resume();
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, t);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
    osc.connect(g).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + ms / 1000 + 0.02);
  } catch {
    /* sound is decoration; never let it break the console */
  }
}

/** Two rising tones — an alert escalating. */
export function chime() {
  blip(660, 70, 0.04);
  setTimeout(() => blip(990, 110, 0.04), 80);
}
