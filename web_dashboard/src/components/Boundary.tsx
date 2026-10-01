import { Component, type ReactNode } from "react";

/**
 * Renders `fallback` if anything below it throws — including a lazy chunk
 * that fails to load. Used around the 3D host graph, so a machine without
 * WebGL, or a GPU context that is lost, drops back to the 2D graph instead of
 * blanking the page.
 */
export default class Boundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    console.warn("[host graph] 3D view unavailable, showing 2D:", error);
  }

  render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

let webgl: boolean | null = null;

/**
 * True when the browser can create a WebGL context at all.
 *
 * Probes once and caches. Browsers cap live WebGL contexts (about 16) and
 * evict the oldest when the cap is hit — which is the real renderer. An
 * uncached probe called during render created a context per stream tick and
 * killed the 3D canvas within seconds. The probe context is released
 * immediately for the same reason.
 */
export function hasWebGL(): boolean {
  if (webgl !== null) return webgl;
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl2") ?? c.getContext("webgl")) as WebGLRenderingContext | null;
    webgl = Boolean(gl);
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    webgl = false;
  }
  return webgl;
}
