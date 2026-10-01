/**
 * Host graph in three dimensions — the console's cinematic viewport.
 *
 * This canvas is the one deliberate exception to the design system's no-glow
 * rule (see DESIGN_SYSTEM.md §7). Inside it: a deep void, holographic layer
 * grids, glass hosts with an internal emissive core, a crimson attacker that
 * lights the glass around it, and bloom. Outside it, nothing glows.
 *
 * Everything is still driven by the discovery stream, not a fixed layout:
 *   - hosts appear when traffic is first seen and ease into place; a quiet
 *     host dims (stale) and is evicted on its TTL
 *   - links are observed flows, so they come and go with the traffic
 *   - packets travel in the direction each flow was initiated, at a density
 *     taken from the bytes that crossed the link in the latest window
 *   - hops the campaign forecasts but has not observed are dashed
 *
 * Labels are one DOM overlay projected each frame rather than a React root
 * per label. Loaded lazily from Network.tsx, so three.js and the post chain
 * are only fetched when the view opens.
 */

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  Environment,
  Grid,
  Lightformer,
  OrbitControls,
  QuadraticBezierLine,
  Sparkles,
  type QuadraticBezierLineRef,
} from "@react-three/drei";
import { Bloom, EffectComposer, ToneMapping, Vignette } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { Topology, TopologyNode } from "../api/types";
import type { PredictionEnvelope } from "../types/live";
import type { Campaign } from "../types/campaign";
import type { FlowRecord } from "../types/evidence";
import { dmzAnchor, hasZones, isInternal, siteName, zoneCidrs, zoneOfIp, type ZoneKey } from "../design/site";

/* ════════════════════════════════════════════════════════════════════════
   Viewport palette — fixed, not theme-driven. The void stays a void in PAPER.
   ════════════════════════════════════════════════════════════════════════ */

const VIZ = {
  void: "#030712",
  cell: "#1e293b",
  section: "#334155",
  rim: "#38bdf8",
  benign: "#059669",
  client: "#64748b",
  warning: "#f59e0b",
  elevated: "#fb923c",
  critical: "#e11d48",
  forecast: "#60a5fa",
  baseline: "#475569",
  packet: "#e2e8f0",
};

/** HDR colour: >1 components survive tone mapping into the bloom pass. */
function hdr(hex: string, k: number): THREE.Color {
  return new THREE.Color(hex).multiplyScalar(k);
}

/* ════════════════════════════════════════════════════════════════════════
   Layout
   ════════════════════════════════════════════════════════════════════════ */


/** The console's bands, top to bottom. Hosts are placed by the site's zones (design/site.ts). */
const ZONES: { key: ZoneKey; label: string; y: number }[] = [
  { key: "external", label: "External", y: 4.5 },
  { key: "dmz", label: "DMZ", y: 1.5 },
  { key: "servers", label: "Servers", y: -1.5 },
  { key: "users", label: "Users", y: -4.5 },
];

/** A site that declares no zones has one internal band. */
const zoneLabel = (z: (typeof ZONES)[number]) => (z.key === "users" && !hasZones() ? "Internal" : z.label);

const LAYER_R = 7.2;

type Host = TopologyNode & {
  risk?: number;
  zone?: string;
  role?: string;
  stale?: boolean;
  bytes_in?: number;
  bytes_out?: number;
  tier?: "core" | "background";
};

function zoneOf(n: Host): ZoneKey {
  if (n.zone === "external" || n.zone === "dmz" || n.zone === "servers" || n.zone === "users") return n.zone;
  // The site's declared zones (config/sites/*.yaml), not the lab's subnets.
  return zoneOfIp(n.ip ?? n.id);
}

const isCore = (n: Host) => n.tier !== "background";

/**
 * Target positions. Per layer: the attacker (or dmz-web) at the centre, core
 * hosts on an inner ring, background population on an outer ring. Ordered by
 * address, so positions are stable between windows.
 */
function layout(nodes: Host[], attacker: string | null): Map<string, THREE.Vector3> {
  const out = new Map<string, THREE.Vector3>();
  const byAddr = (a: Host, b: Host) => (a.ip ?? a.id).localeCompare(b.ip ?? b.id, undefined, { numeric: true });

  ZONES.forEach((z, zi) => {
    const inZone = nodes.filter((n) => zoneOf(n) === z.key).sort(byAddr);
    const centre =
      inZone.find((n) => n.id === attacker) ?? (z.key === "dmz" ? inZone.find((n) => n.ip === dmzAnchor()) : undefined);
    const core = inZone.filter((n) => n !== centre && isCore(n));
    const bg = inZone.filter((n) => n !== centre && !isCore(n));

    if (centre) out.set(centre.id, new THREE.Vector3(0, z.y, 0));

    const ring = (list: Host[], r: number, phase: number) => {
      list.forEach((n, i) => {
        const a = (i / Math.max(1, list.length)) * Math.PI * 2 + phase;
        out.set(n.id, new THREE.Vector3(Math.cos(a) * r, z.y, Math.sin(a) * r));
      });
    };

    if (!centre && core.length === 1 && bg.length === 0) {
      out.set(core[0].id, new THREE.Vector3(0, z.y, 0));
    } else {
      ring(core, core.length <= 3 ? 2.4 : 3.0, zi * 0.6);
    }
    ring(bg, z.key === "external" ? 5.4 : 5.6, zi * 0.6 + 0.2);
  });
  return out;
}

/** Control point for a curved link, bowed away from the scene axis. */
const _d = new THREE.Vector3();
const _p = new THREE.Vector3();
function midpoint(a: THREE.Vector3, b: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
  out.copy(a).add(b).multiplyScalar(0.5);
  _d.copy(b).sub(a);
  _p.set(-_d.z, 0, _d.x);
  if (_p.lengthSq() < 1e-4) _p.set(out.x, 0, out.z);
  if (_p.lengthSq() < 1e-4) _p.set(1, 0, 0);
  return out.add(_p.normalize().multiplyScalar(_d.length() * 0.14));
}

function bezier(a: THREE.Vector3, m: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3): THREE.Vector3 {
  const u = 1 - t;
  return out.set(
    u * u * a.x + 2 * u * t * m.x + t * t * b.x,
    u * u * a.y + 2 * u * t * m.y + t * t * b.y,
    u * u * a.z + 2 * u * t * m.z + t * t * b.z
  );
}

function useReducedMotion(): boolean {
  const [r, setR] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const on = () => setR(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return r;
}

/* ════════════════════════════════════════════════════════════════════════
   Layers — holographic grids with a lit rim
   ════════════════════════════════════════════════════════════════════════ */

function Layer({ y }: { y: number }) {
  const rim = useMemo(() => hdr(VIZ.rim, 1.35), []);
  const grid = useRef<THREE.Mesh>(null);

  // Flat and transparent: one pass draws both faces. Without this, three.js
  // splits a DoubleSide transparent material into back and front passes and
  // re-selects its program for each, every frame.
  useLayoutEffect(() => {
    const m = grid.current?.material;
    for (const mat of Array.isArray(m) ? m : m ? [m] : []) mat.forceSinglePass = true;
  }, []);

  return (
    <group position={[0, y, 0]}>
      <Grid
        ref={grid}
        args={[LAYER_R * 2.2, LAYER_R * 2.2]}
        cellSize={0.6}
        cellThickness={0.6}
        cellColor={VIZ.cell}
        sectionSize={3}
        sectionThickness={1.1}
        sectionColor={VIZ.section}
        fadeDistance={30}
        fadeStrength={1.4}
        side={THREE.DoubleSide}
      />
      {/* Faint body so the layer reads as a plane, not just lines. */}
      <mesh rotation-x={-Math.PI / 2}>
        <circleGeometry args={[LAYER_R, 96]} />
        <meshBasicMaterial color={VIZ.cell} transparent opacity={0.12} depthWrite={false} side={THREE.DoubleSide} forceSinglePass />
      </mesh>
      {/* Holographic rim. */}
      <mesh rotation-x={-Math.PI / 2}>
        <ringGeometry args={[LAYER_R, LAYER_R + 0.035, 160]} />
        <meshBasicMaterial color={rim} toneMapped={false} transparent opacity={0.4} side={THREE.DoubleSide} forceSinglePass />
      </mesh>
    </group>
  );
}

/** A faint vertical spine through the layer centres — the stack's axis. */
function Spine() {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute([0, ZONES[0].y + 1.2, 0, 0, ZONES[3].y - 0.6, 0], 3));
    return g;
  }, []);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color={VIZ.section} transparent opacity={0.6} />
    </lineSegments>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Hosts
   ════════════════════════════════════════════════════════════════════════ */

interface HostInfo {
  flowsPerSec: number;
  bytes: number;
  techniques: string[];
}

function fmtBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

/** Host detail — the hover tip and the full-screen inspector card share it. */
function HostCard({ node, info, isAttacker }: { node: Host; info: HostInfo | undefined; isAttacker: boolean }) {
  const colour = glowFor(node, isAttacker).colour;
  const stale = Boolean(node.stale) || node.status === "offline";
  return (
    <div className="n3-tip" style={{ borderLeftColor: colour }}>
      <div className="n3-tip-head">
        <span>{node.label}</span>
        <span className="n3-tip-state" style={{ color: colour }}>
          {stale ? "stale" : isAttacker ? "attacker" : node.status}
        </span>
      </div>
      <div className="n3-tip-ip">{node.ip}</div>
      <dl>
        <dt>zone</dt>
        <dd>{zoneOf(node)}</dd>
        <dt>risk</dt>
        <dd style={{ color: colour }}>{node.risk != null ? Number(node.risk).toFixed(3) : "—"}</dd>
        <dt>flows</dt>
        <dd>{(info?.flowsPerSec ?? 0).toFixed(1)}/s</dd>
        <dt>bytes</dt>
        <dd>{fmtBytes(info?.bytes ?? 0)}</dd>
        {info && info.techniques.length > 0 && (
          <>
            <dt>seen</dt>
            <dd>{info.techniques.join(" · ")}</dd>
          </>
        )}
      </dl>
    </div>
  );
}

type Shape = "attacker" | "server" | "host" | "client";

function shapeOf(n: Host, isAttacker: boolean): Shape {
  if (isAttacker) return "attacker";
  const z = zoneOf(n);
  if (z === "external") return "client";
  if (z === "users") return "host";
  return "server";
}

function glowFor(n: Host, isAttacker: boolean): { colour: string; intensity: number } {
  const stale = Boolean(n.stale) || n.status === "offline";
  if (stale) return { colour: VIZ.client, intensity: 0.08 };
  if (isAttacker || n.status === "compromised") return { colour: VIZ.critical, intensity: 2.5 };
  if (n.status === "warning" || n.status === "degraded") return { colour: VIZ.warning, intensity: 1.4 };
  if (zoneOf(n) === "external") return { colour: VIZ.client, intensity: 0.35 };
  return { colour: VIZ.benign, intensity: isCore(n) ? 0.55 : 0.35 };
}

/** A stable phase per host, so its pulse and float don't reset on re-render. */
function phaseOf(id: string): number {
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 997;
  return (h / 997) * Math.PI * 2;
}

function HostNode({
  node,
  target,
  live,
  hovered,
  selected,
  focus,
  forecastTarget,
  isAttacker,
  motion,
  onHover,
  onSelect,
}: {
  node: Host;
  target: THREE.Vector3;
  live: Map<string, THREE.Vector3>;
  hovered: boolean;
  selected: boolean;
  focus: boolean;
  forecastTarget: boolean;
  isAttacker: boolean;
  motion: boolean;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const group = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const mat = useRef<THREE.MeshPhysicalMaterial>(null);
  const halo = useRef<THREE.Mesh>(null);
  const shell = useRef<THREE.Mesh>(null);
  const size = useRef(0.001);
  const phase = useMemo(() => phaseOf(node.id), [node.id]);

  const shape = shapeOf(node, isAttacker);
  const stale = Boolean(node.stale) || node.status === "offline";
  const hostile = isAttacker || node.status === "compromised";
  const { colour, intensity } = glowFor(node, isAttacker);
  const core = isCore(node);

  const base =
    shape === "attacker" ? 0.46 : hostile ? 0.4 : shape === "server" ? (core ? 0.32 : 0.24) : shape === "host" ? (core ? 0.2 : 0.15) : 0.14;

  const haloColour = selected ? "#f8fafc" : hostile ? VIZ.critical : focus ? VIZ.warning : forecastTarget ? VIZ.forecast : null;
  const haloHdr = useMemo(() => (haloColour ? hdr(haloColour, 1.8) : null), [haloColour]);
  const shellHdr = useMemo(() => hdr(VIZ.critical, 1.6), []);

  // Forget the live position on unmount, so a host that returns drops in again.
  useEffect(() => () => void live.delete(node.id), [live, node.id]);

  useFrame((state, dt) => {
    const g = group.current;
    if (!g) return;
    const t = state.clock.elapsedTime;

    let p = live.get(node.id);
    if (!p) {
      p = target.clone().add(new THREE.Vector3(0, 1.6, 0));
      live.set(node.id, p);
    }
    p.lerp(target, 1 - Math.exp(-dt * 2.6));
    g.position.copy(p);

    // Float on the body only, so the anchor the links attach to stays still.
    if (body.current) {
      body.current.position.y = motion ? Math.sin(t * 0.9 + phase) * 0.07 / Math.max(0.1, size.current) : 0;
      if (motion) body.current.rotation.y += dt * (hostile ? 0.9 : 0.25);
    }

    const want = base * (hovered ? 1.2 : 1);
    size.current += (want - size.current) * (1 - Math.exp(-dt * 9));
    g.scale.setScalar(size.current);

    if (mat.current && hostile) {
      mat.current.emissiveIntensity = motion ? intensity + Math.sin(t * 3.2 + phase) * 0.8 : intensity;
    }
    if (halo.current) {
      halo.current.rotation.z += dt * 0.6;
      const k = hostile && motion ? 1 + Math.sin(t * 3.2 + phase) * 0.08 : 1;
      halo.current.scale.setScalar(k);
    }
    if (shell.current && motion) {
      shell.current.rotation.x += dt * 0.4;
      shell.current.rotation.y -= dt * 0.55;
    }
  });

  return (
    <group ref={group}>
      <group ref={body}>
        <mesh
          onPointerOver={(e) => {
            e.stopPropagation();
            onHover(node.id);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={() => {
            onHover(null);
            document.body.style.cursor = "";
          }}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(node.id);
          }}
        >
          {shape === "attacker" ? (
            <icosahedronGeometry args={[1, 0]} />
          ) : shape === "server" ? (
            <octahedronGeometry args={[1, 0]} />
          ) : (
            <sphereGeometry args={[1, 32, 20]} />
          )}
          <meshPhysicalMaterial
            ref={mat}
            color={stale ? "#1e293b" : "#0b1220"}
            emissive={colour}
            emissiveIntensity={intensity}
            transmission={0.9}
            thickness={0.6}
            ior={1.4}
            roughness={0.1}
            metalness={0.3}
            clearcoat={1}
            clearcoatRoughness={0.22}
            envMapIntensity={0.9}
            transparent
            opacity={stale ? 0.35 : 1}
          />
        </mesh>

        {/* The attacker's slowly counter-rotating wire shell. */}
        {shape === "attacker" && (
          <mesh ref={shell} scale={1.55}>
            <icosahedronGeometry args={[1, 1]} />
            <meshBasicMaterial color={shellHdr} wireframe toneMapped={false} transparent opacity={0.35} />
          </mesh>
        )}
      </group>

      {haloHdr && (
        <mesh ref={halo} rotation-x={Math.PI / 2}>
          <torusGeometry args={[1.85, 0.03, 8, 64]} />
          <meshBasicMaterial color={haloHdr} toneMapped={false} transparent opacity={selected ? 1 : 0.85} />
        </mesh>
      )}

    </group>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Links + packets
   ════════════════════════════════════════════════════════════════════════ */

interface Lane {
  a: string;
  b: string;
  path: boolean;
  hot: boolean;
  bytes: number;
}

/*
 * Placeholder endpoints. useFrame writes the real ones through setPoints.
 * They must be stable references: a fresh array literal per render makes drei
 * recompute and reset the curve on every topology tick.
 */
const ORIGIN: [number, number, number] = [0, 0, 0];
const NUDGE: [number, number, number] = [0, 0.01, 0];

/** Only rewrite a curve when an endpoint has actually moved. */
/** Segments per link — drei's QuadraticBezierLine default, and so the size of the buffer it allocates. */
const SEGMENTS = 20;
const _s = new THREE.Vector3();
const _e = new THREE.Vector3();

/**
 * Rewrites a link's vertices inside the buffers drei already allocated.
 *
 * drei's setPoints samples a new curve, builds a new Float32Array and uploads a
 * new GPU buffer on every call. With ~60 links following their hosts as they
 * settle, that was thousands of buffers a second and a garbage-collection pause
 * every few seconds. The segment count never changes, so the buffer is reused.
 * Returns false if the buffers aren't the expected shape, so the caller can
 * fall back to setPoints.
 */
function writeCurve(line: QuadraticBezierLineRef, curve: THREE.QuadraticBezierCurve3, dashed: boolean): boolean {
  const geo = line.geometry;
  const start = geo.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute | undefined;
  if (!start?.data || start.data.count !== SEGMENTS) return false;
  const dist = dashed ? (geo.getAttribute("instanceDistanceStart") as THREE.InterleavedBufferAttribute | undefined) : undefined;
  if (dashed && dist?.data?.count !== SEGMENTS) return false;

  const pos = start.data.array as Float32Array;
  const d = dist?.data.array as Float32Array | undefined;
  let run = 0;
  curve.getPoint(0, _s);
  for (let i = 0; i < SEGMENTS; i++) {
    curve.getPoint((i + 1) / SEGMENTS, _e);
    const o = i * 6;
    pos[o] = _s.x;
    pos[o + 1] = _s.y;
    pos[o + 2] = _s.z;
    pos[o + 3] = _e.x;
    pos[o + 4] = _e.y;
    pos[o + 5] = _e.z;
    if (d) {
      d[i * 2] = run;
      run += _s.distanceTo(_e);
      d[i * 2 + 1] = run;
    }
    _s.copy(_e);
  }
  start.data.needsUpdate = true;
  if (dist) dist.data.needsUpdate = true;
  // Bounds drive culling and picking; both reuse their objects after the first call.
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return true;
}

function useCurve(a: string, b: string, live: Map<string, THREE.Vector3>, dashed = false) {
  const ref = useRef<QuadraticBezierLineRef>(null);
  const curve = useMemo(() => new THREE.QuadraticBezierCurve3(), []);
  const lastA = useMemo(() => new THREE.Vector3(Infinity, 0, 0), []);
  const lastB = useMemo(() => new THREE.Vector3(Infinity, 0, 0), []);

  useFrame(() => {
    const pa = live.get(a);
    const pb = live.get(b);
    const line = ref.current;
    if (!pa || !pb || !line) return;
    if (pa.distanceToSquared(lastA) < 1e-6 && pb.distanceToSquared(lastB) < 1e-6) return;
    lastA.copy(pa);
    lastB.copy(pb);
    curve.v0.copy(pa);
    curve.v2.copy(pb);
    midpoint(pa, pb, curve.v1);
    if (writeCurve(line, curve, dashed)) return;
    line.setPoints(pa, pb, curve.v1);
    if (dashed) line.computeLineDistances();
  });
  return ref;
}

function Link({ lane, live }: { lane: Lane; live: Map<string, THREE.Vector3> }) {
  const ref = useCurve(lane.a, lane.b, live);
  const colour = useMemo(
    () => (lane.path ? hdr(VIZ.critical, lane.hot ? 2.6 : 2.0) : new THREE.Color(VIZ.baseline)),
    [lane.path, lane.hot]
  );
  return (
    <QuadraticBezierLine
      ref={ref}
      start={ORIGIN}
      end={NUDGE}
      color={colour}
      lineWidth={lane.path ? 3.2 : 0.8}
      transparent
      opacity={lane.path ? 1 : 0.32}
      toneMapped={!lane.path}
    />
  );
}

/** A hop the campaign forecasts but has not observed — dashed and cool. */
function ForecastLink({ a, b, live }: { a: string; b: string; live: Map<string, THREE.Vector3> }) {
  const ref = useCurve(a, b, live, true);
  const colour = useMemo(() => hdr(VIZ.forecast, 1.5), []);
  return (
    <QuadraticBezierLine
      ref={ref}
      start={ORIGIN}
      end={NUDGE}
      color={colour}
      lineWidth={1.8}
      dashed
      dashSize={0.24}
      gapSize={0.16}
      toneMapped={false}
    />
  );
}

const MAX_PACKETS = 900;
const TRAIL = 5;

/**
 * Traffic on the wire, one instanced mesh for the whole scene. Attack packets
 * carry a short comet trail; baseline packets are single glints.
 */
/**
 * Crimson light around hostile hosts, from a fixed pool.
 *
 * The number of lights is part of every lit shader's program, so one light per
 * hostile host recompiled the whole scene each time a host turned. The pool
 * keeps the count constant; unused lights sit dark.
 */
const LIGHT_POOL = 3;

function HostileLights({ ids, live, motion }: { ids: string[]; live: Map<string, THREE.Vector3>; motion: boolean }) {
  const refs = useRef<(THREE.PointLight | null)[]>([]);
  const phases = useMemo(() => ids.map(phaseOf), [ids]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    for (let i = 0; i < LIGHT_POOL; i++) {
      const l = refs.current[i];
      if (!l) continue;
      const p = i < ids.length ? live.get(ids[i]) : undefined;
      if (!p) {
        l.intensity = 0;
        continue;
      }
      l.position.copy(p);
      l.intensity = motion ? 9 + Math.sin(t * 3.2 + phases[i]) * 3 : 9;
    }
  });

  return (
    <>
      {Array.from({ length: LIGHT_POOL }, (_, i) => (
        <pointLight
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          color={VIZ.critical}
          intensity={0}
          distance={7}
          decay={2}
        />
      ))}
    </>
  );
}

function Packets({ lanes, live, motion }: { lanes: Lane[]; live: Map<string, THREE.Vector3>; motion: boolean }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const pos = useMemo(() => new THREE.Vector3(), []);
  const mid = useMemo(() => new THREE.Vector3(), []);

  const packets = useMemo(() => {
    const out: { lane: Lane; off: number; speed: number; size: number; seg: number }[] = [];
    for (const lane of lanes) {
      const mag = Math.log10(Math.max(10, lane.bytes));
      const n = lane.path ? Math.round(Math.min(12, 3 + mag * 1.2)) : Math.round(Math.min(3, 0.6 + mag * 0.35));
      for (let k = 0; k < n; k++) {
        const segs = lane.path ? TRAIL : 1;
        for (let s = 0; s < segs && out.length < MAX_PACKETS; s++) {
          out.push({
            lane,
            off: k / n - s * 0.014,
            speed: lane.path ? 0.42 : 0.16,
            size: (lane.path ? 0.075 : 0.035) * (1 - s * 0.17),
            seg: s,
          });
        }
      }
    }
    return out;
  }, [lanes]);

  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const c = new THREE.Color();
    packets.forEach((p, i) => {
      const fade = 1 - p.seg * 0.18;
      if (p.lane.path) c.set(VIZ.critical).multiplyScalar((p.lane.hot ? 4 : 3) * fade);
      else c.set(VIZ.packet).multiplyScalar(0.85);
      m.setColorAt(i, c);
    });
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [packets]);

  useFrame((state) => {
    const m = mesh.current;
    if (!m) return;
    const t = state.clock.elapsedTime;
    packets.forEach((p, i) => {
      const a = live.get(p.lane.a);
      const b = live.get(p.lane.b);
      if (!a || !b) {
        dummy.scale.setScalar(0);
      } else {
        const u = ((((motion ? t * p.speed : 0) + p.off) % 1) + 1) % 1;
        bezier(a, midpoint(a, b, mid), b, u, pos);
        dummy.position.copy(pos);
        dummy.scale.setScalar(p.size);
      }
      dummy.updateMatrix();
      m.setMatrixAt(i, dummy.matrix);
    });
    m.count = packets.length;
    m.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, MAX_PACKETS]} frustumCulled={false}>
      <sphereGeometry args={[1, 10, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Label overlay
   ════════════════════════════════════════════════════════════════════════ */

const ZONE_ANCHOR: Record<ZoneKey, THREE.Vector3> = Object.fromEntries(
  ZONES.map((z) => [z.key, new THREE.Vector3(-LAYER_R, z.y, 0)])
) as Record<ZoneKey, THREE.Vector3>;

/**
 * Projects label anchors to screen space once per frame and writes the
 * transform straight onto the DOM. No React render moves a label, and nothing
 * mounts or unmounts on hover.
 */
function Projector({
  live,
  anchors,
  tip,
  hover,
}: {
  live: Map<string, THREE.Vector3>;
  anchors: RefObject<Map<string, HTMLDivElement>>;
  tip: RefObject<HTMLDivElement | null>;
  hover: string | null;
}) {
  const camera = useThree((st) => st.camera);
  const size = useThree((st) => st.size);
  const v = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const place = (el: HTMLElement, p: THREE.Vector3, lift: number) => {
      v.set(p.x, p.y + lift, p.z).project(camera);
      if (v.z > 1) {
        el.style.visibility = "hidden";
        return;
      }
      el.style.visibility = "visible";
      el.style.transform = `translate3d(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px, 0)`;
    };

    anchors.current.forEach((el, id) => {
      if (id.startsWith("zone:")) {
        place(el, ZONE_ANCHOR[id.slice(5) as ZoneKey], 0);
        return;
      }
      const p = live.get(id);
      if (p) place(el, p, 0.62);
      else el.style.visibility = "hidden";
    });

    if (tip.current) {
      const p = hover ? live.get(hover) : undefined;
      if (p) place(tip.current, p, 0.2);
      else tip.current.style.visibility = "hidden";
    }
  });

  return null;
}

/**
 * Warms the renderer before the first frame, then reports once it runs steadily.
 *
 * The glass material and the post-processing chain are heavy programs. Left
 * alone, three.js compiles them synchronously inside the first frame, which on
 * an integrated GPU blocks the page for seconds. The canvas therefore mounts
 * with its loop stopped; this compiles the mounted scene in parallel
 * (compileAsync, off the main thread where the driver allows), starts the loop,
 * and reveals after a few consecutive frames complete quickly — anything
 * compileAsync could not see, such as the composer's passes, has compiled by
 * then. Ceilings keep a slow driver from holding the boot state.
 */
function ShadersReady({ onWarm, onReady }: { onWarm: () => void; onReady: () => void }) {
  const gl = useThree((st) => st.gl);
  const scene = useThree((st) => st.scene);
  const camera = useThree((st) => st.camera);
  const done = useRef(false);
  const steady = useRef(0);

  useEffect(() => {
    let live = true;
    const finish = () => {
      if (!live || done.current) return;
      done.current = true;
      onWarm();
      onReady();
    };
    const warmCeiling = window.setTimeout(() => live && onWarm(), 8000);
    const readyCeiling = window.setTimeout(finish, 20000);
    // One macrotask, so every sibling's effects — the environment capture,
    // instanced buffers — have attached before the scene is compiled.
    const kick = window.setTimeout(() => {
      gl.compileAsync(scene, camera)
        .catch(() => undefined)
        .then(() => {
          if (!live) return;
          window.clearTimeout(warmCeiling);
          onWarm();
        });
    }, 0);
    return () => {
      live = false;
      window.clearTimeout(kick);
      window.clearTimeout(warmCeiling);
      window.clearTimeout(readyCeiling);
    };
  }, [gl, scene, camera, onWarm, onReady]);

  useFrame((_, delta) => {
    if (done.current) return;
    // A frame that stalls on a late compile resets the count.
    steady.current = delta < 0.1 ? steady.current + 1 : 0;
    if (steady.current < 3) return;
    done.current = true;
    onReady();
  });
  return null;
}

const _dir = new THREE.Vector3();

/**
 * Fits the zone stack to the viewport.
 *
 * A fixed camera distance left a wide panel mostly empty and a narrow one
 * cropped. The fit projects the stack's extremes — every plane's rim, and
 * headroom above the top plane for the attacker and its label — and searches
 * for the closest distance that keeps them inside the frame, for whichever
 * dimension binds. It re-fits, eased, when the viewport changes size (entering
 * full screen included) and on Reset; the first fit is immediate, under the
 * boot overlay.
 */
const STACK_POINTS: THREE.Vector3[] = (() => {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * Math.PI * 2;
    for (const z of ZONES) pts.push(new THREE.Vector3(Math.cos(a) * LAYER_R, z.y, Math.sin(a) * LAYER_R));
    pts.push(new THREE.Vector3(Math.cos(a) * 1.2, ZONES[0].y + 1.3, Math.sin(a) * 1.2));
  }
  return pts;
})();

const _probe = new THREE.PerspectiveCamera();
const _v = new THREE.Vector3();

/** The stack's projected extent (NDC) with the camera `d` from `target` along `dir`. */
function extent(d: number, dir: THREE.Vector3, target: THREE.Vector3) {
  _probe.position.copy(target).addScaledVector(dir, d);
  _probe.lookAt(target);
  _probe.updateMatrixWorld();
  let minY = Infinity;
  let maxY = -Infinity;
  let maxX = 0;
  for (const p of STACK_POINTS) {
    _v.copy(p).project(_probe);
    minY = Math.min(minY, _v.y);
    maxY = Math.max(maxY, _v.y);
    maxX = Math.max(maxX, Math.abs(_v.x));
  }
  return { minY, maxY, maxX };
}

const _aim = new THREE.Vector3();

/**
 * The closest camera distance, and the orbit target's height, that frame the
 * whole stack with `margin` to spare. The target height centres the stack's
 * projection vertically; a few rounds of fit-then-centre converge.
 */
function fitView(fov: number, aspect: number, dir: THREE.Vector3, margin: number): { d: number; aimY: number } {
  _probe.fov = fov;
  _probe.aspect = aspect;
  _probe.near = 0.1;
  _probe.far = 200;
  _probe.updateProjectionMatrix();
  const half = Math.tan(THREE.MathUtils.degToRad(fov / 2));
  let aimY = 0;
  let d = 30;
  for (let round = 0; round < 4; round++) {
    _aim.set(0, aimY, 0);
    let lo = 8;
    let hi = 60;
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) / 2;
      const e = extent(mid, dir, _aim);
      if (Math.max(e.maxX, -e.minY, e.maxY) * margin <= 1) hi = mid;
      else lo = mid;
    }
    d = hi;
    const e = extent(d, dir, _aim);
    // Shift the aim by the projection's off-centre, in world units at distance d.
    aimY += ((e.maxY + e.minY) / 2) * half * d;
  }
  return { d, aimY };
}

function Frame({ full, nonce }: { full: boolean; nonce: number }) {
  const camera = useThree((st) => st.camera) as THREE.PerspectiveCamera;
  const controls = useThree((st) => st.controls) as unknown as OrbitControlsImpl | null;
  const width = useThree((st) => st.size.width);
  const height = useThree((st) => st.size.height);
  const tween = useRef<{ fromD: number; toD: number; fromY: number; toY: number; t0: number } | null>(null);
  const placed = useRef(false);

  useEffect(() => {
    // Wait for the orbit controls: the fit moves their target as well.
    if (!controls || width < 2 || height < 2) return;
    const target = controls.target;
    _dir.copy(camera.position).sub(target).normalize();
    // Tighter in full screen: the HUD sits in the corners, not over the stack.
    const fit = fitView(camera.fov, width / height, _dir, full ? 1.05 : 1.1);
    const toD = THREE.MathUtils.clamp(fit.d, 12, 42);
    if (!placed.current) {
      placed.current = true;
      target.set(0, fit.aimY, 0);
      camera.position.copy(target).addScaledVector(_dir, toD);
      return;
    }
    tween.current = { fromD: camera.position.distanceTo(target), toD, fromY: target.y, toY: fit.aimY, t0: performance.now() };
  }, [width, height, full, nonce, camera, controls]);

  useFrame(() => {
    const tw = tween.current;
    if (!tw || !controls) return;
    const k = Math.min(1, (performance.now() - tw.t0) / 720);
    const e = 1 - Math.pow(1 - k, 3);
    const target = controls.target;
    _dir.copy(camera.position).sub(target).normalize();
    target.y = tw.fromY + (tw.toY - tw.fromY) * e;
    camera.position.copy(target).addScaledVector(_dir, tw.fromD + (tw.toD - tw.fromD) * e);
    if (k >= 1) tween.current = null;
  });
  return null;
}

/** The procedural studio the glass reflects. Memoised: its capture re-runs whenever its children change identity. */
const Studio = memo(function Studio() {
  return (
    <Environment resolution={256} frames={1}>
      {/* Kept low: the servers' upward faces mirror it almost exactly, and at
          full-screen size a brighter ring blooms them white. */}
      <Lightformer form="ring" intensity={0.3} color="#94a3b8" scale={12} position={[0, 12, 0]} rotation-x={Math.PI / 2} />
      <Lightformer form="rect" intensity={0.8} color="#38bdf8" scale={[10, 2, 1]} position={[-10, 2, -6]} />
      <Lightformer form="rect" intensity={0.6} color="#f8fafc" scale={[10, 2, 1]} position={[10, -2, 6]} />
    </Environment>
  );
});

/* ════════════════════════════════════════════════════════════════════════
   Scene
   ════════════════════════════════════════════════════════════════════════ */

interface Network3DProps {
  topology: Topology | null;
  envelope: PredictionEnvelope | null;
  campaign: Campaign | null;
  flows: FlowRecord[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  /** The current verdict, shown across the top in full screen. */
  situation?: { level?: string | null; risk?: number | null; technique?: string | null } | null;
}

function levelColour(level: string | null | undefined): string {
  switch (String(level ?? "").toUpperCase()) {
    case "CRITICAL":
      return VIZ.critical;
    case "ELEVATED":
      return VIZ.elevated;
    case "WARNING":
      return VIZ.warning;
    default:
      return VIZ.benign;
  }
}

export default function Network3D({ topology, envelope, campaign, flows, selected, onSelect, situation }: Network3DProps) {
  const reduced = useReducedMotion();
  const motion = !reduced;

  const [hover, setHover] = useState<string | null>(null);
  const [orbit, setOrbit] = useState(true);
  const [ready, setReady] = useState(false);
  const [warm, setWarm] = useState(false);
  const start = useCallback(() => setWarm(true), []);
  const reveal = useCallback(() => setReady(true), []);
  const controls = useRef<OrbitControlsImpl>(null);
  const root = useRef<HTMLDivElement>(null);

  /* Full screen. Native where the browser allows it — the viewport element
     itself goes full screen, so the scene fills the display with no chrome.
     If the request is refused, the viewport covers the window instead. */
  const [full, setFull] = useState(false);
  const [refit, setRefit] = useState(0);
  const [covering, setCovering] = useState(false);

  useEffect(() => {
    const onChange = () => setFull(document.fullscreenElement === root.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      if (document.fullscreenElement === root.current) void document.exitFullscreen().catch(() => undefined);
    };
  }, []);

  const toggleFull = useCallback(() => {
    const el = root.current;
    if (!el) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => undefined);
      return;
    }
    if (covering) {
      setCovering(false);
      setFull(false);
      return;
    }
    const cover = () => {
      setCovering(true);
      setFull(true);
    };
    if (typeof el.requestFullscreen === "function") el.requestFullscreen({ navigationUI: "hide" }).catch(cover);
    else cover();
  }, [covering]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleFull();
      } else if (e.key === "Escape" && covering) {
        setCovering(false);
        setFull(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleFull, covering]);

  const live = useMemo(() => new Map<string, THREE.Vector3>(), []);
  const anchors = useRef(new Map<string, HTMLDivElement>());
  const tip = useRef<HTMLDivElement>(null);
  // Stable ref callbacks per anchor, so re-renders don't churn the map.
  const refCache = useRef(new Map<string, (el: HTMLDivElement | null) => void>());
  const anchorRef = (id: string) => {
    let cb = refCache.current.get(id);
    if (!cb) {
      cb = (el) => {
        if (el) anchors.current.set(id, el);
        else anchors.current.delete(id);
      };
      refCache.current.set(id, cb);
    }
    return cb;
  };

  const nodes = (topology?.nodes ?? []) as Host[];
  const focus = useMemo(() => new Set(envelope?.focus_ips ?? []), [envelope]);
  const attacker = useMemo(() => [...focus].find((ip) => !isInternal(ip)) ?? null, [focus]);
  const latestWindow = envelope?.state?.window_id ?? null;

  const targets = useMemo(() => layout(nodes, attacker), [nodes, attacker]);
  // The attacker first, so it always holds a light when the pool is full.
  const hostileIds = useMemo(
    () =>
      nodes
        .filter((n) => (n.id === attacker || n.status === "compromised") && !n.stale && n.status !== "offline")
        .sort((a, b) => Number(b.id === attacker) - Number(a.id === attacker))
        .map((n) => n.id),
    [nodes, attacker],
  );
  const present = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes]);

  /* Links from the discovered edges, sized by this window's bytes. */
  const lanes = useMemo<Lane[]>(() => {
    const win = flows.filter((f) => f.window === latestWindow);
    const bytesOf = new Map<string, number>();
    for (const f of win) {
      const k = f.src_ip < f.dst_ip ? `${f.src_ip}|${f.dst_ip}` : `${f.dst_ip}|${f.src_ip}`;
      bytesOf.set(k, (bytesOf.get(k) ?? 0) + f.fwd_bytes + f.bwd_bytes);
    }
    return (topology?.edges ?? [])
      .filter((e) => present.has(e.source) && present.has(e.target))
      .map((e) => {
        const k = e.source < e.target ? `${e.source}|${e.target}` : `${e.target}|${e.source}`;
        return {
          a: e.source,
          b: e.target,
          path: e.status === "suspicious" || e.status === "saturated",
          hot: e.status === "saturated",
          bytes: bytesOf.get(k) ?? 0,
        };
      });
  }, [topology, flows, latestWindow, present]);

  /* Forecast hops: campaign edges whose destination node is not observed yet. */
  const forecastHops = useMemo(() => {
    if (!campaign) return [] as { a: string; b: string }[];
    const byId = new Map(campaign.nodes.map((n) => [n.node_id, n]));
    const seen = new Set<string>();
    const out: { a: string; b: string }[] = [];
    for (const e of campaign.edges) {
      const s = byId.get(e.src);
      const d = byId.get(e.dst);
      if (!s || !d || d.provenance !== "FORECAST" || s.host_ip === d.host_ip) continue;
      const key = `${s.host_ip}>${d.host_ip}`;
      if (seen.has(key) || !present.has(s.host_ip) || !present.has(d.host_ip)) continue;
      seen.add(key);
      out.push({ a: s.host_ip, b: d.host_ip });
    }
    return out;
  }, [campaign, present]);

  const forecastTargets = useMemo(() => new Set(forecastHops.map((h) => h.b)), [forecastHops]);

  /* Per-host detail for the hover card. */
  const info = useMemo(() => {
    const m = new Map<string, HostInfo>();
    const win = flows.filter((f) => f.window === latestWindow);
    for (const n of nodes) {
      const mine = win.filter((f) => f.src_ip === n.id || f.dst_ip === n.id);
      m.set(n.id, {
        flowsPerSec: mine.length / 2.0,
        bytes: mine.reduce((s, f) => s + f.fwd_bytes + f.bwd_bytes, 0),
        techniques: [
          ...new Set((campaign?.nodes ?? []).filter((c) => c.host_ip === n.id && c.provenance === "OBSERVED").map((c) => c.technique_id)),
        ],
      });
    }
    return m;
  }, [nodes, flows, latestWindow, campaign]);

  const counts = useMemo(() => {
    const c: Record<ZoneKey, number> = { external: 0, dmz: 0, servers: 0, users: 0 };
    for (const n of nodes) c[zoneOf(n)] += 1;
    return c;
  }, [nodes]);

  useEffect(() => () => void (document.body.style.cursor = ""), []);

  const hoverNode = hover ? (nodes.find((n) => n.id === hover) ?? null) : null;
  const pinned = full && selected ? (nodes.find((n) => n.id === selected) ?? null) : null;

  // Only hosts that carry the story are labelled; the population stays quiet.
  const labelled = nodes.filter((n) => isCore(n) || n.id === selected || n.id === attacker);

  const rootClass = ["n3", full && "is-full", covering && "is-covering"].filter(Boolean).join(" ");

  return (
    <div className={rootClass} ref={root}>
      <Canvas
        flat
        // Held until the scene's shaders have compiled; see <ShadersReady>.
        frameloop={warm ? "always" : "never"}
        dpr={[1, 1.75]}
        camera={{ position: [19, 7.6, 19], fov: 36, near: 0.1, far: 160 }}
        gl={{ antialias: false, powerPreference: "high-performance", stencil: false }}
        onPointerMissed={() => onSelect(null)}
      >
        <color attach="background" args={[VIZ.void]} />
        <fog attach="fog" args={[VIZ.void, 30, 64]} />

        <ambientLight intensity={0.35} />
        <hemisphereLight args={["#1e293b", VIZ.void, 0.6]} />
        <directionalLight position={[8, 14, 6]} intensity={0.9} />

        {/* Procedural studio for the glass to reflect — no HDR download. */}
        <Studio />

        <Spine />
        {ZONES.map((z) => (
          <Layer key={z.key} y={z.y} />
        ))}

        {lanes.map((l) => (
          <Link key={`${l.a}-${l.b}`} lane={l} live={live} />
        ))}
        {forecastHops.map((h) => (
          <ForecastLink key={`f-${h.a}-${h.b}`} a={h.a} b={h.b} live={live} />
        ))}

        {nodes.map((n) => {
          const target = targets.get(n.id);
          if (!target) return null;
          return (
            <HostNode
              key={n.id}
              node={n}
              target={target}
              live={live}
              hovered={hover === n.id}
              selected={selected === n.id}
              focus={Boolean(n.ip && focus.has(n.ip))}
              forecastTarget={forecastTargets.has(n.id)}
              isAttacker={n.id === attacker}
              motion={motion}
              onHover={setHover}
              onSelect={(id) => onSelect(id === selected ? null : id)}
            />
          );
        })}

        <HostileLights ids={hostileIds} live={live} motion={motion} />
        <Packets lanes={lanes} live={live} motion={motion} />

        <Sparkles count={140} scale={[18, 13, 18]} size={1.4} speed={motion ? 0.22 : 0} opacity={0.35} color="#94a3b8" />

        <Projector live={live} anchors={anchors} tip={tip} hover={hover} />
        <ShadersReady onWarm={start} onReady={reveal} />
        <Frame full={full} nonce={refit} />

        <OrbitControls
          ref={controls}
          makeDefault
          autoRotate={orbit && motion && hover == null}
          autoRotateSpeed={0.4}
          enableDamping
          dampingFactor={0.06}
          enablePan={false}
          minDistance={12}
          maxDistance={42}
          minPolarAngle={0.45}
          maxPolarAngle={1.35}
        />

        <EffectComposer multisampling={4}>
          <Bloom mipmapBlur luminanceThreshold={0.2} luminanceSmoothing={0.9} intensity={1.5} />
          <Vignette offset={0.22} darkness={0.78} />
          <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
        </EffectComposer>
      </Canvas>

      <div className={ready ? "n3-boot is-done" : "n3-boot"} aria-hidden={ready}>
        <span>Initialising renderer</span>
        <i />
      </div>

      {/* ── Labels — projected each frame by <Projector> ─────────────── */}
      <div className="n3-labels" aria-hidden style={{ opacity: ready ? 1 : 0 }}>
        {ZONES.map((z) => (
          <div key={z.key} className="n3-anchor" ref={anchorRef(`zone:${z.key}`)}>
            <div className="n3-zone">
              {zoneLabel(z)}
              <span>
                {zoneCidrs(z.key)} · {counts[z.key]}
              </span>
            </div>
          </div>
        ))}
        {labelled.map((n) => {
          const stale = Boolean(n.stale) || n.status === "offline";
          const cls = stale ? "n3-label is-stale" : n.id === selected ? "n3-label is-selected" : "n3-label";
          return (
            <div key={n.id} className="n3-anchor" ref={anchorRef(n.id)}>
              <div className={cls}>{n.label}</div>
            </div>
          );
        })}
        <div className="n3-anchor" ref={tip}>
          {hoverNode && hoverNode.id !== pinned?.id && (
            <HostCard node={hoverNode} info={info.get(hoverNode.id)} isAttacker={hoverNode.id === attacker} />
          )}
        </div>
      </div>

      {/* ── HUD ──────────────────────────────────────────────────────── */}
      <div className="n3-hud n3-tl">
        <span className="n3-key">
          <i style={{ background: VIZ.benign }} /> internal
        </span>
        <span className="n3-key">
          <i style={{ background: VIZ.client }} /> external client
        </span>
        <span className="n3-key">
          <i style={{ background: VIZ.warning }} /> at risk
        </span>
        <span className="n3-key">
          <i style={{ background: VIZ.critical }} /> hostile
        </span>
        <span className="n3-key">
          <i className="is-dash" style={{ borderColor: VIZ.forecast }} /> forecast hop
        </span>
      </div>

      <div className="n3-hud n3-tr">
        <span className="seg-group">
          <button className="seg" aria-pressed={orbit} onClick={() => setOrbit((o) => !o)}>
            Orbit
          </button>
          <button
            className="seg"
            onClick={() => {
              controls.current?.reset();
              setRefit((n) => n + 1);
            }}
          >
            Reset
          </button>
          <button className="seg" aria-pressed={full} onClick={toggleFull} title={full ? "Exit full screen (F / Esc)" : "Full screen (F)"}>
            {full ? "Exit full screen" : "Full screen"}
          </button>
        </span>
      </div>

      {full && situation && (
        <div className="n3-hud n3-strip" style={{ borderTopColor: levelColour(situation.level) }}>
          <span className="n3-strip-site">{siteName()}</span>
          <span className="n3-strip-level" style={{ color: levelColour(situation.level) }}>
            {String(situation.level ?? "NOMINAL").toUpperCase()}
          </span>
          <span className="n3-strip-risk">
            risk <b style={{ color: levelColour(situation.level) }}>{situation.risk != null ? situation.risk.toFixed(3) : "—"}</b>
          </span>
          {situation.technique && <span className="n3-strip-tech">{situation.technique}</span>}
        </div>
      )}

      {pinned && (
        <div className="n3-card">
          <HostCard node={pinned} info={info.get(pinned.id)} isAttacker={pinned.id === attacker} />
        </div>
      )}

      <div className="n3-hud n3-bl">
        {nodes.length} hosts · {lanes.length} links · drag to orbit · click a host to inspect
        {full ? " · F or Esc to exit" : " · F for full screen"}
      </div>
    </div>
  );
}
