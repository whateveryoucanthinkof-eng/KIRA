/**
 * The site under observation, as /api/site describes it (control_backend/
 * site_config.py, config/sites/*.yaml).
 *
 * The console used to decide "internal" with `ip.startsWith("10.")` and lay
 * the 3D view out on the Containerlab subnets. That is only true of the lab:
 * on the local-default site 192.168.0.0/16 is enterprise address space. Every
 * view asks here instead. Like the temporal contract, this is a live module
 * binding set once /api/site answers; until then RFC 1918 space counts as
 * internal.
 */

import type { SiteInfo } from "../api/types";

export type ZoneKey = "external" | "dmz" | "servers" | "users";

interface Cidr {
  base: number;
  mask: number;
}

let site: SiteInfo | null = null;
let enterprise: Cidr[] = [];
let external: Cidr[] = [];
let zones: { key: ZoneKey; cidrs: Cidr[]; raw: string[] }[] = [];
const names = new Map<string, string>();

const RFC1918 = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"].map(parse).filter(Boolean) as Cidr[];

function ipv4(ip: string): number | null {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ip.trim());
  if (!m) return null;
  const o = m.slice(1).map(Number);
  if (o.some((x) => x > 255)) return null;
  return ((o[0] << 24) | (o[1] << 16) | (o[2] << 8) | o[3]) >>> 0;
}

function parse(cidr: string): Cidr | null {
  const [addr, bits] = cidr.split("/");
  const base = ipv4(addr ?? "");
  const n = bits == null ? 32 : Number(bits);
  if (base == null || !Number.isInteger(n) || n < 0 || n > 32) return null;
  const mask = n === 0 ? 0 : (0xffffffff << (32 - n)) >>> 0;
  return { base: (base & mask) >>> 0, mask };
}

function inAny(ip: string, list: Cidr[]): boolean {
  const v = ipv4(ip);
  return v != null && list.some((c) => ((v & c.mask) >>> 0) === c.base);
}

/** Zone names in site YAML that map onto the console's bands. */
const ZONE_ALIASES: Record<string, ZoneKey> = {
  dmz: "dmz",
  servers: "servers",
  server: "servers",
  users: "users",
  user: "users",
  workstations: "users",
  external: "external",
};

export function setSite(s: SiteInfo | null | undefined): void {
  if (!s) return;
  site = s;
  enterprise = (s.enterprise_cidrs ?? []).map(parse).filter(Boolean) as Cidr[];
  external = (s.external_cidrs ?? []).map(parse).filter(Boolean) as Cidr[];
  zones = Object.entries(s.zones ?? {})
    .filter(([k]) => ZONE_ALIASES[k.toLowerCase()])
    .map(([k, cidrs]) => ({
      key: ZONE_ALIASES[k.toLowerCase()],
      cidrs: (cidrs ?? []).map(parse).filter(Boolean) as Cidr[],
      raw: cidrs ?? [],
    }));
  names.clear();
  for (const a of s.assets_of_interest ?? []) if (a?.ip && a?.name) names.set(a.ip, a.name);
}

/** Inside the site's enterprise address space. */
export function isInternal(ip: string | null | undefined): boolean {
  if (!ip) return false;
  if (inAny(ip, external)) return false;
  if (enterprise.length) return inAny(ip, enterprise);
  return inAny(ip, RFC1918);
}

/** The console band a host belongs to. */
export function zoneOfIp(ip: string): ZoneKey {
  for (const z of zones) if (inAny(ip, z.cidrs)) return z.key;
  if (!isInternal(ip)) return "external";
  // Internal but in no declared zone: the lower band, labelled "Internal"
  // when the site declares no zones at all.
  return "users";
}

/** True when the site declares the lab's dmz / servers / users zones. */
export function hasZones(): boolean {
  return zones.some((z) => z.key !== "external");
}

/** The configured name of a host of interest, or null. */
export function assetName(ip: string): string | null {
  return names.get(ip) ?? null;
}

/** The first configured DMZ asset: the 3D view's centre for that band. */
export function dmzAnchor(): string | null {
  const a = (site?.assets_of_interest ?? []).find((x) => x.role === "dmz");
  return a?.ip ?? null;
}

/** The CIDRs the site declares for a band, for labels. */
export function zoneCidrs(key: ZoneKey): string {
  const declared = zones.filter((z) => z.key === key).flatMap((z) => z.raw);
  if (declared.length) return declared.join(", ");
  if (key === "external") return external.length ? (site?.external_cidrs ?? []).join(", ") : "outside enterprise";
  if (key === "users" && !hasZones()) return (site?.enterprise_cidrs ?? []).join(", ") || "enterprise";
  return "—";
}

/** The site's display name (`display_name` in its YAML). */
export function siteName(): string {
  return site?.name || site?.site_id || "Site";
}
