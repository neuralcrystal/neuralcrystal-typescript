// Phase maps: the fixed kinds (the Python package's build_phase) and a learned map spread over the grid.
import type { LensArraySpec } from "../crystal/types.js";
import type { Geometry } from "./geometry.js";
import { rnd } from "./geometry.js";

const TAU = 2 * Math.PI;
const PI = Math.PI;

/** Kinds that carry no fixed phase: learned maps, nl surfaces, unetched glass, trace-only probes. */
export const NON_FIXED = new Set(["learned", "nl", "glass", "probe"]);

/** A fixed kind's phase on the N×N grid, radians, float32 (computed in float64). */
export function fixedPhase(g: Geometry, kind: string, param: number, arr: LensArraySpec | null = null): Float32Array {
  const { N, dx, lam, nmed } = g;
  const c = (N - 1) / 2;
  const W = N * dx * g.winFrac; // the window's width, µm
  const lens = (r2: number, fMm: number): number => (-PI * nmed * r2) / (lam * fMm * 1000);
  const out = new Float32Array(N * N);
  const f = phaseFn(kind, param, arr, { W, lens, nmed, lam });
  for (let iy = 0; iy < N; iy++) {
    const Y = (iy - c) * dx;
    for (let ix = 0; ix < N; ix++) out[iy * N + ix] = f((ix - c) * dx, Y);
  }
  return out;
}

interface Ctx {
  W: number;
  lens: (r2: number, fMm: number) => number;
  nmed: number;
  lam: number;
}

function phaseFn(kind: string, param: number, arr: LensArraySpec | null, { W, lens, nmed, lam }: Ctx): (X: number, Y: number) => number {
  switch (kind) {
    case "flat":
      return () => 0;
    case "lens":
      return (X, Y) => lens(X * X + Y * Y, param);
    case "diverge":
      return (X, Y) => -lens(X * X + Y * Y, param);
    case "grating":
      return (X) => TAU * (X / param - Math.floor(X / param));
    case "axicon":
      return (X, Y) => (-TAU * Math.sqrt(X * X + Y * Y)) / param;
    case "vortex":
      return (X, Y) => rnd(param) * Math.atan2(Y, X);
    case "cylx":
      return (X) => lens(X * X, param);
    case "lensarr":
      return lensArray(param, arr, W, lens);
    case "bugeye": {
      const L = (3 * param * 1000 * lam) / (nmed * W);
      return (X, Y) => lens(X * X + Y * Y, param) + 1.435 * (Math.cos((TAU * X) / L) + Math.cos((TAU * Y) / L));
    }
    case "bugeye2": {
      const L = (4 * param * 1000 * lam) / (nmed * W);
      return (X, Y) => {
        const sx = Math.floor((2 * X) / L) & 1;
        const sy = Math.floor((2 * Y) / L) & 1;
        return lens(X * X + Y * Y, param) + PI * ((sx + sy) & 1);
      };
    }
    case "lens3x3":
    case "lens2x2":
    case "lens12x12":
    case "zern3x3": {
      const w = W / (kind === "lens12x12" ? 12 : kind === "lens2x2" ? 2 : 3);
      const r = ((lam / nmed) * param * 1000) / w;
      return (X, Y) => {
        const u = X - ((Math.floor((X + W / 2) / w) + 0.5) * w - W / 2);
        const q = Y - ((Math.floor((Y + W / 2) / w) + 0.5) * w - W / 2);
        const v = lens(u * u + q * q, param);
        return kind === "zern3x3" && u * u + q * q <= r * r ? v + PI / 2 : v;
      };
    }
    default:
      throw new Error(`plane kind not supported by the engine: ${kind}`);
  }
}

function lensArray(param: number, arr: LensArraySpec | null, W: number, lens: Ctx["lens"]): (X: number, Y: number) => number {
  const a = arr ?? {};
  const nx = Math.max(1, a.nx || 1);
  const ny = Math.max(1, a.ny || 1);
  const P = Math.max(1e-3, (a.pitch_mm ?? a.pitch ?? 0) || W / 1000 / Math.max(nx, ny)) * 1000;
  const cells = Array.isArray(a.cells) && a.cells.length === nx * ny ? a.cells : null;
  const clamp = (v: number, hi: number): number => Math.min(hi, Math.max(0, v));
  return (X, Y) => {
    const ix = clamp(roundHalfEven(X / P + (nx - 1) / 2), nx - 1);
    const iy = clamp(roundHalfEven(Y / P + (ny - 1) / 2), ny - 1);
    const u = X - (ix - (nx - 1) / 2) * P;
    const w = Y - (iy - (ny - 1) / 2) * P;
    const fc = cells ? cells[iy * nx + ix] : param;
    return fc !== null && fc !== undefined && fc > 0 ? lens(u * u + w * w, fc) : 0;
  };
}

/** torch.round: halves to even. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** an 8-bit step k as its phase in (−π, π], exactly as unpackMap makes it */
export const stepPhase = (k: number): number => ((k > 128 ? k - 256 : k) * 2 * Math.PI) / 256;

/** a map held as 8-bit steps (CrystalData.steps, LoadOptions.stepMaps) rather than radians */
export const isSteps = (map: ArrayLike<number>): boolean => map instanceof Uint8Array;

/** A learned n×n map on the N×N grid: each value fills its (N/n)² block of samples. (A map of 8-bit steps comes back in radians, new.) */
export function spreadMap(map: Float32Array, n: number, N: number): Float32Array {
  const steps = isSteps(map);
  if (n === N && !steps) return map;
  const q = N / n;
  if (!Number.isInteger(q)) throw new Error(`feature map ${n}² does not tile the ${N}² grid`);
  const out = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    const src = Math.floor(y / q) * n;
    for (let x = 0; x < N; x++) {
      const v = map[src + Math.floor(x / q)]!;
      out[y * N + x] = steps ? stepPhase(v) : v;
    }
  }
  return out;
}

/** a learned map's square window (x0, x0) … (x0 + w) on the N×N grid, in radians, written into `out` (w²) — no N² array made */
export function mapWindow(map: Float32Array, n: number, N: number, x0: number, w: number, out: Float32Array): Float32Array {
  const q = N / n, steps = isSteps(map);
  for (let y = 0; y < w; y++) {
    const src = Math.floor((x0 + y) / q) * n;
    for (let x = 0; x < w; x++) {
      const v = map[src + Math.floor((x0 + x) / q)]!;
      out[y * w + x] = steps ? stepPhase(v) : v;
    }
  }
  return out;
}

/** a learned map as N×N 8-bit steps (a map of steps already that size is itself) — the phone's GPU upload (phase8) */
export function spreadSteps(map: Uint8Array, n: number, N: number): Uint8Array {
  if (n === N) return map;
  const q = N / n, out = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) {
    const src = Math.floor(y / q) * n;
    for (let x = 0; x < N; x++) out[y * N + x] = map[src + Math.floor(x / q)]!;
  }
  return out;
}
