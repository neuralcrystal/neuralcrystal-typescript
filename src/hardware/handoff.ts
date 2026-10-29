// THE HAND-OFF BETWEEN STAGES (the Python package's Crystal.hidden, forward only): in a multi-stage crystal the camera photographs one
// stage's output box and the controller shows that picture on the DMD as the next stage's input. Here: the camera's picture, exposed (the
// fitted gain, or per-picture auto-exposure), through the imager's response, dithered to the DMD's levels, turned back if asked, then the
// board ghost, the stage residual and the text band — in that order. Every multi-stage crystal uses it: the four-pass quad crystals and
// Craftax's split crystals alike (the settings live under "quad" in the crystal file, where they first appeared).
import type { ModelSpec, QuadSpec } from "../crystal/types.js";
import type { Geometry } from "../physics/geometry.js";
import type { Field } from "../light/field.js";
import { flip180 } from "../light/field.js";
import { nearestIndex } from "./dmd.js";
import { roundHalfEven } from "../physics/phases.js";

export interface HandoffConfig {
  /** passes through the block */
  hops: number;
  /** the physical section each pass runs through */
  passSection: number[];
  /** DMD amplitude levels (mir² + 1 on the real part) */
  levels: number;
  gains: number[];
  norm: { t: number; s: number } | null;
  ghost: number;
  res: number;
  unrot: boolean;
  band: number;
  bandY0: number;
  /** the hidden picture's size when the input frame is finer (0 = the input's) */
  hidden: number;
}

export function handoffConfig(m: ModelSpec): HandoffConfig {
  const q: QuadSpec | null = m.quad && typeof m.quad === "object" ? m.quad : null;
  if (q?.bnd && q.bnd.length) throw new Error("this crystal uses quad.bnd, which the engine does not support");
  const passes = q?.passes && q.passes.length ? q.passes.map(Number) : null;
  const hops = passes ? passes.length : q ? Math.max(1, q.n || 4) : 1;
  const mir = Math.max(1, q?.mir || 4);
  const gains = [...(q?.gains ?? []).map(Number), ...Array<number>(hops).fill(1)].slice(0, Math.max(0, hops - 1));
  const nm = q?.norm;
  return {
    hops,
    passSection: passes ?? Array.from({ length: hops }, (_, k) => k),
    levels: Math.max(2, q?.levels || mir * mir + 1),
    gains,
    norm: nm && typeof nm === "object" ? { t: nm.t || 0, s: nm.s || 4 } : null,
    ghost: q?.ghost || 0,
    res: q?.res || 0,
    unrot: !!q?.unrot,
    band: q?.band || 0,
    bandY0: q?.bandY0 || 80,
    hidden: q?.hidden || 0,
  };
}

export interface HiddenInputs {
  /** the pass-1 frame (the ghost and the band re-inject it) */
  board: Float32Array;
  /** this section's own input picture (the residual adds it back) */
  stageInput: Float32Array;
}

/** The camera's raw picture: each F×F pixel the light of its block of the output box (nearest-neighbour blocks, the DMD's rule). */
export function cameraPicture(E: Field, g: Geometry, F: number): Float64Array {
  const { so, xo } = g;
  const fi = nearestIndex(so, F);
  const v = new Float64Array(F * F);
  const N = E.N;
  for (let y = 0; y < so; y++) {
    const row = fi[y]! * F;
    const base = (xo + y) * N + xo;
    for (let x = 0; x < so; x++) {
      const k = base + x;
      v[row + fi[x]!]! += E.re[k]! * E.re[k]! + E.im[k]! * E.im[k]!;
    }
  }
  return v;
}

/** Pass k's output → pass k+1's DMD picture, amplitudes on the level grid 0..1. */
export function handoffPicture(E: Field, m: ModelSpec, g: Geometry, q: HandoffConfig, F: number, k: number, inp: HiddenInputs): Float32Array {
  const v = cameraPicture(E, g, F);
  const full = (g.side * g.side) / (F * F);
  const black = m.imager?.b ?? 0.05;
  const gm = m.imager?.gm ?? 1;
  const n = F * F;
  const a = new Float64Array(n);
  if (q.norm) {
    let mu = 0;
    for (let i = 0; i < n; i++) mu += v[i]! / full;
    mu /= n;
    let va = 0;
    for (let i = 0; i < n; i++) va += (v[i]! / full - mu) ** 2;
    const sd = Math.sqrt(va / n + 1e-18);
    const shift = q.norm.t * mu;
    const den = Math.max(1e-9, mu + q.norm.s * sd - shift);
    for (let i = 0; i < n; i++) a[i] = (v[i]! / full - shift) / den - black;
  } else {
    const gain = q.gains[k] ?? 1;
    for (let i = 0; i < n; i++) a[i] = (v[i]! / full) * gain - black;
  }
  const L = q.levels - 1;
  const quant = (x: number): number => roundHalfEven(x * L) / L;
  let out: Float32Array = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = quant(Math.pow(Math.min(1, Math.max(0, a[i]!)), gm));
  if (q.unrot) flip180(out, F);
  if (q.ghost) for (let i = 0; i < n; i++) out[i] = Math.min(1, Math.max(0, out[i]! + (q.ghost / L) * inp.board[i]!));
  if (q.res) for (let i = 0; i < n; i++) out[i] = quant(Math.min(1, Math.max(0, out[i]! + q.res * inp.stageInput[i]!)));
  if (q.band) out = withBand(out, inp.board, F, q.bandY0, q.band / L);
  return out;
}

function withBand(out: Float32Array, board: Float32Array, F: number, y0: number, level: number): Float32Array {
  for (let i = y0 * F; i < F * F; i++) out[i] = board[i]! * level;
  return out;
}

/** @deprecated the old name of HandoffConfig */
export type QuadConfig = HandoffConfig;
/** @deprecated the old name of handoffConfig */
export const quadConfig = handoffConfig;
/** @deprecated the old name of handoffPicture */
export const hiddenPicture = handoffPicture;
