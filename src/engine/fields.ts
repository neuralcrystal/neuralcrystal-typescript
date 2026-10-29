// Field and picture helpers the engine shares: cutting a field to a window and back, averaging a picture down, and the split
// crystal's per-pixel combine of its two input heads.
import type { Crop } from "../light/backend.js";
import type { Field } from "../light/field.js";
import type { CrystalData } from "../crystal/types.js";
import { roundHalfEven } from "../physics/phases.js";

/** an n × n single-precision field to keep a window in */
export function cropShape(n: number): Field {
  return { N: n, re: new Float32Array(n * n), im: new Float32Array(n * n) };
}

/** the square `c` of a field; out: an n × n single-precision field to write into instead of a new one */
export function cropField(f: Field, c: Crop, out?: Field): Field {
  const N = f.N, n = c.n, into = !!out && out.N === n && out.re instanceof Float32Array && out.im instanceof Float32Array;
  const re = into ? (out.re as Float32Array) : new Float32Array(n * n), im = into ? (out.im as Float32Array) : new Float32Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const k = (c.y0 + y) * N + c.x0 + x;
      re[y * n + x] = f.re[k]!;
      im[y * n + x] = f.im[k]!;
    }
  return { N: n, re, im };
}

/**
 * a cropped field back on the whole N grid, zeros outside its square (as it came, when it was never cropped); out: a whole-grid field to
 * write into (cleared first) instead of a new one
 */
export function uncrop(f: Field, c: (Crop & { N: number }) | undefined, N: number, out?: Field): Field {
  if (!c) return f;
  const into = !!out && out.N === N && out.re instanceof Float64Array && out.im instanceof Float64Array;
  const re = into ? (out.re as Float64Array) : new Float64Array(N * N), im = into ? (out.im as Float64Array) : new Float64Array(N * N);
  if (into) {
    re.fill(0);
    im.fill(0);
  }
  for (let y = 0; y < c.n; y++)
    for (let x = 0; x < c.n; x++) {
      const k = (c.y0 + y) * N + c.x0 + x;
      re[k] = f.re[y * c.n + x]!;
      im[k] = f.im[y * c.n + x]!;
    }
  return { N, re, im };
}

/** An F×F picture averaged down to Fh×Fh (F a whole multiple of Fh). */
export function blockMean(a: Float32Array, F: number, Fh: number): Float32Array {
  const q = F / Fh;
  if (!Number.isInteger(q)) throw new Error(`the input frame (${F}) must be a whole multiple of quad.hidden (${Fh})`);
  const out = new Float32Array(Fh * Fh);
  for (let y = 0; y < F; y++) for (let x = 0; x < F; x++) out[Math.floor(y / q) * Fh + Math.floor(x / q)]! += a[y * F + x]! / (q * q);
  return out;
}

/** the split crystal's combine as three F×F arrays a, b, c (quad.split.combine; absent: a = b = 0.75, c = 0) */
export function combineWeights(data: CrystalData, F: number): [Float32Array, Float32Array, Float32Array] {
  const cb = data.model.quad?.split?.combine ?? {}, n = F * F;
  const t = (v: number[] | null | undefined, d: number): Float32Array => {
    if (v && v.length !== n) throw new Error(`quad.split.combine: ${v.length} values for a ${F}×${F} picture`);
    return v ? Float32Array.from(v) : new Float32Array(n).fill(d);
  };
  return [t(cb.a, 0.75), t(cb.b, 0.75), t(cb.c, 0)];
}

/** the controller's per-pixel combine of the two heads' pictures: clip(a·x + b·y + c) to [0, 1], rounded to the DMD's `levels` (in float32) */
export function combinePictures(abc: [Float32Array, Float32Array, Float32Array], h0: Float32Array, h1: Float32Array, levels: number): Float32Array {
  const [ca, cb, cc] = abc, L = levels - 1, x = new Float32Array(h0.length);
  for (let i = 0; i < x.length; i++) {
    const v = Math.fround(Math.fround(Math.fround(ca[i]! * h0[i]!) + Math.fround(cb[i]! * h1[i]!)) + cc[i]!);
    x[i] = roundHalfEven(Math.min(1, Math.max(0, v)) * L) / L;
  }
  return x;
}
