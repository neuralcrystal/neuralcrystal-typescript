// The transfer function of a hop: band-limited angular spectrum in the glass (Matsushima's limit), built in float64 and stored
// as interleaved float32 (re, im) — the Python package's Crystal.H, term for term.
import type { Geometry } from "./geometry.js";

const TAU = 2 * Math.PI;

/** The FFT's frequency of index i on an N grid, cycles/µm. */
export const freqOf = (i: number, N: number, df: number): number => (i < N / 2 ? i : i - N) * df;

/** H for a hop of zUm µm: N×N complex, row-major (row = fy), interleaved [re, im]. */
export function transferH(g: Geometry, zUm: number): Float32Array {
  const { N, df, inv } = g;
  const inv2 = inv * inv;
  const fLim = inv / Math.sqrt((2 * df * Math.abs(zUm)) ** 2 + 1);
  const out = new Float32Array(2 * N * N);
  for (let iy = 0; iy < N; iy++) {
    const fy = freqOf(iy, N, df);
    for (let ix = 0; ix < N; ix++) {
      const fx = freqOf(ix, N, df);
      const f2 = fx * fx + fy * fy;
      if (!(f2 < inv2 && Math.sqrt(f2) <= fLim)) continue;
      const ph = TAU * zUm * Math.sqrt(inv2 - f2);
      const k = 2 * (iy * N + ix);
      out[k] = Math.cos(ph);
      out[k + 1] = Math.sin(ph);
    }
  }
  return out;
}

/**
 * H × the DMD relay's pupil (the Python package's dmd_pupil): zero where fx² + fy² > (NA/λ₀)², the test done in float32 as torch does it
 * (fftfreq's f = i · (1/(N dx)) in float32, the squares and their sum rounded to float32, the bound a float32 scalar).
 */
export function pupilH(g: Geometry, zUm: number): Float32Array {
  const { N } = g;
  const out = transferH(g, zUm);
  const f32 = Math.fround, step = f32(1 / (N * g.dx)), lim = f32((g.dmdNA / g.lam) ** 2);
  for (let iy = 0; iy < N; iy++) {
    const fy = f32((iy < N / 2 ? iy : iy - N) * step), fy2 = f32(fy * fy);
    for (let ix = 0; ix < N; ix++) {
      const fx = f32((ix < N / 2 ? ix : ix - N) * step);
      if (f32(f32(fx * fx) + fy2) <= lim) continue;
      const k = 2 * (iy * N + ix);
      out[k] = 0;
      out[k + 1] = 0;
    }
  }
  return out;
}

/** Transfer functions by hop length, built once each (keys rounded to the nanometre, as the Python package's H_cached). */
export class TransferCache {
  private readonly map = new Map<number, Float32Array>();
  constructor(private readonly g: Geometry) {}
  get(zUm: number): Float32Array {
    const key = Math.round(zUm * 1000) / 1000;
    let h = this.map.get(key);
    if (!h) {
      h = transferH(this.g, key);
      this.map.set(key, h);
    }
    return h;
  }
  private readonly pup = new Map<number, Float32Array>();
  /** the same with the DMD relay's pupil folded in (pupilH) */
  getPupil(zUm: number): Float32Array {
    const key = Math.round(zUm * 1000) / 1000;
    let h = this.pup.get(key);
    if (!h) {
      h = pupilH(this.g, key);
      this.pup.set(key, h);
    }
    return h;
  }
  get size(): number {
    return this.map.size;
  }
}
