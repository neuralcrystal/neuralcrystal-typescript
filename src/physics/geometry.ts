// The grid, the window and the boxes, derived from the recipe exactly as the Python package's Crystal.__init__ does.
import type { ModelSpec } from "../crystal/types.js";

/** JavaScript's Math.round (halves go up) — the simulation's rounding, and what the Python package's rnd() reproduces. */
export const rnd = (x: number): number => Math.floor(x + 0.5);

export interface Geometry {
  N: number;
  /** sample pitch, µm */
  dx: number;
  /** wavelength in vacuum, µm */
  lam: number;
  /** refractive index of the glass */
  nmed: number;
  /** nmed / lam: the medium's wavenumber in cycles/µm */
  inv: number;
  /** frequency step 1/(N dx), cycles/µm */
  df: number;
  /** the window's share of the grid (1 unless the whole block is simulated) */
  winFrac: number;
  /** the input frame's box: side samples at x0 (ext % of the window) */
  side: number;
  x0: number;
  /** the readout's box (outScale < 1 shrinks it) */
  so: number;
  xo: number;
  /** absorbing rim: the window box that survives each hop, or null */
  absorb: { x0: number; size: number } | null;
  /** the longest hop still band-limited at the grid's own Nyquist, µm */
  zMax: number;
  /** longest sub-hop, µm (Infinity: hops run whole) */
  zCut: number;
  /** the input gap before the first plane, µm */
  inGapUm: number;
  /** the DMD relay's numerical aperture (dmdNA): its pupil passes |f| ≤ NA/λ₀ only; 0 = no pupil */
  dmdNA: number;
}

export function geometryOf(m: ModelSpec): Geometry {
  const N = m.N;
  const dx = m.dx_um;
  const lam = m.lam_um ?? 0.617;
  const nmed = m.n_medium ?? 1.4607;
  if (m.solveMirror && !m.sideBlack) throw new Error("reflecting sides (solveMirror) are not supported by the engine yet");
  const whole = !!m.wholeGlass;
  const winUm = m.winUm || N * dx;
  const winFrac = whole ? Math.min(1, winUm / (N * dx)) : 1;
  const ext = m.ext ?? 50;
  const side = Math.max(4, rnd((ext / 100) * winFrac * N));
  const x0 = rnd((N - side) / 2);
  const os = m.outScale || 1;
  const outScale = os > 0 && os < 1 ? os : 1;
  const so = outScale < 1 ? Math.max(4, rnd(side * outScale)) : side;
  const xo = rnd((N - so) / 2);
  let absorb: Geometry["absorb"] = null;
  if (m.absorbRim && whole && winFrac < 1) {
    const size = Math.max(1, rnd(N * winFrac));
    absorb = { x0: rnd((N - size) / 2), size };
  }
  const inv = nmed / lam;
  const q = (2 * inv * dx) ** 2 - 1;
  const zMax = q > 0 ? ((N * dx) / 2) * Math.sqrt(q) : Infinity;
  const subHop = m.subHopUm || 0;
  const zCut = subHop > 0 ? Math.min(zMax, subHop) : m.zMaxSplit ? zMax : Infinity;
  return { N, dx, lam, nmed, inv, df: 1 / (N * dx), winFrac, side, x0, so, xo, absorb, zMax, zCut, inGapUm: (m.inGap_mm ?? 0) * 1000, dmdNA: m.dmdNA ?? 0 };
}

/** A hop of z µm as the Python package's prop_z runs it: n equal sub-hops, each with the transfer function of round(z/n, 3). */
export function hopPlan(g: Geometry, zUm: number): { n: number; zSub: number } {
  const n = Math.max(1, Math.ceil(zUm / g.zCut));
  return { n, zSub: Math.round((zUm / n) * 1000) / 1000 };
}
