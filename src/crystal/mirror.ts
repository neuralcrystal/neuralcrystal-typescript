// A crystal mirrored left to right, for a DMD input mirrored left to right too: every
// learned surface's relief reversed across the grid. Light in glass obeys the
// same physics in a mirror, so this crystal fed the DMD picture mirrored gives exactly the original light, mirrored — the same reading on
// mirrored detector squares (the kit mirrors the frame and the squares when data.mirrored is set); a page shows the die mirrored back,
// so the pad looks as it always does. Fixed surfaces are kept only when they are mirror-symmetric already.
import type { CrystalData } from "./types.js";

/** fixed surface kinds that are their own mirror image left to right (centred lenses, flats, cones, a lens along x) */
const SYMMETRIC = new Set(["learned", "glass", "flat", "lens", "diverge", "axicon", "cylx", "nl", "probe"]);

export function mirrorCrystal(data: CrystalData): CrystalData {
  for (const p of data.model.planes) {
    for (const k of [p.kind, ...(p.kinds ?? [])])
      if (!SYMMETRIC.has(k)) throw new Error(`a ${k} surface is not its own mirror image: this crystal cannot be mirrored`);
  }
  const maps = data.maps.map((secs, i) =>
    secs.map((m) => {
      if (!m) return m;
      // a map of 8-bit steps stays steps (stepMaps): the same kind of array, mirrored
      const n = data.mapN[i] || Math.round(Math.sqrt(m.length)), out = new (m.constructor as new (n: number) => Float32Array)(m.length);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = m[y * n + (n - 1 - x)]!;
      return out;
    }),
  );
  return { ...data, maps, mirrored: true };
}

/** an F × F picture mirrored left to right */
export function mirrorFrame(a: Float32Array, F: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let y = 0; y < F; y++) for (let x = 0; x < F; x++) out[y * F + x] = a[y * F + (F - 1 - x)]!;
  return out;
}
