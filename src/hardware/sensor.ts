// The image sensor: the readout box binned to an F×F picture through the sensor's pixel map, and the imager's response —
// as the Python package's readout_map, _bin and imager do it.
import type { ImagerSpec, ModelSpec } from "../crystal/types.js";
import type { Geometry } from "../physics/geometry.js";
import type { Field } from "../light/field.js";

/** The IMX290's pixel pitch, µm. */
export const SENSOR_PITCH_UM = 2.9;

/** One sample column's share of each frame column: [frame column, weight][] per sample. */
export type ReadoutMap = [number, number][][];

/** Whether the readout reaches the die turned 180° (a relay of k, or per DMD mirror) — the Python package's rule. */
export function sensorTurned(sen: ImagerSpec["sen"]): boolean {
  return typeof sen === "number" || sen === "mirror1" || sen === "mirror2";
}

export function readoutMap(m: ModelSpec, g: Geometry, side: number, F: number): ReadoutMap {
  const sen = m.imager?.sen ?? "ideal";
  const mirrors = m.dmdMirrors ?? 0;
  let P = 0;
  if (sen === "bonded") P = SENSOR_PITCH_UM / g.dx;
  else if ((sen === "mirror1" || sen === "mirror2") && mirrors > 0) {
    const Pm = g.N / ((sen === "mirror2" ? 2 : 1) * mirrors);
    P = Pm > 1 ? Pm : 0;
  }
  const fx = sensorTurned(sen) ? (w: number) => F - 1 - w : (w: number) => w;
  const fw = side / F;
  const map: ReadoutMap = [];
  for (let x = 0; x < side; x++) {
    if (P <= 0) {
      // aligned pixels: the sample [x, x+1) shared between the frame pixels it overlaps, by length
      const w0 = Math.min(F - 1, Math.floor(x / fw));
      const w1 = Math.min(F - 1, Math.floor((x + 1 - 1e-9) / fw));
      if (w1 === w0) map.push([[fx(w0), 1]]);
      else {
        const cut = (w0 + 1) * fw;
        map.push([[fx(w0), cut - x], [fx(w1), x + 1 - cut]]);
      }
      continue;
    }
    const s = Math.floor(x / P);
    const a = s * P;
    const b = Math.min(side, (s + 1) * P);
    const list: [number, number][] = [];
    for (let w = Math.max(0, Math.floor(a / fw)); w < F && w * fw < b; w++) {
      const o = Math.min(b, (w + 1) * fw) - Math.max(a, w * fw);
      if (o > 1e-9) list.push([fx(w), o / (b - a)]);
    }
    map.push(list.length ? list : [[fx(Math.min(F - 1, Math.floor((x * F) / side))), 1]]);
  }
  return map;
}

/** Sum the readout box's intensity into an F×F picture through a map (rows and columns share it). */
export function binThrough(E: Field, x0: number, side: number, F: number, map: ReadoutMap): Float64Array {
  const N = E.N;
  const rows = new Float64Array(side * F); // each box row binned along x
  for (let y = 0; y < side; y++) {
    const base = (x0 + y) * N + x0;
    for (let x = 0; x < side; x++) {
      const k = base + x;
      const I = E.re[k]! * E.re[k]! + E.im[k]! * E.im[k]!;
      if (I === 0) continue;
      for (const [w, wt] of map[x]!) rows[y * F + w]! += I * wt;
    }
  }
  const out = new Float64Array(F * F);
  for (let y = 0; y < side; y++) for (const [h, wt] of map[y]!) for (let w = 0; w < F; w++) out[h * F + w]! += rows[y * F + w]! * wt;
  return out;
}

/** The camera's response: r = clip(g·ℓ − black, 0, 1)^γ, ℓ = a pixel's light over a fully-on pixel's. mode "off" passes v through. */
export function imagerResponse(m: ModelSpec, g: Geometry, v: Float64Array, F: number): Float64Array {
  const im = m.imager ?? {};
  if (im.mode === "off") return v;
  const full = (g.side * g.side) / (F * F);
  const gain = im.g ?? 1;
  const black = im.b ?? 0.05;
  const gm = im.gm ?? 1;
  return v.map((x) => Math.pow(Math.min(1, Math.max(0, (x / full) * gain - black)), gm));
}

