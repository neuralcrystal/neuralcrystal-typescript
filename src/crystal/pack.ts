// THE WEB FORM OF A CRYSTAL: cropped, with 16 bit as an option. The same
// safetensors container and model JSON, but each surface map is
//   · cropped to its window — a window crystal's maps are exactly zero outside it (checkers ck56: every map zero outside its central
//     512 of 1024, a quarter of the samples), and
//   · stored as its phase wrapped to one turn in 16 (or 8) bits — exp(iφ) is all the light sees, and the glass is etched wrapped anyway;
//     16 bits is 0.0001 rad a step, 8 bits 0.025.
// ck56: 168 MB → 21 MB (16 bit) or 10.5 MB (8 bit). Trained phases look like noise, so gzip adds little on top (21 → 19.5 MB).
// The file says so in its "nc.pack" metadata; readCrystal expands it back to full maps (zeros outside the window).
import type { CrystalData } from "./types.js";
import { writeSafetensors } from "./safetensors.js";

export interface PackOptions {
  /** bits per phase sample */
  bits: 8 | 16;
}

/** one map's place in the packed file: its own side n, and the window kept (x0, y0, w × w) */
export interface PackedMap {
  n: number;
  x0: number;
  y0: number;
  w: number;
}

export interface PackMeta {
  format: "nc.pack.v1";
  bits: 8 | 16;
  maps: Record<string, PackedMap>;
}

const TAU = 2 * Math.PI;

/** the smallest square around the map's centre (a multiple of 8, within the map) that holds every nonzero sample */
function windowOf(m: Float32Array, n: number): { x0: number; y0: number; w: number } {
  let r = 0;
  const c = n / 2;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) if (m[y * n + x] !== 0) r = Math.max(r, Math.abs(x + 0.5 - c), Math.abs(y + 0.5 - c));
  let w = Math.min(n, Math.ceil((2 * r) / 8) * 8);
  if (w <= 0) w = Math.min(n, 8);
  const x0 = Math.max(0, Math.round(c - w / 2));
  return { x0, y0: x0, w };
}

/** a crystal in its web form: the bytes of a packed safetensors file */
export function packCrystal(data: CrystalData, o: PackOptions): Uint8Array {
  const Q = o.bits === 16 ? 65536 : 256;
  const tensors: [string, Uint16Array | Uint8Array, number[]][] = [];
  const meta: PackMeta = { format: "nc.pack.v1", bits: o.bits, maps: {} };
  data.maps.forEach((secs, i) => {
    const n = data.mapN[i] ?? 0;
    secs.forEach((m, k) => {
      if (!m || !n) return;
      const name = k === 0 ? `plane.${i}.phi` : `plane.${i}.phi.${k}`;
      const { x0, y0, w } = windowOf(m, n);
      const q = o.bits === 16 ? new Uint16Array(w * w) : new Uint8Array(w * w);
      for (let y = 0; y < w; y++)
        for (let x = 0; x < w; x++) {
          const phi = m[(y0 + y) * n + x0 + x]!, t = phi / TAU - Math.floor(phi / TAU); // one turn, [0, 1)
          q[y * w + x] = Math.round(t * Q) % Q;
        }
      tensors.push([name, q, [w, w]]);
      meta.maps[name] = { n, x0, y0, w };
    });
  });
  return writeSafetensors({ neuralcrystal: JSON.stringify(data.model), "nc.pack": JSON.stringify(meta) }, tensors);
}

/** an 8-bit packed map as its full n × n steps (phase k·2π/256), zeros outside its window — typed as a PhaseMap (CrystalData.steps) */
export function unpackSteps(bytes: Uint8Array, dtype: string, p: PackedMap): Float32Array {
  if (dtype !== "U8") throw new Error(`packed steps: dtype ${dtype}`);
  const out = new Uint8Array(p.n * p.n);
  for (let y = 0; y < p.w; y++) out.set(bytes.subarray(y * p.w, (y + 1) * p.w), (p.y0 + y) * p.n + p.x0);
  return out as unknown as Float32Array;
}

/** a packed map back to a full n × n phase map, radians in (-π, π], zeros outside its window */
export function unpackMap(bytes: Uint8Array, dtype: string, p: PackedMap, bits: 8 | 16): Float32Array {
  const Q = bits === 16 ? 65536 : 256, out = new Float32Array(p.n * p.n);
  const src = dtype === "U16" ? new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2) : bytes;
  if (dtype !== "U16" && dtype !== "U8") throw new Error(`packed map: dtype ${dtype}`);
  for (let y = 0; y < p.w; y++)
    for (let x = 0; x < p.w; x++) {
      let phi = (src[y * p.w + x]! / Q) * TAU;
      if (phi > Math.PI) phi -= TAU;
      out[(p.y0 + y) * p.n + p.x0 + x] = phi;
    }
  return out;
}
