// A CIFAR picture as the DMD shows it (what the CIFAR crystals trained on):
//  · global contrast normalisation: the picture re-centred on mid-grey, its spread (over all 3,072 values, n − 1) set to `gcn` of full scale;
//  · four planes: R, G, B and the "dark" plane 255 − Y (Y = 0.299 R + 0.587 G + 0.114 B);
//  · the "mega" code: a picture pixel is 2R × 2R frame pixels (R = frame / 64), four R × R regions — red top-left, green top-right, blue
//    bottom-left, the fourth plane bottom-right — each region an area-dither cell: q = round(v · R² · (levels − 1)) quanta spread over its
//    R² cells in Bayer order, each cell 0 … levels − 1 (amplitudes in steps of 1 / (levels − 1)). Rounding half to even, as torch.round.
import type { CifarPicture } from "./dataset.js";

export interface CifarFrameSpec {
  frame: number;
  levels: number;
  gcn: number;
  p4: "dark" | "luma" | "carrier" | "none";
}

/** the R × R Bayer order (cifarframe.ranks) */
export function ranks(R: number): number[][] {
  let M = [[0]], s = 1;
  while (s < R) {
    const N: number[][] = [];
    for (let y = 0; y < 2 * s; y++) {
      N.push([]);
      for (let x = 0; x < 2 * s; x++) N[y]!.push(4 * M[y % s]![x % s]! + [0, 2, 3, 1][(y >= s ? 2 : 0) + (x >= s ? 1 : 0)]!);
    }
    M = N;
    s *= 2;
  }
  return M;
}

/** round half to even (torch.round) */
function roundEven(x: number): number {
  const f = Math.floor(x), d = x - f;
  return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1;
}

/** the four planes, 0..255 floats, [4][32 × 32] — after the contrast normalisation */
export function cifarPlanes(p: CifarPicture, gcn: number, p4: CifarFrameSpec["p4"]): Float64Array[] {
  const x = new Float64Array(3072);
  for (let i = 0; i < 3072; i++) x[i] = p[i]!;
  if (gcn) {
    let mu = 0;
    for (let i = 0; i < 3072; i++) mu += x[i]!;
    mu /= 3072;
    let v = 0;
    for (let i = 0; i < 3072; i++) v += (x[i]! - mu) ** 2;
    const sd = Math.max(1e-3, Math.sqrt(v / 3071));
    for (let i = 0; i < 3072; i++) x[i] = Math.min(255, Math.max(0, 127.5 + (255 * gcn * (x[i]! - mu)) / sd));
  }
  const P = [0, 1, 2, 3].map(() => new Float64Array(1024));
  for (let k = 0; k < 1024; k++) {
    const r = x[3 * k]!, g = x[3 * k + 1]!, b = x[3 * k + 2]!, Y = 0.299 * r + 0.587 * g + 0.114 * b;
    P[0]![k] = r;
    P[1]![k] = g;
    P[2]![k] = b;
    P[3]![k] = p4 === "luma" ? Y : p4 === "dark" ? 255 - Y : p4 === "carrier" ? 255 : 0;
  }
  return P;
}

/** the frame: Float32Array(frame²) of amplitudes 0..1 (cifarframe.mega_frames) */
export function cifarFrame(p: CifarPicture, s: CifarFrameSpec): Float32Array {
  const F = s.frame, R = F / 64;
  if (!Number.isInteger(R) || R < 1) throw new Error(`cifar: frame ${F} is not a multiple of 64`);
  const P = cifarPlanes(p, s.gcn, s.p4), Lc = s.levels - 1, Q = R * R * Lc, rk = ranks(R), out = new Float32Array(F * F);
  for (let c = 0; c < 4; c++) {
    const ry = c >> 1, rx = c & 1;
    for (let y = 0; y < 32; y++)
      for (let x = 0; x < 32; x++) {
        const q = roundEven(Math.min(1, Math.max(0, P[c]![y * 32 + x]! / 255)) * Q), base = Math.floor(q / (R * R)), extra = q % (R * R);
        for (let j = 0; j < R; j++)
          for (let i = 0; i < R; i++) {
            const row = y * 2 * R + ry * R + j, col = x * 2 * R + rx * R + i;
            out[row * F + col] = (base + (rk[j]![i]! < extra ? 1 : 0)) / Lc;
          }
      }
  }
  return out;
}
