// Reading a Craftax crystal's answer off its exit faces. The light lands on a 64 × 64 grid of detector cells (the same grid as the chess
// crystals' move map, so the same bands); each move, memory gate and memory value is one cell or a block of cells, and its score is the
// log of that light plus a trained offset.
import type { Field } from "../../light/field.js";
import type { CraftaxCell, CraftaxLayout } from "./layout.js";

/** the cell grid: 64 cells a side, 16/1024 of the window apart, each 8/1024 of it square */
export const CRAFTAX_GRID = { n: 64, pitch: 16 / 1024, half: 4 / 1024 };

const rnd = (x: number): number => Math.round(x); // halves up, as the cells were placed

export interface CellBands {
  /** each cell's first sample on either axis */
  lo: number[];
  /** a cell's width in samples */
  side: number;
}

/** The cells' bands on the N × N grid: cell d covers samples [lo[d], lo[d] + side) on each axis. */
export function cellBands(N: number, winFrac: number): CellBands {
  const w = N * winFrac, c = (N - 1) / 2, G = CRAFTAX_GRID;
  const lo = Array.from({ length: G.n }, (_, d) => rnd(c + ((d - (G.n - 1) / 2) * G.pitch - G.half) * w));
  return { lo, side: rnd(2 * G.half * w) };
}

let cache: { key: string; b: CellBands } | null = null;

/** The light in each of the 4,096 cells (row × 64 + col) of an exit field. geometry: the engine's (N and the window fraction). */
export function cellPowers(geometry: { N: number; winFrac: number }, E: Field): Float64Array {
  const key = `${geometry.N}/${geometry.winFrac}`;
  if (cache?.key !== key) cache = { key, b: cellBands(geometry.N, geometry.winFrac) };
  const { lo, side } = cache.b, N = E.N, n = CRAFTAX_GRID.n, p = new Float64Array(n * n), col = new Float64Array(N);
  // separable: each row band's light per column, then summed over the column bands
  for (let f = 0; f < n; f++) {
    col.fill(0);
    for (let y = Math.max(0, lo[f]!); y < Math.min(N, lo[f]! + side); y++)
      for (let x = 0; x < N; x++) col[x]! += E.re[y * N + x]! ** 2 + E.im[y * N + x]! ** 2;
    for (let t = 0; t < n; t++) {
      let s = 0;
      for (let x = Math.max(0, lo[t]!); x < Math.min(N, lo[t]! + side); x++) s += col[x]!;
      p[f * n + t] = s;
    }
  }
  return p;
}

/** a region's log light: one cell, or a block of cells summed */
export function regionLog(p: ArrayLike<number>, c: CraftaxCell): number {
  if (typeof c === "number") return Math.log(p[c]! + 1e-12);
  let s = 0;
  for (const k of c) s += p[k]!;
  return Math.log(s + 1e-12);
}

export interface CraftaxReading {
  /** each move region's score (43, then TARGET on crystals that have one): log light + gA; a move's chance is the softmax */
  scores: Float64Array;
  /** the memory's update gates, 0 … 1 */
  z: Float64Array;
  /** the memory's new values, 0 … 1 */
  hc: Float64Array;
  /** where-map regions' scores, on crystals that have them */
  where: Float64Array | null;
}

const sig = (x: number): number => 1 / (1 + Math.exp(-x));

/**
 * The scores and the memory update from the cell light: pMoves from the moves' exit, pMemory from the memory's (the same array on a
 * single-head crystal). z and hc are sigmoid(al · (log light − the mean over all z and hc cells) + be).
 */
export function readCraftax(L: CraftaxLayout, pMoves: ArrayLike<number>, pMemory: ArrayLike<number> = pMoves): CraftaxReading {
  const { cells, head } = L, U = L.units, nA = cells.act.length;
  const scores = Float64Array.from(cells.act, (c, a) => regionLog(pMoves, c) + head.gA[a]!);
  const where = cells.where ? Float64Array.from(cells.where, (c, w) => regionLog(pMoves, c) + (head.gA[nA + w] ?? 0)) : null;
  const lz = cells.z.map((c) => regionLog(pMemory, c)), lh = cells.hc.map((c) => regionLog(pMemory, c));
  let mean = 0;
  for (let u = 0; u < U; u++) mean += lz[u]! + lh[u]!;
  mean /= 2 * U;
  const z = new Float64Array(U), hc = new Float64Array(U);
  for (let u = 0; u < U; u++) {
    z[u] = sig(head.al[0][u]! * (lz[u]! - mean) + head.be[0][u]!);
    hc[u] = sig(head.al[1][u]! * (lh[u]! - mean) + head.be[1][u]!);
  }
  return { scores, z, hc, where };
}
