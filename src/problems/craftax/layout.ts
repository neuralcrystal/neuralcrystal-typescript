// What a Craftax crystal's file says about its pictures and its readout (the model's tr.craftax block), and the input pictures built
// from it: the board picture plus the memory, written as 4-bit levels onto the memory mirrors.
import type { ModelSpec } from "../../crystal/types.js";

/** an output region: one cell of the 64 × 64 grid, or a block of cells whose light is summed */
export type CraftaxCell = number | number[];

export interface CraftaxLayout {
  /** the picture's side, F */
  F: number;
  /** memory units (each a 4-bit level, 0 … 1) */
  units: number;
  /** per unit, its 4 bits (high bit first), each one mirror or a block of mirrors */
  mem: CraftaxCell[][];
  /** every memory mirror (cleared before the memory is drawn, single-head crystals) */
  memAll: number[];
  /** the readout's regions: the moves (43, or 44 with a TARGET pseudo-move), the memory gates z and new values hc, and optional where-map regions */
  cells: { act: CraftaxCell[]; z: CraftaxCell[]; hc: CraftaxCell[]; where?: CraftaxCell[] };
  /** the readout's trained offsets: move score = log light + gA; z, hc = sigmoid(al · (log light − mean) + be) */
  head: { gA: number[]; al: [number[], number[]]; be: [number[], number[]] };
  /** split crystals: each input head's picture gathered from the board picture (-1: dark) */
  split: { test: string | null; mapA: Int32Array; mapB: Int32Array } | null;
}

interface CraftaxBlock {
  frame: number; units: number; mem: CraftaxCell[][]; memAll?: number[];
  cells: CraftaxLayout["cells"]; e2eHead: CraftaxLayout["head"];
  split?: { test?: string; mapA: number[]; mapB: number[] };
}

/** The crystal file's tr.craftax block, checked. Throws on a crystal that isn't an end-to-end Craftax crystal. */
export function craftaxLayout(model: ModelSpec): CraftaxLayout {
  const cx = ((model["tr"] as Record<string, unknown> | undefined)?.["craftax"] ?? null) as CraftaxBlock | null;
  if (!cx?.cells || !cx.e2eHead) throw new Error("not a Craftax crystal (no tr.craftax with cells and e2eHead)");
  if (cx.split && !model.quad?.split) throw new Error("tr.craftax.split without quad.split");
  const memAll = cx.memAll ?? cx.mem.flat(2);
  return {
    F: cx.frame, units: cx.units, mem: cx.mem, memAll, cells: cx.cells, head: cx.e2eHead,
    split: cx.split ? { test: cx.split.test ?? null, mapA: Int32Array.from(cx.split.mapA), mapB: Int32Array.from(cx.split.mapB) } : null,
  };
}

/** a level 0 … 1 → its 4-bit step 0 … 15 */
export const memLevel = (v: number): number => Math.round(Math.min(1, Math.max(0, v)) * 15);

/** each unit's 4 bits (high bit first) onto its mirrors, zeros included */
function drawMemory(L: CraftaxLayout, pic: Float32Array, memory: ArrayLike<number>): void {
  for (let u = 0; u < L.units; u++) {
    const q = memLevel(memory[u] ?? 0);
    for (let j = 0; j < 4; j++) {
      const bit = (q >> (3 - j)) & 1;
      for (const px of ([] as number[]).concat(L.mem[u]![j]!)) pic[px] = bit;
    }
  }
}

/**
 * The input pictures from the board picture (F × F) and the memory (units numbers, 0 … 1). A split crystal gets two: the first head's
 * (the board and its key fields, gathered by mapA) and the second's (the slower fields, gathered by mapB, with the memory written in).
 * A single-head crystal gets one: the board with the memory written in.
 */
export function pictures(L: CraftaxLayout, board: ArrayLike<number>, memory: ArrayLike<number>): Float32Array[] {
  const n = L.F * L.F;
  if (L.split) {
    const a = new Float32Array(n), b = new Float32Array(n), { mapA, mapB } = L.split;
    for (let i = 0; i < n; i++) {
      if (mapA[i]! >= 0) a[i] = board[mapA[i]!]!;
      if (mapB[i]! >= 0) b[i] = board[mapB[i]!]!;
    }
    drawMemory(L, b, memory);
    return [a, b];
  }
  const fr = Float32Array.from({ length: n }, (_, i) => board[i]!);
  for (const p of L.memAll) fr[p] = 0;
  drawMemory(L, fr, memory);
  return [fr];
}
