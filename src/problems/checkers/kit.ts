// CHECKERS: a position on the DMD as the crystals were trained to see it (96 × 96: the 32 playable squares as corner brackets, the side
// to move as solid discs at the top), four passes through the quad crystal with the FPGA's auto-exposure between them, and the move read off a 32 × 32
// grid of detector cells on the last section's exit face — the row band is the from-square, the column band the to-square (the
// "move32" layout). The legal mask and the sampling (playT, topK) come from the crystal file's "checkers" block.
import type { CrystalData } from "../../crystal/types.js";
import type { CrystalEngine } from "../../engine/crystal-engine.js";
import type { RunResult } from "../../engine/types.js";
import type { Field } from "../../light/field.js";
import type { Panel } from "../../hardware/screen/panel.js";
import type { ProblemKit } from "../../kits/types.js";
import type { CheckersState, MoveReading, ReadOptions } from "./board.js";
import { checkers } from "./board.js";
import type { CheckersScreenView } from "./screen.js";
import { paintCheckersScreen } from "./screen.js";

/** the move grid: 32 cells a side, 32/1024 of the window apart, each 2 × 8/1024 of it square, as the crystals were trained */
export const MOVE_GRID = { n: 32, pitch: 32 / 1024, half: 8 / 1024 };

export interface CheckersAnswer extends MoveReading {
  /** all 1,024 cells' light (from × 32 + to), for the detector picture */
  powers: Float64Array;
}

export interface CheckersKit extends ProblemKit<CheckersState, CheckersAnswer, CheckersScreenView> {
  /** the reading rules the crystal file asks for (overridable per read) */
  readonly play: Required<Pick<ReadOptions, "temp" | "topK" | "anyCell">>;
  /** the cells' bands on the N grid: rows [lo, lo + side) and the same columns, cell d */
  bands(engine: CrystalEngine): { lo: number[]; side: number };
  /** the 1,024 cells' light on an exit field */
  powers(engine: CrystalEngine, E: Field): Float64Array;
  /** read with other rules than the file's (a deterministic brightest-legal read: { temp: 0 }) */
  readWith(engine: CrystalEngine, run: RunResult, state: CheckersState, o: ReadOptions): CheckersAnswer;
}

const rnd = (x: number): number => Math.round(x); // JS half up, as the crystals' detector cells were placed in training

export function checkersKit(data: CrystalData): CheckersKit {
  const ck = (data.model["checkers"] ?? {}) as { playT?: number; topK?: number; legalMask?: boolean; frame?: number };
  const play = { temp: Math.min(5, Math.max(0, ck.playT ?? 1)), topK: ck.topK ?? 5, anyCell: ck.legalMask === false };
  const F = ck.frame ?? 96;
  let cache: { key: string; b: { lo: number[]; side: number } } | null = null;
  const bands = (engine: CrystalEngine): { lo: number[]; side: number } => {
    const g = engine.geometry, key = `${g.N}/${g.winFrac}`;
    if (cache?.key !== key) {
      const w = g.N * g.winFrac, c = (g.N - 1) / 2, G = MOVE_GRID;
      const lo = Array.from({ length: G.n }, (_, d) => rnd(c + ((d - (G.n - 1) / 2) * G.pitch - G.half) * w));
      cache = { key, b: { lo, side: rnd(2 * G.half * w) } };
    }
    return cache.b;
  };
  const powers = (engine: CrystalEngine, E: Field): Float64Array => {
    const { lo, side } = bands(engine), N = E.N, n = MOVE_GRID.n, p = new Float64Array(n * n);
    // separable: each row band's light per column, then summed over the column bands
    const col = new Float64Array(N);
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
  };
  const readWith = (engine: CrystalEngine, run: RunResult, state: CheckersState, o: ReadOptions): CheckersAnswer => {
    const p = powers(engine, run.exit);
    const r = checkers.readMoveMap(p, state, { ...play, ...o });
    if (!r) throw new Error("checkers: no legal move in this position (the game is over)");
    return { ...r, powers: p };
  };
  return {
    id: "checkers",
    label: "Checkers",
    frameSize: F,
    play,
    bands,
    powers,
    readWith,
    frame(state: CheckersState): Float32Array {
      return checkers.render(state, F !== 96 ? { size: F } : undefined);
    },
    read(engine: CrystalEngine, run: RunResult, state: CheckersState): CheckersAnswer {
      return readWith(engine, run, state, {});
    },
    paint(panel: Panel, view: CheckersScreenView): void {
      paintCheckersScreen(panel, view);
    },
  };
}
