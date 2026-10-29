// CHESS (problem "chesswin"): a position on the DMD as the crystal was trained to see it — board.js's compact
// 96-px art (flat board, the side to move's clock lamp, the castling and en-passant lamps, the crystal's quartz on the side to move)
// with every pixel doubled to the 192-px frame — four passes through the quad crystal, and
// the move read off a 64 × 64 grid of detector cells on the last section's exit face: the row band is the from-square, the column band
// the to-square ("move" layout). Three bands of the diagonal (from = to, never a move) are the value
// readout: win, draw, loss for the side to move (tr.value). The crystal plays black, the colour it was trained on.
import type { CrystalData } from "../../crystal/types.js";
import type { CrystalEngine } from "../../engine/crystal-engine.js";
import type { RunResult } from "../../engine/types.js";
import type { Field } from "../../light/field.js";
import type { Panel } from "../../hardware/screen/panel.js";
import type { ProblemKit } from "../../kits/types.js";
import type { ChessMove, ChessReadOptions, ChessReading, ChessState } from "./board.js";
import { chess, mirror } from "./board.js";
import type { ChessScreenView } from "./screen.js";
import { paintChessScreen } from "./screen.js";

/** the move grid: 64 cells a side, 16/1024 of the window apart, each 2 × 4/1024 of it square, as the crystals were trained */
export const CHESS_GRID = { n: 64, pitch: 16 / 1024, half: 4 / 1024 };

export interface ChessAnswer extends ChessReading {
  /** all 4,096 cells' light (from × 64 + to), the value cells included */
  powers: Float64Array;
  /** the same with the value cells dark: the light the move is read from */
  movePowers: Float64Array;
  /**
   * the other color was to move, so the crystal saw the board's color-swapped mirror. The moves, squares and SAN in the answer are on the
   * real board; cell, top, bestCell, powers and movePowers stay on the crystal's own map (the mirrored board's squares).
   */
  mirrored: boolean;
}

export interface ChessKit extends ProblemKit<ChessState, ChessAnswer, ChessScreenView> {
  /** what the crystal file says about its frames */
  readonly art: { art96x2: boolean; lampClock: boolean; lampCrystal: boolean; flatBoard: boolean; playAs: "w" | "b" | null };
  /** the value readout's cells (win, draw, loss), or null: no thinking */
  readonly valueCells: number[][] | null;
  bands(engine: CrystalEngine): { lo: number[]; side: number };
  /** each legal move's cell on the crystal's move map, with the move on the real board (white's positions are read mirrored) */
  cells(state: ChessState): Map<number, ChessMove>;
  /** the 4,096 cells' light on an exit field */
  powers(engine: CrystalEngine, E: Field): Float64Array;
  /** the cells' light with the value cells taken out */
  moveP(p: Float64Array): Float64Array;
  /** read with explicit rules (the play temperature and shortlist) */
  readWith(engine: CrystalEngine, run: RunResult, state: ChessState, o: ChessReadOptions): ChessAnswer;
}

const rnd = (x: number): number => Math.round(x); // JS half up, as the crystals' detector cells were placed in training

/** tr.value → three lists of cells (win, draw, loss), or null: [lo, hi] = the diagonal cells 65·s, or a block */
export function chessValueCells(model: Record<string, unknown>, G = 64): number[][] | null {
  // chess.value (the published files), else tr.value (training's own files), as the Python package reads it
  const field = (k: string): unknown => (model[k] as Record<string, unknown> | undefined)?.["value"];
  const v = (field("chess") ?? field("tr") ?? null) as Record<string, unknown> | null;
  if (!v || typeof v !== "object") return null;
  const gs: number[][] = [];
  for (const k of ["win", "draw", "loss"]) {
    const r = v[k] as unknown;
    const g: number[] = [];
    if (Array.isArray(r) && r.length === 2) {
      const lo = Math.trunc(+r[0]), hi = Math.trunc(+r[1]);
      if (!(lo >= 0 && lo <= hi && hi < G)) return null;
      for (let s = lo; s <= hi; s++) g.push((G + 1) * s);
    } else if (r && typeof r === "object" && Array.isArray((r as { block?: number[] }).block)) {
      const [r0, c0, r1, c1] = (r as { block: number[] }).block.map((x) => Math.trunc(+x));
      if (!(r0! >= 0 && r0! <= r1! && r1! < G && c0! >= 0 && c0! <= c1! && c1! < G)) return null;
      for (let a = r0!; a <= r1!; a++) for (let b = c0!; b <= c1!; b++) g.push(a * G + b);
    } else return null;
    gs.push(g);
  }
  const all = gs.flat();
  return new Set(all).size === all.length ? gs : null;
}

/** the side to move's expected score from the value cells: {w, d, l, es = w + d/2} */
export function chessValueOf(p: ArrayLike<number>, g: number[][]): { w: number; d: number; l: number; es: number } {
  const V = g.map((cs) => cs.reduce((a, c) => a + Math.max(0, +(p[c] ?? 0)), 0)), S = Math.max(V[0]! + V[1]! + V[2]!, 1e-30);
  const w = V[0]! / S, d = V[1]! / S, l = V[2]! / S;
  return { w, d, l, es: w + 0.5 * d };
}

export function chessKit(data: CrystalData): ChessKit {
  const ch = (data.model["chess"] ?? {}) as { art96x2?: boolean; lampClock?: boolean; lampCrystal?: boolean; flatBoard?: boolean; playAs?: string; frame?: number };
  const art = { art96x2: !!ch.art96x2, lampClock: ch.lampClock !== false, lampCrystal: ch.lampCrystal !== false, flatBoard: !!ch.flatBoard, playAs: (ch.playAs === "w" || ch.playAs === "b" ? ch.playAs : null) as "w" | "b" | null };
  const F = ch.frame ?? 192;
  const valueCells = chessValueCells(data.model as unknown as Record<string, unknown>);
  const valueMask = new Uint8Array(4096);
  for (const c of valueCells?.flat() ?? []) valueMask[c] = 1;
  let cache: { key: string; b: { lo: number[]; side: number } } | null = null;
  const bands = (engine: CrystalEngine): { lo: number[]; side: number } => {
    const g = engine.geometry, key = `${g.N}/${g.winFrac}`;
    if (cache?.key !== key) {
      const w = g.N * g.winFrac, c = (g.N - 1) / 2, G = CHESS_GRID;
      const lo = Array.from({ length: G.n }, (_, d) => rnd(c + ((d - (G.n - 1) / 2) * G.pitch - G.half) * w));
      cache = { key, b: { lo, side: rnd(2 * G.half * w) } };
    }
    return cache.b;
  };
  const powers = (engine: CrystalEngine, E: Field): Float64Array => {
    const { lo, side } = bands(engine), N = E.N, n = CHESS_GRID.n, p = new Float64Array(n * n), col = new Float64Array(N);
    // separable: each from-row band's light per column, then summed over the to-column bands
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
  const moveP = (p: Float64Array): Float64Array => {
    if (!valueCells) return p;
    const q = p.slice();
    for (let c = 0; c < 4096; c++) if (valueMask[c]) q[c] = 0;
    return q;
  };
  /** the board as the crystal sees it: its own color always on move, so the other color's positions are shown color-swapped and mirrored */
  const view = (st: ChessState): ChessState => (art.playAs && st.turn !== art.playAs ? mirror(st) : st);
  /** a reading on the mirrored board, put back on the real one: each move flipped a1 ↔ a8 and matched to the real board's legal move */
  const unmirror = (r: ChessReading, st: ChessState, seen: ChessState): ChessReading => {
    const legal = chess.moves(st);
    const back = (m: ChessMove | null | undefined): ChessMove | null =>
      m ? (legal.find((x) => x.from === (m.from ^ 56) && x.to === (m.to ^ 56) && (x.promo ?? 0) === (m.promo ?? 0)) ?? null) : null;
    const move = back(r.move), best = back(chess.moveCells(seen).get(r.bestCell));
    return {
      ...r, move, uci: chess.uci(move), san: move ? chess.san(st, move) : null, state: move ? chess.make(st, move) : null,
      from: r.from >= 0 ? r.from ^ 56 : r.from, to: r.to >= 0 ? r.to ^ 56 : r.to,
      cand: r.cand?.map((c) => ({ ...c, move: back(c.move) })) ?? null, bestUci: chess.uci(best), bestSan: best ? chess.san(st, best) : null,
    };
  };
  const readWith = (engine: CrystalEngine, run: RunResult, state: ChessState, o: ChessReadOptions): ChessAnswer => {
    const p = powers(engine, run.exit), pm = moveP(p), seen = view(state);
    const r = chess.readMoveMap(pm, seen, o.temp || o.anyCell ? o : undefined);
    if (!r) throw new Error("chess: no legal move in this position (the game is over)");
    return { ...(seen === state ? r : unmirror(r, state, seen)), powers: p, movePowers: pm, mirrored: seen !== state };
  };
  const cells = (st: ChessState): Map<number, ChessMove> => {
    const seen = view(st), out = new Map<number, ChessMove>();
    if (seen === st) return chess.moveCells(st);
    const legal = chess.moves(st);
    for (const [c, m] of chess.moveCells(seen)) {
      const real = legal.find((x) => x.from === (m.from ^ 56) && x.to === (m.to ^ 56) && (x.promo ?? 0) === (m.promo ?? 0));
      if (real) out.set(c, real);
    }
    return out;
  };
  /** the frame the crystal sees: the compact art doubled (art96x2), or the full 192-px layout; the other color's positions mirrored (view) */
  const frame = (position: ChessState): Float32Array => {
    const st = view(position);
    const flat0 = chess.FLATBG;
    chess.FLATBG = art.flatBoard;
    const opts = { noise: 0, noClock: !art.lampClock, crystalSides: art.lampCrystal ? [st.turn] : [] };
    try {
      if (!art.art96x2 || F !== 192) {
        chess.layout(F);
        try {
          return chess.render(st, opts);
        } finally {
          chess.layout(96);
        }
      }
      const v = chess.render(st, opts), o = new Float32Array(192 * 192);
      for (let y = 0; y < 192; y++) {
        const r = (y >> 1) * 96, w = y * 192;
        for (let x = 0; x < 192; x++) o[w + x] = v[r + (x >> 1)]!;
      }
      return o;
    } finally {
      chess.FLATBG = flat0;
    }
  };
  return {
    id: "chess",
    label: "Chess",
    frameSize: F,
    art,
    valueCells,
    bands,
    cells,
    powers,
    moveP,
    readWith,
    frame,
    read(engine: CrystalEngine, run: RunResult, state: ChessState): ChessAnswer {
      return readWith(engine, run, state, {});
    },
    paint(panel: Panel, view: ChessScreenView): void {
      paintChessScreen(panel, view);
    },
  };
}
