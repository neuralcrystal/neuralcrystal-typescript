// The move map: the detector cells a move is read from, in canonical squares (the side to move at the top). A step, a first jump, a later
// jump of a chain and a king's move all land on the same kind of cell, so no special multi-jump output is needed.
//
// Two layouts. "fromto32", the default: 32 × 32 cells, cell = from × 32 + to, of which only 170 are ever a legal move. "board16": the map
// is the board, 16 × 16, each dark square holding its own eight moves as a 2 × 4 block: its back row (kings only) above, its forward row
// below (the side to move plays down the picture), [jump left, step left, step right, jump right] across. The two rows that could only
// hold moves off the board are dropped, so the moves fill rows 0–13; row 14 is a guard and row 15 holds the value cells. Every cell
// number goes through cellOf / cellMove, so the layout is one switch; a crystal names its own.
import type { CheckersMove } from "./moves.js";
import { jumpsFrom, moves, stepsFrom, uci } from "./moves.js";
import type { CheckersState } from "./position.js";
import { colOf, flip, rowOf } from "./position.js";

export type MoveLayout = "fromto32" | "board16";

/** the current layout, its grid side and its cell count */
export const moveMap: { layout: MoveLayout; GRID: number; CELLS: number } = { layout: "fromto32", GRID: 32, CELLS: 1024 };

/** a legal move with its canonical squares and its cell */
export interface ViewMove {
  m: CheckersMove;
  from: number;
  to: number;
  cell: number;
}

// the board16 tables, built the first time that layout is chosen: (from × 32 + to) -> cell, and cell -> the move's squares
const B16 = { cell: new Int16Array(32 * 32).fill(-1), move: [] as { from: number; to: number }[] };

/** a canonical (from, to) pair's cell, or −1 when the layout has none */
export const cellOf = (from: number, to: number): number => (moveMap.layout === "board16" ? B16.cell[from * 32 + to]! : from * 32 + to);
/** a canonical move's cell (already canonical when the move came from viewMoves) */
export const MOVE_CELL = (m: { from: number; to: number }): number => cellOf(m.from, m.to);
/** a cell's canonical squares ({ −1, −1 } for a cell that holds no move) */
export const cellMove = (cell: number): { from: number; to: number } =>
  moveMap.layout === "board16" ? B16.move[cell] || { from: -1, to: -1 } : { from: (cell / 32) | 0, to: cell % 32 };
export function cellName(cell: number): string {
  const m = cellMove(cell);
  return m.from < 0 ? `cell ${cell}` : `${m.from + 1}→${m.to + 1}`;
}

export function setMoveLayout(name: MoveLayout): void {
  if (name !== "board16" && name !== "fromto32") throw new Error(`unknown checkers move layout ${name as string}`);
  moveMap.layout = name;
  moveMap.GRID = name === "board16" ? 16 : 32;
  moveMap.CELLS = moveMap.GRID * moveMap.GRID;
  if (name !== "board16" || B16.move.length) return;
  for (let f = 0; f < 32; f++)
    for (const fwd of [true, false])
      for (const right of [false, true])
        for (const jump of [false, true]) {
          const d = jump ? 2 : 1, tr = rowOf(f) + (fwd ? d : -d), tc = colOf(f) + (right ? d : -d);
          if (tr < 0 || tr > 7 || tc < 0 || tc > 7) continue;
          let t = -1;
          for (let s = 0; s < 32; s++) if (rowOf(s) === tr && colOf(s) === tc) t = s;
          if (t < 0) continue;
          const y = fwd ? 2 * rowOf(f) : 2 * rowOf(f) - 1, x = 4 * (colOf(f) >> 1) + (right ? (jump ? 3 : 2) : jump ? 0 : 1);
          if (y < 0 || y > 13) continue;
          B16.cell[f * 32 + t] = y * 16 + x;
          B16.move[y * 16 + x] = { from: f, to: t };
        }
}

/** the legal moves (or the ones given) with their canonical squares and cells attached */
export function viewMoves(st: CheckersState, ms?: CheckersMove[]): ViewMove[] {
  const f = st.turn > 0 ? (s: number): number => s : flip;
  return (ms || moves(st)).map((m) => ({ m, from: f(m.from), to: f(m.to), cell: cellOf(f(m.from), f(m.to)) }));
}

// The moves that would be legal if captures were not compulsory and no chain were running: every step and every jump each of the side's
// pieces could make, judged one piece at a time. Never played from; it only tells apart the ways an illegal cell can be wrong (classify).
export function pseudoMoves(st: CheckersState): CheckersMove[] {
  const out: CheckersMove[] = [];
  for (let i = 0; i < 32; i++) {
    stepsFrom(st, i, out);
    jumpsFrom(st, i, out);
  }
  return out;
}

export function pseudoCells(st: CheckersState): Set<number> {
  const f = st.turn > 0 ? (x: number): number => x : flip, out = new Set<number>();
  for (const m of pseudoMoves(st)) {
    const c = cellOf(f(m.from), f(m.to));
    if (c >= 0) out.add(c);
  }
  return out;
}

export interface Classified {
  kind: "ok" | "legal-other" | "missed-capture" | "wrong-chain" | "noise";
  cell: number;
  from: number;
  to: number;
}

// What the brightest cell says: "ok" (a move the target asks for, by default any legal move), "legal-other" (a legal move, just not one
// asked for, or a pseudo-legal one when no capture was due), "missed-capture" (a quiet move while a jump existed), "wrong-chain" (mid-chain,
// but from the wrong piece) or "noise" (a move from an empty square, or off a diagonal).
export function classify(p: ArrayLike<number>, st: CheckersState, opts?: { moves?: CheckersMove[]; asked?: Set<number> }): Classified | null {
  const ent = viewMoves(st, opts && opts.moves);
  if (!ent.length) return null;
  const legal = new Set(ent.map((e) => e.cell));
  let top = 0;
  for (let c = 1; c < moveMap.CELLS; c++) if ((p[c] || 0) > (p[top] || 0)) top = c;
  const { from, to } = cellMove(top);
  const asked = opts && opts.asked ? opts.asked : legal;
  if (legal.has(top)) return { kind: asked.has(top) ? "ok" : "legal-other", cell: top, from, to };
  const pseudo = pseudoCells(st);
  if (st.jump >= 0) return { kind: pseudo.has(top) ? "wrong-chain" : "noise", cell: top, from, to };
  const jumpAvailable = ent.some((e) => e.m.cap >= 0);
  if (pseudo.has(top) && jumpAvailable) return { kind: "missed-capture", cell: top, from, to };
  return { kind: pseudo.has(top) ? "legal-other" : "noise", cell: top, from, to };
}

/** each legal cell and the move it plays */
export function legalCells(st: CheckersState, ms?: CheckersMove[]): Map<number, CheckersMove> {
  const out = new Map<number, CheckersMove>();
  for (const e of viewMoves(st, ms)) out.set(e.cell, e.m);
  return out;
}

/** 1 on every legal cell, 0 elsewhere */
export function legalMask(st: CheckersState, ms?: CheckersMove[]): Float32Array {
  const v = new Float32Array(moveMap.CELLS);
  for (const e of viewMoves(st, ms)) v[e.cell] = 1;
  return v;
}

/** a teacher's weights: by notation (11-15), or by cell when the keys are numbers */
export type MoveWeights = Map<string | number, number> | Record<string | number, number>;

// The teacher's target over the map, normalized, zero off the legal cells. Without weights it is uniform over the legal moves, and a
// teacher that liked nothing falls back to the same.
export function target(st: CheckersState, dist?: MoveWeights | null, ms?: CheckersMove[]): Float32Array {
  const v = new Float32Array(moveMap.CELLS), ent = viewMoves(st, ms);
  if (!ent.length) return v;
  let tot = 0;
  for (const e of ent) {
    const u = uci(e.m)!;
    const w = dist === undefined || dist === null ? 1 : dist instanceof Map ? (dist.get(u) ?? dist.get(e.cell) ?? 0) : (dist[u] ?? dist[e.cell] ?? 0);
    if (w > 0) {
      v[e.cell]! += w;
      tot += w;
    }
  }
  if (tot <= 0) {
    for (const e of ent) v[e.cell] = 1 / ent.length;
    return v;
  }
  for (let i = 0; i < v.length; i++) v[i]! /= tot;
  return v;
}

// Move scores (from this side's point of view) -> a soft distribution: a softmax at temperature T, with anything more than `cut` below
// the best dropped. T = 0 is one-hot on the best.
export function softTargets(scores: Record<string, number>, T: number, cut?: number | null): Record<string, number> {
  const ks = Object.keys(scores);
  if (!ks.length) return {};
  let best = -Infinity;
  for (const k of ks) if (scores[k]! > best) best = scores[k]!;
  const out: Record<string, number> = {};
  let Z = 0;
  for (const k of ks) {
    const d = scores[k]! - best;
    if (cut !== undefined && cut !== null && -d > cut) continue;
    const w = T > 0 ? Math.exp(d / T) : d === 0 ? 1 : 0;
    out[k] = w;
    Z += w;
  }
  if (Z <= 0) return { [ks[0]!]: 1 };
  for (const k in out) out[k]! /= Z;
  return out;
}
