// The chess rules, frame and move reader as one object: the engine plays by exactly the rules, and draws exactly the frames, the crystal
// was trained and scored on. Squares a1 = 0 … h8 = 63; pieces +1..+6 white P N B R Q K, negative black. The frame's geometry is module
// state (layout): the engine keeps it on the compact 96-px layout, the one the art96 crystals see (doubled to 192) and the one the
// player's 96 × 96 screen shows. The pieces live in ./rules/.
import type { ChessMove, ChessState } from "./rules/position.js";
import { copy, isLight, mirror, mirrorMove, parseFEN, PIECE, pieceChar, sqName, start } from "./rules/position.js";
import { fen, fromUci, inCheck, key, make, moves, san, uci } from "./rules/moves.js";
import { status } from "./rules/status.js";
import { layout, live, sqAt, sqRect } from "./rules/layout.js";
import type { Sprite } from "./rules/sprites.js";
import { drawText } from "./rules/text.js";
import { render } from "./rules/frame.js";
import type { ChessReading, ChessReadOptions } from "./rules/movemap.js";
import { MOVE_CELL, MOVE_CELLS, moveCells, PROMO_Q, readMoveMap } from "./rules/movemap.js";

export type { ChessMove, ChessState } from "./rules/position.js";
export type { ChessCandidate, ChessReading, ChessReadOptions } from "./rules/movemap.js";
export { mirror, mirrorMove };

export interface ChessStatus {
  over: boolean;
  /** "1-0", "0-1", "1/2-1/2" */
  result?: string;
  why?: string;
  check?: boolean;
}

export interface RenderOptions {
  noise?: number;
  noClock?: boolean;
  /** the crystal's quartz mark on these sides ("w" / "b") */
  crystalSides?: ("w" | "b")[];
  /** the game-over word in the top strip */
  result?: string | null;
  moves?: ChessMove[];
}

type Board = {
  start(): ChessState;
  parseFEN(fen: string): ChessState;
  fen(st: ChessState): string;
  key(st: ChessState): string;
  copy(st: ChessState): ChessState;
  moves(st: ChessState): ChessMove[];
  make(st: ChessState, m: ChessMove): ChessState;
  status(st: ChessState, keys?: string[]): ChessStatus;
  inCheck(st: ChessState): boolean;
  san(st: ChessState, m: ChessMove, moves?: ChessMove[]): string;
  uci(m: ChessMove | null): string | null;
  fromUci(st: ChessState, u: string): ChessMove | null;
  render(st: ChessState, opts?: RenderOptions): Float32Array;
  readMoveMap(p: ArrayLike<number>, st: ChessState, opts?: ChessReadOptions): ChessReading | null;
  moveCells(st: ChessState, opts?: { moves?: ChessMove[] }): Map<number, ChessMove>;
  MOVE_CELL(m: ChessMove): number;
  MOVE_CELLS: number;
  PROMO_Q: number;
  layout(size: number): Board;
  drawText(v: Float32Array, text: string, cx: number, cy: number, scale?: number): Float32Array;
  sqRect(sq: number): [number, number, number, number];
  sqAt(x: number, y: number): number;
  sqName(sq: number): string;
  isLight(sq: number): boolean;
  pieceChar(p: number): string;
  PIECE: string;
  SPRITES: Record<string, Sprite>;
  SIZE: number;
  TILE: number;
  SPR: number;
  ORG: number;
  ORGY: number;
  FLATBG: boolean;
};

export const chess: Board = {
  start,
  parseFEN,
  fen,
  key,
  copy,
  moves,
  make,
  // the status carries null result / why while the game goes on (read as "not given")
  status: status as Board["status"],
  inCheck,
  san,
  uci,
  fromUci,
  render,
  readMoveMap,
  moveCells,
  MOVE_CELL,
  MOVE_CELLS,
  PROMO_Q,
  layout(size: number): Board {
    layout(size);
    return chess;
  },
  drawText,
  sqRect,
  sqAt,
  sqName,
  isLight,
  pieceChar,
  PIECE,
  // the live layout's values, read through
  get SPRITES() {
    return live.SPRITES;
  },
  get SIZE() {
    return live.SIZE;
  },
  get TILE() {
    return live.TILE;
  },
  get SPR() {
    return live.SPR;
  },
  get ORG() {
    return live.ORG;
  },
  get ORGY() {
    return live.ORGY;
  },
  get FLATBG() {
    return live.FLATBG;
  },
  set FLATBG(b: boolean) {
    live.FLATBG = b;
  },
};
chess.layout(96);

/** material from white's side, P1 N3 B3 R5 Q9 */
export function material(st: ChessState): number {
  const V = [0, 1, 3, 3, 5, 9, 0];
  let s = 0;
  for (let i = 0; i < 64; i++) {
    const p = st.sq[i]!;
    s += Math.sign(p) * V[Math.abs(p)]!;
  }
  return s;
}
