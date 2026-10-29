// The checkers rules, picture and move reader in one object, so the engine plays by exactly the rules the crystal was trained and scored
// on. English draughts: squares 0–31 in reading order over the dark squares, row 0 at the top; black (+1) starts on 0–11 and moves first,
// down; captures are forced, one jump a move; a man reaching the far row is crowned and ends the turn; a side with no move loses; 40
// moves with no capture or man move is a draw. The pieces live in rules/.
import type { RenderOptions } from "./rules/frame.js";
import { blank, frame, glyph, GLYPH_OF, render, sqAtPx, sqRect } from "./rules/frame.js";
import { key } from "./rules/key.js";
import type { ViewMove } from "./rules/move-map.js";
import { cellMove, legalCells, moveMap, viewMoves } from "./rules/move-map.js";
import type { CheckersMove, CheckersStatus } from "./rules/moves.js";
import { fromUci, make, moves, status, uci } from "./rules/moves.js";
import type { CheckersState } from "./rules/position.js";
import { clone, flip, parse, start, str, VAL } from "./rules/position.js";
import type { MoveReading, ReadOptions } from "./rules/read.js";
import { readMoveMap } from "./rules/read.js";

export type { CheckersState } from "./rules/position.js";
export type { CheckersMove, CheckersStatus } from "./rules/moves.js";
export type { MoveCandidate, MoveReading, ReadOptions } from "./rules/read.js";
export type { RenderOptions } from "./rules/frame.js";

type Board = {
  start(): CheckersState;
  parse(s: string): CheckersState;
  str(st: CheckersState): string;
  moves(st: CheckersState): CheckersMove[];
  make(st: CheckersState, m: CheckersMove): CheckersState;
  status(st: CheckersState): CheckersStatus;
  uci(m: CheckersMove | null): string | null;
  fromUci(st: CheckersState, s: string): CheckersMove | null;
  render(st: CheckersState, opts?: RenderOptions): Float32Array;
  legalCells(st: CheckersState): Map<number, CheckersMove>;
  readMoveMap(p: ArrayLike<number>, st: CheckersState, opts?: ReadOptions): MoveReading | null;
  flip(sq: number): number;
  readonly SIZE: number;
  readonly MOVE_CELLS: number;
  /** drawing helpers: a cell's size and a glyph's inset, a blank frame, a glyph written into one, a piece's art (canonical: + the side
   *  to move), a square's rect [x, y, w, h] in frame pixels, the square under a frame pixel (−1: none) */
  readonly TILE: number;
  readonly INS: number;
  blank(size?: number): Float32Array;
  glyph(v: Float32Array, sq: number, art: string[]): void;
  GLYPH_OF(p: number): string[] | null;
  sqRect(sq: number): [number, number, number, number];
  sqAtPx(x: number, y: number): number;
  clone(st: CheckersState): CheckersState;
  /** a position's identity (repetition), the legal moves with their canonical cells, a cell's squares, material values */
  key(st: CheckersState): string;
  viewMoves(st: CheckersState, moves?: CheckersMove[]): ViewMove[];
  cellMove(cell: number): { from: number; to: number };
  VAL: { man: number; king: number };
};

export const checkers: Board = {
  start,
  parse,
  str,
  moves,
  make,
  status,
  uci,
  fromUci,
  render,
  legalCells,
  readMoveMap,
  flip,
  // the frame and the move map can be resized (render's size option does so for the one call), so these read the current values
  get SIZE() {
    return frame.SIZE;
  },
  get MOVE_CELLS() {
    return moveMap.CELLS;
  },
  get TILE() {
    return frame.L.TILE;
  },
  get INS() {
    return frame.L.INS;
  },
  blank,
  glyph,
  GLYPH_OF,
  sqRect,
  sqAtPx,
  clone,
  // the key is a 53-bit number; the type says string, as it always has, since callers only compare and store it
  key: key as unknown as (st: CheckersState) => string,
  viewMoves,
  cellMove,
  VAL,
};
