// The checkers position and the board's geometry. English draughts on the 32 dark squares, numbered 0–31 in reading order with row 0 at
// the top (the standard 1–32 of draughts, zero-based). A square is playable when (row + col) is odd, so row 0 holds cols 1, 3, 5, 7 and
// row 1 cols 0, 2, 4, 6.

/** a position: sq[32] (0 empty, ±1 man, ±2 king; + black), the side to move (+1 black, −1 white), the square that must jump on or −1 */
export interface CheckersState {
  sq: Int8Array;
  turn: 1 | -1;
  jump: number;
  /** plies since the last capture or man move (the 40-move draw) */
  idle: number;
}

/** what the frame draws: the pieces and the jumping square seen from the side to move (see view) */
export interface CanonicalView {
  sq: Int8Array;
  jump: number;
}

export const N = 32;
/** the four diagonals, as [drow, dcol] */
export const DIRS: readonly (readonly [number, number])[] = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
/** material values: a king is worth less than two men, since two men that can still crown are usually better */
export const VAL = { man: 100, king: 160 };

export const rowOf = (sq: number): number => sq >> 2;
export const colOf = (sq: number): number => 2 * (sq & 3) + (1 - (rowOf(sq) & 1));
/** the square at a row and column, or −1 off the board or on a light square */
export const sqAt = (r: number, c: number): number => (r < 0 || r > 7 || c < 0 || c > 7 || ((r + c) & 1) === 0 ? -1 : 4 * r + (c >> 1));
/** draughts names a square by its number: 1 … 32 */
export const sqName = (sq: number): string => String(sq + 1);
/** the 180° turn: row r col c goes to row 7−r col 7−c, which is exactly 31 − sq */
export const flip = (sq: number): number => 31 - sq;

// Black (+1) is the army that starts at the top, on squares 0–11, and moves down; white (−1) starts on 20–31 and moves up. Black moves first.
export function mk(sq: ArrayLike<number>, turn: 1 | -1, jump?: number, idle?: number): CheckersState {
  return { sq: Int8Array.from(sq), turn, jump: jump === undefined ? -1 : jump, idle: idle || 0 };
}

export function start(): CheckersState {
  const s = new Int8Array(32);
  for (let i = 0; i < 12; i++) s[i] = 1;
  for (let i = 20; i < 32; i++) s[i] = -1;
  return mk(s, 1, -1, 0);
}

export const clone = (st: CheckersState): CheckersState => mk(st.sq, st.turn, st.jump, st.idle);

const CHARS: Record<number, string> = { 1: "b", 2: "B", [-1]: "w", [-2]: "W", 0: "." };

/** the position as text: 32 squares (b/B black man/king, w/W white, . empty), the side to move, the jumping square (1-based or -), idle */
export function str(st: CheckersState): string {
  let s = "";
  for (let i = 0; i < 32; i++) s += CHARS[st.sq[i]!];
  return `${s} ${st.turn > 0 ? "b" : "w"} ${st.jump < 0 ? "-" : st.jump + 1} ${st.idle}`;
}

export function parse(text: string): CheckersState {
  const p = String(text).trim().split(/\s+/), s = new Int8Array(32), b = p[0] || "";
  for (let i = 0; i < 32; i++) {
    const c = b[i];
    s[i] = c === "b" ? 1 : c === "B" ? 2 : c === "w" ? -1 : c === "W" ? -2 : 0;
  }
  return mk(s, (p[1] || "b") === "w" ? -1 : 1, p[2] && p[2] !== "-" ? +p[2] - 1 : -1, +(p[3] || 0) || 0);
}

// Every frame is drawn from the side to move's point of view: its army solid, at the top, moving down the picture. White's positions are
// therefore turned 180° and their colors swapped, which is exactly the map sq -> 31 − sq. Two positions that are the same game up to
// color become the same picture and the same target, so the crystal never has to learn whose turn it is.
export function view(st: CheckersState): CanonicalView {
  if (st.turn > 0) return { sq: st.sq, jump: st.jump };
  const s = new Int8Array(32);
  for (let i = 0; i < 32; i++) s[flip(i)] = -st.sq[i]!;
  return { sq: s, jump: st.jump < 0 ? -1 : flip(st.jump) };
}

/** a real square -> the square it is drawn on */
export const viewSq = (st: CheckersState, sq: number): number => (st.turn > 0 ? sq : flip(sq));
/** and back (the map is its own inverse) */
export const realSq = (st: CheckersState, sq: number): number => (st.turn > 0 ? sq : flip(sq));
