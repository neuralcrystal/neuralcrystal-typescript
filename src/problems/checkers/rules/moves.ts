// The rules of English draughts (American checkers): men step one square diagonally forward and jump forward only; kings step and jump
// both ways; a jump is compulsory whenever one exists, and a jump chain must be played out, but a man that lands on the far row is crowned
// and its turn ends there, chain or no chain. moves() returns atomic moves: one step, or one jump. The crystal is asked the same question
// at every jump of a chain, and the rules, not the optics, know the chain is still running.
import type { CheckersState } from "./position.js";
import { clone, colOf, DIRS, rowOf, sqAt } from "./position.js";

/** one atomic move: a step, or one jump of a chain */
export interface CheckersMove {
  from: number;
  to: number;
  /** the captured square, or −1 */
  cap: number;
  promo: boolean;
  /** whether the piece moving is a king (moves() always sets it) */
  king?: boolean;
}

export interface CheckersStatus {
  over: boolean;
  winner: 1 | -1 | 0 | null;
  reason: string;
}

/** the side to move's jumps from one square, appended to out */
export function jumpsFrom(st: CheckersState, from: number, out: CheckersMove[]): CheckersMove[] {
  const p = st.sq[from]!;
  if (!p || p > 0 !== st.turn > 0) return out;
  const r = rowOf(from), c = colOf(from), king = Math.abs(p) === 2, fwd = st.turn > 0 ? 1 : -1;
  for (const [dr, dc] of DIRS) {
    if (!king && dr !== fwd) continue;
    const mid = sqAt(r + dr, c + dc), to = sqAt(r + 2 * dr, c + 2 * dc);
    if (mid < 0 || to < 0) continue;
    const q = st.sq[mid]!;
    if (!q || q > 0 === st.turn > 0 || st.sq[to]) continue;
    out.push({ from, to, cap: mid, promo: !king && rowOf(to) === (st.turn > 0 ? 7 : 0), king });
  }
  return out;
}

/** the side to move's steps from one square, appended to out */
export function stepsFrom(st: CheckersState, from: number, out: CheckersMove[]): CheckersMove[] {
  const p = st.sq[from]!;
  if (!p || p > 0 !== st.turn > 0) return out;
  const r = rowOf(from), c = colOf(from), king = Math.abs(p) === 2, fwd = st.turn > 0 ? 1 : -1;
  for (const [dr, dc] of DIRS) {
    if (!king && dr !== fwd) continue;
    const to = sqAt(r + dr, c + dc);
    if (to < 0 || st.sq[to]) continue;
    out.push({ from, to, cap: -1, promo: !king && rowOf(to) === (st.turn > 0 ? 7 : 0), king });
  }
  return out;
}

/** every legal atomic move */
export function moves(st: CheckersState): CheckersMove[] {
  if (st.jump >= 0) return jumpsFrom(st, st.jump, []); // mid-chain: only that piece, only jumps
  const jumps: CheckersMove[] = [];
  for (let i = 0; i < 32; i++) jumpsFrom(st, i, jumps);
  if (jumps.length) return jumps; // captures are compulsory
  const steps: CheckersMove[] = [];
  for (let i = 0; i < 32; i++) stepsFrom(st, i, steps);
  return steps;
}

/** the position after a move (a new state; the turn does not pass while a jump chain continues) */
export function make(st: CheckersState, m: CheckersMove): CheckersState {
  const n = clone(st), p = n.sq[m.from]!;
  n.sq[m.from] = 0;
  if (m.cap >= 0) n.sq[m.cap] = 0;
  const crown = m.promo && Math.abs(p) === 1;
  n.sq[m.to] = crown ? (p > 0 ? 2 : -2) : p;
  n.idle = m.cap >= 0 || Math.abs(p) === 1 ? 0 : n.idle + 1; // a capture or any man move resets the idle count
  if (m.cap >= 0 && !crown && jumpsFrom(n, m.to, []).length) {
    n.jump = m.to; // the chain goes on: same turn, same piece
    return n;
  }
  n.jump = -1;
  n.turn = -n.turn as 1 | -1;
  return n;
}

/** 40 moves a side with no capture and no man move is a draw */
export const IDLE_DRAW = 80;

/** a side with no move loses; the idle limit is a draw */
export function status(st: CheckersState): CheckersStatus {
  if (!moves(st).length)
    return { over: true, winner: -st.turn as 1 | -1, reason: st.sq.some((v) => v && v > 0 === st.turn > 0) ? "blocked" : "no pieces" };
  if (st.idle >= IDLE_DRAW) return { over: true, winner: 0, reason: "40-move rule" };
  return { over: false, winner: 0, reason: "" };
}

export const sameMove = (a: CheckersMove | null | undefined, b: CheckersMove | null | undefined): boolean => !!a && !!b && a.from === b.from && a.to === b.to;

/** draughts notation: 11-15 a step, 11x18 a jump */
export function uci(m: CheckersMove | null): string | null {
  return m ? `${m.from + 1}${m.cap >= 0 ? "x" : "-"}${m.to + 1}` : null;
}

/** the legal move a notation names, or null */
export function fromUci(st: CheckersState, s: string): CheckersMove | null {
  const m = /^(\d+)[x-](\d+)$/.exec(String(s).trim());
  if (!m) return null;
  const from = +m[1]! - 1, to = +m[2]! - 1;
  return moves(st).find((v) => v.from === from && v.to === to) || null;
}
