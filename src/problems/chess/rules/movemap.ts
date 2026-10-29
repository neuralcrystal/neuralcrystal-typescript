// Reading a move off detector light, for crystals that answer with a move rather than a picture of the next position.
//
// Two lit squares: the crystal lights two of 64 detector squares, the one the piece leaves and the one it lands on. With the side to
// move, the unordered pair names one legal move: a move's from-square holds a piece of the side to move and its to-square cannot, so
// only one ordering is ever legal. Castling lights the king's pair. A promotion's four moves share a pair and read as the queen.
//
// The move map: a 64 × 64 grid of from/to cells (cell = from × 64 + to), the whole grid a policy. Every legal move has its own cell,
// several can be lit at once, and their brightness ranks them. Promotions share a cell and read as the queen.
import type { ChessMove, ChessState } from "./position.js";
import { make, moves, san, uci } from "./moves.js";

/** the piece type a promotion reads as: the queen */
export const PROMO_Q = 5;
export const MOVE_CELLS = 4096;
export const MOVE_CELL = (m: ChessMove): number => m.from * 64 + m.to;
/** the two squares a move lights */
export const moveSquares = (m: ChessMove): [number, number] => [m.from, m.to];

/** a shortlisted cell of the move map */
export interface ChessCandidate {
  cell: number;
  move: ChessMove | null;
  w: number;
  q?: number;
}

/** what the move map's light says */
export interface ChessReading {
  move: ChessMove | null;
  uci: string | null;
  san: string | null;
  /** the position after the move */
  state: ChessState | null;
  from: number;
  to: number;
  /** the played cell (from × 64 + to), its light and its lead over the next legal cell */
  cell: number;
  score: number;
  margin: number;
  /** the brightest cell of all [from, to], and whether it is the one played */
  top: [number, number];
  clean: boolean;
  /** the played cell's share of all the light; the share on legal cells */
  share: number;
  legal: number;
  moves: number;
  sampled: boolean;
  odds: number | null;
  cand: ChessCandidate[] | null;
  bestUci: string | null;
  bestSan: string | null;
  bestCell: number;
  /** a cell the rules forbid was played (only with the legal mask off) */
  illegal?: boolean;
}

export interface ChessReadOptions {
  /** sampling temperature (0: the brightest legal cell) */
  temp?: number;
  /** the shortlist sampled from */
  topK?: number;
  /** draw from all 4,096 cells, not the legal ones */
  anyCell?: boolean;
  rng?: () => number;
  moves?: ChessMove[];
}

/** what two lit squares say */
export interface SquaresReading {
  move: ChessMove;
  uci: string;
  san: string;
  state: ChessState;
  from: number;
  to: number;
  score: number;
  /** the lead over the best move on a different pair of squares */
  margin: number;
  /** the two brightest squares as read, and whether they are exactly the move's own */
  top: [number, number];
  clean: boolean;
  /** the share of the light on the move's squares */
  share: number;
}

/** p: the light in each of the 64 detector squares (0 = a1 … 63 = h8); the legal move whose two squares hold the most of it */
export function readSquares(p: ArrayLike<number>, st: ChessState, opts?: { moves?: ChessMove[] }): SquaresReading | null {
  const list = (opts && opts.moves) || moves(st);
  if (!list.length) return null;
  let best = list[0]!, s1 = -Infinity, s2 = -Infinity, key1 = -1;
  for (const m of list) {
    const s = p[m.from]! + p[m.to]!, key = Math.min(m.from, m.to) * 64 + Math.max(m.from, m.to);
    // a promotion's four moves share a pair: the queen is the one read
    if (s > s1 || (s === s1 && key === key1 && m.promo === PROMO_Q)) {
      if (key !== key1) s2 = s1;
      s1 = s;
      key1 = key;
      best = m;
    } else if (key !== key1 && s > s2) s2 = s;
  }
  let t1 = 0, t2 = 1, tot = 0;
  for (let d = 0; d < 64; d++) {
    tot += p[d]!;
    if (p[d]! > p[t1]!) {
      t2 = t1;
      t1 = d;
    } else if (d !== t1 && p[d]! > p[t2]!) t2 = d;
  }
  const clean = (t1 === best.from && t2 === best.to) || (t1 === best.to && t2 === best.from);
  return {
    move: best, uci: uci(best), san: san(st, best, list), state: make(st, best), from: best.from, to: best.to,
    score: s1, margin: s1 - (s2 === -Infinity ? 0 : s2), top: [t1, t2], clean, share: tot > 0 ? s1 / tot : 0,
  };
}

/** every legal move as its cell, promotions collapsed onto the queen's move: the legal mask */
export function moveCells(st: ChessState, opts?: { moves?: ChessMove[] }): Map<number, ChessMove> {
  const list = (opts && opts.moves) || moves(st), out = new Map<number, ChessMove>();
  for (const m of list) {
    const c = MOVE_CELL(m);
    if (!out.has(c) || m.promo === PROMO_Q) out.set(c, m);
  }
  return out;
}

/**
 * p: the light in each of the 4,096 move cells. Returns the legal move whose cell holds the most of it (so a reading is always
 * playable), its margin over the next-best legal cell, the brightest cell of all 4,096, and two shares: `share`, the played move's
 * share of all the light, and `legal`, the share on legal cells at all.
 *
 * At a temperature T > 0 the brightest topK legal cells are kept and one is drawn with odds ∝ light^(1/T), normalized against the
 * brightest so nothing overflows: T = 1 plays the map's own proportions, T → 0 takes the brightest, a large T flattens the shortlist.
 * The shortlist comes from the legal list, so an illegal cell is never drawn however bright it is. With `anyCell` the shortlist is
 * drawn from all 4,096 cells instead, so the crystal can name a move the rules forbid; the reading then says {illegal: true}, with
 * the cell it picked and no move to play. That is the honest test of a crystal: with the mask on, even an untrained map plays legal
 * chess every turn.
 */
export function readMoveMap(p: ArrayLike<number>, st: ChessState, opts?: ChessReadOptions): ChessReading | null {
  const list = (opts && opts.moves) || moves(st);
  if (!list.length) return null;
  let best = list[0]!, s1 = -Infinity, s2 = -Infinity, cell1 = -1;
  for (const m of list) {
    const c = MOVE_CELL(m), s = p[c] || 0;
    if (s > s1 || (s === s1 && c === cell1 && m.promo === PROMO_Q)) {
      if (c !== cell1) s2 = s1;
      s1 = s;
      cell1 = c;
      best = m;
    } else if (c !== cell1 && s > s2) s2 = s;
  }
  let t1 = 0, tot = 0, leg = 0;
  for (let d = 0; d < MOVE_CELLS; d++) {
    const v = p[d] || 0;
    tot += v;
    if (v > p[t1]!) t1 = d;
  }
  const seen = new Set<number>();
  for (const m of list) {
    const c = MOVE_CELL(m);
    if (seen.has(c)) continue;
    seen.add(c);
    leg += p[c] || 0;
  }
  const o = opts || {}, T = Number(o.temp) || 0, K = Math.max(0, (o.topK ?? 0) | 0), any = !!o.anyCell;
  const uniq = moveCells(st, { moves: list });
  let picked: ChessMove | null = best, odds: number | null = null, cand: ChessCandidate[] | null = null, pcell = cell1, psc = s1, illegal = false;
  if (T > 0 || any) {
    const c: ChessCandidate[] = any
      ? Array.from({ length: MOVE_CELLS }, (_, cell) => ({ cell, move: uniq.get(cell) || null, w: Math.max(0, p[cell] || 0) }))
      : [...uniq.entries()].map(([cell, move]) => ({ cell, move, w: Math.max(0, p[cell] || 0) }));
    c.sort((a, b) => b.w - a.w || a.cell - b.cell);
    const keep = K || (any ? 1 : c.length);
    if (c.length > keep) c.length = keep;
    let hit = c[0]!;
    if (!(T > 0)) odds = 1;
    else {
      const mx = c[0]!.w, inv = 1 / T;
      let Z = 0;
      for (const x of c) {
        x.q = mx > 0 ? Math.pow(x.w / mx, inv) : 1;
        Z += x.q;
      }
      for (const x of c) x.q = x.q! / Z;
      const r = o.rng ? o.rng() : Math.random();
      let acc = 0;
      hit = c[c.length - 1]!;
      for (const x of c) {
        acc += x.q!;
        if (r <= acc) {
          hit = x;
          break;
        }
      }
      odds = hit.q!;
    }
    picked = hit.move;
    pcell = hit.cell;
    psc = hit.w;
    illegal = !hit.move;
    cand = c;
  }
  if (illegal || !picked)
    return {
      illegal: true, move: null, uci: null, san: null, state: null, cell: pcell, from: pcell >> 6, to: pcell & 63,
      score: psc, margin: 0, top: [t1 >> 6, t1 & 63], clean: t1 === pcell, sampled: T > 0, odds, cand,
      share: tot > 0 ? psc / tot : 0, legal: tot > 0 ? leg / tot : 0, moves: seen.size, bestUci: uci(best), bestSan: san(st, best, list), bestCell: cell1,
    };
  let s2p = -Infinity;
  if (picked !== best)
    for (const m of list) {
      const c = MOVE_CELL(m);
      if (c === pcell) continue;
      const v = p[c] || 0;
      if (v > s2p) s2p = v;
    }
  else s2p = s2;
  return {
    move: picked, uci: uci(picked), san: san(st, picked, list), state: make(st, picked), from: picked.from, to: picked.to,
    score: psc, margin: psc - (s2p === -Infinity ? 0 : s2p), cell: pcell, top: [t1 >> 6, t1 & 63], clean: t1 === pcell,
    share: tot > 0 ? psc / tot : 0, legal: tot > 0 ? leg / tot : 0, moves: seen.size,
    sampled: T > 0, odds, cand, bestUci: uci(best), bestSan: san(st, best, list), bestCell: cell1,
  };
}
