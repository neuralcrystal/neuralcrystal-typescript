// Reading the detectors' light back into a move. The legal mask is applied by the rules, not by a threshold, so a reading is always
// playable: the legal move whose cell holds the most light, with its share of all the light on the map, the share that landed on legal
// cells at all, and the brightest cell of all whether legal or not. temp / topK sample from the top K legal cells at a temperature instead
// of always taking the brightest; anyCell draws the shortlist from every cell with no mask, so the crystal may name an illegal move and
// the caller is told.
import type { ViewMove } from "./move-map.js";
import { cellMove, moveMap, viewMoves } from "./move-map.js";
import type { CheckersMove } from "./moves.js";
import { make, uci } from "./moves.js";
import type { CheckersState } from "./position.js";

/** a candidate in the reader's shortlist */
export interface MoveCandidate {
  cell: number;
  w: number;
  q?: number;
}

/** what the detectors' light says */
export interface MoveReading {
  /** the cell played: from × 32 + to, in canonical squares (the side to move at the top) */
  cell: number;
  /** its light, and its lead over the next legal cell */
  score: number;
  margin: number;
  /** its chance when sampled (temperature > 0), or null */
  odds: number | null;
  sampled: boolean;
  /** the brightest cell of all, whether it is the one played, and whether it is legal */
  top: number;
  clean: boolean;
  topLegal: boolean;
  /** the played cell's share of all the light; the share that fell on legal cells at all */
  share: number;
  legal: number;
  /** how many distinct legal cells there were */
  cells: number;
  /** the brightest legal move */
  best: CheckersMove | null;
  bestUci: string | null;
  cand: MoveCandidate[];
  /** a brighter illegal cell was played (only with the legal mask off) */
  illegal: boolean;
  move: CheckersMove | null;
  uci: string | null;
  /** the position after the move */
  state: CheckersState | null;
  from: number;
  to: number;
}

export interface ReadOptions {
  /** sampling temperature: 0 plays the brightest cell, 1 picks in proportion to light */
  temp?: number;
  /** the shortlist sampled from (0: all) */
  topK?: number;
  /** consider every cell, not just legal ones (the legal mask off) */
  anyCell?: boolean;
  /** a random source (tests) */
  rng?: () => number;
}

/** a shortlist entry: its cell, the legal move there (null for an illegal cell), its light, and its chance once sampled */
interface Cand extends MoveCandidate {
  e: ViewMove | null;
}

/** p: the light in each cell of the map. null when the side to move has no legal move. */
export function readMoveMap(p: ArrayLike<number>, st: CheckersState, opts?: ReadOptions & { moves?: CheckersMove[] }): MoveReading | null {
  const o = opts || {}, ent = viewMoves(st, o.moves);
  if (!ent.length) return null;
  const uniq = new Map<number, ViewMove>();
  for (const e of ent) if (!uniq.has(e.cell)) uniq.set(e.cell, e);
  const CELLS = moveMap.CELLS;
  let tot = 0, t1 = 0;
  for (let c = 0; c < CELLS; c++) {
    const q = p[c] || 0;
    tot += q;
    if (q > p[t1]!) t1 = c;
  }
  let leg = 0;
  for (const c of uniq.keys()) leg += p[c] || 0;
  const T = +(o.temp ?? 0) || 0, K = Math.max(0, (o.topK ?? 0) | 0), any = !!o.anyCell;
  let cand: Cand[] = any
    ? Array.from({ length: CELLS }, (_, c) => ({ cell: c, e: uniq.get(c) || null, w: Math.max(0, p[c] || 0) }))
    : [...uniq.values()].map((e) => ({ cell: e.cell, e, w: Math.max(0, p[e.cell] || 0) }));
  cand.sort((a, b) => b.w - a.w || a.cell - b.cell);
  const bestLegal = any ? cand.find((c) => c.e) : cand[0];
  if (K && cand.length > K) cand = cand.slice(0, K);
  let hit = cand[0]!;
  if (T > 0 && cand.length > 1) {
    const mx = cand[0]!.w, inv = 1 / T;
    let Z = 0;
    for (const c of cand) {
      c.q = mx > 0 ? Math.pow(c.w / mx, inv) : 1;
      Z += c.q;
    }
    for (const c of cand) c.q! /= Z;
    const r = o.rng ? o.rng() : Math.random();
    let acc = 0;
    hit = cand[cand.length - 1]!;
    for (const c of cand) {
      acc += c.q!;
      if (r <= acc) {
        hit = c;
        break;
      }
    }
  }
  let s2 = -Infinity;
  for (const c of uniq.keys())
    if (c !== hit.cell) {
      const q = p[c] || 0;
      if (q > s2) s2 = q;
    }
  const bm = bestLegal?.e ? bestLegal.e.m : null;
  const base = {
    cell: hit.cell, score: hit.w, margin: hit.w - (s2 === -Infinity ? 0 : s2), odds: hit.q ?? null, sampled: T > 0,
    top: t1, clean: t1 === hit.cell, topLegal: uniq.has(t1), share: tot > 0 ? hit.w / tot : 0, legal: tot > 0 ? leg / tot : 0, cells: uniq.size,
    best: bm, bestUci: bm ? uci(bm) : null, cand,
  };
  if (!hit.e) {
    const cm = cellMove(hit.cell);
    return Object.assign(base, { illegal: true, move: null, uci: null, state: null, from: cm.from, to: cm.to });
  }
  return Object.assign(base, { illegal: false, move: hit.e.m, uci: uci(hit.e.m), state: make(st, hit.e.m), from: hit.e.from, to: hit.e.to });
}
