// A position's identity, for spotting repetitions: Zobrist hashing over the 32 squares, the side to move and the square a jump chain is
// running on. 53 bits, so it stays an exact float64 integer and a plain Map or array can hold it.
import type { CheckersState } from "./position.js";

/** mulberry32: a small seeded random source */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Pair = [number, number];

// the random words, drawn once in a fixed order: four piece kinds a square, the side to move, then the jump square (32 = none)
const Z = (() => {
  const r = rng(0x9e3779b9), pick = (): number => (r() * 4294967296) >>> 0, pair = (): Pair => [pick(), pick()];
  const sq: Pair[][] = [];
  for (let i = 0; i < 32; i++) {
    const a: Pair[] = [];
    for (let k = 0; k < 4; k++) a.push(pair());
    sq.push(a);
  }
  const turn = pair();
  return { sq, turn, jump: Array.from({ length: 33 }, pair) };
})();

/** the piece kind's index: black man, black king, white man, white king */
const KIND: Record<number, number> = { 1: 0, 2: 1, [-1]: 2, [-2]: 3 };

export function key(st: CheckersState): number {
  let hi = 0, lo = 0;
  for (let i = 0; i < 32; i++) {
    const p = st.sq[i]!;
    if (!p) continue;
    const z = Z.sq[i]![KIND[p]!]!;
    hi ^= z[0];
    lo ^= z[1];
  }
  if (st.turn < 0) {
    hi ^= Z.turn[0];
    lo ^= Z.turn[1];
  }
  const j = Z.jump[st.jump < 0 ? 32 : st.jump]!;
  hi ^= j[0];
  lo ^= j[1];
  return ((hi >>> 0) & 0x1fffff) * 4294967296 + (lo >>> 0);
}
