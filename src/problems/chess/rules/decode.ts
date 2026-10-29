// Reading a position back off the sensor. Each square of the readout is read by correlation, means removed, against the twelve pieces
// drawn on a flat background at the square's own fill (no dither in the templates: the dither is what a blurred empty square and a
// piece template would otherwise share). The tile's border is cropped off (what bleeds in from the neighbors), and the reading is
// taken at each bin size against templates blurred by each radius, the best variant counting, so it holds whatever the crystal's
// resolution turns out to be. A square's "empty" score is 1 − its best piece score.
//
// A move is not read square by square: readMove sums the scores of every legal move's resulting position and plays the best, unless
// the unchanged position scores as well or better, or the picture disagrees with the board read on more than READ_OFF squares. Either
// reads as no move, so a crystal that only relays the board, or shows diffraction haze, is not credited with a move.
import type { ChessMove, ChessState } from "./position.js";
import { changed, classOf } from "./position.js";
import { make, moves, san, uci } from "./moves.js";
import { live, sqLight, sqRect } from "./layout.js";
import type { Fills } from "./frame.js";
import { DARK, LIGHT, spriteInto } from "./frame.js";

/**
 * The legibility gate: a move counts only if the picture, read square by square, agrees with the board it is read as to within this
 * many squares, and shows the move's own squares exactly (the from-square empty, the piece on its to-square). Clean targets read 0
 * off through 5 px of blur; an untrained crystal's diffraction haze 17–35; pure noise 8 or more.
 */
export const READ_OFF = 4;

/** per square the 13 scores (class 0 = empty), the best class and its margin */
export interface Decoded {
  score: Float32Array;
  cls: Int8Array;
  margin: Float32Array;
}

/** one bin size's templates: per square color, 12 classes × the blurs, each a unit centered vector of M values */
interface BinBank {
  b: number;
  w: number;
  M: number;
  light: Float32Array;
  dark: Float32Array;
}

interface Bank {
  key: string;
  W: number;
  nb: number;
  bins: BinBank[];
}

let bank: Bank | null = null;

/** a piece on its square's mean fill, no dither */
export function spriteTile(cls: number, light: boolean, opts?: Fills): Float32Array {
  const o = opts || {}, T = live.TILE, lvl = light ? (o.light ?? LIGHT) : (o.dark ?? DARK), v = new Float32Array(T * T).fill(lvl);
  return cls ? spriteInto(v, cls, light) : v;
}

/** a box blur of radius r over a W × W tile, edges clamped */
export function blurTile(t: Float32Array, r: number, W: number): Float32Array {
  if (!r) return t;
  const out = new Float32Array(W * W);
  for (let y = 0; y < W; y++)
    for (let x = 0; x < W; x++) {
      let a = 0, c = 0;
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const Y = Math.min(W - 1, Math.max(0, y + dy)), X = Math.min(W - 1, Math.max(0, x + dx));
          a += t[Y * W + X]!;
          c++;
        }
      out[y * W + x] = a / c;
    }
  return out;
}

/** the W × W tile with c pixels cropped off each edge */
export function innerTile(t: Float32Array, W: number, c: number, out?: Float32Array): Float32Array {
  const w = W - 2 * c, o = out || new Float32Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) o[y * w + x] = t[(y + c) * W + x + c]!;
  return o;
}

/** the W × W tile averaged in b × b bins */
export function binTile(t: Float32Array, b: number, W: number, out?: Float32Array): Float32Array {
  const w = W / b, o = out || new Float32Array(w * w);
  for (let j = 0; j < w; j++)
    for (let i = 0; i < w; i++) {
      let a = 0;
      for (let y = 0; y < b; y++) for (let x = 0; x < b; x++) a += t[(j * b + y) * W + i * b + x]!;
      o[j * w + i] = a / (b * b);
    }
  return o;
}

/** mean removed, scaled to unit length (in place) */
export function unitCentered(t: Float32Array): Float32Array {
  let m = 0;
  for (const q of t) m += q;
  m /= t.length;
  let n = 0;
  for (let k = 0; k < t.length; k++) {
    t[k] = t[k]! - m;
    n += t[k]! * t[k]!;
  }
  n = Math.sqrt(n) || 1;
  for (let k = 0; k < t.length; k++) t[k] = t[k]! / n;
  return t;
}

/** the template bank for the live layout and the given fills (built once, kept until either changes) */
export function templates(opts?: Fills): Bank {
  const key = JSON.stringify([live.SIZE, opts && opts.light, opts && opts.dark, live.DEC]);
  if (bank && bank.key === key) return bank;
  const D = live.DEC, T = live.TILE, W = T - 2 * D.crop, nb = D.blurs.length, built: Bank = { key, W, nb, bins: [] };
  for (const b of D.bins) {
    const w = W / b, M = w * w, per: BinBank = { b, w, M, light: new Float32Array(12 * nb * M), dark: new Float32Array(12 * nb * M) };
    for (const light of [true, false]) {
      const out = light ? per.light : per.dark;
      for (let c = 1; c < 13; c++)
        D.blurs.forEach((rb, j) => {
          const t = unitCentered(binTile(innerTile(blurTile(spriteTile(c, light, opts), rb, T), T, D.crop), b, W));
          out.set(t, ((c - 1) * nb + j) * M);
        });
    }
    built.bins.push(per);
  }
  return (bank = built);
}

/** a square's tile cut out of a frame */
export function tileOf(v: ArrayLike<number>, sq: number, out?: Float32Array): Float32Array {
  const [x0, y0, w, h] = sqRect(sq), o = out || new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) o[y * w + x] = v[(y0 + y) * live.SIZE + x0 + x]!;
  return o;
}

/** the readout (SIZE² intensities) as a piece per square */
export function decode(v: ArrayLike<number>, opts?: Fills): Decoded {
  const tb = templates(opts), D = live.DEC, T = live.TILE, W = tb.W, nb = tb.nb;
  const score = new Float32Array(64 * 13), cls = new Int8Array(64), margin = new Float32Array(64), t = new Float32Array(T * T), ti = new Float32Array(W * W);
  const u = tb.bins.map((p) => new Float32Array(p.M));
  for (let sq = 0; sq < 64; sq++) {
    tileOf(v, sq, t);
    innerTile(t, T, D.crop, ti);
    tb.bins.forEach((p, i) => unitCentered(binTile(ti, p.b, W, u[i])));
    const light = sqLight(sq);
    let best = -2, bc = 0, second = -2;
    for (let c = 1; c < 13; c++) {
      let s = -2;
      tb.bins.forEach((p, i) => {
        const tp = light ? p.light : p.dark, uu = u[i]!;
        for (let j = 0; j < nb; j++) {
          const o = ((c - 1) * nb + j) * p.M;
          let d = 0;
          for (let k = 0; k < p.M; k++) d += uu[k]! * tp[o + k]!;
          if (d > s) s = d;
        }
      });
      score[sq * 13 + c] = s;
      if (s > best) {
        second = best;
        best = s;
        bc = c;
      } else if (s > second) second = s;
    }
    const e = 1 - best;
    score[sq * 13] = e;
    if (e >= best) {
      cls[sq] = 0;
      margin[sq] = e - best;
    } else {
      cls[sq] = bc;
      margin[sq] = best - Math.max(second, e);
    }
  }
  return { score, cls, margin };
}

/** the squares a reading gets wrong against a position */
export function wrong(dec: Decoded, st: ChessState): number {
  let n = 0;
  for (let sq = 0; sq < 64; sq++) if (dec.cls[sq] !== classOf(st.sq[sq]!)) n++;
  return n;
}

/** what a readout of the board says */
export interface BoardReading {
  move: ChessMove | null;
  /** no move: the picture still shows the position given ("unchanged") or is not legible as a board ("illegible") */
  none?: true;
  why?: "unchanged" | "illegible";
  state: ChessState;
  score: number;
  margin: number;
  maxOff: number;
  wrong: number;
  /** every square the move changes reads as the move leaves it */
  shown?: boolean;
  dec: Decoded;
  san: string;
  uci: string | null;
  /** the legal move that came closest */
  nearest?: string;
}

export interface BoardReadOptions extends Fills {
  /** the decoded readout, if already known */
  dec?: Decoded;
  moves?: ChessMove[];
  /** the legibility gate (READ_OFF unless given) */
  maxOff?: number | null;
}

/** the legal move whose resulting position the readout shows best, or no move; null when there is no legal move */
export function readMove(v: ArrayLike<number>, st: ChessState, opts?: BoardReadOptions): BoardReading | null {
  const dec = (opts && opts.dec) || decode(v, opts), list = (opts && opts.moves) || moves(st);
  const sum = (b: ChessState): number => {
    let s = 0;
    for (let sq = 0; sq < 64; sq++) s += dec.score[sq * 13 + classOf(b.sq[sq]!)]!;
    return s;
  };
  // the board unchanged is a candidate too: a crystal that hands the position back must read as no move, not as the move that
  // changes the picture least
  const s0 = sum(st);
  let best: { move: ChessMove; state: ChessState; score: number } | null = null, s1 = -Infinity, s2 = -Infinity;
  for (const m of list) {
    const n = make(st, m), s = sum(n);
    if (s > s1) {
      s2 = s1;
      s1 = s;
      best = { move: m, state: n, score: s };
    } else if (s > s2) s2 = s;
  }
  if (!best) return null;
  const maxOff = opts && opts.maxOff != null ? opts.maxOff : READ_OFF, nearest = san(st, best.move, list);
  if (s0 >= s1) return { move: null, none: true, why: "unchanged", state: st, score: s0, margin: s0 - s1, maxOff, wrong: wrong(dec, st), dec, san: "no move", uci: null, nearest };
  const margin = s1 - Math.max(s2, s0), off = wrong(dec, best.state), after = best.state;
  // the move itself must be in the picture: every square it changes reads as the move leaves it
  const shown = changed(st, after).every((sq) => dec.cls[sq] === classOf(after.sq[sq]!));
  if (off > maxOff || !shown) return { move: null, none: true, why: "illegible", state: st, score: s1, margin, maxOff, wrong: off, shown, dec, san: "no move", uci: null, nearest };
  return { move: best.move, state: after, score: best.score, margin, wrong: off, maxOff, dec, san: nearest, uci: uci(best.move) };
}
