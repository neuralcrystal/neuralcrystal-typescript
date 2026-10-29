// The frame: a position as a binary DMD picture, Float32Array(SIZE²) of 0 / 1. A plain two-tone board (light squares lit, dark squares
// dark) with each piece's sprite centered in its square, in its own color: solid where it contrasts with its square, an outline where it
// matches. The border carries the state: the turn clock beside the army that moves, each side's castling flags, an en-passant lamp under
// the file it can be taken on, the crystal's quartz mark, the winning bar, the result word, and the seed bands: fresh random mirrors
// every time the position is shown (or a rank code in their place), so the crystal's play need not be deterministic. A target frame can
// also ring each square the move changed, so a trained crystal highlights its own move.
import type { ChessMove, ChessState } from "./position.js";
import { classOf, PIECE, pieceOfClass, rng as seeded } from "./position.js";
import { moves } from "./moves.js";
import type { Rect } from "./layout.js";
import { blank, fillRect, live, rect, sqLight, sqRect } from "./layout.js";
import { NOISE, NOISEBLK, NOISECELL, NOISEDARK, NOISEFRAME, NOISEGAP, NOISEMODE, renderCode, renderNoise, seedBands } from "./seed.js";
import { spritePx } from "./sprites.js";
import { drawText } from "./text.js";

/** the fill of a lit and of a dark square */
export const LIGHT = 1;
export const DARK = 0;
/** the winning bar's full scale: ±3 pawns from white's view (a mate is clipped to the end) */
export const VALUE_CP = 300;
/** an 8 × 8 ordered-dither matrix, for squares filled between dark and lit */
export const BAYER = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21];

/** the squares' fills (0..1) */
export interface Fills {
  light?: number;
  dark?: number;
}

export interface FrameOptions extends Fills {
  /** the seed noise's fill (0: none) */
  noise?: number;
  seed?: number;
  rng?: () => number;
  noiseBlk?: number;
  noiseCell?: number;
  noiseGap?: number;
  noiseFrame?: number;
  noiseDark?: number;
  noiseMode?: string;
  /** the seed goes to the right band only (the bottom one is the winning bar's) */
  noBottom?: boolean;
  /** the rank code in the seed bands instead of noise: K cells, cell n lit */
  code?: { n: number; K: number };
  /** squares to ring (a target's changed squares), and the ring's width */
  ring?: number[];
  ringW?: number;
  /** leave the turn clock dark */
  noClock?: boolean;
  /** the legal list, if already known */
  moves?: ChessMove[];
  /** the winning bar's value, centipawns from white's view */
  valueCp?: number | null;
  /** the crystal's quartz mark on these sides ("w" / "b") */
  crystalSides?: ("w" | "b")[];
  /** the game-over word in the top strip */
  result?: string | null;
}

export const fillLevel = (fill: number): number => Math.max(0, Math.min(64, Math.round(fill * 64)));
/** a piece's sprite drawn into a square's tile */
export function spriteInto(v: Float32Array, cls: number, light: boolean): Float32Array {
  const T = live.TILE, p = pieceOfClass(cls), S = live.SPRITES[PIECE[Math.abs(p) - 1]!]!, solid = p > 0, o = (T - live.SPR) / 2;
  for (let y = 0; y < live.SPR; y++)
    for (let x = 0; x < live.SPR; x++) {
      const q = spritePx(S, solid, light, y, x);
      if (q >= 0) v[(y + o) * T + x + o] = q;
    }
  return v;
}

/** one square as the DMD shows it: Float32Array(TILE²), dithered to its fill, with its piece (class 1..12) */
export function tile(cls: number, light: boolean, opts?: Fills): Float32Array {
  const o = opts || {}, T = live.TILE, v = new Float32Array(T * T), lvl = fillLevel(light ? (o.light ?? LIGHT) : (o.dark ?? DARK));
  for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) v[y * T + x] = BAYER[(y & 7) * 8 + (x & 7)]! < lvl ? 1 : 0;
  return cls ? spriteInto(v, cls, light) : v;
}

/** the 64 squares only, into a frame */
export function renderBoard(v: Float32Array, st: ChessState, opts?: Fills): Float32Array {
  const T = live.TILE;
  for (let sq = 0; sq < 64; sq++) {
    const [x0, y0] = sqRect(sq), t = tile(classOf(st.sq[sq]!), sqLight(sq), opts);
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) v[(y0 + y) * live.SIZE + x0 + x] = t[y * T + x]!;
  }
  return v;
}

/** the lamps: turn, castling, en passant (the en-passant lamp only when the capture is legal) */
export function renderState(v: Float32Array, st: ChessState, opts?: { noClock?: boolean; moves?: ChessMove[] }): Float32Array {
  const L = live.LAMP, on = (r: Rect): void => rect(v, r[0], r[1], r[2], r[3], 1), off = (r: Rect): void => rect(v, r[0], r[1], r[2], r[3], 0);
  // noClock leaves the turn clock dark: a crystal trained to play one color only ever sees its own clock lit, so the lamp would cost
  // light and teach nothing
  off(L.turnW);
  off(L.turnB);
  if (!(opts && opts.noClock)) on(st.turn === "w" ? L.turnW : L.turnB);
  for (const k of ["K", "Q", "k", "q"] as const) (st.castle[k] ? on : off)(L.castle[k]);
  if (st.ep >= 0 && (opts && opts.moves ? opts.moves : moves(st)).some((m) => m.ep)) {
    // only the active file, as a dark-bordered lit lamp so it reads over the seed noise
    const r = L.ep(st.ep & 7), e = live.EPIN;
    off(r);
    on([r[0] + e, r[1] + e, r[2] - 2 * e, r[3] - 2 * e]);
  }
  return v;
}

/** a small sideways, double-terminated quartz in a region: solid, or a thick outline */
export function renderQuartz(v: Float32Array, x0: number, y0: number, w: number, h: number, solid: boolean): Float32Array {
  const cy = y0 + (h - 1) / 2, tip = Math.max(2, Math.round(w * 0.3)), halfMax = (h - 2) / 2;
  // pointed at both ends, straight in the middle
  const hhAt = (x: number): number => {
    const rx = x - x0;
    if (rx < tip) return (halfMax * (rx + 1)) / tip;
    if (rx >= w - tip) return (halfMax * (w - rx)) / tip;
    return halfMax;
  };
  const inside = (x: number, y: number): boolean => Math.abs(y - cy) <= hhAt(x) + 0.001, th = Math.max(1, Math.round(h * 0.13));
  for (let x = x0; x < x0 + w; x++)
    for (let y = y0; y < y0 + h; y++) {
      if (!inside(x, y)) continue;
      const on = solid || !inside(x - th, y) || !inside(x + th, y) || !inside(x, y - th) || !inside(x, y + th);
      if (on) v[y * live.SIZE + x] = 1;
    }
  return v;
}

/** which color(s) the crystal plays: the top slot for black, the bottom for white; both for self-play, none when off */
export function renderCrystalMark(v: Float32Array, sides?: readonly string[] | null): Float32Array {
  if (!sides || !sides.length) return v;
  if (sides.indexOf("b") >= 0) renderQuartz(v, ...live.LAMP.crystalT, true);
  if (sides.indexOf("w") >= 0) renderQuartz(v, ...live.LAMP.crystalB, true);
  return v;
}

/** the winning bar, a thermometer from the left: empty = black winning by VALUE_CP, half = equal, full = white winning by VALUE_CP */
export function renderValue(v: Float32Array, cp: number | null | undefined): Float32Array {
  const [x, y, w, h] = live.LAMP.value;
  rect(v, x, y, w, h, 0);
  if (cp === null || cp === undefined || !isFinite(cp)) return v;
  const len = Math.round(Math.max(0, Math.min(1, (cp + VALUE_CP) / (2 * VALUE_CP))) * w);
  if (len > 0) fillRect(v, x, y, x + len, y + h);
  return v;
}

/** the bar read off a picture: columns lit past half the brightest, counted from the left until three dark ones; cp from white's view */
export function readValue(pic: ArrayLike<number>): { cp: number; len: number; lit: number } {
  const [x, y, w, h] = live.LAMP.value, col = new Float32Array(w);
  let mx = 0;
  for (let i = 0; i < w; i++) {
    let s = 0;
    for (let j = y; j < y + h; j++) s += pic[j * live.SIZE + x + i]!;
    col[i] = s;
    if (s > mx) mx = s;
  }
  let len = 0, dark = 0;
  if (mx > 0)
    for (let i = 0; i < w; i++) {
      if (col[i]! >= mx / 2) {
        len = i + 1;
        dark = 0;
      } else if (++dark >= 3) break;
    }
  return { cp: Math.round((len / w) * 2 * VALUE_CP - VALUE_CP), len, lit: mx / h };
}

/** a ring w mirrors wide just inside each square's edge, in the tone that contrasts with the square */
export function ringSquares(v: Float32Array, squares: readonly number[], w?: number): Float32Array {
  const W = w ?? live.RING;
  if (!(W > 0)) return v;
  const T = live.TILE;
  for (const sq of squares) {
    const [x0, y0] = sqRect(sq), on = sqLight(sq) ? 0 : 1;
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (x < W || y < W || x >= T - W || y >= T - W) v[(y0 + y) * live.SIZE + x0 + x] = on;
  }
  return v;
}

/** clear the top strip, then the word */
export function renderBanner(v: Float32Array, text: string): Float32Array {
  const b = live.LAMP.banner;
  for (let y = b[1]; y < b[1] + b[3]; y++) for (let x = b[0]; x < b[0] + b[2]; x++) v[y * live.SIZE + x] = 0;
  drawText(v, text, live.SIZE / 2, b[1] + b[3] / 2, live.TEXT);
  return v;
}

/** the frame for a position */
export function render(st: ChessState, opts?: FrameOptions): Float32Array {
  const o = opts || {}, v = blank();
  renderBoard(v, st, o);
  if (o.ring) ringSquares(v, o.ring, o.ringW);
  const bands = o.noBottom ? seedBands(true) : null; // the bottom band is the winning bar's: the seed goes to the right band only
  if (o.code) renderCode(v, o.code.n, o.code.K, { bands, ...o });
  const fill = o.code ? 0 : (o.noise ?? NOISE);
  if (fill > 0)
    renderNoise(v, fill, o.rng || seeded(o.seed ?? 1), o.noiseBlk ?? NOISEBLK, o.noiseCell ?? NOISECELL, o.noiseGap ?? NOISEGAP, o.noiseFrame ?? NOISEFRAME, o.noiseDark ?? NOISEDARK, o.noiseMode ?? NOISEMODE, bands);
  renderState(v, st, o); // the lamps over the seed band, so the en-passant lamp stays readable
  if (o.valueCp !== undefined && o.valueCp !== null) renderValue(v, o.valueCp);
  if (o.crystalSides) renderCrystalMark(v, o.crystalSides);
  if (o.result) renderBanner(v, o.result);
  return v;
}

/** a target keeps the input's border: the crystal relays the state and the noise, and redraws the board */
export function copyBorder(from: ArrayLike<number>, into: Float32Array): Float32Array {
  const N = live.SIZE;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) if (x < live.ORG || y < live.ORGY || x >= live.ORG + live.BOARD || y >= live.ORGY + live.BOARD) into[y * N + x] = from[y * N + x]!;
  return into;
}

/** a frame with the given squares lit */
export function squareMask(squares: readonly number[]): Float32Array {
  const v = blank();
  for (const sq of squares) rect(v, ...sqRect(sq), 1);
  return v;
}
