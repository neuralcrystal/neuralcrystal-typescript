// The frame's geometry. Two frames: the full one is 192 px (18-px squares, 16-px pieces, a 24-px border for the lamps and the seed
// bands); the compact one is 96 px (10-px squares, 8-px pieces, an 8-px border), for optics that cannot carry the full frame's fine
// pixel. The renderer and the decoder read the live layout (`live`); layout(size) switches it, and every lamp, band, bar, the banner's
// text size, the en-passant lamp's inset, the decoder's crop and bins and the sprite set follow.
import type { SpriteSet } from "./sprites.js";
import { SPRITES16, SPRITES8 } from "./sprites.js";
import { isLight } from "./position.js";
import { inb } from "./moves.js";

/** x, y, w, h in frame pixels */
export type Rect = [number, number, number, number];

export interface Lamps {
  /** the turn clock: black's above, white's below (white plays from the bottom) */
  turnB: Rect;
  turnW: Rect;
  /** a small sideways quartz at each end: top = the crystal plays black, bottom = white */
  crystalT: Rect;
  crystalB: Rect;
  /** the castling rights */
  castle: Record<"K" | "Q" | "k" | "q", Rect>;
  /** the en-passant lamp under file f */
  ep: (f: number) => Rect;
  /** the strip for the result word */
  banner: Rect;
  /** the winning bar */
  value: Rect;
  /** the seed bands */
  noise: Rect[];
}

/** what the decoder reads of a square: the border cropped off, the template blurs, the bin sizes */
export interface DecoderShape {
  crop: number;
  blurs: number[];
  bins: number[];
}

export interface Layout {
  SIZE: number;
  TILE: number;
  SPR: number;
  ORG: number;
  /** the board's top row, when it differs from its left column */
  ORGY?: number;
  BOARD: number;
  /** the banner's text scale */
  TEXT: number;
  /** the banner font's height: 7 (5 × 7) or 5 (5 × 5) */
  FONT?: number;
  /** the en-passant lamp's dark inset */
  EPIN: number;
  /** the ring drawn round each changed square in a target, mirrors wide (0: none) */
  RING: number;
  DEC: DecoderShape;
  LAMP: Lamps;
}

export const LAYOUTS: Record<number, Layout> = {
  192: {
    SIZE: 192, TILE: 18, SPR: 16, ORG: 24, BOARD: 144, TEXT: 2, EPIN: 2, RING: 1, DEC: { crop: 1, blurs: [0, 1, 2], bins: [1, 2] },
    LAMP: {
      turnB: [4, 40, 16, 54], turnW: [4, 98, 16, 54], // tall, in the middle of the left column
      crystalT: [2, 6, 20, 16], crystalB: [2, 170, 20, 16],
      castle: { q: [3, 26, 8, 8], k: [13, 26, 8, 8], Q: [3, 156, 8, 8], K: [13, 156, 8, 8] }, // black's pair below its quartz, white's above its own; queenside on the left
      ep: (f) => [24 + 18 * f + 4, 182, 10, 10], // over the bottom seed band, below the winning bar
      banner: [24, 2, 144, 20],
      value: [26, 175, 140, 6],
      noise: [[24, 170, 144, 22], [170, 24, 22, 144]], // below and right of the board, after a 2-px dark margin
    },
  },
  96: {
    SIZE: 96, TILE: 10, SPR: 8, ORG: 8, ORGY: 8, BOARD: 80, TEXT: 1, FONT: 5, EPIN: 1, RING: 0, DEC: { crop: 1, blurs: [0, 1], bins: [1, 2] },
    LAMP: {
      turnB: [1, 13, 5, 30], turnW: [1, 53, 5, 30], // the left column, mirrored about the board's middle
      crystalT: [1, 1, 6, 6], crystalB: [1, 89, 6, 6],
      castle: { q: [1, 8, 2, 3], k: [4, 8, 2, 3], Q: [1, 85, 2, 3], K: [4, 85, 2, 3] },
      ep: (f) => [8 + 10 * f + 3, 88, 4, 5], // a 2 × 3 dot centered under its square
      banner: [8, 0, 80, 7],
      value: [9, 93, 78, 2],
      noise: [[89, 8, 7, 80]], // one band, down the right
    },
  },
};

/** the layout in force */
export interface LiveLayout {
  SIZE: number;
  TILE: number;
  SPR: number;
  ORG: number;
  ORGY: number;
  BOARD: number;
  TEXT: number;
  EPIN: number;
  RING: number;
  /** the banner font's height */
  FH: number;
  DEC: DecoderShape;
  LAMP: Lamps;
  /** the frame's scale against the 192-px one (the noise settings are given in 192-px units) */
  NS: number;
  SPRITES: SpriteSet;
  /**
   * Draw every square as a dark one. A white piece then shows its solid silhouette and a black piece its outline wherever it
   * stands, and the checkerboard, the same on every frame, is not lit at all. The renderer and the decoder both read it.
   */
  FLATBG: boolean;
}

export const live = { FLATBG: false } as LiveLayout;

/** switch frames: the constants, the lamps, the decoder's shape and the sprite set (an unknown size falls back to 192) */
export function layout(size: number): LiveLayout {
  const L = LAYOUTS[size] || LAYOUTS[192]!;
  live.SIZE = L.SIZE;
  live.TILE = L.TILE;
  live.SPR = L.SPR;
  live.ORG = L.ORG;
  live.BOARD = L.BOARD;
  live.TEXT = L.TEXT;
  live.EPIN = L.EPIN;
  live.RING = L.RING;
  live.ORGY = L.ORGY ?? L.ORG;
  live.FH = L.FONT || 7;
  live.DEC = { ...L.DEC };
  live.LAMP = L.LAMP;
  live.NS = live.SIZE / 192;
  live.SPRITES = live.SPR === 16 ? SPRITES16 : SPRITES8;
  return live;
}
layout(192);

/** an empty frame */
export const blank = (): Float32Array => new Float32Array(live.SIZE * live.SIZE);

/** a block of the frame set to val */
export function rect(v: Float32Array, x: number, y: number, w: number, h: number, val: number): void {
  for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) v[j * live.SIZE + i] = val;
}

/** a lit block from (x0, y0) to (x1, y1), clipped by the caller */
export function fillRect(v: Float32Array, x0: number, y0: number, x1: number, y1: number): void {
  for (let j = y0; j < y1; j++) for (let i = x0; i < x1; i++) v[j * live.SIZE + i] = 1;
}

/** the square's color as drawn */
export const sqLight = (sq: number): boolean => (live.FLATBG ? false : isLight(sq));

/** x, y, w, h of a square's tile: rank 8 at the top, file a at the left */
export const sqRect = (sq: number): Rect => [live.ORG + live.TILE * (sq & 7), live.ORGY + live.TILE * (7 - (sq >> 3)), live.TILE, live.TILE];

/** the square under a frame pixel, or −1 */
export function sqAt(x: number, y: number): number {
  const f = Math.floor((x - live.ORG) / live.TILE), r = 7 - Math.floor((y - live.ORGY) / live.TILE);
  return inb(f, r) && x >= live.ORG && y >= live.ORGY ? r * 8 + f : -1;
}
