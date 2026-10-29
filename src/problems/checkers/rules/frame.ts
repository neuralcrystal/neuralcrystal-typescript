// The frame: a position as a binary picture for the DMD. No checkerboard: the board itself is the same on every frame, so only the 32
// playable squares are marked, each by a corner bracket (a top arm and a left stroke). A piece is a glyph inset inside the bracket's
// corner: the side to move as a solid disc, the opponent as a ring (the frame is canonical, see view). The one piece of state a board
// cannot show is a jump already in progress, so the square whose piece must go on jumping is closed into a box.
//
// Everything is drawn in board units, 8 cells of TILE, and px() maps them onto the frame, which may be bigger: at 256 the 32-unit
// cell fills it exactly, at other sizes the board sits centered at a whole-pixel scale.
import type { GlyphSet, Layout } from "./art.js";
import { LAYOUTS } from "./art.js";
import type { CheckersState } from "./position.js";
import { colOf, rowOf, sqAt, view } from "./position.js";

/** the current frame: its side, frame pixels per board unit, the board's offset, and the cell layout */
export const frame: { SIZE: number; S: number; OX: number; OY: number; L: Layout } = { SIZE: 96, S: 1, OX: 0, OY: 0, L: LAYOUTS[12]! };

export function setLayout(tile: number): void {
  const L = LAYOUTS[tile];
  if (!L) throw new Error("no checkers layout with a " + tile + "-unit cell");
  frame.L = L;
}

/** size the frame: the layout whose 8 cells fill it exactly, else the smallest that fits, scaled by whole pixels and centered */
export function setFrame(size?: number): void {
  size = Math.round(size || 8 * frame.L.TILE);
  const tiles = Object.keys(LAYOUTS).map(Number).sort((a, b) => b - a);
  const exact = tiles.find((t) => 8 * t === size);
  setLayout(exact || tiles.filter((t) => 8 * t <= size!).pop() || tiles[tiles.length - 1]!);
  const T = frame.L.TILE;
  size = Math.max(8 * T, size);
  frame.S = Math.max(1, Math.floor(size / (8 * T)));
  frame.SIZE = size;
  frame.OX = frame.OY = Math.floor((size - 8 * T * frame.S) / 2);
}

export const blank = (size?: number): Float32Array => new Float32Array(size ? size * size : frame.SIZE * frame.SIZE);
/** a canonical square's top-left, in board units */
export const cellXY = (sq: number): [number, number] => [colOf(sq) * frame.L.TILE, rowOf(sq) * frame.L.TILE];

/** a square's rect [x, y, w, h] in frame pixels */
export function sqRect(sq: number): [number, number, number, number] {
  const [x, y] = cellXY(sq), t = frame.L.TILE * frame.S;
  return [frame.OX + x * frame.S, frame.OY + y * frame.S, t, t];
}

/** the square under a frame pixel (−1: none) */
export function sqAtPx(x: number, y: number): number {
  const t = frame.L.TILE * frame.S;
  return sqAt(Math.floor((y - frame.OY) / t), Math.floor((x - frame.OX) / t));
}

/** light one board unit (S × S frame pixels), clipped to the frame */
function px(v: Float32Array, x: number, y: number, q: number): void {
  const { S, SIZE } = frame, X = frame.OX + x * S, Y = frame.OY + y * S;
  for (let j = 0; j < S; j++)
    for (let i = 0; i < S; i++) {
      const fx = X + i, fy = Y + j;
      if (fx < 0 || fy < 0 || fx >= SIZE || fy >= SIZE) continue;
      v[fy * SIZE + fx] = q;
    }
}

const pick = (g: GlyphSet, p: number): string[] | null => (p === 1 ? g.man : p === 2 ? g.king : p === -1 ? g.foe : p === -2 ? g.foek : null);
/** a piece's art, in canonical pieces (+ the side to move) */
export const GLYPH_OF = (p: number): string[] | null => pick(frame.L.GLYPH, p);
/** the same on the square that must jump again */
export const GLYPH7_OF = (p: number): string[] | null => pick(frame.L.MARK, p);

// The jump box is BOX pixels thick on all four sides. Its top and left walls reach into the squares above and to the left, which are
// always unplayable (playable squares are never adjacent), and its bottom and right walls use the cell's own gutter, so it collides
// with nothing. The 12- and 24-unit boxes keep BOX−1 outside the cell at the top and left and all of BOX inside at the bottom and
// right; the 32-unit box sits evenly, the same amount outside on all four sides.
function bracket(v: Float32Array, sq: number, closed: boolean): void {
  const [x0, y0] = cellXY(sq), { ARM: A, TILE: T, STROKE, BOX, BOXEVEN } = frame.L, W = STROKE || 1, B = BOX || 2;
  for (let j = 0; j < W; j++)
    for (let i = 0; i < A; i++) {
      px(v, x0 + i, y0 + j, 1);
      px(v, x0 + j, y0 + i, 1);
    }
  if (!closed) return;
  const out = BOXEVEN ? (B - 1) >> 1 : B - 1, iEnd = BOXEVEN ? T + out : T, back = BOXEVEN ? T - (B - out) : T - B;
  for (let i = -out; i < iEnd; i++)
    for (let j = 0; j < B; j++) {
      px(v, x0 + i, y0 - out + j, 1);
      px(v, x0 + i, y0 + back + j, 1);
      px(v, x0 - out + j, y0 + i, 1);
      px(v, x0 + back + j, y0 + i, 1);
    }
}

/** a piece's art written into a frame on a canonical square, at the layout's inset from the bracket's corner */
export function glyph(v: Float32Array, sq: number, art: string[]): void {
  const [x0, y0] = cellXY(sq), o = frame.L.INS, n = art.length;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (art[y]![x] === "#") px(v, x0 + o + x, y0 + o + y, 1);
}

export interface RenderOptions {
  /** draw into an existing frame */
  v?: Float32Array;
  /** false: brackets only (the empty board) */
  pieces?: boolean;
  /** never close a bracket into the jump box */
  noJumpBox?: boolean;
  /** the frame's side, if not the current one (96) */
  size?: number;
}

/** the frame: a Float32Array(size²) of 0 / 1 */
export function render(st: CheckersState, opts?: RenderOptions): Float32Array {
  const o = opts || {}, was = o.size && o.size !== frame.SIZE ? frame.SIZE : 0;
  if (was) setFrame(o.size);
  try {
    const v = o.v || blank(), V = view(st);
    for (let sq = 0; sq < 32; sq++) {
      const marked = !o.noJumpBox && V.jump === sq;
      bracket(v, sq, marked);
      if (o.pieces === false) continue;
      const p = V.sq[sq]!, art = marked ? GLYPH7_OF(p) || GLYPH_OF(p) : GLYPH_OF(p);
      if (art) glyph(v, sq, art);
    }
    return v;
  } finally {
    if (was) setFrame(was);
  }
}

/** the share of the frame the picture lights */
export function lit(st: CheckersState): number {
  const v = render(st);
  let n = 0;
  for (let i = 0; i < v.length; i++) if (v[i]! > 0) n++;
  return n / v.length;
}
