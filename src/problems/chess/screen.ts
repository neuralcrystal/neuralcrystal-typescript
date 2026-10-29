// The chess player's screen (on the 96 × 96 OLED): the board as you see it — white at the near edge — in the
// compact 96-px layout the crystal's own art comes from: 10-px squares, its 8-px pieces (white as its lit silhouette, black as its
// outline), the light squares washed faintly so the board reads. Marks at their own brightness: the round dots where the piece you picked
// up can go, the crystal's last move, the piece in flight; and a word in the strip above or below the board, on its player's edge —
// the crystal's (black's) clock above, yours below (YOUR MOVE, CHECK, who won).
import type { Panel } from "../../hardware/screen/panel.js";
import type { ChessState } from "./board.js";
import { chess as C } from "./board.js";
import type { GhostPiece } from "./ghost.js";
import { GHOST_LEVEL } from "./ghost.js";

export interface ChessScreenView {
  state: ChessState;
  /** the square of the piece you picked up, and where it can go */
  pick?: number;
  targets?: number[];
  /** the crystal's last move */
  last?: [number, number] | null;
  /** the words: black's edge (the top strip) and white's (the bottom) */
  top?: string | null;
  bottom?: string | null;
  /** a piece in your hand: where it came from and the pointer, in screen pixels */
  drag?: { from: number; fx: number; fy: number } | null;
  /** the crystal's move in flight: the position before it, the move and how far along (0..1) */
  anim?: { pre: ChessState; from: number; to: number; u: number } | null;
  /** the look-ahead's line now being read, as ghost pieces over the board (ghost.ts) */
  ghost?: GhostPiece[] | null;
  /** the value meter while the crystal looks ahead: the board it just read, as its expected score 0..1, drawn as a ghost-red bar
   *  right of the board, filled up from the bottom (top: black's, unused) */
  meter?: { v: number; top: boolean } | null;
  /** how bright the light squares are washed, 0..1 (the chess.lightSquares setting) */
  lightSq?: number;
}

/** a piece's 8-px art at (ox, oy): white lit solid (its silhouette's lit pixels), black as its outline */
function sprite(p: Panel, piece: number, ox: number, oy: number, v = 1, under = false): void {
  const S = C.SPRITES[C.PIECE[Math.abs(piece) - 1]!];
  if (!S) return;
  const white = piece > 0, n = S.m.length;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const lit = white ? S.m[y]![x] === "#" && S.w[y]![x] === "#" : S.o[y]![x] === "#";
      if (lit) {
        if (under) {
          // a ghost ADDS its 15 % to what is under it: seen on the washed light squares (0.14) as well as the dark, never dimming a real piece
          const k = (oy + y) * p.w + ox + x;
          if (k >= 0 && k < p.px.length) p.px[k] = Math.min(1, (p.px[k] ?? 0) + v);
        }
        else p.rect(ox + x, oy + y, 1, 1, v);
      }
    }
}

/** a round dot at a square's centre, anti-aliased (4 × 4 samples a pixel), 1.6 px in radius — the checkers screen's */
function dot(p: Panel, sq: number, v: number): void {
  const [x, y, w, h] = C.sqRect(sq), cx = x + w / 2, cy = y + h / 2, r = 1.6, S = 4;
  for (let yy = Math.floor(cy - r); yy <= Math.ceil(cy + r); yy++)
    for (let xx = Math.floor(cx - r); xx <= Math.ceil(cx + r); xx++) {
      let hit = 0;
      for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) if ((xx + (i + 0.5) / S - cx) ** 2 + (yy + (j + 0.5) / S - cy) ** 2 <= r * r) hit++;
      if (hit) p.rect(xx, yy, 1, 1, Math.max(p.px[yy * p.w + xx] ?? 0, (v * hit) / (S * S)));
    }
}

function box(p: Panel, sq: number, v: number): void {
  const [x, y, w, h] = C.sqRect(sq);
  p.rect(x, y, w, 1, v);
  p.rect(x, y + h - 1, w, 1, v);
  p.rect(x, y, 1, h, v);
  p.rect(x + w - 1, y, 1, h, v);
}

/** a word in the 5 × 5 face, centred at row cy, in the strip above or below the board */
function say(p: Panel, word: string, cy: number): void {
  const t = new Float32Array(C.SIZE * C.SIZE);
  C.drawText(t, word, C.SIZE / 2, cy, 1);
  for (let i = 0; i < t.length; i++) if (t[i]! > 0) p.px[i] = 1;
}

export function paintChessScreen(p: Panel, v: ChessScreenView): void {
  p.clear();
  const anim = v.anim ?? null, board = anim ? anim.pre : v.state, T = C.TILE, sp = (T - C.SPR) >> 1;
  for (let sq = 0; sq < 64; sq++)
    if (C.isLight(sq)) {
      const [x, y] = C.sqRect(sq);
      p.rect(x, y, T, T, v.lightSq ?? 0.2);
    }
  // the board's edge, faint
  p.rect(C.ORG - 1, C.ORGY - 1, 8 * T + 2, 1, 0.3);
  p.rect(C.ORG - 1, C.ORGY + 8 * T, 8 * T + 2, 1, 0.3);
  p.rect(C.ORG - 1, C.ORGY - 1, 1, 8 * T + 2, 0.3);
  p.rect(C.ORG + 8 * T, C.ORGY - 1, 1, 8 * T + 2, 0.3);
  if (v.meter && !anim) {
    // in the free strip right of the board: a 2-px bar the board's height in the ghost pieces' red, filled up from the bottom — a + above
    // it (a win for the crystal) and a − below (a loss) — over a fainter track
    const x = C.ORG + 8 * T + 2, H = 8 * T, n = Math.round(Math.max(0, Math.min(1, v.meter.v)) * H), top = C.ORGY, bot = C.ORGY + H - 1;
    for (let i = 0; i < H; i++) p.rect(x, bot - i, 2, 1, i < n ? 2 * GHOST_LEVEL : GHOST_LEVEL / 3);
    const cx = x + 0.5;
    p.rect(Math.floor(cx) - 1, top - 5, 4, 1, 2 * GHOST_LEVEL); // + : a bar and a post, above the meter
    p.rect(x, top - 6, 2, 3, 2 * GHOST_LEVEL);
    p.rect(Math.floor(cx) - 1, bot + 4, 4, 1, 2 * GHOST_LEVEL); // − : a bar, below it
  }
  if (v.last && !anim) for (const s of v.last) box(p, s, 0.55);
  for (let sq = 0; sq < 64; sq++) {
    const pc = board.sq[sq]!;
    if (!pc || (v.drag && sq === v.drag.from)) continue; // the piece in hand is drawn under the pointer
    if (anim && (sq === anim.from || (sq === anim.to && anim.u >= 0.55))) continue; // in flight; a taken piece goes as the mover arrives, not before
    const [x, y] = C.sqRect(sq);
    sprite(p, pc, x + sp, y + sp);
  }
  // the look-ahead's line, faintly, over the real board (the real pieces stay; a ghost only lights what is darker than it)
  if (v.ghost && !anim)
    for (const g of v.ghost) {
      const [x, y] = C.sqRect(g.sq);
      sprite(p, g.piece, x + sp, y + sp, GHOST_LEVEL, true);
    }
  if (anim) {
    // the crystal's piece sliding from its square to the next, easing in and out; the rook of a castling jumps with it
    const u = anim.u, e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u), pc = anim.pre.sq[anim.from]!;
    const [ax, ay] = C.sqRect(anim.from), [bx, by] = C.sqRect(anim.to);
    sprite(p, pc, Math.round(ax + sp + (bx - ax) * e), Math.round(ay + sp + (by - ay) * e));
  } else if (v.drag) {
    for (const t of v.targets ?? []) dot(p, t, 0.85);
    const pc = v.state.sq[v.drag.from]!;
    if (pc) sprite(p, pc, Math.round(v.drag.fx - C.SPR / 2), Math.round(v.drag.fy - C.SPR / 2));
  } else if (v.pick !== undefined && v.pick >= 0) {
    // the piece picked up gets no box: the dots where it can go say which it is
    for (const t of v.targets ?? []) dot(p, t, 0.85);
  }
  // the words: the top strip is black's (the crystal's) edge, rows 1–5; the bottom strip white's (yours), rows 90–94
  if (v.top) say(p, v.top, 3.5);
  if (v.bottom && !anim) say(p, v.bottom, 92.5);
}

/** the square under a screen pixel (−1: none) */
export function chessScreenSquare(x: number, y: number): number {
  return C.sqAt(x, y);
}
