// The checkers player's screen: the game as YOU see it — your army at the near edge, one orientation for the
// whole game — on a real board: the 32 squares that can never hold a piece washed faintly, the pieces at full. The first player (black)
// is drawn as rings and red as solid discs. Marks, all in the screen's one colour at their own brightness: the
// piece you picked up and where it can go, the pieces that owe a jump, the crystal's last move; and a word only when the board cannot say
// it (THINKING, YOUR MOVE, JUMP AGAIN, who won).
import type { Panel } from "../../hardware/screen/panel.js";
import type { CheckersState } from "./board.js";
import { checkers as C } from "./board.js";
import type { CheckersGhost } from "./ghost.js";
import { CHECKERS_GHOST_LEVEL } from "./ghost.js";

export interface CheckersScreenView {
  state: CheckersState;
  /** your side (+1 black, −1 red): your army is drawn at the near edge; 0 — nobody's seat, black at the near edge */
  seat: 1 | -1 | 0;
  /** the square of the piece you have picked up (real squares), and the squares it can go to */
  pick?: number;
  targets?: number[];
  /** squares whose pieces must jump */
  mustJump?: number[];
  /** the crystal's last move, real squares */
  last?: [number, number] | null;
  /** the word to show, or null (THINKING rides half a cell lower: a new king on the back rank stays in sight) */
  word?: string | null;
  /** the crystal's clock under THINKING ("0.68 MS") */
  clock?: string | null;
  /** a piece in your hand: the square it came from (real) and where the pointer is, in screen pixels */
  drag?: { from: number; fx: number; fy: number } | null;
  /** the crystal's move in flight: the position before it, the move (real squares) and how far along (0..1) */
  anim?: { pre: CheckersState; from: number; to: number; cap: number; u: number } | null;
  /** the look-ahead's line now being read, as ghost pieces over the board (ghost.ts) */
  ghost?: CheckersGhost[] | null;
  /** the value meter while it looks ahead: the board it just read, the crystal's expected score 0..1 (a faint bar up the right edge) */
  meter?: number | null;
  /** where the meter runs: up the right edge, or along the back edge (the top) */
  meterAt?: "right" | "back";
}

/** a 5 × 5 face for the screen's words (the chess board's FONT5): letters, digits and the point */
const FONT5: Record<string, string[]> = {
  A: [".###.", "#...#", "#####", "#...#", "#...#"], B: ["####.", "#...#", "####.", "#...#", "####."], C: [".####", "#....", "#....", "#....", ".####"], D: ["####.", "#...#", "#...#", "#...#", "####."],
  E: ["#####", "#....", "####.", "#....", "#####"], H: ["#...#", "#...#", "#####", "#...#", "#...#"], I: ["#####", "..#..", "..#..", "..#..", "#####"], K: ["#..#.", "#.#..", "##...", "#.#..", "#..#."],
  L: ["#....", "#....", "#....", "#....", "#####"], M: ["#...#", "##.##", "#.#.#", "#...#", "#...#"], N: ["#...#", "##..#", "#.#.#", "#..##", "#...#"], O: [".###.", "#...#", "#...#", "#...#", ".###."],
  R: ["####.", "#...#", "####.", "#.#..", "#..#."], S: [".####", "#....", ".###.", "....#", "####."], T: ["#####", "..#..", "..#..", "..#..", "..#.."], U: ["#...#", "#...#", "#...#", "#...#", ".###."],
  W: ["#...#", "#...#", "#.#.#", "##.##", "#...#"], Y: ["#...#", ".#.#.", "..#..", "..#..", "..#.."],
  F: ["#####", "#....", "####.", "#....", "#...."], G: [".####", "#....", "#.###", "#...#", ".###."], J: ["....#", "....#", "....#", "#...#", ".###."],
  P: ["####.", "#...#", "####.", "#....", "#...."], Q: [".###.", "#...#", "#...#", "#..#.", ".##.#"], V: ["#...#", "#...#", "#...#", ".#.#.", "..#.."],
  X: ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"], Z: ["#####", "...#.", "..#..", ".#...", "#####"],
  "0": [".###.", "#...#", "#...#", "#...#", ".###."], "1": ["..#..", ".##..", "..#..", "..#..", ".###."], "2": ["####.", "....#", ".###.", "#....", "#####"],
  "3": ["####.", "....#", ".###.", "....#", "####."], "4": ["#...#", "#...#", "#####", "....#", "....#"], "5": ["#####", "#....", "####.", "....#", "####."],
  "6": [".###.", "#....", "####.", "#...#", ".###."], "7": ["#####", "....#", "...#.", "..#..", "..#.."], "8": [".###.", "#...#", ".###.", "#...#", ".###."],
  "9": [".###.", "#...#", ".####", "....#", ".###."], ".": [".....", ".....", ".....", ".....", "..#.."],
};

/** a word over a black panel: 5 × 5 letters a pixel apart, centred, two dark pixels all round */
function say(p: Panel, word: string, y0: number): void {
  const chars = [...word.toUpperCase()], CW = 6, w = chars.length * CW - 1, h = 5, x0 = Math.max(1, Math.round((C.SIZE - w) / 2));
  p.rect(x0 - 2, y0 - 2, w + 4, h + 4, 0);
  chars.forEach((ch, i) => {
    const gl = FONT5[ch];
    if (!gl) return;
    for (let ry = 0; ry < h; ry++) for (let rx = 0; rx < 5; rx++) if (gl[ry]![rx] === "#") p.rect(x0 + i * CW + rx, y0 + ry, 1, 1, 1);
  });
}

/**
 * A round dot at a square's centre, where a picked-up piece can go (circles read better than boxes). A 2.2-pixel dot comes
 * out as a 4 × 4 block on the 96-pixel panel; this one is 1.6 pixels in radius and anti-aliased — each pixel lit by how much of it
 * the disc covers (4 × 4 samples) — so it reads as a circle.
 */
function dot(p: Panel, sq: number, v: number): void {
  const [x, y, w, h] = C.sqRect(sq), cx = x + w / 2, cy = y + h / 2, r = 1.6, S = 4;
  for (let yy = Math.floor(cy - r); yy <= Math.ceil(cy + r); yy++)
    for (let xx = Math.floor(cx - r); xx <= Math.ceil(cx + r); xx++) {
      let hit = 0;
      for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) if ((xx + (i + 0.5) / S - cx) ** 2 + (yy + (j + 0.5) / S - cy) ** 2 <= r * r) hit++;
      if (hit) p.rect(xx, yy, 1, 1, Math.max(p.px[yy * p.w + xx] ?? 0, (v * hit) / (S * S)));
    }
}

/** a piece's art drawn with its top-left at (ox, oy) */
function art(p: Panel, a: string[], ox: number, oy: number, v: number): void {
  for (let y = 0; y < a.length; y++) for (let x = 0; x < a[y]!.length; x++) if (a[y]![x] === "#") p.rect(ox + x, oy + y, 1, 1, v);
}

/** the square drawn at a real square's place for this seat (your side at the near edge: black starts at the top, so its seat turns the board) */
export function seatSquare(sq: number, seat: 1 | -1 | 0): number {
  return seat >= 0 ? C.flip(sq) : sq;
}

/** the real square under a screen pixel (−1: none) */
export function screenSquare(x: number, y: number, seat: 1 | -1 | 0): number {
  const d = C.sqAtPx(x, y);
  return d < 0 ? -1 : seatSquare(d, seat); // the seat's turn is its own inverse
}

function box(p: Panel, sq: number, v: number, out = 0): void {
  const [x, y, w, h] = C.sqRect(sq);
  p.rect(x - out, y - out, w + 2 * out, 1, v);
  p.rect(x - out, y + h - 1 + out, w + 2 * out, 1, v);
  p.rect(x - out, y - out, 1, h + 2 * out, v);
  p.rect(x + w - 1 + out, y - out, 1, h + 2 * out, v);
}

export function paintCheckersScreen(p: Panel, v: CheckersScreenView): void {
  p.clear();
  const T = C.TILE;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) if (((r + c) & 1) === 0) p.rect(c * T, r * T, T, T, 0.14);
  const frame = C.blank(), anim = v.anim ?? null, board = anim ? anim.pre : v.state;
  for (let sq = 0; sq < 32; sq++) {
    const pc = board.sq[sq]!;
    if (!pc || (v.drag && sq === v.drag.from)) continue; // the piece in hand is drawn under the pointer
    if (anim && (sq === anim.from || (sq === anim.cap && anim.u >= 0.55))) continue; // in flight; the jumped man goes once it is passed
    const art = C.GLYPH_OF(-pc); // black (+) as rings, red (−) as solid discs
    if (art) C.glyph(frame, seatSquare(sq, v.seat), art);
  }
  for (let i = 0; i < frame.length; i++) if (frame[i]! > 0) p.px[i] = 1;
  // the look-ahead's line, faintly: each ghost ADDS its level to what is under it, never dimming a real piece
  if (v.ghost && !anim) {
    const gf = C.blank();
    for (const gp of v.ghost) {
      const a = C.GLYPH_OF(-gp.piece);
      if (a) C.glyph(gf, seatSquare(gp.sq, v.seat), a);
    }
    for (let i = 0; i < gf.length; i++) if (gf[i]! > 0) p.px[i] = Math.min(1, (p.px[i] ?? 0) + CHECKERS_GHOST_LEVEL);
  }
  // the value meter: the board fills the screen, so a 2-px band over its edge squares — up the right edge, or along the back edge (the
  // crystal's side, the top), filling from the left — at 28 % (between the 22 % and the 35 % tried) over a fainter track; a piece under it
  // stays lit
  if (v.meter !== null && v.meter !== undefined && !anim) {
    const back = v.meterAt === "back", L = back ? p.w : p.h, n = Math.round(Math.max(0, Math.min(1, v.meter)) * L), BW = 2;
    for (let i = 0; i < L; i++)
      for (let b = 0; b < BW; b++) {
        const k = back ? b * p.w + i : (p.h - 1 - i) * p.w + (p.w - 1 - b);
        p.px[k] = Math.max(p.px[k] ?? 0, i < n ? 0.28 : CHECKERS_GHOST_LEVEL / 3);
      }
  }

  for (const s of v.mustJump ?? []) {
    box(p, seatSquare(s, v.seat), 0.9);
    box(p, seatSquare(s, v.seat), 0.45, 1);
  }
  if (v.last && !anim) for (const s of v.last) box(p, seatSquare(s, v.seat), 0.6);
  if (v.drag) {
    // the piece under the pointer, in its own art, and the dots where it can land
    for (const t of v.targets ?? []) dot(p, seatSquare(t, v.seat), 0.85);
    const a = C.GLYPH_OF(-v.state.sq[v.drag.from]!);
    if (a) art(p, a, Math.round(v.drag.fx - a.length / 2), Math.round(v.drag.fy - a.length / 2), 1);
  } else if (anim) {
    // the crystal's piece sliding from its square to the next, easing in and out
    const u = anim.u, e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u), a = C.GLYPH_OF(-anim.pre.sq[anim.from]!);
    const [ax, ay] = C.sqRect(seatSquare(anim.from, v.seat)), [bx, by] = C.sqRect(seatSquare(anim.to, v.seat));
    if (a) art(p, a, Math.round(ax + C.INS + (bx - ax) * e), Math.round(ay + C.INS + (by - ay) * e), 1);
  } else if (v.pick !== undefined && v.pick >= 0) {
    box(p, seatSquare(v.pick, v.seat), 1);
    for (const t of v.targets ?? []) dot(p, seatSquare(t, v.seat), 0.85);
  }
  // the word (none while a piece is in flight: it arrives with the piece); THINKING half a cell lower, the clock under it
  const word = anim ? null : (v.word ?? null);
  if (word) {
    const dn = word === "THINKING" ? C.TILE >> 1 : 0;
    say(p, word, word === "YOUR MOVE" ? 22 : 4 + dn);
    if (word === "THINKING" && v.clock) say(p, v.clock, 11 + dn);
  }
}
