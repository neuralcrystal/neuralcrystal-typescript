// The CIFAR player's screen (on the 96 × 96 OLED): the photograph as it is, in colour, at 2× — 64 × 64 pixels
// near the top — and under it the crystal's answer in the screen's lettering; the true class below it, dimmer, only when they differ.
// While it reads, the answer's place shows the read's time in milliseconds, in grey.
import { glyph } from "../../hardware/screen/face57.js";
import type { Panel } from "../../hardware/screen/panel.js";

// THE SCREEN'S COLORS: hits in the housings' wire green (HSL 164°, 85 %, 50 %), misses in the laser's red (#d8401f), the arrows
// white
export const CIFAR_HIT_RGB: readonly [number, number, number] = [19, 236, 178];
export const CIFAR_MISS_RGB: readonly [number, number, number] = [216, 64, 31];
const WHITE: readonly [number, number, number] = [236, 240, 242];
import type { CifarPicture } from "./dataset.js";

export interface CifarScreenView {
  picture: CifarPicture | null;
  /** the crystal's answer, or null while it reads */
  answer: string | null;
  /** whether the answer is the picture's own class (green) or not (red) */
  correct?: boolean;
  /** the picture's own class */
  truth: string | null;
  /** the arrows either side of the photograph (back and forward through the pile) */
  arrows?: boolean;
  /** the pile dealt from: the chevrons take its colour — green for matches, red for misses, white for random */
  mode?: "hit" | "random" | "miss";
  /** set while the crystal reads: the simulated hardware milliseconds so far, counted in the pile's colour, dimmed, where the answer will be; null when not reading */
  reading?: number | null;
}

/** a word in the 5 × 7 face, centred, in its own colour at brightness v */
function say(p: Panel, word: string, y: number, rgb: readonly [number, number, number], v = 1): void {
  const s = word.toUpperCase(), x0 = Math.max(0, Math.round((p.w - (s.length * 6 - 1)) / 2));
  for (let i = 0; i < s.length; i++) {
    const g = glyph(s[i]!);
    for (let yy = 0; yy < 7; yy++) for (let xx = 0; xx < 5; xx++) if (g[yy]![xx] === "#") p.color(x0 + i * 6 + xx, y + yy, rgb[0] * v, rgb[1] * v, rgb[2] * v);
  }
}

export function paintCifarScreen(p: Panel, v: CifarScreenView): void {
  p.clear();
  const X0 = (p.w - 64) >> 1, Y0 = 3;
  if (v.picture)
    for (let y = 0; y < 32; y++)
      for (let x = 0; x < 32; x++) {
        const k = (y * 32 + x) * 3, r = v.picture[k]!, g = v.picture[k + 1]!, b = v.picture[k + 2]!;
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) p.color(X0 + 2 * x + i, Y0 + 2 * y + j, r, g, b);
      }
  // the arrows take the pile's color (its mode)
  const acc = v.mode === "miss" ? CIFAR_MISS_RGB : v.mode === "hit" ? CIFAR_HIT_RGB : WHITE;
  if (v.arrows) {
    // ◀ ▶ in the columns either side of the photograph, centred on it: a press on either side steps through the pile
    const cy = Y0 + 32, h = 6;
    for (let d = 0; d <= h; d++) {
      // CHEVRONS, not filled triangles: ‹ › two pixels thick, 13 tall, in the columns either side of the photograph
      // d: how far from the point, up and down; the point is the chevron's leftmost (‹) or rightmost (›) pixel
      for (const t of [0, 1]) {
        p.color(X0 - 12 + d + t, cy - d, acc[0], acc[1], acc[2]); // ‹ at x 4..11
        p.color(X0 - 12 + d + t, cy + d, acc[0], acc[1], acc[2]);
        p.color(X0 + 64 + 12 - d - t, cy - d, acc[0], acc[1], acc[2]); // › at x 85..92
        p.color(X0 + 64 + 12 - d - t, cy + d, acc[0], acc[1], acc[2]);
      }
    }
  }
  // the answer in green when it is the picture's class, red when not; then, on a miss, the picture's own class under it in a dim green
  if (v.answer) say(p, v.answer, 71, v.correct === false ? CIFAR_MISS_RGB : CIFAR_HIT_RGB);
  // while it reads: a grey count of the milliseconds so far where the answer will be, to show it is thinking
  else if (v.reading !== null && v.reading !== undefined) say(p, `${v.reading < 10 ? v.reading.toFixed(2) : v.reading < 100 ? v.reading.toFixed(1) : Math.round(v.reading)} ms`, 71, acc, 0.65);
  if (v.truth && v.answer && v.truth !== v.answer) say(p, v.truth, 81, CIFAR_HIT_RGB, 0.45);
}

