// The MNIST player screen, in the panel's own 96 × 96 pixels: left, a box you write in with DRAW above and
// HERE below it, showing the hand the crystal is reading; right, the phone pad with the digit the crystal read filled in.
import type { Panel } from "../../hardware/screen/panel.js";
import { Panel as P } from "../../hardware/screen/panel.js";

const PAD: (number | null)[][] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [null, 0, null]];
/** the digit field's window the box shows (rows 2–25, columns 4–23 hold 99.6 % of MNIST's ink), at 2 panel pixels a field pixel */
export const WIN = { r0: 2, r1: 25, c0: 4, c1: 23, k: 2 };
const INH = (WIN.r1 - WIN.r0 + 1) * WIN.k; // 48
const INW = (WIN.c1 - WIN.c0 + 1) * WIN.k; // 40
const BX0 = 47, CW = 16, ROWH = 22, ROWY = 4;
const GRIDB = ROWY + 3 * ROWH - 1; // 69: the rule above the 0 — the box's foot sits on it
const BOXX = Math.round((BX0 - 1 - (INW + 2)) / 2);
const BOXY = GRIDB - (INH + 1);
/** the writable interior, panel pixels */
export const DRAW_BOX = { x: BOXX + 1, y: BOXY + 1, w: INW, h: INH };

export interface DigitScreenView {
  /** the hand on show, 28 × 28 bytes (or 0..1 floats), or null */
  digit: Uint8Array | Float32Array | null;
  /** the digit the crystal read (null: no answer — e.g. the laser is off) */
  read: number | null;
  /** brightness of the ink (a finished digit may be dimmed) */
  ink?: number;
}

export function paintDigitScreen(p: Panel, v: DigitScreenView): void {
  p.clear();
  p.rect(BOXX, BOXY, INW + 2, 1);
  p.rect(BOXX, BOXY + INH + 1, INW + 2, 1);
  p.rect(BOXX, BOXY, 1, INH + 2);
  p.rect(BOXX + INW + 1, BOXY, 1, INH + 2);
  const word = (w: string, y: number): void => p.text(w, Math.round(BOXX + (INW + 2 - P.textWidth(w, 2)) / 2), y, 2);
  word("DRAW", Math.round((BOXY - 14) / 2));
  word("HERE", ROWY + 3 * ROWH + Math.round((ROWH - 14) / 2));
  if (v.digit) paintHand(p, v.digit, v.ink ?? 1);
  for (let r = 0; r < PAD.length; r++)
    for (let c = 0; c < 3; c++) {
      const d = PAD[r]![c];
      if (d === null || d === undefined) continue;
      const cellX = BX0 + c * CW, rowY = ROWY + r * ROWH;
      const gx = cellX + Math.round((CW - 10) / 2), gy = rowY + Math.round((ROWH - 14) / 2);
      if (d === v.read) {
        p.rect(cellX, rowY, CW, ROWH - 1);
        p.text(String(d), gx, gy, 2, 0);
      } else p.text(String(d), gx, gy, 2);
    }
  for (let r = 1; r < PAD.length; r++) p.rect(BX0, ROWY + r * ROWH - 1, 3 * CW, 1);
  for (let c = 1; c < 3; c++) p.rect(BX0 + c * CW - 1, ROWY, 1, GRIDB - ROWY + 1);
}

function paintHand(p: Panel, digit: Uint8Array | Float32Array, ink: number): void {
  const scale = digit instanceof Uint8Array ? 1 : 255;
  const sw = WIN.c1 - WIN.c0 + 1, sh = WIN.r1 - WIN.r0 + 1, k = WIN.k;
  const x0 = DRAW_BOX.x + Math.round((INW - sw * k) / 2), y0 = DRAW_BOX.y + Math.round((INH - sh * k) / 2);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      const v = digit[(y + WIN.r0) * 28 + x + WIN.c0]! * scale;
      if (v > 24) p.rect(x0 + x * k, y0 + y * k, k, k, ink * Math.min(1, v / 255));
    }
}

/**
 * Where a press starts a stroke (so drawing does not drag the view instead): the whole
 * writing side of the screen — the box, its frame, DRAW and HERE — and a few pixels past the screen's edge, not just the box's inside,
 * which is under a hundred screen pixels wide from the default view. The stroke is clamped into the box (panelToField(…, true)).
 */
export function inDrawZone(px: number, py: number): boolean {
  const M = 4;
  return px >= -M && px < BX0 - 1 && py >= -M && py < 96 + M;
}

/** A point on the panel → the digit field's pixel coordinates (a stroke off its edge clamped to the box), or null outside it. */
export function panelToField(px: number, py: number, clampOnly = false): { x: number; y: number } | null {
  const inside = px >= DRAW_BOX.x && px < DRAW_BOX.x + DRAW_BOX.w && py >= DRAW_BOX.y && py < DRAW_BOX.y + DRAW_BOX.h;
  if (!inside && !clampOnly) return null;
  const cx = Math.max(0, Math.min(DRAW_BOX.w, px - DRAW_BOX.x)), cy = Math.max(0, Math.min(DRAW_BOX.h, py - DRAW_BOX.y));
  return { x: WIN.c0 + cx / WIN.k, y: WIN.r0 + cy / WIN.k };
}
