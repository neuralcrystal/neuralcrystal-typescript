// The seed bands: fresh random mirrors every time a position is shown, so the crystal's play need not be deterministic, or a rank code
// in their place that asks for the crystal's n-th choice.
import type { Rect } from "./layout.js";
import { fillRect, live } from "./layout.js";

// The seed bands, in 192-px units (the compact frame scales them by live.NS): each band is a lit FRAME-px border, a NOISEDARK-px dark
// pad, then the pattern. MODE "2d" draws QR-style cells (NOISECELL bits a side, a NOISEGAP-px dark gutter between cells), "1d" a bar code
// along the band; NOISEBLK is a bit's or a bar's size and NOISE a unit's chance of being on.
export const NOISE = 0.5;
export const NOISEBLK = 2;
export const NOISECELL = 3;
export const NOISEGAP = 1;
export const NOISEFRAME = 0;
export const NOISEDARK = 3;
export const NOISEMODE = "2d";

/** a band's lit frame, then its interior past the frame and the dark pad (null when nothing is left) */
function bandInterior(v: Float32Array, [x, y, w, h]: Rect, frame: number, dark: number): Rect | null {
  for (let j = y; j < y + h; j++)
    for (let i = x; i < x + w; i++) if (i < x + frame || i >= x + w - frame || j < y + frame || j >= y + h - frame) v[j * live.SIZE + i] = 1;
  const ix = x + frame + dark, iy = y + frame + dark, iw = w - 2 * (frame + dark), ih = h - 2 * (frame + dark);
  return iw <= 0 || ih <= 0 ? null : [ix, iy, iw, ih];
}

/** the seed bands: both, or the right column alone when the bottom one is the winning bar's */
export const seedBands = (rightOnly?: boolean): Rect[] => (rightOnly ? live.LAMP.noise.filter((bd) => bd[3] > bd[2]) : live.LAMP.noise);

/**
 * The seed noise: in each band a lit `frame`-px border, a dark `dark`-px pad, then the pattern. Mode "2d" draws QR-style cells
 * (`cell` × `cell` bits of side `blk`, a `gap`-px dark gutter between cells), "1d" a bar code of `blk`-wide units along the band. A unit
 * is on with probability `fill`. The settings are in 192-px units, scaled to the live frame.
 */
export function renderNoise(v: Float32Array, fill: number, rand: () => number, blk: number, cell: number, gap: number, frame?: number, dark?: number, mode?: string, bands?: Rect[] | null): Float32Array {
  blk = Math.max(1, (blk | 0) || 1);
  cell = Math.max(1, (cell | 0) || NOISECELL);
  gap = Math.max(0, gap | 0);
  frame = Math.max(0, frame ?? NOISEFRAME);
  dark = Math.max(0, dark ?? NOISEDARK);
  mode = mode || NOISEMODE;
  const ns = live.NS || 1;
  if (ns !== 1) {
    blk = Math.max(1, Math.round(blk * ns));
    gap = Math.max(0, Math.round(gap * ns));
    frame = Math.floor(frame * ns);
    dark = Math.floor(dark * ns);
  }
  const step = cell * blk + gap; // one cell plus its gutter
  for (const band of bands || live.LAMP.noise) {
    const box = bandInterior(v, band, frame, dark);
    if (!box) continue;
    const [ix, iy, iw, ih] = box;
    if (mode === "1d") {
      // bars across the band's long axis, each spanning the short one
      if (iw >= ih) {
        for (let bx = ix; bx < ix + iw; bx += blk) if (rand() < fill) fillRect(v, bx, iy, Math.min(bx + blk, ix + iw), iy + ih);
      } else {
        for (let by = iy; by < iy + ih; by += blk) if (rand() < fill) fillRect(v, ix, by, ix + iw, Math.min(by + blk, iy + ih));
      }
    } else {
      // every bit draws a number, clipped or not, so the pattern stays the same for a seed
      for (let cy = iy; cy < iy + ih; cy += step)
        for (let cx = ix; cx < ix + iw; cx += step)
          for (let r = 0; r < cell; r++)
            for (let c = 0; c < cell; c++) {
              if (rand() >= fill) continue;
              const bx = cx + c * blk, by = cy + r * blk;
              fillRect(v, bx, by, Math.min(bx + blk, ix + iw), Math.min(by + blk, iy + ih));
            }
    }
  }
  return v;
}

/** the seed bands as a signal: K cells along each band, cell n (1-based) lit, "play your n-th choice"; a horizontal band counts from the left, a vertical one from the bottom */
export function renderCode(v: Float32Array, n: number, K: number, opts?: { noiseFrame?: number; noiseDark?: number; bands?: Rect[] | null; noBottom?: boolean }): Float32Array {
  const o = opts || {};
  K = Math.max(1, (K | 0) || 1);
  n = Math.min(K, Math.max(1, (n | 0) || 1));
  const ns = live.NS || 1, frame = Math.floor(Math.max(0, o.noiseFrame ?? NOISEFRAME) * ns), dark = Math.floor(Math.max(0, o.noiseDark ?? NOISEDARK) * ns), gap = Math.max(1, Math.round(2 * ns));
  for (const band of o.bands || (o.noBottom ? seedBands(true) : live.LAMP.noise)) {
    const box = bandInterior(v, band, frame, dark);
    if (!box) continue;
    const [ix, iy, iw, ih] = box;
    if (iw >= ih) {
      const cw = (iw - gap * (K - 1)) / K, x0 = Math.round(ix + (n - 1) * (cw + gap));
      fillRect(v, x0, iy, Math.round(x0 + cw), iy + ih);
    } else {
      // rank 1, the best move, is the lowest cell
      const chh = (ih - gap * (K - 1)) / K, y0 = Math.round(iy + (K - n) * (chh + gap));
      fillRect(v, ix, y0, ix + iw, Math.round(y0 + chh));
    }
  }
  return v;
}

