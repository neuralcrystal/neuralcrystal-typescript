// The player's screen: a 96 × 96 one-color OLED (a 0.96″ SSD1317 panel). Screens are drawn in its own pixels into a brightness buffer
// (0..1 a pixel), which a page can show as a 2D tile or use as a 3D texture — one drawing, any number of places.
import { glyph } from "./face57.js";

export const OLED_PX = 96;
/** the lit color of the hardware's screens and tiles */
export const OLED_RGB: readonly [number, number, number] = [230, 102, 71];

export class Panel {
  readonly w: number;
  readonly h: number;
  /** brightness per pixel, row-major */
  readonly px: Float32Array;
  /** pixels drawn in their own colour (a photograph on a colour screen, CIFAR): their RGB, and which pixels carry one */
  readonly tint: Uint8ClampedArray;
  readonly tinted: Uint8Array;
  constructor(w = OLED_PX, h = OLED_PX) {
    this.w = w;
    this.h = h;
    this.px = new Float32Array(w * h);
    this.tint = new Uint8ClampedArray(w * h * 3);
    this.tinted = new Uint8Array(w * h);
  }
  clear(): void {
    this.px.fill(0);
    this.tinted.fill(0);
  }
  /** a filled rectangle at brightness v (1 = lit, 0 = dark), in the screen's colour */
  rect(x: number, y: number, w: number, h: number, v = 1): void {
    for (let yy = Math.max(0, y); yy < Math.min(this.h, y + h); yy++)
      for (let xx = Math.max(0, x); xx < Math.min(this.w, x + w); xx++) {
        this.px[yy * this.w + xx] = v;
        this.tinted[yy * this.w + xx] = 0;
      }
  }
  /** one pixel in its own colour, full brightness */
  color(x: number, y: number, r: number, g: number, b: number): void {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    const k = y * this.w + x;
    this.px[k] = 1;
    this.tinted[k] = 1;
    this.tint[3 * k] = r;
    this.tint[3 * k + 1] = g;
    this.tint[3 * k + 2] = b;
  }
  /** text in the 5 × 7 face at `scale` panel pixels a stroke, one blank column between characters */
  text(s: string, x: number, y: number, scale = 1, v = 1): void {
    for (let i = 0; i < s.length; i++) {
      const g = glyph(s[i]!);
      const gx = x + i * (5 * scale + 1);
      for (let yy = 0; yy < 7; yy++) for (let xx = 0; xx < 5; xx++) if (g[yy]![xx] === "#") this.rect(gx + xx * scale, y + yy * scale, scale, scale, v);
    }
  }
  /** width of a string in panel pixels */
  static textWidth(s: string, scale = 1): number {
    return s.length * 5 * scale + Math.max(0, s.length - 1);
  }
  /** RGBA bytes (for a canvas ImageData or a texture) in the screen's colour */
  rgba(rgb: readonly [number, number, number] = OLED_RGB): Uint8ClampedArray {
    const out = new Uint8ClampedArray(this.w * this.h * 4);
    for (let i = 0; i < this.px.length; i++) {
      const v = this.px[i]!, t = this.tinted[i] === 1;
      out[4 * i] = (t ? this.tint[3 * i]! : rgb[0]) * v;
      out[4 * i + 1] = (t ? this.tint[3 * i + 1]! : rgb[1]) * v;
      out[4 * i + 2] = (t ? this.tint[3 * i + 2]! : rgb[2]) * v;
      out[4 * i + 3] = 255;
    }
    return out;
  }
}
