// Handwriting → a picture for the DMD. Strokes are round-capped thick lines in the field's own pixels; the raster is their coverage
// (supersampled). normalised() then prepares it the way MNIST prepared its digits — the ink's longer side sized to 20 of the 28,
// the centre of mass moved to the field's centre (done on the way out so the screen still shows the hand
// as written).

export interface StrokePoint {
  x: number;
  y: number;
  /** pen width, field pixels */
  w: number;
}

const SS = 4; // subsamples a pixel, each axis

export class StrokeInput {
  private strokes: StrokePoint[][] = [];
  constructor(readonly size = 28) {}

  begin(p: StrokePoint): void {
    this.strokes.push([p]);
  }
  to(p: StrokePoint): void {
    const s = this.strokes[this.strokes.length - 1];
    if (s) s.push(p);
    else this.begin(p);
  }
  clear(): void {
    this.strokes = [];
  }
  get empty(): boolean {
    return this.strokes.length === 0;
  }
  /** strokes drawn */
  get count(): number {
    return this.strokes.length;
  }
  /** drop every stroke before stroke k (the older ones belonged to the previous digit) */
  keepFrom(k: number): void {
    this.strokes = this.strokes.slice(k);
  }

  /** coverage 0..1, size × size, at `ss` subsamples a pixel per axis (the fine raster normalised() works from) */
  raster(ss = 1, from = 0): Float32Array {
    return this.coverage(this.size * ss, ss, from);
  }

  /** The ink sized and centred like an MNIST digit, size × size, 0..1; null when nothing is drawn. */
  normalised(box = 20, threshold = 24 / 255, from = 0): Float32Array | null {
    return normaliseInk(this.coverage(this.size * SS, SS, from), this.size * SS, this.size, box, threshold);
  }

  private coverage(n: number, ss: number, from = 0): Float32Array {
    const out = new Float32Array(n * n);
    for (const s of this.strokes.slice(from)) {
      const segs = s.length === 1 ? [[s[0]!, s[0]!]] : s.slice(1).map((p, i) => [s[i]!, p]);
      for (const [a, b] of segs as [StrokePoint, StrokePoint][]) this.segment(out, n, ss, a, b);
    }
    return out;
  }

  /** mark every subsample within the pen's half-width of segment a–b (fine raster of n × n, ss fine pixels a field pixel) */
  private segment(out: Float32Array, n: number, ss: number, a: StrokePoint, b: StrokePoint): void {
    const r = (Math.max(a.w, b.w) * ss) / 2;
    const ax = a.x * ss, ay = a.y * ss, bx = b.x * ss, by = b.y * ss;
    const xlo = Math.max(0, Math.floor(Math.min(ax, bx) - r)), xhi = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + r));
    const ylo = Math.max(0, Math.floor(Math.min(ay, by) - r)), yhi = Math.min(n - 1, Math.ceil(Math.max(ay, by) + r));
    const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
    const q = 2; // subsamples per fine pixel for the coverage estimate
    for (let y = ylo; y <= yhi; y++)
      for (let x = xlo; x <= xhi; x++) {
        let hit = 0;
        for (let j = 0; j < q; j++)
          for (let i = 0; i < q; i++) {
            const px = x + (i + 0.5) / q, py = y + (j + 0.5) / q;
            const t = L2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2)) : 0;
            const ex = px - (ax + t * dx), ey = py - (ay + t * dy);
            const wr = ((a.w + t * (b.w - a.w)) * ss) / 2;
            if (ex * ex + ey * ey <= wr * wr) hit++;
          }
        const k = y * n + x;
        out[k] = Math.max(out[k]!, hit / (q * q));
      }
  }
}

/**
 * Any picture of ink (hi: n × n, 0..1) → size × size, prepared the way MNIST prepared its digits: the ink's longer side sized to
 * `box` of the `size`, its centre of mass moved to the centre. null when there is no ink.
 */
export function normaliseInk(hi: Float32Array, n: number, size: number, box = 20, threshold = 24 / 255): Float32Array | null {
  let x0 = n, y0 = n, x1 = -1, y1 = -1, mx = 0, my = 0, m = 0;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const v = hi[y * n + x]!;
      if (v > threshold) {
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
      if (v) {
        mx += (x + 0.5) * v;
        my += (y + 0.5) * v;
        m += v;
      }
    }
  if (x1 < 0 || !m) return null;
  const q = n / size; // fine pixels an output pixel
  const k = (n * box) / size / Math.max(x1 - x0 + 1, y1 - y0 + 1);
  const cx = mx / m, cy = my / m;
  const S = Math.max(1, Math.round(q)) * 2; // sample points an output pixel, each axis
  const out = new Float32Array(size * size);
  for (let Y = 0; Y < size; Y++)
    for (let X = 0; X < size; X++) {
      let t = 0;
      for (let j = 0; j < S; j++)
        for (let i = 0; i < S; i++) {
          const u = (X + (i + 0.5) / S) * q, v = (Y + (j + 0.5) / S) * q;
          t += bilinear(hi, n, (u - n / 2) / k + cx - 0.5, (v - n / 2) / k + cy - 0.5);
        }
      out[Y * size + X] = t / (S * S);
    }
  return out;
}

function bilinear(a: Float32Array, n: number, x: number, y: number): number {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const at = (xx: number, yy: number): number => (xx < 0 || yy < 0 || xx >= n || yy >= n ? 0 : a[yy * n + xx]!);
  return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}
