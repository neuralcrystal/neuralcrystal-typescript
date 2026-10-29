// A complex FFT on the CPU: iterative radix-2, in place, float64. The reference the WebGPU kernels are checked against, and the
// fallback for browsers without WebGPU compute. Unnormalised forward; the inverse divides by n (numpy / torch conventions).

export class FFT {
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly rev: Uint32Array;
  constructor(readonly n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = -Math.sin((2 * Math.PI * k) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** In-place transform of n values at re[off + k·stride], im[…]. inverse: conjugate twiddles and divide by n. */
  run(re: Float64Array, im: Float64Array, off: number, stride: number, inverse: boolean): void {
    const n = this.n;
    const { cos, sin, rev } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i]!;
      if (j <= i) continue;
      const a = off + i * stride;
      const b = off + j * stride;
      const tr = re[a]!;
      const ti = im[a]!;
      re[a] = re[b]!;
      im[a] = im[b]!;
      re[b] = tr;
      im[b] = ti;
    }
    const sgn = inverse ? -1 : 1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = cos[k * step]!;
          const wi = sgn * sin[k * step]!;
          const a = off + (start + k) * stride;
          const b = off + (start + k + half) * stride;
          const xr = re[b]! * wr - im[b]! * wi;
          const xi = re[b]! * wi + im[b]! * wr;
          re[b] = re[a]! - xr;
          im[b] = im[a]! - xi;
          re[a] = re[a]! + xr;
          im[a] = im[a]! + xi;
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i++) {
        re[off + i * stride]! /= n;
        im[off + i * stride]! /= n;
      }
    }
  }

  /** 2-D transform of an n×n row-major field: every row, then every column. */
  run2d(re: Float64Array, im: Float64Array, inverse: boolean): void {
    const n = this.n;
    for (let y = 0; y < n; y++) this.run(re, im, y * n, 1, inverse);
    for (let x = 0; x < n; x++) this.run(re, im, x, n, inverse);
  }
}
