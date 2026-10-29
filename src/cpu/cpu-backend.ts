// The CPU light path: FFT · H · IFFT per hop, e^{iφ} per plane, the absorbing rim after every sub-hop. Float64 throughout, with
// H and φ taken from the same float32 tables the GPU uses, so the two differ only by the GPU's arithmetic.
import type { LightBackend } from "../light/backend.js";
import type { Field } from "../light/field.js";
import { cloneField } from "../light/field.js";
import type { Geometry } from "../physics/geometry.js";
import type { SectionPlan } from "../physics/plan.js";
import { TransferCache } from "../physics/transfer.js";
import { FFT } from "./fft.js";

export class CpuBackend implements LightBackend {
  readonly name = "cpu";
  private readonly fft: FFT;
  private readonly H: TransferCache;
  private readonly trig = new Map<string, { c: Float64Array; s: Float64Array }>();

  constructor(private readonly g: Geometry) {
    this.fft = new FFT(g.N);
    this.H = new TransferCache(g);
  }

  async runSection(input: Field, plan: SectionPlan): Promise<Field> {
    const E = cloneField(input);
    for (const op of plan.ops) {
      if (op.op === "hop") {
        const H = this.H.get(op.zSub);
        for (let i = 0; i < op.n; i++) this.hop(E, op.pupil && i === 0 ? this.H.getPupil(op.zSub) : H);
      } else this.phase(E, op.key, op.phi);
    }
    return E;
  }

  dispose(): void {
    this.trig.clear();
  }

  private hop(E: Field, H: Float32Array): void {
    const { re, im } = E;
    // the CPU path works in double precision; single-precision fields are only the GPU's read-backs (its traces)
    if (!(re instanceof Float64Array) || !(im instanceof Float64Array)) throw new Error("cpu backend: a field in single precision");
    this.fft.run2d(re, im, false);
    for (let k = 0; k < re.length; k++) {
      const hr = H[2 * k]!;
      const hi = H[2 * k + 1]!;
      const r = re[k]!;
      re[k] = r * hr - im[k]! * hi;
      im[k] = r * hi + im[k]! * hr;
    }
    this.fft.run2d(re, im, true);
    const a = this.g.absorb;
    if (a) absorbOutside(E, a.x0, a.size);
  }

  private phase(E: Field, key: string, phi: () => Float32Array): void {
    let t = this.trig.get(key);
    if (!t) {
      const p = phi();
      t = { c: new Float64Array(p.length), s: new Float64Array(p.length) };
      for (let k = 0; k < p.length; k++) {
        t.c[k] = Math.cos(p[k]!);
        t.s[k] = Math.sin(p[k]!);
      }
      this.trig.set(key, t);
    }
    const { re, im } = E;
    for (let k = 0; k < re.length; k++) {
      const r = re[k]!;
      re[k] = r * t.c[k]! - im[k]! * t.s[k]!;
      im[k] = r * t.s[k]! + im[k]! * t.c[k]!;
    }
  }
}

/** Black sides: zero everything outside the window box. */
export function absorbOutside(E: Field, x0: number, size: number): void {
  const N = E.N;
  for (let y = 0; y < N; y++) {
    const inY = y >= x0 && y < x0 + size;
    for (let x = 0; x < N; x++) {
      if (inY && x >= x0 && x < x0 + size) continue;
      E.re[y * N + x] = 0;
      E.im[y * N + x] = 0;
    }
  }
}
