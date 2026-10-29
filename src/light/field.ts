// A complex N×N field, split into real and imaginary planes, row-major (row = y).

/**
 * Float16Array where the browser has one (Safari 18.2+, Chrome 135+): a phone's light traces are kept in half precision, half the memory
 * (a traced chess move held ~1 GB of fields and an iPhone restarted). Typed as the Float32Array it reads like.
 */
export const Float16: (new (from: ArrayBuffer | number) => Float32Array) | null =
  (globalThis as { Float16Array?: new (from: ArrayBuffer | number) => Float32Array }).Float16Array ?? null;

export interface Field {
  N: number;
  /** double precision on the CPU; the GPU's read-backs stay single precision (all it computes in) — half the memory per trace */
  re: Float64Array | Float32Array;
  im: Float64Array | Float32Array;
}

export function makeField(N: number): Field {
  return { N, re: new Float64Array(N * N), im: new Float64Array(N * N) };
}

export function cloneField(f: Field): Field {
  return { N: f.N, re: f.re.slice(), im: f.im.slice() };
}

/** |E|² per sample. */
export function intensity(f: Field): Float64Array {
  const out = new Float64Array(f.N * f.N);
  for (let k = 0; k < out.length; k++) out[k] = f.re[k]! * f.re[k]! + f.im[k]! * f.im[k]!;
  return out;
}

/** Turn a field 180° in place (both axes reversed). */
export function flip180<T extends Float64Array | Float32Array>(a: T, n: number): T {
  const len = n * n;
  for (let k = 0; k < len / 2; k++) {
    const j = len - 1 - k;
    const t = a[k]!;
    a[k] = a[j]!;
    a[j] = t;
  }
  return a;
}
