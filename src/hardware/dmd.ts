// The DMD: an F×F picture (amplitudes 0..1) → the input field. Nearest-neighbour blocks in the frame's box, the polarity switch,
// the 180° pre-flip — as the Python package's Crystal.dmd does it.
import type { Geometry } from "../physics/geometry.js";
import type { Field } from "../light/field.js";
import { flip180, makeField } from "../light/field.js";

export interface DmdOptions {
  /** the imager's polarity switch: the frame shown inverted (input frames only, never a quad's hidden picture) */
  inv: boolean;
  /** the 180° pre-flip */
  flip: boolean;
}

/** Sample y of a box of `side` samples → frame pixel floor((y + ½)·F/side), a half-pixel nearest neighbour. */
export function nearestIndex(side: number, F: number): Int32Array {
  const fi = new Int32Array(side);
  for (let y = 0; y < side; y++) fi[y] = Math.min(F - 1, Math.floor(((y + 0.5) * F) / side));
  return fi;
}

/** out: a field to write into (cleared first) instead of a new one — the trace pool's */
export function dmdField(g: Geometry, frame: Float32Array, F: number, o: DmdOptions, hidden = false, out?: Field): Field {
  const { N, side, x0 } = g;
  const E = out && out.N === N ? out : makeField(N);
  if (E === out) {
    E.re.fill(0);
    E.im.fill(0);
  }
  const fi = nearestIndex(side, F);
  const inv = o.inv && !hidden;
  for (let y = 0; y < side; y++) {
    const row = fi[y]! * F;
    for (let x = 0; x < side; x++) {
      const v = frame[row + fi[x]!]!;
      E.re[(x0 + y) * N + x0 + x] = inv ? 1 - v : v;
    }
  }
  if (o.flip) flip180(E.re, N);
  return E;
}
