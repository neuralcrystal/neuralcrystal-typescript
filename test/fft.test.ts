import { describe, expect, it } from "vitest";
import { FFT } from "../src/index.js";

describe("FFT", () => {
  it("matches a direct DFT and inverts", () => {
    const n = 16;
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 1.3) + 0.2 * i);
    const im = Float64Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    const r2 = re.slice();
    const i2 = im.slice();
    new FFT(n).run(r2, i2, 0, 1, false);
    for (let k = 0; k < n; k++) {
      let sr = 0;
      let si = 0;
      for (let t = 0; t < n; t++) {
        const a = (-2 * Math.PI * k * t) / n;
        sr += re[t]! * Math.cos(a) - im[t]! * Math.sin(a);
        si += re[t]! * Math.sin(a) + im[t]! * Math.cos(a);
      }
      expect(r2[k]).toBeCloseTo(sr, 10);
      expect(i2[k]).toBeCloseTo(si, 10);
    }
    new FFT(n).run(r2, i2, 0, 1, true);
    for (let t = 0; t < n; t++) expect(r2[t]).toBeCloseTo(re[t]!, 12);
  });
  it("rejects sizes that are not a power of two", () => {
    expect(() => new FFT(12)).toThrow();
  });
});
