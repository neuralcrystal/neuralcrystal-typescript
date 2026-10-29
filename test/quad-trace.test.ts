// traceStages: a quad crystal's move traced pass by pass (the rays and glass a caller draws) is the same move run() makes — the same hidden
// pictures between the passes, the same exit — whether the traces are cut to the window or kept whole.
import { describe, expect, it } from "vitest";
import { CrystalEngine } from "../src/index.js";
import { fieldErr, fixture } from "./helpers.js";

describe("traceStages", () => {
  const fx = fixture("quad");
  const frame = fx.t("expect.frame");
  for (const crop of [true, false]) {
    it(`makes run()'s move (${crop ? "cut to the window" : "whole grid"})`, async () => {
      const eng = new CrystalEngine(fx.data);
      const r = await eng.run(frame, fx.F), t = await eng.traceStages(frame, fx.F, crop ? eng.window : null);
      expect(t.passes.length).toBe(eng.quad.hops);
      expect(t.hiddens.length).toBe(r.hiddens.length);
      t.hiddens.forEach((h, k) => expect(Array.from(h)).toEqual(Array.from(r.hiddens[k]!)));
      expect(fieldErr(t.run.exit, Float32Array.from(r.exit.re), Float32Array.from(r.exit.im))).toBeLessThan(1e-6); // staged surface by surface, the CPU rounds a little differently (3e-8)
      const w = eng.window, tr = t.passes[0]!;
      expect(tr.arriving[0]!.N).toBe(crop ? w.n : eng.geometry.N);
      expect(tr.leaving.length).toBe(fx.data.model.planes.length);
      if (crop) expect(tr.crop).toEqual({ ...w, N: eng.geometry.N });
    });
  }
});
