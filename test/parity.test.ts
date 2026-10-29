// The engine against the training code's Crystal.run on the fixtures test/fixtures/make_fixtures.py writes.
import { describe, expect, it } from "vitest";
import { CrystalEngine, makeField } from "../src/index.js";
import { handoffPicture } from "../src/hardware/handoff.js";
import { fieldErr, fixture, mismatch, relErr } from "./helpers.js";

for (const name of ["quad", "single", "roles", "pupil"]) {
  describe(`parity with the Python package: ${name}`, () => {
    const fx = fixture(name);
    const eng = new CrystalEngine(fx.data);
    const frame = fx.t("expect.frame");

    // the pupil (dmdNA) rides on the input gap's first sub-hop, so the engine's DMD field is the one before it: compared by the passes
    it.skipIf(name === "pupil")("launches exactly the same field from the DMD", () => {
      const last = eng.quad.hops - 1;
      const pic = last === 0 ? frame : fx.t(`expect.hidden.${last - 1}`);
      const E = eng.dmd(pic, fx.F, last > 0);
      expect(fieldErr(E, fx.t("expect.entry.re"), fx.t("expect.entry.im"))).toBe(0);
    });

    it("runs every section to float32 accuracy from the trainer's own input", async () => {
      for (let k = 0; k < eng.quad.hops; k++) {
        const pic = k === 0 ? frame : fx.t(`expect.hidden.${k - 1}`);
        const E = await eng.backend.runSection(eng.dmd(pic, fx.F, k > 0), eng.plan(eng.quad.passSection[k] ?? k));
        expect(fieldErr(E, fx.t(`expect.pass.${k}.re`), fx.t(`expect.pass.${k}.im`))).toBeLessThan(2e-5);
      }
    });

    it("writes the same hidden pictures (bar a rare dither level on a rounding edge)", () => {
      for (let k = 0; k + 1 < eng.quad.hops; k++) {
        const E = makeField(eng.geometry.N);
        E.re.set(fx.t(`expect.pass.${k}.re`));
        E.im.set(fx.t(`expect.pass.${k}.im`));
        const stageInput = k === 0 ? frame : fx.t(`expect.hidden.${k - 1}`);
        const pic = handoffPicture(E, fx.data.model, eng.geometry, eng.quad, fx.F, k, { board: frame, stageInput });
        expect(mismatch(pic, fx.t(`expect.hidden.${k}`))).toBeLessThan(0.01);
      }
    });

    it("runs end to end and reads out the same picture", async () => {
      const r = await eng.run(frame, fx.F);
      expect(fieldErr(r.exit, fx.t("expect.exit.re"), fx.t("expect.exit.im"))).toBeLessThan(2e-5);
      expect(relErr(eng.readout(r.exit, fx.Fr), fx.t("expect.readout"))).toBeLessThan(2e-5);
      expect(relErr(eng.picture(r.exit, fx.Fr), fx.t("expect.picture"))).toBeLessThan(2e-4);
    });
  });
}
