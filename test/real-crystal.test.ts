// A real trained crystal (mnist-padrot.safetensors in NC_CRYSTALS, not shipped) vs the training code. Skipped when the crystal or its fixture
// is not on this machine; make_fixtures.py writes the fixture when NC_CRYSTALS points at the crystal.
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CrystalEngine, loadCrystal } from "../src/index.js";
import { fieldErr, fixture, relErr, realCrystal } from "./helpers.js";

const crystal = realCrystal("mnist-padrot.safetensors");
const fx = fileURLToPath(new URL("./fixtures/mnist-real.safetensors", import.meta.url));

describe.runIf(existsSync(crystal) && existsSync(fx))("a real crystal: mnist-padrot", () => {
  it("runs to float32 accuracy and reads out the same picture", async () => {
    const data = await loadCrystal(new Uint8Array(readFileSync(crystal)));
    const f = fixture("mnist-real");
    const eng = new CrystalEngine(data);
    const r = await eng.run(f.t("expect.frame"), f.F);
    expect(fieldErr(r.exit, f.t("expect.exit.re"), f.t("expect.exit.im"))).toBeLessThan(2e-5);
    expect(relErr(eng.readout(r.exit, f.Fr), f.t("expect.readout"))).toBeLessThan(2e-5);
    expect(relErr(eng.picture(r.exit, f.Fr), f.t("expect.picture"))).toBeLessThan(2e-4);
  });
});
