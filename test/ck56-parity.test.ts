// A real checkers crystal (checkers-ck56.safetensors in NC_CRYSTALS, 168 MB, not shipped) against the Python package on one position: the three hidden
// pictures between the passes and the move map. HEAVY — a 1024² quad solve on the CPU, ~13 s — so it runs only with NC_HEAVY=1.
// Measured 2026-09-29: move map within 2.2e-6 of the port, hidden pictures identical, the same move (22-17).
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { realCrystal } from "./helpers.js";
import { CrystalEngine, checkers, checkersKit, readCrystal } from "../src/index.js";

const crystal = realCrystal("checkers-ck56.safetensors");
const fx = fileURLToPath(new URL("./fixtures/ck56-one.json", import.meta.url));

describe.runIf(process.env["NC_HEAVY"] === "1" && existsSync(crystal) && existsSync(fx))("a real checkers crystal: ck56", () => {
  it("matches the Python package: the hidden pictures exactly, the move map to float32 accuracy, the same move", async () => {
    const Y = JSON.parse(readFileSync(fx, "utf8")) as { pos: string; powers: number[]; hidden: number[][] };
    const data = readCrystal(new Uint8Array(readFileSync(crystal))), eng = new CrystalEngine(data), kit = checkersKit(data);
    const st = checkers.parse(Y.pos), run = await eng.run(kit.frame(st), kit.frameSize);
    run.hiddens.forEach((h, k) => expect(Array.from(h)).toEqual(Y.hidden[k]));
    const p = kit.powers(eng, run.exit);
    let num = 0, den = 0;
    for (let i = 0; i < p.length; i++) {
      num += (p[i]! - Y.powers[i]!) ** 2;
      den += Y.powers[i]! ** 2;
    }
    expect(Math.sqrt(num / den)).toBeLessThan(1e-5);
    expect(kit.readWith(eng, run, st, { temp: 0 }).cell).toBe(Y.powers.indexOf(Math.max(...Y.powers)));
  }, 120000);
});
