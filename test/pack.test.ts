// The web form of a crystal (packCrystal): cropped to each map's window, phase wrapped to 16 or 8 bits, read back by readCrystal.
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CrystalEngine, loadCrystal, packCrystal, readCrystal } from "../src/index.js";
import type { CrystalData } from "../src/index.js";
import { fieldErr, fixture, realCrystal } from "./helpers.js";

/** the largest error in exp(iφ) between two crystals' maps — all the light sees of a phase */
function phasorErr(a: CrystalData, b: CrystalData): number {
  let worst = 0;
  a.maps.forEach((secs, i) =>
    secs.forEach((m, k) => {
      const o = b.maps[i]?.[k];
      if (!m || !o) {
        if (!!m !== !!o) worst = Infinity;
        return;
      }
      for (let j = 0; j < m.length; j++) worst = Math.max(worst, Math.hypot(Math.cos(m[j]!) - Math.cos(o[j]!), Math.sin(m[j]!) - Math.sin(o[j]!)));
    }),
  );
  return worst;
}

function synthetic(): CrystalData {
  const n = 64, maps = [0, 1].map((p) =>
    [0, 1].map((k) => {
      const m = new Float32Array(n * n);
      for (let y = 16; y < 48; y++) for (let x = 16; x < 48; x++) m[y * n + x] = -300 * ((x - 32) ** 2 + (y - 32) ** 2) / 512 + p + 0.7 * k; // a lens, unwrapped, as trained
      return m;
    }),
  );
  return { model: { N: n, dx_um: 1, planes: [{ kind: "learned" }, { kind: "learned" }] } as unknown as CrystalData["model"], maps, mapN: [n, n] };
}

describe("packCrystal", () => {
  it("round-trips 16-bit maps to within one step of a turn, cropped to the window", () => {
    const d = synthetic(), bytes = packCrystal(d, { bits: 16 }), back = readCrystal(bytes);
    expect(back.mapN).toEqual(d.mapN);
    expect(phasorErr(d, back)).toBeLessThan((2 * Math.PI) / 65536);
    expect(bytes.length).toBeLessThan(2 * 2 * 32 * 32 * 2 + 4096); // four 32² windows of 2 bytes, plus the header
  });
  it("round-trips 8-bit maps to within one step", () => {
    const d = synthetic(), back = readCrystal(packCrystal(d, { bits: 8 }));
    expect(phasorErr(d, back)).toBeLessThan((2 * Math.PI) / 256);
  });
});

const crystal = realCrystal("mnist-padrot.safetensors");
const fx = fileURLToPath(new URL("./fixtures/mnist-real.safetensors", import.meta.url));
describe.runIf(existsSync(crystal) && existsSync(fx))("packCrystal on mnist-padrot", () => {
  it("the 16-bit web form gives the same light as the full file", async () => {
    const data = await loadCrystal(new Uint8Array(readFileSync(crystal)));
    const packed = readCrystal(packCrystal(data, { bits: 16 }));
    const f = fixture("mnist-real");
    const r = await new CrystalEngine(packed).run(f.t("expect.frame"), f.F);
    expect(fieldErr(r.exit, f.t("expect.exit.re"), f.t("expect.exit.im"))).toBeLessThan(2e-4);
  });
});
