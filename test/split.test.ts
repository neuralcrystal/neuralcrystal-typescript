// Split crystals (quad.split, as Craftax's crystals use): runSplit and traceSplit on the quad fixture with two split layouts laid over it —
// shared output heads, and the earlier split where each head has its own branch. The hashes were taken from the engine before it was split
// into engine/, so they pin the exact bytes; traceSplit must also end on runSplit's answer.
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { CrystalEngine } from "../src/index.js";
import { fieldErr, fixture } from "./helpers.js";

const hash = (a: Float32Array | Float64Array): string => createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex").slice(0, 16);

const CASES = {
  "shared outputs": {
    split: { in: [0, 1], mid: [1], out: [2, 0] },
    exits: ["a871243f37fae6a0a4ab2aba5c52b69d", "b0fae3550a5a17f9987b0b0118419b78"],
    hiddens: ["cc0c5d3644fd6a24", "f6489deee8a9045a", "0ffe609a4b3ed4c2"],
    turns: "0012*2",
  },
  "a branch per head": {
    split: { in: [0, 1], mid: [2], out: [2, 0], branch: [[1, 2], [0]] },
    exits: ["7282420698c0553c1f98404cb4b966e6", "eeb3075058ad7988dbed0de8b7f7bb51"],
    hiddens: ["cc0c5d3644fd6a24", "f6489deee8a9045a", "2e2eeb9e752bbc31", "72660afadf7bae8a"],
    turns: "001223*",
  },
};

describe("split crystals", () => {
  const fx = fixture("quad"), F = fx.F;
  const fa = fx.t("expect.frame"), fb = Float32Array.from(fa, (v, i) => (i % 7 === 0 ? 1 - v : v));
  for (const [name, c] of Object.entries(CASES)) {
    const data = { ...fx.data, model: { ...fx.data.model, quad: { ...(fx.data.model.quad as object), split: c.split } } } as typeof fx.data;
    it(`runSplit gives the same bytes as before (${name})`, async () => {
      const r = await new CrystalEngine(data).runSplit(fa, fb, F);
      expect(r.exits.map((x) => hash(x.re as Float64Array) + hash(x.im as Float64Array))).toEqual(c.exits);
      expect(r.hiddens.map(hash)).toEqual(c.hiddens);
    });
    it(`runSplit with reuseReads gives the same answer (${name})`, async () => {
      const e = new CrystalEngine(data);
      e.reuseReads = true;
      await e.runSplit(fb, fa, F); // a step before, so the kept fields hold something else
      const r = await e.runSplit(fa, fb, F);
      expect(r.exits.map((x) => hash(Float64Array.from(x.re)) + hash(Float64Array.from(x.im)))).toEqual(c.exits);
    });
    it(`traceSplit ends on runSplit's answer (${name})`, async () => {
      const e = new CrystalEngine(data), r = await e.runSplit(fa, fb, F), t = await e.traceSplit(fa, fb, F);
      t.exits.forEach((x, k) => expect(fieldErr(x, Float32Array.from(r.exits[k]!.re), Float32Array.from(r.exits[k]!.im))).toBeLessThan(1e-6));
      expect(t.pictures.map(hash)).toEqual(c.hiddens);
      expect(t.passes.map((p) => `${p.turn}${p.aim ? "*" : ""}`).join("")).toBe(c.turns);
    });
  }
});
