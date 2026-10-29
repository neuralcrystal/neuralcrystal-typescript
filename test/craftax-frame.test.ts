import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { craftaxAtlas, unpackBits } from "../src/problems/craftax/atlas.js";
import { ATLAS_JSON } from "../src/problems/craftax/atlas.gen.js";
import { frame, viewCoarse } from "../src/problems/craftax/frame.js";
import type { CraftaxState } from "../src/problems/craftax/state.js";
import { actionMask } from "../src/problems/craftax/state.js";

// Real game states from the Craftax WebAssembly engine (plus two edited ones for the status corners), each with the picture the game
// page's frame.js drew for it, its coarse view, and the engine's own action mask (null for the edited states).
interface Case { label: string; mode: number; stamp: boolean; state: CraftaxState; frame: string; coarse: number[]; mask: string | null }
const { cases } = JSON.parse(readFileSync(new URL("./fixtures/craftax-frames.json", import.meta.url), "utf8")) as { cases: Case[] };

describe("the Craftax board picture", () => {
  it("unpacks the atlas", () => {
    const A = craftaxAtlas();
    expect(A.frame).toBe(192);
    expect(A.glyphBits.length).toBe(A.n_glyphs * 16 * 16);
    expect(A.staticBits.length).toBe(192 * 192);
  });

  it("is the data in atlas.json", () => {
    expect(ATLAS_JSON).toEqual(JSON.parse(readFileSync(new URL("../src/problems/craftax/atlas.json", import.meta.url), "utf8")));
  });

  for (const c of cases)
    it(`draws the same picture as frame.js: ${c.label}`, () => {
      const got = frame(c.state, { mode: c.mode, stamp: c.stamp }), want = unpackBits(c.frame, 192 * 192);
      let off = 0;
      for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) off++;
      expect(off).toBe(0);
      expect(Array.from(viewCoarse(c.state))).toEqual(c.coarse);
    });

  it("allows the moves the game engine allows", () => {
    const real = cases.filter((c) => c.mask);
    expect(real.length).toBeGreaterThan(5);
    for (const c of real) expect(actionMask(c.state).map(Number).join(""), c.label).toBe(c.mask);
  });

  it("allows only noop asleep", () => {
    const s = { ...cases[0]!.state, sleeping: true };
    expect(actionMask(s).filter(Boolean).length).toBe(1);
  });
});
