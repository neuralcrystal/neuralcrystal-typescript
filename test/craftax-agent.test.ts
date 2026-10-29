import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Field } from "../src/light/field.js";
import { CraftaxAgent } from "../src/problems/craftax/agent.js";
import { frame } from "../src/problems/craftax/frame.js";
import { pictures } from "../src/problems/craftax/layout.js";
import { cellPowers, readCraftax } from "../src/problems/craftax/read.js";
import type { CraftaxState } from "../src/problems/craftax/state.js";
import { actionMask } from "../src/problems/craftax/state.js";

// No fixture crystal has split heads, so the light is a stand-in: an engine whose runs record their input pictures and return fixed
// exit fields. That checks the agent's plumbing (pictures, memory, mask, sampling); the optics are CrystalEngine's own tests.
const rng = (seed: number) => () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32);
const N = 256, F = 192, U = 4;
const field = (r: () => number): Field => {
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let i = 0; i < re.length; i++) { re[i] = r() - 0.5; im[i] = r() - 0.5; }
  return { N, re, im };
};

function stub(split: boolean) {
  const r = rng(3), cell = () => Math.floor(r() * 4096), vec = (n: number, s = 2) => Array.from({ length: n }, () => (r() * 2 - 1) * s);
  // memory mirrors in the bottom-right corner (blank in the board picture), unit 3's bits each a block of two mirrors
  const mem = Array.from({ length: U }, (_, u) => Array.from({ length: 4 }, (_, j) => (u === 3 ? [F * F - 1 - 8 * j, F * F - 2 - 8 * j] : F * F - 40 - 4 * u - j)));
  const mapA = Array.from({ length: F * F }, (_, i) => (i < (F * F) / 2 ? i : -1)), mapB = Array.from({ length: F * F }, (_, i) => F * F - 1 - i);
  const craftax = {
    frame: F, units: U, mem, cells: { act: Array.from({ length: 43 }, (_, a) => (a % 7 === 0 ? [cell(), cell()] : cell())), z: Array.from({ length: U }, cell), hc: Array.from({ length: U }, cell) },
    e2eHead: { gA: vec(43, 3), al: [vec(U), vec(U)], be: [vec(U), vec(U)] }, ...(split ? { split: { test: "A", mapA, mapB } } : {}),
  };
  const model = { N, planes: [], tr: { craftax }, quad: split ? { split: { in: [0, 1], mid: [], out: [2, 3] } } : {} };
  const exits = [field(r), field(r)], seen: Float32Array[][] = [];
  const engine = {
    data: { model }, geometry: { N, winFrac: 0.875 },
    runSplit: async (fa: Float32Array, fb: Float32Array) => { seen.push([fa.slice(), fb.slice()]); return { exits, entry: exits[0]!, hiddens: [] }; },
    run: async (fr: Float32Array) => { seen.push([fr.slice()]); return { exit: exits[0]!, entry: exits[0]!, hiddens: [] }; },
  };
  return { engine: engine as never, exits, seen, mem };
}

const { cases } = JSON.parse(readFileSync(new URL("./fixtures/craftax-frames.json", import.meta.url), "utf8")) as { cases: { state: CraftaxState }[] };
const state = cases[1]!.state;
const bitsOf = (pic: Float32Array, mem: (number | number[])[][], u: number) => mem[u]!.map((m) => ([] as number[]).concat(m).map((p) => pic[p]));

describe("the Craftax agent", () => {
  it("writes the memory into the second picture and steps it as h + z (hc − h), 4-bit", async () => {
    const s = stub(true), agent = new CraftaxAgent(s.engine, { temperature: 0 }), L = agent.layout;
    const want = readCraftax(L, cellPowers({ N, winFrac: 0.875 }, s.exits[0]!), cellPowers({ N, winFrac: 0.875 }, s.exits[1]!));
    const m1 = await agent.act(state);
    expect(Array.from(m1.memoryIn)).toEqual([0, 0, 0, 0]);
    const h1 = Array.from({ length: U }, (_, u) => Math.fround(Math.round(Math.min(1, Math.max(0, want.z[u]! * want.hc[u]!)) * 15) / 15));
    expect(Array.from(agent.memory)).toEqual(h1);
    expect(h1.filter((v) => v > 0 && v < 1).length).toBeGreaterThan(1); // levels other than 0 and 1, so the bits are a real check
    // the first picture is the board gathered by mapA; the second carries the memory as each unit's 4 bits, high bit first
    const board = frame(state);
    expect(s.seen[0]![0]!.slice(0, 100)).toEqual(Float32Array.from(board.slice(0, 100)));
    await agent.act(state);
    const pic = s.seen[1]![1]!;
    for (let u = 0; u < U; u++) {
      const q = Math.round(h1[u]! * 15), bits = [3, 2, 1, 0].map((k) => (q >> k) & 1);
      expect(bitsOf(pic, s.mem, u)).toEqual(bits.map((b) => (u === 3 ? [b, b] : [b])));
    }
    expect(pictures(L, board, h1)[1]).toEqual(pic);
    agent.reset();
    expect(Array.from(agent.memory)).toEqual([0, 0, 0, 0]);
  });

  it("plays the most likely allowed move at temperature 0, and samples by the softmax otherwise", async () => {
    const s = stub(true), agent = new CraftaxAgent(s.engine, { temperature: 0 });
    const m = await agent.act(state, false);
    const best = m.scores.indexOf(Math.max(...m.scores));
    expect(m.action).toBe(best);
    const mask = Array.from({ length: 43 }, (_, i) => i !== best && i % 2 === 0);
    const m2 = await agent.act(state, mask);
    expect(mask[m2.action]).toBe(true);
    expect(m2.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (let i = 0; i < 43; i++) if (!mask[i]) expect(m2.probs[i]).toBe(0);
    const sampler = new CraftaxAgent(stub(true).engine, { temperature: 1, random: () => 0 });
    const m3 = await sampler.act(state, mask);
    expect(m3.action).toBe(mask.indexOf(true)); // r = 0 lands on the first allowed move
    expect(actionMask(state)[(await agent.act(state)).action]).toBe(true); // by default, the game's own rules
  });

  it("holds the memory and plays noop asleep or resting", async () => {
    const s = stub(true), agent = new CraftaxAgent(s.engine);
    await agent.act(state);
    const h = agent.memory.slice(), m = await agent.act({ ...state, sleeping: true });
    expect(m.action).toBe(0);
    expect(agent.memory).toEqual(h);
  });

  it("runs a single-head crystal on one picture, the memory cells cleared and redrawn", async () => {
    const s = stub(false), agent = new CraftaxAgent(s.engine, { temperature: 0 });
    expect(agent.layout.split).toBe(null);
    await agent.act(state);
    const h = agent.memory.slice();
    await agent.act(state);
    expect(s.seen[1]!.length).toBe(1);
    expect(s.seen[1]![0]).toEqual(pictures(agent.layout, frame(state), h)[0]);
  });
});
