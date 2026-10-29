import { describe, expect, it } from "vitest";
import type { CrystalData } from "../src/crystal/types.js";
import type { Field } from "../src/light/field.js";
import { chessKit } from "../src/problems/chess/kit.js";
import { cellPowers, readCraftax } from "../src/problems/craftax/read.js";
import type { CraftaxLayout } from "../src/problems/craftax/layout.js";

// a seeded generator, so the synthetic exits are the same every run
const rng = (seed: number) => () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32);

function exitField(N: number, r: () => number): Field {
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let i = 0; i < re.length; i++) { re[i] = r() - 0.5; im[i] = r() - 0.5; }
  return { N, re, im };
}

// a synthetic layout: 43 moves (some a block of cells), a TARGET region, where regions, 8 memory units
function layout(r: () => number): CraftaxLayout {
  const U = 8, cell = () => Math.floor(r() * 4096), block = () => [cell(), cell(), cell()];
  const act = Array.from({ length: 44 }, (_, a) => (a % 5 === 0 ? block() : cell()));
  const vec = (n: number) => Array.from({ length: n }, () => r() * 4 - 2);
  return {
    F: 192, units: U, mem: [], memAll: [], split: null,
    cells: { act, z: Array.from({ length: U }, cell), hc: Array.from({ length: U }, cell), where: [block(), cell()] },
    head: { gA: vec(46), al: [vec(U), vec(U)], be: [vec(U), vec(U)] },
  };
}

// the reading as the game's crystal worker does it today: the chess kit's cell powers, then the logits and gates inline
function workerReading(L: CraftaxLayout, p1: Float64Array, p2: Float64Array) {
  const regionLog = (p: Float64Array, c: number | number[]) => (Array.isArray(c) ? Math.log(c.reduce((s, k) => s + p[k]!, 0) + 1e-12) : Math.log(p[c]! + 1e-12));
  const sig = (x: number) => 1 / (1 + Math.exp(-x)), U = L.units, h = L.head;
  const scores = L.cells.act.map((c, a) => regionLog(p1, c) + h.gA[a]!);
  const lz = L.cells.z.map((c) => regionLog(p2, c)), lh = L.cells.hc.map((c) => regionLog(p2, c));
  let mean = 0; for (let u = 0; u < U; u++) mean += lz[u]! + lh[u]!; mean /= 2 * U;
  const z = lz.map((v, u) => sig(h.al[0][u]! * (v - mean) + h.be[0][u]!)), hc = lh.map((v, u) => sig(h.al[1][u]! * (v - mean) + h.be[1][u]!));
  return { scores, z, hc };
}

describe("reading the Craftax crystal", () => {
  for (const [N, winFrac] of [[256, 0.875], [1024, 0.875], [512, 0.8]] as const)
    it(`reads the same cell light as the chess kit (N ${N}, window ${winFrac})`, () => {
      const r = rng(N), engine = { geometry: { N, winFrac } } as never, kit = chessKit({ model: { N, planes: [] } } as unknown as CrystalData);
      for (let k = 0; k < 2; k++) {
        const E = exitField(N, r);
        expect(Array.from(cellPowers({ N, winFrac }, E))).toEqual(Array.from(kit.powers(engine, E)));
      }
    });

  it("reads the scores and the memory update as the crystal worker does", () => {
    const r = rng(7), N = 256, g = { N, winFrac: 0.875 }, kit = chessKit({ model: { N, planes: [] } } as unknown as CrystalData);
    const L = layout(r), E1 = exitField(N, r), E2 = exitField(N, r);
    const p1 = kit.powers({ geometry: g } as never, E1), p2 = kit.powers({ geometry: g } as never, E2);
    const got = readCraftax(L, cellPowers(g, E1), cellPowers(g, E2)), want = workerReading(L, p1, p2);
    expect(Array.from(got.scores)).toEqual(want.scores);
    expect(Array.from(got.z)).toEqual(want.z);
    expect(Array.from(got.hc)).toEqual(want.hc);
    expect(got.where!.length).toBe(2);
    expect(got.where![1]).toBe(Math.log(p1[L.cells.where![1] as number]! + 1e-12) + L.head.gA[45]!);
  });
});
