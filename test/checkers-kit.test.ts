// The checkers kit: the move cells sit where the trainer puts them and read the exit field's light; and, when NC_DATA points at the
// original data set (a folder holding checkers/board.js, the original untyped rules, and checkers/test-000.json), its frames and reads
// match theirs on real data-set positions. Those tests skip when NC_DATA is unset or the files are missing.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";
import { checkers, checkersKit } from "../src/index.js";
import type { CheckersState, CrystalData, CrystalEngine, Field } from "../src/index.js";

const DATA = process.env["NC_DATA"];
const SRC = DATA ? join(DATA, "checkers/board.js") : "";
const TEST = DATA ? join(DATA, "checkers/test-000.json") : "";
const have = existsSync(SRC);

/** the original board.js, run as a browser page runs it (a script that sets ncCheckers on the global) */
function labBoard(): typeof checkers {
  const g: Record<string, unknown> = {};
  runInNewContext(readFileSync(SRC, "utf8"), g);
  return g["ncCheckers"] as typeof checkers;
}

/** a seeded random source */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

const fakeData = { model: { N: 1024, dx_um: 1, wholeGlass: true, winUm: 512, planes: [], checkers: { playT: 1, topK: 5, legalMask: true } } } as unknown as CrystalData;
const fakeEngine = { geometry: { N: 1024, winFrac: 0.5 } } as unknown as CrystalEngine;

describe("checkers kit", () => {
  const kit = checkersKit(fakeData);

  it("puts the move cells where the trainer does: 8-sample bands at 260 + 16d on a 512 µm window of 1024", () => {
    const b = kit.bands(fakeEngine);
    expect(b.side).toBe(8);
    expect(b.lo).toEqual(Array.from({ length: 32 }, (_, d) => 260 + 16 * d));
  });

  it("reads each cell's light off the exit field", () => {
    const N = 1024, E: Field = { N, re: new Float32Array(N * N), im: new Float32Array(N * N) };
    const b = kit.bands(fakeEngine), from = 9, to = 13;
    for (let y = b.lo[from]!; y < b.lo[from]! + b.side; y++) for (let x = b.lo[to]!; x < b.lo[to]! + b.side; x++) E.re[y * N + x] = 1;
    const p = kit.powers(fakeEngine, E);
    expect(p[from * 32 + to]).toBe(64);
    expect(p.reduce((a, v) => a + v, 0)).toBe(64);
  });
});

describe.runIf(have)("checkers kit against the original board.js", () => {
  const positions: CheckersState[] = existsSync(TEST) ? (JSON.parse(readFileSync(TEST, "utf8")) as { pos: string[] }).pos.slice(0, 300).map((p) => checkers.parse(p)) : [checkers.start()];
  const lab = have ? labBoard() : (null as unknown as typeof checkers); // a skipped suite still runs this body
  const kit = checkersKit(fakeData);

  it("draws the same frame for every position", () => {
    for (const st of positions) expect(Array.from(kit.frame(st))).toEqual(Array.from(lab.render(lab.parse(checkers.str(st)))));
  });

  it("plays the same move for the same light (brightest, and sampled with the same dice)", () => {
    const r = rng(7);
    for (const st of positions.slice(0, 120)) {
      const p = Float64Array.from({ length: 1024 }, () => r() ** 3);
      const a = checkers.readMoveMap(p, st, { temp: 0 }), b = lab.readMoveMap(p, lab.parse(checkers.str(st)), { temp: 0 });
      expect(a?.uci).toBe(b?.uci);
      const s1 = rng(11), s2 = rng(11);
      expect(checkers.readMoveMap(p, st, { ...kit.play, rng: s1 })?.uci).toBe(lab.readMoveMap(p, lab.parse(checkers.str(st)), { ...kit.play, rng: s2 })?.uci);
    }
  });
});
