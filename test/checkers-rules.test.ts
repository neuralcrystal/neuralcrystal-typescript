// The checkers rules: a few positions checked by hand (the opening, a forced capture, a jump chain), and golden digests of everything the
// rules, the frame and the move reader produce over a few thousand positions from seeded random games. The digests were taken from the
// original untyped rules before they were retyped, so these modules must reproduce them byte for byte.
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { checkers } from "../src/problems/checkers/board.js";
import { lit } from "../src/problems/checkers/rules/frame.js";
import { cellName, classify, legalMask, pseudoCells, setMoveLayout, softTargets, target } from "../src/problems/checkers/rules/move-map.js";

// the wrapper plus the rules it leaves out, under the names the original rules object used
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const api: any = { ...checkers, lit, cellName, classify, legalMask, pseudoCells, setMoveLayout, softTargets, target };

/** a seeded random source */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

const hash = (s: string | Uint8Array): string => createHash("sha256").update(s).digest("hex").slice(0, 16);
const bytes = (v: Float32Array): Uint8Array => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);

/** positions from seeded random games, every one of them (jump chains, kings and white to move all turn up) */
function playouts(n: number): unknown[] {
  const r = rng(12345), out: unknown[] = [];
  for (let g = 0; g < n; g++) {
    let st = api.start();
    for (let ply = 0; ply < 160; ply++) {
      if (api.status(st).over) break;
      out.push(st);
      const ms = api.moves(st);
      st = api.make(st, ms[Math.floor(r() * ms.length)]);
    }
    out.push(st);
  }
  return out;
}

const NAMED = {
  start: "bbbbbbbbbbbb........wwwwwwwwwwww b - 0",
  kings: "b.B..b.b....w...b...W..w.ww...B. w - 7",
  chain: ".b...w.......w................W. b 2 0",
  chainW: "...b.....B..w....b.....w.....b.. w 13 3",
  capture: "b........w....w..........w...... b - 0",
  endgame: "........B..............W........ w - 40",
};

function digest(): Record<string, string> {
  const P = playouts(60), d: Record<string, string> = {};
  d["count"] = String(P.length);
  d["str"] = hash(P.map((s) => api.str(s)).join("\n"));
  d["moves"] = hash(JSON.stringify(P.map((s) => api.moves(s))));
  d["status"] = hash(JSON.stringify(P.map((s) => api.status(s))));
  d["key"] = hash(P.map((s) => api.key(s)).join(","));
  d["uci"] = hash(JSON.stringify(P.map((s) => api.moves(s).map((m: unknown) => api.uci(m)))));
  d["fromUci"] = hash(JSON.stringify(P.map((s) => api.moves(s).map((m: unknown) => api.fromUci(s, api.uci(m))))));
  const h96 = createHash("sha256");
  for (const s of P) h96.update(bytes(api.render(s)));
  d["render96"] = h96.digest("hex").slice(0, 16);
  for (const size of [128, 192, 256, 300]) {
    const h = createHash("sha256");
    for (const s of P.filter((_, i) => i % 7 === 0)) h.update(bytes(api.render(s, { size })));
    d[`render${size}`] = h.digest("hex").slice(0, 16);
  }
  const hOpt = createHash("sha256");
  for (const s of P.filter((_, i) => i % 5 === 0)) {
    hOpt.update(bytes(api.render(s, { noJumpBox: true })));
    hOpt.update(bytes(api.render(s, { pieces: false })));
  }
  d["renderOpts"] = hOpt.digest("hex").slice(0, 16);
  for (const [k, s] of Object.entries(NAMED)) d[`named.${k}`] = hash(bytes(api.render(api.parse(s)))) + " " + hash(JSON.stringify(api.moves(api.parse(s))));
  const r = rng(99), sub = P.filter((_, i) => i % 4 === 0);
  const maps = sub.map(() => Float64Array.from({ length: 1024 }, () => r() ** 3));
  const reads: unknown[] = [];
  sub.forEach((s, i) => {
    const p = maps[i]!;
    reads.push(api.readMoveMap(p, s, { temp: 0 }));
    reads.push(api.readMoveMap(p, s, { temp: 1, topK: 5, rng: rng(i) }));
    reads.push(api.readMoveMap(p, s, { temp: 0.5, topK: 0, rng: rng(i + 7) }));
    reads.push(api.readMoveMap(p, s, { temp: 0, anyCell: true }));
    reads.push(api.readMoveMap(p, s, { temp: 2, topK: 3, anyCell: true, rng: rng(i + 3) }));
  });
  d["read"] = hash(JSON.stringify(reads));
  d["classify"] = hash(JSON.stringify(sub.map((s, i) => api.classify(maps[i], s))));
  d["pseudo"] = hash(JSON.stringify(sub.map((s) => [...api.pseudoCells(s)].sort((a: number, b: number) => a - b))));
  d["mask"] = hash(JSON.stringify(sub.map((s) => Array.from(api.legalMask(s)))));
  const targets: unknown[] = [];
  for (const s of sub) {
    targets.push(Array.from(api.target(s)));
    const ms = api.moves(s), obj: Record<string, number> = {}, map = new Map<number | string, number>();
    ms.forEach((m: unknown, j: number) => (obj[api.uci(m)] = j));
    api.viewMoves(s).forEach((e: { cell: number }, j: number) => map.set(e.cell, (j * 3) % 4));
    targets.push(Array.from(api.target(s, obj)), Array.from(api.target(s, map)));
  }
  d["target"] = hash(JSON.stringify(targets));
  d["legalCells"] = hash(JSON.stringify(sub.map((s) => [...api.legalCells(s).entries()])));
  d["soft"] = hash(JSON.stringify([api.softTargets({ a: 10, b: 5, c: -20 }, 10), api.softTargets({ a: 10, b: 5, c: -20 }, 10, 12), api.softTargets({ a: 1, b: 1 }, 0), api.softTargets({}, 1)]));
  d["lit"] = String(api.lit(api.start()));
  d["sqRect"] = hash(JSON.stringify(Array.from({ length: 32 }, (_, s) => api.sqRect(s))));
  d["sqAtPx"] = hash(Array.from({ length: 100 * 100 }, (_, i) => api.sqAtPx((i % 100) - 2, ((i / 100) | 0) - 2)).join(","));
  d["glyphs"] = hash(JSON.stringify([2, 1, 0, -1, -2].map((p) => api.GLYPH_OF(p))));
  api.setMoveLayout("board16");
  try {
    d["b16.cells"] = hash(JSON.stringify(P.map((s) => api.viewMoves(s))));
    d["b16.read"] = hash(JSON.stringify(sub.map((s, i) => [api.readMoveMap(maps[i], s, { temp: 1, topK: 5, rng: rng(i) }), api.readMoveMap(maps[i], s, { temp: 0, anyCell: true })])));
    d["b16.target"] = hash(JSON.stringify(sub.map((s) => Array.from(api.target(s)))));
    d["b16.classify"] = hash(JSON.stringify(sub.map((s, i) => api.classify(maps[i], s))));
    d["b16.cellMove"] = hash(JSON.stringify(Array.from({ length: 256 }, (_, c) => [api.cellMove(c), api.cellName(c)])));
  } finally {
    api.setMoveLayout("fromto32");
  }
  return d;
}

/** the digests of the original rules */
const GOLDEN: Record<string, string> = {
  "count": "5289",
  "str": "394ec596126a89ee",
  "moves": "ba9b2662bb29d0d7",
  "status": "fe74648c74a331a5",
  "key": "0750d45837350c0e",
  "uci": "96ef16ef59a41218",
  "fromUci": "ba9b2662bb29d0d7",
  "render96": "fae39ec954553794",
  "render128": "baa269eb129f6512",
  "render192": "83b818b000928862",
  "render256": "7b116a66e207d625",
  "render300": "124cf1973cbeaff8",
  "renderOpts": "f2491f8875c5213a",
  "named.start": "24907579f4a83eaa 481b1eb0c678f5b9",
  "named.kings": "abf36660204c5ec0 f6a663287e44cbd0",
  "named.chain": "d4599db761b95fb2 f7140de04fa73553",
  "named.chainW": "b1f89ed2b34bf986 4f53cda18c2baa0c",
  "named.capture": "0b1d71cb38562284 e407a4ba448fed0d",
  "named.endgame": "f86b9d32f153779a c9acf87368426aab",
  "read": "5042e79802de1c86",
  "classify": "53f3e8f508694be2",
  "pseudo": "3e5c39676f28dbbc",
  "mask": "c9f2bf53ba687604",
  "target": "2fe283f320113f0a",
  "legalCells": "af4718b18aadc7d9",
  "soft": "e10424f8f9482b13",
  "lit": "0.16666666666666666",
  "sqRect": "1698165ff90399d9",
  "sqAtPx": "e9ec942576100813",
  "glyphs": "47af50bb489d7af8",
  "b16.cells": "0185346180ac24a6",
  "b16.read": "de1517eefa3ecbde",
  "b16.target": "68d26912a45b9b68",
  "b16.classify": "fabed0bebe6a97f4",
  "b16.cellMove": "aca596ee94d51397",
};

describe("checkers rules", () => {
  it("opens with 7 moves for black", () => {
    const st = checkers.start();
    expect(checkers.str(st)).toBe(NAMED.start);
    expect(checkers.moves(st).map((m) => checkers.uci(m))).toEqual(["9-13", "9-14", "10-14", "10-15", "11-15", "11-16", "12-16"]);
  });

  it("forces a capture: only the jump is legal while quiet moves exist", () => {
    const st = checkers.parse("b........b....w..........w...... b - 0"), ms = checkers.moves(st); // black men on 1 and 10, white on 15 and 26
    expect(ms).toEqual([{ from: 9, to: 18, cap: 14, promo: false, king: false }]);
    expect(checkers.fromUci(st, "1-5")).toBeNull();
  });

  it("keeps the turn through a jump chain, then passes it", () => {
    const st = checkers.parse(".b...w.......w................W. b - 0");
    const a = checkers.make(st, checkers.fromUci(st, "2x9")!);
    expect(a.turn).toBe(1);
    expect(a.jump).toBe(8);
    expect(checkers.moves(a).map((m) => checkers.uci(m))).toEqual(["9x18"]);
    const b = checkers.make(a, checkers.moves(a)[0]!);
    expect(b.turn).toBe(-1);
    expect(b.jump).toBe(-1);
    expect(checkers.str(b)).toBe(".................b............W. w - 0");
  });

  it("covers chains, kings and white to move in the random games", () => {
    const P = playouts(60) as ReturnType<typeof checkers.start>[];
    expect(P.some((s) => s.jump >= 0 && s.turn < 0)).toBe(true);
    expect(P.some((s) => s.jump >= 0 && s.turn > 0)).toBe(true);
    expect(P.some((s) => s.sq.some((v) => Math.abs(v) === 2))).toBe(true);
  });

  it("reproduces the original rules, frames and move reader byte for byte", () => {
    expect(digest()).toEqual(GOLDEN);
    expect(checkers.SIZE).toBe(96); // a sized render puts the frame back
    expect(checkers.MOVE_CELLS).toBe(1024);
  });
});
