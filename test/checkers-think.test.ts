// The crystal's rules round its reads: the value cells, the lookahead's choice, the repetition rule.
import { describe, expect, it } from "vitest";
import { checkers as C, repNext, repRule, think, valueCells, valueOf } from "../src/index.js";

const model = { tr: { value: { win: { block: [0, 28, 3, 31] }, draw: { block: [6, 28, 9, 31] }, loss: { block: [0, 22, 3, 25] } } } };

describe("checkers thinking", () => {
  it("reads the value cells off the file's blocks (16 each) and scores win + draw / 2", () => {
    const g = valueCells(model)!;
    expect(g.map((x) => x.length)).toEqual([16, 16, 16]);
    expect(g[0]).toContain(28);
    const p = new Float64Array(1024);
    for (const c of g[0]!) p[c] = 1; // all win light
    for (const c of g[1]!) p[c] = 1; // as much draw
    expect(valueOf(p, g)).toBeCloseTo(0.75);
  });

  it("looks ahead when unsure and plays the move whose worst case is best", async () => {
    const st = C.start(), cells = [...new Set(C.viewMoves(st).map((e) => e.cell))];
    const p = new Float64Array(1024);
    cells.slice(0, 3).forEach((c, i) => (p[c] = 3 - i * 0.2)); // three near-equal moves: unsure
    const rd = C.readMoveMap(p, st, { temp: 0 })!;
    const g = valueCells(model)!;
    // every board after the SECOND candidate is judged a win for us, the others a loss: the search must switch to it
    const second = C.make(st, C.viewMoves(st).find((e) => e.cell === cells[1])!.m);
    const look = async (b: typeof st): Promise<Float64Array> => {
      const q = new Float64Array(1024);
      for (const e of C.viewMoves(b)) q[e.cell] = 1;
      const good = C.str(b).split(" ")[0] === C.str(second).split(" ")[0];
      for (const c of good ? g[2]! : g[0]!) q[c] = 1; // on their move: our win is their loss
      return q;
    };
    const t = await think(st, p, rd, { mode: "always", share: 0.5, widths: [3], valueCells: g, look });
    expect(t.rd.cell).toBe(cells[1]);
    expect(t.note?.looked).toBe(3);
    const off = await think(st, p, rd, { mode: "off", share: 0.5, widths: [3], valueCells: g, look });
    expect(off.rd).toBe(rd);
  });

  it("will not repeat a position when not behind, if another of the top five has a quarter of the light", () => {
    // a king each: black's steps out, white's steps out, black's steps back — white stepping back now would recreate the start
    const st = C.parse("B.............................W. b - 0");
    let rep = [C.key(st)];
    const go = (s0: typeof st, m: ReturnType<typeof C.moves>[number]): typeof st => {
      const n = C.make(s0, m);
      rep = repNext(rep, s0, m, n);
      return n;
    };
    const m1 = C.moves(st)[0]!, a = go(st, m1);
    const m2 = C.moves(a)[0]!, b = go(a, m2);
    const back1 = C.moves(b).find((m) => m.from === m1.to && m.to === m1.from)!, c = go(b, back1);
    const ent = C.viewMoves(c), again = ent.find((e) => e.m.from === m2.to && e.m.to === m2.from)!;
    expect(rep).toContain(C.key(C.make(c, again.m))); // white's step back repeats the start
    const other = ent.find((e) => e.cell !== again.cell)!, p = new Float64Array(1024);
    p[again.cell] = 1;
    p[other.cell] = 0.3;
    const rd = C.readMoveMap(p, c, { temp: 0 })!;
    expect(rd.cell).toBe(again.cell);
    const r = repRule(rd, p, c, rep);
    expect(r.rd.cell).toBe(other.cell);
    expect(r.text).toMatch(/no repetition/);
    p[other.cell] = 0.1; // under a quarter of the light: the repeat stands
    expect(repRule(rd, p, c, rep).rd.cell).toBe(again.cell);
  });
});
