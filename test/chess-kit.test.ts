import { describe, expect, it } from "vitest";
import { chess, chessKit, chessMirror, chessMirrorMove, chessRules, chessThink, chessValueCells, CHESS_RULES, ghostPieces } from "../src/index.js";
import type { CrystalData } from "../src/index.js";

// the art96z crystal's own blocks (its header), without its 335 MB of maps
const model = { N: 1024, dx_um: 1, winUm: 896, ext: 85.7142857, chess: { art96x2: true, lampClock: true, lampCrystal: true, flatBoard: true, playAs: "b", frame: 192 }, tr: { det: "move", value: { win: [0, 15], draw: [24, 39], loss: [48, 63] } }, planes: [] };
const kit = chessKit({ model } as unknown as CrystalData);
const fakeEngine = { geometry: { N: 1024, winFrac: 0.875 } } as never;

describe("the chess kit", () => {
  it("puts the 64 × 64 cells where the crystals were trained with them: x = 67 + 14·to, 7 × 7", () => {
    const b = kit.bands(fakeEngine);
    expect(b.side).toBe(7);
    expect(b.lo.slice(0, 3)).toEqual([67, 81, 95]);
    expect(b.lo[63]).toBe(67 + 14 * 63);
  });

  it("reads the value cells off the diagonal (65·s) and keeps them out of the move read", () => {
    const g = chessValueCells(model)!;
    expect(g.map((x) => x.length)).toEqual([16, 16, 16]);
    expect(g[0]![1]).toBe(65);
    const p = new Float64Array(4096).fill(1), q = kit.moveP(p);
    expect(q[65]).toBe(0);
    expect(q[64]).toBe(1);
  });

  it("draws the frame as the compact art doubled, the crystal's mark on the side to move", () => {
    const st = chess.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    const f = kit.frame(st);
    expect(f.length).toBe(192 * 192);
    for (let y = 0; y < 192; y += 2) for (let x = 0; x < 192; x += 2) expect(f[y * 192 + x + 1]).toBe(f[y * 192 + x]);
    // the quartz mark, top-left (black to move), [1, 1, 6, 6] at 96 → rows 2–13 at 192
    let lit = 0;
    for (let y = 2; y < 14; y++) for (let x = 2; x < 14; x++) lit += f[y * 192 + x]!;
    expect(lit).toBeGreaterThan(0);
    expect(chess.FLATBG).toBe(false); // put back for the screen
  });

  it("plays the brightest legal cell at temperature 0, and a rule takes a mate in one", () => {
    const st = chess.parseFEN("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"), p = new Float64Array(4096);
    const mate = chess.moves(st).find((m) => chess.uci(m) === "a1a8")!, quiet = chess.moves(st).find((m) => chess.uci(m) === "g1f1")!;
    p[chess.MOVE_CELL(quiet)] = 3;
    p[chess.MOVE_CELL(mate)] = 1;
    const rd = chess.readMoveMap(p, st)!;
    expect(rd.uci).toBe("g1f1");
    const ruled = chessRules(rd, p, st, [chess.key(st)], CHESS_RULES);
    expect(ruled.uci).toBe("a1a8");
    expect(ruled.ruled).toContain("mate");
  });

  it("answers for either color: white's positions go in color-swapped and mirrored, and the move comes back on the real board", () => {
    // any light on the exit face: the kit must read a white position exactly as it reads that position's mirror, flipped back
    let seed = 7;
    const rnd = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const N = 1024, exit = { N, re: Float64Array.from({ length: N * N }, rnd), im: Float64Array.from({ length: N * N }, rnd) };
    const run = { exit, entry: exit, hiddens: [] };
    for (const fen of ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "r3k2r/pp1b1ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1", "8/P5k1/8/8/8/8/6K1/8 w - - 0 1"]) {
      const white = chess.parseFEN(fen), seen = chessMirror(white);
      expect(Array.from(kit.frame(white))).toEqual(Array.from(kit.frame(seen)));
      const a = kit.read(fakeEngine, run, white), m = kit.read(fakeEngine, run, seen);
      expect(a.mirrored).toBe(true);
      expect(m.mirrored).toBe(false);
      expect(a.uci).toBe(chess.uci(chessMirrorMove(m.move!)));
      expect(a.san).toBe(chess.san(white, a.move!));
      expect(chess.moves(white).some((x) => chess.uci(x) === a.uci)).toBe(true);
      expect(chess.fen(a.state!)).toBe(chess.fen(chess.make(white, a.move!)));
      expect(a.cell).toBe(m.cell);
      const cells = kit.cells(white);
      expect(cells.get(a.cell)).toEqual(a.move);
      expect(cells.size).toBe(chess.moveCells(seen).size); // the four promotions share one cell
    }
  });

  it("looks ahead from a white position as from its mirror, the candidates flipped back", async () => {
    let seed = 11;
    const rnd = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const pw = Float64Array.from({ length: 4096 }, rnd), raw = Float64Array.from({ length: 4096 }, rnd);
    const o = { mode: "always" as const, share: 1, widths: [3, 2], side: "b" as const, keys: [], valueCells: kit.valueCells!, moveP: (p: Float64Array) => kit.moveP(p), look: async () => raw };
    const white = chess.parseFEN("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"), seen = chessMirror(white);
    const a = (await chessThink(white, pw, o))!, m = (await chessThink(seen, pw, o))!;
    expect(a.cand.map((c) => chess.uci(c.move))).toEqual(m.cand.map((c) => chess.uci(chessMirrorMove(c.move))));
    expect(a.cand.map((c) => c.cell)).toEqual(m.cand.map((c) => c.cell));
    for (const c of a.cand) expect(chess.moves(white).some((x) => chess.uci(x) === chess.uci(c.move))).toBe(true);
  });

  it("mirrors a position back onto itself", () => {
    const st = chess.parseFEN("r3k2r/pp1b1ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1");
    expect(chess.fen(chessMirror(chessMirror(st)))).toBe(chess.fen(st));
  });

  it("thinks 2-ply 3 × 5 within 18 reads, reading every board with its trained colour to move", async () => {
    const st = chess.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    const pw = new Float64Array(4096);
    for (const [c] of chess.moveCells(st)) pw[c] = 1; // all even: unsure
    const seen: string[] = [];
    const th = await chessThink(st, pw, {
      mode: "unsure", share: 0.5, widths: [3, 5], side: "b", keys: [chess.key(st)], valueCells: chessValueCells(model)!, moveP: (p) => kit.moveP(p),
      look: async (b) => {
        seen.push(b.turn);
        const p = new Float64Array(4096);
        for (const [c] of chess.moveCells(b)) p[c] = 1 + (c % 7);
        p[0] = 1; // a little of everything on the value cells
        p[65 * 30] = 2;
        return p;
      },
    });
    expect(th!.thought).toBe(true);
    expect(th!.looked).toBe(18);
    expect(new Set(seen)).toEqual(new Set(["b"]));
  });

  it("narrows the sure lines: a sure inner board follows its one brightest move, at the same depth", async () => {
    const st = chess.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    const pw = new Float64Array(4096);
    for (const [c] of chess.moveCells(st)) pw[c] = 1;
    const run = (narrow: { ours: number; theirs: number } | null) =>
      chessThink(st, pw, {
        mode: "always", share: 0.5, widths: [3, 3, 3, 3], side: "b", keys: [chess.key(st)], valueCells: chessValueCells(model)!, moveP: (p) => kit.moveP(p), narrow,
        look: async (b) => {
          const p = new Float64Array(4096), cells = [...chess.moveCells(b).keys()];
          cells.forEach((c, i) => (p[c] = i === 0 ? 100 : 1)); // one move far brighter than the rest: every board is sure
          p[0] = 1;
          return p;
        },
      });
    const full = await run(null), sure = await run({ ours: 0.6, theirs: 0.8 });
    expect(full!.looked).toBe(120);
    expect(sure!.looked).toBe(3 * 4); // each candidate: one line of four boards
    expect(sure!.narrowed).toBe(9);
    expect(sure!.most).toBe(120);
  });

  it("ghosts the line being read: the moved pieces where they end up, and tells each line before its read", async () => {
    const st = chess.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    const e5 = chess.fromUci(st, "e7e5")!, n = chess.make(st, e5), nf3 = chess.fromUci(n, "g1f3")!;
    expect(ghostPieces(st, [e5, nf3]).map((g) => `${chess.sqName(g.sq)}${g.piece}`).sort()).toEqual(["e5-1", "f32"]);
    const pw = new Float64Array(4096);
    for (const [c] of chess.moveCells(st)) pw[c] = 1;
    const lines: number[] = [];
    await chessThink(st, pw, {
      mode: "always", share: 0.5, widths: [3, 5], side: "b", keys: [chess.key(st)], valueCells: chessValueCells(model)!, moveP: (p) => kit.moveP(p),
      onLook: (line) => lines.push(line.length),
      look: async (b) => {
        const p = new Float64Array(4096);
        for (const [c] of chess.moveCells(b)) p[c] = 1;
        p[0] = 1;
        return p;
      },
    });
    expect(lines.length).toBe(18);
    expect(lines.filter((l) => l === 1).length).toBe(3); // each candidate, then its five replies
    expect(lines.filter((l) => l === 2).length).toBe(15);
  });

  it("thinks 4-ply 3 × 3 × 3 × 3 within 120 reads, 5 × 3 × 3 × 1 within 110, and 3 × 7 within 24", async () => {
    const st = chess.parseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    const pw = new Float64Array(4096);
    for (const [c] of chess.moveCells(st)) pw[c] = 1;
    for (const [w, n] of [[[3, 3, 3, 3], 120], [[3, 7], 24], [[5, 3, 3, 1], 110], [[5, 5], 30], [[7, 3], 28], [[7, 5], 42], [[5, 3, 3, 2], 155]] as const) {
      const seen = new Set<string>();
      const th = await chessThink(st, pw, {
        mode: "always", share: 0.5, widths: [...w], side: "b", keys: [chess.key(st)], valueCells: chessValueCells(model)!, moveP: (p) => kit.moveP(p),
        look: async (b) => {
          seen.add(b.turn);
          const p = new Float64Array(4096);
          for (const [c] of chess.moveCells(b)) p[c] = 1 + (c % 5);
          p[0] = 1;
          return p;
        },
      });
      expect(th!.looked).toBe(n);
      expect(seen).toEqual(new Set(["b"]));
    }
  });
});
