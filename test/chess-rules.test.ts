// The chess rules against the standard move counts (perft), and the frame, decoder and move readers against outputs captured from the
// original implementation they were ported from: a hash of every rendered frame (both layouts, board and flat) and of every reading.
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import type { ChessMove, ChessState } from "../src/problems/chess/rules/position.js";
import { parseFEN, rng } from "../src/problems/chess/rules/position.js";
import { fen, fromUci, key, make, moves, san, uci } from "../src/problems/chess/rules/moves.js";
import { status } from "../src/problems/chess/rules/status.js";
import { layout, live } from "../src/problems/chess/rules/layout.js";
import { readValue, render } from "../src/problems/chess/rules/frame.js";
import { decode, readMove } from "../src/problems/chess/rules/decode.js";
import { moveCells, readMoveMap, readSquares } from "../src/problems/chess/rules/movemap.js";
import { chess } from "../src/problems/chess/board.js";

const R = {
  layout, setFlat: (b: boolean): void => void (live.FLATBG = b), SIZE: (): number => live.SIZE,
  parseFEN, rng, fen, fromUci, key, make, moves, san, uci, status, render, readValue, decode, readMove, readSquares, readMoveMap, moveCells,
};

const sha = (a: ArrayBufferView): string => createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex").slice(0, 16);
const FENS = [
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
  "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
];

/** every case's output, hashed */
function cases(): Record<string, string> {
  const out: Record<string, string> = {};
  const summary = (o: unknown): string =>
    createHash("sha256")
      .update(JSON.stringify(o, (k, v) => (k === "dec" ? undefined : k === "state" && v ? R.fen(v) : ArrayBuffer.isView(v) ? sha(v) : v === -Infinity ? "-inf" : v)))
      .digest("hex")
      .slice(0, 16);
  for (const size of [192, 96]) {
    R.layout(size);
    for (const flat of [false, true]) {
      R.setFlat(flat);
      FENS.forEach((f, i) => {
        const st = R.parseFEN(f), k = `${size}/${flat ? "flat" : "board"}/${i}`;
        out[`${k}/plain`] = sha(R.render(st));
        out[`${k}/marks`] = sha(R.render(st, { seed: 7, ring: [12, 28, 63], result: "WHITE WINS", valueCp: 120, crystalSides: ["w", "b"], noClock: true }));
        out[`${k}/code`] = sha(R.render(st, { code: { n: 2, K: 5 }, noBottom: true }));
        out[`${k}/bars`] = sha(R.render(st, { noise: 0.3, noiseMode: "1d", seed: 3, valueCp: -1000, crystalSides: ["b"] }));
        out[`${k}/text`] = sha(R.render(st, { noise: 0, result: "0.001082 SEC", code: { n: 9, K: 3 } }));
      });
    }
    R.setFlat(false);
    // the decoder on synthetic readouts
    const st = R.parseFEN(FENS[0]!), e4 = R.fromUci(st, "e2e4")!, target = R.render(R.make(st, e4), { noise: 0 });
    const blur = new Float32Array(target.length), n = R.SIZE();
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        let a = 0, c = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const Y = y + dy, X = x + dx; if (Y >= 0 && Y < n && X >= 0 && X < n) { a += target[Y * n + X]!; c++; } }
        blur[y * n + x] = a / c;
      }
    const rng = R.rng(11), haze = new Float32Array(n * n).map(() => rng());
    out[`${size}/decode/target`] = sha(R.decode(target).score);
    out[`${size}/decode/blur`] = sha(R.decode(blur).score);
    out[`${size}/readMove/target`] = summary(R.readMove(target, st));
    out[`${size}/readMove/blur`] = summary(R.readMove(blur, st));
    out[`${size}/readMove/same`] = summary(R.readMove(R.render(st, { noise: 0 }), st));
    out[`${size}/readMove/haze`] = summary(R.readMove(haze, st));
    out[`${size}/readMove/strict`] = summary(R.readMove(blur, st, { maxOff: 0, light: 0.9, dark: 0.1 }));
    out[`${size}/readValue`] = summary(R.readValue(R.render(st, { valueCp: 120 })));
  }
  R.layout(96);
  FENS.forEach((f, i) => {
    const st = R.parseFEN(f), moves = R.moves(st);
    out[`moves/${i}`] = summary(moves.map((m: ChessMove) => [m, R.san(st, m), R.uci(m), R.fen(R.make(st, m))]));
    out[`status/${i}`] = summary([R.status(st), R.status(st, [R.key(st), R.key(st)]), R.status(st, [R.key(st), R.key(st), R.key(st)])]);
    out[`fen/${i}`] = R.fen(st);
    const r1 = R.rng(100 + i), p64 = Float64Array.from({ length: 64 }, () => r1());
    out[`readSquares/${i}`] = summary(R.readSquares(p64, st));
    const p4k = Float64Array.from({ length: 4096 }, () => r1());
    out[`readMoveMap/${i}/plain`] = summary(R.readMoveMap(p4k, st));
    out[`readMoveMap/${i}/temp`] = summary(R.readMoveMap(p4k, st, { temp: 1, topK: 5, rng: R.rng(5) }));
    out[`readMoveMap/${i}/any`] = summary(R.readMoveMap(p4k, st, { anyCell: true }));
    out[`readMoveMap/${i}/anyTemp`] = summary(R.readMoveMap(p4k, st, { anyCell: true, temp: 0.5, topK: 3, rng: R.rng(6) }));
    out[`moveCells/${i}`] = summary([...R.moveCells(st).entries()]);
  });
  const ENDS = ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/8/6Q1/8/8/8/8/K7 b - - 0 1", "8/8/4k3/8/2B5/8/4K3/8 w - - 0 1", "8/2b5/4k3/8/2B5/8/4K3/8 w - - 0 1", "8/2b5/4k3/8/3B4/8/4K3/8 w - - 0 1", "8/8/4k3/8/8/8/4KR2/8 w - - 100 80"];
  ENDS.forEach((f, i) => (out[`end/${i}`] = summary(R.status(R.parseFEN(f)))));
  return out;
}

const GOLDEN: Record<string, string> = {
  "192/board/0/plain": "10fdf57c6927c359", "192/board/0/marks": "e675ccf55815f5ce", "192/board/0/code": "e6a7043af4101198", "192/board/0/bars": "c985f99e69986127",
  "192/board/0/text": "a87d96c6467d437a", "192/board/1/plain": "4a10b8f947900960", "192/board/1/marks": "28270651b268f3a3", "192/board/1/code": "b0756df136f47b2f",
  "192/board/1/bars": "64c2c10d584e8f23", "192/board/1/text": "16f9d91d9b022d87", "192/board/2/plain": "fa7e94b7e6893263", "192/board/2/marks": "619f7d2383fdcdb9",
  "192/board/2/code": "5f001c2fa2709c31", "192/board/2/bars": "2ab9b2b370da0ca3", "192/board/2/text": "f367c5d56e166e15", "192/board/3/plain": "cc300a55cb8b3262",
  "192/board/3/marks": "28e8a87cb35e3f2e", "192/board/3/code": "c38ca840c5d8f62e", "192/board/3/bars": "c704ca4365e1e003", "192/board/3/text": "471ed1ef9d7bc0d8",
  "192/board/4/plain": "d5cb0590ceae0e47", "192/board/4/marks": "3f668ad864bb0d7f", "192/board/4/code": "6026efe694f74383", "192/board/4/bars": "cde4ab3583fc23ee",
  "192/board/4/text": "29bf854da7502446", "192/flat/0/plain": "52bd6a76cfee6c3a", "192/flat/0/marks": "b9752cc7eae78d72", "192/flat/0/code": "9bab2d4f117560ff",
  "192/flat/0/bars": "bb2d91444494b9b0", "192/flat/0/text": "59d90313b92818ef", "192/flat/1/plain": "583bdb4b9c5724ab", "192/flat/1/marks": "4842f3cef01e9f3b",
  "192/flat/1/code": "2f4ec3186b3b40d9", "192/flat/1/bars": "1980167e6e34cc00", "192/flat/1/text": "8b5bd077c8582c79", "192/flat/2/plain": "cc2907d3a1870e7d",
  "192/flat/2/marks": "a06e746d9681a967", "192/flat/2/code": "bc65a73fc4d411c7", "192/flat/2/bars": "1c12f5968e2366ec", "192/flat/2/text": "9ece2c8163cf6585",
  "192/flat/3/plain": "bae94f3954f6f716", "192/flat/3/marks": "b2d5db1c2c22b04b", "192/flat/3/code": "6dcc3b5a5bb7f104", "192/flat/3/bars": "f2a021c7f3f857dc",
  "192/flat/3/text": "8c0cabfae11c0079", "192/flat/4/plain": "1ed3754df2ba45f5", "192/flat/4/marks": "51441f889387acdb", "192/flat/4/code": "c4abd4e4eaa86480",
  "192/flat/4/bars": "4382fab329187458", "192/flat/4/text": "6de7a6356c45d870", "192/decode/target": "90254d7c31f39f1b", "192/decode/blur": "7d8f048b01b9daea",
  "192/readMove/target": "226bb37db593d800", "192/readMove/blur": "cbfcab5d2809e4a0", "192/readMove/same": "c97b48a20c4f0746", "192/readMove/haze": "e627cc1960cf234d",
  "192/readMove/strict": "14753ad66cc05cca", "192/readValue": "fe574b378eb5bf75", "96/board/0/plain": "12ff605ea0ea2502", "96/board/0/marks": "cd64c8b28aaff915",
  "96/board/0/code": "b93d02fc6b7fd5bb", "96/board/0/bars": "5a41231f9cdcad51", "96/board/0/text": "14b80c2910b78dc1", "96/board/1/plain": "d6de51605be248b8",
  "96/board/1/marks": "202bb419544d3869", "96/board/1/code": "19b412fb2bd3d2f6", "96/board/1/bars": "665278bbd77602e7", "96/board/1/text": "a024b32b2e7e747e",
  "96/board/2/plain": "6d7be97e9fbc7a2d", "96/board/2/marks": "31cbf792a22c201e", "96/board/2/code": "47f756ee1bcfc07b", "96/board/2/bars": "f6628f39010cb4cd",
  "96/board/2/text": "0b456d6e611c2a44", "96/board/3/plain": "b0281ad7291c9db5", "96/board/3/marks": "4b0bb1e750788b9f", "96/board/3/code": "3082ea3a3f3d0374",
  "96/board/3/bars": "b194f68bc084ff14", "96/board/3/text": "1481e96f0a4781a1", "96/board/4/plain": "91d4c7ada0f6f764", "96/board/4/marks": "eb4370120906efd2",
  "96/board/4/code": "bb068ac311f6ceab", "96/board/4/bars": "dc1f517349997224", "96/board/4/text": "da6d1eea719db1f7", "96/flat/0/plain": "786c5725cbbcfe16",
  "96/flat/0/marks": "be72cc17380921ed", "96/flat/0/code": "3282258e6854d15a", "96/flat/0/bars": "e894a27ac56ee47b", "96/flat/0/text": "1521a0ac9ff3d1cb",
  "96/flat/1/plain": "3a44fe286c466ee1", "96/flat/1/marks": "af5786f508b72c3b", "96/flat/1/code": "7ef41df08e6adca1", "96/flat/1/bars": "b706d098f96862dd",
  "96/flat/1/text": "9d51486fbc01bb16", "96/flat/2/plain": "5d1dd22468137a24", "96/flat/2/marks": "7600b2e2c524fd24", "96/flat/2/code": "be73fef600d82b3c",
  "96/flat/2/bars": "e9d9158e0aa663d8", "96/flat/2/text": "fa9d585fcaa9c01f", "96/flat/3/plain": "86c88f5e8d0ca9a8", "96/flat/3/marks": "52f0cd919dd9f759",
  "96/flat/3/code": "64d32d38f79d8944", "96/flat/3/bars": "4a9e77bb566d7832", "96/flat/3/text": "a4df2e7effea9c85", "96/flat/4/plain": "2b06709d9939f692",
  "96/flat/4/marks": "ebdff8ae94370dc0", "96/flat/4/code": "7c4ea5d266118c10", "96/flat/4/bars": "e6e8b3565d357f11", "96/flat/4/text": "b8fb7f4175bac56c",
  "96/decode/target": "b327478751aabccc", "96/decode/blur": "503c32cb545df0b7", "96/readMove/target": "78c5984ce54b41fb", "96/readMove/blur": "514a454d390ff132",
  "96/readMove/same": "437dfb334971e989", "96/readMove/haze": "7d04530e6a2f3dce", "96/readMove/strict": "1914357fac1061dc", "96/readValue": "dccc48d1ec8029b0",
  "moves/0": "af483f4882b344f0", "status/0": "921c99bc788eafca", "fen/0": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "readSquares/0": "1463df717f9f2314",
  "readMoveMap/0/plain": "0f6b776fd17b32d4", "readMoveMap/0/temp": "531b50fcf21ec4e1", "readMoveMap/0/any": "bc76bf5bf3e75eda", "readMoveMap/0/anyTemp": "031c8634e6f3eb80",
  "moveCells/0": "2a3f664f9b20945d", "moves/1": "2975875ab7a1e595", "status/1": "c002fb9760e93ea8", "fen/1": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "readSquares/1": "e7f44afe09e327df", "readMoveMap/1/plain": "ffccda77396d1019", "readMoveMap/1/temp": "c499ff5bcd516642", "readMoveMap/1/any": "4c507b3da90c48d5",
  "readMoveMap/1/anyTemp": "bbc6f1701efb73ce", "moveCells/1": "e00e42689c89d721", "moves/2": "58fbb9767d00a582", "status/2": "28bfc77b9139b82e",
  "fen/2": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "readSquares/2": "321cbc0d2f573b87", "readMoveMap/2/plain": "36e01dcbfd4090aa", "readMoveMap/2/temp": "973f18515ecced07",
  "readMoveMap/2/any": "ce468a6b68a33ca1", "readMoveMap/2/anyTemp": "a7fddc204a303d8e", "moveCells/2": "7fc39dcff14c1ba6", "moves/3": "94adddf05b1fc99d",
  "status/3": "7499667df2ca1866", "fen/3": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "readSquares/3": "7799877331cc7531", "readMoveMap/3/plain": "256625327d015028",
  "readMoveMap/3/temp": "819fc95540366d29", "readMoveMap/3/any": "cfb08c68cca8c9fe", "readMoveMap/3/anyTemp": "6a29baf80b61a4ff", "moveCells/3": "6ac823eae2e04d8e",
  "moves/4": "db2956cac0e4d95e", "status/4": "41c0380cee9612f2", "fen/4": "8/P6k/8/8/8/8/6Kp/8 w - - 0 1", "readSquares/4": "b4c359dddf3868e0",
  "readMoveMap/4/plain": "c5c8112b4394a360", "readMoveMap/4/temp": "b64bf621460593f6", "readMoveMap/4/any": "303093e4671fe2e0", "readMoveMap/4/anyTemp": "c781b41d920ec67d",
  "moveCells/4": "f650b9c2dd431031", "end/0": "4edf6ba577905af4", "end/1": "559e63c328e8ff74", "end/2": "489fce9ed245e97a",
  "end/3": "a77733f8e32ab07d", "end/4": "489fce9ed245e97a", "end/5": "9bb8cf63bfc1a493",
};

function perft(st: ChessState, depth: number): number {
  if (depth === 0) return 1;
  let n = 0;
  for (const m of moves(st)) n += depth === 1 ? 1 : perft(make(st, m), depth - 1);
  return n;
}

describe("the chess rules", () => {
  it("counts the standard perft numbers", () => {
    const at = (f: string, d: number): number => perft(parseFEN(f), d);
    expect(at(FENS[0]!, 1)).toBe(20);
    expect(at(FENS[0]!, 2)).toBe(400);
    expect(at(FENS[0]!, 3)).toBe(8902);
    expect(at(FENS[3]!, 1)).toBe(48); // Kiwipete
    expect(at(FENS[3]!, 2)).toBe(2039);
    expect(at("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3)).toBe(2812); // position 3: en passant and pins
    expect(at("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2)).toBe(264); // position 4: promotions
  });

  it("reads a clean target as its move, and the board handed back as no move", () => {
    for (const size of [192, 96]) {
      layout(size);
      const st = parseFEN(FENS[0]!), e4 = fromUci(st, "e2e4")!;
      expect(readMove(render(make(st, e4), { noise: 0 }), st)?.uci).toBe("e2e4");
      expect(readMove(render(st, { noise: 0 }), st)?.why).toBe("unchanged");
    }
    layout(96);
  });

  it("matches the original implementation's frames and readings", () => {
    const got = cases();
    expect(Object.keys(got).length).toBe(Object.keys(GOLDEN).length);
    for (const k in GOLDEN) expect([k, got[k]]).toEqual([k, GOLDEN[k]]);
    layout(96); // the engine's layout, as board.ts leaves it
    expect(chess.SIZE).toBe(96);
  });
});
