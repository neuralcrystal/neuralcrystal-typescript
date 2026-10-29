// Ask the chess crystal for a move, optionally let it think two moves ahead, and draw what the light did.
//   npx vite-node examples/chess.ts [crystal] ["<FEN>"] [out.png] [--think]
// The crystal plays black; a position with white to move is shown to it color-swapped and mirrored, and the move comes back on the real
// board, so any FEN works.
import { chess as ch, chessKit, chessThink, chessValueOf, solve, huggingFaceUrl } from "../src/index.js";
import { engine, frameTile, lightTile, load, outline, savePng } from "./lib.js";

const args = process.argv.slice(2).filter((a) => !a.startsWith("--")), think = process.argv.includes("--think");
const src = args[0] || huggingFaceUrl("chess"), fen = args[1] || "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", out = args[2] || "examples/out/chess.png";
const crystal = await load(src), eng = await engine(crystal), kit = chessKit(crystal);

const pos = ch.parseFEN(fen);
const { answer: a, run } = await solve(kit, eng, pos); // the brightest legal move
const wdl = chessValueOf(a.powers, kit.valueCells!); // win, draw, loss for the side to move
console.log(`${pos.turn === "w" ? "white" : "black"} to move: the crystal plays ${a.san} (${a.uci})`);
console.log(`  win / draw / loss for the side to move: ${[wdl.w, wdl.d, wdl.l].map((x) => `${(100 * x).toFixed(0)} %`).join(" / ")}`);
const top = [...kit.cells(pos)].map(([cell, m]) => ({ m, p: a.movePowers[cell]! })).sort((x, y) => y.p - x.p).slice(0, 5);
for (const t of top) console.log(`  ${ch.san(pos, t.m).padEnd(7)} ${"#".repeat(Math.round((40 * t.p) / top[0]!.p))}`);

let move = a.move!;
if (think) {
  // think=(3, 5): its 3 brightest moves against the opponent's 5 brightest replies to each, judged by the win/draw/loss reading
  const th = await chessThink(pos, a.movePowers, {
    mode: "always", share: 1, widths: [3, 5], side: kit.art.playAs, keys: [], valueCells: kit.valueCells!, moveP: (p) => kit.moveP(p),
    look: async (s) => kit.powers(eng, (await eng.run(kit.frame(s), kit.frameSize)).exit),
  });
  if (th?.best !== undefined) move = th.cand[th.best]!.move;
  console.log(`  thought about ${th?.cand.map((c) => ch.uci(c.move)).join(", ")}, expected scores ${th?.es?.map((e) => e?.toFixed(2)).join(", ")}`);
  console.log(`  after looking ahead: ${ch.san(pos, move)}`);
}

// the DMD frame, the camera's picture after passes 1–3, and the move map: legal moves in teal, the crystal's move in white
const map = lightTile(a.movePowers, 64, 4);
for (const cell of kit.cells(pos).keys()) outline(map, (cell % 64) * 4, Math.floor(cell / 64) * 4, 4, 4, [60, 170, 160]);
outline(map, (a.cell % 64) * 4 - 1, Math.floor(a.cell / 64) * 4 - 1, 6, 6, [255, 255, 255]);
savePng(out, [[frameTile(kit.frame(pos), kit.frameSize, 2), ...run.hiddens.map((h) => frameTile(h, kit.frameSize, 2)), map]]);
console.log(`wrote ${out}`);
eng.dispose();
