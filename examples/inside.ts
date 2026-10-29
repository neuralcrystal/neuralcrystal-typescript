// Look inside a crystal: the light arriving at every surface of every pass, the camera's pictures between passes, and the answer.
//   npx vite-node examples/inside.ts cifar10 [crystal] [photo.png]
//   npx vite-node examples/inside.ts checkers [crystal] ["<position>"]
//   npx vite-node examples/inside.ts chess [crystal] ["<FEN>"]
//   npx vite-node examples/inside.ts mnist [crystal] [digit.png]
// Writes examples/out/inside-<problem>.png, one row per pass: what the DMD launched, the light arriving at each of the section's surfaces,
// and the light at its exit face (each tile scaled to its own brightest point, square-root brightness so faint light shows).
import { checkers as ck, checkersKit, chess as ch, chessKit, cifarKit, mnistKit, huggingFaceUrl } from "../src/index.js";
import { engine, exitTile, frameTile, load, readPng, savePng } from "./lib.js";
import type { Tile } from "./lib.js";
import type { PublishedCrystal } from "../src/index.js";

const problem = process.argv[2] ?? "cifar10", what = process.argv[4];
const crystal = await load(process.argv[3] ?? huggingFaceUrl(problem as PublishedCrystal)), eng = await engine(crystal);
let frame: Float32Array, F: number;
if (problem === "checkers") { const kit = checkersKit(crystal); frame = kit.frame(ck.parse(what ?? "bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0")); F = kit.frameSize; }
else if (problem === "chess") { const kit = chessKit(crystal); frame = kit.frame(ch.parseFEN(what ?? "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")); F = kit.frameSize; }
else if (problem === "mnist") { const kit = mnistKit(crystal); frame = kit.frame(readPng(what ?? "examples/assets/digit-4.png").px); F = kit.frameSize; }
else { const kit = cifarKit(crystal); frame = kit.frame(readPng(what ?? "examples/assets/frog.png").px); F = kit.frameSize; }

// the light at every surface: one pass for a single-section crystal, every stage (and the pictures between) for a multi-stage one
const T = 128, rows: Tile[][] = [];
const passes = eng.quad.hops > 1 ? await eng.traceStages(frame, F) : { passes: [await eng.trace(frame, F)], hiddens: [] };
passes.passes.forEach((tr, k) => {
  const launched = k === 0 ? frame : passes.hiddens[k - 1]!;
  rows.push([frameTile(launched, F, Math.max(1, Math.round(T / F))), ...tr.arriving.map((E) => exitTile(E, T))]);
  console.log(`pass ${k + 1}: launched a ${F} × ${F} picture, light traced at ${tr.arriving.length - 1} surfaces and the exit face`);
});
const out = `examples/out/inside-${problem}.png`;
savePng(out, rows, 4);
console.log(`wrote ${out}`);
eng.dispose();
