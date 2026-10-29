// A position in, a move out: four passes through the checkers crystal onto a 32 × 32 move map (row = from-square, column = to-square);
// the brightest legal cell is the move.
//   npx vite-node examples/checkers.ts [crystal] ["<position>"] [out.png]
import { checkers as ck, checkersKit, huggingFaceUrl } from "../src/index.js";
import { engine, frameTile, lightTile, load, outline, savePng } from "./lib.js";

const [src = huggingFaceUrl("checkers"), text = "bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0", out = "examples/out/checkers.png"] = process.argv.slice(2);
const crystal = await load(src), eng = await engine(crystal), kit = checkersKit(crystal);

// 32 squares in draughts order, side to move, jumping piece, idle moves
const pos = ck.parse(text);
const run = await eng.run(kit.frame(pos), kit.frameSize);
const a = kit.readWith(eng, run, pos, { temp: 0 }); // the brightest legal move ({ temp, topK } to sample among the brightest instead)
console.log(`the crystal plays ${a.uci} (${a.cells} legal moves; ${(100 * a.share).toFixed(1)} % of all the light on its cell)`);
// if the same side must jump again (next.jump >= 0), ask again
const next = ck.make(pos, a.move!);
console.log(`after it: ${ck.str(next)}`);

// the board frame, the camera's picture after passes 1–3, and the move map with the crystal's move in white
const map = lightTile(a.powers, 32, 8);
outline(map, (a.cell % 32) * 8, Math.floor(a.cell / 32) * 8, 8, 8, [255, 255, 255]);
savePng(out, [[frameTile(kit.frame(pos), kit.frameSize, 2), ...run.hiddens.map((h) => frameTile(h, kit.frameSize, 2)), map]]);
console.log(`wrote ${out}`);
eng.dispose();
