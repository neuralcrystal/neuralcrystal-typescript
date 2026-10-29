// A crystal's input and output, drawn large (MNIST, CIFAR-10 and checkers; chess has its own: chess-input-output.ts).
//   npx vite-node examples/input-output.ts mnist [crystal] [digit.png]
//   npx vite-node examples/input-output.ts cifar10 [crystal] [photo.png]
//   npx vite-node examples/input-output.ts checkers [crystal] ["<position>"]
// Writes examples/out/<problem>-input.png (what the micromirrors show) and <problem>-output.png (the light the answer is read from).
import { checkers as ck, checkersKit, cifarKit, mnistKit, solve, huggingFaceUrl } from "../src/index.js";
import { engine, exitTile, frameTile, lightTile, load, outline, readPng, savePng } from "./lib.js";

const problem = process.argv[2] ?? "mnist";
const crystal = await load(process.argv[3] ?? huggingFaceUrl(problem === "cifar10" || problem === "checkers" ? problem : "mnist")), eng = await engine(crystal);
const input = `examples/out/${problem}-input.png`, output = `examples/out/${problem}-output.png`;

if (problem === "checkers") {
  const kit = checkersKit(crystal), pos = ck.parse(process.argv[4] ?? "bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0");
  const run = await eng.run(kit.frame(pos), kit.frameSize), a = kit.readWith(eng, run, pos, { temp: 0 });
  savePng(input, [[frameTile(kit.frame(pos), kit.frameSize, 6)]]);
  const S = 16, map = lightTile(a.powers, 32, S);
  for (const cell of ck.legalCells(pos).keys()) outline(map, (cell % 32) * S, Math.floor(cell / 32) * S, S, S, [70, 120, 230]);
  outline(map, (a.cell % 32) * S, Math.floor(a.cell / 32) * S, S, S, [255, 150, 40]);
  savePng(output, [[map]]);
  console.log(`the crystal plays ${a.uci}`);
} else {
  const kit = problem === "cifar10" ? cifarKit(crystal) : mnistKit(crystal);
  const pic = readPng(process.argv[4] ?? (problem === "cifar10" ? "examples/assets/frog.png" : "examples/assets/digit-4.png")).px;
  const { answer, run } = await solve(kit as ReturnType<typeof mnistKit>, eng, pic);
  savePng(input, [[frameTile(kit.frame(pic), kit.frameSize, problem === "cifar10" ? 3 : 16)]]);
  const size = 512, exit = exitTile(run.exit, size), k = size / eng.geometry.N;
  kit.rects(eng).forEach((r, i) => outline(exit, Math.round(r.x0 * k), Math.round(r.y0 * k), Math.round(r.w * k), Math.round(r.h * k), i === answer.pick ? [255, 150, 40] : [70, 120, 230]));
  savePng(output, [[exit]]);
  console.log(`the crystal reads ${"name" in answer ? (answer as { name: string }).name : answer.pick}`);
}
console.log(`wrote ${input} and ${output}`);
eng.dispose();
